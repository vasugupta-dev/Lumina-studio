"""BitsAndBytes NF4/4-bit adapter for safetensors container.

Handles the bnb quantized format where weights are stored as:
- <name>.weight: U8 quantized values
- <name>.weight.absmax: U8 absmax per block  
- <name>.weight.quant_state.bitsandbytes__nf4: U8 quantization state
"""
from __future__ import annotations

import json
import struct

import numpy as np

from ..memory import ExpertPage, TensorContainer  # noqa: F401
from . import LoadedModel, ModelSpec, TensorTable, register

_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
    "BOOL": np.bool_,
}


def _read_dtype(dt: str, buf: bytes, shape, offset: int):
    if dt == "BF16":
        u = np.frombuffer(buf, dtype=np.uint16, count=int(np.prod(shape)), offset=offset)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    arr = np.frombuffer(buf, dtype=_DTYPES[dt], count=int(np.prod(shape)), offset=offset)
    return arr.reshape(shape).copy()


# NF4 quantization constants (from bitsandbytes)
_NF4_VALUES = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003923177719, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=np.float32)


def _dequantize_nf4(qweight: np.ndarray, absmax: np.ndarray, quant_state: np.ndarray,
                     output_shape: tuple) -> np.ndarray:
    """Dequantize NF4 weights to FP32.
    
    qweight: (n_blocks * block_size, 1) - packed 4-bit values (2 per byte)
    absmax: (n_blocks,) - scale per block
    quant_state: contains block_size, etc.
    """
    block_size = 64  # default NF4 block size
    # Parse quant_state if needed
    if quant_state is not None and len(quant_state) >= 8:
        # First bytes often encode block_size
        pass
    
    # NF4: 2 values per byte (4-bit each)
    n_blocks = absmax.shape[0]
    flat_len = qweight.shape[0]
    expected = n_blocks * block_size
    
    # Reshape qweight to 1D
    qflat = qweight.flatten()
    
    # Unpack 4-bit values (2 per byte)
    n_vals = qflat.shape[0] * 2
    # Use vectorized unpacking
    high = (qflat >> 4) & 0xF
    low = qflat & 0xF
    unpacked = np.empty(n_vals, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    unpacked = unpacked[:expected]
    
    # Dequantize using NF4 lookup table
    absmax_f32 = absmax.astype(np.float32).flatten()
    scales = absmax_f32.repeat(block_size)[:expected]
    
    dequant = _NF4_VALUES[unpacked] * scales
    
    return dequant.reshape(output_shape)


class BNBAdapter:
    name = "bnb"
    extensions = (".safetensors",)  # bnb uses safetensors container

    @staticmethod
    def probe(path: str) -> bool:
        """Detect bnb format by checking for bnb-specific tensor names."""
        if not path.endswith(".safetensors"):
            return False
        try:
            import os
            size = os.path.getsize(path)
            if size < 16 or size > (64 << 30):
                return False
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                if not (0 < n < size - 8):
                    return False
                hdr = json.loads(f.read(n))
            if not isinstance(hdr, dict):
                return False
            # Check for bnb-specific patterns
            for k in hdr:
                if "bitsandbytes" in k or ".absmax" in k or ".quant_state" in k or "nested_absmax" in k or "nested_quant_map" in k or "quant_map" in k:
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        
        import os
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
            base = 8 + n
            buf = f.read()
        
        meta = hdr.pop("__metadata__", {}) if isinstance(hdr.get("__metadata__"), dict) else {}
        
        # First pass: collect all tensor info
        tensor_info = {}
        for name, info in hdr.items():
            if not isinstance(info, dict) or "dtype" not in info:
                continue
            tensor_info[name] = info
        
        # Group by base weight name - look for patterns
        weight_groups = {}
        for name, info in tensor_info.items():
            # Find the base name (everything before the last .weight or other suffix)
            base = name
            # Remove suffixes to get base
            suffixes = [".weight.absmax", ".weight.quant_state.bitsandbytes__nf4",
                       ".weight.nested_absmax", ".weight.nested_quant_map", ".weight.quant_map",
                       ".weight", ".absmax", ".quant_state.bitsandbytes__nf4",
                       ".nested_absmax", ".nested_quant_map", ".quant_map"]
            for suffix in suffixes:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            if base not in weight_groups:
                weight_groups[base] = {}
            weight_groups[base][name] = info
        
        # Load and dequantize
        tensors = {}
        bpw = {}
        
        for base, group in weight_groups.items():
            # Find the weight tensor and its quantization metadata
            weight_name = None
            absmax_name = None
            qstate_name = None
            
            for name in group:
                if name.endswith(".weight") and not name.endswith(".absmax") and "quant_state" not in name:
                    weight_name = name
                elif name.endswith(".weight.absmax"):
                    absmax_name = name
                elif "quant_state" in name:
                    qstate_name = name
            
            if weight_name and absmax_name:
                w_info = group[weight_name]
                a_info = group[absmax_name]
                q_info = group.get(qstate_name)
                
                s, e = w_info["data_offsets"]
                qweight = _read_dtype(w_info["dtype"], buf[s:e], w_info["shape"], 0)
                
                s, e = a_info["data_offsets"]
                absmax = _read_dtype(a_info["dtype"], buf[s:e], a_info["shape"], 0)
                
                quant_state = None
                if q_info:
                    s, e = q_info["data_offsets"]
                    quant_state = _read_dtype(q_info["dtype"], buf[s:e], q_info["shape"], 0)
                
                out_shape = w_info["shape"]
                if out_shape[-1] == 1:
                    out_shape = out_shape[:-1]
                
                try:
                    dequant = _dequantize_nf4(qweight, absmax, quant_state, out_shape)
                    tensors[weight_name] = dequant
                    bpw[weight_name] = 4.0
                except Exception as e:
                    tensors[weight_name] = qweight.reshape(out_shape)
                    bpw[weight_name] = 8.0
            else:
                # Regular tensor
                for name, info in group.items():
                    if name not in tensors:
                        s, e = info["data_offsets"]
                        arr = _read_dtype(info["dtype"], buf[s:e], info["shape"], 0)
                        tensors[name] = arr
                        bpw[name] = {"F32": 32.0, "F16": 16.0, "BF16": 16.0, "F64": 64.0, "U8": 8.0, "I8": 8.0}.get(info["dtype"], 8.0)
        
        spec = _spec_from_meta(meta, tensors)
        pages, dense = _build_pages(tensors, spec)
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw), expert_pages=pages,
                           dense_pages=dense, format_name="bnb",
                           source_file=path, notes=["bnb NF4 quantized; dequantized on load"])


def _spec_from_meta(meta: dict, tensors: dict[str, np.ndarray]) -> ModelSpec:
    cfg = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    return ModelSpec(
        name=meta.get("name", "bnb-model"),
        arch=cfg.get("arch", meta.get("arch", "swiglu-moe")),
        n_layer=cfg.get("n_layer", meta.get("n_layer", 1)),
        n_expert=cfg.get("n_expert", meta.get("n_expert", 1)),
        n_shared_expert=cfg.get("n_shared_expert", 1),
        d_model=cfg.get("d_model", meta.get("d_model", 64)),
        d_ff=cfg.get("d_ff", meta.get("d_ff", 128)),
        top_k=cfg.get("top_k", meta.get("top_k", 1)),
        router_scaling=cfg.get("router_scaling", 1.0),
        vocab=cfg.get("vocab", meta.get("vocab", 512)),
    )


def _build_pages(tensors: dict[str, np.ndarray], spec: ModelSpec):
    pages: dict[str, ExpertPage] = {}
    dense: dict[str, "TensorContainer"] = {}
    from ..memory import TensorContainer as TC

    for l in range(spec.n_layer):
        shared = {k: v for k, v in tensors.items()
                  if f".{l}.shared." in k or f"layers.{l}.shared" in k}
        if shared:
            pages[f"L{l}-shared"] = ExpertPage(
                f"L{l}-shared", {p: TC(shared[p]) for p in sorted(shared)}, l, arch="shared")
        for e in range(spec.n_expert):
            ex = {k: v for k, v in tensors.items()
                  if f".{l}.experts.{e}." in k or f"layers.{l}.experts.{e}." in k}
            if ex:
                pages[f"L{l}-e{e}"] = ExpertPage(
                    f"L{l}-e{e}", {p: TC(ex[p]) for p in sorted(ex)}, l)

    skip = ("experts.", "shared", "router")
    for name, arr in tensors.items():
        if any(s in name for s in skip):
            continue
        if arr.ndim >= 1 and arr.size >= 1 and arr.dtype.kind == "f":
            dense[name] = TC(arr)
    return pages, dense


register(BNBAdapter)
