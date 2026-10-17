"""Lumina native format adapter — stores fully quantized, execution-ready models.

Container: magic(4) + version(4) + header_len(8) + JSON header + tensor blobs.
Each tensor stores: codes + scales (packed) with exact byte accounting.
"""
from __future__ import annotations

import json
import struct

import numpy as np

from ..memory import ExpertPage, TensorContainer
from . import LoadedModel, ModelSpec, TensorTable, register

LUMINA_MAGIC = 0x4E494D4C  # "LMIN"
LUMINA_VERSION = 1


class LuminaAdapter:
    name = "lumina"
    extensions = (".lumina",)

    @staticmethod
    def probe(path: str) -> bool:
        if not path.endswith(".lumina"):
            return False
        try:
            with open(path, "rb") as f:
                magic, ver = struct.unpack("<II", f.read(8))
                return magic == LUMINA_MAGIC
        except Exception:
            return False

    @staticmethod
    def load(path: str) -> LoadedModel:
        with open(path, "rb") as f:
            magic, ver, hlen = struct.unpack("<IIQ", f.read(16))
            hdr = json.loads(f.read(hlen))
            buf = f.read()
        
        tensors = {}
        bpw = {}
        tensors_hdr = hdr.get("tensors", {})
        
        for name, info in tensors_hdr.items():
            # Handle old fixture format (F32 reference)
            if "codec" in info and info["codec"] == "f32-reference":
                s, e = info["offset"], info["offset"] + info["length"]
                arr = np.frombuffer(buf[s:e], dtype=np.float32).reshape(info["shape"])
                tensors[name] = arr
                bpw[name] = 32.0
                continue
            
            # Handle new quantized format
            if "kind" in info:
                kind = info.get("kind", "int4")
                s = info["offset"]
                e = s + info["codes_len"]
                codes = np.frombuffer(buf[s:e], dtype=np.uint8).reshape(info["codes_shape"])
                s2, e2 = s + info["codes_len"], s + info["codes_len"] + info["scales_len"]
                scales = np.frombuffer(buf[s2:e2], dtype=np.float32).reshape(info["scales_shape"])
                
                if kind in ("int4", "int3", "int2"):
                    levels = np.linspace(-1.0, 1.0, 2 ** int(kind[-1]))
                    gs = info.get("group_size", 64)
                    s_expanded = np.repeat(scales, gs, axis=1)[:, :codes.shape[1]]
                    arr = s_expanded * levels[codes.astype(np.int64)]
                    tensors[name] = arr.astype(np.float32)
                    bpw[name] = float(kind[-1])
                elif kind == "binary":
                    N, K_packed = codes.shape
                    K = info["scales_shape"][1] * info.get("group_size", 64)
                    bits = ((codes[:, :, None] >> np.arange(8, dtype=np.uint8)[None, None, :]) & 1)
                    signs = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
                    signs = signs.reshape(N, -1)[:, :K]
                    s_expanded = np.repeat(scales, info.get("group_size", 64), axis=1)[:, :K]
                    arr = signs * s_expanded
                    tensors[name] = arr.astype(np.float32)
                    bpw[name] = 1.0
                continue
            
            # Default: treat as raw F32
            if "offset" in info and "length" in info:
                s, e = info["offset"], info["offset"] + info["length"]
                arr = np.frombuffer(buf[s:e], dtype=np.float32).reshape(info["shape"])
                tensors[name] = arr
                bpw[name] = 32.0
        
        # Get spec from hdr (fixture format) or model_spec (new format)
        spec_hdr = hdr.get("spec", {}) or hdr.get("model_spec", {})
        spec = ModelSpec(
            name=hdr.get("name", spec_hdr.get("name", "lumina")),
            arch=hdr.get("arch", spec_hdr.get("arch", "swiglu-moe")),
            n_layer=spec_hdr.get("n_layer", 1),
            n_expert=spec_hdr.get("n_expert", 1),
            n_shared_expert=spec_hdr.get("n_shared_expert", 1),
            d_model=spec_hdr.get("d_model", 64),
            d_ff=spec_hdr.get("d_ff", 128),
            top_k=spec_hdr.get("top_k", 1),
            router_scaling=spec_hdr.get("router_scaling", 1.0),
            vocab=spec_hdr.get("vocab", 512),
        )
        
        # Build expert pages like other adapters
        from ..memory import ExpertPage, TensorContainer as TC
        pages: dict[str, ExpertPage] = {}
        dense: dict[str, TC] = {}
        
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
        
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw), 
                           expert_pages=pages, dense_pages=dense,
                           format_name="lumina", source_file=path,
                           notes=["Lumina native quantized format"])

    @staticmethod
    def save(path: str, tensors: dict[str, np.ndarray], spec: ModelSpec,
             quantization: dict[str, str] | None = None) -> str:
        """Save quantized model to Lumina format."""
        # This is used by the converter pipeline
        # quantization: tensor_name -> "int4"|"int3"|"int2"|"binary"|"fp32"
        from ..format.codecs.ptq_ladder import quantize_int, quantize_binary, pack_signs
        
        blobs = []
        tensor_info = {}
        off = 0
        
        for name, arr in tensors.items():
            qtype = quantization.get(name, "fp32") if quantization else "fp32"
            if qtype == "fp32":
                a = np.ascontiguousarray(arr, dtype=np.float32)
                raw = a.tobytes()
                tensor_info[name] = {"kind": "raw", "dtype": "F32", "shape": list(a.shape),
                                     "offset": off, "length": len(raw)}
                blobs.append(raw)
                off += len(raw)
            elif qtype == "binary":
                q = quantize_binary(arr, group_size=64)
                codes = q.codes
                scales = q.scales
                cb = codes.tobytes()
                sb = scales.astype(np.float32).tobytes()
                tensor_info[name] = {"kind": "binary", "codes_dtype": "uint8",
                                     "codes_shape": list(codes.shape),
                                     "scales_dtype": "F32", "scales_shape": list(scales.shape),
                                     "offset": off, "codes_len": len(cb), "scales_len": len(sb),
                                     "group_size": 64}
                blobs.append(cb)
                blobs.append(sb)
                off += len(cb) + len(sb)
            else:
                bits = int(qtype[-1])
                q = quantize_int(arr, bits, group_size=64)
                codes = q.codes.astype(np.uint8)
                scales = q.scales
                cb = codes.tobytes()
                sb = scales.astype(np.float32).tobytes()
                tensor_info[name] = {"kind": f"int{bits}", "codes_dtype": "uint8",
                                     "codes_shape": list(codes.shape),
                                     "scales_dtype": "F32", "scales_shape": list(scales.shape),
                                     "offset": off, "codes_len": len(cb), "scales_len": len(sb),
                                     "group_size": 64, "code_bits": bits}
                blobs.append(cb)
                blobs.append(sb)
                off += len(cb) + len(sb)
        
        hdr = {"format": "lumina", "version": LUMINA_VERSION, "name": spec.name,
               "arch": spec.arch, "model_spec": {"n_layer": spec.n_layer,
                   "d_model": spec.d_model, "d_ff": spec.d_ff, "top_k": spec.top_k,
                   "vocab": spec.vocab}, "tensors": tensor_info}
        hb = json.dumps(hdr).encode()
        
        with open(path, "wb") as f:
            f.write(struct.pack("<IIQ", LUMINA_MAGIC, LUMINA_VERSION, len(hb)))
            f.write(hb)
            for b in blobs:
                f.write(b)
        return path


register(LuminaAdapter)
