"""Safetensors adapter — the canonical FP source format.

Implements the real container:
  8-byte little-endian header length, then a JSON header, then raw tensor data.
Supports F32/F16/BF16 and -- crucially -- returns the tensors unchanged so the
framework's quantization layer is the only thing that decides precision.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from ..memory import ExpertPage, TensorContainer  # noqa: F401
from . import LoadedModel, ModelSpec, TensorTable, register

_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
    "BOOL": np.bool_,
}


def _read_dtype(dt: str, buf: bytes, shape, offset: int):
    if dt == "BF16":
        u = np.frombuffer(buf, dtype=np.uint16, count=int(np.prod(shape)), offset=offset)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    arr = np.frombuffer(buf, dtype=_DTYPES[dt], count=int(np.prod(shape)), offset=offset)
    return arr.reshape(shape).copy()


def _write_dtype(arr: np.ndarray, dt: str) -> bytes:
    if dt == "BF16":
        u = (np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
        return u.tobytes()
    return np.ascontiguousarray(arr).astype(_DTYPES[dt]).tobytes()


def _dtype_of(arr: np.ndarray) -> str:
    m = {np.dtype("float32"): "F32", np.dtype("float16"): "F16",
         np.dtype("int8"): "I8", np.dtype("int32"): "I32", np.dtype("int64"): "I64"}
    return m.get(arr.dtype, "BF16" if arr.dtype == np.uint16 else "F32")


class SafetensorsAdapter:
    name = "safetensors"
    extensions = (".safetensors",)

    @staticmethod
    def probe(path: str) -> bool:
        """Strict: .safetensors extension, OR a valid safetensors JSON header.

        Must NOT accept arbitrary binaries — that would steal files from the other
        format adapters in the registry.
        """
        if path.endswith(".safetensors"):
            return True
        try:
            size = os.path.getsize(path)
            if size < 16 or size > (64 << 30):
                return False
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                if not (0 < n < size - 8):
                    return False
                hdr = json.loads(f.read(n))
            if not isinstance(hdr, dict) or not hdr:
                return False
            # every non-__metadata__ entry must look like a tensor entry
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                if not isinstance(v, dict):
                    return False
                if not {"dtype", "shape", "data_offsets"} <= set(v):
                    return False
                if not isinstance(v["data_offsets"], list) or len(v["data_offsets"]) != 2:
                    return False
            return any(k != "__metadata__" for k in hdr)
        except Exception:
            return False

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
            base = 8 + n
            buf = f.read()
        meta = hdr.pop("__metadata__", {}) if isinstance(hdr.get("__metadata__"), dict) else {}
        tensors: dict[str, np.ndarray] = {}
        bpw: dict[str, float] = {}
        for name, info in hdr.items():
            if not isinstance(info, dict) or "dtype" not in info:
                continue
            dtype, shape = info["dtype"], info["shape"]
            s, e = info["data_offsets"]
            arr = _read_dtype(dtype, buf[s:e], shape, 0)
            tensors[name] = arr
            bpw[name] = {"F32": 32.0, "F16": 16.0, "BF16": 16.0, "F64": 64.0}.get(dtype, 8.0)
        spec = _spec_from_meta(meta, tensors)
        pages, dense = _build_pages(tensors, spec)
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw), expert_pages=pages,
                           dense_pages=dense, format_name="safetensors",
                           source_file=path, notes=["FP source; framework assigns quantization"])

    @staticmethod
    def save(path: str, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> str:
        hdr: dict = {}
        if metadata:
            hdr["__metadata__"] = metadata
        offset = 0
        blobs = []
        for name, arr in tensors.items():
            dt = _dtype_of(arr)
            raw = _write_dtype(arr, dt)
            hdr[name] = {"dtype": dt, "shape": list(arr.shape),
                         "data_offsets": [offset, offset + len(raw)]}
            offset += len(raw)
            blobs.append(raw)
        hbytes = json.dumps(hdr).encode()
        pad = (-len(hbytes)) % 8
        hbytes += b" " * pad
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hbytes)))
            f.write(hbytes)
            for b in blobs:
                f.write(b)
        return path


def _spec_from_meta(meta: dict, tensors: dict[str, np.ndarray]) -> ModelSpec:
    cfg = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    return ModelSpec(
        name=meta.get("name", "unnamed"),
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
    """Group tensors into expert pages / dense pages using naming conventions."""
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


register(SafetensorsAdapter)
