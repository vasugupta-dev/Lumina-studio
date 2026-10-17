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


def pack_codes_nibble(codes: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pack int4 codes, 2 per byte (low nibble first). Pads odd widths."""
    o, i = codes.shape
    pad = (-i) % 2
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)))
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    return ((c[0::2] & 0xF) | ((c[1::2] & 0xF).astype(np.uint8) << 4)), (o, i)


def pack_codes_2bit(codes: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pack int2 codes, 4 per byte (c0 in bits 0-1 .. c3 in bits 6-7)."""
    o, i = codes.shape
    pad = (-i) % 4
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)))
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    return (((c[0::4] & 0x3).astype(np.uint8)
             | ((c[1::4] & 0x3).astype(np.uint8) << 2)
             | ((c[2::4] & 0x3).astype(np.uint8) << 4)
             | ((c[3::4] & 0x3).astype(np.uint8) << 6)), (o, i))


def lumina_header(path: str) -> dict:
    """Raw JSON header of a .lumina file (meta without weight bytes)."""
    with open(path, "rb") as f:
        _m, _v, hlen = struct.unpack("<IIQ", f.read(16))
        return json.loads(f.read(hlen))


def unpack_codes(buf: bytes, shape: tuple[int, int], flag: str) -> np.ndarray:
    """Inverse of the packers above; "none"/unknown = legacy 1 byte per code.

    Packed rows are stride-padded (odd widths), so the row stride is derived
    from the buffer length — never assumed from the matrix width.
    """
    o, i = shape
    if flag == "nibble":
        raw = np.frombuffer(buf, dtype=np.uint8)
        wp = (raw.size * 2) // o
        assert wp * o == raw.size * 2 and wp >= i, "bad nibble blob"
        flat = np.empty((o, wp), dtype=np.uint8)
        flat[:, 0::2] = raw.reshape(o, -1) & 0xF
        flat[:, 1::2] = raw.reshape(o, -1) >> 4
        return flat[:, :i]
    if flag == "2bit":
        raw = np.frombuffer(buf, dtype=np.uint8)
        wp = (raw.size * 4) // o
        assert wp * o == raw.size * 4 and wp >= i, "bad 2bit blob"
        r = raw.reshape(o, -1)
        flat = np.empty((o, wp), dtype=np.uint8)
        flat[:, 0::4] = r & 0x3
        flat[:, 1::4] = (r >> 2) & 0x3
        flat[:, 2::4] = (r >> 4) & 0x3
        flat[:, 3::4] = (r >> 6) & 0x3
        return flat[:, :i]
    return np.frombuffer(buf, dtype=np.uint8).reshape(o, i)


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

            # Raw fp32 (sensitive escapes, norms, small mats)
            if info.get("kind") == "raw":
                s, e = info["offset"], info["offset"] + info["length"]
                arr = np.frombuffer(buf[s:e], dtype=np.float32).reshape(info["shape"])
                tensors[name] = arr
                bpw[name] = 32.0
                continue

            # Quantized: unpack codes when the writer bit-packed them
            # ("packed" flag; files without it are legacy unpacked layout).
            if "kind" in info:
                kind = info.get("kind", "int4")
                gs = int(info.get("group_size", 64))
                s = info["offset"]
                e = s + info["codes_len"]
                s2, e2 = s + info["codes_len"], s + info["codes_len"] + info["scales_len"]
                scales = np.frombuffer(buf[s2:e2], dtype=np.float32).reshape(info["scales_shape"])
                shape = [int(x) for x in info.get("shape", info["codes_shape"])]
                flag = info.get("packed", "none")
                codes = unpack_codes(buf[s:e], (shape[0], shape[1]), flag)

                if kind in ("int4", "int3", "int2"):
                    levels = np.linspace(-1.0, 1.0, 2 ** int(kind[-1]))
                    s_expanded = np.repeat(scales, gs, axis=1)[:, :codes.shape[1]]
                    arr = s_expanded * levels[codes.astype(np.int64)]
                    tensors[name] = arr.astype(np.float32)
                    bpw[name] = float(kind[-1])
                elif kind == "binary":
                    N, K_packed = codes.shape
                    K = info["scales_shape"][1] * gs
                    bits = ((codes[:, :, None] >> np.arange(8, dtype=np.uint8)[None, None, :]) & 1)
                    signs = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
                    signs = signs.reshape(N, -1)[:, :K]
                    s_expanded = np.repeat(scales, gs, axis=1)[:, :K]
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
        from lumina_format.codecs.ptq_ladder import quantize_binary, quantize_int

        blobs = []
        tensor_info = {}
        off = 0
        group_size = 64

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
                q = quantize_binary(arr, group_size=group_size)
                codes = q.codes
                scales = q.scales
                cb = codes.tobytes()
                sb = scales.astype(np.float32).tobytes()
                tensor_info[name] = {"kind": "binary", "codes_dtype": "uint8",
                                     "codes_shape": list(codes.shape),
                                     "scales_dtype": "F32", "scales_shape": list(scales.shape),
                                     "shape": list(codes.shape), "packed": "none",
                                     "group_size": group_size,
                                     "offset": off, "codes_len": len(cb), "scales_len": len(sb)}
                blobs.append(cb)
                blobs.append(sb)
                off += len(cb) + len(sb)
            else:
                bits = int(qtype[-1])
                q = quantize_int(arr, bits, group_size=group_size)
                codes = np.ascontiguousarray(q.codes, dtype=np.uint8)
                scales = q.scales
                pack_flag = "none"
                shape = list(codes.shape)
                if qtype == "int4":
                    packed_arr, shape = pack_codes_nibble(codes)
                    cb = packed_arr.tobytes()
                    pack_flag = "nibble"
                elif qtype == "int2":
                    packed_arr, shape = pack_codes_2bit(codes)
                    cb = packed_arr.tobytes()
                    pack_flag = "2bit"
                else:
                    cb = codes.tobytes()
                sb = scales.astype(np.float32).tobytes()
                tensor_info[name] = {"kind": f"int{bits}", "codes_dtype": "uint8",
                                     "codes_shape": list(codes.shape),
                                     "scales_dtype": "F32", "scales_shape": list(scales.shape),
                                     "shape": list(shape), "packed": pack_flag,
                                     "group_size": group_size,
                                     "offset": off, "codes_len": len(cb), "scales_len": len(sb)}
                blobs.append(cb)
                blobs.append(sb)
                off += len(cb) + len(sb)
        
        hdr = {"format": "lumina", "version": LUMINA_VERSION, "name": spec.name,
               "arch": spec.arch, "model_spec": {"n_layer": spec.n_layer,
                   "n_expert": spec.n_expert,
                   "n_shared_expert": spec.n_shared_expert,
                   "d_model": spec.d_model, "d_ff": spec.d_ff,
                   "top_k": spec.top_k,
                   "router_scaling": spec.router_scaling,
                   "router_dtype": spec.router_dtype,
                   "vocab": spec.vocab}, "tensors": tensor_info}
        hb = json.dumps(hdr).encode()
        
        with open(path, "wb") as f:
            f.write(struct.pack("<IIQ", LUMINA_MAGIC, LUMINA_VERSION, len(hb)))
            f.write(hb)
            for b in blobs:
                f.write(b)
        return path


register(LuminaAdapter)
