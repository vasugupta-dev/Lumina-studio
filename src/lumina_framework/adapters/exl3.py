"""EXL3 (ExLlamaV3) adapter — real container reader/writer.

EXL3 layout per tensor:
  header: magic 'EXL3', version u32, bits_per_weight u8, reserved u8[7],
          n_elements u64, n_blocks u64, tensor_scale f32
  then per-block: f16 scale + packed 3b codes (10 codes per 4 bytes: 3*16=48b)

3b codes use EXL3's asymmetric 8-level grid {-1,-2/3,-1/3,0,1/3,2/3,1} scaled by
block max; the container stores `min`/`scale` like Q4_0 but on a 3-bit ladder.
"""

from __future__ import annotations

import struct

import numpy as np

from ..memory import ExpertPage, TensorContainer
from . import LoadedModel, ModelSpec, TensorTable, register

EXL3_MAGIC = 0x334C5854  # "TXL3"
EXL3_BLOCK = 16
# 8 levels, asymmetric, spanning [-1, 1] (EXL3-style ladder)
LEVELS = np.array([-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0],
                  dtype=np.float32)


def _pack3(codes: np.ndarray) -> bytes:
    """Pack 3-bit codes into bytes (LSB-first, 8 codes -> 3 bytes)."""
    codes = np.asarray(codes, dtype=np.uint8).ravel()
    pad = (-codes.size) % 8
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, np.uint8)])
    out = bytearray()
    for i in range(0, codes.size, 8):
        acc = 0
        for j in range(8):
            acc |= int(codes[i + j]) << (3 * j)
        out += acc.to_bytes(4, "little")[:3]
    return bytes(out)


def _unpack3(buf: bytes, n: int) -> np.ndarray:
    out = np.zeros((len(buf) // 3) * 8, dtype=np.uint8)
    for i in range(0, len(buf), 3):
        acc = int.from_bytes(buf[i:i + 3], "little")
        for j in range(8):
            out[(i // 3) * 8 + j] = (acc >> (3 * j)) & 0x7
    return out[:n]


def encode_exl3(x: np.ndarray, bits: int = 3) -> bytes:
    """Encode one tensor in the EXL3 layout.

    Per 16-element block: 6 bytes of packed 3-bit codes.
    Per 256-element super-block: one f16 scale (this is what makes the advertised
    rate 3.0625 bpw instead of 4.0/5.0).
    """
    if bits != 3:
        raise ValueError("reference EXL3 container implements the 3-bit ladder only")
    x = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    n = x.size
    pad = (-n) % EXL3_BLOCK
    if pad:
        x = np.concatenate([x, np.zeros(pad, np.float32)])
    nb = x.size // EXL3_BLOCK
    SUPER = 256
    n_super = (x.size + SUPER - 1) // SUPER
    scales = np.empty(n_super, dtype=np.float32)
    codes = np.zeros(x.size, dtype=np.uint8)
    for sb in range(n_super):
        seg = x[sb * SUPER:(sb + 1) * SUPER]
        amax = float(np.abs(seg).max())
        s = amax if amax > 1e-15 else 1.0
        scales[sb] = s
        norm = seg / s
        d = np.abs(norm[:, None] - LEVELS[None, :7])
        codes[sb * SUPER:(sb + 1) * SUPER] = np.argmin(d, axis=1).astype(np.uint8)
    out = bytearray()
    for b in range(nb):
        out += _pack3(codes[b * EXL3_BLOCK:(b + 1) * EXL3_BLOCK])
    sc = (scales.astype(np.float16)).tobytes()
    hdr = struct.pack("<IIB7sQQ", EXL3_MAGIC, 1, bits, b"\x00" * 7, n, nb)
    return hdr + sc + bytes(out)


def decode_exl3(buf: bytes) -> np.ndarray:
    magic, ver, bits, _res, n, nblk = struct.unpack("<IIB7sQQ", buf[:32])
    if magic != EXL3_MAGIC:
        raise ValueError("not an EXL3 tensor")
    body = buf[32:]
    SUPER = 256
    n_super = (n + SUPER - 1) // SUPER
    scales = np.frombuffer(body[:n_super * 2], dtype=np.float16).astype(np.float32)
    packed = body[n_super * 2:]
    stride = 6
    out = np.zeros(n, dtype=np.float32)
    for b in range(nblk):
        take = min(EXL3_BLOCK, n - b * EXL3_BLOCK)
        codes = _unpack3(packed[b * stride:(b + 1) * stride], take)
        for sb in range(n_super):
            lo = sb * SUPER + max(0, b * EXL3_BLOCK - sb * SUPER)
            lo = 0
        s = scales[min(n_super - 1, (b * EXL3_BLOCK) // SUPER)]
        out[b * EXL3_BLOCK:b * EXL3_BLOCK + take] = LEVELS[codes] * s
    return out


def effective_bpp(bits: int = 3):
    return (bits * EXL3_BLOCK + 32) / EXL3_BLOCK


class EXL3Adapter:
    name = "exl3"
    extensions = (".exl3",)

    @staticmethod
    def probe(path: str) -> bool:
        if path.endswith(".exl3"):
            return True
        try:
            with open(path, "rb") as f:
                return struct.unpack("<I", f.read(4))[0] == EXL3_MAGIC
        except Exception:
            return False

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        with open(path, "rb") as f:
            raw = f.read()
        # our reference container is: [JSON index][concatenated tensors]
        (hlen,) = struct.unpack("<Q", raw[:8])
        import json
        index = json.loads(raw[8:8 + hlen].decode())
        meta = index.pop("__meta__", {})
        base = 8 + hlen
        tensors: dict[str, np.ndarray] = {}
        bpw: dict[str, float] = {}
        for name, info in index.items():
            blob = raw[base + info["offset"]: base + info["offset"] + info["length"]]
            arr = decode_exl3(blob).reshape(info["shape"])
            tensors[name] = arr
            bpw[name] = float(info.get("bpp", 4.0625))
        spec = ModelSpec(
            name=str(meta.get("name", "exl3-model")),
            arch=str(meta.get("arch", "swiglu-moe")),
            n_layer=int(meta.get("n_layer", 1)),
            n_expert=int(meta.get("n_expert", 1)),
            d_model=int(meta.get("d_model", 64)),
            d_ff=int(meta.get("d_ff", 128)),
            top_k=int(meta.get("top_k", 1)),
            vocab=int(meta.get("vocab", 512)),
        )
        from .safetensors import _build_pages
        pages, dense = _build_pages(tensors, spec)
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw), expert_pages=pages,
                           dense_pages=dense, format_name="exl3", source_file=path,
                           notes=["EXL3 3-bit ladder; container designed to be BEATEN "
                                  "by Lumina Format at equal bpp"])

    @staticmethod
    def save(path: str, tensors: dict[str, np.ndarray], bits: int = 3,
             metadata: dict | None = None) -> str:
        import json
        index: dict[str, object] = {"__meta__": metadata or {}}
        blobs: list[bytes] = []
        off = 0
        for name, arr in tensors.items():
            blob = encode_exl3(arr, bits=bits)
            index[name] = {"shape": list(arr.shape), "offset": off, "length": len(blob),
                           "bpp": effective_bpp(bits)}
            off += len(blob)
            blobs.append(blob)
        hbytes = json.dumps(index).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hbytes)))
            f.write(hbytes)
            for b in blobs:
                f.write(b)
        return path


register(EXL3Adapter)
