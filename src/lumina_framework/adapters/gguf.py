"""GGUF (llama.cpp) adapter — real container reader/writer.

Layout (little-endian):
  magic "GGUF", version u32, tensor_count u64, kv_count u64,
  KV metadata pairs, tensor infos, padding to alignment, tensor data.

Quantization types implemented (both directions):
  Q8_0   : 34B blocks = int8 x32 + f16 scale
  Q4_0   : 18B blocks = f16 scale + 16x 4-bit pairs (asymmetric, -8..7)
  F16/F32: raw
"""

from __future__ import annotations

import struct

import numpy as np

from ..memory import ExpertPage, TensorContainer
from . import LoadedModel, ModelSpec, TensorTable, register

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

GGML_TYPE = {
    "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3,
    "Q8_0": 8, "Q8_1": 9, "Q2_K": 10, "Q3_K": 11,
    "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
}
GGML_TYPE_REV = {v: k for k, v in GGML_TYPE.items()}

GGUF_VALUE = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
              10: "u64", 11: "i64", 12: "f64", 8: "str", 9: "arr"}
_FMT = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
        "f32": "<f", "u64": "<Q", "i64": "<q", "f64": "<d"}

Q8_0 = 32
Q4_0 = 32

GH = 0x1


class _Writer:
    def __init__(self):
        self.buf = bytearray()

    def u32(self, v):
        self.buf += struct.pack("<I", v)

    def u64(self, v):
        self.buf += struct.pack("<Q", v)

    def raw(self, b: bytes):
        self.buf += b

    def f16(self, v):
        self.buf += struct.pack("<e", float(v))

    def str(self, s: str):
        b = s.encode()
        self.u64(len(b))
        self.buf += b

    def value(self, t: int, v):
        if t == 8:
            self.str(v)
        elif t == 9:  # array: (elem_type, count, values...)
            et, vals = v
            self.u32(et)
            self.u64(len(vals))
            for x in vals:
                self.value(et, x)
        else:
            self.buf += struct.pack(_FMT[GGUF_VALUE[t]], v)

    def keyval(self, k: str, t: int, v):
        self.str(k)
        self.u32(t)
        self.value(t, v)

    def bytes(self) -> bytes:
        return bytes(self.buf)


# ---------------------------------------------------------------------------
# quantizers
# ---------------------------------------------------------------------------

def quantize_q8_0(x: np.ndarray) -> tuple[bytes, float]:
    x = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    n = x.size
    pad = (-n) % Q8_0
    if pad:
        x = np.concatenate([x, np.zeros(pad, np.float32)])
    blocks = x.reshape(-1, Q8_0)
    amax = np.abs(blocks).max(axis=1, keepdims=True).clip(1e-15)
    scale = amax / 127.0
    q = np.round(blocks / scale).clip(-127, 127).astype(np.int8)
    out = bytearray()
    for i in range(q.shape[0]):
        out += np.float16(scale[i, 0]).tobytes()
        out += q[i].tobytes()
    return bytes(out), 8.0


def dequantize_q8_0(buf: bytes, n: int) -> np.ndarray:
    if len(buf) % 34:
        raise ValueError("Q8_0 buffer not block aligned")
    nb = len(buf) // 34
    out = np.empty(nb * Q8_0, dtype=np.float32)
    mv = memoryview(buf)
    for i in range(nb):
        blk = mv[i * 34:(i + 1) * 34]
        s = np.frombuffer(blk[:2], dtype=np.float16)[0]
        q = np.frombuffer(blk[2:], dtype=np.int8).astype(np.float32)
        out[i * Q8_0:(i + 1) * Q8_0] = q * np.float32(s)
    return out[:n]


def quantize_q4_0(x: np.ndarray) -> tuple[bytes, float]:
    """GGUF Q4_0: 32 weights per 18-byte block = 2B f16 scale + 16B packed 4-bit.

    Canonical GGUF reconstruction is `d * (q - 8)` with q in 0..15.
    """
    x = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    n = x.size
    pad = (-n) % Q4_0
    if pad:
        x = np.concatenate([x, np.zeros(pad, np.float32)])
    nb = x.size // Q4_0
    final = bytearray()
    for b in range(nb):
        blk = x[b * Q4_0:(b + 1) * Q4_0]
        amax = float(np.abs(blk).max())
        d = (amax / 7.0) if amax > 1e-15 else 1.0
        q = np.round(blk / d + 8.0).clip(0, 15).astype(np.uint8)
        final += np.float16(d).tobytes()
        final += bytes(int(q[j]) | (int(q[j + 1]) << 4) for j in range(0, Q4_0, 2))
    return bytes(final), 4.5


def dequantize_q4_0(buf: bytes, n: int) -> np.ndarray:
    if len(buf) % 18:
        raise ValueError(f"Q4_0 buffer not block aligned: {len(buf)} bytes")
    nb = len(buf) // 18
    out = np.empty(nb * Q4_0, dtype=np.float32)
    mv = memoryview(buf)
    for i in range(nb):
        blk = mv[i * 18:(i + 1) * 18]
        d = float(np.frombuffer(blk[:2], dtype=np.float16)[0])
        raw = np.frombuffer(blk[2:], dtype=np.uint8)
        vals = np.empty(Q4_0, dtype=np.float32)
        for j in range(32):
            byte = raw[j // 2]
            v = (byte & 0x0F) if j % 2 == 0 else (byte >> 4)
            vals[j] = (float(v) - 8.0) * d
        out[i * Q4_0:(i + 1) * Q4_0] = vals
    return out[:n] if n <= out.size else out


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

GGML_TO_HF = [
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output.weight", "lm_head.weight"),
    ("output_norm.weight", "model.norm.weight"),
]


def _ggml_layer_name(name: str) -> str | None:
    """blk.{l}.{ggml} -> model.layers.{l}.{hf}; None if not a ggml name."""
    import re as _re
    m = _re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None
    lyr, rest = m.group(1), m.group(2)
    table = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_qkv.weight": "self_attn.qkv_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_output.bias": "self_attn.o_proj.bias",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
        "ffn_down_exps": "mlp.experts.down_proj",
        "ffn_gate_exps": "mlp.experts.gate_up_proj",
    }
    hf = table.get(rest)
    return f"model.layers.{lyr}.{hf}" if hf else None


def ggml_to_hf_name(name: str) -> str | None:
    for g, h in GGML_TO_HF:
        if name == g:
            return h
    return _ggml_layer_name(name)


def gguf_chat_config(kvs: dict, arch: str = "") -> dict:
    """Build a ChatModel cfg from GGUF metadata (llama/qwen/mistral keys)."""
    pre = (arch + ".") if arch and not arch.endswith(".") else arch
    n_layer = int(kvs.get(pre + "block_count", kvs.get("block_count", 1)))
    toks = kvs.get("tokenizer.ggml.tokens", [])
    cfg = {"num_hidden_layers": n_layer,
           "hidden_size": int(kvs.get(pre + "embedding_length",
                                      kvs.get("embedding_length", 64))),
           "num_attention_heads": int(kvs.get(pre + "attention.head_count",
                                              kvs.get("attention.head_count", 8))),
           "num_key_value_heads": int(kvs.get(pre + "attention.head_count_kv",
                                             kvs.get("attention.head_count_kv",
                                                     kvs.get(pre + "attention.head_count",
                                                             kvs.get("attention.head_count", 8))))),
           "vocab_size": (len(toks) if isinstance(toks, list) and toks else 32000),
           "tie_word_embeddings": False,
           "rms_norm_eps": 1e-5,
           "rope_theta": (lambda rb: float(rb) if isinstance(rb, (int, float)) else 10000.0)(
               kvs.get(pre + "rope.freq_base", kvs.get("rope.freq_base", 10000.0)))}
    return cfg


class GGUFAdapter:
    name = "gguf"
    extensions = (".gguf",)

    @staticmethod
    def probe(path: str) -> bool:
        if path.endswith(".gguf"):
            return True
        try:
            with open(path, "rb") as f:
                return struct.unpack("<I", f.read(4))[0] == GGUF_MAGIC
        except Exception:
            return False

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        with open(path, "rb") as f:
            magic, version = struct.unpack("<II", f.read(8))
            if magic != GGUF_MAGIC:
                raise ValueError(f"bad GGUF magic {magic:#x}")
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            kvs: dict[str, object] = {}
            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", f.read(8))
                key = f.read(klen).decode()
                (t,) = struct.unpack("<I", f.read(4))
                kvs[key] = _read_value(f, t)
            infos = []
            for _ in range(n_tensors):
                (nlen,) = struct.unpack("<Q", f.read(8))
                name = f.read(nlen).decode()
                ndim = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack("<" + "Q" * ndim, f.read(8 * ndim))
                (tid,) = struct.unpack("<I", f.read(4))
                (off,) = struct.unpack("<Q", f.read(8))
                infos.append((name, list(dims), tid, off))
            base = f.tell()
            f.seek(0, 2)
            total = f.tell()
            f.seek(base)
            data = f.read(total - base)
            bufs: dict[str, np.ndarray] = {}
            bpw: dict[str, float] = {}
            for name, dims, tid, off in infos:
                n = int(np.prod(dims))
                tname = GGML_TYPE_REV.get(tid, "F32")
                hf = ggml_to_hf_name(name)
                # NOTE: real-GGUF dims are stored reversed ([in,out]); the HF
                # array is reshape(bytes, reversed(dims)) with NO transpose.
                # (reshape(recorded)+transpose is a different permutation and
                # silently scrambles weights — caught by exact-value tests.)
                rshape = list(reversed(dims)) if hf is not None else dims
                if tname == "Q8_0":
                    nbytes = ((n + Q8_0 - 1) // Q8_0) * 34
                    arr = dequantize_q8_0(data[off:off + nbytes], n)
                    bits = 8.0625
                elif tname == "Q4_0":
                    nbytes = ((n + Q4_0 - 1) // Q4_0) * 18
                    arr = dequantize_q4_0(data[off:off + nbytes], n)
                    bits = 4.5
                elif tname == "F16":
                    nbytes = n * 2
                    arr = np.frombuffer(data[off:off + nbytes],
                                        dtype=np.float16).astype(np.float32)
                    bits = 16.0
                else:
                    nbytes = n * 4
                    arr = np.frombuffer(data[off:off + nbytes],
                                        dtype=np.float32).copy()
                    bits = 32.0
                arr = arr.reshape(rshape)
                if hf is not None:
                    name = hf
                bufs[name] = arr
                bpw[name] = bits
        spec = _spec_from_kv(kvs)
        pages, dense = _build_pages(bufs, spec)
        return LoadedModel(spec=spec, tensors=TensorTable(bufs, bpw), expert_pages=pages,
                           dense_pages=dense, format_name="gguf", source_file=path,
                           notes=[f"GGUF v{version}; quantized payload parsed"])

    @staticmethod
    @staticmethod
    def save(path: str, tensors: dict[str, np.ndarray], ttype: str = "Q8_0",
             metadata: dict | None = None) -> str:
        _rewrite(path, tensors, ttype, kvs_from_metadata(metadata))
        return path


def kvs_from_metadata(metadata: dict | None) -> list[tuple[str, int, object]]:
    """Turn a python metadata dict into (key, gguf_type_id, value) triples."""
    out: list[tuple[str, int, object]] = []
    if metadata is None:
        return out
    for k, v in metadata.items():
        if isinstance(v, str):
            out.append((k, 8, v))
        elif isinstance(v, bool):
            out.append((k, 7, v))
        elif isinstance(v, int):
            out.append((k, 4, v))
        elif isinstance(v, float):
            out.append((k, 6, v))
        else:
            out.append((k, 8, str(v)))
    return out

def _rewrite(path, tensors, ttype, kvs):
    """Second pass: emit header with correct offsets then data (keeps it simple)."""
    w = _Writer()
    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(tensors))
    w.u64(len(kvs))
    for k, t, v in kvs:
        w.keyval(k, t, v)
    payloads = []
    for name, arr in tensors.items():
        if ttype == "Q8_0":
            blob, _ = quantize_q8_0(arr)
            tid = GGML_TYPE["Q8_0"]
        elif ttype == "Q4_0":
            blob, _ = quantize_q4_0(arr)
            tid = GGML_TYPE["Q4_0"]
        elif ttype == "F16":
            blob = np.ascontiguousarray(arr, dtype=np.float16).tobytes()
            tid = GGML_TYPE["F16"]
        else:
            blob = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            tid = GGML_TYPE["F32"]
        payloads.append((name, arr, tid, blob))
    off = 0
    for name, arr, tid, blob in payloads:
        w.str(name)
        w.u32(arr.ndim)
        for d in arr.shape:
            w.u64(int(d))
        w.u32(tid)
        w.u64(off)
        off += len(blob)
    hdr = w.bytes()
    with open(path, "wb") as f:
        f.write(hdr)
        for _, _, _, blob in payloads:
            f.write(blob)


def _read_value(f, t: int):
    if t == 8:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, et) for _ in range(n)]
    if t in (0, 2, 4, 10):  # unsigned
        return struct.unpack(_FMT[GGUF_VALUE[t]], f.read(struct.calcsize(_FMT[GGUF_VALUE[t]])))[0]
    if t in (1, 3, 5, 11):
        return struct.unpack(_FMT[GGUF_VALUE[t]], f.read(struct.calcsize(_FMT[GGUF_VALUE[t]])))[0]
    if t == 7:  # bool is u8
        return struct.unpack("<B", f.read(1))[0] != 0
    return struct.unpack(_FMT[GGUF_VALUE[t]], f.read(struct.calcsize(_FMT[GGUF_VALUE[t]])))[0]


def _spec_from_kv(kvs: dict) -> ModelSpec:
    def _g(k, d):
        return kvs.get(k, d)
    cfg = kvs.get("lumina.config", {})
    arch = str(cfg.get("arch", _g("general.architecture", "swiglu-moe")))
    spec = ModelSpec(
        name=str(_g("general.name", "gguf-model")),
        arch=arch,
        n_layer=int(cfg.get("n_layer", _g("lumina.n_layer", 1))),
        n_expert=int(cfg.get("n_expert", _g("lumina.n_expert", 1))),
        d_model=int(cfg.get("d_model", _g("lumina.d_model", 64))),
        d_ff=int(cfg.get("d_ff", _g("lumina.d_ff", 128))),
        top_k=int(cfg.get("top_k", _g("lumina.top_k", 1))),
        vocab=int(cfg.get("vocab", _g("lumina.vocab", 512))),
    )
    # chat-ready config distilled from GGUF metadata (no tokenizer list kept)
    try:
        spec.extra["gguf_chat"] = gguf_chat_config(kvs, arch)
    except Exception:
        pass
    return spec


def _build_pages(bufs: dict[str, np.ndarray], spec: ModelSpec):
    pages: dict[str, ExpertPage] = {}
    dense: dict[str, TensorContainer] = {}
    TC = TensorContainer
    for l in range(spec.n_layer):
        for tag, prefix in (("shared", f"layers.{l}.shared"), ("shared", f".{l}.shared")):
            got = {k: v for k, v in bufs.items() if prefix in k}
            if got:
                pages[f"L{l}-shared"] = ExpertPage(f"L{l}-shared", {k: TC(v) for k, v in got.items()}, l, arch="shared")
                break
        for e in range(spec.n_expert):
            got = {k: v for k, v in bufs.items()
                   if (f"layers.{l}.experts.{e}." in k or f".{l}.experts.{e}." in k)}
            if got:
                pages[f"L{l}-e{e}"] = ExpertPage(f"L{l}-e{e}", {k: TC(v) for k, v in got.items()}, l)
    for name, arr in bufs.items():
        if ("experts." in name) or ("shared" in name) or ("router" in name):
            continue
        if arr.ndim >= 1 and arr.dtype.kind == "f":
            dense[name] = TC(arr)
    return pages, dense


register(GGUFAdapter)
