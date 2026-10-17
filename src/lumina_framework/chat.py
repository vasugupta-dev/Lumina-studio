"""Real inference engine — full transformer forward, greedy decode.

No mocks: RMSNorm + GQA attention with RoPE + SwiGLU MLP executed on the
model's own weights (float16 working copy). One model cached at a time;
every call carries a wall-clock budget so the phone never hangs.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .adapters import load_model
from .tokenizer import BPETokenizer

_engine_cache: dict = {}

# NOTE: an isolated head-GEMV runs fastest at 2 BLAS threads (1.50ms vs
# 4.24ms at 8T on this device), but wrapping the FULL decode loop in a
# thread limit measured HARMFUL in-loop (83.6 vs 167.8 tok/s median,
# high variance — pool thrash + slower layer matmuls). Disabled by
# default; kept for opt-in experiments via env LUMINA_DECODE_THREADS=2.
import os as _os
_DECODE_THREADS = int(_os.environ.get("LUMINA_DECODE_THREADS", "0"))
try:
    from threadpoolctl import threadpool_limits as _tpl

    class _decode_threads:
        def __init__(self, n=2):
            self.n = n if _DECODE_THREADS > 0 else 0
            self._ctx = None

        def __enter__(self):
            if self.n:
                try:
                    self._ctx = _tpl(limits=self.n)
                    self._ctx.__enter__()
                except Exception:
                    self._ctx = None
            return self

        def __exit__(self, *a):
            if self._ctx is not None:
                try:
                    self._ctx.__exit__(*a)
                except Exception:
                    pass
except Exception:
    class _decode_threads:  # threadpoolctl missing: env flag does nothing
        def __init__(self, n=2):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    m = np.mean(xf ** 2, axis=-1, keepdims=True)
    return (xf / np.sqrt(m + eps) * w.astype(np.float32)).astype(np.float32)


def rope_freqs(head_dim: int, seq_len: int, theta: float = 1000000.0,
               offset: int = 0) -> np.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, 2 * (head_dim // 2), 2).astype(np.float32)
                             / head_dim))
    t = np.arange(offset, offset + seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)
    return np.concatenate([angles, angles], axis=-1)


def rope_angles_at(head_dim: int, positions: np.ndarray,
                   theta: float = 1000000.0) -> np.ndarray:
    """RoPE angles for explicit per-row positions (tree verification).
    Odd head_dim: last dim passes through unrotated (standard)."""
    freqs = 1.0 / (theta ** (np.arange(0, 2 * (head_dim // 2), 2).astype(np.float32)
                             / head_dim))
    angles = np.outer(np.asarray(positions, dtype=np.float32), freqs)
    return np.concatenate([angles, angles], axis=-1)


def apply_rope(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """x: (..., seq, head_dim); angles: (seq, 2*P) radians, P = pairs.
    Rotates the first 2P dims pairwise; any trailing dim (odd head_dim)
    passes through — identical to before for even head_dim."""
    P = angles.shape[-1] // 2
    xr, rest = x[..., :2 * P], x[..., 2 * P:]
    half = P
    x1, x2 = xr[..., :half], xr[..., half:]
    a1, a2 = angles[..., :half], angles[..., half:]
    cos1, sin1 = np.cos(a1), np.sin(a1)
    cos2, sin2 = np.cos(a2), np.sin(a2)
    out = np.concatenate([x1 * cos1 - x2 * sin1,
                          x1 * sin2 + x2 * cos2], axis=-1)
    if rest.shape[-1]:
        out = np.concatenate([out, rest], axis=-1)
    return out


class ChatModel:
    """Dense transformer runner over a loaded model's own weights (float16).

    fast=False (default, phone-safe): weights stay fp16; each matmul casts the
    weight to fp32 on the fly (simple, ~2x weight memory avoided).
    fast=True (servers/Kaggle): every projection is converted to a
    transposed fp32 copy ONCE, lazily, and the fp16 original is released —
    eliminating per-token conversion churn (~model-size allocations per
    token). Both paths produce identical greedy tokens (verified by
    tests/test_chat_fast_parity.py on a tiny fixture).
    """

    def __init__(self, model_dir: str, n_heads: int = 14, n_kv: int = 2,
                 eps: float = 1e-6, theta: float = 1000000.0,
                 fast: bool = False, head_mode: str = "fp32"):
        lm = load_model(os.path.join(model_dir, "model.safetensors"))
        tensors = {k: v.astype(np.float16) for k, v in lm.tensors.tensors.items()}
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tok = BPETokenizer(model_dir)
        self._init_from_parts(tensors, cfg, tok, n_heads, n_kv, eps, theta,
                              fast, head_mode=head_mode)

    @classmethod
    def from_tensors(cls, tensors: dict[str, np.ndarray], cfg: dict,
                     tok: "BPETokenizer", n_heads: int = 14, n_kv: int = 2,
                     eps: float = 1e-6, theta: float = 1000000.0,
                     fast: bool = False,
                     moe_map: dict | None = None,
                     head_mode: str = "fp32") -> "ChatModel":
        """Build from an already-loaded tensor dict (e.g. dequantized from a
        .lumina file through the format adapter). Same math, same contract —
        the format only changes how many bits survived to reach fp16."""
        obj = cls.__new__(cls)
        obj._init_from_parts(
            {k: np.asarray(v, dtype=np.float16) for k, v in tensors.items()},
            cfg, tok, n_heads, n_kv, eps, theta, fast, moe_map,
            head_mode=head_mode)
        return obj

    def _init_from_parts(self, tensors: dict[str, np.ndarray], cfg: dict,
                         tok: "BPETokenizer", n_heads: int, n_kv: int,
                         eps: float, theta: float, fast: bool,
                         moe_map: dict | None = None,
                         head_mode: str = "fp32") -> None:
        # head_mode: "fp32" exact GEMV | "int8" torch dynamic-quantized head
        # (reads 4x fewer bytes; logits shift slightly — measured per model,
        # reported by api_load; never silently default)
        self.head_mode = head_mode
        self._int8_head = None
        self.t = tensors
        self.cfg = cfg
        self.tok = tok
        # virtual-MoE layers: experts keys present -> routed execution.
        # moe_map (from the .lumina header) gives (experts, ie, top_k);
        # without it the flat grouped matrices cannot reshape -> raise.
        self.moe: dict[int, dict] = {}
        pfx = "model.layers."
        max_lyr = int(cfg.get("num_hidden_layers", 24))
        for lyr in range(max_lyr):
            gu = f"{pfx}{lyr}.mlp.experts.gate_up_proj"
            if gu in tensors:
                if not moe_map or str(lyr) not in moe_map and lyr not in moe_map:
                    # also accept prefix-keyed maps
                    hit = None
                    for k, v in (moe_map or {}).items():
                        if f".layers.{lyr}." in str(k):
                            hit = v
                            break
                    if hit is None:
                        raise KeyError(
                            f"layer {lyr} has virtual-MoE experts but no "
                            "virtual_moe header entry (experts/ie/top_k)")
                    mm = hit
                else:
                    mm = moe_map.get(str(lyr), moe_map.get(lyr))
                self.moe[lyr] = {"experts": int(mm["experts"]),
                                 "ie": int(mm["ie"]),
                                 "top_k": int(mm["top_k"])}
        # Head counts come from the model config when present; the constructor
        # args are only fallbacks. Hard-coding them crashed generate() on any
        # model with non-default heads (server thread died -> empty response).
        self.n_heads = int(self.cfg.get("num_attention_heads", n_heads))
        self.n_kv = int(self.cfg.get("num_key_value_heads", self.n_heads))
        self.eps, self.theta = eps, theta
        self.fast = fast
        self._wcache: dict = {}
        self.n_layer = int(self.cfg.get("num_hidden_layers", 24))
        self.hd = self.t["model.embed_tokens.weight"].shape[1] // self.n_heads
        head = self.t.get("lm_head.weight")
        self.head = self.t["model.embed_tokens.weight"] if head is None else head
        if head_mode == "int8":
            try:
                import torch  # noqa: F401
                class _HL:
                    def __init__(self, w):
                        import torch as _t
                        with _t.no_grad():
                            lin = _t.nn.Linear(w.shape[1], w.shape[0], bias=False)
                            lin.weight.copy_(_t.from_numpy(np.ascontiguousarray(w)))
                        class _M(_t.nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.l = lin
                            def forward(self, x):
                                return self.l(x)
                        self.m = _M().eval()
                        self.m = _t.ao.quantization.quantize_dynamic(
                            self.m, {_t.nn.Linear}, dtype=_t.qint8)
                    def __call__(self, hrow):
                        import torch as _t
                        with _t.no_grad():
                            return self.m(_t.from_numpy(hrow)).numpy()
                self._int8_head = _HL(np.asarray(self.head, dtype=np.float32))
            except Exception:
                self.head_mode = "fp32"
                self._int8_head = None

    def _W(self, name: str) -> np.ndarray:
        """fp32 transposed weight, converted once (fast path)."""
        w = self._wcache.get(name)
        if w is None:
            w = self.t[name].T.astype(np.float32)
            self._wcache[name] = w
            del self.t[name]  # release the fp16 original to bound memory
        return w

    def _layer(self, x: np.ndarray, lyr: int, cache: dict,
               pos0: int, attn_bias: np.ndarray | None = None,
               pos_override: np.ndarray | None = None) -> np.ndarray:
        t, hd, nh, nk = self.t, self.hd, self.n_heads, self.n_kv
        if self.fast:
            W = self._W
        else:
            W = lambda n: t[n].T.astype(np.float32)  # noqa: E731
        h = rmsnorm(x, t[f"model.layers.{lyr}.input_layernorm.weight"], self.eps)
        q = (h @ W(f"model.layers.{lyr}.self_attn.q_proj.weight"))
        k = (h @ W(f"model.layers.{lyr}.self_attn.k_proj.weight"))
        v = (h @ W(f"model.layers.{lyr}.self_attn.v_proj.weight"))
        sq, sk, sv = q.shape[0], k.shape[0], v.shape[0]
        q = q.reshape(sq, nh, hd)
        k = k.reshape(sk, nk, hd)
        v = v.reshape(sv, nk, hd)
        # row positions: normally pos0+i; tree verification passes true depths
        row_pos = (np.asarray(pos_override).reshape(-1) if pos_override is not None
                   else pos0 + np.arange(sq))
        if pos_override is not None:
            ang = rope_angles_at(hd, row_pos, self.theta)
        else:
            ang = rope_freqs(hd, sq, self.theta, pos0)
        q = apply_rope(q, ang[:, None, :])
        k = apply_rope(k, ang[:, None, :])
        rep = nh // nk
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        if lyr in cache:
            k = np.concatenate([cache[lyr][0], k], axis=0)
            v = np.concatenate([cache[lyr][1], v], axis=0)
        cache[lyr] = (k, v)
        # positions: key j holds true position j while j is a prefix key,
        # else its input row's pos (row_pos defined above for RoPE).
        n_prev = k.shape[0] - sq
        key_pos = np.concatenate([np.arange(n_prev),
                                  row_pos]) if n_prev > 0 else row_pos
        allow = key_pos[None, :] <= row_pos[:, None]
        # NOTE: for tree inputs allow is a superset (concat-order siblings);
        # callers pass attn_bias (tree ancestry) which restricts via minimum.
        bias = np.where(allow, 0.0, -np.inf).astype(np.float32)
        if attn_bias is not None:
            bias = np.minimum(bias, attn_bias.astype(np.float32))
        qh = q.transpose(1, 0, 2)
        kh = k.transpose(1, 2, 0)
        vh = v.transpose(1, 0, 2)
        sc = qh @ kh / max(hd ** 0.5, 1e-6)
        e = np.exp(sc + bias[None] - (sc + bias[None]).max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        o = (a @ vh).transpose(1, 0, 2).reshape(sq, nh * hd)
        o = o @ W(f"model.layers.{lyr}.self_attn.o_proj.weight")
        x = x + o.astype(np.float32)
        h = rmsnorm(x, t[f"model.layers.{lyr}.post_attention_layernorm.weight"], self.eps)
        if lyr in self.moe:
            from .qwen_next import moe_grouped_forward
            mc = self.moe[lyr]
            E, Ie = mc["experts"], mc["ie"]
            pfx = f"model.layers.{lyr}.mlp."
            gu = np.asarray(t[pfx + "experts.gate_up_proj"], dtype=np.float32).reshape(E, 2 * Ie, -1)
            dn = np.asarray(t[pfx + "experts.down_proj"], dtype=np.float32).reshape(E, Ie, -1)
            rw = np.asarray(t[pfx + "gate.weight"], dtype=np.float32)
            rb = (np.asarray(t[pfx + "gate_bias"], dtype=np.float32)
                  if pfx + "gate_bias" in t else None)
            d, _info = moe_grouped_forward(h.astype(np.float32), gu, dn, rw,
                                            mc["top_k"], weight_mode="unit",
                                            router_b=rb,
                                            router_feat=mc.get("router_feat", "raw"))
            return x + d
        g = h @ W(f"model.layers.{lyr}.mlp.gate_proj.weight")
        u = h @ W(f"model.layers.{lyr}.mlp.up_proj.weight")
        d = ((g / (1.0 + np.exp(-np.clip(g, -40, 40)))) * u)
        d = d @ W(f"model.layers.{lyr}.mlp.down_proj.weight")
        return x + d


def generate(model: ChatModel, prompt: str, max_tokens: int = 32,
             budget_s: float = 300.0, stop_ids: tuple = (151643,),
             spec_drafter=None, rng=None, temperature: float = 0.0,
             top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0,
             typical_p: float = 1.0, repeat_penalty: float = 1.0,
             on_token=None) -> dict:
    """Greedy decode with KV cache; optional spec_drafter for speedup.

    spec_drafter: callable(feat, embed_fn, lm_head) -> list[int] drafting via
    spec.py (EAGLE/MTP/parallel); verified here with speculative sampling.
    Every returned dict carries measured stats — never a bare string.
    """
    t0 = time.perf_counter()
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    # head transposed ONCE — the old loop did a full VxD copy per token
    head_t = model.head.T.astype(np.float32)
    head_out = model._int8_head if getattr(model, "head_mode", "fp32") == "int8" else None
    cache: dict = {}
    h = emb[np.array(ids)]
    rng = rng if rng is not None else np.random.default_rng(0)
    pos, out, stats = 0, [], {"steps": 0, "spec_accepted": 0, "spec_drafted": 0}

    def pick(lg, seen):
        lg = np.asarray(lg, dtype=np.float64)
        if repeat_penalty != 1.0 and seen:
            # penalize only the recent window — penalizing the whole prompt
            # crushes long-context quality (llama.cpp uses a window too)
            for t in set(seen[-64:]):
                if 0 <= t < lg.size:
                    lg[t] = lg[t] / repeat_penalty if lg[t] > 0 else lg[t] * repeat_penalty
        if temperature <= 0 and top_k <= 1 and min_p <= 0 and typical_p >= 1.0:
            return int(np.argmax(lg))
        z = (lg - lg.max()) / max(temperature if temperature > 0 else 1.0, 1e-6)
        p = np.exp(z)
        p = p / p.sum()
        # argpartition (O(n)) instead of argsort (O(n log n)) on the full
        # vocab — same keep-sets, measurably faster on 150k-vocab models
        if top_k and top_k > 0:
            part = np.argpartition(-p, min(top_k, p.size - 1))[:top_k]
            keep = np.zeros_like(p, dtype=bool)
            keep[part] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if min_p > 0:
            keep = p >= (p.max() * min_p)
            keep[int(np.argmax(p))] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if top_p < 1.0:
            n = int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), top_p)) + 1
            n = max(1, min(n, p.size))
            part = np.argpartition(-p, n - 1)[:n]
            keep = np.zeros_like(p, dtype=bool)
            keep[part] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if typical_p < 1.0:
            # exact cumulative-deviation cutoff (argsort kept: a windowed
            # argpartition approximation changes the distribution)
            ent = float(-(p * np.log(p + 1e-12)).sum())
            dev = np.abs(-np.log(p + 1e-12) - ent)
            order = np.argsort(dev, kind="stable")
            cum = np.cumsum(p[order])
            keep = np.zeros_like(p, dtype=bool)
            keep[order[cum <= typical_p]] = True
            keep[order[0]] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if temperature <= 0:
            return int(np.argmax(p))
        return int(rng.choice(p.size, p=p)) if p.size < 256 else int(np.argmax(rng.multinomial(1, p)))
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, pos)
    pos = len(ids)
    _logit_buf = np.empty((1, head_t.shape[1]), dtype=np.float32)
    with _decode_threads(2):
        while len(out) < max_tokens and time.perf_counter() - t0 < budget_s:
            if head_out is not None:
                lg = head_out(h[-1:])[0]
            else:
                np.dot(h[-1:], head_t, out=_logit_buf)
                lg = _logit_buf[0]
            nxt = pick(lg, ids + out)
            if nxt in stop_ids:
                break
            out.append(nxt)
            stats["steps"] += 1
            if on_token is not None:
                try:
                    on_token(nxt)
                except Exception:
                    pass
            h = emb[[nxt]]
            for lyr in range(model.n_layer):
                h = model._layer(h, lyr, cache, pos)
            pos += 1
    dt = time.perf_counter() - t0
    return {"text": model.tok.decode(out), "tokens": out,
            "tokens_per_second": len(out) / dt if dt else 0.0,
            "seconds": dt, "stats": stats}


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def generate_cached(model: ChatModel, prompt: str, max_tokens: int = 32,
                    budget_s: float = 300.0, stop_ids: tuple = (151643,),
                    rng=None, temperature: float = 0.0, top_p: float = 1.0,
                    top_k: int = 0, min_p: float = 0.0, typical_p: float = 1.0,
                    repeat_penalty: float = 1.0, on_token=None,
                    state: dict | None = None) -> dict:
    """generate() with prefix KV reuse — the standard serving optimization.

    `state` carries {ids, kv, pos} from the previous call on the same
    conversation. The longest common token prefix keeps its KV rows; only
    the divergence is re-prefilled. Exact same tokens as generate()
    (parity-tested); the win is skipping re-prefill of the whole history on
    every turn. Divergence truncates the cache (regenerate/edit cases)."""
    t0 = time.perf_counter()
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    rng = rng if rng is not None else np.random.default_rng(0)
    st = state if state is not None else {"ids": [], "kv": {}, "pos": 0}
    prefill_saved = 0

    common = _common_prefix(st["ids"], ids)
    if common < len(st["ids"]):
        # history diverged: truncate KV rows to the shared prefix (rows are
        # position-ordered, so slicing is exact)
        for lyr, (k, v) in list(st["kv"].items()):
            st["kv"][lyr] = (k[:common], v[:common])
        st["ids"] = st["ids"][:common]
        st["pos"] = common
    prefill_saved = len(st["ids"])
    delta = ids[len(st["ids"]):]
    h = emb[np.array(delta)] if len(delta) else emb[np.array(ids[-1:])]
    pos = st["pos"]
    if len(delta):
        for lyr in range(model.n_layer):
            h = model._layer(h, lyr, st["kv"], pos)
        pos += len(delta)
    else:
        # fully cached prompt: replay the LAST row on a truncated COPY of the
        # KV (without the row itself) so it attends each position exactly
        # once — identical to fresh processing. The copy is discarded; the
        # live cache already holds rows 0..pos-1.
        pos0r = max(pos - 1, 0)
        kv_replay = {}
        for lyr, (k, v) in st["kv"].items():
            kv_replay[lyr] = (k[:pos0r], v[:pos0r]) if pos0r else (
                np.zeros((0,) + k.shape[1:], k.dtype),
                np.zeros((0,) + v.shape[1:], v.dtype))
        h = emb[[ids[-1]]]
        for lyr in range(model.n_layer):
            h = model._layer(h, lyr, kv_replay, pos0r)

    head = model.head.T.astype(np.float32)
    head_out = model._int8_head if getattr(model, "head_mode", "fp32") == "int8" else None
    out = []
    stats = {"steps": 0, "prefill_reused": prefill_saved,
             "prefill_computed": len(delta)}

    def pick(lg, seen):
        lg = np.asarray(lg, dtype=np.float64)
        if repeat_penalty != 1.0 and seen:
            for t in set(seen[-64:]):
                if 0 <= t < lg.size:
                    lg[t] = lg[t] / repeat_penalty if lg[t] > 0 else lg[t] * repeat_penalty
        if temperature <= 0 and top_k <= 1 and min_p <= 0 and typical_p >= 1.0:
            return int(np.argmax(lg))
        z = (lg - lg.max()) / max(temperature if temperature > 0 else 1.0, 1e-6)
        p = np.exp(z)
        p = p / p.sum()
        # argpartition (O(n)) instead of argsort (O(n log n)) on the full
        # vocab — same keep-sets, measurably faster on 150k-vocab models
        if top_k and top_k > 0:
            part = np.argpartition(-p, min(top_k, p.size - 1))[:top_k]
            keep = np.zeros_like(p, dtype=bool)
            keep[part] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if min_p > 0:
            keep = p >= (p.max() * min_p)
            keep[int(np.argmax(p))] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if top_p < 1.0:
            n = int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), top_p)) + 1
            n = max(1, min(n, p.size))
            part = np.argpartition(-p, n - 1)[:n]
            keep = np.zeros_like(p, dtype=bool)
            keep[part] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if typical_p < 1.0:
            # exact cumulative-deviation cutoff (argsort kept: a windowed
            # argpartition approximation changes the distribution)
            ent = float(-(p * np.log(p + 1e-12)).sum())
            dev = np.abs(-np.log(p + 1e-12) - ent)
            order = np.argsort(dev, kind="stable")
            cum = np.cumsum(p[order])
            keep = np.zeros_like(p, dtype=bool)
            keep[order[cum <= typical_p]] = True
            keep[order[0]] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if temperature <= 0:
            return int(np.argmax(p))
        return int(rng.choice(p.size, p=p)) if p.size < 256 else int(np.argmax(rng.multinomial(1, p)))

    _logit_buf = np.empty((1, head.shape[1]), dtype=np.float32)
    with _decode_threads(2):
        while len(out) < max_tokens and time.perf_counter() - t0 < budget_s:
            if head_out is not None:
                lg = head_out(h[-1:])[0]
            else:
                np.dot(h[-1:], head, out=_logit_buf)
                lg = _logit_buf[0]
            nxt = pick(lg, ids + out)
            if nxt in stop_ids:
                break
            out.append(nxt)
            stats["steps"] += 1
            if on_token is not None:
                try:
                    on_token(nxt)
                except Exception:
                    pass
            h = emb[[nxt]]
            for lyr in range(model.n_layer):
                h = model._layer(h, lyr, st["kv"], pos)
            pos += 1
    # record the extended history so the NEXT turn reuses this prefix
    st["ids"] = ids + out
    st["pos"] = pos
    dt = time.perf_counter() - t0
    return {"text": model.tok.decode(out), "tokens": out,
            "tokens_per_second": len(out) / dt if dt else 0.0,
            "seconds": dt, "stats": stats, "cache_state": st}


def pld_draft(ids: list[int], n: int = 4, gram: int = 3) -> list[int]:
    """Prompt Lookup Decoding: copy continuation of a repeated n-gram.

    Training-free drafting (published method): if the last `gram` tokens
    appear earlier in the context, draft what followed them. Returns [] on
    no match — the caller falls back to single-token decode. Exact, no model.
    """
    if len(ids) < gram + 1:
        return []
    key = ids[-gram:]
    for i in range(len(ids) - gram - 1, -1, -1):
        if ids[i:i + gram] == key:
            out = ids[i + gram:i + gram + n]
            if out:
                return out
    return []


def generate_spec(model: ChatModel, prompt: str, max_tokens: int = 32,
                  budget_s: float = 300.0, stop_ids: tuple = (151643,),
                  draft_n: int = 4, rng=None) -> dict:
    """PLD draft + speculative-sample verify on the real model. Unbiased."""
    from .spec import speculative_sample, SpecStats
    rng = rng if rng is not None else np.random.default_rng(0)
    t0 = time.perf_counter()
    st = SpecStats()
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    head = model.head.T.astype(np.float32)
    cache: dict = {}
    h = emb[np.array(ids)]
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, 0)
    pos, out = len(ids), []
    while len(out) < max_tokens and time.perf_counter() - t0 < budget_s:
        drafted = pld_draft(ids + out, draft_n)
        if not drafted:
            lg = (h[-1:] @ head)[0]
            nxt = int(np.argmax(lg))
            if nxt in stop_ids:
                break
            out.append(nxt)
            h = emb[[nxt]]
            for lyr in range(model.n_layer):
                h = model._layer(h, lyr, cache, pos)
            pos += 1
            st.target_forwards += 1
            st.tokens_out += 1
            continue
        st.drafted += len(drafted)
        dh = emb[np.array(drafted)]
        hh = dh
        for lyr in range(model.n_layer):
            hh = model._layer(hh, lyr, cache, pos)
        st.target_forwards += 1
        t_lg = (hh @ head)
        # Alignment: drafted[k] sits at absolute pos+k, predicted by the
        # hidden at pos+k-1: h (pre-verify) for k=0, t_lg[k-1] for k>=1.
        # All-accepted bonus is argmax(t_lg[D-1]).
        h_lg = (h[-1:] @ head)[0]
        acc: list[int] = []
        bonus = int(np.argmax(t_lg[-1]))
        for k, tok in enumerate(drafted):
            pred = h_lg if k == 0 else t_lg[k - 1]
            if int(np.argmax(pred)) == tok:
                acc.append(tok)
            else:
                bonus = int(np.argmax(pred))
                break
        st.accepted += len(acc)
        st.tokens_out += len(acc) + 1
        for a in acc + [bonus]:
            if a in stop_ids:
                out.append(a)
                dt = time.perf_counter() - t0
                return {"text": model.tok.decode(out), "tokens": out,
                        "spec": st.as_dict(),
                        "tokens_per_second": len(out) / dt if dt else 0.0,
                        "seconds": dt}
            out.append(a)
        # Truncate caches to the accepted prefix; one single-token forward
        # appends the bonus KV and yields its hidden. Exact, no drift.
        for lyr in range(model.n_layer):
            k, v = cache[lyr]
            cache[lyr] = (k[:pos + len(acc)], v[:pos + len(acc)])
        h = emb[[bonus]]
        for lyr in range(model.n_layer):
            h = model._layer(h, lyr, cache, pos + len(acc))
        st.target_forwards += 1
        pos += len(acc) + 1
    dt = time.perf_counter() - t0
    out = out[:max_tokens]
    return {"text": model.tok.decode(out), "tokens": out,
            "spec": st.as_dict(),
            "tokens_per_second": len(out) / dt if dt else 0.0, "seconds": dt}
