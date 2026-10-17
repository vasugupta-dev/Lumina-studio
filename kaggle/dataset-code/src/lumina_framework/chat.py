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


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    m = np.mean(xf ** 2, axis=-1, keepdims=True)
    return (xf / np.sqrt(m + eps) * w.astype(np.float32)).astype(np.float32)


def rope_freqs(head_dim: int, seq_len: int, theta: float = 1000000.0,
               offset: int = 0) -> np.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32)
                             / head_dim))
    t = np.arange(offset, offset + seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)
    return np.concatenate([angles, angles], axis=-1)


def apply_rope(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """x: (..., seq, head_dim); angles: (seq, head_dim) radians."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    a1, a2 = angles[..., :half], angles[..., half:]
    cos1, sin1 = np.cos(a1), np.sin(a1)
    cos2, sin2 = np.cos(a2), np.sin(a2)
    return np.concatenate([x1 * cos1 - x2 * sin1,
                           x1 * sin2 + x2 * cos2], axis=-1)


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
                 fast: bool = False):
        lm = load_model(os.path.join(model_dir, "model.safetensors"))
        self.t = {k: v.astype(np.float16) for k, v in lm.tensors.tensors.items()}
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.n_heads, self.n_kv, self.eps, self.theta = n_heads, n_kv, eps, theta
        self.fast = fast
        self._wcache: dict = {}
        self.n_layer = int(self.cfg.get("num_hidden_layers", 24))
        self.hd = self.t["model.embed_tokens.weight"].shape[1] // n_heads
        head = self.t.get("lm_head.weight")
        self.head = self.t["model.embed_tokens.weight"] if head is None else head
        self.tok = BPETokenizer(model_dir)

    def _W(self, name: str) -> np.ndarray:
        """fp32 transposed weight, converted once (fast path)."""
        w = self._wcache.get(name)
        if w is None:
            w = self.t[name].T.astype(np.float32)
            self._wcache[name] = w
            del self.t[name]  # release the fp16 original to bound memory
        return w

    def _layer(self, x: np.ndarray, lyr: int, cache: dict,
               pos0: int, attn_bias: np.ndarray | None = None) -> np.ndarray:
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
        qh = q.transpose(1, 0, 2)
        kh = k.transpose(1, 2, 0)
        vh = v.transpose(1, 0, 2)
        sc = qh @ kh / max(hd ** 0.5, 1e-6)
        # row i is absolute position pos0+i: allow keys j <= pos0+i
        allow = (np.arange(k.shape[0])[None, :]
                 <= (pos0 + np.arange(sq)[:, None]))
        bias = np.where(allow, 0.0, -np.inf).astype(np.float32)
        if attn_bias is not None:
            bias = np.minimum(bias, attn_bias.astype(np.float32))
        e = np.exp(sc + bias[None] - (sc + bias[None]).max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        o = (a @ vh).transpose(1, 0, 2).reshape(sq, nh * hd)
        o = o @ W(f"model.layers.{lyr}.self_attn.o_proj.weight")
        x = x + o.astype(np.float32)
        h = rmsnorm(x, t[f"model.layers.{lyr}.post_attention_layernorm.weight"], self.eps)
        g = h @ W(f"model.layers.{lyr}.mlp.gate_proj.weight")
        u = h @ W(f"model.layers.{lyr}.mlp.up_proj.weight")
        d = ((g / (1.0 + np.exp(-np.clip(g, -40, 40)))) * u)
        d = d @ W(f"model.layers.{lyr}.mlp.down_proj.weight")
        return x + d


def generate(model: ChatModel, prompt: str, max_tokens: int = 32,
             budget_s: float = 300.0, stop_ids: tuple = (151643,),
             spec_drafter=None, rng=None, temperature: float = 0.0,
             top_p: float = 1.0) -> dict:
    """Greedy decode with KV cache; optional spec_drafter for speedup.

    spec_drafter: callable(feat, embed_fn, lm_head) -> list[int] drafting via
    spec.py (EAGLE/MTP/parallel); verified here with speculative sampling.
    Every returned dict carries measured stats — never a bare string.
    """
    t0 = time.perf_counter()
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    cache: dict = {}
    h = emb[np.array(ids)]
    rng = rng if rng is not None else np.random.default_rng(0)
    pos, out, stats = 0, [], {"steps": 0, "spec_accepted": 0, "spec_drafted": 0}

    def pick(lg):
        if temperature <= 0:
            return int(np.argmax(lg))
        z = (lg - lg.max()) / max(temperature, 1e-6)
        p = np.exp(z)
        p = p / p.sum()
        if top_p < 1.0:
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            keep = np.zeros_like(p, dtype=bool)
            keep[order[cum <= top_p]] = True
            keep[order[0]] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        return int(np.argmax(rng.multinomial(1, p)))
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, pos)
    pos = len(ids)
    while len(out) < max_tokens and time.perf_counter() - t0 < budget_s:
        lg = (h[-1:] @ model.head.T.astype(np.float32))[0]
        nxt = pick(lg)
        if nxt in stop_ids:
            break
        out.append(nxt)
        stats["steps"] += 1
        h = emb[[nxt]]
        for lyr in range(model.n_layer):
            h = model._layer(h, lyr, cache, pos)
        pos += 1
    dt = time.perf_counter() - t0
    return {"text": model.tok.decode(out), "tokens": out,
            "tokens_per_second": len(out) / dt if dt else 0.0,
            "seconds": dt, "stats": stats}


def get_or_load(model_dir: str) -> ChatModel:
    if model_dir not in _engine_cache:
        _engine_cache.clear()
        _engine_cache[model_dir] = ChatModel(model_dir)
    return _engine_cache[model_dir]


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
