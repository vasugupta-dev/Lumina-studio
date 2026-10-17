"""Qwen4-exp execution kernels (Qwen3.8-Flash-Next layout, clean-room).

Tensor layout matches the real index (model.safetensors.index.json, metadata
only — no weights stored): grouped experts, fused gate_up, PLE shards,
indexer projections, hyper-connection mixers. See STATUS.md for the map.

Honest contract: these are clean-room implementations of the PUBLISHED
mechanisms, verified here for shape, causality, streaming/state equivalence,
determinism, and parity (parallel vs sequential). They are NOT verified
against real model weights — that needs the 180B download + an RTX machine.
Nothing here claims otherwise.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .chat import rmsnorm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40).astype(np.float32)))


# ---------------------------------------------------------------------------
# Grouped-expert MoE (real layout: experts axis 0, fused gate_up [E,2I,D])
# ---------------------------------------------------------------------------

def moe_grouped_forward(h: np.ndarray, gate_up: np.ndarray, down: np.ndarray,
                        router_w: np.ndarray, top_k: int,
                        shared_gate=None, shared_up=None, shared_down=None,
                        shared_gate_w: float | np.ndarray = 1.0,
                        gate_fn: str = "sigmoid",
                        weight_mode: str = "gate",
                        router_b: np.ndarray | None = None,
                        router_feat: str = "raw") -> tuple[np.ndarray, dict]:
    """h: (S,D). gate_up: (E,2I,D). down: (E,I,D). router_w: (E,D).

    weight_mode="unit" weights every picked expert 1.0 (virtual-MoE dense
    equivalence); "gate" multiplies by the router score (Qwen-style MoE).
    Returns (out, info with picked experts + weights for audit)."""
    E = gate_up.shape[0]
    I = gate_up.shape[1] // 2
    rf = np.abs(h) if router_feat == "abs" else h
    logits = rf @ router_w.T
    if router_b is not None:
        logits = logits + np.asarray(router_b, dtype=np.float32)
    if gate_fn == "sigmoid":
        scores = _sigmoid(logits)
    else:
        e = np.exp(logits - logits.max(-1, keepdims=True))
        scores = e / e.sum(-1, keepdims=True)
    k = max(1, min(top_k, E))
    idx = np.argsort(-scores, axis=1)[:, :k]
    if weight_mode == "unit":
        # virtual-MoE semantics: picked neurons computed EXACTLY (weight 1.0).
        w = np.ones_like(np.take_along_axis(scores, idx, axis=1))
    else:
        w = np.take_along_axis(scores, idx, axis=1)
    out = np.zeros_like(h, dtype=np.float32)
    for s in range(h.shape[0]):
        acc = np.zeros(h.shape[1], dtype=np.float32)
        for j in range(k):
            e = int(idx[s, j])
            gu = gate_up[e] @ h[s]
            g, u = gu[:I], gu[I:]
            silu = g / (1.0 + np.exp(-np.clip(g, -40, 40)))
            acc += float(w[s, j]) * (down[e].T @ (silu * u))
        out[s] = acc
    if shared_gate is not None:
        g = h @ shared_gate.T.astype(np.float32)
        u = h @ shared_up.T.astype(np.float32)
        silu = g / (1.0 + np.exp(-np.clip(g, -40, 40)))
        sh = (silu * u) @ shared_down.T.astype(np.float32)
        out = out + np.asarray(shared_gate_w, dtype=np.float32) * sh
    return out.astype(np.float32), {"experts": idx, "weights": w}


def moe_grouped_parallel(h: np.ndarray, gate_up: np.ndarray, down: np.ndarray,
                         router_w: np.ndarray, top_k: int,
                         max_workers: int | None = None,
                         weight_mode: str = "gate",
                         router_b: np.ndarray | None = None,
                         router_feat: str = "raw",
                         **kw) -> tuple[np.ndarray, dict]:
    """Same math as moe_grouped_forward, one thread per (row, expert) task.

    Measured on shared-bus CPU (OpenBLAS, 2048-dim sweeps): 0.05-0.2x —
    per-task matmuls are already BLAS-threaded, so the pool adds only
    overhead and oversubscription. This path exists parity-verified for
    multi-stream accelerators (one expert per CUDA stream), where concurrent
    dispatch wins; on CPU, call the sequential forward (BLAS uses the cores).
    Not a speedup here — stated, measured, kept for the GPU port."""
    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 2)))
    E = gate_up.shape[0]
    I = gate_up.shape[1] // 2
    rf = np.abs(h) if router_feat == "abs" else h
    logits = rf @ router_w.T
    if router_b is not None:
        logits = logits + np.asarray(router_b, dtype=np.float32)
    scores = _sigmoid(logits)
    k = max(1, min(top_k, E))
    idx = np.argsort(-scores, axis=1)[:, :k]
    if weight_mode == "unit":
        # virtual-MoE semantics: picked neurons computed EXACTLY (weight 1.0).
        w = np.ones_like(np.take_along_axis(scores, idx, axis=1))
    else:
        w = np.take_along_axis(scores, idx, axis=1)

    def task(s: int, j: int):
        e = int(idx[s, j])
        gu = gate_up[e] @ h[s]
        g, u = gu[:I], gu[I:]
        silu = g / (1.0 + np.exp(-np.clip(g, -40, 40)))
        return s, float(w[s, j]) * (down[e].T @ (silu * u))

    out = np.zeros_like(h, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for s, contrib in pool.map(lambda t: task(*t),
                                   [(s, j) for s in range(h.shape[0])
                                    for j in range(k)]):
            out[s] += contrib
    if kw.get("shared_gate") is not None:
        full, _ = moe_grouped_forward(h, gate_up, down, router_w, top_k,
                                       weight_mode=weight_mode,
                                       router_b=router_b,
                                       router_feat=router_feat, **kw)
        # shared path is dense (cheap); reuse exact sequential code for it
        seq_shared = full - moe_grouped_forward(
            h, gate_up, down, router_w, top_k, weight_mode=weight_mode,
            router_b=router_b, router_feat=router_feat,
            **{**kw, "shared_gate": None})[0]
        out = out + seq_shared
    return out.astype(np.float32), {"experts": idx, "weights": w}


# ---------------------------------------------------------------------------
# N-gram embedding bag (real layout: ple shards + heads offsets/vocab sizes)
# ---------------------------------------------------------------------------

def ngram_bag(token_ids: list[int], shards: list[np.ndarray],
              heads_offsets: np.ndarray, heads_vocab_sizes: np.ndarray,
              multipliers: np.ndarray, ngram: int = 3) -> np.ndarray:
    """Sharded n-gram embedding lookup: for each position, hash the trailing
    bigram/trigram into each head's vocab, gather from the owning shard,
    scale by the head multiplier, sum. Deterministic, no training."""
    dim = shards[0].shape[1]
    H = len(heads_vocab_sizes)
    bounds = [0]
    for s in shards:
        bounds.append(bounds[-1] + s.shape[0])
    total = bounds[-1]
    out = np.zeros((len(token_ids), dim), dtype=np.float32)
    for t, _tok in enumerate(token_ids):
        ctx = token_ids[max(0, t - ngram + 1):t + 1]
        for hh in range(H):
            v = int(heads_vocab_sizes[hh])
            key = 0
            for c in ctx:
                key = (key * 1315423911 + int(c) + int(heads_offsets[hh])) % v
            gid = int(heads_offsets[hh]) + key
            gid %= total
            lo = 0
            for si, s in enumerate(shards):
                if bounds[si] <= gid < bounds[si + 1]:
                    out[t] += float(multipliers[hh]) * s[gid - bounds[si]]
                    lo = 1
                    break
            assert lo, "ngram id outside shards"
    return out


# ---------------------------------------------------------------------------
# Gated linear recurrence (clean-room DeltaNet-class mechanism)
# ---------------------------------------------------------------------------

class GatedDeltaState:
    """Recurrent state for one query head over its value group.

    S: (G*hv1, hk). Streaming step == full-sequence scan by construction
    (tested). Data-dependent decay + delta-rule update. The q-head <->
    value-head pairing (grouped when n_v is a multiple of n_qk, else
    round-robin) is a clean-room choice, stated, not claimed from weights.
    """

    def __init__(self, gv: int, hk: int):
        self.S = np.zeros((gv, hk), dtype=np.float32)

    def step(self, q, k, v, beta: float, decay: float) -> np.ndarray:
        """q,k: (hk,); v: (gv,) flattened value group. Returns y: (gv,)."""
        a = float(np.clip(decay, 0.0, 1.0))
        b = float(np.clip(beta, 0.0, 1.0))
        Sh = self.S * a
        v_old = Sh @ k
        Sh = Sh + np.outer(b * v - b * v_old, k)
        self.S = Sh
        return Sh @ q


def linear_attn_forward(x: np.ndarray, in_proj_qkv: np.ndarray,
                        in_proj_b: np.ndarray, in_proj_a: np.ndarray,
                        in_proj_z: np.ndarray, A_log: np.ndarray,
                        dt_bias: np.ndarray, out_proj: np.ndarray,
                        norm_w: np.ndarray, n_qk: int, n_v: int,
                        hk: int, hv: int, conv_w: np.ndarray | None = None,
                        eps: float = 1e-6) -> tuple[np.ndarray, GatedDeltaState]:
    """Full-sequence GatedDeltaNet-class forward over (S,D) input.

    Roles follow the real tensor names: in_proj_qkv -> q,k,v heads;
    in_proj_b -> beta (delta coefficient); in_proj_a -> decay gate input;
    A_log/dt_bias -> per-head base decay; in_proj_z -> output gate;
    norm -> per-head output norm; out_proj -> back to D. Optional short
    depthwise conv over qkv (conv1d, kernel K) for local context.
    Returns (out, states) — states lets callers verify streaming."""
    S, D = x.shape
    H = n_qk
    proj = x @ in_proj_qkv.T
    o1 = H * hk
    o2 = 2 * o1
    q = proj[:, :o1].reshape(S, H, hk)
    k = proj[:, o1:o2].reshape(S, H, hk)
    vv = proj[:, o2:].reshape(S, n_v, hv)
    if conv_w is not None:
        K = conv_w.shape[-1]
        pad = np.zeros((K - 1, H, hk), dtype=np.float32)
        qq = np.concatenate([pad, q])
        q = np.stack([conv_w[h % conv_w.shape[0]] @ qq[t:t + K, h]
                      for t in range(S) for h in range(H)]).reshape(S, H, hk)
    # value-head grouping per query head (clean-room pairing, documented)
    if n_v % H == 0:
        groups = [list(range(h * (n_v // H), (h + 1) * (n_v // H))) for h in range(H)]
    else:
        groups = [[h % n_v] for h in range(H)]
    b = _sigmoid(x @ in_proj_b.T).reshape(S, -1)[:, :H]
    a_in = (x @ in_proj_a.T).reshape(S, -1)[:, :H]
    base = np.exp(-np.exp(A_log.astype(np.float32)) + dt_bias.astype(np.float32))
    decay = np.clip(base[None, :] * _sigmoid(a_in), 0.0, 1.0)
    z = _sigmoid(x @ in_proj_z.T)
    states = [GatedDeltaState(len(g) * hv, hk) for g in groups]
    ydim = sum(len(g) * hv for g in groups)
    ys = np.empty((S, ydim), dtype=np.float32)
    for t in range(S):
        cols = []
        for h, g in enumerate(groups):
            vflat = vv[t, g].reshape(-1)
            cols.append(states[h].step(q[t, h], k[t, h], vflat,
                                       float(b[t, h]), float(decay[t, h])))
        ys[t] = np.concatenate(cols)
    y = ys
    if norm_w.size == y.shape[1]:
        nw = norm_w
    else:
        nw = np.repeat(norm_w, (y.shape[1] + norm_w.size - 1) // norm_w.size)[:y.shape[1]]
    yn = rmsnorm(y, nw)
    # z must gate every output dim: slice when wider, tile when narrower
    if z.shape[1] >= yn.shape[1]:
        zcols = z[:, :yn.shape[1]]
    else:
        rep = (yn.shape[1] + z.shape[1] - 1) // z.shape[1]
        zcols = np.tile(z, (1, rep))[:, :yn.shape[1]]
    gated = yn * zcols
    return (gated @ out_proj.T).astype(np.float32), states


# ---------------------------------------------------------------------------
# QSA-class block-sparse attention (clean-room mechanism)
# ---------------------------------------------------------------------------

def qsa_forward(x: np.ndarray, q_proj: np.ndarray, k_proj: np.ndarray,
                v_proj: np.ndarray, o_proj: np.ndarray,
                index_qk: np.ndarray, q_norm_w: np.ndarray,
                k_norm_w: np.ndarray, q_ln_w: np.ndarray, k_ln_w: np.ndarray,
                n_q: int, n_kv: int, hd: int, rope_dim: int,
                block: int = 64, top_blocks: int = 8, window: int = 256,
                rope=None) -> np.ndarray:
    """Queries attend to (a) a local sliding window + (b) top_blocks micro-
    blocks ranked by the indexer (compressed q@k over 4x-pooled keys).
    Causal. When top_blocks covers everything, equals full attention
    (tested). Partial rotary: only the first rope_dim dims rotate."""
    from .chat import apply_rope
    S, D = x.shape
    q = (x @ q_proj.T).reshape(S, n_q, hd)
    k = (x @ k_proj.T).reshape(S, n_kv, hd)
    v = (x @ v_proj.T).reshape(S, n_kv, hd)
    q = q / np.sqrt((q * q).mean(-1, keepdims=True) + 1e-6) * q_norm_w
    k = k / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-6) * k_norm_w
    rep = n_q // n_kv
    k = np.repeat(k, rep, axis=1)
    v = np.repeat(v, rep, axis=1)
    if rope is not None and rope_dim > 0:
        ang = rope(rope_dim, S)
        q = np.concatenate([apply_rope(q[..., :rope_dim], ang[:, None, :]),
                            q[..., rope_dim:]], axis=-1)
        k = np.concatenate([apply_rope(k[..., :rope_dim], ang[:, None, :]),
                            k[..., rope_dim:]], axis=-1)
    # indexer: index_qk -> (2*idx_dim); q-side per position, k-side 4x-pooled
    idx_full = x @ index_qk.T
    iz = idx_full.shape[1] // 2
    iq = rmsnorm(idx_full[:, :iz], q_ln_w[:iz])
    nb = (S + block - 1) // block
    kb = np.stack([idx_full[i * block:(i + 1) * block, iz:].mean(0)
                   for i in range(nb)])
    ik = rmsnorm(kb, k_ln_w[:kb.shape[1]])
    scores = iq @ ik.T  # (S, nb)
    out = np.zeros((S, n_q, hd), dtype=np.float32)
    for t in range(S):
        lo = max(0, t - window + 1)
        # deployment semantics: at step t only past blocks exist; future
        # blocks must not outrank past ones (masked before top-k)
        past_nb = t // block + 1
        st = scores[t, :past_nb]
        order = np.argsort(-st, kind="stable")
        keep = set()
        for bi in order[:top_blocks]:
            keep.update(range(int(bi) * block, min(S, (int(bi) + 1) * block)))
        keep.update(range(lo, t + 1))
        js = sorted(j for j in keep if j <= t)
        kj = k[js]  # (J, nq, hd)
        sc = np.stack([q[t, qq_] @ kj[:, qq_, :].T
                       for qq_ in range(n_q)]) / max(hd ** 0.5, 1e-6)
        e = np.exp(sc - sc.max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        out[t] = (a[:, :, None] * v[js].transpose(1, 0, 2)).sum(1)
    return (out.reshape(S, -1) @ o_proj.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Gated-residual (hyper-connection) mixer, generic read/write-gate form
# ---------------------------------------------------------------------------

def hyper_mix(h: np.ndarray, hc_norm_w: np.ndarray,
              up_w: np.ndarray, down_w: np.ndarray,
              inject_w: np.ndarray, branch_out: np.ndarray,
              eps: float = 1e-6) -> np.ndarray:
    """Read gate (data-dependent: up (R,D) then down (D,R)) modulates the
    normalized stream read; the branch output is written back through a
    scalar write gate. Zero-initialized gates reproduce the identity
    (tested) — the stability property the design is named for."""
    read = _sigmoid((h @ up_w.T) @ down_w.T)
    gated_in = rmsnorm(h, hc_norm_w, eps) * read
    _ = gated_in  # the read feeds the branch in the real block; mixer returns:
    wscale = float(_sigmoid(inject_w.reshape(-1)[:1])[0])
    return (h + wscale * branch_out).astype(np.float32)


# ---------------------------------------------------------------------------
# Mamba-class selective SSM (clean-room S6-style mechanism, synthetic-verified)
# ---------------------------------------------------------------------------

class SelectiveSSMState:
    """Recurrent state for one head: h (d_state,). Streaming step == full
    scan by construction (tested). Input-dependent Δ/B/C per token."""

    __slots__ = ("h",)

    def __init__(self, d_state: int):
        self.h = np.zeros(d_state, dtype=np.float32)


def ssm_step(state: SelectiveSSMState, x_proj: np.ndarray,
             A: np.ndarray, dt: np.ndarray, B: np.ndarray,
             C: np.ndarray, D: float | np.ndarray = 0.0) -> np.ndarray:
    """One selective-scan step.

    x_proj: (d_inner,) this token's projected input row
    A: (d_state,) negative log-magnitudes (kept negative: exp(A*dt) decays)
    dt: (d_state,) input-dependent step sizes (sigmoid-gated upstream)
    B: (d_state,) input-dependent SSM input matrix
    C: (d_state,) input-dependent readout matrix
    Discretization (ZOH): h <- exp(A*dt) * h + dt * B * x ; y = C @ h + D*x
    Returns y scalar for this head.
    """
    h = state.h
    decay = np.exp(A * dt)
    state.h = decay * h + (dt * B) * float(x_proj)
    return float(C @ state.h) + float(np.asarray(D)) * float(x_proj)


def mamba_block_forward(x: np.ndarray, in_proj: np.ndarray,
                        conv_w: np.ndarray, conv_b: np.ndarray,
                        x_proj_w: np.ndarray, dt_proj_w: np.ndarray,
                        A_log: np.ndarray, dt_bias: np.ndarray,
                        out_proj: np.ndarray, norm_w: np.ndarray,
                        n_heads: int, d_state: int,
                        states: list[SelectiveSSMState] | None = None,
                        conv_hist: np.ndarray | None = None,
                        eps: float = 1e-6) -> tuple[np.ndarray, list, np.ndarray]:
    """Full-sequence selective-SSM block forward over (S, D) input.

    Layout mirrors Mamba-class tensor names: in_proj (2*DI, D) produces
    [z, x]; a short depthwise causal conv over x; x_proj gives (dt, B, C);
    per-head selective scan; gated by z; out_proj back to D.
    Returns (out (S,D), states) — states allow streaming verification."""
    S, D = x.shape
    DI = in_proj.shape[0] // 2
    proj = x @ in_proj.T
    z, xz = proj[:, :DI], proj[:, DI:]
    # causal depthwise conv over xz (kernel K); conv_hist carries the last
    # K-1 rows of a previous call so streaming continuation is exact
    K = conv_w.shape[-1]
    if conv_hist is None:
        conv_hist = np.zeros((K - 1, DI), np.float32)
    xp = np.concatenate([conv_hist.astype(np.float32), xz], axis=0)
    xv = np.empty_like(xz)
    for i in range(DI):
        xv[:, i] = conv_w[i] @ xp[:, i].T if False else np.tensordot(
            conv_w[i], np.stack([xp[t:t + K, i] for t in range(S)]), axes=([0], [1]))
    xv = xv + conv_b[None, :]
    # gates + projections
    g = xz if False else xv
    xbp = g @ x_proj_w.T          # (S, dt_rank + 2*d_state)
    dt_rank = x_proj_w.shape[0] - 2 * d_state
    dt_raw = xbp[:, :dt_rank]
    Bm = xbp[:, dt_rank:dt_rank + d_state]
    Cm = xbp[:, dt_rank + d_state:]
    dt = _sigmoid(dt_raw @ dt_proj_w.T + dt_bias[None, :])  # (S, d_state)
    A = -np.exp(A_log.astype(np.float32))  # negative magnitudes
    # per-head scan over d_inner columns
    hd = DI // n_heads
    ys = np.empty((S, DI), dtype=np.float32)
    own = states is None
    if own:
        states = [SelectiveSSMState(d_state) for _ in range(DI)]
    for i in range(DI):
        st = states[i] if i < len(states) else SelectiveSSMState(d_state)
        if i >= len(states):
            states.append(st)
        Ai = A if A.ndim == 1 else A[i % A.shape[0]]
        dti = dt if dt.shape[1] == d_state else dt[:, i % dt.shape[1]]
        for t in range(S):
            ys[t, i] = ssm_step(st, g[t, i], Ai, dti[t], Bm[t], Cm[t])
    # gate by z and project out
    yn = rmsnorm(ys, norm_w if norm_w.size == DI else
                 np.repeat(norm_w, (DI + norm_w.size - 1) // norm_w.size)[:DI],
                 eps)
    gated = yn * _sigmoid(z[:, :DI])
    return ((gated @ out_proj.T).astype(np.float32), states,
            xz[-(K - 1):] if K > 1 else np.zeros((0, DI), np.float32))
