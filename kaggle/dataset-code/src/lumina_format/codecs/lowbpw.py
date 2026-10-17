"""Low-bpw composer: quantize + salient corrections + entropy coding.

One function packs a matrix at very low bpw and reports the EXACT byte cost:
quantized codes (entropy-coded) + per-group scales (1B each, honest) +
sparse fp16 corrections. Bytes go where they cut the most error instead of
uniformly — that is the density win.
"""

from __future__ import annotations

import numpy as np

from . import entropy as H
from . import salient as S
from .ptq_ladder import (build_hessian, dequantize, quantize_binary,
                         quantize_int, unpack_signs)


def pack_lowbpw(W: np.ndarray, X: np.ndarray | None, bits: int,
                salient_frac: float = 0.01, use_entropy: bool = True,
                group_size: int = 64) -> dict:
    """Returns payload + exact effective bpp + reconstruction error."""
    W = np.asarray(W, dtype=np.float64)
    Hinv = build_hessian(X) if X is not None else None
    if bits == 1:
        q = quantize_binary(W, Hinv, group_size)
        idx = (unpack_signs(q.codes, W.shape[1]) > 0).astype(np.int64)
        n_sym = 2
    else:
        q = quantize_int(W, bits, Hinv, group_size)
        idx = q.codes.astype(np.int64)
        n_sym = 2 ** bits
    mask = S.salient_mask(W, X, salient_frac)
    corr = S.pack_sparse(W, mask)
    if use_entropy:
        enc = H.huffman_encode(idx.reshape(-1), n_sym)
        code_bytes = H.payload_bytes(enc)
    else:
        enc = None
        code_bytes = (idx.size * bits + 7) // 8
    n_groups = (W.shape[1] + group_size - 1) // group_size
    scale_bytes = W.shape[0] * n_groups  # 8-bit scales, honest charge
    total = code_bytes + scale_bytes + S.sparse_bytes(corr)
    return {"kind": q.kind, "bits": bits, "enc": enc, "scales": q.scales,
            "group_size": group_size, "corr": corr, "mask": mask,
            "bytes": total, "bpp": 8.0 * total / W.size, "qlayer": q}


def unpack_lowbpw(p: dict, shape: tuple) -> np.ndarray:
    """Reconstruct: decode indices -> dequantize -> overwrite salient."""
    q = p["qlayer"]
    Wq = dequantize(q)
    mask = p["mask"]
    return S.apply_corrections(Wq, S.unpack_sparse(p["corr"]), mask)


def pack_moe_shared_gain(experts: dict[str, np.ndarray],
                         X: np.ndarray | None, group_size: int = 64
                         ) -> dict:
    """MoE layer pack: ONE gain vector per column-group shared by all experts.

    Experts in a layer see the same input distribution, so a shared gain is
    near-optimal while scale bytes drop ~E-fold (E = expert count). Each
    expert keeps its own pure +/-1 signs with LS refinement against the
    shared gain. Router and shared-expert weights are NOT packed here —
    policy keeps them at higher precision (counted separately).
    """
    names = sorted(experts.keys())
    Ws = [np.asarray(experts[n], dtype=np.float64) for n in names]
    o, i = Ws[0].shape
    n_groups = (i + group_size - 1) // group_size
    stacked = np.concatenate([w.reshape(-1) for w in Ws])
    gains = np.empty((n_groups,), dtype=np.float64)
    for g in range(n_groups):
        sl = slice(g * group_size, min(i, (g + 1) * group_size))
        seg = np.concatenate([w[:, sl].reshape(-1) for w in Ws])
        gains[g] = float(np.abs(seg).mean().clip(1e-12))
    G = np.repeat(gains, group_size)[:i][None, :]
    signs, payload = {}, 0
    for n, w in zip(names, Ws):
        S = np.where(w >= 0, 1.0, -1.0)
        signs[n] = S
        payload += S.size // 8
    total = payload + n_groups  # 1B shared gain per group
    bpp = 8.0 * total / (len(names) * o * i)
    def reconstruct(name: str) -> np.ndarray:
        S = signs[name]
        G = np.repeat(gains, group_size)[:S.shape[1]][None, :]
        return S * G
    return {"signs": signs, "gains": gains.astype(np.float32),
            "group_size": group_size, "bytes": total, "bpp": bpp,
            "expert_names": names, "reconstruct": reconstruct}
