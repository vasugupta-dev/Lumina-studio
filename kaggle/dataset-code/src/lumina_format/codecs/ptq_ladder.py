"""Lumina Format PTQ ladder — 4-bit down to sub-1-bit, post-training only.

Measured the way the project owner defines it: convert a model into Lumina Format
and compare the WHOLE MODEL's behaviour against the FP16/BF16 base, not a
per-matrix reconstruction proxy.

Technique: GPTQ error feedback. Independent rounding ignores that a weight's
error changes what the function computes. GPTQ propagates each error into the
not-yet-quantized weights through the inverse Hessian of the calibration
activations:

    W[:, j+1:] -= ((w_j - q_j) / Hinv[j,j]) * Hinv[j, j+1:]

The Hessian comes from real calibration forward passes. Without calibration data
the codec falls back to H = I and records that fact, because the assumption
changes measured quality.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
try:
    from scipy import optimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# bit packing
# ---------------------------------------------------------------------------


def pack_signs(S: np.ndarray) -> np.ndarray:
    """Pack {-1,+1} into 1 bit per weight. Consecutive columns share a byte."""
    S = np.asarray(S)
    pos = (S > 0).astype(np.uint8)
    n, m = pos.shape
    pad = (-m) % 8
    if pad:
        pos = np.concatenate([pos, np.zeros((n, pad), np.uint8)], axis=1)
    Bg = pos.reshape(n, -1, 8)
    w = (np.uint8(1) << np.arange(8, dtype=np.uint8)).astype(np.uint8)
    return (Bg * w[None, None, :]).sum(axis=2).astype(np.uint8)


def unpack_signs(packed: np.ndarray, m: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    n = packed.shape[0]
    bits = (packed[:, :, None] >> np.arange(8, dtype=np.uint8)[None, None, :]) & np.uint8(1)
    signs = np.where(bits == 1, 1, -1).astype(np.float32)
    return signs.reshape(n, -1)[:, :m]


def _levels(nbits: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, 2 ** nbits)


# ---------------------------------------------------------------------------
# Hessian
# ---------------------------------------------------------------------------


def build_hessian(X: np.ndarray, damp: float = 0.01) -> np.ndarray:
    """Damped inverse Hessian from calibration inputs X of shape (in, n_samples).

    WARNING: if n_samples < in_features the Hessian is rank-deficient. The
    reference raises no error but callers MUST pass enough samples; the pipeline
    records how many were used so a rank-deficient calibration is visible.
    """
    X = np.asarray(X, dtype=np.float64)
    i = X.shape[0]
    if X.shape[1] == 0:
        return np.eye(i)
    n = X.shape[1]
    return build_hessian_from_gram((X @ X.T), n)


def build_hessian_from_gram(G: np.ndarray, n: int,
                            damp: float = 0.01) -> np.ndarray:
    """Exact damped inverse Hessian from an accumulated Gram G = X @ X.T.

    Streaming twin of build_hessian: hooks accumulate float32 Grams per layer
    (never retaining activations), then each Gram inverts here. Identical
    math, fraction of the memory.
    """
    G = np.asarray(G, dtype=np.float64)
    i = G.shape[0]
    n = max(int(n), 1)
    H = G * (2.0 / n) + 1e-8 * np.eye(i)
    d = np.diag(H).copy()
    mean = float(d[d > 0].mean()) if (d > 0).any() else 1.0
    H = H + damp * mean * np.eye(i)
    H = 0.5 * (H + H.T)
    try:
        L = np.linalg.cholesky(H)
        Linv = np.linalg.solve(L, np.eye(i))
        return Linv.T @ Linv
    except np.linalg.LinAlgError:
        return np.diag(1.0 / np.maximum(np.diag(H), 1e-12))



# ---------------------------------------------------------------------------
# Activation-aware scaling (AWQ-lite)
# ---------------------------------------------------------------------------


def awq_scale(W: np.ndarray, X: np.ndarray, nbits: int = 4,
              alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-input-channel scaling to reduce quantization difficulty.

    Returns (W_scaled, X_scaled, s) such that W_scaled = W * diag(s),
    X_scaled = X / diag(s) (conceptually) and the scaled weights are easier
    to quantize. This is a lightweight approximation of AWQ: scales are chosen
    proportional to (a_i / w_i)^alpha where a_i, w_i are activation/weight mags.
    """
    W = np.asarray(W, dtype=np.float64)
    if X is None or X.shape[1] == 0:
        s = np.ones(W.shape[1])
        return W, X, s
    X = np.asarray(X, dtype=np.float64)
    # activation magnitude per input channel
    a = np.abs(X).mean(axis=1)  # (in,)
    w = np.abs(W).mean(axis=0)  # (in,)
    a = np.maximum(a, 1e-12)
    w = np.maximum(w, 1e-12)
    s = (a / w) ** alpha
    s = np.clip(s, 0.01, 100.0)
    # apply scaling: W_scaled_{:,i} *= s_i
    W_scaled = W * s[None, :]
    return W_scaled, X, s
# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


class QLayer:
    """Quantized matrix with exact, auditable byte accounting."""

    __slots__ = ("kind", "out", "inn", "codes", "scales", "extra", "code_bits",
                 "group_size", "scale_bits")

    def __init__(self, kind, out, inn, codes, scales, extra=None,
                 code_bits=None, group_size=64, scale_bits=8):
        self.kind = kind
        self.out, self.inn = out, inn
        self.codes = codes
        self.scales = scales
        self.extra = extra or {}
        self.code_bits = code_bits
        self.group_size = group_size
        self.scale_bits = scale_bits

    def n_groups(self) -> int:
        return max(1, (self.inn + self.group_size - 1) // self.group_size)

    def payload_bytes(self) -> int:
        if self.kind == "sub1bit":
            return int(self.extra.get("bytes", 0))
        bits = self.out * self.inn * (self.code_bits or 1)
        bits += self.out * self.n_groups() * self.scale_bits
        bits += int(self.extra.get("escape_bits", 0))
        return int(math.ceil(bits / 8))

    def effective_bpp(self) -> float:
        return 8.0 * self.payload_bytes() / float(self.out * self.inn)

    def summary(self) -> dict:
        return {"kind": self.kind, "shape": (self.out, self.inn),
                "payload_bytes": self.payload_bytes(),
                "effective_bpp": self.effective_bpp()}


def dequantize(q: QLayer) -> np.ndarray:
    """Full-precision reconstruction (used by tests and the reference forward)."""
    o, i = q.out, q.inn
    if q.kind == "binary":
        S = unpack_signs(q.codes, i)
        s = np.repeat(q.scales, q.group_size, axis=1)[:, :i]
        W = S * s
    else:
        if q.extra.get("custom_grid") and "learned_grid" in q.extra:
            levels = q.extra["learned_grid"]
        else:
            levels = _levels(q.code_bits)
        
        # Handle per-channel vs per-group scales
        if q.scales.ndim == 2 and q.scales.shape[1] == i:
            # Per-channel scales (output_channels, input_channels)
            s = q.scales
        elif q.scales.ndim == 2:
            # Per-group scales (output_channels, n_groups) - expand
            s = np.repeat(q.scales, q.group_size, axis=1)[:, :i]
        else:
            s = q.scales
        
        W = s * levels[q.codes]
        if q.extra.get("escape_idx") is not None:
            W[q.extra["escape_idx"][0]] = q.extra["escape_vals"]
    
    if q.extra.get("awq_applied") and "awq_s" in q.extra:
        awq_s = np.asarray(q.extra["awq_s"], dtype=W.dtype).reshape(1, -1)[:, :i]
        W = W / np.maximum(awq_s, 1e-12)
    return W


# ---------------------------------------------------------------------------
# GPTQ core
# ---------------------------------------------------------------------------


def group_scales(W: np.ndarray, group_size: int, nbits: int) -> np.ndarray:
    """Per-row, per-group scales chosen by MSE, not by max.

    Scaling the grid to the group's absolute max wastes most of the dynamic range
    on a single outlier: for Gaussian weights the max is ~3.2 sigma, so the grid
    spacing becomes 3.2x coarser than it needs to be. Searching a small set of
    multipliers and keeping the one that minimizes squared error is the standard
    MSE-optimal fit and is a real quality lever at every rung.
    """
    o, i = W.shape
    n_groups = (i + group_size - 1) // group_size
    lv = _levels(nbits)
    out = np.zeros((o, n_groups))
    # candidate multipliers of the group absmax
    for g in range(n_groups):
        sl = slice(g * group_size, min(i, (g + 1) * group_size))
        Wg = W[:, sl]                                   # (o, gs)
        amax = np.abs(Wg).max(axis=1).clip(1e-12)
        best_s = amax.copy()
        best_e = None
        for f in (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2):
            s = np.clip(amax * f, 1e-12, None)
            wn = Wg / s[:, None]
            d = np.abs(wn[:, :, None] - lv[None, None, :])
            idx = np.argmin(d, axis=2)
            err = (wn - lv[idx]) ** 2
            e = err.mean(axis=1)
            if best_e is None:
                best_e, best_s = e, s
            else:
                take = e < best_e
                best_e = np.where(take, e, best_e)
                best_s = np.where(take, s, best_s)
        out[:, g] = best_s
    return out


def gptq(W: np.ndarray, Hinv, nbits: int, group_size: int = 64,
         act_order: bool = True, per_channel_scale: bool = False,
         levels: np.ndarray | None = None,
         dtype=np.float64, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise quantization with GPTQ error feedback — block fast path.

    act_order=True quantizes columns in descending Hessian-diagonal order.
    per_channel_scale=True uses per-output-channel scales (more granular).

    Exact-math fast path (the official GPTQ scheme): columns are permuted once,
    feedback within the current block is applied sequentially, and feedback to
    all later columns is deferred to one BLAS matmul per block:
        W[:, jend:] -= Err_block @ Hinv_block_to_rest
    This is algebraically identical to the per-column outer-product update
    (same terms, same order per target column; only the summation grouping
    differs), but replaces O(o*i) interpreted updates per column with one
    contiguous BLAS call per block. Verified against the reference loop by
    tests/test_format_framework.py (codes and relW must match to 1e-6).
    """
    W = np.asarray(W, dtype=dtype).copy()
    o, i = W.shape

    if per_channel_scale:
        scales = np.abs(W).max(axis=1).clip(1e-12)  # (o,)
    else:
        scales = group_scales(W, group_size, nbits)  # (o, n_groups)

    if Hinv is None:
        Hinv = np.eye(i)
    else:
        Hinv = np.asarray(Hinv, dtype=np.float64)

    if levels is None:
        levels = _levels(nbits)
    levels = np.asarray(levels, dtype=dtype)
    # int32, not int64: level indices never exceed 2^bits-1; halves the
    # transient allocation on big matrices (embed-scale codes can exceed 1GB
    # as int64). dequantize indexes levels[codes] identically.
    codes = np.zeros((o, i), dtype=np.int32)

    diag_all = np.maximum(np.diag(Hinv).copy(), 1e-12)
    if act_order:
        perm = np.argsort(-diag_all)
    else:
        perm = np.arange(i)

    # Permute once: all later work runs on contiguous slices.
    W = W[:, perm]
    Hinv = Hinv[np.ix_(perm, perm)]
    diag = diag_all[perm]
    gcol = (perm // group_size) if not per_channel_scale else None

    Err = np.empty((o, block), dtype=W.dtype)
    for j0 in range(0, i, block):
        jend = min(j0 + block, i)
        w_block = jend - j0
        for jj in range(j0, jend):
            if per_channel_scale:
                s = scales
            else:
                s = scales[:, gcol[jj]]
            wn = W[:, jj] / s
            idx = np.abs(wn[:, None] - levels[None, :]).argmin(axis=1)
            codes[:, jj] = idx
            q = s * levels[idx]
            err = (W[:, jj] - q) / diag[jj]
            Err[:, jj - j0] = err
            # intra-block feedback stays sequential (exact GPTQ order)
            if jj + 1 < jend:
                W[:, jj + 1:jend] -= np.outer(err, Hinv[jj, jj + 1:jend])
        # cross-block feedback: one BLAS matmul instead of per-column outers
        if jend < i:
            W[:, jend:] -= Err[:, :w_block] @ Hinv[j0:jend, jend:]

    # codes were computed in permuted order; map back to original columns
    codes_orig = np.zeros_like(codes)
    codes_orig[:, perm] = codes
    return codes_orig, scales



# ---------------------------------------------------------------------------
# Advanced quantization: AWQ + GPTQ combined, learned grids, mixed precision
# ---------------------------------------------------------------------------


def awq_scale_per_channel(W: np.ndarray, X: np.ndarray, nbits: int = 4,
                          alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel AWQ scaling: scale each input channel independently.
    
    Returns (W_scaled, X_scaled, s_per_channel) where W_scaled = W * diag(s).
    This is the full AWQ approach vs the per-group approximation.
    """
    W = np.asarray(W, dtype=np.float64)
    if X is None or X.shape[1] == 0:
        s = np.ones(W.shape[1])
        return W, X, s
    X = np.asarray(X, dtype=np.float64)
    # Activation magnitude per input channel
    a = np.abs(X).mean(axis=1)  # (in,)
    w = np.abs(W).mean(axis=0)  # (in,)
    a = np.maximum(a, 1e-12)
    w = np.maximum(w, 1e-12)
    s = (a / w) ** alpha
    s = np.clip(s, 0.01, 100.0)
    W_scaled = W * s[None, :]
    # Note: X would need inverse scaling at inference, but for PTQ we just scale W
    return W_scaled, X, s


def learn_optimal_grid(W: np.ndarray, nbits: int, num_iters: int = 10) -> np.ndarray:
    """Learn optimal non-uniform quantization grid via Lloyd-Max / k-means.
    
    For a given weight distribution, finds the grid points that minimize MSE.
    This is a significant quality lever at 1-3 bits.
    """
    W_flat = W.flatten()
    n_levels = 2 ** nbits
    
    # Initialize with uniform grid
    grid = np.linspace(W_flat.min(), W_flat.max(), n_levels)
    
    for _ in range(num_iters):
        # Assign each weight to nearest grid point
        dist = np.abs(W_flat[:, None] - grid[None, :])
        assignments = np.argmin(dist, axis=1)
        
        # Update grid points to centroids of their clusters
        new_grid = np.zeros_like(grid)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                new_grid[i] = W_flat[mask].mean()
            else:
                new_grid[i] = grid[i]
        
        if np.allclose(grid, new_grid):
            break
        grid = new_grid
    
    return grid.astype(np.float32)


def mixed_precision_allocation(W_layers: dict[str, np.ndarray], 
                               X_layers: dict[str, np.ndarray],
                               target_bpp: float,
                               min_bits: int = 1, max_bits: int = 4) -> dict[str, int]:
    """Allocate bits per layer based on sensitivity (Hessian trace).
    
    Returns layer_name -> bits allocation that minimizes total error at target bpp.
    """
    sensitivities = {}
    for name, W in W_layers.items():
        X = X_layers.get(name)
        if X is not None:
            # Sensitivity = trace(Hessian) * ||W||^2
            H = (X @ X.T) * (2.0 / X.shape[1])
            sens = np.trace(H) * np.sum(W ** 2)
        else:
            sens = np.sum(W ** 2)
        sensitivities[name] = sens
    
    # Sort by sensitivity (descending)
    sorted_layers = sorted(sensitivities.items(), key=lambda x: -x[1])
    
    # Greedy allocation: start all at max_bits, reduce least sensitive first
    total_params = sum(W.size for W in W_layers.values())
    target_bits = target_bpp * total_params / 8.0  # total bits budget
    
    allocation = {name: max_bits for name in W_layers}
    current_bits = sum(W.size * max_bits for W in W_layers.values()) / 8.0
    
    for name, _ in sorted_layers:
        if current_bits <= target_bits:
            break
        for bits in range(max_bits - 1, min_bits - 1, -1):
            if current_bits <= target_bits:
                break
            delta = W_layers[name].size * (allocation[name] - bits) / 8.0
            if current_bits - delta >= target_bits:
                current_bits -= delta
                allocation[name] = bits
    
    return allocation


def quantize_int(W: np.ndarray, nbits: int, Hinv=None,
                 group_size: int = 64, X: np.ndarray = None,
                 use_awq: bool = False, awq_alpha: float = 0.5,
                 use_learned_grid: bool = False, per_channel: bool = False,
                 dtype=np.float64) -> QLayer:
    W_in = W
    s_awq = None
    grid = _levels(nbits)
    if use_awq and X is not None:
        W_s, X_s, s_awq = awq_scale(W_in, X, nbits=nbits, alpha=awq_alpha)
    else:
        W_s = W_in
    if use_learned_grid:
        # learn in globally-normalized space so the grid matches the
        # per-group scales gptq fits (both live in [-1, 1])
        amax = float(np.abs(W_s).max().clip(1e-12))
        grid = learn_optimal_grid(W_s / amax, nbits)
    o, i = W_s.shape
    codes, scales = gptq(W_s, Hinv, nbits, group_size=group_size,
                         per_channel_scale=per_channel, levels=grid,
                         dtype=dtype)
    ql = QLayer(f"int{nbits}", o, i, codes, scales, code_bits=nbits,
                group_size=group_size)
    if s_awq is not None:
        ql.extra["awq_s"] = s_awq.astype(np.float32)
        ql.extra["awq_applied"] = True
    if use_learned_grid:
        ql.extra["learned_grid"] = grid.astype(np.float32)
        ql.extra["custom_grid"] = True
    return ql


def quantize_binary(W: np.ndarray, Hinv=None, group_size: int = 64,
                       X: np.ndarray = None, use_awq: bool = False,
                       awq_alpha: float = 0.5, dtype=np.float64) -> QLayer:
    """1-bit sign weights, per-group scale, GPTQ error feedback, then a least
    squares refinement of every group scale against the ORIGINAL matrix."""
    W_in = W
    s_awq = None
    if use_awq and X is not None:
        W_s, X_s, s_awq = awq_scale(W_in, X, nbits=1, alpha=awq_alpha)
    else:
        W_s = W_in
    o, i = W_s.shape
    n_groups = (i + group_size - 1) // group_size

    scales0 = group_scales(W_s, group_size, 1)
    Hinv = np.eye(i) if Hinv is None else np.asarray(Hinv, dtype=dtype)
    diag = np.maximum(np.diag(Hinv).copy(), 1e-12)

    Wc = W_s.astype(dtype).copy()
    sign = np.zeros_like(Wc)
    # Block fast path: intra-block feedback sequential, cross-block one BLAS
    # matmul per block — algebraically identical to the per-column loop.
    BLOCK = 32
    Err = np.empty((o, BLOCK), dtype=Wc.dtype)
    for j0 in range(0, i, BLOCK):
        jend = min(j0 + BLOCK, i)
        wb = jend - j0
        for jj in range(j0, jend):
            g = jj // group_size
            s = scales0[:, g]
            sg = np.where(Wc[:, jj] >= 0, 1.0, -1.0)
            sign[:, jj] = sg
            q = s * sg
            err = (Wc[:, jj] - q) / diag[jj]
            Err[:, jj - j0] = err
            if jj + 1 < jend:
                Wc[:, jj + 1:jend] -= np.outer(err, Hinv[jj, jj + 1:jend])
        if jend < i:
            Wc[:, jend:] -= Err[:, :wb] @ Hinv[j0:jend, jend:]

    scales = scales0.copy()
    for g in range(n_groups):
        sl = slice(g * group_size, min(i, (g + 1) * group_size))
        num = np.einsum("oj,oj->o", W_s[:, sl], sign[:, sl])
        den = np.einsum("oj,oj->o", sign[:, sl], sign[:, sl]).clip(1e-12)
        scales[:, g] = num / den

    payload = pack_signs(sign)
    ql = QLayer("binary", o, i, payload, scales.astype(np.float32),
                code_bits=1, group_size=group_size)
    if s_awq is not None:
        ql.extra["awq_s"] = s_awq.astype(np.float32)
        ql.extra["awq_applied"] = True
    return ql
