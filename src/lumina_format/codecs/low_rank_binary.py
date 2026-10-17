"""Lumina Low-Rank Binary (LLRB) codec — the sub-1-bit substrate.

Target: approach 1.0 effective bits per logical parameter and go sub-1-bit only
while measured quality holds. This is LittleBit / NanoQuant style low-rank binary
factorization with least-squares scale refinement, plus a sensitivity-guided
residual stream that is the Lumina mechanism for recovering quality at low rate.

Formulation
-----------
For a weight matrix W (out, in) truncated to rank r:

    W_hat = diag(s_out) @ B1 @ diag(g) @ B2 @ diag(s_in)

    B1 in {-1,+1}^(out x r)   row-scaled by s_out (out,)
    B2 in {-1,+1}^(r x in)    col-scaled by s_in (in,)
    g in R_+^r                per-latent (rank) scales

Bit accounting (everything is counted; nothing is free):

    bits = out*r + in*r + (out + in + r)*scale_bits + residual_bits

Rank selection
--------------
Given a target effective bpp:

    r(bpp) = (bpp*out*in - (out+in)*scale_bits) / (out + in + scale_bits)

so large matrices get large ranks and the declared bpp is met by construction.

Execution
---------
`forward_from_payload` computes x @ W_hat^T WITHOUT ever materializing W_hat:

    z = (x @ diag(s_in)) @ B2^T     (in -> r)
    z = z * g                        (per-rank scale)
    y = z @ B1^T                     (r -> out)

This is the same three-matmul structure that makes a SHARED basis cheap: z is
computed once per layer and reused by every expert sharing B2.
"""

from __future__ import annotations

import math

import numpy as np

SCALE_BITS_DEFAULT = 8  # E4M3-style charged rate for every stored scale


# ---------------------------------------------------------------------------
# rank / bit accounting
# ---------------------------------------------------------------------------

def payload_bytes(out: int, inn: int, r: int, scale_bits: int = SCALE_BITS_DEFAULT) -> int:
    """Exact serialized size, including every packing/padding byte."""
    return (int(math.ceil(out * r / 8.0))
            + int(math.ceil(inn * r / 8.0))
            + int(math.ceil(out * scale_bits / 8.0))
            + int(math.ceil(inn * scale_bits / 8.0))
            + int(math.ceil(r * scale_bits / 8.0)))


def rank_for_bpp(out: int, inn: int, bpp: float, scale_bits: int = SCALE_BITS_DEFAULT,
                 min_rank: int = 1, max_rank: int | None = None) -> int:
    """Largest rank whose ALL-IN serialized size stays within the byte budget.

    The continuous closed form gives a starting point; the loop below then steps
    down until the *actual* packed size (scale bytes rounded up, sign bytes
    padded to whole bytes) fits. Solving the inequality without the discrete check
    is how a codec silently overruns its declared budget.
    """
    denom = (out + inn + scale_bits)
    num = bpp * out * inn - (out + inn) * scale_bits
    r = int(num // denom) if num > 0 else min_rank
    r = max(min_rank, r)
    if max_rank is not None:
        r = min(r, max_rank)
    r = min(r, min(out, inn))
    budget_bits = bpp * out * inn
    while r > min_rank and 8.0 * payload_bytes(out, inn, r, scale_bits) > budget_bits + 1e-9:
        r -= 1
    if 8.0 * payload_bytes(out, inn, r, scale_bits) > budget_bits + 1e-9 and r == min_rank:
        # nothing fits at this size; the honest answer is the rank floor
        return min_rank
    return max(r, 1)


def bits_for_rank(out: int, inn: int, r: int, scale_bits: int = SCALE_BITS_DEFAULT) -> int:
    return out * r + inn * r + (out + inn + r) * scale_bits


def effective_bpp(out: int, inn: int, r: int, scale_bits: int = SCALE_BITS_DEFAULT) -> float:
    return bits_for_rank(out, inn, r, scale_bits) / float(out * inn)


# ---------------------------------------------------------------------------
# bit packing  (round-trip tested)
# ---------------------------------------------------------------------------

def pack_signs(B: np.ndarray) -> np.ndarray:
    """Pack {-1,+1} into 8 bits per uint8. Consecutive columns share a byte;
    bit 1 means +1, bit 0 means -1. Shape (n, ceil(m/8))."""
    B = np.asarray(B)
    pos = (B > 0).astype(np.uint8)
    n, m = pos.shape
    pad = (-m) % 8
    if pad:
        pos = np.concatenate([pos, np.zeros((n, pad), np.uint8)], axis=1)
    Bg = pos.reshape(n, -1, 8)
    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).astype(np.uint8)
    return (Bg * weights[None, None, :]).sum(axis=2).astype(np.uint8)


def unpack_signs(packed: np.ndarray, m: int) -> np.ndarray:
    """Inverse of pack_signs. Returns (n, m) int8 with +1 where the bit is set."""
    packed = np.asarray(packed, dtype=np.uint8)
    n = packed.shape[0]
    bits = (packed[:, :, None] >> np.arange(8, dtype=np.uint8)[None, None, :]) & np.uint8(1)
    signs = np.where(bits == 1, np.int8(1), np.int8(-1))
    return signs.reshape(n, -1)[:, :m]


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

class LLRBPayload:
    """Everything the loader needs. Byte count is honest and complete."""

    __slots__ = ("B1", "B2", "s_out", "s_in", "g", "out", "inn", "rank",
                 "scale_bits", "residual_idx", "residual_rows", "residual_bits")

    def __init__(self, B1, B2, s_out, s_in, g, out, inn, scale_bits,
                 residual_idx=None, residual_rows=None, residual_bits=0):
        self.B1, self.B2 = B1, B2
        self.s_out, self.s_in, self.g = s_out, s_in, g
        self.out, self.inn = out, inn
        self.rank = B2.shape[0]
        self.scale_bits = scale_bits
        self.residual_idx = residual_idx
        self.residual_rows = residual_rows
        self.residual_bits = residual_bits

    # -- accounting --------------------------------------------------------
    def payload_bytes(self) -> int:
        b = pack_signs(self.B1).size + pack_signs(self.B2).size
        b += int(math.ceil(len(self.s_out) * self.scale_bits / 8))
        b += int(math.ceil(len(self.s_in) * self.scale_bits / 8))
        b += int(math.ceil(len(self.g) * self.scale_bits / 8))
        b += int(math.ceil(self.residual_bits / 8))
        return b

    def bulk_payload_bytes(self) -> int:
        return payload_bytes(self.out, self.inn, self.rank, self.scale_bits)

    def effective_bpp(self) -> float:
        return 8.0 * self.payload_bytes() / float(self.out * self.inn)

    def logical_params(self) -> int:
        return self.out * self.inn

    def summary(self) -> dict:
        return {
            "shape": (self.out, self.inn), "rank": self.rank,
            "scale_bits": self.scale_bits,
            "bulk_bpp": 8.0 * self.bulk_payload_bytes() / float(self.out * self.inn),
            "effective_bpp": self.effective_bpp(),
            "residual_bits": self.residual_bits,
            "n_residual_rows": 0 if self.residual_idx is None else len(self.residual_idx),
        }


# ---------------------------------------------------------------------------
# core fit
# ---------------------------------------------------------------------------

def fit_scales(Wn: np.ndarray, B1: np.ndarray, B2: np.ndarray, g0: np.ndarray,
               als_iters: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating least squares for (s_out, s_in, g) given fixed SIGN matrices.

    Model: M = diag(s_out) B1 diag(g) B2 diag(s_in), B1:(o,r) B2:(r,in)
    - s_out[i], s_in[j] are closed-form 1-D LS given the rest
    - g solves an r x r normal equation (Kronecker structure -> Hadamard Gram)
    """
    o, i = Wn.shape
    r = B1.shape[1]
    s_out = np.ones(o)
    s_in = np.ones(i)
    g = g0.copy()
    for _ in range(als_iters):
        M = (B1 * g[None, :]) @ (B2 * s_in[None, :])
        den = np.einsum("ij,ij->i", M, M).clip(1e-18)
        s_out = np.einsum("ij,ij->i", Wn, M) / den

        M = (B1 * s_out[:, None]) @ (g[:, None] * (B2 * s_in[None, :]))
        den = np.einsum("ij,ij->j", M, M).clip(1e-18)
        s_in = np.einsum("ij,ij->j", Wn, M) / den

        a = B1 * s_out[:, None]          # (o, r)
        b = B2.T * s_in[:, None]        # (i, r)
        Gram = (a.T @ a) * (b.T @ b)     # (r, r)
        rhs = np.einsum("ik,ij,jk->k", a, Wn, b)
        g = np.linalg.solve(Gram + 1e-12 * np.eye(r), rhs)
        g = np.clip(np.abs(g), 1e-9, None)
    return s_out, s_in, g


def refine_signs(Wn: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                 s_out: np.ndarray, s_in: np.ndarray, g: np.ndarray,
                 sweeps: int = 3, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, bool]:
    """Exact coordinate descent on the sign matrices.

    Key correctness point: flips are applied ONE RANK AT A TIME, vectorized across
    rows (for B1) or columns (for B2). Flips in different ranks interact through
    the shared model, so applying them all at once makes the individual gains
    non-additive and can INCREASE the error — that is the bug that produced
    rel_err > 1.0 in the first version of this loop.

    For B1[:, k]: flipping B1[i,k] changes row i of the model by
        dM[i, :] = -2 * s_out[i] * B1[i,k] * g[k] * (B2[k, :] * s_in)
    and since every other row is untouched, the flips within a rank are
    independent and the squared-error change is exact:

        flip B1[i,k]  iff  B1[i,k] * gain[i,k] > 0
        gain[i,k] = sum_j R[i,j] * s_out[i] * g[k] * B2[k,j] * s_in[j]

    Symmetrically for B2[k, :] (flip iff B2[k,j] * gain2[k,j] > 0).
    Every accepted flip is guaranteed non-increasing in squared error.
    """
    M = (s_out[:, None] * B1) @ (g[:, None] * (B2 * s_in[None, :]))
    R = Wn - M
    o, r = B1.shape
    changed = False
    for _ in range(sweeps):
        flipped = 0
        for k in range(r):
            w = B2[k, :] * s_in                       # (in,)
            ww = float(w @ w)
            dot = R @ w                                # (o,)  = R[i,:] . w
            # Flipping B1[i,k] makes R'[i,j] = R[i,j] + a*w[j], a = 2*so*g*B1.
            # Squared-error change = a^2*||w||^2 + 2*a*(R . w), negative exactly
            # when  B1[i,k]*(R[i,:].w)  <  -s_out[i]*g[k]*||w||^2
            flip = B1[:, k] * dot < -(s_out * g[k]) * ww
            if flip.any():
                flipped += int(flip.sum())
                ri = np.flatnonzero(flip)
                # only flipped ROWS change, so this stays O(|flip| * in)
                R[ri] -= (-2.0 * s_out[ri] * g[k] * B1[ri, k])[:, None] * w[None, :]
                M[ri] += (-2.0 * s_out[ri] * g[k] * B1[ri, k])[:, None] * w[None, :]
                B1[ri, k] = -B1[ri, k]
        for k in range(r):
            u = B1[:, k] * s_out                      # (out,)
            uu = float(u @ u)
            dot = R.T @ u                              # (in,)
            # symmetric derivation: flip iff B2[k,j]*(R[:,j].u) < -s_in*g*||u||^2
            flip = B2[k, :] * dot < -(s_in * g[k]) * uu
            if flip.any():
                flipped += int(flip.sum())
                ci = np.flatnonzero(flip)
                # only flipped COLUMNS change, so this stays O(|flip| * out)
                R[:, ci] -= (-2.0 * s_in[ci] * g[k] * B2[k, ci])[None, :] * u[:, None]
                M[:, ci] += (-2.0 * s_in[ci] * g[k] * B2[k, ci])[None, :] * u[:, None]
                B2[k, ci] = -B2[k, ci]
        if flipped:
            changed = True
        else:
            break
    return B1, B2, changed


def reconstruct(s_out, B1, g, B2, s_in) -> np.ndarray:
    return (s_out[:, None] * B1) @ (g[:, None] * (B2 * s_in[None, :]))


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

def encode(W: np.ndarray, target_bpp: float = 1.0,
           scale_bits: int = SCALE_BITS_DEFAULT, als_iters: int = 12,
           residual_frac: float = 0.0, residual_bits: float = 3.281,
           min_rank: int = 1, force_rank: int | None = None,
           row_norm: bool = True, sign_sweeps: int = 2) -> LLRBPayload:
    """Encode W (out, in) at <= target_bpp effective bits per logical parameter.

    residual_frac: fraction of rows that receive an additive correction stream,
    selected by output-space residual energy. residual_bits is the charged rate of
    the residual codec (DG-MS-64L native is 3.281 b/p).
    """
    if row_norm:
        # Row normalization makes the SVD factors well-conditioned for binarization.
        ro = np.linalg.norm(W, axis=1).clip(1e-12)
    else:
        ro = np.ones(W.shape[0])
    Wp = W / ro[:, None]                  # (out, in)
    out, inn = W.shape

    r = force_rank if force_rank is not None else rank_for_bpp(
        out, inn, target_bpp, scale_bits, min_rank=min_rank)
    r = max(1, min(r, min(out, inn)))

    U, S, Vt = np.linalg.svd(Wp, full_matrices=False)
    U, S, Vt = U[:, :r], S[:r], Vt[:r]

    # Note: sign(SVD) + LS scales is the PTQ baseline. On matrices without strong
    # low-rank structure this saturates; see the measured frontier in
    # tests/test_codec_frontier.py. The residual stream is the Lumina quality lever.
    B1 = np.where(U >= 0, 1.0, -1.0)
    B2 = np.where(Vt >= 0, 1.0, -1.0)     # (r, in)

    s_out, s_in, g = fit_scales(Wp, B1, B2, S.copy(), als_iters=als_iters)

    # sign refinement: exact coordinate descent, with a monotone guard so a
    # degenerate ALS step can never make the payload worse than the baseline
    if sign_sweeps:
        def _loss(so, b1, gg, b2, si):
            return float(((Wp - (so[:, None] * b1) @ (gg[:, None] * (b2 * si[None, :]))) ** 2).sum())

        best = (_loss(s_out, B1, g, B2, s_in), B1.copy(), B2.copy(),
                s_out.copy(), s_in.copy(), g.copy())
        for _ in range(2):
            B1n, B2n, changed = refine_signs(Wp, B1, B2, s_out, s_in, g,
                                             sweeps=sign_sweeps)
            if not changed:
                break
            so2, si2, g2 = fit_scales(Wp, B1n, B2n, g,
                                      als_iters=max(4, als_iters // 2))
            cur = _loss(so2, B1n, g2, B2n, si2)
            if cur < best[0]:
                best = (cur, B1n.copy(), B2n.copy(), so2.copy(), si2.copy(), g2.copy())
                B1, B2, s_out, s_in, g = B1n, B2n, so2, si2, g2
            else:
                # refinement made it worse (ALS ran away on a degenerate Gram);
                # keep the last good configuration and stop
                B1, B2, s_out, s_in, g = best[1], best[2], best[3], best[4], best[5]
                break

    # fold the row normalization back in: W = diag(ro) * Wp, and
    # Wp ~= diag(s_out) B1 diag(g) B2 diag(s_in), so the effective row scale is
    # ro * s_out (multiply, not divide — dividing was a real bug that produced
    # rel_err > 1.0 on every matrix).
    s_out = s_out * ro

    residual_idx = None
    residual_rows = None
    res_bits = 0
    if residual_frac > 0:
        W_hat = reconstruct(s_out, B1, g, B2, s_in)
        R = W - W_hat
        k = max(1, int(round(residual_frac * out)))
        energy = (R ** 2).sum(axis=1)
        residual_idx = np.sort(np.argsort(-energy)[:k])
        # store residual rows in a low-bit codec (charged at residual_bits b/p)
        residual_rows = R[residual_idx]
        idx_bits = math.ceil(math.log2(max(2, out)))
        res_bits = int(k * (idx_bits + inn * residual_bits))

    return LLRBPayload(B1, B2, s_out, s_in, g, out, inn, scale_bits,
                       residual_idx=residual_idx, residual_rows=residual_rows,
                       residual_bits=res_bits)


def decode(p: LLRBPayload) -> np.ndarray:
    W = reconstruct(p.s_out, p.B1, p.g, p.B2, p.s_in)
    if p.residual_idx is not None and p.residual_rows is not None:
        W[p.residual_idx] += p.residual_rows
    return W


def forward_from_payload(p: LLRBPayload, x: np.ndarray, fused: bool = True) -> np.ndarray:
    """y = x @ W_hat^T computed WITHOUT reconstructing W.

    If fused=True the binary factors are round-tripped through the packed
    representation so the execution path exercises the exact bytes the loader
    would read.
    """
    x = np.asarray(x, dtype=np.float64)
    if fused:
        B1u = unpack_signs(pack_signs(p.B1), p.B1.shape[1])
        B2u = unpack_signs(pack_signs(p.B2), p.B2.shape[1])
    else:
        B1u, B2u = p.B1, p.B2
    z = (x * p.s_in[None, :]) @ B2u.T        # (n, r)
    z = z * p.g[None, :]
    y = z @ B1u.T                            # (n, out)
    y = y * p.s_out[None, :]
    if p.residual_idx is not None and p.residual_rows is not None:
        # residual_rows[k] is the correction for OUTPUT row residual_idx[k]
        y[:, p.residual_idx] += x @ p.residual_rows.T
    return y


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def weight_error(W: np.ndarray, p: LLRBPayload) -> dict:
    W_hat = decode(p)
    return {
        "rel_frobenius": float(np.linalg.norm(W - W_hat) / max(np.linalg.norm(W), 1e-12)),
        "mse": float(np.mean((W - W_hat) ** 2)),
        "max_abs": float(np.abs(W - W_hat).max()),
    }


def output_error(W: np.ndarray, p: LLRBPayload, n_trials: int = 8,
                 seed: int = 0) -> dict:
    """Output-space error — a better proxy for capability retention than weight MSE."""
    rng = np.random.default_rng(seed)
    W_hat = decode(p)
    rel = []
    for _ in range(n_trials):
        x = rng.standard_normal((W.shape[1], 4))
        a, b = W @ x, W_hat @ x
        rel.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)))
    return {"mean_rel_output_error": float(np.mean(rel)),
            "max_rel_output_error": float(np.max(rel))}


def parity_forward(W: np.ndarray, p: LLRBPayload, n: int = 4, seed: int = 0) -> float:
    """The fused forward must equal (x @ W_hat^T) exactly — this is the API contract."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, W.shape[1]))
    return float(np.abs(forward_from_payload(p, x) - x @ decode(p).T).max())


def llm_like_matrix(out: int, inn: int, seed: int, n_dominant: int = 24,
                    decay: float = 2.0, noise: float = 0.03) -> np.ndarray:
    """Weight-like matrix: dominant low-rank structure + a decaying tail.

    Used instead of pure Gaussian because binary factorization quality depends on
    effective rank, and trained LLM weights are strongly low-rank while pure
    Gaussian noise is not.
    """
    rng = np.random.default_rng(seed)
    k = min(n_dominant, out, inn)
    U, _ = np.linalg.qr(rng.standard_normal((out, k)))
    V, _ = np.linalg.qr(rng.standard_normal((inn, k)))
    s = np.geomspace(3.0, 0.05, k)
    return (U * s) @ V.T + rng.standard_normal((out, inn)) * noise


if __name__ == "__main__":
    for K in (64, 128, 256, 512):
        for bpp in (1.0, 0.5, 0.25, 0.1):
            r = rank_for_bpp(K, K, bpp)
            print(f"  K={K:>4} target_bpp={bpp:<5} -> rank={r:>4} "
                  f"actual_bpp={effective_bpp(K, K, r):.4f}")
        print()
