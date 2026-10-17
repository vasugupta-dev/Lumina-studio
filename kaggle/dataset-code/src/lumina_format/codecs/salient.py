"""Salient-weight sparse corrections — the "1.x bpw beats pure 1-bit" lever.

A tiny fraction of weights (outliers by |W| x input-magnitude) dominate the
error. Storing the top-k in fp16 CSR while the rest goes to 1-2 bits buys
more quality per byte than spending those bytes on uniformly higher bits.
Every sparse byte is counted in the effective bpp.
"""

from __future__ import annotations

import numpy as np


def salient_mask(W: np.ndarray, X: np.ndarray | None, frac: float
                 ) -> np.ndarray:
    """Boolean mask of the top-`frac` weights by |W| * input magnitude."""
    W = np.asarray(W, dtype=np.float64)
    if X is not None:
        amp = np.abs(np.asarray(X, dtype=np.float64)).mean(axis=1)
        amp = amp / max(amp.max(), 1e-12)
        score = np.abs(W) * amp[None, :]
    else:
        score = np.abs(W)
    k = max(1, int(score.size * frac))
    thr = np.partition(score.reshape(-1), -k)[-k]
    return score >= thr


def pack_sparse(W: np.ndarray, mask: np.ndarray) -> dict:
    """CSR-ish payload: int32 flat indices + fp16 values."""
    idx = np.flatnonzero(mask).astype(np.int32)
    vals = W.reshape(-1)[idx].astype(np.float16)
    return {"idx": idx, "vals": vals, "shape": tuple(W.shape)}


def unpack_sparse(payload: dict, shape: tuple | None = None) -> np.ndarray:
    shape = tuple(payload["shape"]) if shape is None else shape
    out = np.zeros(int(np.prod(shape)), dtype=np.float32)
    out[payload["idx"].astype(np.int64)] = payload["vals"].astype(np.float32)
    return out.reshape(shape)


def sparse_bytes(payload: dict) -> int:
    """Exact bytes: 4B index + 2B value per entry."""
    return int(payload["idx"].size) * 6


def apply_corrections(Wq: np.ndarray, W_orig: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """Overwrite quantized entries at salient positions with fp16 truth."""
    out = Wq.copy()
    out[mask] = W_orig[mask]
    return out
