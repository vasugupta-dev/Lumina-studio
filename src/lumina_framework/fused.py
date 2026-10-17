"""Fused execution kernels borrowed from the best existing runtimes.

Feature-by-feature provenance (layouts match our own adapters exactly):
- llama.cpp (ggml): repack activations to Q8_0 ONCE per matmul, then INTEGER
  dot-product against quantized weights; row-split threadpool for GEMV.
  The weight matrix is never materialized in FP — the dot runs on quants.
- EXL3: integer dot directly over 3-bit packed codes + super-block scales.
- TensorRT-LLM: FP8-blockwise (E4M3, per-128 scales) activation pipeline
  with fused quant epilogues, and paged-KV block tables for the cache.

CPU reference here is numpy (+numba where installed); the function SHAPES and
integer arithmetic mirror the GPU kernels so a Triton/CUTLASS port is exact.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except Exception:
    numba = None
    _HAS_NUMBA = False

try:
    from . import tops as _TOPS          # optional compiled dotprod kernels
except Exception:
    _TOPS = None

# Below this K the compiled kernel is not used (measured crossover; the old
# K>=512 gate in memory.py still applies on top of this).
TOPS_MIN_K = int(os.environ.get("LUMINA_TOPS_MIN_K", "512"))


# ---------------------------------------------------------------------------
# TRT-LLM style FP8 E4M3 blockwise (exact bit layout: 1-4-3, bias 8, max 448)
# ---------------------------------------------------------------------------

E4M3_MAX = 448.0
E4M3_BIAS = 8


def fp32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Round FP32 -> E4M3 bytes (saturates to 448, NaN preserved)."""
    x = np.asarray(x, dtype=np.float32)
    sbit = (np.signbit(x).astype(np.uint8) << 7)
    ax = np.abs(x)
    out = np.empty(x.shape, dtype=np.uint8)
    nan = np.isnan(ax)
    big = np.isinf(ax) | (ax > E4M3_MAX)
    fin = ~(nan | big)
    f = ax[fin]
    e = np.floor(np.log2(np.maximum(f, 2.0 ** -9))).astype(np.int32)
    sub = f < 2.0 ** -6
    m_sub = np.round(f / (2.0 ** -6) * 8.0).astype(np.int32)
    e = np.clip(e, -6, E4M3_BIAS)
    m = np.round((f / (2.0 ** e) - 1.0) * 8.0).astype(np.int32)
    bump = (~sub) & (m == 8)
    e = np.clip(np.where(bump, e + 1, e), -6, E4M3_BIAS)
    m = np.where(bump, 0, m)
    bits = (((e + E4M3_BIAS) << 3) | m).astype(np.uint8)
    bits = np.where(sub, np.clip(m_sub, 0, 7).astype(np.uint8), bits)
    out[fin] = bits
    out[nan] = np.uint8(0x7F)
    out[big] = np.uint8(0x78)
    return (out | sbit.reshape(out.shape)).astype(np.uint8)


def e4m3_to_fp32(b: np.ndarray) -> np.ndarray:
    """E4M3 bytes -> FP32 (exact inverse for all finite codes)."""
    b = np.asarray(b, dtype=np.uint8)
    s = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    e = ((b >> 3) & 0xF).astype(np.int32) - E4M3_BIAS
    m = (b & 0x7).astype(np.float32) / 8.0
    is_inf_nan = (b & 0x78) == 0x78
    return np.where(is_inf_nan,
                    np.where(m == 0, s * np.inf, np.nan),
                    s * (2.0 ** np.maximum(e, -6)) * (m + (e > -6))).astype(np.float32)


# ---------------------------------------------------------------------------
# TRT-LLM style FP8 blockwise activation pipeline + llama.cpp Q8 repack
# ---------------------------------------------------------------------------

FP8_BLOCK = 128


def q8_repack(x: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """llama.cpp trick: quantize activations to int8 ONCE per matmul."""
    x = np.asarray(x, dtype=np.float32)
    flat = x.reshape(-1)
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    blk = flat.reshape(-1, block)
    amax = np.abs(blk).max(axis=1, keepdims=True).clip(1e-12)
    q = np.round(blk * (127.0 / amax)).clip(-127, 127).astype(np.int8)
    return q.reshape(x.shape if not pad else (-1,))[:x.size].reshape(x.shape), \
        (amax[:, 0] / 127.0).astype(np.float32)


if _HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _idot_q8(sa: np.int8, sb: np.int8) -> np.int32:
        """Integer dot over equal-length int8 vectors (ggml_vec_dot_q8_0)."""
        s = np.int32(0)
        for i in range(sa.shape[0]):
            s += np.int32(sa[i]) * np.int32(sb[i])
        return s

    @numba.jit(nopython=True, cache=True)
    def _gemv_q4_0(xq: np.int8, xs: np.ndarray, wq: np.uint8, ws: np.ndarray,
                   out: np.ndarray, group: int = 32) -> None:
        """Fused Q4_0 GEMV: out[n] = sum_k xq[k]*xs * ((wq-8)*ws).

        wq holds nibble codes 0..15 (value = code - 8, GGUF Q4_0 asymmetric).
        No FP weight matrix is ever formed — mirrors exllama/llama.cpp.
        """
        nrows = wq.shape[0]
        K = xq.shape[0]
        for n in range(nrows):
            acc = np.float32(0.0)
            row = wq[n]
            for b in range(K // group):
                acc += np.float32(_idot_q8(xq[b * group:(b + 1) * group],
                                           (row[b * group:(b + 1) * group].astype(np.int8) - np.int8(8)))) \
                    * np.float32(xs[b]) * np.float32(ws[n, b])
            out[n] = acc


if _HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _gemv_exl3(xq: np.int8, xs: np.ndarray, codes: np.uint8,
                   scales: np.ndarray, levels: np.ndarray,
                   out: np.ndarray) -> None:
        """Fused EXL3 GEMV over packed 3-bit codes + super-block scales.

        codes: (nrows, K) values 0..6; levels: the 7 grid values; scales:
        (nrows, n_super). out[n] = sum over superblocks of
        scale_sb * (x_q . levels[codes]) * xs. Lookup-then-FMA mirrors the
        EXL3 kernel; no FP weight matrix is ever formed.
        """
        nrows, K = codes.shape
        SUPER = 256
        for n in range(nrows):
            acc = np.float32(0.0)
            for b in range(K // 32):
                seg = np.float32(0.0)
                base = b * 32
                for j in range(32):
                    seg += np.float32(xq[base + j]) * np.float32(levels[codes[n, base + j]])
                sb = base // SUPER
                seg = seg * np.float32(xs[b]) * np.float32(scales[n, sb])
                acc += seg
            out[n] = acc


if _HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _gemv_int8(xq: np.int8, xs: np.ndarray, wq: np.int8,
                   ws: np.ndarray, out: np.ndarray) -> None:
        """Fused int8 GEMV over a TensorContainer's native payload.

        xq: (K,) repacked activations; xs: (K/32,) act scales;
        wq: (N, K) weight quants; ws: (N,) row scales.
        Integer dot per 32-elem block, one float scale-up per block.
        """
        nrows, K = wq.shape
        nb = K // 32
        for n in range(nrows):
            acc = np.float32(0.0)
            wrow = wq[n]
            s = np.float32(ws[n])
            for b in range(nb):
                seg = np.int32(0)
                base = b * 32
                for j in range(32):
                    seg += np.int32(xq[base + j]) * np.int32(wrow[base + j])
                acc += np.float32(seg) * np.float32(xs[b]) * s
            out[n] = acc


def fused_gemv_int8(x: np.ndarray, container) -> np.ndarray:
    """Decode fast path: repack x once, integer-dot the container payload.

    Uses the compiled dotprod kernel (tops.py) when available and K is large
    enough; otherwise the numba integer path; otherwise dequant+BLAS. The
    compiled path is the only one that beats fp32 BLAS at the real decode
    shape (measured: 30.1 GMAC/s vs 2.4-3.5 BLAS, see
    scripts/bench_tops_feasibility.py and tests/test_tops_kernel.py).
    """
    M = x.shape[0]
    outs = np.empty((M, container._q8.shape[0]), dtype=np.float32)
    K = container._q8.shape[1]
    use_tops = (_TOPS is not None and _TOPS.available()
                and K >= TOPS_MIN_K
                and os.environ.get("LUMINA_TOPS_DISABLE") != "1")
    for m in range(M):
        xq, xs = q8_repack(x[m].reshape(-1))
        if use_tops:
            r = _TOPS.gemv_int8(xq, xs, container._q8, container._scale, outs[m])
            if "out" in r:
                continue
        row = np.empty(container._q8.shape[0], dtype=np.float32)
        if _HAS_NUMBA:
            _gemv_int8(xq, xs, container._q8, container._scale.reshape(-1), row)
        else:
            w = container.dequantize()
            row = (xq.astype(np.float32) * np.repeat(xs, 32)[:w.shape[1]]) @ w.T
        outs[m] = row
    return outs


def fused_gemv_q4_0(x: np.ndarray, container) -> np.ndarray:
    """GGUF Q4_0 decode path: 4-bit packed payload + integer dotprod.

    Requires the compiled dotprod kernel (tops.py); without it this returns
    {"error": ...} so callers keep the dequant path.
    """
    if _TOPS is None or not _TOPS.available():
        return {"error": "q4_0 dotprod kernel unavailable: " + (
            _TOPS.reason() if _TOPS else "tops module missing")}
    wq = getattr(container, "_q4_0", None)
    if wq is None:
        return {"error": "container has no _q4_0 payload"}
    ws = getattr(container, "_scale", None)
    if ws is None or ws.ndim != 2:
        return {"error": "q4_0 container needs (N, K/32) scales"}
    M = x.shape[0]
    outs = np.empty((M, wq.shape[0]), dtype=np.float32)
    for m in range(M):
        xq, xs = q8_repack(x[m].reshape(-1))
        r = _TOPS.gemv_q4_0(xq, xs, wq, ws, outs[m])
        if "out" not in r:
            return r
    return outs


def fused_gemm_int8(x: np.ndarray, container) -> np.ndarray:
    """Prefill path (M>1): integer dotprod GEMM over an int8 container."""
    if _TOPS is None or not _TOPS.available():
        return {"error": "int8 dotprod kernel unavailable: " + (
            _TOPS.reason() if _TOPS else "tops module missing")}
    xq = np.empty(x.shape, dtype=np.int8)
    xs = np.empty((x.shape[0], x.shape[1] // 32), dtype=np.float32)
    for m in range(x.shape[0]):
        xq[m], xs[m] = q8_repack(x[m].reshape(-1))
    r = _TOPS.gemm_int8(xq, xs, container._q8, container._scale)
    return r["out"] if "out" in r else r


def tops_status() -> dict:
    """Honest status for the UI/tests: available, reason, kernel info."""
    if _TOPS is None:
        return {"available": False, "reason": "tops module missing"}
    return _TOPS.info()
