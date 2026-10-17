"""Block-GPTQ parity: the fast path must reproduce the reference per-column
loop (same codes up to float summation grouping, same relW to 1e-6) and be
measurably faster. Phone-safe: 256/512-sized matrices only.

  python3 tests/test_gptq_block_parity.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_format.codecs.ptq_ladder import (  # noqa: E402
    gptq, quantize_int, quantize_binary, dequantize, group_scales, _levels,
)

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" — " + detail) if not cond else ""))
    if not cond:
        FAILS.append(name)


def ref_gptq(W, Hinv, nbits, group_size, act_order, per_channel_scale, dtype=np.float64):
    """The original per-column loop, kept verbatim as the reference."""
    W = np.asarray(W, dtype=dtype).copy()
    o, i = W.shape
    if per_channel_scale:
        scales = np.abs(W).max(axis=1).clip(1e-12)
    else:
        scales = group_scales(W, group_size, nbits)
    if Hinv is None:
        Hinv = np.eye(i)
    else:
        Hinv = np.asarray(Hinv, dtype=np.float64)
    diag = np.maximum(np.diag(Hinv).copy(), 1e-12)
    levels = np.asarray(_levels(nbits), dtype=dtype)
    codes = np.zeros((o, i), dtype=np.int64)
    if act_order:
        perm = np.argsort(-diag)
    else:
        perm = np.arange(i)
    for p in range(i):
        j = int(perm[p])
        rest = perm[p + 1:]
        s = scales if per_channel_scale else scales[:, j // group_size]
        wn = W[:, j] / s
        idx = np.abs(wn[:, None] - levels[None, :]).argmin(axis=1).astype(np.int64)
        codes[:, j] = idx
        q = s * levels[idx]
        err = (W[:, j] - q) / diag[j]
        if rest.size:
            W[:, rest] -= np.outer(err, Hinv[j, rest])
    return codes, scales


def relw(W, q_layer):
    Wq = dequantize(q_layer)
    return float(np.linalg.norm(W - Wq) / np.linalg.norm(W))


def main() -> int:
    rng = np.random.default_rng(7)
    for (o, i) in [(96, 96), (256, 256)]:
        W = rng.standard_normal((o, i)) * (rng.random(i)[None, :] * 2 + 0.3)
        X = rng.standard_normal((i, 256))
        H = (X @ X.T) * (2.0 / X.shape[1]) + 0.02 * np.eye(i)
        # damped inverse Hessian like the pipeline uses
        Hinv = np.linalg.inv(H + 0.01 * np.mean(np.diag(H)) * np.eye(i))
        for nbits in (4, 2):
            for gs in (32, 64):
                for ao in (True, False):
                    c_new, s_new = gptq(W, Hinv, nbits, group_size=gs, act_order=ao)
                    c_ref, s_ref = ref_gptq(W, Hinv, nbits, gs, ao, False)
                    ql_new = quantize_int(W, nbits, Hinv, group_size=gs)
                    ql_ref = quantize_int.__wrapped__(W, nbits, Hinv, group_size=gs) if hasattr(quantize_int, "__wrapped__") else None
                    mism = float((c_new != c_ref).mean())
                    check(f"gptq({o}x{i}, {nbits}b, gs{gs}, ao={ao}) codes match ref",
                          mism < 1e-9, f"mismatch rate {mism:.2e}")
        # relW equality through the public codec
        r1 = relw(W, quantize_int(W, 4, Hinv, group_size=64))
        c_ref, _ = ref_gptq(W, Hinv, 4, 64, True, False)
        s_ref = group_scales(W, 64, 4)
        W_ref = s_ref[:, c_ref // 64 if False else np.zeros((o, i), int)]  # placeholder
        # build ref reconstruction directly
        levels = np.asarray(_levels(4))
        W_ref = np.zeros_like(W)
        for j in range(i):
            g = j // 64
            W_ref[:, j] = s_ref[:, g] * levels[c_ref[:, j]]
        d = abs(r1 - float(np.linalg.norm(W - W_ref) / np.linalg.norm(W)))
        check(f"gptq({o}x{i}, int4) relW equals reference", d < 1e-6, f"delta {d:.2e}")

    # binary parity
    rng2 = np.random.default_rng(11)
    W = rng2.standard_normal((128, 128))
    X = rng2.standard_normal((128, 200))
    Hinv = np.linalg.inv((X @ X.T) * (2.0 / 200) + 0.05 * np.eye(128))
    qb = quantize_binary(W, Hinv, group_size=64)
    Wb = dequantize(qb)
    # reference binary loop
    Wc = W.astype(np.float64).copy(); i = 128; gs = 64
    s0 = group_scales(W, gs, 1); diag = np.maximum(np.diag(Hinv).copy(), 1e-12)
    sign = np.zeros_like(Wc)
    for j in range(i):
        sg = np.where(Wc[:, j] >= 0, 1.0, -1.0); sign[:, j] = sg
        err = (Wc[:, j] - s0[:, j // gs] * sg) / diag[j]
        if j + 1 < i:
            Wc[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
    check("binary fast path matches reference signs",
          float((np.sign(Wb) != sign).mean()) < 1e-9)

    # speed: fast vs reference on 512x512 (phone-safe size)
    rng3 = np.random.default_rng(3)
    W = rng3.standard_normal((512, 512))
    X = rng3.standard_normal((512, 128))
    Hinv = np.linalg.inv((X @ X.T) * (2.0 / 128) + 0.02 * np.eye(512))
    t0 = time.perf_counter(); gptq(W, Hinv, 4, group_size=64); t_new = time.perf_counter() - t0
    t0 = time.perf_counter(); ref_gptq(W, Hinv, 4, 64, True, False); t_ref = time.perf_counter() - t0
    check(f"block-GPTQ faster than reference on 512x512 (new {t_new:.2f}s vs ref {t_ref:.2f}s)",
          t_new < t_ref, "no speedup!")

    print("---")
    if FAILS:
        print("RESULT: FAIL (%d)" % len(FAILS))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
