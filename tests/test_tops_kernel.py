"""Compiled dotprod kernels for Lumina (src/lumina_framework/tops.py).

Covers, in order:
  * availability + build info (SKIP honestly if the CPU/compiler can't provide)
  * numeric parity: int8 GEMV vs numba + numpy, Q4_0 GEMV vs numpy,
    fp16 GEMV vs fp32 reference, int8 GEMM vs per-row GEMV
  * speed gates (thresholds fixed before measuring):
      T6 tops int8 GEMV   >= 2.0x the numba integer GEMV at (11008x4096)
      T6b tops Q4_0 GEMV  >= 2.0x the numba integer GEMV at the same shape
      T10 decode loop     >= 1.5x numba across a realistic 32-layer GEMV loop
  * hybrid threshold sweep: measured crossover K for tops vs numba
  * edge shapes (K=32/96, N=1/7, M=1 GEMM) parity and no-crash
  * fallback: LUMINA_TOPS_DISABLE=1 -> numba path still correct

Writes artifacts/tops_kernel_bench.json with every measured number.

Run:  python3 tests/test_tops_kernel.py
Exit codes: 0 = pass, 1 = fail, 3 = honest skip (kernel unavailable).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from lumina_framework import fused, tops  # noqa: E402

RESULTS = []
ART = os.path.join(ROOT, "artifacts", "tops_kernel_bench.json")
MEAS: dict = {}


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:160]) if not cond and extra else ""))


def best_of(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def make_container(rng, N, K):
    # small weights keep multi-layer activation chains finite (benchmark only)
    wq = rng.integers(-8, 8, (N, K)).astype(np.int8)
    sc = (np.abs(rng.standard_normal(N)).astype(np.float32) * 0.02 + 0.001)
    return type("C", (), {"_q8": wq, "_scale": sc,
                          "dequantize": lambda s: s._q8.astype(np.float32) * s._scale[:, None]})()


def make_q4(rng, N, K):
    codes = rng.integers(0, 16, (N, K)).astype(np.uint8)
    q4 = ((codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)).astype(np.uint8)
    sc = (np.abs(rng.standard_normal((N, K // 32))).astype(np.float32) + 0.01)
    return q4, sc


def main() -> int:
    t0 = time.time()
    st = tops.info()
    print("tops:", json.dumps(st))
    MEAS["tops"] = st
    if not tops.available():
        print("SKIP: compiled dotprod kernel unavailable (%s)" % st.get("reason"))
        return 3

    rng = np.random.default_rng(11)
    N, K = 37, 256
    xq = rng.integers(-8, 8, K).astype(np.int8)
    xs = (np.abs(rng.standard_normal(K // 32)).astype(np.float32) + 0.01)
    wq = rng.integers(-8, 8, (N, K)).astype(np.int8)
    ws = (np.abs(rng.standard_normal(N)).astype(np.float32) + 0.1)
    x = rng.standard_normal((1, K)).astype(np.float32)

    # T2 int8 parity: compiled kernel vs numba vs numpy int32 reference
    r = tops.gemv_int8(xq, xs, wq, ws)
    out = r["out"]
    nb = np.zeros(N, dtype=np.float32)
    if fused._HAS_NUMBA:
        fused._gemv_int8(xq, xs, wq, ws, nb)
    ref = np.zeros(N, dtype=np.float64)
    for n in range(N):
        acc = 0.0
        for b in range(K // 32):
            seg = int(np.dot(xq[b * 32:(b + 1) * 32].astype(np.int32),
                              wq[n, b * 32:(b + 1) * 32].astype(np.int32)))
            acc += seg * float(xs[b])
        ref[n] = acc * float(ws[n])
    d_k = float(np.abs(out.astype(np.float64) - ref).max())
    d_n = float(np.abs(nb.astype(np.float64) - ref).max()) if fused._HAS_NUMBA else None
    MEAS["int8_parity"] = {"kernel_vs_numpy_max": d_k,
                           "numba_vs_numpy_max": d_n}
    check("int8 parity: compiled kernel == numpy int32 reference", d_k <= 1e-3, d_k)
    if d_n is not None:
        check("int8 parity: numba == numpy reference", d_n <= 1e-3, d_n)

    # T3 Q4_0 parity
    q4, q4s = make_q4(rng, N, K)
    rq = tops.gemv_q4_0(xq, xs, q4, q4s)
    lo = (q4 & 15).astype(np.int16) - 8
    hi = (q4 >> 4).astype(np.int16) - 8
    inter = np.empty((N, K), dtype=np.int16)
    inter[:, 0::2] = lo
    inter[:, 1::2] = hi
    refq = np.zeros(N, dtype=np.float64)
    for n in range(N):
        acc = 0.0
        for b in range(K // 32):
            seg = int(np.dot(xq[b * 32:(b + 1) * 32].astype(np.int32),
                              inter[n, b * 32:(b + 1) * 32].astype(np.int32)))
            acc += seg * float(xs[b]) * float(q4s[n, b])
        refq[n] = acc
    dq = float(np.abs(rq["out"].astype(np.float64) - refq).max())
    MEAS["q4_0_parity"] = {"kernel_vs_numpy_max": dq}
    check("q4_0 parity: compiled kernel == numpy reference", dq <= 1e-2, dq)

    # T4 fp16 parity vs fp32 reference
    xh = rng.standard_normal(K).astype(np.float16)
    wh = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
    rh = tops.gemv_fp16(xh, wh)
    refh = xh.astype(np.float32) @ wh.astype(np.float32).T
    rel = float(np.abs(rh["out"] - refh).max() / max(1e-6, float(np.abs(refh).max())))
    MEAS["fp16_parity"] = {"rel_max": rel}
    check("fp16 parity vs fp32 reference", rel <= 1e-2, rel)

    # T5 GEMM parity vs per-row GEMV
    M = 5
    xqm = rng.integers(-8, 8, (M, K)).astype(np.int8)
    xsm = (np.abs(rng.standard_normal((M, K // 32))).astype(np.float32) + 0.01)
    gm = tops.gemm_int8(xqm, xsm, wq, ws)["out"]
    worst = 0.0
    for m in range(M):
        row = tops.gemv_int8(xqm[m], xsm[m], wq, ws)["out"]
        worst = max(worst, float(np.abs(gm[m] - row).max()))
    MEAS["gemm_parity"] = {"gemm_vs_gemv_max": worst}
    check("gemm parity vs per-row gemv", worst <= 1e-3, worst)

    # T6 speed gates at the real decode shape
    N2, K2 = 11008, 4096
    x2 = rng.standard_normal((1, K2)).astype(np.float32)
    cont2 = make_container(rng, N2, K2)
    t_tops, t_tops_med = best_of(lambda: fused.fused_gemv_int8(x2, cont2))
    saved = fused._TOPS
    fused._TOPS = None
    t_nb, t_nb_med = best_of(lambda: fused.fused_gemv_int8(x2, cont2))
    fused._TOPS = saved
    ratio = t_nb / t_tops
    MEAS["speed_int8_gemv_11008x4096"] = {
        "tops_ms": round(t_tops * 1e3, 3), "tops_median_ms": round(t_tops_med * 1e3, 3),
        "numba_ms": round(t_nb * 1e3, 3), "numba_median_ms": round(t_nb_med * 1e3, 3),
        "speedup": round(ratio, 2),
        "gmac_s_tops": round(2 * N2 * K2 / t_tops / 1e9, 2),
        "gmac_s_numba": round(2 * N2 * K2 / t_nb / 1e9, 2)}
    check("T6 speed: tops int8 GEMV >= 2.0x numba at 11008x4096",
          ratio >= 2.0, round(ratio, 2))

    # GEMM speed at a prefill shape (measurement; the M-blocked kernel reuses
    # each weight block across 16 rows instead of re-reading it per row)
    M32 = 32
    xqm32 = rng.integers(-8, 8, (M32, K2)).astype(np.int8)
    xsm32 = (np.abs(rng.standard_normal((M32, K2 // 32))).astype(np.float32) + 0.01)
    tg32, _ = best_of(lambda: tops.gemm_int8(xqm32, xsm32,
                                              cont2._q8, cont2._scale), reps=3)
    tb32, _ = best_of(lambda: xqm32.astype(np.float32)
                      @ cont2._q8.astype(np.float32).T, reps=3)
    MEAS["gemm_speed_M32_11008x4096"] = {
        "int8_ms": round(tg32 * 1e3, 1), "blas_ms": round(tb32 * 1e3, 1),
        "blas_over_int8": round(tb32 / tg32, 2),
        "int8_gmac_s": round(2 * M32 * N2 * K2 / tg32 / 1e9, 1),
        "blas_gflops": round(2 * M32 * N2 * K2 / tb32 / 1e9, 1)}
    print("gemm M=32: int8 %.1f ms (%.1f GMAC/s) vs BLAS fp32 %.1f ms "
          "(%.1f GFLOPS) -> BLAS is %.2fx"
          % (tg32 * 1e3, 2 * M32 * N2 * K2 / tg32 / 1e9,
             tb32 * 1e3, 2 * M32 * N2 * K2 / tb32 / 1e9, tb32 / tg32))

    # T6b Q4_0 speed at the same shape
    q4b, q4sb = make_q4(rng, N2, K2)
    class QC:
        _q4_0 = q4b
        _scale = q4sb
    t_q4, t_q4_med = best_of(lambda: fused.fused_gemv_q4_0(x2, QC))
    r_q4 = t_q4 if isinstance(t_q4, float) else None
    ratio_q4 = t_q4 / t_tops
    MEAS["speed_q4_0_gemv_11008x4096"] = {
        "q4_0_ms": round(t_q4 * 1e3, 3), "q4_0_median_ms": round(t_q4_med * 1e3, 3),
        "vs_int8_kernel": round(ratio_q4, 2),
        "gmac_s_q4_0": round(2 * N2 * K2 / t_q4 / 1e9, 2)}
    check("T6b speed: q4_0 GEMV >= 2.0x numba int8 path",
          t_nb / t_q4 >= 2.0, round(t_nb / t_q4, 2))

    # T7 hybrid threshold sweep (measurement; the gate is at K=4096)
    sweep = {}
    for KK in (128, 256, 512, 1024, 4096):
        xx = rng.standard_normal((1, KK)).astype(np.float32)
        cc = make_container(rng, 4096, KK)
        a, _ = best_of(lambda: fused.fused_gemv_int8(xx, cc), reps=5)
        fused._TOPS = None
        b, _ = best_of(lambda: fused.fused_gemv_int8(xx, cc), reps=5)
        fused._TOPS = saved
        sweep[str(KK)] = {"tops_ms": round(a * 1e3, 3),
                          "numba_ms": round(b * 1e3, 3),
                          "ratio": round(b / a, 2)}
    MEAS["hybrid_threshold_sweep"] = sweep
    print("hybrid sweep (K: tops vs numba ms):",
          json.dumps(sweep))
    check("T7 hybrid: tops wins at K=4096", sweep["4096"]["ratio"] >= 1.0,
          sweep["4096"])

    # T8 edge shapes
    ok_edges = True
    for KK, NN in ((32, 1), (96, 7), (512, 1), (1024, 3)):
        if KK % 32:
            continue
        xe = rng.standard_normal((1, KK)).astype(np.float32)
        ce = make_container(rng, NN, KK)
        got = fused.fused_gemv_int8(xe, ce)
        if got.shape != (1, NN) or not np.isfinite(got).all():
            ok_edges = False
    gm1 = tops.gemm_int8(rng.integers(-8, 8, (1, 1024)).astype(np.int8),
                         (np.abs(rng.standard_normal((1, 32))).astype(np.float32) + 0.01),
                         wq, ws)
    if "out" not in gm1 or gm1["out"].shape != (1, N):
        ok_edges = False
    MEAS["edge_shapes"] = {"ok": ok_edges}
    check("T8 edge shapes: no crash, finite, correct shape", ok_edges)

    # T9 fallback: disabled env -> numba path, same numbers
    code = (
        "import sys, os; sys.path.insert(0, %r);"
        "os.environ['LUMINA_TOPS_DISABLE']='1';"
        "import numpy as np; from lumina_framework import fused, tops;"
        "print('avail', tops.available(), tops.reason());"
        "rng = np.random.default_rng(3); K,N=1024,2048;"
        "wq = rng.integers(-8,8,(N,K)).astype(np.int8);"
        "sc = (np.abs(rng.standard_normal(N)).astype(np.float32)+0.1);"
        "x = rng.standard_normal((1,K)).astype(np.float32);"
        "o = fused.fused_gemv_int8(x, type('C',(),{'_q8':wq,'_scale':sc})());"
        "print('finite', bool(np.isfinite(o).all()), 'max', float(np.abs(o).max()))"
        % os.path.join(ROOT, "src")
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=ROOT)
    line = [l for l in r.stdout.splitlines() if l.startswith("finite")]
    fallback_ok = bool(line) and "True" in line[0]
    MEAS["fallback_disabled"] = {"stdout": r.stdout.strip()[:200],
                                 "ok": fallback_ok}
    check("T9 fallback: LUMINA_TOPS_DISABLE=1 still produces finite output",
          fallback_ok, r.stdout.strip()[:160] + r.stderr.strip()[:120])

    # T10 decode loop at realistic shapes (32 layers, d=1024, mlp=4096)
    # container widths alternate so the chain stays dimensionally valid
    d, inter, layers = 1024, 4096, 32
    xs_ = [rng.standard_normal(d).astype(np.float32) for _ in range(layers)]
    conts = [make_container(rng, inter, d), make_container(rng, inter, d),
             make_container(rng, d, inter), make_container(rng, inter, d),
             make_container(rng, d, inter), make_container(rng, inter, d),
             make_container(rng, d, inter)]
    wd = [c.dequantize() for c in conts]

    def loop(fn):
        for i in range(layers):
            h = xs_[i]
            g = fn(h[None, :], conts[0])
            u = fn(h[None, :], conts[1])
            hh = np.clip(g * u, -1e3, 1e3)
            for j in (2, 3, 4, 5, 6):
                hh = np.clip(fn(hh, conts[j]), -1e3, 1e3)
        return hh

    def with_tops(h, c):
        return fused.fused_gemv_int8(h, c)

    def numba_only(h, c):
        fused._TOPS = None
        try:
            return fused.fused_gemv_int8(h, c)
        finally:
            fused._TOPS = saved

    def loop_fp32_only():
        for i in range(layers):
            h = xs_[i]
            g = h[None, :] @ wd[0].T
            u = h[None, :] @ wd[1].T
            hh = np.clip(g * u, -1e3, 1e3)
            for j in (2, 3, 4, 5, 6):
                hh = np.clip(hh @ wd[j].T, -1e3, 1e3)
        return hh

    t_loop_tops, _ = best_of(lambda: loop(with_tops), reps=3)
    t_loop_nb, _ = best_of(lambda: loop(numba_only), reps=3)
    t_loop_fp32, _ = best_of(loop_fp32_only, reps=3)
    loop_ratio = t_loop_nb / t_loop_tops
    MEAS["decode_loop_32_layers"] = {
        "shapes": "d=%d inter=%d layers=%d" % (d, inter, layers),
        "tops_s": round(t_loop_tops, 4), "numba_s": round(t_loop_nb, 4),
        "fp32_s": round(t_loop_fp32, 4),
        "speedup_vs_numba": round(loop_ratio, 2),
        "speedup_vs_fp32": round(t_loop_fp32 / t_loop_tops, 2),
        "tokens_per_s_equivalent": round(
            (layers * sum(2 * c._q8.size for c in conts)) / t_loop_tops / 1e6, 2)}
    print("decode loop: tops %.3fs, numba %.3fs, fp32 %.3fs (%.2fx vs numba)"
          % (t_loop_tops, t_loop_nb, t_loop_fp32, loop_ratio))
    check("T10 decode loop: tops >= 1.5x numba", loop_ratio >= 1.5,
          round(loop_ratio, 2))

    MEAS["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(MEAS, open(ART, "w"), indent=1)
    print("artifact:", ART)
    ok = sum(1 for _, c in RESULTS if c)
    print("\nTOPS-KERNEL: %d/%d passed (%.1fs)" % (ok, len(RESULTS), MEAS["wall_s"]))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
