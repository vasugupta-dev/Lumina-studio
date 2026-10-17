"""TOPS-class compiled kernels for Lumina (ARMv8.2+ dotprod).

The feasibility study (``scripts/bench_tops_feasibility.py``) measured that an
int8 dotprod GEMV is 3-6x faster than fp32 BLAS GEMV on the real decode shape,
while the shipped scalar (numba) integer GEMV measures 0.54-0.89x BLAS. This
module makes the measured win usable without adding a runtime dependency:

  * the C source lives next to this file (``kernels/tops_int8.c``) and is
    compiled on demand with the system C compiler into a cache directory
    (``_build/``, git-ignored; override with ``LUMINA_TOPS_BUILD``)
  * the CPU must advertise ``asimddp`` (ARMv8.2 dotprod) or nothing is built
  * a numeric self-test against numpy int32 references runs before the kernel
    is trusted; any failure marks the module unavailable
  * every public call returns ``{"error": ...}`` style dicts or raises nothing:
    callers fall back to the numba/numpy paths

Air-gap switch: ``LUMINA_TOPS_DISABLE=1`` forces unavailable.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "kernels", "tops_int8.c")
VERSION = 4

_STATE: dict | None = None
_LOCK = threading.Lock()


def _build_dir() -> str:
    """Executable cache dir: the sdcard mount is noexec, so default to tmpfs."""
    d = os.environ.get("LUMINA_TOPS_BUILD")
    if not d:
        d = os.path.join(os.environ.get("TMPDIR") or "/tmp", "opencode",
                         "lumina_tops")
    return d


def _cpu_has_dotprod() -> tuple[bool, str]:
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("features"):
                feats = line.split(":", 1)[1].split()
                return ("asimddp" in feats), "asimddp " + (
                    "present" if "asimddp" in feats else "absent")
    except Exception as e:  # noqa: BLE001
        return False, "cpuinfo unreadable: " + str(e)[:80]
    return False, "no features line in cpuinfo"


def _compile(src: bytes, tag: str) -> tuple[str | None, str]:
    """Try to build the shared object; return (path, error)."""
    d = _build_dir()
    key = hashlib.sha1(src).hexdigest()[:12]
    out = os.path.join(d, "tops_int8_%s_%s.so" % (key, tag))
    if os.path.exists(out):
        return out, ""
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        return None, "build dir unusable: " + str(e)[:100]
    variants = [
        (["-O3", "-shared", "-fPIC", "-march=armv8.2-a+dotprod+fp16",
          "-DTOPS_USE_PTHREAD", "-pthread"], "pthread"),
        (["-O3", "-shared", "-fPIC", "-march=armv8.2-a+dotprod+fp16"], "serial"),
    ]
    last = ""
    for flags, name in variants:
        cand = os.path.join(d, "tops_int8_%s_%s.so" % (key, name))
        cmd = ["gcc"] + flags + ["-o", cand, SRC, "-lm"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except Exception as e:  # noqa: BLE001
            last = "gcc failed: " + str(e)[:100]
            continue
        if r.returncode == 0 and os.path.exists(cand):
            return cand, ""
        last = (r.stderr or "").strip().splitlines()[-1][:160] if r.stderr else "compile failed"
    return None, last


def _bind(path: str) -> dict:
    lib = ctypes.CDLL(path)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.c_int
    lib.tops_version.restype = ip
    if int(lib.tops_version()) != VERSION:
        raise RuntimeError("kernel version mismatch")
    lib.tops_gemv_int8.argtypes = [i8p, f32p, i8p, f32p, f32p, ip, ip]
    lib.tops_gemv_q4_0.argtypes = [i8p, f32p, u8p, f32p, f32p, ip, ip]
    lib.tops_gemv_q4_0_v2.argtypes = [i8p, f32p, u8p, f32p, f32p, ip, ip]
    lib.tops_gemv_int8_mt.argtypes = [i8p, f32p, i8p, f32p, f32p, ip, ip]
    lib.tops_gemv_fp16.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   f32p, ip, ip]
    lib.tops_gemm_int8.argtypes = [i8p, f32p, i8p, f32p, f32p, ip, ip, ip]
    lib.tops_set_threads.argtypes = [ip]
    lib.tops_get_threads.argtypes = []
    lib.tops_get_threads.restype = ip
    for fn in (lib.tops_gemv_int8, lib.tops_gemv_q4_0, lib.tops_gemv_q4_0_v2,
               lib.tops_gemv_int8_mt, lib.tops_gemv_fp16, lib.tops_gemm_int8,
               lib.tops_set_threads):
        fn.restype = None
    return {"lib": lib}


def _ref_int8(xq, xs, wq, ws) -> np.ndarray:
    """Scalar numpy reference (int32 exact) for tops_gemv_int8."""
    N, K = wq.shape
    nb = K // 32
    out = np.zeros(N, dtype=np.float64)
    for n in range(N):
        acc = 0.0
        for b in range(nb):
            seg = int(np.dot(xq[b * 32:(b + 1) * 32].astype(np.int32),
                              wq[n, b * 32:(b + 1) * 32].astype(np.int32)))
            acc += seg * float(xs[b])
        out[n] = acc * float(ws[n])
    return out


def _ref_q4_0(xq, xs, wq, ws) -> np.ndarray:
    """wq is the packed payload (N, K/2); ws is (N, K/32)."""
    N = wq.shape[0]
    K = wq.shape[1] * 2
    nb = K // 32
    out = np.zeros(N, dtype=np.float64)
    # interleaved nibble order the kernel produces: [lo0, hi0, lo1, hi1, ...]
    lo = (wq.astype(np.uint8) & 15).astype(np.int16) - 8
    hi = (wq.astype(np.uint8) >> 4).astype(np.int16) - 8
    inter = np.empty((N, K), dtype=np.int16)
    inter[:, 0::2] = lo
    inter[:, 1::2] = hi
    for n in range(N):
        acc = 0.0
        for b in range(nb):
            seg = int(np.dot(xq[b * 32:(b + 1) * 32].astype(np.int32),
                              inter[n, b * 32:(b + 1) * 32].astype(np.int32)))
            acc += seg * float(xs[b]) * float(ws[n, b])
        out[n] = acc
    return out


def _self_test(lib: dict) -> str:
    """Return '' when all kernels match the numpy references."""
    rng = np.random.default_rng(11)
    N, K = 37, 256
    xq = rng.integers(-8, 8, K).astype(np.int8)
    xs = (np.abs(rng.standard_normal(K // 32)).astype(np.float32) + 0.01)
    wq = rng.integers(-8, 8, (N, K)).astype(np.int8)
    ws = (np.abs(rng.standard_normal(N)).astype(np.float32) + 0.1)
    out = np.zeros(N, dtype=np.float32)
    lib["lib"].tops_gemv_int8(xq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              wq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              N, K)
    ref = _ref_int8(xq, xs, wq, ws)
    d = float(np.abs(out.astype(np.float64) - ref).max())
    if d > 1e-3:
        return "int8 self-test max diff %g" % d
    # multithreaded variant must match the serial one exactly
    lib["lib"].tops_set_threads(4)
    out_mt = np.zeros(N, dtype=np.float32)
    lib["lib"].tops_gemv_int8_mt(
        xq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_mt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, K)
    d_mt = float(np.abs(out_mt.astype(np.float64) - ref).max())
    lib["lib"].tops_set_threads(8)
    if d_mt > 1e-3:
        return "int8_mt self-test max diff %g" % d_mt
    # q4_0: packed payload (N, K/2) bytes; byte j = weights 2j (lo) and 2j+1 (hi)
    codes = rng.integers(0, 16, (N, K)).astype(np.uint8)
    q4 = ((codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)).astype(np.uint8)
    q4s = (np.abs(rng.standard_normal((N, K // 32))).astype(np.float32) + 0.01)
    out2 = np.zeros(N, dtype=np.float32)
    lib["lib"].tops_gemv_q4_0(
        xq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q4.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        q4s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, K)
    ref2 = _ref_q4_0(xq, xs, q4, q4s)
    d2 = float(np.abs(out2.astype(np.float64) - ref2).max())
    if d2 > 1e-2:
        return "q4_0 self-test max diff %g" % d2
    # v2 unpack must match the reference too
    out2b = np.zeros(N, dtype=np.float32)
    lib["lib"].tops_gemv_q4_0_v2(
        xq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q4.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        q4s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out2b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, K)
    d2b = float(np.abs(out2b.astype(np.float64) - ref2).max())
    if d2b > 1e-2:
        return "q4_0_v2 self-test max diff %g" % d2b
    # fp16
    xh = rng.standard_normal(K).astype(np.float16)
    wh = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
    out3 = np.zeros(N, dtype=np.float32)
    lib["lib"].tops_gemv_fp16(xh.ctypes.data, wh.ctypes.data,
                              out3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              N, K)
    ref3 = (xh.astype(np.float32) @ wh.astype(np.float32).T)
    rel = float(np.abs(out3 - ref3).max() / max(1e-6, float(np.abs(ref3).max())))
    if rel > 1e-2:
        return "fp16 self-test rel err %g" % rel
    # gemm vs per-row gemv
    M = 5
    xqm = rng.integers(-8, 8, (M, K)).astype(np.int8)
    xsm = (np.abs(rng.standard_normal((M, K // 32))).astype(np.float32) + 0.01)
    outm = np.zeros((M, N), dtype=np.float32)
    lib["lib"].tops_gemm_int8(
        xqm.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        xsm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        outm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, N, K)
    worst = 0.0
    for m in range(M):
        refm = _ref_int8(xqm[m], xsm[m], wq, ws)
        worst = max(worst, float(np.abs(outm[m].astype(np.float64) - refm).max()))
    if worst > 1e-2:
        return "gemm self-test max diff %g" % worst
    return ""


def _init() -> dict:
    if os.environ.get("LUMINA_TOPS_DISABLE") == "1":
        return {"ok": False, "reason": "disabled by LUMINA_TOPS_DISABLE=1"}
    try:
        if not os.path.exists(SRC):
            return {"ok": False, "reason": "kernel source missing"}
        has, why = _cpu_has_dotprod()
        if not has:
            return {"ok": False, "reason": "cpu: " + why}
        src = open(SRC, "rb").read()
        path, err = _compile(src, "auto")
        if not path:
            return {"ok": False, "reason": "compile failed: " + err}
        bound = _bind(path)
        ste = _self_test(bound)
        if ste:
            return {"ok": False, "reason": ste}
        return {"ok": True, "path": path, "lib": bound["lib"],
                "reason": "built and self-tested"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "%s: %s" % (type(e).__name__, str(e)[:120])}


def state() -> dict:
    global _STATE
    if _STATE is None:
        with _LOCK:
            if _STATE is None:
                _STATE = _init()
    return _STATE


def available() -> bool:
    return bool(state().get("ok"))


def reason() -> str:
    s = state()
    return "available: " + str(s.get("reason")) if s.get("ok") else \
        "unavailable: " + str(s.get("reason"))


def info() -> dict:
    s = state()
    return {"available": bool(s.get("ok")), "reason": s.get("reason"),
            "path": s.get("path"), "version": VERSION,
            "source": os.path.relpath(SRC, os.path.dirname(HERE)),
            "build_dir": _build_dir(),
            "disabled_env": os.environ.get("LUMINA_TOPS_DISABLE")}


def _ptr(arr, ctype):
    return np.ascontiguousarray(arr).ctypes.data_as(ctypes.POINTER(ctype))


def gemv_int8(xq, xs, wq, ws, out=None):
    """out[n] = ws[n] * sum_b dot(xq_block, wq_row_block) * xs[b]."""
    if not available():
        return {"error": reason()}
    N, K = int(wq.shape[0]), int(wq.shape[1])
    xq = np.ascontiguousarray(xq, dtype=np.int8)
    wq = np.ascontiguousarray(wq, dtype=np.int8)
    xs = np.ascontiguousarray(xs, dtype=np.float32)
    ws = np.ascontiguousarray(ws, dtype=np.float32).reshape(-1)
    if out is None:
        out = np.zeros(N, dtype=np.float32)
    if xq.shape[0] < K:
        return {"error": "xq shorter than K (pad to a 32-block first)"}
    state()["lib"].tops_gemv_int8(_ptr(xq, ctypes.c_int8),
                                  _ptr(xs, ctypes.c_float),
                                  _ptr(wq, ctypes.c_int8),
                                  _ptr(ws, ctypes.c_float),
                                  _ptr(out, ctypes.c_float), N, K)
    return {"out": out}


def gemv_q4_0(xq, xs, wq, ws, out=None):
    """Q4_0 GEMV: wq is the packed payload (N, K//2) uint8; ws is (N, K//32)."""
    if not available():
        return {"error": reason()}
    K = int(wq.shape[1]) * 2
    N = int(wq.shape[0])
    xq = np.ascontiguousarray(xq, dtype=np.int8)
    wq = np.ascontiguousarray(wq, dtype=np.uint8)
    xs = np.ascontiguousarray(xs, dtype=np.float32)
    ws = np.ascontiguousarray(ws, dtype=np.float32)
    if out is None:
        out = np.zeros(N, dtype=np.float32)
    state()["lib"].tops_gemv_q4_0(_ptr(xq, ctypes.c_int8),
                                  _ptr(xs, ctypes.c_float),
                                  _ptr(wq, ctypes.c_uint8),
                                  _ptr(ws, ctypes.c_float),
                                  _ptr(out, ctypes.c_float), N, K)
    return {"out": out}


def set_threads(n: int) -> dict:
    """Thread count for the _mt kernels (default 8; clamped to >= 1)."""
    if not available():
        return {"error": reason()}
    state()["lib"].tops_set_threads(max(1, int(n)))
    return {"threads": max(1, int(n))}


def threads() -> int:
    if not available():
        return 1
    return int(state()["lib"].tops_get_threads())


def gemv_int8_mt(xq, xs, wq, ws, out=None):
    """Multithreaded (row-split) int8 dotprod GEMV."""
    if not available():
        return {"error": reason()}
    N, K = int(wq.shape[0]), int(wq.shape[1])
    xq = np.ascontiguousarray(xq, dtype=np.int8)
    wq = np.ascontiguousarray(wq, dtype=np.int8)
    xs = np.ascontiguousarray(xs, dtype=np.float32)
    ws = np.ascontiguousarray(ws, dtype=np.float32).reshape(-1)
    if out is None:
        out = np.zeros(N, dtype=np.float32)
    state()["lib"].tops_gemv_int8_mt(_ptr(xq, ctypes.c_int8),
                                     _ptr(xs, ctypes.c_float),
                                     _ptr(wq, ctypes.c_int8),
                                     _ptr(ws, ctypes.c_float),
                                     _ptr(out, ctypes.c_float), N, K)
    return {"out": out}


def gemv_q4_0_v2(xq, xs, wq, ws, out=None):
    """Q4_0 GEMV with the strided-dot unpack (see tops_int8.c)."""
    if not available():
        return {"error": reason()}
    K = int(wq.shape[1]) * 2
    N = int(wq.shape[0])
    xq = np.ascontiguousarray(xq, dtype=np.int8)
    wq = np.ascontiguousarray(wq, dtype=np.uint8)
    xs = np.ascontiguousarray(xs, dtype=np.float32)
    ws = np.ascontiguousarray(ws, dtype=np.float32)
    if out is None:
        out = np.zeros(N, dtype=np.float32)
    state()["lib"].tops_gemv_q4_0_v2(_ptr(xq, ctypes.c_int8),
                                     _ptr(xs, ctypes.c_float),
                                     _ptr(wq, ctypes.c_uint8),
                                     _ptr(ws, ctypes.c_float),
                                     _ptr(out, ctypes.c_float), N, K)
    return {"out": out}


def gemv_fp16(x, w, out=None):
    if not available():
        return {"error": reason()}
    N, K = int(w.shape[0]), int(w.shape[1])
    x = np.ascontiguousarray(x, dtype=np.float16)
    w = np.ascontiguousarray(w, dtype=np.float16)
    if out is None:
        out = np.zeros(N, dtype=np.float32)
    state()["lib"].tops_gemv_fp16(x.ctypes.data, w.ctypes.data,
                                  _ptr(out, ctypes.c_float), N, K)
    return {"out": out}


def gemm_int8(xq, xs, wq, ws, out=None):
    """out[m,n] = ws[n] * sum_b dot(xq[m]_block, wq[n]_block) * xs[m,b]."""
    if not available():
        return {"error": reason()}
    M, K = int(xq.shape[0]), int(xq.shape[1])
    N = int(wq.shape[0])
    xq = np.ascontiguousarray(xq, dtype=np.int8)
    wq = np.ascontiguousarray(wq, dtype=np.int8)
    xs = np.ascontiguousarray(xs, dtype=np.float32)
    ws = np.ascontiguousarray(ws, dtype=np.float32).reshape(-1)
    if out is None:
        out = np.zeros((M, N), dtype=np.float32)
    state()["lib"].tops_gemm_int8(_ptr(xq, ctypes.c_int8),
                                  _ptr(xs, ctypes.c_float),
                                  _ptr(wq, ctypes.c_int8),
                                  _ptr(ws, ctypes.c_float),
                                  _ptr(out, ctypes.c_float), M, N, K)
    return {"out": out}
