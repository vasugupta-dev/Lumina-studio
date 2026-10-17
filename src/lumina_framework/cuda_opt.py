"""Compute inventory for the Lumina runtime.

What this module is:
  * hardware_concurrency() — every compute resource we can see: CPU cores,
    RAM, torch/CUDA, BLAS threading. No guessing, all measured or queried.
  * recommend_threads() — one honest thread policy (BLAS + pools).

What this module is NOT: hand-written Triton kernels, or torch compute ops
without a caller. The previous revision shipped Triton kernels with dummy
loads and `pass` loop bodies that computed nothing — decorative code that
could never run correctly. Triton and torch compute helpers return only
when a caller exists and a numeric test ships.
"""
from __future__ import annotations

import os

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


def cuda_available() -> bool:
    return bool(_HAS_TORCH and torch.cuda.is_available())


def get_device_props() -> dict:
    if not cuda_available():
        return {}
    d = torch.cuda.get_device_properties(0)
    return {"name": d.name, "sm": d.major * 10 + d.minor,
            "mem": d.total_memory,
            "multiprocessors": d.multi_processor_count}


def _mem_available_mb() -> int | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def hardware_concurrency() -> dict:
    """Every compute resource visible to this process (measured, not claimed)."""
    import threading
    out = {
        "cpu_count": os.cpu_count() or 1,
        "threading_active": threading.active_count(),
        "mem_available_mb": _mem_available_mb(),
        "torch": None,
        "cuda": None,
        "blas": None,
    }
    if _HAS_TORCH:
        out["torch"] = {"version": torch.__version__,
                        "threads": torch.get_num_threads(),
                        "cuda": cuda_available()}
        if cuda_available():
            out["cuda"] = get_device_props()
    try:
        import threadpoolctl

        pools = [{"lib": p["filepath"].split("/")[-1],
                  "threads": p["num_threads"]}
                 for p in threadpoolctl.threadpool_info()]
        out["blas"] = pools
    except Exception:
        out["blas"] = None
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(k):
            out.setdefault("env", {})[k] = os.environ[k]
    return out


def recommend_threads() -> dict:
    """One thread policy: BLAS gets the cores; our pools stay small so they
    never oversubscribe the BLAS workers doing the real FLOPs."""
    n = os.cpu_count() or 1
    return {"blas_threads": max(1, n),
            "pool_workers": max(1, min(4, n // 2)),
            "prefetch_workers": 1}


def apply_thread_policy(policy: dict | None = None) -> dict:
    """Set thread counts (BLAS env must be set before numpy imports to bite;
    torch can be set any time). Returns what was applied and what was late."""
    p = policy or recommend_threads()
    late = False
    try:
        import numpy  # noqa: F401 — if already imported, env is advisory only
        late = True
    except Exception:
        pass
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(p["blas_threads"]))
    if _HAS_TORCH:
        try:
            torch.set_num_threads(p["blas_threads"])
        except Exception:
            pass
    out = dict(p)
    out["blas_env_late"] = late
    return out


# ---------------------------------------------------------------------------
# Tested fused ops (torch, device-agnostic: CPU here, CUDA unchanged)
# ---------------------------------------------------------------------------


def torch_swiglu(x, gate_w, up_w, down_w):
    """Full SwiGLU MLP block: silu(x@G) * (x@U) @ D. All three projections —
    the deleted Triton kernel never applied the down projection."""
    if not _HAS_TORCH:
        raise RuntimeError("torch not installed")
    g = x.float() @ gate_w.float().T
    u = x.float() @ up_w.float().T
    return (torch.nn.functional.silu(g) * u) @ down_w.float().T

