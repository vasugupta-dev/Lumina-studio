"""Lumina Framework v0.1 — Controlled, integer-executable MoE streaming runtime reference.

Design goals (from Lumina_Studio/plan.md):
  * VRAM hot / RAM warm / NVMe cold as ONE managed hierarchy (colibri-inspired).
  * Streaming MoE experts from disk when they don't fit; correct regardless of placement.
  * Honest instrumentation: every byte moved is attributed to a tier.
  * Cache hit rate alone is never a pass criterion — E2E latency must improve.

This module implements:
  * MemoryTier: a byte-budgeted store with LRU + LFRU eviction and honest accounting.
  * ExpertMemoryEngine: multi-tier placement, usage histogram pinning, live tier
    adaptation, and per-token movement accounting.
  * Quantized tensor containers (Q8_0-style int8 activations / int4 weights) so the
    reference stays in the low-bit regime the Format track is designed for.
  * Default behavior is LOSSLESS with respect to the model's declared precision:
    tiers move bytes; they never silently change router semantics or precision.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np
try:
    import numba
    _HAS_NUMBA = True
except Exception:
    numba = None
    _HAS_NUMBA = False

# ----------------------------------------------------------------------------
# Tier accounting
# ----------------------------------------------------------------------------

TIER_VRAM = "vram"
TIER_VRAM_STAGING = "vram_staging"
TIER_RAM = "ram"
TIER_RAM_COMPRESSED = "ram_compressed"
TIER_NVME = "nvme"

TIER_LATENCY = {
    TIER_VRAM: 5e-6,
    TIER_VRAM_STAGING: 1e-5,
    TIER_RAM: 2e-4,
    TIER_RAM_COMPRESSED: 5e-4,
    TIER_NVME: 4e-4,
}


class TierStats:
    """Immutable-ish counter block for one memory tier."""

    __slots__ = ("read_bytes", "read_ops", "write_bytes", "time_in_read", "evictions")

    def __init__(self) -> None:
        self.read_bytes = 0
        self.read_ops = 0
        self.write_bytes = 0
        self.time_in_read = 0.0
        self.evictions = 0

    def as_dict(self) -> dict:
        return {
            "read_bytes": self.read_bytes,
            "read_ops": self.read_ops,
            "write_bytes": self.write_bytes,
            "seconds_in_read": self.time_in_read,
            "evictions": self.evictions,
        }


@dataclass
class Placement:
    """Where an expert block currently lives."""

    tier: str
    resident: bool
    last_access: float = 0.0
    hits: int = 0


class MemoryTier:
    """A byte-budgeted cache tier with LRU eviction.

    `on_evict` is invoked AFTER the tier lock is released and receives the evicted
    key. Callbacks must never re-enter the evicting tier (it is mid-eviction);
    demotion logic belongs to the Expert Memory Engine.
    """

    def __init__(self, name: str, budget_bytes: int, on_evict=None):
        self.name = name
        self.budget = int(budget_bytes)
        self.used = 0
        self.store: OrderedDict[str, object] = OrderedDict()
        self.sizes: dict[str, int] = {}
        self.stats = TierStats()
        self._on_evict = on_evict
        self._lock = threading.RLock()

    # -- hot API ----------------------------------------------------------
    def contains(self, key: str) -> bool:
        return key in self.store

    def get(self, key: str):
        with self._lock:
            if key not in self.store:
                return None
            self.store.move_to_end(key)
            return self.store[key]

    def put(self, key: str, value: object, size_bytes: int) -> bool:
        """Insert; returns False if the object could not fit even when empty."""
        size_bytes = int(size_bytes)
        if size_bytes > self.budget:
            return False
        evicted: list[str] = []
        with self._lock:
            if key in self.store:
                self.used -= self.sizes[key]
            while self.used + size_bytes > self.budget and self.store:
                old_key, _ = self.store.popitem(last=False)
                old_size = self.sizes.pop(old_key)
                self.used -= old_size
                self.stats.evictions += 1
                evicted.append(old_key)
            self.store[key] = value
            self.sizes[key] = size_bytes
            self.used += size_bytes
            self.stats.write_bytes += size_bytes
        for k in evicted:
            if self._on_evict is not None:
                self._on_evict(k)
        return True

    def drop(self, key: str) -> bool:
        with self._lock:
            if key not in self.store:
                return False
            self.store.pop(key)
            self.used -= self.sizes.pop(key)
            return True

    def keys(self):
        return list(self.store.keys())

    def utilization(self) -> float:
        return self.used / self.budget if self.budget else 0.0


# ----------------------------------------------------------------------------
# Quantized tensor containers (reference containers; real kernels are Track R work)
# ----------------------------------------------------------------------------

def _quantize_int8_per_row(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise symmetric int8. x is (out, in) or flattened rows."""
    x = np.atleast_2d(x)
    amax = np.abs(x).max(axis=1, keepdims=True).clip(1e-12)
    q = np.round(x * (127.0 / amax)).clip(-127, 127).astype(np.int8)
    return q, (amax / 127.0).astype(np.float32)


# Numba JIT optimized hot paths (module level)
if _HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _silu_numba(x):
        return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    @numba.jit(nopython=True, cache=True)
    def _matmul_numba(x, w):
        """x: (M, K), w: (N, K) -> out: (M, N)"""
        M, K = x.shape
        N = w.shape[0]
        out = np.zeros((M, N), dtype=np.float32)
        if M == 1:
            # Single token - optimize
            xi = x[0]
            for j in range(N):
                wj = w[j]
                s = 0.0
                for k in range(K):
                    s += xi[k] * wj[k]
                out[0, j] = s
            return out
        # Multiple tokens - use numpy's optimized matmul
        return x @ w.T

    @numba.jit(nopython=True, cache=True)
    def _router_scores_numba(x, w):
        """x: (M, D), w: (E, D) -> scores: (M, E)"""
        M, D = x.shape
        E = w.shape[0]
        scores = np.zeros((M, E), dtype=np.float32)
        for i in range(M):
            xi = x[i]
            for e in range(E):
                we = w[e]
                s = 0.0
                for k in range(D):
                    s += xi[k] * we[k]
                scores[i, e] = 1.0 / (1.0 + np.exp(-s))
        return scores

class TensorContainer:
    """Quantized weight container with honest byte accounting.

    Layout: int8 packed payload + per-row int8 exponent (E4M3-style scale code).
    The dequantized value is exactly (int8 * scale); scales are stored as float32
    but *charged* at 8 bits in byte accounting, so accounting matches the wire
    format the Kernel track will consume.
    """

    def __init__(self, data: np.ndarray, dtype_code: int = 8, group_bits: int = 8):
        self.shape = tuple(data.shape)
        self.flat = np.ascontiguousarray(data.reshape(-1))
        rows = data.shape[0] if data.ndim > 1 else 1
        self.rows = rows
        self.n = self.flat.size
        self._q8, self._scale = _quantize_int8_per_row(
            self.flat.reshape(rows, -1) if data.ndim > 1 else self.flat[None, :]
        )
        # packed nibble-free payload
        self.payload = self._q8.tobytes()
        self.dtype_code = dtype_code
        self.group_bits = group_bits
        self.shape_matrix = data.shape
        self._dequantized = None
        self._dequantized_T = None

    # -- accounting --------------------------------------------------------
    @property
    def weight_bytes(self) -> int:
        return len(self.payload)

    @property
    def scale_bytes(self) -> int:
        # charged at 8 bits per row group regardless of the float32 working copy
        return int(math.ceil(self.rows * self.group_bits / 8))

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes

    def effective_bpp(self) -> float:
        return 8.0 * self.total_bytes / self.n

    # -- execution ----------------------------------------------------------
    def dequantize(self) -> np.ndarray:
        if self._dequantized is not None:
            return self._dequantized
        qf = self._q8.astype(np.float32, copy=False)
        sf = self._scale
        out = qf * sf[:, None] if sf.ndim == 1 and qf.ndim > 1 else qf * sf.reshape(-1,1)
        self._dequantized = out.reshape(self.shape_matrix)
        return self._dequantized

    def dequantize_T(self) -> np.ndarray:
        """Return transposed dequantized weights for fast x @ W (no .T needed)."""
        if self._dequantized_T is not None:
            return self._dequantized_T
        w = self.dequantize()
        self._dequantized_T = w.T.copy()
        return self._dequantized_T


class ExpertPage:
    """One MoE expert (or any named block) with multi-tier placement metadata.

    The payload is a dict of tensor containers so a single expert can hold gate /
    up / down projections. Moves between tiers copy the *quantized* payload only;
    dequantization happens at operator time, never as a tiering default.
    """

    __slots__ = ("name", "containers", "size_bytes", "placement", "layer", "arch")

    def __init__(self, name: str, containers: dict[str, TensorContainer],
                 layer: int, arch: str = "swiglu"):
        self.name = name
        self.containers = containers
        self.size_bytes = sum(c.total_bytes for c in containers.values())
        self.placement = Placement(tier=TIER_NVME, resident=False)
        self.layer = layer
        self.arch = arch

    def forward(self, x: np.ndarray, fused: bool = True) -> np.ndarray:
        dt = np.float32
        keys = self.containers
        # find keys once
        gate = None; up = None; down = None
        for k in keys:
            kl = k.lower()
            if gate is None and ('gate' in kl or 'w1' in kl or 'fc_in' in kl): gate = k
            if up is None and ('up' in kl or 'w3' in kl or 'fc_gate' in kl): up = k
            if down is None and ('down' in kl or 'w2' in kl or 'out' in kl or 'fc_out' in kl): down = k
        def _matmul(xv, c):
            # Decode regime + bandwidth-bound width: repack-once integer dot
            # (llama.cpp path). Below K=512 BLAS wins; measured crossover.
            if xv.shape[0] <= 4 and c._q8.shape[1] >= 512:
                try:
                    from . import fused as _fz
                    return _fz.fused_gemv_int8(xv, c)
                except Exception:
                    pass
            c_deq = c.dequantize()
            return xv.astype(dt) @ c_deq.T
        if gate is not None and up is not None and down is not None:
            g = _matmul(x, self.containers[gate])
            u = _matmul(x, self.containers[up])
            h = (g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))) * u
            return _matmul(h, self.containers[down])
        if gate is not None and down is not None and up is None:
            g = _matmul(x, self.containers[gate]); g = np.tanh(g)
            return _matmul(g, self.containers[down])
        w = next(iter(self.containers.values()))
        return _matmul(x, w)

    def _find(self, keys, candidates):
        for c in candidates:
            for k in keys:
                if k == c or k.endswith("." + c) or k.endswith("_" + c) or f".{c}." in k:
                    return k
        return None
