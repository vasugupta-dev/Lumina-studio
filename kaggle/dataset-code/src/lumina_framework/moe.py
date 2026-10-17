"""MoE layer with streaming expert dispatch, batch-union reads, and lookahead.

Layer semantics follow the colibri / GLM/DeepSeek-style reference:
  sigmoid router (no aux-loss routing correction) with routed_scaling_factor,
  shared expert always active, routed experts streamed from the Expert Memory
  Engine, first-N dense layers optional.

Correctness invariants the reference tests assert:
  I1. Setting every expert resident and disabling speculation must reproduce
      dense FP16 forward within documented tolerance.
  I2. Batch-union dispatch must visit each unique expert exactly once per batch
      and produce the same per-expert output as per-token dispatch.
  I3. Prefetch must never change results; it only moves bytes earlier.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import numpy as np
try:
    import numba
    _HAS_NUMBA = True
except Exception:
    numba = None
    _HAS_NUMBA = False

from .expert_memory import ExpertMemoryEngine
from .memory import TIER_RAM, TIER_VRAM, ExpertPage, TensorContainer
from . import kernels


# Numba router scores
if _HAS_NUMBA:
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

class Router:
    """Sigmoid router without auxiliary loss (DeepSeek/GLM style)."""

    def __init__(self, weight: np.ndarray, scaling: float = 1.0):
        self.weight = weight  # (n_expert, d_model)
        self.scaling = scaling

    def scores(self, x: np.ndarray) -> np.ndarray:
        x32 = x.astype(np.float32, copy=False)
        wT = self.weight.T
        if _HAS_NUMBA:
            z = x32 @ wT
            return 1.0 / (1.0 + np.exp(-z))
        # fallback: use numba for large matrices
        if x32.shape[0] > 1 and _HAS_NUMBA:
            return _router_scores_numba(x32, self.weight)
        z = x32 @ wT
        return 1.0 / (1.0 + np.exp(-z))

    def route(self, x: np.ndarray, top_k: int, top_p: float | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (expert_index_per_token, combine_weight_per_token)."""
        s = self.scores(x) * self.scaling
        if top_p is not None:
            # adaptive top-k: smallest set of experts covering `top_p` routing mass
            order = np.argsort(-s, axis=1)
            cum = np.cumsum(np.take_along_axis(s, order, 1), axis=1)
            keep = cum <= top_p
            keep[:, 0] = True
            idx, w = [], []
            for i in range(s.shape[0]):
                mask = keep[i]
                e = order[i][mask]
                idx.append(e)
                w.append(s[i][e] / s[i][e].sum())
            n = max(len(e) for e in idx)
            idx_a = np.zeros((s.shape[0], n), dtype=np.int64)
            w_a = np.zeros((s.shape[0], n), dtype=np.float32)
            for i, (e, ww) in enumerate(zip(idx, w)):
                idx_a[i, :len(e)] = e
                w_a[i, :len(e)] = ww
            return idx_a, w_a
        k = min(top_k, self.weight.shape[0])
        idx = np.argsort(-s, axis=1)[:, :k]
        w = np.take_along_axis(s, idx, 1)
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-9, None)
        return idx, w


class RoutePredictor:
    """Router-lookahead predictor.

    colibri measured 71.6% recall for "apply layer L+1's router to layer L's
    post-attention state" vs 41.3% for "same experts as last token". This class
    implements that prediction rule and reports its own recall so it can be
    validated rather than assumed.

    `use```variance_cutoff``` additionally drops predictions below the router's
    confidence horizon; long-horizon guesses are advisory.
    """

    def __init__(self, confidence: float = 0.0):
        self.confidence = confidence
        self.tp = 0
        self.fn = 0
        self.n_correct_tokens = 0
        self.n_tokens = 0

    def predict_next(self, router: Router, hidden_post_attention: np.ndarray,
                     top_k: int) -> np.ndarray:
        pred, _ = router.route(hidden_post_attention, top_k)
        return pred

    def update_recall(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        for p_row, a_row in zip(predicted, actual):
            self.n_tokens += 1
            ap = set(p_row.tolist())
            aa = set(a_row.tolist())
            self.tp += len(ap & aa)
            self.fn += len(aa - ap)
            self.n_correct_tokens += int(len(ap & aa) == len(aa))

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def token_accuracy(self) -> float:
        return self.n_correct_tokens / self.n_tokens if self.n_tokens else 0.0


class DenseLayer:
    """Dense (non-MoE) transformer layer with MLP."""

    def __init__(self, layer_id: int, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray):
        self.layer_id = layer_id
        self.gate = gate_w
        self.up = up_w
        self.down = down_w

    def forward(self, x: np.ndarray, engine=None, target: str = "ram") -> np.ndarray:
        """SwiGLU forward."""
        g = x @ self.gate.T
        u = x @ self.up.T
        h = g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0))) * u
        return h @ self.down.T

class Prefetcher:
    """Background prefetch with async execution."""

    def __init__(self, engine: ExpertMemoryEngine, threads: int = 12):
        try:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=threads)
            self._has_pool = True
        except Exception:
            self._pool = None
            self._has_pool = False
        self.engine = engine
        self.threads = threads
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self.n_issued = 0
        self.bytes_moved = 0
        self._futures = []

    def submit(self, keys: list[str], target: str = TIER_RAM) -> None:
        batch = []
        with self._lock:
            for k in keys:
                rt = self.engine.resident_tier(k)
                if rt is None and k not in self._inflight:
                    self._inflight.add(k)
                    batch.append(k)
        if not batch:
            return
        if self._has_pool:
            fut = self._pool.submit(self._run, batch, target)
            self._futures.append(fut)
        else:
            self._run(batch, target)

    def _run(self, batch: list[str], target: str) -> None:
        pages = self.engine.get(batch, target=target)
        for p, cold in pages:
            if cold:
                self.bytes_moved += p.size_bytes
        with self._lock:
            for k in batch:
                self._inflight.discard(k)
            self.n_issued += 1

    def flush(self, target: str = TIER_RAM) -> dict:
        t0 = time.perf_counter()
        if self._has_pool:
            try:
                from concurrent.futures import wait
                wait(self._futures)
                self._futures.clear()
            except Exception:
                pass
        return {"seconds": time.perf_counter() - t0, "bytes_prefetched": self.bytes_moved}


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class MoELayer:
    """One routed+shared expert layer."""

    def __init__(self, layer_id: int, n_expert: int, d_model: int, d_ff: int,
                 top_k: int = 2, router_scaling: float = 1.0, seed: int = 0):
        self.layer_id = layer_id
        self.n_expert = n_expert
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        rng = np.random.default_rng(seed)
        self.router = Router(rng.standard_normal((n_expert, d_model)).astype(np.float32) * 0.05,
                             scaling=router_scaling)
        self.shared = ExpertPage(
            name=f"L{layer_id}-shared",
            containers={
                "gate": TensorContainer(rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.05),
                "up": TensorContainer(rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.05),
                "down": TensorContainer(rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.05),
            },
            layer=layer_id, arch="shared")
        self.experts = [
            ExpertPage(
                name=f"L{layer_id}-e{i}",
                containers={
                    "gate": TensorContainer(rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.05),
                    "up": TensorContainer(rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.05),
                    "down": TensorContainer(rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.05),
                },
                layer=layer_id, arch="swiglu")
            for i in range(n_expert)
        ]

    def forward(self, x: np.ndarray, engine: ExpertMemoryEngine,
                top_p: float | None = None, target: str = TIER_RAM,
                return_routing: bool = False):
        idx, w = self.router.route(x, self.top_k, top_p=top_p)
        # fast unique
        used_set = set(idx.ravel())
        used = tuple(sorted(used_set))
        pages = [self.experts[i] for i in used]
        # batch get
        eng_get = engine.get
        eng_get([p.name for p in pages], target=target)
        # shared
        sh = self.shared.forward(x)
        out = sh.copy()
        # group
        rows_by_expert = {}
        xshape0 = x.shape[0]
        idxshape1 = idx.shape[1]
        for r in range(xshape0):
            idxr = idx[r]
            for j in range(idxshape1):
                e = int(idxr[j])
                if e < 0: continue
                if e not in rows_by_expert:
                    rows_by_expert[e] = []
                rows_by_expert[e].append((r,j))
        for e, rows_list in rows_by_expert.items():
            try:
                page = pages[used.index(e)]
            except Exception:
                page = self.experts[e]
            # fast gather
            xr_rows = [rr[0] for rr in rows_list]
            xr = x[xr_rows]
            y = page.forward(xr)
            w_mat = w
            out_mat = out
            for k,(r,j) in enumerate(rows_list):
                out_mat[r] += w_mat[r,j] * y[k]
        if return_routing:
            return out, idx, w
        return out
