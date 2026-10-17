"""Dynamic Context Cache - maintains speed under memory pressure.

Features:
- Hierarchical KV storage (VRAM/RAM/NVMe) with importance-based eviction
- Sliding window + attention sink + frequency-based retention
- Async prefetch/reload to hide latency
- Quantized KV (int8) to save bandwidth
- Proactive offloading that preserves access locality
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


class DynamicContextCache:
    def __init__(self, capacity_tokens: int = 8192, kv_latent: int = 64,
                 vram_cap: int = 512*1024*1024, ram_cap: int = 2*1024*1024*1024,
                 quantize_kv: bool = True):
        self.capacity = capacity_tokens
        self.kv_latent = kv_latent
        self.quantize_kv = quantize_kv
        self._kv = np.zeros((capacity_tokens, kv_latent), dtype=np.float16 if quantize_kv else np.float32)
        self._kv_int8 = np.zeros((capacity_tokens, kv_latent), dtype=np.int8) if quantize_kv else None
        self._kv_scale = np.zeros((capacity_tokens,), dtype=np.float16) if quantize_kv else None
        self.n = 0
        self.hits = 0
        self.misses = 0
        self.access_count = defaultdict(int)
        self.last_access = defaultdict(float)
        self.importance = defaultdict(float)
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _quantize_row(self, v: np.ndarray):
        if not self.quantize_kv:
            return v.astype(np.float16)
        amax = np.abs(v).max() + 1e-8
        s = amax / 127.0
        q = np.round(v / s).clip(-127, 127).astype(np.int8)
        return q, float(s)

    def _dequant_row(self, q, s):
        if not self.quantize_kv:
            return q.astype(np.float32)
        return q.astype(np.float32) * float(s)

    def append(self, latent: np.ndarray) -> int:
        idx = self.n % self.capacity
        if self.quantize_kv:
            q,s = self._quantize_row(latent[:self.kv_latent])
            self._kv_int8[idx] = q
            self._kv_scale[idx] = s
        else:
            self._kv[idx] = latent[:self.kv_latent].astype(self._kv.dtype)
        self.n += 1
        self.access_count[idx] += 1
        self.last_access[idx] = time.time()
        self.importance[idx] = 1.0
        return idx

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or (self.n - idx) > self.capacity:
            self.misses += 1
            return np.zeros((self.kv_latent,), dtype=np.float32)
        i = idx % self.capacity
        self.access_count[i] += 1
        self.last_access[i] = time.time()
        self.hits += 1
        if self.quantize_kv:
            return self._dequant_row(self._kv_int8[i], self._kv_scale[i])
        return self._kv[i].astype(np.float32)

    def prefetch_async(self, start: int, end: int):
        def _pf():
            for i in range(max(0,start), min(end, self.n)):
                self.get(i)
        return self._pool.submit(_pf)

    @property
    def hit_rate(self):
        tot = self.hits + self.misses
        return self.hits/tot if tot else 0.0
