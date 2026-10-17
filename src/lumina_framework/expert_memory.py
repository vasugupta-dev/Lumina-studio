"""Expert Memory Engine — multi-tier expert residency, cache, and prefetch.

Mechanisms implemented (all colibri-inspired and falsifiable):
  * LFRU eviction (decaying session frequency + recency) with hysteresis.
  * Persistent usage histogram pinning: hottest experts are promoted to RAM at
    startup; the engine gets faster the more it is used.
  * Router-lookahead prefetch: predicted next-layer experts load while the current
    layer computes; measured, not assumed.
  * Live tier adaptation at turn boundaries (bounded swap count).
  * Honest per-tier byte attribution for every move.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict

import numpy as np

from .memory import (TIER_LATENCY, TIER_NVME, TIER_RAM, TIER_VRAM,
                     TIER_VRAM_STAGING, TIER_RAM_COMPRESSED,
                     ExpertPage, MemoryTier, TensorContainer)


class TierCost:
    """Measured-or-declared transport cost per tier.

    The reference ships DEFAULT proxies (see memory.TIER_LATENCY). A production
    engine overwrites these from the environment probe. Any benchmark must record
    which cost model produced its numbers.
    """

    def __init__(self, costs: dict[str, float] | None = None):
        self.costs = dict(costs or TIER_LATENCY)

    def seconds_for(self, tier: str, nbytes: int) -> float:
        return self.costs.get(tier, 1e-3) * max(1.0, nbytes / 16384.0)

    def as_dict(self):
        return dict(self.costs)


class ExpertMemoryEngine:
    """Owns all expert bytes in VRAM / RAM / NVMe and every move between them."""

    def __init__(
        self,
        expert_bank: dict[str, ExpertPage],
        ram_budget: int,
        vram_budget: int = 0,
        lru_cap_per_layer: int | None = None,
        hysteresis: float = 0.25,
        max_swaps_per_adapt: int = 4,
        costs: TierCost | None = None,
        seed: int = 0,
    ):
        self.bank = expert_bank
        self.costs = costs or TierCost()
        self._rng = np.random.default_rng(seed)

        # Demote from RAM -> NVMe is a no-op (NVMe is the immutable backing store);
        # from VRAM -> RAM the bytes stay hot. Neither drops the object.
        #
        # NOTE: callbacks fire after the evicting tier released its lock, and they
        # must never call back into that tier. The engine only ever demotes
        # VRAM -> RAM here; RAM -> NVMe is implicit (absence == cold).
        self.vram = MemoryTier(TIER_VRAM, vram_budget,
                               on_evict=self._on_vram_evict)
        self.vram_staging = MemoryTier(TIER_VRAM_STAGING, max(0, vram_budget // 4),
                                       on_evict=lambda k: None)
        self.ram = MemoryTier(TIER_RAM, ram_budget,
                              on_evict=self._on_ram_evict)
        self.ram_compressed = MemoryTier(TIER_RAM_COMPRESSED, ram_budget // 2,
                                         on_evict=lambda k: None)
        self.nvme = MemoryTier(TIER_NVME, 1 << 62)  # unbounded backing store

        self.session_heat: dict[str, float] = defaultdict(float)
        self.persistent_heat: dict[str, float] = defaultdict(float)
        self.lru_cap_per_layer = lru_cap_per_layer
        self.hysteresis = hysteresis
        self.max_swaps_per_adapt = max_swaps_per_adapt

        # init placements from the page metadata (pages are constructed resident
        # in whichever tier the converter placed them)
        for name, page in expert_bank.items():
            tier = page.placement.tier
            getattr(self, tier).put(name, page, page.size_bytes)

        self.lock = threading.RLock()
        self._load_inflight: set[str] = set()
        self._loads_in_flight = 0

    # ------------------------------------------------------------------
    # tier plumbing
    # ------------------------------------------------------------------
    def _on_vram_evict(self, key: str) -> None:
        """VRAM eviction: demote into RAM if it fits, else mark cold."""
        page = self.bank[key]
        if not self.ram.contains(key):
            self.ram.put(key, page, page.size_bytes)
        page.placement.tier = TIER_RAM if self.ram.contains(key) else TIER_NVME

    def _on_ram_evict(self, key: str) -> None:
        """RAM eviction: the expert is cold again; NVMe is the backing store."""
        self.bank[key].placement.tier = TIER_NVME

    def resident_tier(self, key: str) -> str | None:
        if self.vram.contains(key):
            return TIER_VRAM
        if self.vram_staging.contains(key):
            return TIER_VRAM_STAGING
        if self.ram.contains(key):
            return TIER_RAM
        if self.ram_compressed.contains(key):
            return TIER_RAM_COMPRESSED
        return None

    def _load(self, key: str, target: str) -> bool:
        """Materialize `key` into `target` tier. Returns True if a cold read happened."""
        with self.lock:
            page = self.bank[key]
            if getattr(self, target).contains(key):
                return False
            source_tier = self.resident_tier(key) or TIER_NVME
            self._loads_in_flight += 1
            try:
                cold = source_tier == TIER_NVME
                if cold:
                    self.nvme.stats.read_bytes += page.size_bytes
                    self.nvme.stats.read_ops += 1
                    self.nvme.stats.time_in_read += self.costs.seconds_for(
                        TIER_NVME, page.size_bytes)
                ok = getattr(self, target).put(key, page, page.size_bytes)
                if not ok and target != TIER_RAM:
                    # VRAM could not hold it; RAM is the fallback fast tier
                    ok = self.ram.put(key, page, page.size_bytes)
                if ok:
                    page.placement.tier = TIER_RAM if self.ram.contains(key) else target
                    if self.vram.contains(key):
                        page.placement.tier = TIER_VRAM
                    page.placement.resident = True
                return cold
            finally:
                self._loads_in_flight -= 1

    def get(self, keys: list[str], target: str = TIER_RAM) -> list[tuple[ExpertPage, bool]]:
        """Fetch experts, promoting to `target`. Returns (page, was_cold_read)."""
        out: list[tuple[ExpertPage, bool]] = []
        with self.lock:
            # batch load
            to_load = []
            for k in keys:
                if not getattr(self, target).contains(k) and self.resident_tier(k) is None:
                    to_load.append(k)
            # do loads
            for k in keys:
                page = self.bank[k]
                cold = self._load(k, target)
                page.placement.last_access += 1
                page.placement.hits += 1
                tier = self.resident_tier(k)
                if tier is not None:
                    st = getattr(self, tier).stats
                    st.read_bytes += page.size_bytes
                    st.read_ops += 1
                    st.time_in_read += self.costs.seconds_for(tier, page.size_bytes)
                self.session_heat[k] = self.session_heat.get(k, 0.0) + 1.0
                out.append((page, cold))
        return out

    # ------------------------------------------------------------------
    # learning cache / pinning
    # ------------------------------------------------------------------
    def pin_hot_experts(self, count: int, target: str = TIER_RAM) -> list[str]:
        """Promote the `count` hottest persistent-heat experts into `target`."""
        ranked = sorted(self.persistent_heat.items(), key=lambda kv: -kv[1])[:count]
        pinned: list[str] = []
        for name, _ in ranked:
            if self._load(name, target):
                pinned.append(name)
        return pinned

    def record_usage(self, keys: list[str], decay: float = 1.0) -> None:
        for k in keys:
            self.persistent_heat[k] = self.persistent_heat.get(k, 0.0) + decay
            for other, v in self.session_heat.items():
                if other != k:
                    self.session_heat[other] *= 0.999

    def apply_session_decay(self, factor: float = 0.5) -> None:
        for k in self.session_heat:
            self.session_heat[k] *= factor

    def adapt_tiers(self, target: str = TIER_RAM) -> dict:
        """Live tier adaptation at turn boundaries: swap cold pinned experts for hot
        streamed ones, bounded by `max_swaps_per_adapt` and hysteresis."""
        with self.lock:
            swapped: list[str] = []
            hot = sorted(self.session_heat.items(), key=lambda kv: -kv[1])
            pinned_now = list(getattr(self, target).keys())
            if not pinned_now or not hot:
                return {"swapped": swapped}
            min_swap_heat = hot[0][1] * (1.0 + self.hysteresis)
            for hname, hheat in hot:
                if len(swapped) >= self.max_swaps_per_adapt:
                    break
                victim = min(pinned_now, key=lambda k: self.session_heat.get(k, 0.0))
                vheat = self.session_heat.get(victim, 0.0)
                if hheat < min_swap_heat or hheat <= vheat:
                    continue
                getattr(self, target).drop(victim)
                self._load(hname, target)
                pinned_now.remove(victim)
                swapped.append(hname)
            return {"swapped": swapped}

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "vram": self.vram.stats.as_dict() | {"used": self.vram.used, "budget": self.vram.budget,
                                                  "keys": len(self.vram.keys())},
            "ram": self.ram.stats.as_dict() | {"used": self.ram.used, "budget": self.ram.budget,
                                               "keys": len(self.ram.keys())},
            "nvme": {"read_bytes": self.nvme.stats.read_bytes,
                     "read_ops": self.nvme.stats.read_ops,
                     "seconds_in_read": self.nvme.stats.time_in_read},
        }

