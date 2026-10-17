#!/usr/bin/env python3
"""Run experiment model on CPU-only with aggressive offloading."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from lumina_framework.runtime import RuntimeFabric, StubModel
from lumina_framework.expert_memory import ExpertMemoryEngine, TierCost
from lumina_framework.memory import TIER_RAM, TIER_NVME

# CPU-optimized config
m = StubModel(n_layer=6, n_expert=12, d_model=256, d_ff=512, top_k=2, seed=42)
# build bank
bank = {}
for l in m.layers:
    for e in l.experts:
        bank[e.name] = e
    bank[l.shared.name] = l.shared

# CPU-only: pin few experts, rely on streaming
eng = ExpertMemoryEngine(bank, ram_budget=256*1024*1024, vram_budget=0)
# pin hottest subset only
eng.pin_hot_experts(6, target=TIER_RAM)
rt = RuntimeFabric(m, eng, prefetch=True)

prompt = np.array([10, 20, 30, 40, 50])
t0 = time.time()
res = rt.decode(prompt, 100, target=TIER_RAM)
t1 = time.time()
print(f"CPU decode: {res['tokens_per_second']:.2f} tok/s in {t1-t0:.2f}s")
