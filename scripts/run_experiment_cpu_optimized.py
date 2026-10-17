#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from lumina_framework.runtime import RuntimeFabric, StubModel
from lumina_framework.expert_memory import ExpertMemoryEngine, TierCost
from lumina_framework.memory import TIER_RAM, TIER_NVME

# Smaller model, more aggressive pinning for CPU
m = StubModel(n_layer=2, n_expert=6, d_model=128, d_ff=256, top_k=1, seed=42)
bank = {}
for l in m.layers:
    for e in l.experts:
        bank[e.name] = e
    bank[l.shared.name] = l.shared
eng = ExpertMemoryEngine(bank, ram_budget=512*1024*1024, vram_budget=0)
# pin everything
eng.pin_hot_experts(20, target=TIER_RAM)
rt = RuntimeFabric(m, eng, prefetch=False)  # less overhead
prompt = np.array([1,2,3,4])
t0 = time.time()
res = rt.decode(prompt, 200, target=TIER_RAM)
t1 = time.time()
print(f"CPU (optimized): {res['tokens_per_second']:.2f} tok/s in {t1-t0:.2f}s")
