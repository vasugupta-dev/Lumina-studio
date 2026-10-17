#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from lumina_framework.runtime import RuntimeFabric, StubModel
from lumina_framework.expert_memory import ExpertMemoryEngine
from lumina_framework.memory import TIER_RAM

m = StubModel(n_layer=3, n_expert=8, d_model=128, d_ff=256, top_k=2, seed=7)
bank = {}
for l in m.layers:
    for e in l.experts: bank[e.name] = e
    bank[l.shared.name] = l.shared
eng = ExpertMemoryEngine(bank, ram_budget=64*1024*1024, vram_budget=0)  # tight
eng.pin_hot_experts(2, target=TIER_RAM)  # minimal pinning = heavy offload
rt = RuntimeFabric(m, eng, prefetch=True)
prompt = np.array([1,2,3,4])
t0 = time.time()
res = rt.decode(prompt, 100, target=TIER_RAM)
t1 = time.time()
print(f"Heavy offload: {res['tokens_per_second']:.2f} tok/s in {t1-t0:.2f}s (ram_budget=64MB, pinned=2)")
