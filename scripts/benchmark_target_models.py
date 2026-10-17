#!/usr/bin/env python3
"""Benchmark target model configurations at 4-bit quantization."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from lumina_framework.runtime import RuntimeFabric, StubModel
from lumina_framework.expert_memory import ExpertMemoryEngine
from lumina_framework.memory import TIER_RAM
from lumina_framework.target_models import get_spec, ModelSpec

def build_model_from_spec(spec: ModelSpec, seed: int = 42) -> StubModel:
    """Build a StubModel matching the target spec (simplified)."""
    return StubModel(
        n_layer=spec.n_layer,
        n_expert=spec.n_expert,
        d_model=spec.d_model,
        d_ff=spec.d_ff,
        top_k=spec.top_k,
        seed=seed
    )

def benchmark(spec: ModelSpec, ram_mb: int, pin_n: int, tokens: int = 200, prefetch: bool = True) -> float:
    m = build_model_from_spec(spec)
    bank = {}
    for l in m.layers:
        for e in l.experts: bank[e.name] = e
        bank[l.shared.name] = l.shared
    eng = ExpertMemoryEngine(bank, ram_budget=ram_mb*1024*1024, vram_budget=0)
    eng.pin_hot_experts(pin_n, target=TIER_RAM)
    rt = RuntimeFabric(m, eng, prefetch=prefetch)
    prompt = np.array([1,2,3,4,5])
    t0 = time.time()
    res = rt.decode(prompt, tokens, target=TIER_RAM)
    t1 = time.time()
    return res['tokens_per_second']

def main():
    print("=" * 70)
    print("Target Model Benchmarks at 4-bit (CPU-only)")
    print("=" * 70)
    
    # Qwen 3.8 Flash Next - moderate offload
    print("\nQwen 3.8 Flash Next:")
    for ram, pin in [(256, 32), (512, 64), (1024, 128)]:
        tps = benchmark(get_spec("qwen3.8-flash-next"), ram, pin, tokens=200)
        print(f"  RAM={ram}MB, pinned={pin}: {tps:.1f} tok/s")
    
    # GLM 5.3 Flash - moderate offload
    print("\nGLM 5.3 Flash:")
    for ram, pin in [(256, 32), (512, 64), (1024, 128)]:
        tps = benchmark(get_spec("glm-5.3-flash"), ram, pin, tokens=200)
        print(f"  RAM={ram}MB, pinned={pin}: {tps:.1f} tok/s")
    
    # GLM 5.3 - heavy offload
    print("\nGLM 5.3:")
    for ram, pin in [(128, 16), (256, 32), (512, 64)]:
        tps = benchmark(get_spec("glm-5.3"), ram, pin, tokens=200)
        print(f"  RAM={ram}MB, pinned={pin}: {tps:.1f} tok/s")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
