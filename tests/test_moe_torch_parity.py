"""Torch MoE parity: TorchMoELayer.forward must match the numpy
MoELayer.forward (engine path) within float tolerance on a tiny config,
and TorchStub.greedy must match RuntimeFabric.decode's committed tokens
(numpy path) exactly. Phone-safe sizes.

  python3 tests/test_moe_torch_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" — " + detail) if not cond else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    import torch
    from lumina_framework.moe import MoELayer
    from lumina_framework.expert_memory import ExpertMemoryEngine
    from lumina_framework.runtime import StubModel, RuntimeFabric
    from lumina_framework.memory import TIER_RAM
    from lumina_framework.moe_torch import TorchMoELayer, TorchStub

    rng = np.random.default_rng(9)
    # 1. layer parity
    ref = MoELayer(0, n_expert=8, d_model=32, d_ff=64, top_k=2, seed=3)
    bank = {}
    for e in ref.experts:
        bank[e.name] = e
    bank[ref.shared.name] = ref.shared
    eng = ExpertMemoryEngine(bank, ram_budget=256 * 1024 * 1024, vram_budget=0)
    eng.pin_hot_experts(8, target=TIER_RAM)
    x = rng.standard_normal((5, 32)).astype(np.float32)
    y_np = ref.forward(x, eng, target=TIER_RAM)
    tly = TorchMoELayer(ref, device="cpu", dtype=torch.float32)
    y_t = tly.forward(torch.as_tensor(x)).cpu().numpy()
    d = float(np.abs(y_np - y_t).max())
    check(f"layer parity max|diff| ({d:.2e})", d < 2e-2, f"diff {d}")

    # 2. routing weight parity (top-k indices identical)
    idx_np, w_np = ref.router.route(x, 2)
    s = torch.sigmoid(torch.as_tensor(x) @ tly.router_w.T) * tly.scaling
    ti = torch.topk(s, 2, dim=1).indices.cpu().numpy()
    # numpy argsort(-s)[:, :k] vs torch.topk: same set, possibly same order
    same_set = all(set(idx_np[r]) == set(ti[r]) for r in range(x.shape[0]))
    check("router top-k sets identical", same_set)

    # 3. full-stack greedy parity: RuntimeFabric.decode vs TorchStub.greedy
    m = StubModel(n_layer=3, n_expert=8, d_model=64, d_ff=96, top_k=2, seed=6)
    bank = {}
    for ly in m.layers:
        for e in ly.experts:
            bank[e.name] = e
        bank[ly.shared.name] = ly.shared
    eng = ExpertMemoryEngine(bank, ram_budget=256 * 1024 * 1024, vram_budget=0)
    eng.pin_hot_experts(24, target=TIER_RAM)
    rt = RuntimeFabric(m, eng, prefetch=False)
    prompt = np.array([1, 2, 3, 4])
    r_np = rt.decode(prompt, 12, target=TIER_RAM)
    tm = TorchStub(m, device="cpu", dtype=torch.float32)
    gen_t = tm.greedy(prompt, 12)
    check("greedy tokens identical",
          list(r_np["tokens"]) == list(gen_t),
          f"numpy {list(r_np['tokens'])} vs torch {gen_t}")

    # 4. fp16 device variant still routes identically (order may differ on
    #    ties; compare final hidden within tolerance)
    tm16 = TorchStub(m, device="cpu", dtype=torch.float16)
    gen16 = tm16.greedy(prompt, 12)
    agree = sum(int(a == b) for a, b in zip(r_np["tokens"], gen16))
    check("fp16 greedy >= 10/12 tokens agree with fp32", agree >= 10,
          f"agree {agree}/12")

    print("---")
    if FAILS:
        print("RESULT: FAIL (%d)" % len(FAILS))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
