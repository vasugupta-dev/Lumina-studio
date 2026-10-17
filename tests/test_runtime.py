"""Reference smoke test + honest accounting for the Lumina Framework.

Runs a small StubModel MoE through the Expert Memory Engine and asserts the
correctness invariants from moe.py docstring, then prints the accounting.

Run:  python3 tests/test_runtime.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumina_framework.expert_memory import ExpertMemoryEngine, TierCost  # noqa: E402
from lumina_framework.memory import TIER_RAM, TIER_NVME, TIER_VRAM, ExpertPage, TensorContainer  # noqa: E402
from lumina_framework.moe import MoELayer, Router  # noqa: E402
from lumina_framework.runtime import RuntimeFabric, StubModel  # noqa: E402

RNG = np.random.default_rng(7)


def dense_reference(layer: MoELayer, x: np.ndarray, top_k: int) -> np.ndarray:
    """All-resident FP32 reference forward with no memory engine involved."""
    idx, w = layer.router.route(x, top_k)
    out = layer.shared.forward(x)
    for e in range(layer.n_expert):
        rows = [r for r in range(x.shape[0]) if e in idx[r].tolist()]
        if not rows:
            continue
        y = layer.experts[e].forward(x[rows])
        for k, r in enumerate(rows):
            j = idx[r].tolist().index(e)
            out[r] += w[r, j] * y[k]
    return out


def test_invariant_I1_resident_equals_dense():
    layer = MoELayer(0, n_expert=8, d_model=64, d_ff=128, top_k=2, seed=1)
    x = RNG.standard_normal((16, 64)).astype(np.float32)

    pages = [layer.shared] + layer.experts
    engine = ExpertMemoryEngine({p.name: p for p in pages}, ram_budget=10**12)

    got = layer.forward(x, engine)
    ref = dense_reference(layer, x, top_k=2)
    err = np.abs(got - ref).max()
    print(f"[I1] streaming forward vs dense reference: max_abs_err={err:.3e}  pass={err < 1e-3}")
    assert err < 1e-3, f"I1 FAIL: {err}"


def test_invariant_I2_batch_union_reads_each_expert_once():
    layer = MoELayer(0, n_expert=8, d_model=64, d_ff=128, top_k=2, seed=2)
    x = RNG.standard_normal((32, 64)).astype(np.float32)
    pages = [layer.shared] + layer.experts
    engine = ExpertMemoryEngine({p.name: p for p in pages}, ram_budget=10**12)

    # make everything cold
    for p in layer.experts:
        engine.ram.drop(p.name)

    before = {p.name: engine.nvme.stats.read_bytes for p in layer.experts}
    layer.forward(x, engine)
    after = {p.name: engine.nvme.stats.read_bytes for p in layer.experts}
    moved = [n for n, b in before.items() if after[n] > b]
    idx, _ = layer.router.route(x, 2)
    unique_used = set(idx.ravel().tolist())
    print(f"[I2] cold experts read: {len(moved)}  unique experts routed: {len(unique_used)}  "
          f"pass={len(moved) == len(unique_used)}")
    assert len(moved) == len(unique_used), "I2 FAIL: expert read count mismatch"


def test_invariant_I3_prefetch_is_result_neutral():
    layer = MoELayer(0, n_expert=8, d_model=64, d_ff=128, top_k=2, seed=3)
    x = RNG.standard_normal((16, 64)).astype(np.float32)
    pages = [layer.shared] + layer.experts
    engine = ExpertMemoryEngine({p.name: p for p in pages}, ram_budget=10**12)
    idx, _ = layer.router.route(x, 2)
    predicted = layer.router.route(x, 2)[0]  # lookahead = same state here
    for e in sorted(set(idx.ravel().tolist())):
        engine.ram.drop(layer.experts[e].name)
    for e in sorted(set(predicted.ravel().tolist())):
        engine.ram.drop(layer.experts[e].name)
    nvme_before = engine.nvme.stats.read_bytes
    out_ref = layer.forward(x.copy(), engine).copy()
    for e in sorted(set(predicted.ravel().tolist())):
        engine.ram.drop(layer.experts[e].name)
    engine.nvme.stats.read_bytes = nvme_before
    out_pre = layer.forward(x.copy(), engine)
    err = np.abs(out_ref - out_pre).max()
    print(f"[I3] prefetched-run vs reference-run: max_abs_err={err:.3e}  pass={err < 1e-6}")
    assert err < 1e-6, "I3 FAIL: prefetch changed the result"


def test_tiered_execution_and_accounting():
    """Small MoE stack with RAM budget small enough to force streaming."""
    n_layer, n_expert, d_model, d_ff, top_k = 4, 16, 128, 256, 2
    model = StubModel(n_layer=n_layer, n_expert=n_expert, d_model=d_model,
                      d_ff=d_ff, top_k=top_k, seed=11)
    pages: dict[str, ExpertPage] = {}
    for layer in model.layers:
        pages[layer.shared.name] = layer.shared
        for p in layer.experts:
            pages[p.name] = p
    per_expert_bytes = pages[f"L0-e0"].size_bytes
    print(f"per-expert container bytes: {per_expert_bytes} "
          f"(bpp={per_expert_bytes / (3 * d_ff * d_model):.3f} for 3 projections)")

    # tiny RAM: only shared experts + a few routed experts fit
    shared_bytes = sum(1 for k in pages if k.endswith("shared")) * per_expert_bytes
    engine = ExpertMemoryEngine(pages, ram_budget=shared_bytes + 3 * per_expert_bytes,
                                 vram_budget=0, seed=0)
    # cold run
    prompt = np.arange(8)
    fabric = RuntimeFabric(model, engine, top_k=top_k)
    r1 = fabric.decode(prompt, n_tokens=8)

    # learning cache warm-up, then pinned run over the SAME prompt
    engine.record_usage([n for n in pages])
    pinned = engine.pin_hot_experts(4)
    r2 = fabric.decode(prompt, n_tokens=8)

    print(f"\n[rt] cold-ish decode: {r1['tokens_per_second']:.1f} tok/s  "
          f"({len(pinned)} experts pinned after learning)")
    print(f"[rt] pinned decode:    {r2['tokens_per_second']:.1f} tok/s")
    s = engine.stats()
    print(f"[rt] bytes read RAM={s['ram']['read_bytes']} NVMe={s['nvme']['read_bytes']} "
          f"RAM evictions={s['ram']['evictions']}")
    assert r1["tokens"] == r2["tokens"], "I4 FAIL: tier placement changed decode output"


def test_usage_pinning_changes_placement():
    pages = {}
    rng = np.random.default_rng(5)
    for i in range(16):
        pages[f"e{i}"] = ExpertPage(
            f"e{i}", {"gate": TensorContainer(rng.standard_normal((64, 32)) * .1),
                      "up": TensorContainer(rng.standard_normal((64, 32)) * .1),
                      "down": TensorContainer(rng.standard_normal((32, 64)) * .1)}, 0)
    engine = ExpertMemoryEngine(pages, ram_budget=4 * pages["e0"].size_bytes)
    for _ in range(20):
        engine.record_usage(["e3", "e9"])
    pinned = engine.pin_hot_experts(2)
    print(f"[pin] pinned={pinned} (expected exactly e3 and e9 by heat)")
    assert set(pinned) == {"e3", "e9"}, f"pinning FAIL: {pinned}"


if __name__ == "__main__":
    test_invariant_I1_resident_equals_dense()
    test_invariant_I2_batch_union_reads_each_expert_once()
    test_invariant_I3_prefetch_is_result_neutral()
    test_usage_pinning_changes_placement()
    test_tiered_execution_and_accounting()
    print("\nALL FRAMEWORK REFERENCE TESTS PASSED")
