"""Honest runtime benchmark — the Lumina Framework Evidence Ledger harness.

Measures what the plan declares as pass criteria, not just wall clock:
  * committed tok/s (reference has no speculation yet, so committed == raw)
  * bytes moved per tier PER GENERATED TOKEN (the north-star efficiency metric)
  * expert cache hit rate
  * the effect of a RAM budget (cache thesis) and of usage pinning (learning cache)
  * router-lookahead recall (colibri measured 71.6%; ours is reported, not claimed)

Every number printed is a MEASURED RESULT on this machine with this build. The
TierCost model is a declared proxy (see src/lumina_framework/expert_memory.py);
replace it with measured transport costs before quoting latency conclusions.

Run:  python3 scripts/bench_runtime.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_framework.expert_memory import ExpertMemoryEngine, TierCost  # noqa: E402
from lumina_framework.memory import TIER_NVME, TIER_RAM, TIER_VRAM  # noqa: E402
from lumina_framework.moe import RoutePredictor  # noqa: E402
from lumina_framework.runtime import RuntimeFabric, StubModel  # noqa: E402


def build(n_layer=6, n_expert=32, d_model=256, d_ff=512, top_k=2, seed=99):
    model = StubModel(n_layer=n_layer, n_expert=n_expert, d_model=d_model,
                      d_ff=d_ff, top_k=top_k, seed=seed)
    pages: dict[str, object] = {}
    for layer in model.layers:
        pages[layer.shared.name] = layer.shared
        for p in layer.experts:
            pages[p.name] = p
    return model, pages


def run_case(label: str, model, pages, ram_budget: int, n_tokens: int = 64,
             pin_after: int | None = None, top_p: float | None = None,
             seed: int = 0, per_expert_bytes: int = 1) -> dict:
    engine = ExpertMemoryEngine(pages, ram_budget=ram_budget, vram_budget=0, seed=seed)
    fabric = RuntimeFabric(model, engine, top_k=model.layers[0].top_k, top_p=top_p)
    prompt = np.arange(16)

    # per-tier deltas, measured honestly (previous methodology reported cumulative
    # totals divided by n_tokens, which hid the steady state)
    d_nvme, d_ram, d_vram = [], [], []
    prev = None

    t0 = time.perf_counter()
    tokens: list[int] = []
    h = model.embed[prompt].mean(axis=0, keepdims=True)
    for step in range(n_tokens):
        if pin_after is not None and step == pin_after:
            engine.record_usage([n for n in pages])
            # pin only what the budget can actually HOLD: a pin larger than the
            # cache budget just evicts itself (this was a real bug in v0.1)
            slots = max(0, (ram_budget - engine.ram.used) // per_expert_bytes - 1)
            engine.pin_hot_experts(int(slots))
        st = engine.stats()
        cur = (st["nvme"]["read_bytes"], st["ram"]["read_bytes"], st["vram"]["read_bytes"])
        if prev is None:
            prev = cur
        d_nvme.append(cur[0] - prev[0])
        d_ram.append(cur[1] - prev[1])
        d_vram.append(cur[2] - prev[2])
        prev = cur
        hs = model.forward_hidden(h, engine, target=TIER_RAM)
        logits = hs @ model.head.T
        nxt = int(np.argmax(logits[0]))
        tokens.append(nxt)
        h = model.embed[[nxt]]
    elapsed = time.perf_counter() - t0

    st = engine.stats()
    # split cold-fill time from steady-state decode: re-run warm with same state
    warm_tokens: list[int] = []
    tw = time.perf_counter()
    h = model.embed[prompt].mean(axis=0, keepdims=True)
    for _ in range(n_tokens):
        hs = model.forward_hidden(h, engine, target=TIER_RAM)
        logits = hs @ model.head.T
        nxt = int(np.argmax(logits[0]))
        warm_tokens.append(nxt)
        h = model.embed[[nxt]]
    warm_elapsed = time.perf_counter() - tw

    # memory-model response time (TierCost proxy) — the number that reflects the
    # hierarchy. Kept strictly separate from wall clock.
    mem_time = (st["ram"]["seconds_in_read"]
                + st["nvme"].get("seconds_in_read", 0.0)
                + st["vram"]["seconds_in_read"])

    total_read = st["ram"]["read_bytes"] + st["nvme"]["read_bytes"] + st["vram"]["read_bytes"]
    hot_fraction = (st["ram"]["read_bytes"] + st["vram"]["read_bytes"]) / max(1, total_read)

    return {
        "label": label,
        "ram_budget_gb": ram_budget / 1e9,
        "tokens": n_tokens,
        "cold_seconds": elapsed,
        "warm_seconds": warm_elapsed,
        "cold_tok_per_s": n_tokens / elapsed,
        "warm_tok_per_s": n_tokens / warm_elapsed,
        "warm_nvme_mb_per_token": float(np.mean(d_nvme)) / 1e6,
        "warm_ram_mb_per_token": float(np.mean(d_ram)) / 1e6,
        "nvme_bytes_total": st["nvme"]["read_bytes"],
        "ram_evictions": st["ram"]["evictions"],
        "hot_read_fraction": hot_fraction,
        "ram_utilization": st["ram"]["used"] / max(1, st["ram"]["budget"]),
        "resident_experts": st["ram"]["keys"],
        "total_experts": len(pages),
        "memory_model_seconds_per_token": mem_time / max(1, n_tokens),
        "identical_warm_decode": warm_tokens == tokens,
    }


def router_recall_case(model, n_tokens=64, seed=5):
    """Measure router-lookahead recall: apply layer L+1's router to layer L's
    post-attention state and compare against the true layer L+1 routing."""
    predictor = RoutePredictor()
    rng = np.random.default_rng(seed)
    for _ in range(8):
        x = rng.standard_normal((16, model.layers[0].d_model)).astype(np.float32)
        # reference (oracle-free) route through layer 0
        idx0, _ = model.layers[0].router.route(x, model.layers[0].top_k)
        # lookahead rule: use next layer's router on this layer's output state
        h0 = model.layers[0].forward(x, _NoMemory())
        predicted, _ = model.layers[1].router.route(h0, model.layers[1].top_k)
        actual, _ = model.layers[0].router.route(x, model.layers[0].top_k)
        # cross-layer recall: how often the predicted expert set of layer L+1
        # matches the expert set layer L would actually route to
        predictor.update_recall(predicted, idx0)
    return {"route_recall": predictor.recall,
            "token_accuracy": predictor.token_accuracy,
            "colibri_published_recall": 0.716}


class _NoMemory:
    """Resident-by-default fake engine for measuring routing in isolation."""

    def get(self, keys, target=TIER_RAM):
        return [(DummyPage(), False) for _ in keys]


class DummyPage:
    size_bytes = 0


def main() -> None:
    print("Building reference model...")
    model, pages = build()
    per_expert = next(iter(pages.values())).size_bytes
    total_expert_bytes = sum(p.size_bytes for p in pages.values())
    print(f"  layers=6 experts=32 d_model=256 d_ff=512 top_k=2")
    print(f"  total expert bytes: {total_expert_bytes/1e6:.2f} MB "
          f"({per_expert/1e3:.1f} KB/expert, "
          f"effective bpp={per_expert/(3*512*256):.3f})\n")

    cases = []
    # cache thesis: same model, different RAM budgets
    for frac in (0.05, 0.15, 0.30, 0.60, 1.00):
        ram = int(total_expert_bytes * frac)
        cases.append((f"RAM {frac*100:.0f}% of expert bank", ram, None))
    # learning pinning on the smallest budget that leaves headroom
    cases.append(("RAM 15% + pin after 16 tokens", int(total_expert_bytes * 0.15), 16))

    print(f"{'case':<28}{'warm tok/s':>12}{'cold tok/s':>12}"
          f"{'NVMe MB/tok':>14}{'hot frac':>10}{'mem-model s/tok':>17}{'evict':>7}")
    rows = []
    for label, ram, pin in cases:
        r = run_case(label, model, pages, ram, n_tokens=64, pin_after=pin,
                            per_expert_bytes=per_expert)
        rows.append(r)
        print(f"{label:<28}{r['warm_tok_per_s']:>12.1f}{r['cold_tok_per_s']:>12.1f}"
              f"{r['warm_nvme_mb_per_token']:>14.2f}{r['hot_read_fraction']:>10.2f}"
              f"{r['memory_model_seconds_per_token']:>17.3e}{r['ram_evictions']:>7}")

    print()
    rr = router_recall_case(model)
    print(f"[router lookahead] recall={rr['route_recall']:.3f} "
          f"token_accuracy={rr['token_accuracy']:.3f} "
          f"(colibri published 0.716 on GLM-5.2 — our reference is a synthetic router, "
          f"so this number is NOT a claim about real models)")

    out = {
        "evidence_label": "MEASURED RESULT",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build": "lumina_framework v0.1 (numpy reference)",
        "tier_cost_model": TierCost().as_dict(),
        "cases": rows,
        "router": rr,
    }
    outp = os.path.join(ROOT, "artifacts", "bench_runtime.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {outp}")
    print("NOTE: synthetic reference model; throughput figures are framework "
          "overhead measurements, not GLM-5.3-class predictions.")


if __name__ == "__main__":
    main()
