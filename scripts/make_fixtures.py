"""Build a tiny deterministic MoE model and export it to EVERY supported format.

This is the bridge between the Format track and the Framework track: it produces
one canonical FP32 model, then writes it into safetensors / GGUF / EXL3 / .lumina
so the framework's format-agnostic loading can be tested on all of them.

Run:  python3 scripts/make_fixtures.py [--outdir artifacts/fixtures]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_framework.adapters import gguf, exl3, lumina, safetensors  # noqa: E402


def build_model(n_layer: int = 2, n_expert: int = 4, d_model: int = 64, d_ff: int = 128,
                vocab: int = 512, seed: int = 1234) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    t: dict[str, np.ndarray] = {}
    t["embed.weight"] = (rng.standard_normal((vocab, d_model)) * 0.05).astype(np.float32)
    t["head.weight"] = (rng.standard_normal((vocab, d_model)) * 0.05).astype(np.float32)
    for l in range(n_layer):
        t[f"layers.{l}.router.weight"] = (rng.standard_normal((n_expert, d_model)) * 0.08).astype(np.float32)
        t[f"layers.{l}.shared.gate.weight"] = (rng.standard_normal((d_ff, d_model)) * 0.05).astype(np.float32)
        t[f"layers.{l}.shared.up.weight"] = (rng.standard_normal((d_ff, d_model)) * 0.05).astype(np.float32)
        t[f"layers.{l}.shared.down.weight"] = (rng.standard_normal((d_model, d_ff)) * 0.05).astype(np.float32)
        for e in range(n_expert):
            for proj in ("gate", "up", "down"):
                if proj == "down":
                    shape = (d_model, d_ff)
                else:
                    shape = (d_ff, d_model)
                t[f"layers.{l}.experts.{e}.{proj}.weight"] = (
                    rng.standard_normal(shape) * 0.05).astype(np.float32)
    meta = {
        "name": "lumina-fixture-moe",
        "arch": "swiglu-moe",
        "n_layer": n_layer, "n_expert": n_expert, "d_model": d_model, "d_ff": d_ff,
        "top_k": 2, "vocab": vocab, "router_scaling": 1.0,
        "config": {"n_layer": n_layer, "n_expert": n_expert, "d_model": d_model,
                   "d_ff": d_ff, "top_k": 2, "vocab": vocab},
    }
    return t, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "artifacts", "fixtures"))
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-expert", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-ff", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tensors, meta = build_model(args.n_layer, args.n_expert, args.d_model, args.d_ff)
    n_params = sum(a.size for a in tensors.values())
    print(f"fixture: {len(tensors)} tensors, {n_params/1e3:.1f}K params")

    p = os.path.join(args.outdir, "model.safetensors")
    safetensors.SafetensorsAdapter.save(p, tensors, metadata=meta)
    print(f"  wrote {p}  ({os.path.getsize(p)} bytes)")

    # float16 payload: lossless for the framework's float32 working copy
    p = os.path.join(args.outdir, "model.gguf")
    gguf.GGUFAdapter.save(p, tensors, ttype="F16", metadata={
        "general.name": meta["name"],
        "general.architecture": meta["arch"],
        "lumina.n_layer": meta["n_layer"],
        "lumina.n_expert": meta["n_expert"],
        "lumina.d_model": meta["d_model"],
        "lumina.d_ff": meta["d_ff"],
        "lumina.top_k": meta["top_k"],
        "lumina.vocab": meta["vocab"],
    })
    print(f"  wrote {p}  ({os.path.getsize(p)} bytes)")

    p = os.path.join(args.outdir, "model.exl3")
    exl3.EXL3Adapter.save(p, tensors, bits=3, metadata=meta)
    print(f"  wrote {p}  ({os.path.getsize(p)} bytes, "
          f"{exl3.effective_bpp(3):.3f} b/p declared)")

    p = os.path.join(args.outdir, "model.lumina")
    from lumina_framework.adapters import ModelSpec
    spec = ModelSpec(name=meta["name"], arch=meta["arch"], n_layer=meta["n_layer"],
                     n_expert=meta["n_expert"], d_model=meta["d_model"],
                     d_ff=meta["d_ff"], top_k=meta["top_k"], vocab=meta["vocab"])
    lumina.LuminaAdapter.save(p, tensors, spec=spec)
    print(f"  wrote {p}  ({os.path.getsize(p)} bytes)")

    print("\nAll formats written. Now run:  python3 tests/test_cross_format.py")


if __name__ == "__main__":
    main()
