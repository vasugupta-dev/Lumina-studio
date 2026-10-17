"""Honest-bpp ladder: hit exact average-bpp rungs with best quality.

Each rung = (bulk codec, head policy, blk0 policy, salient frac). ALL bytes
counted: codes + scales + sparse + escapes. Reports measured (bpp, loss) per
rung — goals are targets, measurements are truth.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import torch
from mini_lm import MiniLM, evaluate
from lumina_format.codecs.ptq_ladder import (build_hessian, dequantize,
                                             quantize_binary, quantize_int)
from lumina_format.codecs import salient as S
from lumina_format.codecs import low_rank_binary as LRB
from convert_and_eval import collect_activations

POLICIES = [
    # (tag, goal, bulk, head, blk0, salient_frac_on_bulk)
    ("r4", 4.0, 4, ("int", 4, 0.02), ("int", 4, 0.0)),
    ("r3", 3.0, 3, ("int", 8, 0.0), ("int", 4, 0.0)),
    ("r2", 2.0, 2, ("int", 8, 0.0), ("int", 4, 0.0)),
    ("r1.58", 1.58, 1, ("int", 8, 0.0), ("int", 4, 0.0)),
    ("r1", 1.0, 1, ("bin", 0.05), ("int", 2, 0.0)),
    ("r0.85", 0.85, 1, ("bin", 0.05), ("bin", 0.02)),
    ("r0.75", 0.75, 1, ("bin", 0.03), ("bin", 0.02)),
    ("r0.5", 0.5, ("llrb", 0.35), ("bin", 0.05), ("bin", 0.02)),
    ("r0.4", 0.4, ("llrb", 0.25), ("bin", 0.05), ("bin", 0.02)),
    ("r0.2", 0.2, ("llrb", 0.1), ("bin", 0.02), ("bin", 0.01)),
    ("r0.1", 0.1, ("llrb", 0.05), ("bin", 0.02), ("bin", 0.01)),
]


def quant_layer(W, X, spec, group=64):
    """spec: bulk int | ('int',b,f) | ('bin',f) | ('llrb',bpp). -> (Wq, bits)."""
    if isinstance(spec, int):
        H = build_hessian(X) if X is not None else None
        q = quantize_binary(W, H, group) if spec == 1 \
            else quantize_int(W, spec, H, group)
        return dequantize(q), q.payload_bytes() * 8
    kind = spec[0]
    if kind == "int":
        _, b, f = spec
        H = build_hessian(X) if X is not None else None
        q = quantize_int(W, b, H, group)
        Wq, bits = dequantize(q), q.payload_bytes() * 8
    elif kind == "bin":
        _, f = spec
        H = build_hessian(X) if X is not None else None
        q = quantize_binary(W, H, group)
        Wq, bits = dequantize(q), q.payload_bytes() * 8
    else:
        p = LRB.encode(W, target_bpp=spec[1], sign_sweeps=1, als_iters=6)
        return LRB.decode(p), p.payload_bytes() * 8
    if kind in ("int", "bin") and spec[-1] > 0:
        mk = S.salient_mask(W, X, spec[-1])
        Wq = S.apply_corrections(Wq, W, mk)
        bits += S.sparse_bytes(S.pack_sparse(W, mk))
    return Wq, bits


def run_policy(base, seq, bl, names, tag, goal, bulk, head_spec, blk0_spec):
    m = copy.deepcopy(base)
    qb = bb = 0
    acts = collect_activations(m, seq, n_calib=16)
    for idx, n in enumerate(names):
        if idx and idx % 2 == 0:
            acts = collect_activations(m, seq, n_calib=16)
        mod = dict(m.named_modules())[n]
        W = mod.weight.detach().numpy().astype(np.float64)
        bb += W.size * 32
        X = acts.get(n)
        spec = head_spec if n == "head" else (
            blk0_spec if n.startswith("blocks.0") else bulk)
        Wq, bits = quant_layer(W, X, spec)
        qb += bits
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(Wq.astype(np.float32)))
    loss = evaluate(m, seq)
    return {"tag": tag, "goal": goal, "bpp": qb / bb * 32,
            "loss": loss, "rel_pct": (loss - bl) / bl * 100.0}


def main() -> int:
    ck = torch.load(os.path.join(ROOT, "artifacts", "reference.pt"),
                     map_location="cpu", weights_only=False)
    base = MiniLM()
    base.load_state_dict(ck["state"])
    seq = ck["seq"]
    bl = evaluate(base, seq)
    print(f"base={bl:.4f}")
    names = [n for n, _ in base.named_modules()
             if isinstance(_, torch.nn.Linear)]
    out = []
    for tag, goal, bulk, hs, bs in POLICIES:
        t0 = time.time()
        r = run_policy(base, seq, bl, names, tag, goal, bulk, hs, bs)
        r["seconds"] = round(time.time() - t0, 1)
        ok = "PASS" if abs(r["rel_pct"]) < 5.0 else "MISS"
        bpp = r["bpp"]; rel = r["rel_pct"]; sec = r["seconds"]
        print(tag, "goal=", goal, "bpp=", round(bpp, 2),
              "rel=", round(rel, 2), "[" + ok + "]", flush=True)
        out.append(r)
    json.dump({"base_loss": bl, "rungs": out},
              open(os.path.join(ROOT, "artifacts", "bpp_ladder.json"), "w"),
              indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
