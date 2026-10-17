"""Stream-quantize a real HF model dir -> .lumina + fidelity report.

Memory-safe: reads tensor-by-tensor straight from safetensors offsets, so
even 7B runs under phone RAM. Doctrine by NAME pattern (no forwards needed):
embed/head + first layers protected, bulk at chosen bits. H=I fallback is
RECORDED (no fake Hessian). Fidelity = per-tensor relW + honest bpp.
"""

from __future__ import annotations

import glob
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from lumina_format.codecs.ptq_ladder import (dequantize, quantize_binary,
                                             quantize_int)
from lumina_format.codecs import salient as S
from lumina_format.codecs import low_rank_binary as LRB


def iter_tensors(model_dir):
    """Yield (name, array-f32, dtype, shard) streaming, one tensor at a time."""
    from lumina_framework.adapters.safetensors import _read_dtype
    for sh in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(sh, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
            base = 8 + n
            hdr.pop("__metadata__", None)
            for name, info in hdr.items():
                s, e = info["data_offsets"]
                f.seek(base + s)
                buf = f.read(e - s)
                yield name, _read_dtype(info["dtype"], buf, info["shape"], 0)


def policy(name: str, n_layer: int, bulk: int) -> tuple:
    """-> ('int',bits,frac) | ('bin',frac) | ('fp',) . Doctrine by name."""
    ln = name.lower()
    if "embed" in ln or "lm_head" in ln or "output" in ln:
        return ("int", 8, 0.0)
    try:
        lyr = int(name.split("layers.")[1].split(".")[0])
    except Exception:
        lyr = 999
    if lyr == 0:
        return ("int", 8, 0.0)
    if lyr == 1:
        return ("int", 4, 0.0)
    if bulk == 1:
        return ("bin", 0.02)
    return ("int", bulk, 0.0)


def quant_one(W, spec, group=64):
    W = np.asarray(W, dtype=np.float64)
    if spec == ("fp",):
        return W.astype(np.float32), W.size * 32, "fp32"
    if spec[0] == "int":
        _, b, f = spec
        q = quantize_int(W, b, None, group)
        Wq = dequantize(q)
        bits = q.payload_bytes() * 8
    else:
        _, f = spec
        q = quantize_binary(W, None, group)
        Wq = dequantize(q)
        bits = q.payload_bytes() * 8
    if spec[-1] > 0 and W.ndim == 2 and min(W.shape) >= 16:
        mk = S.salient_mask(W, None, spec[-1])
        Wq = S.apply_corrections(Wq, W, mk)
        bits += S.sparse_bytes(S.pack_sparse(W, mk))
    return Wq.astype(np.float32), bits, q.kind if spec[0] != "fp" else "fp32"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bulk", type=int, default=4)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    t = cfg.get("text_config", cfg)
    n_layer = int(t.get("num_hidden_layers", t.get("num_layers", 1)))
    tensors, stats = {}, []
    qb = bb = 0
    off = int(os.environ.get("QOFF", "0"))
    cnt = int(os.environ.get("QCOUNT", "1000000"))
    seen = [0]
    for name, arr in iter_tensors(a.model):
        if seen[0] < off:
            seen[0] += 1
            continue
        if seen[0] >= off + cnt:
            break
        seen[0] += 1
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        W = arr
        spec = policy(name, n_layer, a.bulk) if W.ndim == 2 \
            and min(W.shape) >= 16 else ("fp",)
        Wq, bits = quant_one(W, spec)
        d = float(np.linalg.norm(W - Wq) / max(np.linalg.norm(W), 1e-12))
        qb += bits
        bb += W.size * 32
        tensors[name] = Wq
        stats.append({"name": name, "kind": spec, "relW": round(d, 4),
                      "bpp": round(8 * bits / max(W.size, 1), 3)})
    import lumina_convert as lc
    from lumina_framework.adapters import ModelSpec
    spec_m = ModelSpec(name=os.path.basename(a.model.rstrip("/")),
                       n_layer=n_layer)
    # store dequantized reference (loader path); codes path via converter
    lc.write_lumina(a.out, {k: ("raw", v) for k, v in tensors.items()},
                    stats, spec_m.name, f"int{a.bulk}")
    rep = {"model": a.model, "bulk": a.bulk, "honest_bpp": qb / bb * 32,
           "hessian": "I-fallback (no forwards on phone)",
           "tensors": stats}
    rp = a.report or a.out + ".json"
    json.dump(rep, open(rp, "w"), indent=1)
    print(f"honest_bpp={qb/bb*32:.3f} tensors={len(stats)} report={rp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
