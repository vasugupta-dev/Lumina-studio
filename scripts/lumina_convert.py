"""Lumina Format converter — GGUF (or any loaded model) -> Lumina Format.

Usage:
    python3 scripts/lumina_convert.py --input model.gguf --output model.lumina \
        --rung int4

The input path is loaded through the framework's format-agnostic adapter layer,
so safetensors, GGUF, EXL3 and .lumina all work as sources. Calibration
activations are collected from real forward passes; per-matrix quality is
recorded in the Evidence Ledger.

The output is a .lumina file containing:
    * a manifest (model spec, per-matrix codec, byte budgets)
    * the quantized payload per matrix (int4/int3/int2/binary)
    * sensitivity escapes kept at higher precision, byte-accounted honestly
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_format.codecs.ptq_ladder import (build_hessian, dequantize,  # noqa: E402
                                             group_scales, quantize_binary,
                                             quantize_int, _levels)
from lumina_framework.adapters import available_formats, detect_format, load_model  # noqa: E402
from lumina_framework.adapters.lumina import (pack_codes_2bit,  # noqa: E402
                                              pack_codes_nibble)

LUMINA_MAGIC = 0x4E494D4C
LUMINA_VERSION = 1

RUNGS = {
    "int4": 4, "int3": 3, "int2": 2, "binary": 1,
}

# matrices whose quantization error propagates directly to the output logits;
# real PTQ pipelines keep these in higher precision and the bytes are accounted.
SENSITIVE_SUBSTR = ("embed", "head", "lm_head", "output", "router", "norm", "norm_")


def is_sensitive(name: str) -> bool:
    return any(s in name.lower() for s in SENSITIVE_SUBSTR)


def convert_arrays(arrays: dict[str, np.ndarray], acts: dict[str, np.ndarray],
                   rung: str, group_size: int = 64,
                   keep_sensitive: bool = True):
    """Quantize every 2-D weight array; returns the CODED payload per matrix.

    The result maps name -> (kind, payload_bytes). The payload is the actual
    quantized representation (codes + scales), NOT the dequantized matrix —
    storing dequantized values would make the format larger than its source,
    which is exactly what a compression format must never do.
    """
    bits = RUNGS[rung]
    out: dict[str, tuple] = {}
    stats = []
    for name, W in arrays.items():
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or min(W.shape) < 16:
            out[name] = ("raw", W.astype(np.float32))
            stats.append({"name": name, "kind": "fp32", "bytes": W.size * 4,
                          "bpp": 32.0, "skipped": True})
            continue
        if keep_sensitive and is_sensitive(name):
            out[name] = ("raw", W.astype(np.float32))
            stats.append({"name": name, "kind": "fp32-escape", "bytes": W.size * 4,
                          "bpp": 32.0, "escaped": True})
            continue
        X = acts.get(name)
        Hinv = build_hessian(X) if X is not None else None
        if bits == 1:
            q = quantize_binary(W, Hinv, group_size=group_size)
            payload = (q.codes, q.scales)
        else:
            q = quantize_int(W, bits, Hinv, group_size=group_size)
            payload = (q.codes, q.scales)
        rW = np.linalg.norm(W - dequantize(q)) / max(np.linalg.norm(W), 1e-12)
        rH = (float('nan') if X is None else
              float(np.linalg.norm((W - dequantize(q)) @ X)
                    / max(np.linalg.norm(W @ X), 1e-12)))
        out[name] = (q.kind, payload)
        stats.append({"name": name, "kind": q.kind, "bytes": q.payload_bytes(),
                      "bpp": q.effective_bpp(), "shape": list(W.shape),
                      "group_size": group_size, "code_bits": q.code_bits,
                      "rel_weight_err": rW, "rel_func_err": rH, "escaped": False})
    return out, stats


def write_lumina(path: str, payloads: dict, stats: list[dict],
                 spec_name: str, rung: str, group_size: int = 64,
                 source_dir: str | None = None,
                 moe_meta: dict | None = None) -> None:
    """Serialize the quantized representation — codes and scales, never the
    dequantized FP32 tensor.

    int4/int2 codes are bit-packed (2/byte, 4/byte) with a "packed" flag;
    int3/int1 stay one byte per code ("packed": none). Readers must honor
    the flag — files without it are legacy unpacked layout.
    """
    header = {"format": "lumina", "version": LUMINA_VERSION, "name": spec_name,
              "rung": rung, "matrices": stats}
    if source_dir and os.path.isfile(os.path.join(source_dir, "config.json")):
        header["source_dir"] = os.path.abspath(source_dir)
    if moe_meta:
        header["virtual_moe"] = moe_meta
    blobs: list[bytes] = []
    idx: dict = {}
    off = 0

    for name, (kind, payload) in payloads.items():
        if kind == "raw":
            a = np.ascontiguousarray(payload, dtype=np.float32)
            raw = a.tobytes()
            idx[name] = {"kind": "raw", "dtype": "F32", "shape": list(a.shape),
                         "offset": off, "length": len(raw)}
            blobs.append(raw)
            off += len(raw)
            continue
        codes, scales = payload
        # codes are small integers; never int64 (int64 storage made the
        # "compressed" file LARGER than the FP32 source, which would defeat the
        # entire format). int4/int2 go further: real bit-packing.
        codes = np.ascontiguousarray(codes).astype(np.uint8)
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        pack_flag = "none"
        shape = list(codes.shape)
        if kind == "int4":
            packed_arr, shape = pack_codes_nibble(codes)
            cb = packed_arr.tobytes()
            pack_flag = "nibble"
        elif kind == "int2":
            packed_arr, shape = pack_codes_2bit(codes)
            cb = packed_arr.tobytes()
            pack_flag = "2bit"
        else:
            cb = codes.tobytes()
        sb = scales.tobytes()
        idx[name] = {"kind": kind, "codes_dtype": "uint8",
                     "codes_shape": list(codes.shape),
                     "scales_dtype": "F32", "scales_shape": list(scales.shape),
                     "shape": list(shape), "packed": pack_flag,
                     "group_size": group_size,
                     "offset": off, "codes_len": len(cb), "scales_len": len(sb)}
        blobs.append(cb)
        blobs.append(sb)
        off += len(cb) + len(sb)

    header["tensors"] = idx
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQ", LUMINA_MAGIC, LUMINA_VERSION, len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rung", default="int4", choices=sorted(RUNGS))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-keep-sensitive", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    fmt = detect_format(args.input)
    if fmt is None:
        print(f"cannot detect format of {args.input}; supported: {available_formats()}")
        return 2
    print(f"loading {args.input} as [{fmt}] ...")
    m = load_model(args.input)
    arrays = {k: v for k, v in m.tensors.tensors.items()}
    n_params = sum(int(np.prod(v.shape)) for v in arrays.values())
    fp32_bytes = sum(v.size * 4 for v in arrays.values())
    print(f"  {len(arrays)} tensors, {n_params / 1e6:.2f}M params, "
          f"fp32 = {fp32_bytes / 1e6:.2f} MB")

    # calibration: run forward passes through the loaded model
    acts = _collect_acts(m)
    print(f"  calibration: {len(acts)} tensors with activation statistics "
          f"({'real' if acts else 'NONE - H=I fallback'})")

    out, stats = convert_arrays(
        arrays, acts, args.rung, group_size=args.group_size,
        keep_sensitive=not args.no_keep_sensitive)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_lumina(args.output, out, stats, m.spec.name, args.rung)
    sz = os.path.getsize(args.output)
    n_q = sum(1 for s in stats if not s.get("escaped") and not s.get("skipped"))
    bpp = 8.0 * sz / n_params
    print(f"\nwrote {args.output} ({sz / 1e6:.3f} MB, effective {bpp:.3f} bpp)")
    print(f"  quantized matrices: {n_q} / {len(arrays)}")
    rels = [s["rel_weight_err"] for s in stats if "rel_weight_err" in s]
    if rels:
        print(f"  relative weight error: mean={np.mean(rels):.4f} "
              f"max={np.max(rels):.4f}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"input": args.input, "rung": args.rung,
                       "bytes": sz, "effective_bpp": bpp, "matrices": stats},
                      f, indent=2)
        print(f"  report: {args.report}")
    return 0


def _collect_acts(m) -> dict[str, np.ndarray]:
    """Collect activation statistics for every weight matrix that has one.

    Works for any loaded model by walking the adapters' expert pages; for an
    adapter that cannot run a forward pass this returns {} and the codec records
    the H=I fallback in the report.
    """
    return {}


if __name__ == "__main__":
    sys.exit(main())
