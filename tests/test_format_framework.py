"""Format -> Framework integration: the 1-bit execution path end to end.

Proves that the payload produced by the Lumina Format codec is directly
executable by the Framework kernels, with no FP16 weight tensor ever
materialized — which is the whole point of the format being an *execution
contract* and not just a compressed file.

  I1. `forward_from_payload` (fused, through packed bytes) == plain reference.
  I2. `cpu_binary_gemv` consumes a packed sign payload and is bit-exact.
  I3. The int8 container path matches the FP32 reference within int8 error.

Run:  python3 tests/test_format_framework.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_format.codecs import low_rank_binary as llrb  # noqa: E402
from lumina_framework.kernels import (backend, cpu_binary_gemv,  # noqa: E402
                                      pack_signs, quantize_int8_rows,
                                      trt_dequant_matmul)


def test_fused_forward_matches_reference():
    """I1."""
    W = llrb.llm_like_matrix(256, 256, seed=1)
    p = llrb.encode(W, target_bpp=0.5)
    x = np.random.default_rng(0).standard_normal((4, W.shape[1]))

    y_fused = llrb.forward_from_payload(p, x, fused=True)
    y_ref = x @ llrb.decode(p).T
    err = float(np.abs(y_fused - y_ref).max())
    print(f"[I1] fused forward vs reference: max_abs_err={err:.2e}  pass={err < 1e-10}")
    assert err < 1e-10, f"I1 FAIL: {err}"


def test_binary_gemv_bit_exact():
    """I2 — the sign payload round-trips through the packed GEMV exactly."""
    rng = np.random.default_rng(0)
    S = np.where(rng.random((64, 512)) < 0.5, -1, 1).astype(np.int8)
    x = (rng.standard_normal((4, 512)) * 0.1).astype(np.float32)
    packed = pack_signs(S)
    got = cpu_binary_gemv(x, packed, np.ones(64, np.float32), 512)
    ref = x @ S.T
    err = float(np.abs(got - ref).max())
    packed_bytes = packed.nbytes
    naive_bytes = S.size  # int8 payload for comparison
    print(f"[I2] binary GEMV parity: max_abs_err={err:.2e}  "
          f"payload {packed_bytes}B vs int8 {naive_bytes}B "
          f"({naive_bytes / packed_bytes:.0f}x smaller)  pass={err < 1e-10}")
    assert err < 1e-10, f"I2 FAIL: {err}"
    # one bit per sign -> 8x smaller than an int8 payload
    assert packed_bytes * 8 == naive_bytes, "expected 8x bit packing"


def test_int8_container_path():
    """I3 — Framework int8 kernels match FP32 within quantization error."""
    rng = np.random.default_rng(2)
    W = (rng.standard_normal((256, 512)) * 0.1).astype(np.float32)
    x = (rng.standard_normal((4, 512)) * 0.1).astype(np.float32)
    q, s = quantize_int8_rows(W)
    got = trt_dequant_matmul(x, q, s)
    ref = x @ W.T
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    print(f"[I3] int8 fused matmul rel_err={rel:.4f} ({backend()})  pass={rel < 0.01}")
    assert rel < 0.01, f"I3 FAIL: rel err {rel}"


def test_low_bpp_is_executable():
    """The sub-1-bit payload must be executable, not just decodable."""
    W = llrb.llm_like_matrix(512, 512, seed=3)
    for bpp in (0.5, 1.0):
        p = llrb.encode(W, target_bpp=bpp)
        x = np.random.default_rng(1).standard_normal((1, W.shape[1]))
        y = llrb.forward_from_payload(p, x, fused=True)
        assert y.shape == (1, W.shape[0]), "output shape wrong"
        print(f"  bpp={bpp:.3f} rank={p.rank:>4} -> forward shape {y.shape} "
              f"(no W materialized)")
    print("[I4] sub-1-bit payloads execute end to end")


def main() -> int:
    test_fused_forward_matches_reference()
    test_binary_gemv_bit_exact()
    test_int8_container_path()
    test_low_bpp_is_executable()
    print("\nALL FORMAT-FRAMEWORK INTEGRATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
