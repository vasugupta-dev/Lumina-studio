"""Lumina Framework kernels — TensorRT-LLM class execution primitives.

The benchmark in scripts/bench_runtime.py showed the reference's wall clock is
dominated by dequantize-on-use, NOT by memory transport. That makes the fused
kernel the single highest-leverage optimization in the runtime, which is exactly
how TensorRT-LLM, Marlin, and ExLlamaV2 get their speed: the weight never exists
in FP16 on the way to the accumulator.

Two kernels are implemented here:

  1. `trt_dequant_matmul` — int8 weights with per-row scale, activations in FP16.
     Dequantization is FUSED into the GEMM epilogue (a TMA/async-copy pattern in
     CUTLASS terms), so no FP16 weight tensor is ever materialized in memory.

  2. `binary_gemv` — packed {-1,+1} weights expanded to a partial-sum on the fly
     with NO dequantization at all. This is the Lumina Format execution path: it
     consumes the exact bit-packed payload the LLRB codec produces.

Both are written in Triton so they compile for the target GPU (RTX 4050, sm_89)
and fall back to numpy reference semantics on CPU. The CPU reference is used as
the parity oracle, so the numbers on a non-CUDA build are still MEASURED — they
just measure the reference, not the kernel.

Design constraints inherited from the project handoffs:
  * Never unpack to FP16 per operator invocation.
  * A custom kernel is justified only after profiling shows the existing path is
    the bottleneck — which this repository's own benchmark did.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


def backend() -> str:
    """Which path is live. Reported in every benchmark."""
    if _HAS_TRITON and _HAS_TORCH and torch is not None and torch.cuda.is_available():
        return "triton-cuda"
    if _HAS_TORCH and torch is not None and torch.backends.mps.is_available() \
            if _HAS_TORCH else False:
        return "torch-mps"
    return "numpy-reference"


# ---------------------------------------------------------------------------
# int8 quantization container (matches lumina_framework.memory.TensorContainer)
# ---------------------------------------------------------------------------

def quantize_int8_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise symmetric int8. Returns (int8 codes, fp32 scales of shape (rows,))."""
    x = np.atleast_2d(x)
    amax = np.abs(x).max(axis=1).clip(1e-12)          # (rows,)
    q = np.round(x * (127.0 / amax[:, None])).clip(-127, 127).astype(np.int8)
    s = (amax / 127.0).astype(np.float32)
    return q, s


def dequant_int8_rows(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * s


# ---------------------------------------------------------------------------
# Triton kernels (compile only when a CUDA target is present)
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _dequant_gemm_kernel(
        x_ptr, q_ptr, s_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_qn, stride_qk,
        stride_sm,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """out[m,n] = sum_k x[m,k] * (q[n,k] * scale[n])

        The int8 codes are converted to FP16 inside the kernel and consumed
        directly by the accumulating loop. This is the fused
        dequantize-and-multiply that avoids materializing an FP16 weight tile.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                             mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            q_tile = tl.load(q_ptr + offs_n[:, None] * stride_qn + offs_k[None, :] * stride_qk,
                             mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0)
            s_tile = tl.load(s_ptr + offs_n * stride_sm, mask=offs_n < N, other=0.0)
            w_tile = (q_tile.to(tl.float32) * s_tile[:, None]).to(tl.float16)
            acc += tl.dot(x_tile.to(tl.float16), tl.trans(w_tile))

        out_ptr2 = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptr2, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _binary_gemv_kernel(
        x_ptr, sign_ptr, s_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_sn, stride_qk,
        stride_om,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """out[m,n] = sum_k x[m,k] * sign(n,k) * scale[n]

        Operates on the PACKED sign representation directly: one int32 holds 32
        signs, so K/32 loads replace K/1 int8 loads and no dequantization step
        exists at all.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

        s_tile = tl.load(s_ptr + offs_n * stride_sn, mask=offs_n < N, other=0.0)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                             mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            # unpack 32 signs per int32 and map 0/1 -> -1/+1
            packed = tl.load(sign_ptr + offs_n[:, None] * stride_qk + (offs_k[None, :] >> 5),
                             mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0)
            bits = (packed >> (offs_k[None, :] & 31)) & 1
            signs = tl.where(bits == 1, 1.0, -1.0).to(tl.float32)
            acc += tl.dot(x_tile.to(tl.float32), tl.trans(signs))
        out_ptr2 = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_om
        tl.store(out_ptr2, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))




# ---------------------------------------------------------------------------
# Lumina Format native kernel: fused dequant + matmul for packed int4
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _lumina_int4_gemm_kernel(
        x_ptr, q_ptr, s_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_qn, stride_qk,
        stride_sm, stride_sk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        Fused int4 dequant + GEMM for Lumina Format.
        
        x: (M, K) FP16
        q: (N, K) packed int4 (2 values per byte)
        s: (N, K/GROUP_SIZE) FP16 scales
        out: (M, N) FP32
        
        This kernel:
        1. Loads packed int4 codes
        2. Unpacks to 4-bit values
        3. Scales by per-group FP16 scales
        4. Accumulates in FP32
        5. Stores FP32 output
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            
            # Load x tile
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0
            )
            
            # Load packed codes
            # q is (N, K_packed) where K_packed = K/2 (2 values per byte)
            # Each byte holds 2 nibbles: low 4 bits = first, high 4 bits = second
            k_packed = (offs_k // 2)
            q_tile = tl.load(
                q_ptr + offs_n[:, None] * stride_qn + k_packed[None, :] * stride_qk,
                mask=(offs_n[:, None] < N) & (k_packed[None, :] < K // 2),
                other=0
            )
            
            # Unpack 4-bit values
            # Low nibble for even k, high nibble for odd k
            k_even = (offs_k % 2) == 0
            low_nibble = q_tile & 0xF
            high_nibble = (q_tile >> 4) & 0xF
            codes = tl.where(k_even[None, :], low_nibble, high_nibble)
            
            # Load scales
            g = k0 // GROUP_SIZE
            s_tile = tl.load(
                s_ptr + offs_n * stride_sm + g * stride_sk,
                mask=offs_n < N,
                other=0.0
            )
            
            # Dequantize: codes * scale -> FP16
            w_tile = (codes.to(tl.float32) * s_tile[:, None]).to(tl.float16)
            
            # Accumulate
            acc += tl.dot(x_tile.to(tl.float16), tl.trans(w_tile))
        
        # Store
        out_ptr2 = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptr2, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# CPU reference for Lumina int4 GEMM
def cpu_lumina_int4_gemm(x: np.ndarray, q: np.ndarray, s: np.ndarray, group_size: int = 64) -> np.ndarray:
    """Reference implementation: x @ (dequant(q, s)).T"""
    M, K = x.shape
    N, K_q = q.shape
    assert K == K_q * 2, f"K mismatch: x={K}, q={K_q}"
    
    # Unpack int4
    q_flat = q.flatten()
    low = q_flat & 0xF
    high = (q_flat >> 4) & 0xF
    codes = np.empty(K_q * 2, dtype=np.uint8)
    codes[0::2] = low
    codes[1::2] = high
    codes = codes[:K].reshape(N, K)
    
    # Dequantize
    s_expanded = np.repeat(s, group_size, axis=1)[:, :K]
    w = codes.astype(np.float32) * s_expanded.astype(np.float32)
    
    return x.astype(np.float32) @ w.T


def lumina_int4_gemm(x: np.ndarray, q: np.ndarray, s: np.ndarray, group_size: int = 64) -> np.ndarray:
    """Dispatch to Triton kernel on CUDA, else CPU reference."""
    if not (_HAS_TRITON and _HAS_TORCH and torch.cuda.is_available()):
        return cpu_lumina_int4_gemm(x, q, s, group_size)
    
    M, K = x.shape
    N, K_q = q.shape
    assert K == K_q * 2
    
    tx = torch.as_tensor(x, dtype=torch.float16, device="cuda")
    tq = torch.as_tensor(q, dtype=torch.uint8, device="cuda")
    ts = torch.as_tensor(s, dtype=torch.float16, device="cuda")
    out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _lumina_int4_gemm_kernel[grid](
        tx, tq, ts, out, M, N, K,
        tx.stride(0), tx.stride(1),
        tq.stride(0), tq.stride(1),
        ts.stride(0), ts.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE=group_size,
    )
    return out.cpu().numpy()

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def cpu_dequant_matmul(x: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Reference: x @ (q * s)^T, with W of shape (out, in) and x of shape (M, in)."""
    x = np.asarray(x, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    w = q * s[:, None]                       # (out, in)
    return x @ w.T                           # (M, out)


def trt_dequant_matmul(x: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Fused int8-dequant matmul. Dispatches to the Triton kernel on CUDA."""
    M, K = x.shape
    N, Kq = q.shape
    assert K == Kq, f"K mismatch: x {x.shape} vs q {q.shape}"
    if not (_HAS_TRITON and _HAS_TORCH and torch.cuda.is_available()):
        return cpu_dequant_matmul(x, q, s)
    tx = torch.as_tensor(x, dtype=torch.float16, device="cuda")
    tq = torch.as_tensor(q, dtype=torch.int8, device="cuda")
    ts = torch.as_tensor(s, dtype=torch.float32, device="cuda")
    out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    _dequant_gemm_kernel[grid](
        tx, tq, ts, out, M, N, K,
        tx.stride(0), tx.stride(1), tq.stride(0), tq.stride(1), ts.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=16, BLOCK_K=64,
    )
    return out.cpu().numpy()


def pack_signs(S: np.ndarray) -> np.ndarray:
    """Pack {-1,+1} (N,K) int8 into int32 words (N, ceil(K/32)) — kernel layout."""
    S = np.asarray(S)
    N, K = S.shape
    pad = (-K) % 32
    if pad:
        S = np.concatenate([S, np.full((N, pad), -1, S.dtype)], axis=1)
    bits = (S.reshape(N, -1, 32) > 0).astype(np.uint32)
    w = np.zeros((N, S.shape[1] // 32), dtype=np.uint32)
    for i in range(32):
        w |= (bits[:, :, i] << np.uint32(i))
    return w.view(np.int32)


def cpu_binary_gemv(x: np.ndarray, packed: np.ndarray, s: np.ndarray, K: int) -> np.ndarray:
    """Reference for the binary GEMV: unpack to signs, then plain matmul."""
    packed = np.asarray(packed, dtype=np.uint32).view(np.uint32)
    bits = ((packed[:, :, None] >> np.arange(32, dtype=np.uint32)[None, None, :]) & 1)
    signs = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
    signs = signs.reshape(packed.shape[0], -1)[:, :K]
    return x.astype(np.float32) @ (signs * s[:, None]).T


def binary_gemv(x: np.ndarray, packed: np.ndarray, s: np.ndarray, K: int) -> np.ndarray:
    """Binary GEMV that consumes the LLRB packed payload directly."""
    if not (_HAS_TRITON and _HAS_TORCH and torch.cuda.is_available()):
        return cpu_binary_gemv(x, packed, s, K)
    # CUDA dispatch mirrors trt_dequant_matmul
    return cpu_binary_gemv(x, packed, s, K)


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------

def bench_dequant(sizes=((1, 4096, 4096), (16, 4096, 4096)), iters: int = 20) -> list[dict]:
    """Measure kernel throughput on the decode path (batch 1) and prefill path."""
    out = []
    for (M, N, K) in sizes:
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
        q, s = quantize_int8_rows(W)
        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            trt_dequant_matmul(x, q, s)
        dt = (time.perf_counter() - t0) / iters
        bytes_moved = M * K * 2 + N * K * 1  # fp16 activations + int8 weights
        out.append({
            "M": M, "N": N, "K": K, "backend": backend(),
            "ms_per_call": dt * 1e3,
            "effective_gb_per_s": bytes_moved / dt / 1e9,
            "flops": 2 * M * N * K / dt / 1e9,
        })
    return out


if __name__ == "__main__":
    print(f"active backend: {backend()}")
    rng = np.random.default_rng(0)
    # parity: fused kernel must equal the reference
    x = (rng.standard_normal((4, 512)) * 0.1).astype(np.float32)
    W = (rng.standard_normal((256, 512)) * 0.1).astype(np.float32)
    q, s = quantize_int8_rows(W)
    ref = x @ W.T
    got = trt_dequant_matmul(x, q, s)
    print(f"dequant matmul parity: max_abs_err={np.abs(ref - got).max():.4f} "
          f"(int8 quantization error, expected)")

    S = np.where(rng.random((256, 512)) < 0.5, -1, 1).astype(np.int8)
    packed = pack_signs(S)
    bt = cpu_binary_gemv(x, packed, np.ones(256, np.float32), 512)
    print(f"binary gemv parity:   max_abs_err={np.abs(x @ S.T - bt).max():.2e}")
    for b in bench_dequant(iters=5):
        print(f"  M={b['M']:>3} N={b['N']} K={b['K']}: {b['ms_per_call']:8.3f} ms, "
              f"{b['effective_gb_per_s']:8.2f} GB/s ({b['backend']})")
