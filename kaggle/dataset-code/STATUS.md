# Lumina Studio — Target Status (honest, measured)

## Block-GPTQ fast path (format speed — the thing that killed Kaggle v10)
- Root cause of the dead v10 kernel: per-column `np.outer` error feedback with
  act-order fancy indexing — O(o·i²) interpreted updates; hours per big matrix
  on Kaggle's 2 CPUs, kernel died grinding it (3.5h silent, hard OOM-kill).
- Fix: official GPTQ block scheme — permute once, sequential intra-block
  feedback, ONE BLAS matmul per block for cross-block feedback.
- Verified (tests/test_gptq_block_parity.py, phone): BIT-EXACT code parity vs
  the reference loop (mismatch 0.0 across 4/2-bit, gs 32/64, act-order on/off,
  96² and 256²), binary path sign-exact, 512² int4 speedup 5.4× (0.34s vs
  1.85s) — gap widens with matrix size (14B MoE layers ~100× bigger).
- Dead code removed: gptq_mixed (never called, latent NameError).
- Label: **Implemented and verified** (parity phone-verified; real-model
  numbers pending Kaggle v12).

## ChatModel fast path (framework decode on real weights)
- Flaw found: `_layer` cast every projection weight fp16→fp32 EVERY token
  (~model-size allocation churn per token); also chat.py was missing
  `import json` entirely — ChatModel crashed on config load (latent NameError
  masked by the app's error bubble).
- Fix: `fast=True` converts each weight to transposed fp32 ONCE, lazily, and
  releases the fp16 original (bounded memory); default stays False
  (phone-safe fp16). `import json` restored.
- Verified (tests/test_chat_fast_parity.py, tiny fixture): identical greedy
  tokens, stable across regenerations, fp16 originals released.
- Fake-CUDA decode path removed (it round-tripped CPU→GPU→CPU per token —
  slower than pure CPU, misleading "fast path" label).
- Label: **Implemented and verified (parity)**. Real-model tok/s measured on
  Kaggle v12 Phase 5.

## QAT v2 recipe (ramp + cosine) — firewall broken on the reference rig
- Recipe upgrades in scripts/qat_run.py: gradual-quantization ramp (W_eff =
  l*Wq + (1-l)*W_fp annealed over --ramp steps; kills the init crater — step-0
  loss 897 -> 1.78), cosine LR (--cosine, eta_min 1e-5), longer budgets,
  honest-bpp accounting of the final packed weights (head fp16 counted).
- 2-bit, 1600 steps, ramp 200, cosine lr 2e-4: end-to-end loss +1.22% @ 2.603
  honest bpp (base 2.4510 -> 2.4810). Previous 2-bit recipe: +68.5%.
  That is 56x better and UNDER the 5% line — at lower bpp than the PTQ
  int2+crits rung (+1.32% @ 3.87 bpp). All bytes counted, no criticals.
- Label: **Implemented and verified (reference rig; small-rig caveat:
  percentages on the 1.43M rig do not transfer as-is to production models)**.
  Real-model QAT stays a Kaggle/notebook task.
- 1-bit v2 (2400 steps, ramp 300, cosine): +5.33% @ 1.638 honest bpp
  (was +10.34% @ 800 steps). Beats the old PTQ 2.81-bpp rung (+5.60%) by
  1.2 bpp at equal-or-better quality — the measured PTQ firewall moved.
- 1-bit v3 (4000 steps, ramp 400, lr 3e-4, cosine): **+4.07% @ 1.638 honest
  bpp — UNDER the 5% line at 1.6 bpp.** The measured <5% frontier moved
  from 3.87 bpp (PTQ int2 + int8 criticals) to 1.638 bpp (pure 1-bit bulk,
  no criticals) on the reference rig. Reference-rig caveat stands: real-model
  proof is Kaggle Phase 7 (QAT v2 recipe, 0.5B).

## Kaggle doctrine eval
- v10: DIED — slow per-column GPTQ (root cause above).
- v11: failed in 22s — kernel attached while the dataset version was still
  processing ("codec unavailable"). Loader hardened with retries + forced
  re-download.
- v12: pushed with block-GPTQ codec, incremental results.json after EVERY
  rung (a death never loses data), per-10-layer progress prints, per-rung
  quant timing, parity self-test cell, Phase 5 framework speed on real 0.5B
  weights. Status: RUNNING.
- Label: **In flight — no real-model numbers claimed until results.json lands.**

## Framework speed target
- Target: 50–70 tok/s raw at 4-bit (target model class); >90 tok/s for
  small-class (Qwen ~0.5–0.8B) CPU-only in this environment.
- Measured (proxy StubModel, CPU-only, int8 containers + numba + cached dequant):
  - tiny pinned: ~2890 tok/s
  - Qwen3.8B proxy (4L/8E/d768/topk2): ~100 tok/s
  - heavy offload / moderate / fully-pinned small configs: 300+ / 540+ / 1400+ tok/s
- Label: **Implemented and verified (proxy)**. Real-weight verification on
  Qwen/GLM deferred (needs desktop RAM/GPU; phone cannot hold the weights).

## Format quality target
- Target: 4-bit ~0% loss; 1-bit 99% of base.
- Measured (PTQ: GPTQ act-order + MSE scales + sequential calibration):
  - per-matrix int4 relW ~0.12 (worst-case Gaussian); binary ~0.69
  - end-to-end reference model int4: +15% (documented in artifacts)
- 1-bit 99%: **requires QAT/distillation (documented)**. `qat_distill.py`
  implements STE + KL distillation + sensitivity ranking; training run deferred.
- Label: **Implemented, partially verified**. 4-bit near-lossless and 1-bit 99%
  on real models are **deferred with documented reason** (need training runs).

## Format speed target (beat EXL3 / TensorRT-LLM)
- `lumina_int4_gemm` Triton fused kernel written; CPU reference verified.
- Label: **Compatible in principle, not tested** (no CUDA on this phone).

## Borrowed-kernel features (fused.py — all parity-verified on CPU)
- llama.cpp: Q8 repack-once + integer dot (Q4_0 parity 1.2e-07; container
  GEMV rel-err 0.0047 repack noise); decode fast path gated at K>=512
  (measured: 25x faster than BLAS at K=1024, BLAS wins below 512).
- EXL3: int-dot over packed 3-bit codes + superblock scales (parity verified).
- TRT-LLM: exact E4M3 codec (roundtrip 2.3%, specials exact) + FP8-blockwise
  pipeline + paged-KV block tables (append/get verified).
- Speculative: EAGLE draft head + static/dynamic trees + parallel streams +
  DeepSeek-style MTP stack; PLD+verify proven EXACT-equal to greedy decode.
- Label: Implemented and verified (CPU reference). GPU (Triton/CUTLASS) port
  is shape-compatible by construction, not yet executed.

## Low-bpw density frontier (lowbpw.py — measured on 128x128 matrices)
- Entropy (canonical Huffman): -49.6% code bytes on peaked dists, +1.2% worst
  case; exact roundtrip. Saves ~0.27 bpp at int2.
- Salient fp16 corrections (CSR, byte-counted): on heavy-tailed weights
  (realistic), 1-bit+1%salient relW 0.836->0.680 at 1.61bpp; 1-bit+2%salient
  0.628 at 2.09bpp — BEATS pure int2 (0.952) at matched bytes.
- On pure Gaussian (pessimistic, no outliers) int2 still dominates; the
  salient lever is distribution-dependent — stated, not hidden.
- Label: Implemented and verified (synthetic matrices). Real-weight frontier
  is the deferred run.

## Code tree + MoE density (codetree.py, lowbpw.pack_moe_shared_gain)
- Gain-shape residual binary tree (GSRQ-grounded): strictly +/-1 splits
  (purity asserted), best-first expansion, Huffman leaves. Measured
  (heavy-tail 128x128): 1.01bpp/0.76, 2.02bpp/0.60, 3.04bpp/0.53, 4.09/0.48.
- MoE shared gain: one gain/group/layer across experts. Measured (4 experts):
  bpp 1.125->1.000 (-11% bytes) at relW 0.7509->0.7569 (+0.8%).
- Binary rung is pure 1-bit (assert_signed enforced); zero-level ladders exist
  only on the 3-bit EXL3 rung (declared, not the 1-bit path).
- Label: Implemented and verified (synthetic). No PEAGLE spec exists publicly
  (arXiv 0, GH 0; sole HF hit is an unrelated 2024 merge name).

## Enhancement sweep for extreme targets (measured, heavy-tail 128x128)
- int3 group 64->32: relW 0.49->0.39 (real win, +0.125 bpp cost).
- Learned Lloyd-Max grids: NO gain (0.39->0.42) — uniform + MSE scales wins.
- Per-row "per-channel" scales: HARMFUL (0.78) — coarser than groups; dropped.
- LLRB sub-1-bit: 1.0bpw/0.56-0.72, 0.5/0.69-0.86, 0.25/0.75-0.95, 0.1/~1.0.
- VERDICT vs asked targets (99.5% @1bpw, lossless @3bit, 95% @0.1bpw,
  98% @0.5bpw, all PTQ-only): NOT REACHABLE — by measurement AND by
  rate-distortion (0.1bpw = 320:1; top components alone exceed that budget).
  Best measured PTQ: int4 relW ~0.17-0.22, 1-bit ~0.63-0.69, 0.5bpw ~0.69.
  Closing the gap needs training (QAT/distill, module ready) — physics, not effort.

## Critical-matrix doctrine (END-TO-END quality, reference model, base=2.451)
- Attribution: every body layer alone <=0.04%, HEAD alone +14.3%. Earlier
  "escapes don't help" was a test-script bug (name mismatch, escape never
  applied) — corrected.
- Doctrine = sequential GPTQ + int8 criticals (head + early block), bytes counted:
  - int4 bulk, trim crits: +0.42% @ 4.85bpp — TARGET PASS (<=0.5%)
  - int3 bulk + crits: +0.41% @ 4.59bpp — PASS (near-lossless 3-bit)
  - int2 bulk + crits: +1.32% @ 3.87bpp — close
  - binary bulk + crits: +66.5% @ 3.06bpp
  - binary + 2% salient + crits: +33.8% @ 3.14bpp (salient HALVED damage)
  - int2 + salient: +4.5% (post-hoc salient disturbs GPTQ comp — SpQR note)
- QAT proof: scripts/qat_run.py (correct in-graph STE via scratch-copy
  discipline); 2-bit/400-step run in progress (artifacts/qat_2bit.json).
- Standing verdict: PTQ-only 1-bit 99.5% and sub-0.5bpw 95-98% remain
  out of reach; doctrine + training is the path. Quality measured
  end-to-end, never by relW alone.

## Honest-bpp ladder, <5% goal (END-TO-END, base 2.451, all bytes counted)
| honest bpp | config | rel loss | <5% |
|---|---|---|---|
| 5.23 | int4 + head/blk0 int8 | +0.07% | YES |
| 4.85 | int4 + head/blk0-partial int8 | +0.42% | YES |
| 4.80 | binary+2%sal + head/blk0/blk1 int8 | +3.94% | YES (1-bit bulk) |
| 4.59 | int3 + head/blk0 int8 | +0.41% | YES |
| 3.87 | int2 + head/blk0 int8 | +1.32% | YES |
| 3.21 | int2 + head/blk0-partial | +10.75% | no |
| 2.81 | int2 + head-i8/blk0-i4 | +5.60% | no (just over) |
| 2.18 | binary + head-i8/blk0-i4 | +35.6% | no (collapse) |
| ~2.2 | QAT 1-bit 800 steps | +10.34% | no (improving, unflattened? flattened) |
- FIREWALL: below ~3.8 honest bpp PTQ-only cannot hold <5%: criticals
  (head+blk0 int8 ~= 2.3bpp) are mandatory; under-protection collapses.
  Requested 1.58/1.0/0.5/0.1 rungs need QAT (in flight) or bigger models
  where bulk damage dilutes — stated as hypothesis, not claim.
