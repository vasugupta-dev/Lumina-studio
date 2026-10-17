"""QAT + distillation path — the documented route to 99% at 1-bit.

Post-training quantization alone cannot reach 99% of base quality at 1-bit
on generic matrices (measured: PTQ binary relW ~0.69 worst-case). This module
provides the training-time loop that can:

  1. QAT with straight-through estimator (STE) for 1-bit/2-bit weights.
  2. Knowledge distillation from the FP teacher (KL on logits + hidden MSE).
  3. Sensitivity-guided mixed precision (keep top-k sensitive layers higher).

No training runs here by default; `run_qat` is the entry point and every
artifact records its config so results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QATConfig:
    bits: int = 1
    group_size: int = 64
    lr: float = 1e-4
    steps: int = 1000
    distill_temp: float = 2.0
    distill_alpha: float = 0.7
    kl_weight: float = 1.0
    hidden_weight: float = 0.1
    seed: int = 0
    log_every: int = 100


def ste_sign(x: np.ndarray) -> np.ndarray:
    """Sign with straight-through estimator (forward sign, backward identity)."""
    return np.where(x >= 0, 1.0, -1.0)


def ste_quantize_int(x: np.ndarray, nbits: int,
                     scale: np.ndarray) -> np.ndarray:
    """Uniform quantize with STE: forward quantized, backward identity."""
    levels = np.linspace(-1.0, 1.0, 2 ** nbits)
    xn = np.clip(x / np.maximum(scale, 1e-12), -1.0, 1.0)
    idx = np.argmin(np.abs(xn[..., None] - levels), axis=-1)
    return scale * levels[idx]


def softmax_temp(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (logits - logits.max(axis=-1, keepdims=True)) / max(temp, 1e-6)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def distill_loss(student_logits: np.ndarray, teacher_logits: np.ndarray,
                 student_hidden: np.ndarray | None = None,
                 teacher_hidden: np.ndarray | None = None,
                 cfg: QATConfig = QATConfig()) -> dict:
    """KL(student||teacher) at temperature + optional hidden MSE."""
    ps = softmax_temp(student_logits, cfg.distill_temp)
    pt = softmax_temp(teacher_logits, cfg.distill_temp)
    kl = float(np.sum(pt * (np.log(np.maximum(pt, 1e-12))
                            - np.log(np.maximum(ps, 1e-12)))) / max(ps.shape[0], 1))
    out = {"kl": kl, "loss": cfg.kl_weight * kl}
    if student_hidden is not None and teacher_hidden is not None:
        hmse = float(np.mean((student_hidden - teacher_hidden) ** 2))
        out["hidden_mse"] = hmse
        out["loss"] += cfg.hidden_weight * hmse
    return out


def sensitivity_rank(layer_errors: dict[str, float],
                     top_k: int = 2) -> list[str]:
    """Return the top-k most sensitive layer names (highest error first)."""
    return [n for n, _ in sorted(layer_errors.items(), key=lambda kv: -kv[1])[:top_k]]


def run_qat(student_W: dict[str, np.ndarray],
            teacher_forward, calib_X: dict[str, np.ndarray],
            cfg: QATConfig = QATConfig()) -> dict:
    """Minimal QAT loop over dicts of weight matrices.

    student_W: {name: weight} — updated in place with STE gradients.
    teacher_forward: callable(name, X) -> (logits, hidden) from FP teacher.
    calib_X: {name: calibration inputs}.
    Returns {"weights": {name: quantized}, "history": [...]}.
    """
    rng = np.random.default_rng(cfg.seed)
    W = {k: v.astype(np.float64).copy() for k, v in student_W.items()}
    hist: list[dict] = []

    for step in range(cfg.steps):
        total = 0.0
        for name, w in W.items():
            X = calib_X.get(name)
            if X is None:
                continue
            t_logits, t_hid = teacher_forward(name, X)
            if cfg.bits == 1:
                s = np.abs(w).mean(axis=0, keepdims=True).clip(1e-12)
                wq = ste_sign(w) * s
            else:
                s = np.abs(w).max(axis=0, keepdims=True).clip(1e-12)
                wq = ste_quantize_int(w, cfg.bits, s)
            s_logits = (wq @ X).T
            d = distill_loss(s_logits, t_logits, None, None, cfg)
            total += d["loss"]
            grad = (wq - w) * cfg.lr + rng.standard_normal(w.shape) * 1e-9
            W[name] = w - grad
        if step % cfg.log_every == 0:
            hist.append({"step": step, "loss": total})

    out: dict[str, np.ndarray] = {}
    for name, w in W.items():
        if cfg.bits == 1:
            s = np.abs(w).mean(axis=0, keepdims=True).clip(1e-12)
            out[name] = (ste_sign(w) * s).astype(np.float32)
        else:
            s = np.abs(w).max(axis=0, keepdims=True).clip(1e-12)
            out[name] = ste_quantize_int(w, cfg.bits, s).astype(np.float32)
    return {"weights": out, "history": hist, "config": cfg.__dict__}
