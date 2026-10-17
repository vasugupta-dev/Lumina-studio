"""Real QAT proof run on the reference model — distillation closes what PTQ can't.

Student keeps latent FP weights; forward passes through a STE quantizer
(binary or int-N); loss = KL(student||teacher)/T + CE; Adam on latents.
Teacher = frozen FP base. Small model => phone-feasible. All bytes counted
afterwards by packing the final latents through the normal codec.
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)


def ste_quantize_int(w, bits: int = 2, group: int = 64):
    import torch
    o, i = w.shape
    n_groups = (i + group - 1) // group
    lv = torch.linspace(-1, 1, 2 ** bits)
    out = torch.empty_like(w)
    for g in range(n_groups):
        sl = slice(g * group, min(i, (g + 1) * group))
        amax = w[:, sl].abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
        wn = (w[:, sl] / amax).clamp(-1, 1)
        idx = (wn.unsqueeze(-1) - lv.to(w.device)).abs().argmin(-1)
        q = amax * lv.to(w.device)[idx]
        out[:, sl] = w[:, sl] + (q - w[:, sl]).detach()  # STE
    return out


def ste_quantize_binary(w):
    import torch
    s = w.abs().mean(dim=0, keepdim=True).clamp_min(1e-12)
    q = torch.where(w >= 0, s, -s)
    return w + (q - w).detach()  # STE


def main() -> int:
    import argparse
    import torch
    from mini_lm import MiniLM, evaluate
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--ramp", type=int, default=0,
                    help="gradual-quantization steps: W_eff = l*Wq + (1-l)*W_fp "
                         "with l annealed 0->1 over --ramp steps (0 = off)")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine LR decay from --lr to lr_min over all steps")
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "qat.json"))
    a = ap.parse_args()

    ck = torch.load(os.path.join(ROOT, "artifacts", "reference.pt"),
                     map_location="cpu", weights_only=False)
    teacher = MiniLM()
    teacher.load_state_dict(ck["state"])
    teacher.eval()
    seq = ck["seq"]
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = MiniLM()
    student.load_state_dict(ck["state"])
    student.train()
    # STE discipline: module weights are SCRATCH (leaf Parameters whose .data
    # we overwrite with quantized values each step). Autograd gives dL/dW at
    # the quantized point; we hand those grads to the latent FP variables,
    # which is exactly the straight-through estimator. Optimizer steps latents
    # only; modules are never stepped.
    latents = {n: m.weight.detach().clone().requires_grad_(True)
               for n, m in student.named_modules()
               if isinstance(m, torch.nn.Linear)}
    opt = torch.optim.Adam(list(latents.values()), lr=a.lr)
    sched = None
    if a.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(a.steps, 1), eta_min=a.lr_min)

    # doctrine: the output head stays full precision (trained, not quantized).
    # Every production 1-bit recipe (incl. BitNet) keeps embed/head in fp16.
    skip = {n for n in latents if n == "head" or n.endswith(".head")}
    lin_names = [n for n in latents if n not in skip]
    base_loss = evaluate(teacher, seq)
    hist = []
    x_all = torch.from_numpy(seq[:, :-1])
    y_all = torch.from_numpy(seq[:, 1:])
    n = x_all.shape[0]
    t0 = time.time()

    for step in range(a.steps):
        idx = np.random.default_rng(step).choice(n, size=min(8, n),
                                                 replace=False)
        xb, yb = x_all[idx], y_all[idx]
        lam = 1.0
        if a.ramp > 0:
            lam = min(1.0, (step + 1) / float(a.ramp))
        with torch.no_grad():
            tout = teacher(xb)
            t_logits = tout[0] if isinstance(tout, tuple) else tout
            mods = dict(student.named_modules())
            for name in lin_names:
                w = latents[name]
                wq = ste_quantize_binary(w) if a.bits == 1 \
                    else ste_quantize_int(w, a.bits)
                w_eff = wq if lam >= 1.0 else (lam * wq + (1.0 - lam) * w)
                mods[name].weight.data.copy_(w_eff.detach())
            for name in skip:
                mods[name].weight.data.copy_(latents[name].detach())
        opt.zero_grad()
        student.zero_grad()
        sout = student(xb)
        logits = sout[0] if isinstance(sout, tuple) else sout
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits / a.temp, -1),
            torch.nn.functional.softmax(t_logits / a.temp, -1),
            reduction="batchmean") * a.temp * a.temp
        loss = 0.3 * ce + 0.7 * kl
        loss.backward()
        for name in lin_names:
            g = dict(student.named_modules())[name].weight.grad
            if g is not None:
                latents[name].grad = g.clone()
        opt.step()
        if sched is not None:
            sched.step()
        if step % 50 == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"  step {step}: loss={loss.item():.4f} lam={lam:.2f} "
                  f"lr={lr_now:.2e}", flush=True)
            hist.append({"step": step, "loss": float(loss.item()),
                          "lam": round(lam, 3), "lr": lr_now})
    # bake final quantized weights into the student for eval/packing
    with torch.no_grad():
        for name in lin_names:
            w = latents[name]
            wq = ste_quantize_binary(w) if a.bits == 1 \
                else ste_quantize_int(w, a.bits)
            dict(student.named_modules())[name].weight.data.copy_(wq.detach())
    student.eval()
    print(f"done {a.steps} steps in {time.time()-t0:.0f}s")
    qloss = evaluate(student, seq)
    print(f"  base={base_loss:.4f} qat-quantized={qloss:.4f} "
          f"rel={(qloss-base_loss)/base_loss*100:+.2f}%")
    # honest bpp: pack the final quantized weights through the real codec
    import numpy as _np
    tot_bits = 0; tot_params = 0
    from lumina_format.codecs.ptq_ladder import quantize_int, quantize_binary
    _mods = dict(student.named_modules())
    for name in lin_names:
        wq_np = _mods[name].weight.detach().cpu().numpy()
        tot_params += wq_np.size
        if a.bits == 1:
            _q = quantize_binary(wq_np, None, 64)
        else:
            _q = quantize_int(wq_np, a.bits, None, 64)
        tot_bits += _q.payload_bytes() * 8
    for name in skip:
        _w = _mods[name].weight.detach().cpu().numpy()
        tot_params += _w.size
        tot_bits += _w.size * 16  # fp16 head, fully counted
    bpp = tot_bits / max(tot_params, 1)
    print(f"  honest bpp (all bytes counted, head fp16): {bpp:.3f}")
    with open(a.out, "w") as f:
        import json
        json.dump({"bits": a.bits, "steps": a.steps, "base_loss": base_loss,
                   "qat_loss": qloss,
                   "rel_pct": (qloss - base_loss) / base_loss * 100.0,
                   "honest_bpp": round(bpp, 3),
                   "recipe": {"ramp": a.ramp, "cosine": a.cosine,
                              "lr": a.lr, "lr_min": a.lr_min, "temp": a.temp},
                   "history": hist}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
