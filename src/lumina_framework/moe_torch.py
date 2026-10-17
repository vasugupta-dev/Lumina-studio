"""Torch mirror of the MoE runtime — the real GPU path.

Semantics MUST match moe.MoELayer.forward exactly:
  sigmoid router (no aux loss) -> top-k -> normalize combine weights over k,
  shared expert always active, SwiGLU routed experts (gate/up/down),
  residual handled by the caller. Verified against the numpy reference by
  tests/test_moe_torch_parity.py (CPU torch, tiny config, phone-safe);
  GPU speed is measured on the Kaggle notebook (Phase 6), never claimed here.
"""
from __future__ import annotations

import numpy as np
import torch


def _sig(x):
    return torch.sigmoid(x)


class TorchMoELayer(torch.nn.Module):
    def __init__(self, ref, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.top_k = int(ref.top_k)
        self.n_expert = int(ref.n_expert)
        self.scaling = float(ref.router.scaling)
        self.router_w = torch.as_tensor(
            np.asarray(ref.router.weight), dtype=dtype, device=device)
        self.shared_gate = torch.as_tensor(
            ref.shared.containers["gate"].dequantize(), dtype=dtype, device=device)
        self.shared_up = torch.as_tensor(
            ref.shared.containers["up"].dequantize(), dtype=dtype, device=device)
        self.shared_down = torch.as_tensor(
            ref.shared.containers["down"].dequantize(), dtype=dtype, device=device)
        self.gates = torch.nn.ParameterList()
        self.ups = torch.nn.ParameterList()
        self.downs = torch.nn.ParameterList()
        for e in ref.experts:
            self.gates.append(torch.nn.Parameter(torch.as_tensor(
                e.containers["gate"].dequantize(), dtype=dtype, device=device),
                requires_grad=False))
            self.ups.append(torch.nn.Parameter(torch.as_tensor(
                e.containers["up"].dequantize(), dtype=dtype, device=device),
                requires_grad=False))
            self.downs.append(torch.nn.Parameter(torch.as_tensor(
                e.containers["down"].dequantize(), dtype=dtype, device=device),
                requires_grad=False))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M, D) -> (M, D). Matches numpy MoELayer.forward routing exactly."""
        s = _sig(x @ self.router_w.T) * self.scaling        # (M, E)
        k = min(self.top_k, self.n_expert)
        idx = torch.topk(s, k, dim=1).indices                 # (M, k)
        w = torch.gather(s, 1, idx)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)    # normalize over k
        # shared expert (always)
        sh = torch.nn.functional.silu(x @ self.shared_gate.T) * (x @ self.shared_up.T)
        out = sh @ self.shared_down.T
        # routed experts, grouped per expert (like the numpy engine path)
        idx_np = idx.cpu().numpy()
        for e in range(self.n_expert):
            rows, cols = np.nonzero(idx_np == e)
            if rows.size == 0:
                continue
            xe = x[torch.as_tensor(rows, device=x.device, dtype=torch.long)]
            h = torch.nn.functional.silu(xe @ self.gates[e].T) * (xe @ self.ups[e].T)
            ye = h @ self.downs[e].T
            wc = w[torch.as_tensor(rows, device=x.device, dtype=torch.long),
                   torch.as_tensor(cols, device=x.device, dtype=torch.long)].unsqueeze(1)
            out.index_add_(0, torch.as_tensor(rows, device=x.device, dtype=torch.long),
                           ye * wc)
        return out


class TorchStub(torch.nn.Module):
    """Full stub stack (embed -> N MoE layers -> head) on one torch device."""

    def __init__(self, ref_model, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.embed = torch.as_tensor(np.asarray(ref_model.embed), dtype=dtype, device=device)
        self.head = torch.as_tensor(np.asarray(ref_model.head), dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList(
            [TorchMoELayer(ly, device=device, dtype=dtype)
             for ly in ref_model.layers])
        self.device = device

    @torch.no_grad()
    def forward_hidden(self, h: torch.Tensor) -> torch.Tensor:
        for ly in self.layers:
            h = h + ly.forward(h)
        return h

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed[tokens]
        return self.forward_hidden(h) @ self.head.T

    @torch.no_grad()
    def greedy(self, prompt, n_tokens: int):
        """tokens: 1-D list/ndarray. Returns generated token list."""
        toks = [int(t) for t in np.asarray(prompt).ravel()]
        gen = []
        cur = torch.as_tensor(toks, dtype=torch.long, device=self.device)
        h = self.embed[cur].mean(dim=0, keepdim=True)
        for _ in range(n_tokens):
            lg = self.forward_hidden(h) @ self.head.T
            nxt = int(lg[0].argmax().item())
            gen.append(nxt)
            h = self.embed[torch.as_tensor([nxt], device=self.device)]
        return gen
