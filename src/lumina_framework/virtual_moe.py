"""Virtual MoE: run DENSE models as MoE for better performance.

MoEfication: split each dense SwiGLU MLP (intermediate dim F) into E expert
groups by activation affinity, route top-k per token, compute only the
routed experts + nothing else changes. Same weights, fewer FLOPs per token:
dense costs 2*D*F multiply-adds per token per layer; virtual-MoE costs
2*D*F*(k/E) plus a tiny (D,E) router matmul.

Clustering (no training): per-neuron signatures = mean |silu(gx)·(ux)| over
calibration inputs; greedy farthest-point centroids + nearest assignment with
a capacity cap for balance. Router row e = mean gate-row direction of its
members (the input direction that lights the cluster). Deterministic.

Honest contract: routing is exact (audited top-k), execution reuses the
tested grouped MoE forward. Quality delta vs dense is REPORTED per layer
(functional error on calibration acts) — on real weights that number
decides whether the speedup is worth it; on random-weight toys only speed +
exactness are claimed.
"""
from __future__ import annotations

import numpy as np


def neuron_signatures(gate: np.ndarray, up: np.ndarray,
                      acts: np.ndarray) -> np.ndarray:
    """acts: (N,D) calibration inputs. Returns (F,) mean |activation| per neuron."""
    g = acts @ gate.T
    u = acts @ up.T
    silu = g / (1.0 + np.exp(-np.clip(g, -40, 40)))
    return np.abs(silu * u).mean(axis=0)


def cluster_neurons(sig_mat: np.ndarray, n_experts: int,
                    cap: float = 1.5) -> np.ndarray:
    """Greedy farthest-point + capacity-capped assignment.

    sig_mat: (F, N) per-neuron signature vectors (or (F,) magnitudes —
    reshaped to (F,1)). Returns (F,) expert ids 0..E-1. Deterministic.
    """
    F = sig_mat.shape[0]
    E = max(1, min(n_experts, F))
    X = sig_mat.reshape(F, -1).astype(np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    centroids = [int(np.argmax(np.linalg.norm(Xn, axis=1)))]
    for _ in range(1, E):
        d = np.min([np.linalg.norm(Xn - Xn[c], axis=1) for c in centroids], axis=0)
        d[centroids] = -1.0
        centroids.append(int(np.argmax(d)))
    C = np.array(centroids)
    assign = np.argmin([[np.linalg.norm(Xn[f] - Xn[c]) for c in C] for f in range(F)], axis=1)
    # capacity cap: overflow members move to their next-nearest open expert
    maxn = int(np.ceil(F / E * cap))
    counts = np.bincount(assign, minlength=E)
    # eliminate empty experts: split the largest cluster's farthest members
    for e in range(E):
        if counts[e] > 0:
            continue
        src = int(np.argmax(counts))
        if counts[src] <= 1:
            break
        members = np.nonzero(assign == src)[0]
        dists = sorted(((float(np.linalg.norm(Xn[f] - Xn[C[src]])), int(f))
                        for f in members), reverse=True)
        move = [f for _, f in dists[:max(1, counts[src] // 2)]]
        for f in move:
            assign[f] = e
        counts[src] -= len(move)
        counts[e] += len(move)
    if np.any(counts > maxn):
        order = np.argsort([np.linalg.norm(Xn[f] - Xn[C[assign[f]]]) for f in range(F)])[::-1]
        for f in order:
            e = assign[f]
            if counts[e] <= maxn:
                continue
            dists = sorted(((float(np.linalg.norm(Xn[f] - Xn[c])), ee)
                            for ee, c in enumerate(C) if counts[ee] < maxn))
            if dists:
                assign[f] = dists[0][1]
                counts[e] -= 1
                counts[dists[0][1]] += 1
    return assign.astype(np.int64)


def fit_router(gate: np.ndarray, up: np.ndarray, assign: np.ndarray,
               acts: np.ndarray, n_experts: int, steps: int = 300,
               lr: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train a linear router (W, b) on calibration acts.

    Labels come from the dense layer itself: per input, the expert whose
    member neurons produce the largest total |output|. Full-batch softmax
    cross-entropy descent, fixed steps/seed — deterministic. Returns
    (W (E,D), b (E,)) in fp32.
    """
    rng = np.random.default_rng(seed)
    # energy features: expert choice depends on |input| magnitude patterns,
    # which are linearly separable where raw zero-mean inputs are not
    # (x and -x share a label but opposite dots). Runtime uses the same.
    X = np.abs(np.asarray(acts, dtype=np.float64))
    N, D = X.shape
    E = n_experts
    g = X @ np.asarray(gate, dtype=np.float64).T
    u = X @ np.asarray(up, dtype=np.float64).T
    silu = g / (1.0 + np.exp(-np.clip(g, -40, 40)))
    mag = np.abs(silu * u)
    # labels: argmax over per-expert output magnitude totals
    totals = np.zeros((N, E))
    for e in range(E):
        members = np.nonzero(assign == e)[0]
        totals[:, e] = mag[:, members].sum(axis=1) if len(members) else -1.0
    labels = totals.argmax(axis=1)
    W = (rng.normal(size=(E, D)) * 0.01)
    b = np.zeros(E)
    for _ in range(steps):
        lg = X @ W.T + b
        lg -= lg.max(axis=1, keepdims=True)
        P = np.exp(lg)
        P /= P.sum(axis=1, keepdims=True)
        Y = np.zeros_like(P)
        Y[np.arange(N), labels] = 1.0
        G = (P - Y) / N
        W -= lr * (G.T @ X)
        b -= lr * G.sum(axis=0)
    return W.astype(np.float32), b.astype(np.float32)


def build_virtual_moe(gate: np.ndarray, up: np.ndarray, down: np.ndarray,
                      acts: np.ndarray, n_experts: int,
                      top_k: int) -> dict:
    """Split one dense SwiGLU layer into grouped experts + router.

    Returns dict with gate_up (E,2Ie,D), down (E,Ie,D), router (E,D),
    assign (F,), top_k, and quality metrics. Ie varies per expert (ragged
    groups are padded to max with zero rows; pad mask recorded).
    """
    F, D = gate.shape
    E = max(1, min(n_experts, F))
    k = max(1, min(top_k, E))
    sig = neuron_signatures(gate, up, acts)
    assign = cluster_neurons(sig, E)
    sizes = np.bincount(assign, minlength=E)
    Ie = int(sizes.max())
    gu = np.zeros((E, 2 * Ie, D), dtype=np.float32)
    dn = np.zeros((E, Ie, D), dtype=np.float32)
    mask = np.zeros((E, Ie), dtype=bool)
    router = np.zeros((E, D), dtype=np.float32)
    for e in range(E):
        members = np.nonzero(assign == e)[0]
        ne = len(members)
        if ne == 0:
            continue
        gu[e, :ne, :] = gate[members]
        gu[e, Ie:Ie + ne, :] = up[members]
        dn[e, :ne, :] = down[:, members].T  # down is (D,F): experts own columns
        mask[e, :ne] = True
        router[e] = gate[members].mean(axis=0)
    router, router_b = fit_router(gate, up, assign, acts, E)
    return {"gate_up": gu, "down": dn, "router": router, "router_b": router_b,
            "router_feat": "abs",
            "mask": mask, "assign": assign, "top_k": k, "n_experts": E, "ie": Ie}


def virtual_moe_quality(gate, up, down, routed: dict,
                        acts: np.ndarray) -> dict:
    """Per-layer functional error: dense output vs routed top-k output."""
    from .qwen_next import moe_grouped_forward
    g = acts @ gate.T
    u = acts @ up.T
    dense = (((g / (1.0 + np.exp(-np.clip(g, -40, 40)))) * u) @ down.T)
    out, info = moe_grouped_forward(acts.astype(np.float32), routed["gate_up"],
                                    routed["down"], routed["router"],
                                    routed["top_k"], weight_mode="unit",
                                    router_b=routed.get("router_b"),
                                    router_feat=routed.get("router_feat", "raw"))
    # zero-pad rows contribute nothing (mask) — verify, don't assume
    denom = float(np.linalg.norm(dense)) or 1.0
    rel = float(np.linalg.norm(dense - out) / denom)
    return {"rel_func_err": rel,
            "experts_used": sorted(set(info["experts"].ravel().tolist()))}


def apply_to_arrays(arrays: dict[str, np.ndarray],
                    acts: dict[str, np.ndarray] | None,
                    n_experts: int, top_k: int,
                    seed: int = 0) -> tuple[dict, dict, dict]:
    """Replace dense MLP triplets with flat grouped experts + fp32 router.

    Matches names `<prefix>.mlp.gate_proj.weight / up_proj / down_proj`
    (any prefix). Grouped matrices are flattened to 2-D
    ((E*2Ie,D) / (E*Ie,D)) so the normal quantizer handles them; moe_meta
    records (E, Ie, top_k) per layer for the runtime reshape. Signatures come
    from calibration acts when present, else SEEDED random probes (recorded).
    Returns (new_arrays, moe_meta, report).
    """
    rng = np.random.default_rng(seed)
    out = dict(arrays)
    meta: dict[str, dict] = {}
    report: dict[str, dict] = {"layers": {}, "signatures": None}
    used_probes = False
    prefixes: set[str] = set()
    for name in arrays:
        if name.endswith(".mlp.gate_proj.weight"):
            prefixes.add(name[: -len("gate_proj.weight")])
    for pfx in sorted(prefixes):
        gn, un, dn_ = pfx + "gate_proj.weight", pfx + "up_proj.weight", pfx + "down_proj.weight"
        if not (gn in out and un in out and dn_ in out):
            continue
        gate, up, down = (np.asarray(out[gn], dtype=np.float32),
                          np.asarray(out[un], dtype=np.float32),
                          np.asarray(out[dn_], dtype=np.float32))
        if gate.ndim != 2 or gate.shape != up.shape:
            continue
        F, D = gate.shape
        if F < 2 * max(2, n_experts):
            continue
        cal = None
        if acts:
            for cand in (gn, pfx + "input_layernorm.weight"):
                if cand in acts and np.asarray(acts[cand]).ndim == 2:
                    a = np.asarray(acts[cand], dtype=np.float32)
                    cal = a if a.shape[1] == D else None
                    break
        if cal is None:
            cal = (rng.normal(size=(256, D)) * 0.5).astype(np.float32)
            used_probes = True
        routed = build_virtual_moe(gate, up, down, cal, n_experts, top_k)
        E, Ie = routed["n_experts"], routed["ie"]
        out[pfx + "experts.gate_up_proj"] = routed["gate_up"].reshape(E * 2 * Ie, D)
        out[pfx + "experts.down_proj"] = routed["down"].reshape(E * Ie, D)
        out[pfx + "gate.weight"] = routed["router"]  # fp32 router (kept raw below)
        out[pfx + "gate_bias"] = np.asarray(routed["router_b"], dtype=np.float32)
        for n in (gn, un, dn_):
            del out[n]
        lyr = pfx.rstrip(".").split(".")[-2] if ".layers." in pfx or ".layers" in pfx else pfx
        meta[pfx] = {"experts": E, "ie": Ie, "top_k": routed["top_k"],
                     "router_feat": routed.get("router_feat", "raw"),
                     "layer": lyr}
        report["layers"][pfx] = {"experts": E, "top_k": routed["top_k"],
                                 **virtual_moe_quality(gate, up, down, routed, cal)}
    report["signatures"] = "random-probes" if used_probes else "calibration-acts"
    report["router_precision"] = "fp32-raw"
    return out, meta, report
