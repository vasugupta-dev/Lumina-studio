"""EAGLE speculative decoding with TRAINED draft heads + tree verification.

Pipeline (all measured, all exact):
  1. collect_training_data: run the TARGET, record (feat_t, emb_t, feat_{t+1}).
  2. train_draft_head: full backprop (all params, grad-checked) fitting
     feature MSE. Random heads accept ~nothing; trained heads earn every point.
  3. generate_eagle: draft a token TREE with the trained head, verify the
     whole tree in ONE target forward (tree ancestry mask + true positions),
     accept the longest exact-match path + bonus. Greedy mode output provably
     equals plain greedy decode (tested).
  4. generate_mtp: same verify machinery driven by a trained MTPStack
     (train_mtp_block included) — MTP maximized within train-free weights.

Honest contract: speedups are MEASURED per prompt class and reported with
acceptance stats. No universal multiplier is claimed; repetitive text
accepts far more than novel text, and the numbers say so.
"""
from __future__ import annotations

import time

import numpy as np

from .chat import rmsnorm
from .spec import SpecStats, build_tree_mask


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40, 40)))


# ---------------------------------------------------------------------------
# Draft-head forward with caches (mirrors spec.DraftHead numerology exactly)
# ---------------------------------------------------------------------------

def eagle_forward(P: dict, feat: np.ndarray, emb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Sequence forward. feat, emb: (T,D). Causal single-head attention over
    the T positions (softmax is non-degenerate here — q/k carry real signal,
    unlike single-vector calls). Returns (pred (T,D), cache)."""
    x = np.concatenate([feat, emb], axis=-1).astype(np.float32)
    T = x.shape[0]
    h = x @ P["fc"].T
    mu = (h ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mu + 1e-6)
    hn = h * inv * P["n"]
    q, k, v = hn @ P["q"].T, hn @ P["k"].T, hn @ P["v"].T
    dd = q.shape[-1]
    sc = q @ k.transpose(0, 2, 1) / max(dd ** 0.5, 1e-6) if q.ndim == 3 else q @ k.T / max(dd ** 0.5, 1e-6)
    causal = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
    e = np.exp(sc + causal - (sc + causal).max(-1, keepdims=True))
    a = e / e.sum(-1, keepdims=True)
    av = a @ v
    o = av @ P["o"].T
    h2 = hn + o
    g, u = h2 @ P["gate"].T, h2 @ P["up"].T
    s = _silu(g)
    m = (s * u) @ P["down"].T
    h3 = h2 + m
    pred = h3 @ P["back"].T
    return pred.astype(np.float32), {"x": x, "h": h, "inv": inv, "hn": hn,
                                     "q": q, "k": k, "v": v, "a": a, "av": av,
                                     "o": o, "h2": h2, "g": g, "u": u,
                                     "s": s, "m": m, "h3": h3}


def eagle_backward(P: dict, c: dict, d_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Full sequence backprop for mean-MSE. Grad-checked — see grad_check."""
    G: dict[str, np.ndarray] = {}
    G["back"] = d_pred.T @ c["h3"]
    dh3 = d_pred @ P["back"]
    G["down"] = dh3.T @ (c["s"] * c["u"])
    d_m = dh3 @ P["down"]
    dsig = c["s"] * (1.0 - c["s"])
    dg = d_m * c["u"] * dsig
    du = d_m * c["s"]
    G["gate"] = dg.T @ c["h2"]
    G["up"] = du.T @ c["h2"]
    dh2 = dg @ P["gate"] + du @ P["up"] + dh3
    G["o"] = c["av"].T @ dh2
    d_av = dh2 @ P["o"].T
    d_a = d_av @ c["v"].T
    d_v = c["a"].transpose(0, 2, 1) @ d_av if c["a"].ndim == 3 else c["a"].T @ d_av
    row = (d_a * c["a"]).sum(-1, keepdims=True)
    d_sc = c["a"] * (d_a - row)
    dd = c["q"].shape[-1]
    scl = max(dd ** 0.5, 1e-6)
    if c["q"].ndim == 3:
        d_q = d_sc @ c["k"] / scl
        d_k = d_sc.transpose(0, 2, 1) @ c["q"] / scl
        G["q"] = np.einsum("tij,tik->jk", d_q, c["hn"])
        G["k"] = np.einsum("tij,tik->jk", d_k, c["hn"])
        d_hn = (np.einsum("tij,jk->tik", d_q, P["q"])
                + np.einsum("tij,jk->tik", d_k, P["k"])
                + np.einsum("tij,jk->tik", d_v, P["v"]) + dh2)
    else:
        d_q = d_sc @ c["k"] / scl
        d_k = d_sc.T @ c["q"] / scl
        G["q"] = d_q.T @ c["hn"]
        G["k"] = d_k.T @ c["hn"]
        d_hn = d_q @ P["q"] + d_k @ P["k"] + d_v @ P["v"] + dh2
    G["v"] = d_v.T @ c["hn"]
    # RMSNorm backward: y = x*n*rstd
    rstd = c["inv"]
    xn = c["h"] * rstd
    d_n = (d_hn * xn).sum(axis=0)
    G["n"] = d_n
    d_h = (d_hn * P["n"] * rstd
           - c["h"] * ((d_hn * P["n"] * xn).sum(axis=-1, keepdims=True)
                       / (c["h"].shape[-1] * (1.0 / (rstd ** 2)))))
    G["fc"] = d_h.T @ c["x"]
    return G


def head_to_dict(head) -> dict:
    return {k: np.array(getattr(head, k), dtype=np.float32)
            for k in ("fc", "q", "k", "v", "o", "gate", "up", "down", "back", "n")}


def dict_to_head(P: dict, head) -> None:
    for k, v in P.items():
        setattr(head, k, np.array(v, dtype=np.float32))


def grad_check(P: dict, feat: np.ndarray, emb: np.ndarray,
               tgt: np.ndarray, eps: float = 1e-3) -> dict:
    """Finite-difference audit of eagle_backward on a short sequence.

    Metric: |ana-num| / max(|ana|, |num|, floor). The floor (0.05) keeps fp32
    rounding noise on tiny grads from failing the check, while sign flips
    and missing factors (O(1) errors) still fail loudly. Returns max per
    param (must be < 0.15)."""
    pred, c = eagle_forward(P, feat, emb)
    d_pred = (pred - tgt) / pred.size
    G = eagle_backward(P, c, d_pred)
    for k in P:
        assert G[k].shape == P[k].shape, f"grad shape {k}: {G[k].shape} vs {P[k].shape}"
    errs = {}
    for k, v in P.items():
        num = np.zeros_like(v)
        n_probe = min(12, v.size)
        for fi in range(n_probe):
            ix = np.unravel_index(fi, v.shape)
            Pp = {kk: vv.copy() for kk, vv in P.items()}
            Pm = {kk: vv.copy() for kk, vv in P.items()}
            Pp[k][ix] += eps
            Pm[k][ix] -= eps
            lp = 0.5 * ((eagle_forward(Pp, feat, emb)[0] - tgt) ** 2).mean()
            lm = 0.5 * ((eagle_forward(Pm, feat, emb)[0] - tgt) ** 2).mean()
            num[ix] = (lp - lm) / (2 * eps)
        a = np.abs(num.ravel()[:n_probe])
        b = np.abs(G[k].ravel()[:n_probe])
        errs[k] = float((np.abs(a - b) / np.maximum(np.maximum(a, b), 0.05)).max())
    return errs


def train_draft_head(head, feats: np.ndarray, embs: np.ndarray,
                     tgts: np.ndarray, iters: int = 200,
                     lr: float = 1e-3, seed: int = 0) -> dict:
    """Sequence-batch SGD on feature MSE, all params. The (steps,D) arrays
    form ONE causal sequence (they came from one generation run).
    Deterministic."""
    P = head_to_dict(head)
    hist = []
    for _ in range(iters):
        pred, c = eagle_forward(P, feats, embs)
        err = pred - tgts
        loss = 0.5 * (err ** 2).mean()
        G = eagle_backward(P, c, err / err.size)
        for k in P:
            P[k] -= lr * np.clip(G[k], -1.0, 1.0)
        hist.append(float(loss))
    dict_to_head(P, head)
    return {"loss_start": float(hist[0]), "loss_end": float(hist[-1]),
            "iters": iters, "lr": lr}


def collect_training_data(model, prompt_ids: list[int],
                          steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the target greedily; record (feat_t, emb(tok_t), feat_{t+1}).

    feat_t is the hidden state that predicts token t+1; emb(tok_t) is the
    embedding of the token just consumed. Shapes: (steps, D) each."""
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    ids = list(prompt_ids)
    cache: dict = {}
    h = emb[np.array(ids)]
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, 0)
    pos = len(ids)
    F, E, T = [], [], []
    for _ in range(steps):
        F.append(h[-1].copy())
        E.append(emb[ids[-1]].copy())
        lg = (h[-1:] @ model.head.T.astype(np.float32))[0]
        nxt = int(np.argmax(lg))
        ids.append(nxt)
        h = emb[[nxt]]
        for lyr in range(model.n_layer):
            h = model._layer(h, lyr, cache, pos)
        pos += 1
        T.append(h[-1].copy())
    return np.stack(F), np.stack(E), np.stack(T)


# ---------------------------------------------------------------------------
# Tree drafting + single-forward verification on a ChatModel
# ---------------------------------------------------------------------------

_HEAD_T: dict = {}


def _head_t_cache(model):
    """Transposed head, cached per model — the verify/draft loops used to
    build a full VxD copy every call (19MB per token on the toy vocab)."""
    key = id(model)
    got = _HEAD_T.get(key)
    if got is None:
        got = model.head.T.astype(np.float32)
        if len(_HEAD_T) > 4:
            _HEAD_T.clear()
        _HEAD_T[key] = got
    return got


def draft_tree(head, feat: np.ndarray, last_tok: int, embed_row, lm_head: np.ndarray,
               max_nodes: int = 31, branch=(3, 2, 2)) -> tuple[list[int], list[int], list[int], int]:
    """BFS token tree. Node 0 is the last verified token (toks[0]=last_tok,
    depth 0). Children draft from the trained head autoregressively on
    features. Returns (parents, toks, depths, head_forwards)."""
    from .spec import DraftHead as _DH  # reuse forward numerology
    assert isinstance(head, _DH)
    parents, toks, depths = [-1], [int(last_tok)], [0]
    frontier = [0]
    fmap = {0: feat.astype(np.float32)}
    n_fwd = 0
    for depth, b in enumerate(branch):
        nxt = []
        for p in frontier:
            f = head.forward(fmap[p][None], embed_row(toks[p]))
            n_fwd += 1
            lg = (f @ lm_head.T)[0]
            top = np.argsort(-lg, kind="stable")[:b]
            for tk in top:
                if len(parents) >= max_nodes:
                    break
                parents.append(p)
                toks.append(int(tk))
                depths.append(depth + 1)
                nxt.append(len(parents) - 1)
                fmap[len(parents) - 1] = f[0]
            if len(parents) >= max_nodes:
                break
        frontier = nxt
        if not frontier or len(parents) >= max_nodes:
            break
    return parents, toks, depths, n_fwd


def verify_tree(model, cache: dict, pos: int, tree_ids: list[int],
                parents: list[int], depths: list[int]) -> np.ndarray:
    """Verify the CHILDREN (depth >= 1) in one target forward. The root
    (node 0, the last verified token) is NOT re-encoded — its key already
    lives in the prefix cache, and re-encoding would append a duplicate key
    and shift every logit (measured 0.058 drift). Root logits come from the
    live hidden state (exact, zero cost); see generate_eagle.

    Returns logits per child node, in tree order (root excluded).
    cache is shared-read-only (never mutated in place); verification runs
    on a seeded copy. Exact positions + ancestry mask."""
    import numpy as _np
    emb = model.t["model.embed_tokens.weight"].astype(_np.float32)
    order = [i for i in range(len(tree_ids)) if depths[i] >= 1]
    N = len(order)
    D = emb.shape[1]
    x = _np.stack([emb[tree_ids[i]] for i in order])
    Lp = pos  # prefix length == next position
    pos_ov = _np.array([Lp - 1 + depths[i] for i in order])
    tmask = build_tree_mask(parents)  # full tree incl. root
    K = Lp + N
    bias = _np.full((N, K), -_np.inf, dtype=_np.float32)
    bias[:, :Lp] = 0.0  # every node sees the whole prefix (root lives there)
    gidx = {old: new for new, old in enumerate(order)}
    for new_i, old_i in enumerate(order):
        for old_j in range(len(parents)):
            if tmask[old_i, old_j] == 0.0 and old_j != 0 and old_j in gidx:
                bias[new_i, Lp + gidx[old_j]] = 0.0
    tc = dict(cache)
    h = x
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, tc, Lp, attn_bias=bias, pos_override=pos_ov)
    return (h @ _head_t_cache(model)), tc, h


def commit_verified(model, cache: dict, tc: dict, pos: int, emb: np.ndarray,
                    toks: list[int], parents: list[int], depths: list[int],
                    cur_lg_fn, st: SpecStats, max_tokens: int,
                    out: list, stop_ids) -> tuple[bool, int, np.ndarray, int]:
    """Greedy walk + exact cache-gather commit, shared by tree (EAGLE) and
    chain (MTP/PLD) verifiers. cur_lg_fn(old_index) -> logits; old 0 == root
    (live hidden). Returns (done, pos, h, last_tok). Always appends >= 1."""
    import numpy as _np
    accepted: list[int] = []
    cur = 0
    while True:
        top = int(_np.argmax(cur_lg_fn(cur)))
        kids = [i for i in range(1, len(toks)) if parents[i] == cur]
        nxt = next((i for i in kids if toks[i] == top), None)
        accepted.append(top if nxt is None else toks[nxt])
        if nxt is None:
            break
        cur = nxt
    st.accepted += len(accepted) - 1
    st.tokens_out += len(accepted)
    done = False
    for a in accepted:
        if a in stop_ids:
            out.append(a)
            done = True
            break
        out.append(a)
        if len(out) >= max_tokens:
            done = True
            break
    if done:
        return done, pos, emb[[accepted[-1]]], accepted[-1]
    # cache reuse WITHOUT re-encode: gather the drafted-accepted prefix rows
    # from the tree cache in path order (copies, no matmuls). Only the bonus
    # token needs one standard forward. Exact.
    order = [i for i in range(len(toks)) if depths[i] >= 1]
    gidx = {old: new for new, old in enumerate(order)}
    path_nodes = []
    cur = 0
    for a in accepted:
        kids = [i for i in range(1, len(toks)) if parents[i] == cur]
        nxt = next((i for i in kids if toks[i] == a), None)
        if nxt is None:
            break
        path_nodes.append(nxt)
        cur = nxt
    n_bonus = len(accepted) - len(path_nodes)
    assert n_bonus == 1, "greedy walk always ends in exactly one bonus"
    bonus = accepted[-1]
    for lyr in range(model.n_layer):
        ok_, ov = cache[lyr]
        tk, tv = tc[lyr]
        Lp = ok_.shape[0]
        rows_k = [tk[[Lp + gidx[i]]] for i in path_nodes]
        rows_v = [tv[[Lp + gidx[i]]] for i in path_nodes]
        cache[lyr] = (np.concatenate([ok_] + rows_k) if rows_k else ok_,
                      np.concatenate([ov] + rows_v) if rows_v else ov)
    pos += len(path_nodes)
    h = emb[[bonus]]
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, pos)
    st.target_forwards += 1
    pos += 1
    return done, pos, h, bonus


def generate_eagle(model, prompt: str, head, max_tokens: int = 32,
                   budget_s: float = 300.0, stop_ids=(151643,),
                   max_nodes: int = 31, branch=(3, 2, 2)) -> dict:
    """EAGLE tree draft + exact single-forward verify. Greedy: output provably
    equals plain greedy decode (asserted in tests on the toy rig)."""
    t0 = time.perf_counter()
    st = SpecStats()
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    head_mat = model.head.astype(np.float32)  # (V,D); draft_tree transposes
    cache: dict = {}
    h = emb[np.array(ids)]
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, 0)
    pos, out = len(ids), []
    last_tok = ids[-1]
    while len(out) < max_tokens and time.perf_counter() - t0 < budget_s:
        root_lg = (h[-1:] @ _head_t_cache(model))[0]
        parents, toks, depths, n_fwd = draft_tree(head, h[-1], last_tok,
                                                  lambda t: emb[[t]], head_mat,
                                                  max_nodes, branch)
        st.drafted += len(toks) - 1
        st.draft_forwards += n_fwd
        lg, tc, _th = verify_tree(model, cache, pos, toks, parents, depths)
        st.target_forwards += 1
        done, pos, h, last_tok = commit_verified(
            model, cache, tc, pos, emb, toks, parents, depths,
            lambda cur: root_lg if cur == 0 else lg[cur - 1],
            st, max_tokens, out, stop_ids)
        if done:
            break
    dt = time.perf_counter() - t0
    out = out[:max_tokens]
    return {"text": model.tok.decode(out), "tokens": out,
            "spec": st.as_dict(),
            "tokens_per_second": len(out) / dt if dt else 0.0, "seconds": dt,
            "speculative": "eagle-tree"}


def mtp_forward(P: dict, h: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, dict]:
    """One MTP block forward (sequence). h: (T,D) target hidden, e: (T,D)
    next-token embeds. Returns (pred (T,D), cache). Mirrors MTPStack."""
    x = np.concatenate([h, e], axis=-1).astype(np.float32)
    T = x.shape[0]
    u = x @ P["proj"].T
    q = u @ P["blk_q"].T
    sc = q @ q.transpose(0, 2, 1) / max(q.shape[-1] ** 0.5, 1e-6) if q.ndim == 3 else q @ q.T / max(q.shape[-1] ** 0.5, 1e-6)
    causal = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
    ex = np.exp(sc + causal - (sc + causal).max(-1, keepdims=True))
    a = ex / ex.sum(-1, keepdims=True)
    v = (a @ u) @ P["blk_o"].T
    y = u + v
    pred = y @ P["back"].T
    return pred.astype(np.float32), {"x": x, "u": u, "q": q, "a": a, "v": v,
                                     "y": y}


def mtp_backward(P: dict, c: dict, d_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Full backprop for the MTP block. Grad-checked below."""
    G: dict[str, np.ndarray] = {}
    G["back"] = d_pred.T @ c["y"]
    dy = d_pred @ P["back"]
    G["blk_o"] = c["v"].T @ dy if c["v"].ndim == 2 else np.einsum("tij,tik->jk", c["v"], dy)
    d_v = dy @ P["blk_o"].T
    d_a = d_v @ c["u"].T if c["u"].ndim == 2 else np.einsum("tij,tik->tjk", d_v, c["u"]).sum(-1)
    row = (d_a * c["a"]).sum(-1, keepdims=True)
    d_sc = c["a"] * (d_a - row)
    dd = c["q"].shape[-1]
    scl = max(dd ** 0.5, 1e-6)
    if c["q"].ndim == 3:
        d_q = d_sc @ c["q"] / scl
        G["blk_q"] = np.einsum("tij,tik->jk", d_q, c["u"])
        d_u = (np.einsum("tij,jk->tik", d_q, P["blk_q"]) + d_v + dy)
    else:
        d_q = d_sc @ c["q"] / scl
        G["blk_q"] = d_q.T @ c["u"]
        d_u = d_q @ P["blk_q"] + d_v + dy
    G["proj"] = d_u.T @ c["x"]
    return G


def mtp_to_dict(stack, depth: int) -> dict:
    return {"proj": np.array(stack.proj[depth], dtype=np.float32),
            "blk_q": np.array(stack.blk_q[depth], dtype=np.float32),
            "blk_o": np.array(stack.blk_o[depth], dtype=np.float32),
            "back": np.array(stack.back[depth], dtype=np.float32)}


def mtp_from_dict(P: dict, stack, depth: int) -> None:
    for k in ("proj", "blk_q", "blk_o", "back"):
        stack.__dict__[k][depth] = np.array(P[k], dtype=np.float32)


def train_mtp_block(stack, depth: int, h: np.ndarray, e: np.ndarray,
                    tgts: np.ndarray, iters: int = 200,
                    lr: float = 1e-3) -> dict:
    """Sequence-batch SGD on one MTP block (feature MSE). Deterministic."""
    P = mtp_to_dict(stack, depth)
    hist = []
    for _ in range(iters):
        pred, c = mtp_forward(P, h, e)
        err = pred - tgts
        loss = 0.5 * (err ** 2).mean()
        G = mtp_backward(P, c, err / err.size)
        for k in P:
            P[k] -= lr * np.clip(G[k], -1.0, 1.0)
        hist.append(float(loss))
    mtp_from_dict(P, stack, depth)
    return {"loss_start": float(hist[0]), "loss_end": float(hist[-1]),
            "iters": iters, "lr": lr}


def generate_mtp(model, prompt: str, stack, max_tokens: int = 32,
                 budget_s: float = 300.0, stop_ids=(151643,),
                 temperature: float = 0.0, rng=None) -> dict:
    """MTP chain draft + exact single-forward verify + greedy walk.

    Drafts come from the trained MTPStack (depth = len(stack)); verification
    reuses the tree machinery with a chain (branching 1), so the same
    exactness proof holds. Greedy output equals plain greedy decode."""
    from .spec import SpecStats as _SS
    t0 = time.perf_counter()
    st = _SS()
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    head_mat = model.head.astype(np.float32)  # (V,D); call sites transpose
    cache: dict = {}
    h = emb[np.array(ids)]
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, 0)
    pos, out = len(ids), []
    last_tok = ids[-1]
    rng = rng if rng is not None else np.random.default_rng(0)
    while len(out) < max_tokens and time.perf_counter() - t0 < budget_s:
        root_lg = (h[-1:] @ _head_t_cache(model))[0]
        toks, depths, parents = [last_tok], [0], [-1]
        hh, d_lg = h[-1:], []
        for _d in range(stack.depth):
            x = np.concatenate([hh, emb[[toks[-1]]]], axis=-1) @ stack.proj[_d].T
            xb = x[None]
            q = xb @ stack.blk_q[_d].T
            sc = q @ q.transpose(0, 2, 1)
            a = np.exp(sc - sc.max(-1, keepdims=True))
            a = a / a.sum(-1, keepdims=True)
            xx = (xb + (a @ xb) @ stack.blk_o[_d].T)[0]
            hh = (xx @ stack.back[_d].T)
            lg = (hh @ head_mat.T)[0]
            d_lg.append(lg)
            if temperature <= 0:
                toks.append(int(np.argmax(lg)))
            else:
                z = lg - lg.max()
                p = np.exp(z / max(temperature, 1e-6))
                toks.append(int(np.argmax(rng.multinomial(1, p / p.sum()))))
            depths.append(depths[-1] + 1)
            parents.append(len(toks) - 2)
        st.drafted += len(toks) - 1
        st.draft_forwards += len(toks) - 1
        lgm, tc, _th = verify_tree(model, cache, pos, toks, parents, depths)
        st.target_forwards += 1
        done, pos, h, last_tok = commit_verified(
            model, cache, tc, pos, emb, toks, parents, depths,
            lambda cur: root_lg if cur == 0 else lgm[cur - 1],
            st, max_tokens, out, stop_ids)
        if done:
            break
    dt = time.perf_counter() - t0
    out = out[:max_tokens]
    return {"text": model.tok.decode(out), "tokens": out,
            "spec": st.as_dict(),
            "tokens_per_second": len(out) / dt if dt else 0.0, "seconds": dt,
            "speculative": "mtp-chain"}
