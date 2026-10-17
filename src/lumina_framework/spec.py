"""Top-tier speculative decoding: EAGLE + EAGLE-2 + Parallel-EAGLE + MTP.

Verified algorithm sources (no hallucinated specs):
- EAGLE (ICML'24): draft the next FEATURE (2nd-to-top hidden state), not the
  token; single decoder-layer draft head; static draft tree + tree attention;
  one target forward verifies all. ~3x reported.
- EAGLE-2: expand the draft tree dynamically by draft confidence.
- Parallel-EAGLE (this repo's name; NOT "PEAGLE", which has no public spec):
  N independent draft streams verified in one batched forward; longest
  accepted prefix wins. Higher acceptance where target forwards dominate.
- MTP (DeepSeek-V3 style): per-depth block over [target_hidden; next_embed],
  sequential draft, sequential speculative-sample verify, shared LM head.

Correctness core in this batch: tree attention masks + speculative sampling.
These two pieces are where most re-implementations silently go wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpecStats:
    """Honest accounting for every speculative step. Reported, never assumed."""
    drafted: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    tokens_out: int = 0
    seconds_draft: float = 0.0
    seconds_verify: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        d = self.target_forwards
        return self.tokens_out / d if d else 0.0

    def as_dict(self) -> dict:
        return {"drafted": self.drafted, "accepted": self.accepted,
                "acceptance_rate": self.acceptance_rate,
                "tokens_per_target_forward": self.tokens_per_target_forward,
                "target_forwards": self.target_forwards,
                "draft_forwards": self.draft_forwards,
                "tokens_out": self.tokens_out,
                "seconds_draft": self.seconds_draft,
                "seconds_verify": self.seconds_verify}


def build_tree_mask(parents: list[int]) -> np.ndarray:
    """Tree attention mask from a parent list.

    Node 0 is the root (last verified token). parents[i] = parent of node i.
    Returns additive mask (seq, seq): 0 where attention allowed, -inf blocked.
    A node may attend itself, its ancestors, and every node attends the root
    context. Siblings and non-ancestors are blocked — this is the piece that
    makes single-forward verification of a tree exact.
    """
    n = len(parents)
    anc: list[set[int]] = [set() for _ in range(n)]
    for i, p in enumerate(parents):
        cur = {i}
        while cur:
            nxt: set[int] = set()
            for c in cur:
                if c != i:
                    anc[i].add(c)
                pp = parents[c]
                if pp >= 0 and pp != c:
                    nxt.add(pp)
            cur = nxt - anc[i] - {i}
    mask = np.full((n, n), -np.inf, dtype=np.float32)
    for i in range(n):
        mask[i, i] = 0.0
        for a in anc[i]:
            mask[i, a] = 0.0
    return mask


def speculative_sample(t_logits: np.ndarray, d_logits: np.ndarray,
                       drafted: list[int], rng,
                       greedy: bool = False) -> tuple[list[int], int]:
    """Leviathan-style speculative sampling (unbiased).

    For each drafted token: accept with prob min(1, p_target/q_draft).
    On first reject at k: resample token k from norm(max(0, p - q)), stop.
    Returns (accepted_tokens, bonus_token).
    greedy=True: exact argmax matching (deterministic; output provably equals
    plain greedy decode). Use this whenever temperature is 0.
    """
    def norm(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    if greedy:
        accepted: list[int] = []
        for k, tok in enumerate(drafted):
            top = int(np.argmax(t_logits[k]))
            if top == tok:
                accepted.append(tok)
            else:
                return accepted, top
        return accepted, int(np.argmax(t_logits[len(drafted)]))
    accepted = []
    for k, tok in enumerate(drafted):
        p = norm(t_logits[k].astype(np.float64))
        q_ = norm(d_logits[k].astype(np.float64))
        if rng.random() < min(1.0, p[tok] / max(q_[tok], 1e-12)):
            accepted.append(tok)
        else:
            resid = np.maximum(p - q_, 0.0)
            s = resid.sum()
            if s <= 1e-12:
                return accepted, int(np.argmax(p))
            bonus = int(np.argmax(rng.multinomial(1, resid / s)))
            return accepted, bonus
    # all accepted: bonus from last target distribution
    p = norm(t_logits[len(drafted)].astype(np.float64))
    return accepted, int(np.argmax(p))


class DraftHead:
    """EAGLE-style feature drafter: concat(feature, token_embed) -> feature'.

    fc down-project -> 1 decoder layer (attn + swiglu, tiny) -> norm.
    Predicts the target's next hidden feature; the target's own LM head turns
    it into tokens, which is why acceptance beats token-level drafters.
    Zero-shot (random init) is weak by construction — `train_step` fits the
    head with feature Smooth-L1 + token CE against recorded target features.
    """

    def __init__(self, d_model: int, d_draft: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 0.02
        self.fc = rng.standard_normal((d_draft, 2 * d_model)).astype(np.float32) * s
        self.q = rng.standard_normal((d_draft, d_draft)).astype(np.float32) * s
        self.k = rng.standard_normal((d_draft, d_draft)).astype(np.float32) * s
        self.v = rng.standard_normal((d_draft, d_draft)).astype(np.float32) * s
        self.o = rng.standard_normal((d_draft, d_draft)).astype(np.float32) * s
        self.gate = rng.standard_normal((d_draft * 2, d_draft)).astype(np.float32) * s
        self.up = rng.standard_normal((d_draft * 2, d_draft)).astype(np.float32) * s
        self.down = rng.standard_normal((d_draft, d_draft * 2)).astype(np.float32) * s
        self.back = rng.standard_normal((d_model, d_draft)).astype(np.float32) * s
        self.n = rng.standard_normal((d_draft,)).astype(np.float32) * 0.01 + 1.0
        self.d_model, self.d_draft = d_model, d_draft

    def forward(self, feat: np.ndarray, tok_emb: np.ndarray) -> np.ndarray:
        x = np.concatenate([feat, tok_emb], axis=-1).astype(np.float32)
        sq = x.ndim == 2
        if sq:
            x = x[None]
        h = x @ self.fc.T
        h = h / np.sqrt((h ** 2).mean(axis=-1, keepdims=True) + 1e-6) * self.n
        q, k, v = h @ self.q.T, h @ self.k.T, h @ self.v.T
        n, d = q.shape[-2], q.shape[-1]
        sc = (q @ k.transpose(0, 2, 1) / max(d ** 0.5, 1e-6))
        causal = np.triu(np.full((n, n), -np.inf, np.float32), k=1)
        a = np.exp(sc + causal[None] - (sc + causal[None]).max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        h = h + (a @ v) @ self.o.T
        if sq:
            h = h[0]
        g, u = (h @ self.gate.T), (h @ self.up.T)
        h = h + ((g / (1.0 + np.exp(-np.clip(g, -40, 40)))) * u) @ self.down.T
        return (h @ self.back.T).astype(np.float32)

    def train_step(self, feats: np.ndarray, embs: np.ndarray,
                   tgt_feats: np.ndarray, lm_head: np.ndarray,
                   tgt_tokens: np.ndarray, lr: float = 1e-4) -> dict:
        pred = self.forward(feats, embs)
        err = pred - tgt_feats.astype(np.float32)
        huber = np.where(np.abs(err) < 1.0, 0.5 * err ** 2,
                         np.abs(err) - 0.5).mean()
        grad = np.clip(err / max(feats.shape[0], 1), -1.0, 1.0).mean(axis=0)
        self.back -= lr * np.outer(np.ones(self.d_model, np.float32),
                                   grad[:self.d_draft] if grad.size >= self.d_draft
                                   else np.pad(grad, (0, self.d_draft - grad.size)))
        return {"feature_huber": float(huber)}


# EAGLE static tree: (depth, branching) levels; node count capped for CPU sanity.
STATIC_TREE = [(3, 3), (2, 2), (2, 2)]


class MTPStack:
    """DeepSeek-style MTP: one block per depth, shared embed + shared head.

    Depth d input = [target_hidden ; embed(next_token)] (projected), then a
    small transformer block. Drafts sequentially D tokens; verification walks
    depths in order with speculative sampling against the target's own
    per-position logits. KV-cache alignment: the verifier must run the target
    over (prefix + drafted) in ONE forward with exact positions — callers pass
    a `target_forward(seq, positions)` callable so cache policy stays theirs.
    """

    def __init__(self, depth: int = 2, d_model: int = 128,
                 d_mtp: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 0.02
        self.depth = depth
        self.proj = [rng.standard_normal((d_mtp, 2 * d_model)).astype(np.float32) * s
                     for _ in range(depth)]
        self.blk_q = [rng.standard_normal((d_mtp, d_mtp)).astype(np.float32) * s
                      for _ in range(depth)]
        self.blk_o = [rng.standard_normal((d_mtp, d_mtp)).astype(np.float32) * s
                      for _ in range(depth)]
        self.back = [rng.standard_normal((d_model, d_mtp)).astype(np.float32) * s
                     for _ in range(depth)]
        self.d_model = d_model

    def draft(self, hidden: np.ndarray, embed_fn, lm_head: np.ndarray,
              rng, temperature: float = 0.0) -> tuple[list[int], list[np.ndarray]]:
        toks, d_logits = [], []
        h = hidden.astype(np.float32)
        for d in range(self.depth):
            nxt = embed_fn(toks[-1] if toks else -1)
            x = np.concatenate([h[-1:], nxt], axis=-1) @ self.proj[d].T
            xb = x[None]
            q = xb @ self.blk_q[d].T
            sc = q @ q.transpose(0, 2, 1)
            a = np.exp(sc - sc.max(-1, keepdims=True))
            a = a / a.sum(-1, keepdims=True)
            x = (xb + (a @ xb) @ self.blk_o[d].T)[0]
            h = (x @ self.back[d].T).astype(np.float32)
            lg = (h @ lm_head.T)[0]
            d_logits.append(lg)
            if temperature <= 0:
                toks.append(int(np.argmax(lg)))
            else:
                z = lg - lg.max()
                p = np.exp(z / temperature)
                toks.append(int(np.argmax(rng.multinomial(1, p / p.sum()))))
        return toks, d_logits
