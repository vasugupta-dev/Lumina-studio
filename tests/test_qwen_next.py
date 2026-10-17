"""Qwen4-exp clean-room kernels: mechanism verification (NOT weight-verified).

Every test here checks structure the math guarantees at any scale —
determinism, causality, streaming/state equivalence, parity, auditability —
on synthetic weights in the REAL tensor layout. Nothing here proves the
kernels match the real 180B weights; that needs the download + RTX machine.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_framework import qwen_next as Q  # noqa: E402
from lumina_framework.chat import apply_rope, rope_freqs  # noqa: E402

rng = np.random.default_rng(3)
RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def moe_case():
    S, D, E, I = 5, 32, 16, 24
    h = rng.normal(size=(S, D)).astype(np.float32)
    gu = (rng.normal(size=(E, 2 * I, D)) * 0.1).astype(np.float32)
    dn = (rng.normal(size=(E, I, D)) * 0.1).astype(np.float32)
    rw = (rng.normal(size=(E, D)) * 0.1).astype(np.float32)
    o1, i1 = Q.moe_grouped_forward(h, gu, dn, rw, 4)
    o2, _ = Q.moe_grouped_forward(h, gu, dn, rw, 4)
    check("moe deterministic", np.array_equal(o1, o2))
    sig = 1 / (1 + np.exp(-np.clip(h @ rw.T, -40, 40)))
    check("moe top-k audit",
          all(set(r) == set(np.argsort(-s)[:4]) for r, s in zip(i1["experts"], sig)))
    sh = {"g": (rng.normal(size=(I, D)) * 0.1).astype(np.float32),
          "u": (rng.normal(size=(I, D)) * 0.1).astype(np.float32),
          "d": (rng.normal(size=(D, I)) * 0.1).astype(np.float32)}
    o3, _ = Q.moe_grouped_forward(h, gu, dn, rw, 4, shared_gate=sh["g"],
                                  shared_up=sh["u"], shared_down=sh["d"],
                                  shared_gate_w=0.7)
    g = h @ sh["g"].T
    u = h @ sh["u"].T
    term = 0.7 * (((g / (1 + np.exp(-np.clip(g, -40, 40)))) * u) @ sh["d"].T)
    check("moe shared term exact", np.allclose(o3, o1 + term, atol=1e-5))
    op, _ = Q.moe_grouped_parallel(h, gu, dn, rw, 4)
    check("moe parallel exact parity", np.array_equal(o1, op))
    op2, _ = Q.moe_grouped_parallel(h, gu, dn, rw, 4, shared_gate=sh["g"],
                                    shared_up=sh["u"], shared_down=sh["d"],
                                    shared_gate_w=0.7)
    check("moe parallel shared exact", np.allclose(op2, o3, atol=1e-5))


def ngram_case():
    dim, Hh = 16, 3
    shards = [(rng.normal(size=(s, dim)) * 0.1).astype(np.float32) for s in (50, 70, 40)]
    offs = np.array([0, 1000, 2000])
    vs = np.array([100, 200, 150])
    mult = np.array([1.0, 0.5, 2.0])
    ids = [5, 17, 42, 9, 30]
    n1 = Q.ngram_bag(ids, shards, offs, vs, mult)
    check("ngram deterministic", np.array_equal(n1, Q.ngram_bag(ids, shards, offs, vs, mult)))
    exp = np.zeros(dim, np.float32)
    bounds = [0, 50, 120, 160]
    for hh in range(Hh):
        key = (5 + int(offs[hh])) % int(vs[hh])
        gid = (int(offs[hh]) + key) % 160
        si = next(i for i in range(3) if bounds[i] <= gid < bounds[i + 1])
        exp += float(mult[hh]) * shards[si][gid - bounds[si]]
    check("ngram manual exact", np.allclose(n1[0], exp, atol=1e-6))
    check("ngram finite", bool(np.all(np.isfinite(n1))))


def linear_case():
    Sq, Dq, Hq, Hv2, hk, hv = 9, 48, 4, 4, 8, 6
    x = rng.normal(size=(Sq, Dq)).astype(np.float32)
    P = Hq * hk * 2 + Hv2 * hv
    kw = dict(in_proj_qkv=(rng.normal(size=(P, Dq)) * 0.1).astype(np.float32),
              in_proj_b=(rng.normal(size=(Hq, Dq)) * 0.1).astype(np.float32),
              in_proj_a=(rng.normal(size=(Hq, Dq)) * 0.1).astype(np.float32),
              in_proj_z=(rng.normal(size=(Hq * hv, Dq)) * 0.1).astype(np.float32),
              A_log=(rng.normal(size=(Hq,)) * 0.1).astype(np.float32),
              dt_bias=np.zeros(Hq, np.float32),
              out_proj=(rng.normal(size=(Dq, Hq * hv)) * 0.1).astype(np.float32),
              norm_w=np.ones(Hq * hv, np.float32),
              n_qk=Hq, n_v=Hv2, hk=hk, hv=hv)
    full, _ = Q.linear_attn_forward(x, **kw)
    pre, _ = Q.linear_attn_forward(x[:5], **kw)
    check("linear prefix==full prefix", np.allclose(pre, full[:5], atol=1e-5))
    x2 = x.copy()
    x2[6:] = rng.normal(size=(3, Dq)).astype(np.float32) * 3.0
    full2, _ = Q.linear_attn_forward(x2, **kw)
    check("linear causal", np.allclose(full2[:6], full[:6], atol=1e-4))
    check("linear finite+deterministic",
          bool(np.all(np.isfinite(full)))
          and np.array_equal(full, Q.linear_attn_forward(x, **kw)[0]))
    kw2 = dict(kw, n_v=8,
               in_proj_qkv=(rng.normal(size=(Hq * hk * 2 + 8 * hv, Dq)) * 0.1).astype(np.float32),
               out_proj=(rng.normal(size=(Dq, 8 * hv)) * 0.1).astype(np.float32))
    fa, _ = Q.linear_attn_forward(x, **kw2)
    check("linear asymmetric pairing runs",
          fa.shape == (Sq, Dq) and bool(np.all(np.isfinite(fa))))


def qsa_case():
    Sq2, Dd, nq, nkv, hd = 10, 40, 4, 2, 8
    xx = rng.normal(size=(Sq2, Dd)).astype(np.float32)
    Qp = (rng.normal(size=(nq * hd, Dd)) * 0.1).astype(np.float32)
    Kp = (rng.normal(size=(nkv * hd, Dd)) * 0.1).astype(np.float32)
    Vp = (rng.normal(size=(nkv * hd, Dd)) * 0.1).astype(np.float32)
    Op = (rng.normal(size=(Dd, nq * hd)) * 0.1).astype(np.float32)
    IZ = 16
    IQ = (rng.normal(size=(2 * IZ, Dd)) * 0.1).astype(np.float32)
    qn = np.ones(hd, np.float32)
    kn = np.ones(hd, np.float32)
    ql = np.ones(IZ, np.float32)
    kl = np.ones(IZ, np.float32)
    rd = 4
    out_full = Q.qsa_forward(xx, Qp, Kp, Vp, Op, IQ, qn, kn, ql, kl, nq, nkv,
                             hd, rd, block=2, top_blocks=99, window=99,
                             rope=rope_freqs)
    qq = (xx @ Qp.T).reshape(Sq2, nq, hd)
    kk = (xx @ Kp.T).reshape(Sq2, nkv, hd)
    vv = (xx @ Vp.T).reshape(Sq2, nkv, hd)
    qq = qq / np.sqrt((qq * qq).mean(-1, keepdims=True) + 1e-6) * qn
    kk = kk / np.sqrt((kk * kk).mean(-1, keepdims=True) + 1e-6) * kn
    rep = nq // nkv
    kk = np.repeat(kk, rep, 1)
    vv = np.repeat(vv, rep, 1)
    ang = rope_freqs(rd, Sq2)
    qq = np.concatenate([apply_rope(qq[..., :rd], ang[:, None, :]), qq[..., rd:]], -1)
    kk = np.concatenate([apply_rope(kk[..., :rd], ang[:, None, :]), kk[..., rd:]], -1)
    ref = np.zeros((Sq2, nq, hd), np.float32)
    for t in range(Sq2):
        sc = np.stack([qq[t, q] @ kk[:t + 1][:, q, :].T for q in range(nq)]) / max(hd ** 0.5, 1e-6)
        e = np.exp(sc - sc.max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        ref[t] = (a[:, :, None] * vv[:t + 1].transpose(1, 0, 2)).sum(1)
    ref = (ref.reshape(Sq2, -1) @ Op.T).astype(np.float32)
    check("qsa full-coverage == dense", np.allclose(out_full, ref, atol=1e-4))
    qkw = dict(block=2, top_blocks=1, window=3, rope=rope_freqs)
    out_sp = Q.qsa_forward(xx, Qp, Kp, Vp, Op, IQ, qn, kn, ql, kl, nq, nkv,
                           hd, rd, **qkw)
    check("qsa sparse runs+finite",
          out_sp.shape == (Sq2, Dd) and bool(np.all(np.isfinite(out_sp))))
    for block in (1, 2):
        kw2 = dict(qkw, block=block)
        a = Q.qsa_forward(xx, Qp, Kp, Vp, Op, IQ, qn, kn, ql, kl, nq, nkv,
                          hd, rd, **kw2)
        xx2 = xx.copy()
        xx2[7:] = rng.normal(size=(3, Dd)).astype(np.float32) * 3.0
        b = Q.qsa_forward(xx2, Qp, Kp, Vp, Op, IQ, qn, kn, ql, kl, nq, nkv,
                          hd, rd, **kw2)
        check(f"qsa causal block={block}", np.allclose(a[:7], b[:7], atol=1e-4))
    check("qsa deterministic",
          np.array_equal(out_sp, Q.qsa_forward(xx, Qp, Kp, Vp, Op, IQ, qn, kn,
                                               ql, kl, nq, nkv, hd, rd, **qkw)))


def hyper_case():
    Hm, Dm, Rr = 6, 24, 8
    hm = rng.normal(size=(Hm, Dm)).astype(np.float32)
    bo = rng.normal(size=(Hm, Dm)).astype(np.float32)
    got = Q.hyper_mix(hm, np.ones(Dm, np.float32),
                      np.zeros((Rr, Dm), np.float32),
                      np.zeros((Dm, Rr), np.float32),
                      np.full((4,), -50.0, np.float32), bo)
    check("hyper zero-gates == identity", np.allclose(got, hm, atol=1e-5))
    up = (rng.normal(size=(Rr, Dm)) * 0.1).astype(np.float32)
    dn = (rng.normal(size=(Dm, Rr)) * 0.1).astype(np.float32)
    g2 = Q.hyper_mix(hm, np.ones(Dm, np.float32), up, dn,
                     np.zeros(4, np.float32), bo)
    check("hyper active finite+deterministic",
          bool(np.all(np.isfinite(g2)))
          and np.array_equal(g2, Q.hyper_mix(hm, np.ones(Dm, np.float32), up,
                                             dn, np.zeros(4, np.float32), bo)))




def mamba_case():
    from lumina_framework.qwen_next import mamba_block_forward
    r = np.random.default_rng(8)
    S, D, DI, ds, nh, K = 7, 24, 16, 8, 4, 4
    x = (r.normal(size=(S, D)) * 0.5).astype(np.float32)
    kw = dict(
        in_proj=(r.normal(size=(2 * DI, D)) * 0.1).astype(np.float32),
        conv_w=(r.normal(size=(DI, K)) * 0.1).astype(np.float32),
        conv_b=np.zeros(DI, np.float32),
        x_proj_w=(r.normal(size=(4 + 2 * ds, DI)) * 0.1).astype(np.float32),
        dt_proj_w=(r.normal(size=(ds, 4)) * 0.1).astype(np.float32),
        A_log=(r.normal(size=(ds,)) * 0.3).astype(np.float32),
        dt_bias=np.zeros(ds, np.float32),
        out_proj=(r.normal(size=(D, DI)) * 0.1).astype(np.float32),
        norm_w=np.ones(DI, np.float32),
        n_heads=nh, d_state=ds)
    full, states, hist = mamba_block_forward(x, **kw)
    pre, st2, h2 = mamba_block_forward(x[:4], **kw)
    check("mamba prefix==full prefix", np.allclose(pre, full[:4], atol=1e-5))
    p1, st, h1 = mamba_block_forward(x[:3], **kw)
    p2, st3, h3 = mamba_block_forward(x[3:], **kw, states=st, conv_hist=h1)
    check("mamba state+conv carry == full",
          np.allclose(np.vstack([p1, p2]), full, atol=1e-5))
    x2 = x.copy()
    x2[5:] = (r.normal(size=(2, D)) * 2).astype(np.float32)
    full2, _, _ = mamba_block_forward(x2, **kw)
    check("mamba causal", np.allclose(full2[:5], full[:5], atol=1e-4))
    f3, _, _ = mamba_block_forward(x, **kw)
    check("mamba deterministic", np.array_equal(full, f3))
    check("mamba finite", bool(np.all(np.isfinite(full))))


def main() -> int:
    t0 = time.perf_counter()
    moe_case()
    vmoe_case()
    mamba_case()
    ngram_case()
    linear_case()
    qsa_case()
    hyper_case()
    n = sum(1 for _, c in RESULTS if c)
    print(f"\nQWEN-NEXT: {n}/{len(RESULTS)} passed in {time.perf_counter()-t0:.1f}s")
    print("RESULT:", "ALL PASS" if n == len(RESULTS) else "FAILURES PRESENT")
    return 0 if n == len(RESULTS) else 1

def vmoe_case():
    import sys as _s
    _s.path.insert(0, os.path.join(ROOT, "src"))
    from lumina_framework import virtual_moe as V
    from lumina_framework.qwen_next import moe_grouped_forward as _mgf
    r = np.random.default_rng(4)
    D, F, E = 64, 64, 8
    g = np.zeros((F, D), np.float32)
    u = np.zeros((F, D), np.float32)
    d = np.zeros((D, F), np.float32)
    for e in range(E):
        sl = slice(e * 8, (e + 1) * 8)
        g[sl, sl] = np.eye(8, dtype=np.float32) * 3.0
        u[sl, sl] = np.eye(8, dtype=np.float32) * 3.0
        d[sl, sl] = np.eye(8, dtype=np.float32)
    acts = np.zeros((256, D), np.float32)
    for i in range(256):
        e = i % E
        acts[i, e * 8:(e + 1) * 8] = r.normal(size=8) * 2.0
    routed = V.build_virtual_moe(g, u, d, acts, E, 2)
    q = V.virtual_moe_quality(g, u, d, routed, acts)
    check("vmoe specialized top-2 rel==0", q["rel_func_err"] < 1e-4)
    o1, _ = _mgf(acts[:4], routed["gate_up"], routed["down"], routed["router"],
                 2, weight_mode="unit", router_b=routed["router_b"],
                 router_feat="abs")
    from lumina_framework.qwen_next import moe_grouped_parallel as _mgp
    o2, _ = _mgp(acts[:4], routed["gate_up"], routed["down"], routed["router"],
                 2, weight_mode="unit", router_b=routed["router_b"],
                 router_feat="abs")
    check("vmoe unit+abs+bias parity", np.array_equal(o1, o2))
    r8 = V.build_virtual_moe(g, u, d, acts, 4, 4)
    check("vmoe full-k exact",
          V.virtual_moe_quality(g, u, d, r8, acts)["rel_func_err"] < 1e-5)


if __name__ == "__main__":
    sys.exit(main())
