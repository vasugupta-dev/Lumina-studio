"""Train an EAGLE draft head for any loaded ChatModel directory.

Run:  python3 scripts/train_eagle.py [--model-dir /tmp/tiny-chat]
        [--prompts "hello world|repeat after me"] [--steps 256] [--iters 150]

Collects target (feat, emb, next-feat) triples by running the target, trains
the draft head with full backprop (grad-checked), saves eagle_head.npz next
to the model dir. /api/chat {"spec": "eagle"} picks it up automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_framework.chat import ChatModel  # noqa: E402
from lumina_framework.spec import DraftHead, MTPStack  # noqa: E402
from lumina_framework import eagle as E  # noqa: E402
from lumina_framework.eagle import train_mtp_block  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/tmp/tiny-chat")
    ap.add_argument("--prompts", default="hello world|the quick brown fox|repeat after me now")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-draft", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    model = ChatModel(a.model_dir)
    D = model.t["model.embed_tokens.weight"].shape[1]
    d_draft = a.d_draft or D
    head = DraftHead(D, d_draft=d_draft, seed=0)
    F, Em, Tg = [], [], []
    for p in a.prompts.split("|"):
        ids = model.tok.encode(p.strip() or "hello")
        f, e, t = E.collect_training_data(model, ids, a.steps)
        F.append(f)
        Em.append(e)
        Tg.append(t)
    F = np.concatenate(F).astype(np.float32)
    Em = np.concatenate(Em).astype(np.float32)
    Tg = np.concatenate(Tg).astype(np.float32)
    print(f"data: {F.shape[0]} steps, D={D}, d_draft={d_draft}")
    rep = E.train_draft_head(head, F, Em, Tg, iters=a.iters, lr=a.lr)
    print(f"train: loss {rep['loss_start']:.4f} -> {rep['loss_end']:.4f} "
          f"({rep['iters']} iters, {time.perf_counter()-t0:.0f}s)")
    mst = MTPStack(depth=2, d_model=D, d_mtp=D, seed=0)
    for dd in range(mst.depth):
        # depth-d block maps (feat_t, emb_t) -> feat_{t+d+1}
        Xe = Em[:len(Em)-dd-1] if dd else Em
        Xh = F[:len(F)-dd-1] if dd else F
        Xt = Tg[dd+1:] if len(Tg) > dd + 1 else Tg
        n = min(len(Xh), len(Xe), len(Xt))
        rep = train_mtp_block(mst, dd, Xh[:n], Xe[:n], Xt[:n], iters=a.iters, lr=a.lr)
        print(f"mtp depth-{dd}: loss {rep['loss_start']:.4f} -> {rep['loss_end']:.4f}")
    mout = os.path.join(a.model_dir, "mtp_stack.npz")
    mp = {"depth": mst.depth, "d_model": D, "d_mtp": D}
    for dd in range(mst.depth):
        for k in ("proj", "blk_q", "blk_o", "back"):
            mp[f"d{dd}_{k}"] = getattr(mst, {"proj": "proj", "blk_q": "blk_q", "blk_o": "blk_o", "back": "back"}[k])[dd]
    np.savez(mout, **{k: np.asarray(v) for k, v in mp.items() if not isinstance(v, int)})
    json.dump({"depth": mst.depth, "d_model": D, "d_mtp": D},
              open(os.path.join(a.model_dir, "mtp_stack.json"), "w"))
    print("saved", mout)
    out = os.path.join(a.model_dir, "eagle_head.npz")
    payload = {"d_model": D, "d_draft": d_draft}
    payload.update({k: getattr(head, k) for k in
                    ("fc", "q", "k", "v", "o", "gate", "up", "down", "back", "n")})
    np.savez(out, **{k: np.asarray(v) for k, v in payload.items()
                     if not isinstance(v, int)})
    meta = {"d_model": D, "d_draft": d_draft}
    json.dump(meta, open(os.path.join(a.model_dir, "eagle_head.json"), "w"))
    print("saved", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
