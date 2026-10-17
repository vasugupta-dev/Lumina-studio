"""Train and SAVE the reference model so measurements are reproducible."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mini_lm import MiniLM, train, evaluate
import torch

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "artifacts", "reference.pt")
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 900
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
sharp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6

m, seq = train(MiniLM(), steps=steps, seed=seed, sharpness=sharp)
loss = evaluate(m, seq)
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"state": m.state_dict(), "seq": seq, "loss": loss,
            "steps": steps, "seed": seed, "sharpness": sharp}, out)
print(f"saved {out}  loss={loss:.4f}  steps={steps} seed={seed} sharp={sharp}")
