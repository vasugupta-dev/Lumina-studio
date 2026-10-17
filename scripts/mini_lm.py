"""Tiny but REAL transformer LM, used only to measure true function preservation.

Trained briefly on a structured synthetic corpus so it has a non-trivial loss to
preserve. This is a measurement instrument, not a model for deployment.
"""
import numpy as np, torch, torch.nn as nn

VOCAB, D, NH, L, FF = 256, 192, 4, 4, 512

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3*D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FF, bias=False)
        self.fc2 = nn.Linear(FF, D, bias=False)
    def attn(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(D, dim=2)
        q = q.view(B,T,NH,C//NH).transpose(1,2)
        k = k.view(B,T,NH,C//NH).transpose(1,2)
        v = v.view(B,T,NH,C//NH).transpose(1,2)
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(B,T,C))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))
        return x + h

class MiniLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.embed.weight  # tied
    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x)), x

def train(model, steps=300, seed=0, sharpness=0.08):
    """Train on a Markov-chain corpus. `sharpness` controls corpus entropy:
    low = very predictable (memorizable), high = noisy."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.ones(VOCAB)*sharpness, size=VOCAB).astype(np.float32)
    seq = np.zeros((16, 64), dtype=np.int64)
    for b in range(16):
        seq[b,0] = rng.integers(VOCAB)
        for t in range(1,64):
            seq[b,t] = rng.choice(VOCAB, p=trans[seq[b,t-1]])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for s in range(steps):
        i = s % 16
        x = torch.from_numpy(seq[i, :-1]); y = torch.from_numpy(seq[i, 1:])
        logits, _ = model(x[None])
        loss = torch.nn.functional.cross_entropy(logits[0], y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, seq

def evaluate(model, seq):
    model.eval(); tot=0.0; n=0
    with torch.no_grad():
        for i in range(seq.shape[0]):
            x = torch.from_numpy(seq[i,:-1]); y = torch.from_numpy(seq[i,1:])
            logits,_ = model(x[None])
            tot += torch.nn.functional.cross_entropy(logits[0], y, reduction='sum').item(); n += len(y)
    return tot/n
