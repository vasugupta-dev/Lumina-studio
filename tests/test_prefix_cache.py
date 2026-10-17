"""Prefix KV caching: parity + reuse accounting.

generate_cached must produce IDENTICAL tokens to generate across: first
turn, follow-up turn (the win), a diverged/edited turn (truncation), and a
fully-cached repeat — the case where the naive replay double-counted a
position (found by critique, fixed, locked here).
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_framework.chat import ChatModel, generate, generate_cached  # noqa: E402

MODEL_DIR = "/tmp/tiny-chat"
RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def turn(p):
    return "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"


def build_fixture():
    """Rebuild the tiny Qwen-format chat model if /tmp was cleared (phone-safe,
    ~1MB weights). Same builder pattern as test_chat_fast_parity."""
    if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        return
    import json

    from safetensors.numpy import save_file
    os.makedirs(MODEL_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    n_layer, d, inter, vocab, nh, nk = 2, 64, 128, 48, 4, 2
    w = {}
    w["model.embed_tokens.weight"] = rng.standard_normal((vocab, d)) * 0.05
    w["lm_head.weight"] = rng.standard_normal((vocab, d)) * 0.05
    for l in range(n_layer):
        p = f"model.layers.{l}"
        w[f"{p}.input_layernorm.weight"] = np.ones(d)
        w[f"{p}.post_attention_layernorm.weight"] = np.ones(d)
        w[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((d, d)) * 0.05
        w[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((32, d)) * 0.05
        w[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((32, d)) * 0.05
        w[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((d, d)) * 0.05
        w[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((inter, d)) * 0.05
        w[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((inter, d)) * 0.05
        w[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, inter)) * 0.05
    save_file({k: v.astype(np.float32) for k, v in w.items()},
              os.path.join(MODEL_DIR, "model.safetensors"))
    json.dump({"num_hidden_layers": n_layer, "hidden_size": d,
               "num_attention_heads": nh, "num_key_value_heads": nk,
               "vocab_size": vocab},
              open(os.path.join(MODEL_DIR, "config.json"), "w"))
    for fn, txt in (("vocab.json", '{"<s>": 0, "</s>": 1, "a": 2, "b": 3, "hello": 4, "world": 5, "the": 6, "quick": 7}'),
                    ("merges.txt", "\n")):
        open(os.path.join(MODEL_DIR, fn), "w").write(txt)


def main() -> int:
    build_fixture()
    m = ChatModel(MODEL_DIR)
    p1 = turn("hello world tell me things")
    r1 = generate(m, p1, max_tokens=8, rng=np.random.default_rng(0))
    p2 = p1 + r1["text"] + "<|im_end|>\n" + turn("and more")
    r2 = generate(m, p2, max_tokens=8, rng=np.random.default_rng(0))
    p4 = p2 + r2["text"] + "<|im_end|>\n" + turn("one more")
    r4 = generate(m, p4, max_tokens=8, rng=np.random.default_rng(0))
    p3 = turn("hello THERE instead")  # diverges at the first turn
    r3 = generate(m, p3, max_tokens=6, rng=np.random.default_rng(1))

    st = {"ids": [], "kv": {}, "pos": 0}
    c1 = generate_cached(m, p1, max_tokens=8, rng=np.random.default_rng(0), state=st)
    check("turn1 cached == fresh", r1["tokens"] == c1["tokens"])
    c2 = generate_cached(m, p2, max_tokens=8, rng=np.random.default_rng(0), state=st)
    check("turn2 cached == fresh", r2["tokens"] == c2["tokens"])
    check("turn2 reuses the old prefix",
          c2["stats"]["prefill_reused"] > 0
          and c2["stats"]["prefill_computed"] < len(m.tok.encode(p2)))
    c3 = generate_cached(m, p3, max_tokens=6, rng=np.random.default_rng(1), state=st)
    check("diverged turn cached == fresh", r3["tokens"] == c3["tokens"])
    c4 = generate_cached(m, p3, max_tokens=6, rng=np.random.default_rng(1), state=st)
    check("fully-cached repeat == fresh", r3["tokens"] == c4["tokens"])
    check("fully-cached repeat reuses everything",
          c4["stats"]["prefill_computed"] == 0)
    check("state ids track history",
          len(st["ids"]) == len(m.tok.encode(p3)) + len(r3["tokens"]))
    c5 = generate_cached(m, p4, max_tokens=8, rng=np.random.default_rng(0), state=st)
    check("turn after repeat cached == fresh", r4["tokens"] == c5["tokens"])

    n = sum(1 for _, c in RESULTS if c)
    print(f"\nPREFIX-CACHE: {n}/{len(RESULTS)} passed")
    print("RESULT:", "ALL PASS" if n == len(RESULTS) else "FAILURES PRESENT")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
