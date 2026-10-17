"""ChatModel fast-path parity: fast=True (fp32 preconverted, transposed-once
weights) must produce IDENTICAL greedy tokens to fast=False, on a tiny
synthetic Qwen-format fixture (phone-safe: ~1MB weights).

Also micro-times both paths at this tiny scale (indicative only — real
speed claims come from the Kaggle notebook, not from this fixture).

  python3 tests/test_chat_fast_parity.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "lumina_framework"))

FIX = os.path.join("/tmp/opencode", "chatfix")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" — " + detail) if not cond else ""))
    if not cond:
        FAILS.append(name)


def build_fixture():
    os.makedirs(FIX, exist_ok=True)
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
        w[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((nk * d // nh * nh // 1, d)) * 0.05
        # keep shapes exact: hd = d // nh = 16; k/v project to nk*hd = 32
        w[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((32, d)) * 0.05
        w[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((32, d)) * 0.05
        w[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((d, d)) * 0.05
        w[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((inter, d)) * 0.05
        w[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((inter, d)) * 0.05
        w[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, inter)) * 0.05
    # safetensors writer (stdlib-free: use the installed safetensors)
    from safetensors.numpy import save_file
    save_file({k: v.astype(np.float32) for k, v in w.items()},
              os.path.join(FIX, "model.safetensors"))
    json.dump({"num_hidden_layers": n_layer, "hidden_size": d,
               "num_attention_heads": nh, "num_key_value_heads": nk,
               "vocab_size": vocab},
              open(os.path.join(FIX, "config.json"), "w"))
    # minimal BPE files so BPETokenizer loads
    for fn, txt in (("vocab.json", '{"<s>": 0, "</s>": 1, "a": 2, "b": 3, "hello": 4, "world": 5, "the": 6, "quick": 7}'),
                    ("merges.txt", "\n")):
        open(os.path.join(FIX, fn), "w").write(txt)


def main() -> int:
    build_fixture()
    from lumina_framework.chat import ChatModel, generate
    m_slow = ChatModel(FIX, n_heads=4, n_kv=2, fast=False)
    r_slow = generate(m_slow, "hello world", max_tokens=8)
    m_fast = ChatModel(FIX, n_heads=4, n_kv=2, fast=True)
    r_fast = generate(m_fast, "hello world", max_tokens=8)
    check("fast path produces identical tokens",
          r_slow["text"] == r_fast["text"],
          f"{r_slow['text']!r} vs {r_fast['text']!r}")
    # second generation reuses the cache — must stay identical too
    r_fast2 = generate(m_fast, "hello world", max_tokens=8)
    check("fast path stable across regenerations",
          r_fast2["text"] == r_fast["text"])
    # memory contract: fp16 originals of converted matrices are released
    gone = all(f"model.layers.0.self_attn.{p}_proj.weight" not in m_fast.t
               for p in ("q", "k", "v", "o"))
    check("fp16 originals released after conversion", gone)
    # indicative micro-timing only (tiny fixture — NOT a speed claim)
    t0 = time.perf_counter()
    generate(m_slow, "hello world", max_tokens=4)
    t_slow = time.perf_counter() - t0
    m_fast2 = ChatModel(FIX, n_heads=4, n_kv=2, fast=True)
    t0 = time.perf_counter()
    generate(m_fast2, "hello world", max_tokens=4)
    t_fast = time.perf_counter() - t0
    print(f"  indicative tiny-scale: slow {t_slow*1000:.1f}ms fast {t_fast*1000:.1f}ms "
          "(real speed measured on Kaggle notebook)")
    print("---")
    if FAILS:
        print("RESULT: FAIL (%d)" % len(FAILS))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
