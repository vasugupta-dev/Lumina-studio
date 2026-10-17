"""Shared test fixtures for the new-generation suites (compaction/grounding).

The tiny Qwen-format chat model is rebuilt if /tmp was cleared (phone-safe,
~1 MB of weights). Same builder as tests/test_prefix_cache.py — that file
keeps its own copy on purpose (existing suites stay untouched).
"""
import json
import os

import numpy as np

MODEL_DIR = "/tmp/tiny-chat"


def build_tiny_chat(model_dir: str = MODEL_DIR) -> str:
    """Rebuild the tiny chat model if missing. Returns the model dir."""
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return model_dir
    from safetensors.numpy import save_file
    os.makedirs(model_dir, exist_ok=True)
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
              os.path.join(model_dir, "model.safetensors"))
    json.dump({"num_hidden_layers": n_layer, "hidden_size": d,
               "num_attention_heads": nh, "num_key_value_heads": nk,
               "vocab_size": vocab},
              open(os.path.join(model_dir, "config.json"), "w"))
    for fn, txt in (("vocab.json", '{"<s>": 0, "</s>": 1, "a": 2, "b": 3, "hello": 4, "world": 5, "the": 6, "quick": 7}'),
                    ("merges.txt", "\n")):
        open(os.path.join(model_dir, fn), "w").write(txt)
    return model_dir
