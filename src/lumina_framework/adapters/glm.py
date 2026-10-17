"""GLM adapter — Zhipu GLM family (GLM-4/5/5.3, chatglm + MoE/DSA).

Detects a HuggingFace directory (config.json model_type starting with
`glm`/`chatglm`) or a safetensors file with such a sibling config, and builds
an accurate ModelSpec from the real config: layers, routed experts, top-k,
hidden sizes, vocab. Weights load as FP reference; MLA/DSA attention is
executed through the framework's MLACache path.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from ..memory import ExpertPage, TensorContainer
from . import LoadedModel, ModelSpec, TensorTable, register
from .safetensors import SafetensorsAdapter

GLM_TYPES = ("glm", "chatglm", "glm_moe", "glm5_next", "glm_moe_dsa")


def _cfg_of(path: str) -> tuple[dict, str] | tuple[None, None]:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    cj = os.path.join(d, "config.json")
    if not os.path.isfile(cj):
        return None, None
    try:
        cfg = json.load(open(cj))
    except Exception:
        return None, None
    return cfg, d


def _text_cfg(cfg: dict) -> dict:
    if isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg


def _is_glm(cfg: dict) -> bool:
    if not isinstance(cfg, dict):
        return False
    t = _text_cfg(cfg)
    mt = str(t.get("model_type", "")).lower()
    arch = " ".join(t.get("architectures", [])).lower()
    return mt.startswith(GLM_TYPES) or "glm" in arch or "chatglm" in arch


class GLMAdapter:
    name = "glm"
    extensions = (".safetensors", "")

    @staticmethod
    def probe(path: str) -> bool:
        cfg, d = _cfg_of(path)
        if cfg is None or not _is_glm(cfg):
            return False
        if os.path.isdir(path):
            return any(glob.glob(os.path.join(d, "*.safetensors")))
        return path.endswith(".safetensors")

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        cfg, d = _cfg_of(path)
        t = _text_cfg(cfg or {})
        shards = ([path] if path.endswith(".safetensors")
                  else sorted(glob.glob(os.path.join(d, "*.safetensors"))))
        if not shards:
            raise ValueError("glm: no safetensors shards found")
        tensors: dict[str, np.ndarray] = {}
        bpw: dict[str, float] = {}
        for sh in shards:
            lm = SafetensorsAdapter.load(sh)
            for k, v in lm.tensors.tensors.items():
                if k not in tensors:
                    tensors[k] = v
                    bpw[k] = lm.tensors.source_bits_per_weight.get(k, 16.0)
        n_exp = int(t.get("n_routed_experts",
                          t.get("num_experts", t.get("n_experts", 1))))
        spec = ModelSpec(
            name="glm",
            arch=str(t.get("model_type", "glm")),
            n_layer=int(t.get("num_hidden_layers",
                              t.get("num_layers", 1))),
            n_expert=n_exp,
            n_shared_expert=int(t.get("n_shared_experts", 1)),
            d_model=int(t.get("hidden_size", 64)),
            d_ff=int(t.get("intermediate_size",
                           t.get("ffn_hidden_size", 128))),
            top_k=int(t.get("num_experts_per_tok",
                            t.get("top_k", 1))),
            vocab=int(t.get("vocab_size",
                            t.get("padded_vocab_size", 512))),
            extra={"moe_intermediate_size": t.get("moe_intermediate_size"),
                   "first_k_dense": t.get("first_k_dense_replace", 0),
                   "q_lora_rank": t.get("q_lora_rank", 0),
                   "kv_lora_rank": t.get("kv_lora_rank", 0)},
        )
        pages, dense = _build_pages(tensors, spec)
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw),
                           expert_pages=pages, dense_pages=dense,
                           format_name="glm", source_file=path,
                           notes=["GLM family import; MLA/DSA via MLACache"])


def _build_pages(tensors, spec):
    pages: dict[str, ExpertPage] = {}
    dense: dict[str, TensorContainer] = {}
    from ..memory import TensorContainer as TC
    for lyr in range(spec.n_layer):
        patr = (f"model.layers.{lyr}.mlp", f"layers.{lyr}.mlp",
                f"transformer.layers.{lyr}.mlp")
        mlp = {k: v for k, v in tensors.items()
               if any(p in k for p in patr)}
        if not mlp:
            continue
        exp = {k: v for k, v in mlp.items() if ".experts." in k}
        if exp:
            by_e: dict[int, dict] = {}
            for k, v in exp.items():
                try:
                    e = int(k.split(".experts.")[1].split(".")[0])
                except Exception:
                    continue
                by_e.setdefault(e, {})[k] = v
            for e, d_ in by_e.items():
                pages[f"L{lyr}-e{e}"] = ExpertPage(
                    f"L{lyr}-e{e}",
                    {p: TC(d_[p]) for p in sorted(d_)}, lyr)
        else:
            pages[f"L{lyr}-mlp"] = ExpertPage(
                f"L{lyr}-mlp",
                {p: TC(mlp[p]) for p in sorted(mlp)}, lyr, arch="swiglu")
    skip = ("experts.", "shared", "router")
    for name, arr in tensors.items():
        if any(s in name for s in skip):
            continue
        if arr.ndim >= 1 and arr.size >= 1 and arr.dtype.kind == "f":
            dense[name] = TC(arr)
    return pages, dense


register(GLMAdapter)
