"""TensorRT-LLM checkpoint adapter (import path, CPU/GPU-agnostic).

Reads TRT-LLM checkpoints as stored on disk: a directory with `config.json`
(architecture + mapping + quantization blocks) and rank safetensors shards.
Weights load as FP reference into the framework contract; execution uses the
Lumina runtime. TRT-LLM's own GPU runtime is NOT required and NOT bundled —
the adapter reports that honestly in `notes`.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from ..memory import ExpertPage, TensorContainer
from . import LoadedModel, ModelSpec, TensorTable, register
from .safetensors import SafetensorsAdapter


def _cfg_of(path: str) -> dict | None:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    cj = os.path.join(d, "config.json")
    if not os.path.isfile(cj):
        return None
    try:
        return json.load(open(cj))
    except Exception:
        return None


def _is_trtllm(cfg: dict) -> bool:
    if not isinstance(cfg, dict):
        return False
    blob = json.dumps(cfg).lower()
    return ("tensorrt_llm" in blob or "tensorrt-llm" in blob
            or ("mapping" in cfg and "quantization" in cfg))


class TRTLLMAdapter:
    name = "trtllm"
    extensions = (".safetensors", "")

    @staticmethod
    def probe(path: str) -> bool:
        cfg = _cfg_of(path)
        if cfg is None:
            return False
        if _is_trtllm(cfg):
            if os.path.isdir(path):
                return any(glob.glob(os.path.join(path, "*.safetensors")))
            return path.endswith(".safetensors")
        return False

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        d = path if os.path.isdir(path) else os.path.dirname(path)
        cfg = _cfg_of(path) or {}
        shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        if not shards:
            raise ValueError("trtllm: no safetensors shards found")
        tensors: dict[str, np.ndarray] = {}
        bpw: dict[str, float] = {}
        for sh in shards:
            lm = SafetensorsAdapter.load(sh)
            for k, v in lm.tensors.tensors.items():
                if k not in tensors:
                    tensors[k] = v
                    bpw[k] = lm.tensors.source_bits_per_weight.get(k, 16.0)
        n_layer = int(cfg.get("num_hidden_layers", 1))
        spec = ModelSpec(
            name=cfg.get("architecture", "trtllm-model"),
            arch="swiglu-moe",
            n_layer=n_layer,
            n_expert=int(cfg.get("num_experts", 1)),
            n_shared_expert=int(cfg.get("num_shared_experts", 1)),
            d_model=int(cfg.get("hidden_size", 64)),
            d_ff=int(cfg.get("intermediate_size", 128)),
            top_k=int(cfg.get("num_experts_per_tok",
                              cfg.get("top_k", 1))),
            vocab=int(cfg.get("vocab_size", 512)),
            extra={"trtllm_mapping": cfg.get("mapping", {}),
                   "trtllm_quant": cfg.get("quantization", {})},
        )
        pages, dense = _build_pages(tensors, spec)
        notes = ["TRT-LLM checkpoint import; executed by Lumina runtime, "
                 "not by TensorRT (needs NVIDIA GPU + trtllm runtime)"]
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw),
                           expert_pages=pages, dense_pages=dense,
                           format_name="trtllm", source_file=path, notes=notes)


def _build_pages(tensors, spec):
    pages: dict[str, ExpertPage] = {}
    dense: dict[str, TensorContainer] = {}
    from ..memory import TensorContainer as TC
    for lyr in range(spec.n_layer):
        shared = {k: v for k, v in tensors.items()
                  if f".{lyr}.shared." in k}
        if shared:
            pages[f"L{lyr}-shared"] = ExpertPage(
                f"L{lyr}-shared",
                {p: TC(shared[p]) for p in sorted(shared)}, lyr,
                arch="shared")
        for e in range(spec.n_expert):
            ex = {k: v for k, v in tensors.items()
                  if f".{lyr}.experts.{e}." in k}
            if ex:
                pages[f"L{lyr}-e{e}"] = ExpertPage(
                    f"L{lyr}-e{e}",
                    {p: TC(ex[p]) for p in sorted(ex)}, lyr)
    skip = ("experts.", "shared", "router")
    for name, arr in tensors.items():
        if any(s in name for s in skip):
            continue
        if arr.ndim >= 1 and arr.size >= 1 and arr.dtype.kind == "f":
            dense[name] = TC(arr)
    return pages, dense


register(TRTLLMAdapter)
