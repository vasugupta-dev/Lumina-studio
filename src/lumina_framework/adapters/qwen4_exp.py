"""Qwen4-exp adapter — Qwen3.8-Flash-Next family (hybrid Gated DeltaNet + QSA MoE).

Probed from the real HF config (model_type "qwen4_exp", researched 2026-10-16):
48 layers in 12x[3x(Gated DeltaNet -> MoE) + 1x(QSA sparse -> MoE)], 512 routed
experts + shared expert per layer (top-10), 20M-entry n-gram table, 1-layer MTP
head, vision encoder, BF16, 180B params over ~131 shards.

Two modes:
  * meta_only=True (default for inventory): reads config + shard HEADERS only
    (names/shapes/dtypes, zero weight bytes). Safe on any machine.
  * full load: reads all shards into FP reference tensors. Needs ~360GB RAM
    for the real model — refuses otherwise with the measured shortfall.

Chat execution is NOT supported: Gated DeltaNet recurrence, the QSA indexer,
the n-gram table path and the MTP head have no verified kernels here. The
server refuses qwen4_exp with that exact list instead of decoding garbage.
"""

from __future__ import annotations

import glob
import json
import os
import struct

import numpy as np

from ..memory import ExpertPage, TensorContainer
from . import LoadedModel, ModelSpec, TensorTable, register
from .safetensors import SafetensorsAdapter

QWEN4_TYPES = ("qwen4_exp",)
# layer types the framework cannot execute yet (exact gap list, shared with serve.py)
QWEN4_UNSUPPORTED = ("gated_deltanet recurrence", "QSA sparse indexer",
                     "n-gram embedding table path", "MTP draft head",
                     "vision encoder")


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


def _is_qwen4(cfg: dict) -> bool:
    if not isinstance(cfg, dict):
        return False
    t = _text_cfg(cfg)
    mt = str(t.get("model_type", "")).lower()
    arch = " ".join(t.get("architectures", [])).lower()
    return mt in QWEN4_TYPES or "qwen4exp" in arch.replace("_", "")


def shard_headers(shards: list[str]) -> dict[str, dict]:
    """Name -> {shape, dtype, shard} from headers only (no weight bytes)."""
    out: dict[str, dict] = {}
    for sh in shards:
        try:
            with open(sh, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
        except Exception:
            continue
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k not in out:
                out[k] = {"shape": list(v.get("shape", [])),
                          "dtype": v.get("dtype", "?"), "shard": sh}
    return out


def layer_inventory(t: dict) -> list[str]:
    """Per-layer attention kind from the real config (linear/full pattern)."""
    lt = t.get("layer_types") or []
    inv = []
    for x in lt:
        inv.append("gated_deltanet" if "linear" in str(x) else
                   "qsa_sparse" if "full" in str(x) else str(x))
    return inv


class Qwen4ExpAdapter:
    name = "qwen4_exp"
    extensions = (".safetensors", "")

    @staticmethod
    def probe(path: str) -> bool:
        cfg, d = _cfg_of(path)
        if cfg is None or not _is_qwen4(cfg):
            return False
        if os.path.isdir(path):
            return any(glob.glob(os.path.join(d, "*.safetensors")))
        return path.endswith(".safetensors")

    @staticmethod
    def load(path: str, meta_only: bool = True, **kw) -> LoadedModel:
        cfg, d = _cfg_of(path)
        t = _text_cfg(cfg or {})
        shards = ([path] if path.endswith(".safetensors")
                  else sorted(glob.glob(os.path.join(d, "*.safetensors"))))
        if not shards:
            raise ValueError("qwen4_exp: no safetensors shards found")
        n_exp = int(t.get("num_experts", 512))
        spec = ModelSpec(
            name="qwen4_exp",
            arch=str(t.get("model_type", "qwen4_exp")),
            n_layer=int(t.get("num_hidden_layers", 48)),
            n_expert=n_exp,
            n_shared_expert=1,
            d_model=int(t.get("hidden_size", 2560)),
            d_ff=int(t.get("moe_intermediate_size", 640)),
            top_k=int(t.get("num_experts_per_tok", 10)),
            vocab=int(t.get("vocab_size", 248320)),
            extra={"layer_inventory": layer_inventory(t),
                   "ngram_vocab": t.get("ngram_vocab_size_base", 20000000),
                   "mtp_layers": t.get("mtp_num_hidden_layers", 1),
                   "unsupported": list(QWEN4_UNSUPPORTED)},
        )
        if meta_only:
            heads = shard_headers(shards)
            n_params = 0
            bpw: dict[str, float] = {}
            for k, v in heads.items():
                n_params += int(np.prod(v["shape"])) if v["shape"] else 0
                bpw[k] = {"F32": 32.0, "F16": 16.0, "BF16": 16.0}.get(v["dtype"], 16.0)
            notes = ["header-only inventory: 0 weight bytes read",
                     "chat unsupported: " + ", ".join(QWEN4_UNSUPPORTED),
                     f"tensors={len(heads)} shards={len(shards)} "
                     f"params~{n_params / 1e9:.1f}B"]
            return LoadedModel(spec=spec, tensors=TensorTable({}, bpw),
                               expert_pages={}, dense_pages={},
                               format_name="qwen4_exp", source_file=path,
                               notes=notes)
        # full load: needs room for every byte
        need_gb = sum(os.path.getsize(s) for s in shards) / 1e9
        avail_gb = None
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) / 1024 / 1024
        except Exception:
            pass
        if avail_gb is not None and need_gb * 1.5 > avail_gb:
            raise MemoryError(
                f"qwen4_exp full load needs ~{need_gb:.0f}GB weights "
                f"(+50% working), only {avail_gb:.1f}GB free")
        tensors: dict[str, np.ndarray] = {}
        bpw = {}
        for sh in shards:
            lm = SafetensorsAdapter.load(sh)
            for k, v in lm.tensors.tensors.items():
                if k not in tensors:
                    tensors[k] = v
                    bpw[k] = lm.tensors.source_bits_per_weight.get(k, 16.0)
        pages, dense = _build_pages(tensors, spec)
        return LoadedModel(spec=spec, tensors=TensorTable(tensors, bpw),
                           expert_pages=pages, dense_pages=dense,
                           format_name="qwen4_exp", source_file=path,
                           notes=["full FP load; chat still unsupported: "
                                  + ", ".join(QWEN4_UNSUPPORTED)])

    @staticmethod
    def save(*a, **k):
        raise NotImplementedError("qwen4_exp is read-only (180B source)")


def _build_pages(tensors, spec):
    pages: dict[str, ExpertPage] = {}
    dense: dict[str, TensorContainer] = {}
    from ..memory import TensorContainer as TC
    for lyr in range(spec.n_layer):
        exp = {k: v for k, v in tensors.items() if f".{lyr}." in k and ".experts." in k}
        by_e: dict[int, dict] = {}
        for k, v in exp.items():
            try:
                e = int(k.split(".experts.")[1].split(".")[0])
            except Exception:
                continue
            by_e.setdefault(e, {})[k] = v
        for e, d_ in by_e.items():
            pages[f"L{lyr}-e{e}"] = ExpertPage(
                f"L{lyr}-e{e}", {p: TC(d_[p]) for p in sorted(d_)}, lyr)
    skip = ("experts.", "shared", "router")
    for name, arr in tensors.items():
        if any(s in name for s in skip):
            continue
        if arr.ndim >= 1 and arr.size >= 1 and arr.dtype.kind == "f":
            dense[name] = TC(arr)
    return pages, dense


register(Qwen4ExpAdapter)
