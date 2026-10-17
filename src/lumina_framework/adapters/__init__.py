"""Format-agnostic model adapters.

The Lumina Framework never reads a model format directly. It consumes:
  * ModelSpec      — architecture description
  * TensorTable    — named FP reference tensors
  * ExpertBank     — named ExpertPages (quantized containers)

Every format adapter converts its file into that contract. The framework's
execution is therefore identical for every format; the *only* thing a format
changes is how many bits survived to reach the framework, which the adapter
reports honestly.

Adding a new format = one module + one entry in REGISTRY. Nothing else changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..memory import ExpertPage, TensorContainer


@dataclass
class ModelSpec:
    """Architecture description sufficient to build an executable model."""

    name: str
    arch: str = "swiglu-moe"
    n_layer: int = 1
    n_expert: int = 1
    n_shared_expert: int = 1
    d_model: int = 64
    d_ff: int = 128
    top_k: int = 1
    router_scaling: float = 1.0
    router_dtype: str = "sigmoid"
    vocab: int = 512
    extra: dict = field(default_factory=dict)


@dataclass
class TensorTable:
    """Named FP reference tensors plus the precision the format actually gave us."""

    tensors: dict[str, np.ndarray]
    source_bits_per_weight: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.tensors[k]

    def names(self) -> list[str]:
        return list(self.tensors.keys())


@dataclass
class LoadedModel:
    spec: ModelSpec
    tensors: TensorTable
    expert_pages: dict[str, ExpertPage]
    dense_pages: dict[str, TensorContainer]
    format_name: str
    source_file: str
    notes: list[str] = field(default_factory=list)


class ModelAdapter(Protocol):
    """Every format module implements exactly this."""

    name: str
    extensions: tuple[str, ...]

    @staticmethod
    def probe(path: str) -> bool:
        """Return True if this adapter owns `path` (magic bytes / extension)."""
        ...

    @staticmethod
    def load(path: str, **kw) -> LoadedModel:
        """Read the file into the framework contract."""
        ...


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

_ADAPTERS: list[ModelAdapter] = []


def register(adapter: ModelAdapter) -> ModelAdapter:
    _ADAPTERS.append(adapter)
    return adapter


def _lazy_import() -> None:
    """Import adapter modules so their @register decorators run."""
    # Order matters: more specific formats first
    from . import glm as _       # noqa: F401 (GLM family)
    from . import qwen4_exp as _  # noqa: F401 (Qwen3.8-Flash-Next family)
    from . import trtllm as _    # noqa: F401 (TensorRT-LLM ckpt)
    from . import bnb as _      # noqa: F401 (BitsAndBytes NF4)
    from . import exl3 as _     # noqa: F401 (EXL3 quantized)
    from . import gguf as _     # noqa: F401 (GGUF)
    from . import lumina as _   # noqa: F401 (Lumina native)
    from . import safetensors as _  # noqa: F401 (FP source)


def detect_format(path: str) -> str | None:
    """Probe magic bytes first, then extension."""
    _lazy_import()
    for a in _ADAPTERS:
        try:
            if a.probe(path):
                return a.name
        except Exception:
            continue
    return None


def available_formats() -> list[str]:
    _lazy_import()
    return [a.name for a in _ADAPTERS]


def load_model(path: str, **kw) -> LoadedModel:
    """Detect the format of `path` and load it through the right adapter."""
    _lazy_import()
    probe_errors = []
    for a in _ADAPTERS:
        try:
            if a.probe(path):
                return a.load(path, **kw)
        except Exception as e:  # noqa: BLE001
            probe_errors.append(f"{a.name}: {e}")
    raise ValueError(
        f"No adapter could read {path!r}. registered={available_formats()}; "
        f"errors={probe_errors}"
    )
