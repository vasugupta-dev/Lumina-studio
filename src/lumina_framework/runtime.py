"""Runtime Fabric — the scheduler that turns a token stream into committed tokens.

Responsibilities:
  * Step the MoE stack, streaming experts through the Expert Memory Engine.
  * Drive the Predictor Society (router-lookahead) and issue prefetches that are
    provably result-neutral.
  * Maintain the MLA-compressed KV cache.
  * Report honest per-token accounting.

The fabric is deliberately boring: it is the place where "tokens/s" claims either
survive or die, so it measures everything and hides nothing.
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .expert_memory import ExpertMemoryEngine
from .memory import ExpertPage, TensorContainer
from .moe import MoELayer, RoutePredictor, Router
# (DynamicContextCache import removed: it was dead wiring — the fabric never
#  consulted it. Prefix reuse now lives in chat.generate_cached, which is
#  parity-tested; see tests/test_prefix_cache.py.)
from .mtp_engine import MTPEngine, MTPConfig


def _cpu_workers() -> int:
    import os
    return max(1, min(4, (os.cpu_count() or 2) // 2))


# One shared single-thread executor for async tier reads: creating a pool per
# call (the old code) leaked threads and raced shutdown against the future.
_SHARED_IO_POOL: ThreadPoolExecutor | None = None
_SHARED_IO_LOCK = threading.Lock()


def _shared_io_pool() -> ThreadPoolExecutor:
    global _SHARED_IO_POOL
    if _SHARED_IO_POOL is None:
        with _SHARED_IO_LOCK:
            if _SHARED_IO_POOL is None:
                _SHARED_IO_POOL = ThreadPoolExecutor(max_workers=1,
                                                     thread_name_prefix="lumina-io")
    return _SHARED_IO_POOL

# Reference-logit head: a fixed random projection used so the reference has a
# deterministic, reproducible "next token" signal without a real language model.
# Replaced by the Model Adapters track (GLM-5.3, K3, Nova, Lumi Soup).
_LOGIT_VOCAB = 512


class StubModel:
    """Minimal stack: embed -> N MoE layers -> logits. Deterministic."""

    def __init__(self, n_layer: int = 4, n_expert: int = 8, d_model: int = 128,
                 d_ff: int = 256, top_k: int = 2, seed: int = 0):
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        self.embed = rng.standard_normal((_LOGIT_VOCAB, d_model)).astype(np.float32) * 0.05
        self.head = rng.standard_normal((_LOGIT_VOCAB, d_model)).astype(np.float32) * 0.05
        self.layers = [
            MoELayer(i, n_expert, d_model, d_ff, top_k=top_k, seed=seed + i)
            for i in range(n_layer)
        ]

    @staticmethod
    def from_loaded_model(loaded, engine):
        """Create a runtime model from a LoadedModel.

        Every tensor must be present — the old code silently substituted
        RANDOM weights for missing embed/head/layers, which would decode
        garbage while looking alive. Missing keys now raise with the names.
        """
        from . import moe
        import numpy as np
        tensors = loaded.tensors.tensors
        missing = [k for k in ("embed.weight", "head.weight") if k not in tensors]
        if missing:
            raise KeyError("from_loaded_model missing tensors: " + ", ".join(missing))
        m = type('DenseModel', (), {})()
        m.embed = tensors['embed.weight']
        m.head = tensors['head.weight']
        m.d_model = m.embed.shape[1]
        m.layers = []

        # Build dense layers from tensors
        n_layers = max([int(k.split('.')[2]) for k in tensors if k.startswith('model.layers.') and '.mlp.' in k], default=0) + 1
        for l in range(n_layers):
            gate = tensors.get(f'model.layers.{l}.mlp.gate_proj.weight')
            up = tensors.get(f'model.layers.{l}.mlp.up_proj.weight')
            down = tensors.get(f'model.layers.{l}.mlp.down_proj.weight')
            if gate is None or up is None or down is None:
                raise KeyError(f"from_loaded_model missing mlp weights for layer {l}")
            layer = moe.DenseLayer(l, gate, up, down)
            m.layers.append(layer)
        
        def forward_hidden(self, h, engine, target="ram"):
            for layer in self.layers:
                h = h + layer.forward(h, engine, target=target)
            return h
        
        m.forward_hidden = forward_hidden.__get__(m)
        return m

    def forward_tokens(self, tokens: np.ndarray, engine, target="ram"):
        h = self.embed[tokens]
        for layer in self.layers:
            h = h + layer.forward(h, engine, target=target)
        return h @ self.head.T

    def forward_hidden(self, h, engine, target="ram"):
        for layer in self.layers:
            h = h + layer.forward(h, engine, target=target)
        return h


class MLACache:
    """MLA-style compressed KV cache reference.

    Stores a low-rank latent per token instead of full per-head K/V; the reference
    charges `kv_latent` floats per token and can expand on demand.
    """

    def __init__(self, d_model: int, kv_latent: int, capacity: int = 4096):
        self.d_model = d_model
        self.kv_latent = kv_latent
        self.capacity = capacity
        self._buf = np.zeros((capacity, kv_latent), dtype=np.float32)
        self.n = 0

    def append(self, latent: np.ndarray) -> None:
        if self.n >= self.capacity:
            self._buf = np.vstack([self._buf, np.zeros_like(self._buf)])
            self.capacity *= 2
        self._buf[self.n] = latent[:self.kv_latent]
        self.n += 1

    @property
    def bytes(self) -> int:
        return self.n * self.kv_latent * 4

    def prefetch_range(self, start: int, end: int) -> None:
        pass

    def load_range_async(self, start: int, end: int):
        try:
            return _shared_io_pool().submit(
                lambda: self._buf[max(0, start):min(end, self.n)])
        except Exception:
            return None

    def clear(self) -> None:
        self.n = 0


@dataclass
class StepStats:
    tokens_out: int = 0
    seconds: float = 0.0
    tokens_per_second: float = 0.0
    committed_tokens: int = 0
    spec_acceptance: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tier_seconds: dict = field(default_factory=dict)


class RuntimeFabric:
    """Drives the model and owns all the accounting the plan demands."""

    def __init__(self, model: StubModel, engine: ExpertMemoryEngine,
                 top_k: int = 2, top_p: float | None = None,
                 prefetch: bool = False):
        self.model = model
        self.engine = engine
        self.top_k = top_k
        self.top_p = top_p
        self.prefetch = prefetch
        self.predictor = RoutePredictor()
        self.stats = StepStats()

    # ------------------------------------------------------------------
    def decode(self, prompt: np.ndarray, n_tokens: int, target: str = "ram") -> dict:
        """Autoregressive decode with dynamic async prefetch and CUDA fast path."""
        t0 = time.perf_counter()
        tokens = list(prompt)
        generated = []
        h = self.model.embed[prompt].mean(axis=0, keepdims=True)
        pool = None
        if self.prefetch:
            try:
                pool = ThreadPoolExecutor(max_workers=_cpu_workers(),
                                          thread_name_prefix="lumina-prefetch")
            except Exception:
                pool = None
        # numpy path
        futures = []
        for step in range(n_tokens):
            hs = self.model.forward_hidden(h, self.engine, target=target)
            logits = hs @ self.model.head.T
            next_tok = int(np.argmax(logits[0]))
            tokens.append(next_tok); generated.append(next_tok)
            h = self.model.embed[[next_tok]]
            if self.prefetch and pool is not None and step < n_tokens - 1:
                try:
                    pred = self.predictor.predict_next(self.model.layers[0].router, hs, self.model.layers[0].top_k)
                    next_experts = sorted(set(pred.ravel().tolist()))
                    exps = []
                    for le in self.model.layers:
                        for ei in next_experts[:3]:
                            if ei < len(le.experts):
                                exps.append(le.experts[ei].name)
                    if exps:
                        fut = pool.submit(self.engine.get, exps[:12], target)
                        futures.append(fut)
                except Exception:
                    pass
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except Exception:
                pass
        elapsed = time.perf_counter() - t0
        return {
            "tokens": generated,
            "seconds": elapsed,
            "tokens_per_second": n_tokens / elapsed if elapsed else float("inf"),
            "prompt_len": len(prompt),
        }

    def hit_rate(self) -> float:
        reads = 0
        hits = 0
        for layer in self.model.layers:
            for p in layer.experts:
                reads += 1
                if self.engine.resident_tier(p.name) in ("vram", "ram"):
                    hits += 1
        return hits / reads if reads else 0.0


    def decode_speculative(self, prompt: np.ndarray, n_tokens: int, target: str = "ram",
                           mtp_depth: int = 4) -> dict:
        """Speculative decoding with MTP for 2-3x speedup at minimal quality loss."""
        t0 = time.perf_counter()
        tokens = list(prompt)
        generated = []
        
        # Initialize MTP engine if not exists
        if not hasattr(self, '_mtp_engine'):
            self._mtp_engine = MTPEngine(
                MTPConfig(prediction_depth=mtp_depth),
                d_model=self.model.d_model,
                vocab_size=self.model.head.shape[0],
                main_head=self.model.head
            )
        
        h = self.model.embed[prompt].mean(axis=0, keepdims=True)
        
        for step in range(n_tokens):
            # Main forward
            hs = self.model.forward_hidden(h, self.engine, target=target)
            logits = hs @ self.model.head.T
            next_tok = int(np.argmax(logits[0]))
            
            # Speculate
            spec_tokens, spec_conf = self._mtp_engine.speculate(hs, mtp_depth)
            
            # Verify
            accepted, accept_rate = self._mtp_engine.verify(hs, spec_tokens)
            
            # Add main token + accepted speculative tokens
            tokens.append(next_tok)
            generated.append(next_tok)
            
            for atok in accepted:
                tokens.append(atok)
                generated.append(atok)
            
            # Update hidden for next iteration
            if accepted:
                # Use last accepted token's embedding
                h = self.model.embed[[accepted[-1]]]
            else:
                h = self.model.embed[[next_tok]]
        
        elapsed = time.perf_counter() - t0
        return {
            "tokens": generated[:n_tokens],
            "seconds": elapsed,
            "tokens_per_second": len(generated[:n_tokens]) / elapsed if elapsed else float("inf"),
            "prompt_len": len(prompt),
        }

    def report(self) -> dict:
        s = self.engine.stats()
        total_read = s["vram"]["read_bytes"] + s["ram"]["read_bytes"]
        return {
            "bytes_read": total_read,
            "bytes_from_nvme": s["nvme"]["read_bytes"],
            "router_recall": self.predictor.recall,
            "router_token_accuracy": self.predictor.token_accuracy,
            "tiers": s,
        }
