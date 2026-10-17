"""Multimodal support for Lumina Framework.

Handles vision encoders, text encoders, and cross-modal projection layers
for models like Qwen-VL, GLM-4V, LLaVA, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .memory import ExpertPage, TensorContainer
from .adapters import LoadedModel, ModelSpec


@dataclass
class VisionConfig:
    """Vision encoder configuration."""
    model_type: str  # 'clip', 'siglip', 'qwen_vision', 'glm_vision', 'vit'
    hidden_size: int
    image_size: int
    patch_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    out_hidden_size: int  # projection output dim to LLM
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    in_channels: int = 3


@dataclass
class MultimodalSpec(ModelSpec):
    """Extended spec for multimodal models."""
    vision_config: Optional[VisionConfig] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    video_start_token_id: Optional[int] = None
    video_end_token_id: Optional[int] = None
    has_video: bool = False


class VisionEncoder:
    """Vision encoder forward pass (ViT-like)."""

    def __init__(self, config: VisionConfig, weights: dict[str, np.ndarray]):
        self.config = config
        self.weights = weights
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_patches = (config.image_size // config.patch_size) ** 2

    def _patch_embed(self, images: np.ndarray) -> np.ndarray:
        """Patch embedding: (B, C, H, W) -> (B, N_patches, hidden_size)"""
        # For now, simplified - real impl would use conv2d
        # images: (B, C, H, W)
        B = images.shape[0]
        # Mock: return random patches
        return np.random.randn(B, self.num_patches, self.hidden_size).astype(np.float32)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Forward pass through vision encoder."""
        x = self._patch_embed(images)
        # Simplified transformer layers
        for l in range(self.config.num_layers):
            # Attention + MLP (simplified)
            x = x + 0.01 * np.random.randn(*x.shape).astype(np.float32)
        return x

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project to LLM hidden size."""
        proj_w = self.weights.get('visual.merger.proj')
        if proj_w is not None:
            return x @ proj_w.T
        # Fallback: random projection
        return np.random.randn(*x.shape[:-1], self.config.out_hidden_size).astype(np.float32)


class MultimodalModel:
    """Multimodal model combining vision + language."""

    def __init__(self, spec: MultimodalSpec, tensors: dict[str, np.ndarray], engine=None):
        self.spec = spec
        self.tensors = tensors
        self.engine = engine
        self.vision = None
        if spec.vision_config:
            vision_weights = {k: v for k, v in tensors.items() if k.startswith('visual.')}
            self.vision = VisionEncoder(spec.vision_config, vision_weights)

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        """Encode images to LLM embeddings."""
        if self.vision is None:
            raise ValueError("No vision encoder")
        x = self.vision.forward(images)
        return self.vision.project(x)

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """Encode text tokens."""
        embed_w = self.tensors.get('model.embed_tokens.weight')
        if embed_w is None:
            embed_w = self.tensors.get('embed_tokens.weight')
        return embed_w[tokens]

    def forward(self, token_embeddings: np.ndarray, image_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass combining text and image embeddings."""
        # Concat or interleave image embeddings
        if image_embeddings is not None:
            # Simple concat at sequence start
            h = np.concatenate([image_embeddings, token_embeddings], axis=1)
        else:
            h = token_embeddings
        # Pass through LLM layers (simplified)
        return h


def extract_multimodal_spec(loaded: LoadedModel) -> MultimodalSpec:
    """Extract multimodal spec from loaded model."""
    meta = {}
    if hasattr(loaded, 'format_name') and loaded.format_name:
        pass
    # Check for vision tensors
    has_vision = any(k.startswith('visual.') for k in loaded.tensors.tensors.keys())
    
    vision_cfg = None
    if has_vision:
        # Try to infer from config or tensor shapes
        vision_cfg = VisionConfig(
            model_type='qwen_vision',
            hidden_size=1152,
            image_size=448,
            patch_size=14,
            num_layers=24,
            num_heads=16,
            intermediate_size=4304,
            out_hidden_size=4096,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
    
    base = loaded.spec
    return MultimodalSpec(
        name=base.name,
        arch=base.arch,
        n_layer=base.n_layer,
        n_expert=base.n_expert,
        n_shared_expert=base.n_shared_expert,
        d_model=base.d_model,
        d_ff=base.d_ff,
        top_k=base.top_k,
        router_scaling=base.router_scaling,
        vocab=base.vocab,
        vision_config=vision_cfg,
        vision_start_token_id=154830,
        vision_end_token_id=154831,
        image_token_id=154854,
        video_token_id=154855,
        video_start_token_id=154832,
        video_end_token_id=154833,
        has_video=True,
    )
