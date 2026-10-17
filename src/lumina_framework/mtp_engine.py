"""Multi-Token Prediction (MTP) Engine for speculative decoding.

Implements the MTP approach from GLM-4/5 and other recent models:
- Small auxiliary head predicts next N tokens in parallel
- Verification via main model
- Acceptance rate determines speedup
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class MTPConfig:
    num_mtp_layers: int = 1
    mtp_hidden_size: int = 1024
    prediction_depth: int = 4  # predict 4 tokens ahead
    acceptance_threshold: float = 0.5


class MTPHead:
    """Single MTP layer: takes hidden state, predicts next token embedding."""
    
    def __init__(self, d_model: int, vocab_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.norm = rng.standard_normal((d_model,)).astype(np.float32) * 0.01
        self.output = rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02
    
    def forward(self, h: np.ndarray) -> np.ndarray:
        # h: (1, d_model) or (seq_len, d_model)
        h_norm = h * self.norm  # simple layer norm approx
        h_proj = h_norm @ self.proj.T
        # ReLU activation
        h_proj = np.maximum(h_proj, 0)
        logits = h_proj @ self.output.T
        return logits


class MTPEngine:
    """Manages MTP heads and speculative decoding."""
    
    def __init__(self, config: MTPConfig, d_model: int, vocab_size: int, main_head: np.ndarray):
        self.config = config
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.main_head = main_head
        self.mtp_heads = [
            MTPHead(d_model, vocab_size, seed=42 + i)
            for i in range(config.num_mtp_layers)
        ]
    
    def speculate(self, hidden: np.ndarray, depth: Optional[int] = None) -> tuple[List[int], List[float]]:
        """Generate speculative tokens from MTP heads.
        
        Returns (predicted_tokens, confidence_scores)
        """
        depth = depth or self.config.prediction_depth
        tokens = []
        confidences = []
        h = hidden
        
        for _ in range(depth):
            # Use last MTP head for deepest prediction
            logits = self.mtp_heads[-1].forward(h)
            probs = self._softmax(logits[0])
            next_token = int(np.argmax(probs))
            confidence = float(probs[next_token])
            
            tokens.append(next_token)
            confidences.append(confidence)
            
            # Update hidden for next step (simplified)
            h = h @ self.mtp_heads[-1].proj.T
            h = np.maximum(h, 0)
        
        return tokens, confidences
    
    def verify(self, hidden: np.ndarray, speculative_tokens: List[int]) -> tuple[List[int], float]:
        """Verify speculative tokens using main model head.
        
        Returns (accepted_tokens, acceptance_rate)
        """
        h = hidden
        accepted = []
        for tok in speculative_tokens:
            logits = h @ self.main_head.T
            probs = self._softmax(logits[0])
            pred = int(np.argmax(probs))
            if pred == tok:
                accepted.append(tok)
                h = h + 0.01  # simplified state update
            else:
                break
        acceptance_rate = len(accepted) / len(speculative_tokens) if speculative_tokens else 0.0
        return accepted, acceptance_rate
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

