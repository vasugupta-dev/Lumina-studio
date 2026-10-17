"""Gain-shape residual binary tree — sub-1-bit with strictly binary splits.

Grounded in GSRQ (gain-shape residual quantization) + AF1 (genuine 1-bit
accounting): split magnitude (gain, shared per group) from direction (shape,
pure +/-1 signs — NEVER ternary, no zero level), then encode the RESIDUAL
with further binary stages arranged as a tree. Sub-1-bit average rate comes
from two honest sources: (1) best-first expansion (only high-error leaves get
deeper levels, EAGLE-2 style); (2) Huffman-coded leaf signs (peaked dists).

Every node costs exactly: code bytes (entropy) + 1B gain + 2b structure.
"""

from __future__ import annotations

import heapq

import numpy as np

from . import entropy as H


def fit_gain_shape(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Binary shape + LS gain: S=sign(R), g=<R,S>/<S,S>. Pure +/-1."""
    S = np.where(R >= 0, 1.0, -1.0)
    g = float(np.sum(R * S) / max(np.sum(S * S), 1e-12))
    return S, g


class TreeNode:
    __slots__ = ("cols", "gain", "signs", "children")

    def __init__(self, cols: np.ndarray, gain: float = 0.0,
                 signs: np.ndarray | None = None):
        self.cols = cols          # column indices owned by this node
        self.gain = gain
        self.signs = signs        # +/-1 over (rows, len(cols)) or None
        self.children: list["TreeNode"] = []


def build_tree(W: np.ndarray, max_nodes: int = 7,
               min_cols: int = 8) -> TreeNode:
    """Best-first binary residual tree. Root owns all columns, gain-only."""
    W = np.asarray(W, dtype=np.float64)
    root = TreeNode(np.arange(W.shape[1]))
    approx = np.zeros_like(W)
    # Root is ALWAYS fitted: it is the base binary stage (plain 1-bit).
    S, g = fit_gain_shape(W)
    root.signs, root.gain = S, g
    approx += g * S
    nodes = 1
    # max-heap by negative residual energy: (-err, tiebreak, node)
    heap: list = []
    cnt = [0]

    def push(node: TreeNode) -> None:
        R = W - approx
        err = float(np.sum(R[:, node.cols] ** 2))
        cnt[0] += 1
        heapq.heappush(heap, (-err, cnt[0], node))

    if max_nodes > 1:
        mid = root.cols.size // 2
        for half in (root.cols[:mid], root.cols[mid:]):
            child = TreeNode(half)
            root.children.append(child)
            nodes += 1
            push(child)
    fitted = 1  # root always fitted
    while heap and fitted < max_nodes:
        _, _, node = heapq.heappop(heap)
        cols = node.cols
        if cols.size < 2 * min_cols:
            continue
        R = W - approx
        S, g = fit_gain_shape(R[:, cols])
        node.signs, node.gain = S, g
        approx[:, cols] += g * S
        fitted += 1
        mid = cols.size // 2
        for half in (cols[:mid], cols[mid:]):
            child = TreeNode(half)
            node.children.append(child)
            nodes += 1
            push(child)
    return root


def _walk(node: TreeNode):
    yield node
    for c in node.children:
        yield from _walk(c)


def pack_tree(root: TreeNode, shape: tuple) -> dict:
    """Serialize: Huffman-coded +/-1 leaves + 1B gains + 2b/node structure."""
    o, i = shape
    blobs, gains, struct_bits = [], [], 0
    nnodes = 0
    for node in _walk(root):
        nnodes += 1
        struct_bits += 2  # 00=leaf, 01/10/11=child config (we store n_children)
        if node.signs is None:
            gains.append(np.float16(0.0))
            continue
        idx = (node.signs > 0).astype(np.int64).reshape(-1)
        enc = H.huffman_encode(idx, 2)
        blobs.append(enc)
        gains.append(np.float16(node.gain))
    code_bytes = sum(H.payload_bytes(b) for b in blobs)
    total = code_bytes + len(gains) + (struct_bits + 7) // 8
    return {"blobs": blobs, "gains": np.array(gains, dtype=np.float16),
            "nnodes": nnodes, "bytes": total,
            "bpp": 8.0 * total / (o * i)}


def unpack_tree(p: dict, shape: tuple, skeleton: list) -> np.ndarray:
    """Reconstruct from payload. `skeleton` = [(cols, gain_idx, has_signs)]."""
    o, i = shape
    out = np.zeros((o, i), dtype=np.float64)
    bi = 0
    for cols, gi, hs in skeleton:
        g = float(p["gains"][gi])
        if not hs:
            continue
        idx = H.huffman_decode(p["blobs"][bi])
        bi += 1
        S = np.where(idx.reshape(o, len(cols)) > 0, 1.0, -1.0)
        out[:, cols] += g * S
    return out


def skeleton(root: TreeNode) -> list:
    """Structural metadata needed to decode (cols per node, in walk order)."""
    sk, gi, bi = [], [0], [0]
    for node in _walk(root):
        cols = [int(c) for c in node.cols]
        if node.signs is None:
            sk.append((cols, gi[0], False))
        else:
            sk.append((cols, gi[0], True))
            bi[0] += 1
        gi[0] += 1
    return sk
