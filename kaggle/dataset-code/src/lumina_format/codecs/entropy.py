"""Entropy coding for quantization indices — canonical Huffman per tensor.

Why: at very low bpw the code distribution is strongly peaked (most weights
fall on few grid points). Fixed-width storage wastes ~15-30% vs the entropy
limit. Canonical Huffman closes most of that gap with a tiny codebook, and
every byte is counted exactly.
"""

from __future__ import annotations

import heapq

import numpy as np


def _build_lengths(freq: np.ndarray) -> np.ndarray:
    """Huffman code lengths via heap merge (n <= 256 symbols)."""
    n = int(freq.size)
    nz = [(int(freq[i]), i) for i in range(n) if freq[i] > 0]
    if len(nz) <= 1:
        lens = np.ones(n, dtype=np.int64)
        lens[:] = 1
        return lens
    heap = [(f, [i]) for f, i in nz]
    heapq.heapify(heap)
    depth = {i: 0 for _, i in nz}
    while len(heap) > 1:
        f1, s1 = heapq.heappop(heap)
        f2, s2 = heapq.heappop(heap)
        for i in s1 + s2:
            depth[i] += 1
        heapq.heappush(heap, (f1 + f2, s1 + s2))
    return np.array([depth.get(i, 0) for i in range(n)], dtype=np.int64)


def huffman_encode(codes: np.ndarray, n_symbols: int) -> dict:
    """codes: int array in [0, n_symbols). Returns packed payload dict."""
    flat = np.asarray(codes).reshape(-1)
    freq = np.bincount(flat, minlength=n_symbols).astype(np.int64)
    lens = _build_lengths(freq)
    # canonical codes: sort by (length, symbol)
    order = sorted(range(n_symbols), key=lambda s: (lens[s], s))
    table: dict[int, tuple[int, int]] = {}
    code, last_len = 0, lens[order[0]]
    for s in order:
        ln = int(lens[s])
        if ln == 0:
            continue
        code <<= ln - last_len
        table[s] = (code, ln)
        code += 1
        last_len = ln
    out = bytearray()
    acc, nb = 0, 0
    for v in flat:
        c, ln = table[int(v)]
        acc = (acc << ln) | c
        nb += ln
        while nb >= 8:
            nb -= 8
            out.append((acc >> nb) & 0xFF)
    if nb:
        out.append((acc << (8 - nb)) & 0xFF)
    return {"bytes": bytes(out), "lengths": lens.astype(np.uint8),
            "n": int(flat.size), "n_symbols": n_symbols, "bits": sum(
                int(lens[int(v)]) for v in flat)}


def huffman_decode(payload: dict) -> np.ndarray:
    """Exact inverse of huffman_encode. Roundtrip is bit-exact by test."""
    lens = np.asarray(payload["lengths"], dtype=np.int64)
    n, ns = int(payload["n"]), int(payload["n_symbols"])
    data = payload["bytes"]
    # rebuild canonical table: (code, length) per symbol
    order = sorted(range(ns), key=lambda s: (lens[s], s))
    table: dict[tuple[int, int], int] = {}
    code, last_len = 0, int(lens[order[0]])
    for s in order:
        ln = int(lens[s])
        if ln == 0:
            continue
        code <<= ln - last_len
        table[(code, ln)] = s
        code += 1
        last_len = ln
    maxlen = int(lens.max()) if n else 0
    bits = "".join(f"{b:08b}" for b in data)
    out = np.empty(n, dtype=np.int64)
    pos, i = 0, 0
    cur, ln = 0, 0
    while i < n and pos < len(bits):
        cur = (cur << 1) | (bits[pos] == "1")
        ln += 1
        pos += 1
        if (cur, ln) in table:
            out[i] = table[(cur, ln)]
            i += 1
            cur, ln = 0, 0
        elif ln > maxlen + 1:
            raise ValueError("huffman: corrupt stream")
    if i != n:
        raise ValueError("huffman: short stream")
    return out


def payload_bytes(payload: dict) -> int:
    """Exact stored bytes: bitstream + codebook (lengths) + 8B header."""
    return len(payload["bytes"]) + int(payload["n_symbols"]) + 8
