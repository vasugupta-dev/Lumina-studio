"""Minimal BPE tokenizer from a HF directory (vocab.json + merges.txt).

Real tokenization using the model's own files — no mock vocabulary.
Falls back to byte-level tokens when model files are absent.
"""

from __future__ import annotations

import json
import os


class BPETokenizer:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        self.ok = False
        self._load()

    def _load(self) -> None:
        vp = os.path.join(self.dir, "vocab.json")
        mp = os.path.join(self.dir, "merges.txt")
        if not (os.path.isfile(vp) and os.path.isfile(mp)):
            return
        try:
            self.vocab = json.load(open(vp))
            self.inv = {v: k for k, v in self.vocab.items()}
            lines = open(mp, encoding="utf-8").read().splitlines()[1:]
            for i, ln in enumerate(lines):
                p = ln.split(" ")
                if len(p) == 2:
                    self.bpe_ranks[(p[0], p[1])] = i
            self.ok = True
        except Exception:
            self.ok = False

    def _bpe(self, token: str) -> list[str]:
        word = list(token)
        if len(word) <= 1:
            return word
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 10 ** 9))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        return word

    def encode(self, text: str) -> list[int]:
        if not self.ok:
            return [b % 512 for b in text.encode("utf-8", "ignore")]
        ids: list[int] = []
        for tok in text.split(" "):
            parts = self._bpe("Ġ" + tok) if tok else []
            for p in parts:
                if p in self.vocab:
                    ids.append(self.vocab[p])
                else:
                    ids.extend([b % len(self.vocab)
                                for b in p.encode("utf-8", "ignore")])
        return ids

    def decode(self, ids: list[int]) -> str:
        if not self.ok:
            return "".join(chr(32 + (i % 90)) for i in ids)
        text = "".join(self.inv.get(i, "") for i in ids)
        return text.replace("Ġ", " ").replace("Ċ", "\n")
