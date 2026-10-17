"""Compact BM25 skill-retrieval index for Lumina Studio.

Phone-first, stdlib-only (no numpy needed) inverted index over the skill
library (built-ins from ``serve.SKILLS`` plus user skills). Scoring matches
``serve._rag_score`` exactly: the same tokenizer, the same BM25 constants
(k1=1.2, b=0.75) and the same idf shape

    idf = log(1 + (N - df + 0.5) / (df + 0.5))

Postings are compact: one term -> (``array("i")`` doc ids, ``array("i")`` term
frequencies). Doc ids are appended in increasing order, so the arrays stay
sorted and ``add()`` is incremental without any re-sorting. Only one short
copy of each document text (160 chars, the id/desc head) is retained; the
token list itself is discarded after indexing, so a 100k-doc index stays lean
(about 1 KB per doc, most of it unique id/marker terms in the benchmark).

Calibration beats recall here: ``confident`` is True only when the top score
clears ``min_score`` AND the relative gap to the runner-up clears
``min_margin``. Near-duplicate skills therefore report ``reason == "margin"``
instead of a confident (and misleading) pick. The character-trigram Jaccard
layer may only break close lexical ties over documents that already scored
lexically; it never invents hits, and it is annotated (``"fuzzy"``) whenever
it changes a score. Any non-confident result with hits reports "margin";
``min_score`` failures fold into the same reason because the public reason
vocabulary is None | "no_hits" | "margin".

Determinism: no randomness in this module at all (the benchmark and tests use
``random.Random(7)`` for synthetic corpora). Ties break on the lower doc id.
"""

from __future__ import annotations

import array
import math
import re
import sys
import time

BM25 = {"k1": 1.2, "b": 0.75}

# Identical to serve._toks (same character class, same lowering).
_WORDS_RE = re.compile(r"[a-z0-9\u00c0-\u024f]+")

# Per-field caps for the indexed document text.
DOC_FIELD_CHARS = {"id": 80, "desc": 240, "category": 60, "prompt": 120}

# Display copy kept per document (returned as "desc" in search hits).
DISPLAY_CHARS = 160

# Fuzzy layer: character trigram Jaccard, applied only when the lexical margin
# is below min_margin and only over the top lexical candidates.
FUZZY_MIN_JACCARD = 0.10
FUZZY_WEIGHT = 1.5


def tok(s: str) -> list[str]:
    """Tokenize exactly like serve._toks: lowercase, [a-z0-9\\u00c0-\\u024f]+."""
    return _WORDS_RE.findall(s.lower())


def _field_text(skill: dict, key: str, cap: int, alt: str | None = None) -> str:
    """One doc field, capped; missing/odd types degrade to ""."""
    v = skill.get(key)
    if v is None and alt is not None:
        v = skill.get(alt)
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        v = " ".join(str(x) for x in v)
    return str(v)[:cap]


def _doc_text(skill: dict, caps: dict) -> str:
    """id + desc + category/tags + first N chars of the prompt body."""
    sid = _field_text(skill, "id", caps.get("id", 80))
    desc = _field_text(skill, "desc", caps.get("desc", 240), "description")
    cat_cap = caps.get("category", 60)
    cat = _field_text(skill, "category", cat_cap)
    tags = _field_text(skill, "tags", cat_cap)
    prompt = _field_text(skill, "prompt", caps.get("prompt", 120), "body")
    parts = [p for p in (sid, desc, (cat + " " + tags).strip(), prompt) if p]
    return " ".join(parts)


def _trigrams(s: str) -> set[str]:
    """Character trigrams with space padding (stable for very short strings)."""
    pad = " " + s + " "
    if len(pad) < 3:
        return set()
    return {pad[i:i + 3] for i in range(len(pad) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / (len(a) + len(b) - inter)


class SkillIndex:
    """Inverted BM25 index over skill dicts (id/desc/category/tags/prompt)."""

    def __init__(self, skills: list[dict] | None = None,
                 doc_field_chars: dict | None = None):
        self.field_chars = dict(DOC_FIELD_CHARS)
        if isinstance(doc_field_chars, dict):
            for key, val in doc_field_chars.items():
                try:
                    self.field_chars[str(key)] = max(0, int(val))
                except (TypeError, ValueError):
                    continue
        self._reset()
        if skills is not None:
            self.build(skills)

    # -- construction -----------------------------------------------------

    def _reset(self) -> None:
        self._postings: dict = {}       # term -> (array("i") ids, array("i") tfs)
        self._dl = array.array("i")     # doc length in tokens
        self._ids: list[str] = []
        self._texts: list[str] = []     # capped text: display copy + fuzzy text
        self._n = 0
        self._tokens = 0                # total tokens (avgdl without re-summing)

    def build(self, skills: list[dict]) -> dict:
        """Rebuild from scratch over an iterable of skill dicts."""
        t0 = time.perf_counter()
        if skills is None or isinstance(skills, (str, bytes, dict)):
            return {"error": "build() needs an iterable of skill dicts"}
        self._reset()
        try:
            for skill in skills:
                self.add(skill)
        except TypeError:
            self._reset()
            return {"error": "build() needs an iterable of skill dicts"}
        return {"docs": self._n, "terms": len(self._postings),
                "build_s": time.perf_counter() - t0}

    def add(self, skill: dict) -> None:
        """Index one skill incrementally; postings stay sorted by doc id."""
        if not isinstance(skill, dict):
            return
        text = _doc_text(skill, self.field_chars)
        toks = tok(text)
        idx = self._n
        sid = str(skill.get("id", ""))[:self.field_chars.get("id", 80)]
        self._ids.append(sid)
        self._texts.append(text if len(text) <= DISPLAY_CHARS
                           else text[:DISPLAY_CHARS])
        self._dl.append(len(toks))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        postings = self._postings
        for term, freq in tf.items():
            ent = postings.get(term)
            if ent is None:
                postings[term] = (array.array("i", [idx]),
                                  array.array("i", [freq]))
            else:
                # idx increases monotonically -> arrays stay sorted by doc id
                ent[0].append(idx)
                ent[1].append(freq)
        self._tokens += len(toks)
        self._n += 1

    # -- retrieval --------------------------------------------------------

    def search(self, query: str, k: int = 5, *, min_score: float = 2.0,
               min_margin: float = 0.15, fuzzy: bool = True,
               max_fuzzy_candidates: int = 200) -> dict:
        """Rank skills for a free-text query; calibrated confidence.

        Returns {"query", "skills": [{"id","score","bm25","desc"[,"fuzzy"]}],
        "confident", "reason", "latency_ms", "top1_minus_top2"}.
        Scores are raw floats (serve rounds only for display); the BM25 field
        is exactly serve._rag_score's score for the same doc text.
        """
        t0 = time.perf_counter()
        if not isinstance(query, str):
            return {"error": "query must be a string"}
        try:
            k = max(1, int(k))
        except (TypeError, ValueError):
            return {"error": "k must be an int"}
        try:
            min_score = float(min_score)
            min_margin = float(min_margin)
        except (TypeError, ValueError):
            return {"error": "min_score/min_margin must be numbers"}
        qt = tok(query)
        acc: dict[int, float] = {}
        n = self._n
        if qt and n:
            avgdl = self._tokens / n
            k1 = float(BM25["k1"])
            b = float(BM25["b"])
            postings = self._postings
            dl = self._dl
            for term in qt:                      # duplicates count, as in serve
                ent = postings.get(term)
                if ent is None:
                    continue
                ids, fr = ent
                df = len(ids)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                for i in range(df):
                    d = ids[i]
                    f = float(fr[i])
                    denom = f + k1 * (1.0 - b + b * (dl[d] / avgdl))
                    acc[d] = acc.get(d, 0.0) + idf * (f * (k1 + 1.0)) / denom
        ranked = sorted(((d, s) for d, s in acc.items() if s > 0.0),
                        key=lambda kv: (-kv[1], kv[0]))
        fmap: dict[int, float] = {}
        if ranked:
            top1 = ranked[0][1]
            top2 = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = (top1 - top2) / max(top1, 1e-9)
            if fuzzy and margin < min_margin:
                try:
                    window = max(0, int(max_fuzzy_candidates))
                except (TypeError, ValueError):
                    window = 0
                if window:
                    qg = _trigrams(" ".join(qt))
                    for d, _s in ranked[:window]:
                        if not qg:
                            break
                        j = _jaccard(qg, _trigrams(self._texts[d]))
                        if j >= FUZZY_MIN_JACCARD:
                            fmap[d] = j
                    if fmap:
                        fused = []
                        for d, s in ranked:
                            j = fmap.get(d, 0.0)
                            fused.append((d, s * (1.0 + FUZZY_WEIGHT * j) if j
                                          else s))
                        fused.sort(key=lambda kv: (-kv[1], kv[0]))
                        ranked = fused
        hits = []
        for d, s in ranked[:k]:
            bm25 = acc[d]
            h = {"id": self._ids[d], "score": s, "bm25": bm25,
                 "desc": self._texts[d]}
            if d in fmap and s > bm25:
                h["fuzzy"] = round(fmap[d], 4)
            hits.append(h)
        if ranked:
            top1 = ranked[0][1]
            top2 = ranked[1][1] if len(ranked) > 1 else 0.0
            confident = (top1 >= min_score
                         and ((top1 - top2) / max(top1, 1e-9)) >= min_margin)
            reason = None if confident else "margin"
        else:
            top1 = top2 = 0.0
            confident = False
            reason = "no_hits"
        return {"query": query, "skills": hits, "confident": confident,
                "reason": reason,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "top1_minus_top2": top1 - top2}

    # -- accounting -------------------------------------------------------

    def approx_bytes(self) -> int:
        """Honest estimate of live structure size (dicts + array buffers)."""
        total = sys.getsizeof(self._postings)
        for term, ent in self._postings.items():
            total += sys.getsizeof(term) + sys.getsizeof(ent)
            total += sys.getsizeof(ent[0]) + sys.getsizeof(ent[1])
        for lst in (self._ids, self._texts):
            total += sys.getsizeof(lst)
            for item in lst:
                total += sys.getsizeof(item)
        total += sys.getsizeof(self._dl)
        total += sys.getsizeof(self) + sys.getsizeof(self.field_chars)
        return int(total)
