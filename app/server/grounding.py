"""Evidence grounding: claim-level attribution against retrieved passages.

Honest scope (do not overclaim): this is a MEASURABLE GATE, not a
zero-hallucination guarantee. It combines

  * pre-generation: an evidence-only prompt (answer strictly from numbered
    passages, refuse when the evidence lacks the answer), and a hard refusal
    WITHOUT a model call when retrieval returns nothing;
  * post-generation: the answer is split into claims; every claim is checked
    against the passages for (a) lexical coverage of its content words,
    (b) BM25 support (same formula/constants as serve._rag_score:
    k1=1.2, b=0.75, numerator k1+1), (c) every number it states must exist in
    the passages, (d) identifier-like tokens (e.g. "Zorblax-9") must exist
    too. Failing (c)/(d) means the model invented a specific — the classic
    hallucination — and the claim is reported unsupported.

Policies: "flag" (default) annotates only and never edits the answer text;
"withhold" removes exactly the flagged sentences; if every content-bearing
claim is unsupported the answer is replaced by REFUSAL_TEXT.

Numbers on the fixed fixture are printed by tests/test_grounding.py
(detection rate and false-refusal rate) — the gate lives there, not here.
"""
from __future__ import annotations

import math
import re

REFUSAL_TEXT = "I could not find evidence for this in the provided documents."

_WORDS_RE = re.compile(r"[a-z0-9\u00c0-\u024f]+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# identifier-like: contains BOTH letters and digits, or a Capitalized token
# that is not sentence-initial — the shapes fabricated names usually take.
_IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*\b|\b[A-Z][a-z]{2,}\b")

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "it", "its", "this", "that",
    "with", "as", "at", "by", "from", "has", "have", "had", "not", "but",
    "they", "their", "them", "he", "she", "his", "her", "we", "our", "you",
    "your", "i", "also", "than", "then", "there", "here", "which", "who",
    "what", "when", "where", "how", "all", "any", "each", "into", "over",
    "under", "about", "can", "could", "may", "might", "will", "would",
    "should", "do", "does", "did", "so", "no", "yes", "if", "because",
}

# Capitalized words that are not entities even when absent from the evidence.
_STARTERS = {
    "there", "these", "those", "however", "both", "while", "many", "most",
    "some", "after", "before", "during", "over", "under", "they", "their",
}


def tok(s: str) -> list[str]:
    """Parity copy of serve._toks (same regex, same lowercasing)."""
    return _WORDS_RE.findall(str(s).lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tok(text) if t not in _STOP and len(t) > 1]


_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
}


def extract_numbers(text: str) -> list[str]:
    """Canonical decimal strings: commas removed, trailing .0 stripped.
    '1,000' -> '1000'; '42.0' -> '42'; '3.14' -> '3.14'. Number WORDS
    (one..twenty, tens, hundred/thousand/million/billion) are normalized to
    digits too — otherwise a fabricated "six lanes" would slip past an
    evidence sentence that says "four lanes"."""
    out = []
    for m in _NUM_RE.findall(str(text)):
        n = m.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        if n:
            out.append(n)
    for w in tok(text):
        if w in _NUM_WORDS:
            out.append(_NUM_WORDS[w])
    return out


def bm25_scores(query: str, docs_tokens: list[list[str]]) -> list[float]:
    """Exact copy of serve._rag_score's scoring loop (same constants), over an
    in-memory document set instead of the global RAG store."""
    if not docs_tokens:
        return []
    N = len(docs_tokens)
    qt = tok(query)
    if not qt:
        return [0.0] * N
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(t) for t in docs_tokens) / N
    scores = []
    for toks in docs_tokens:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        sc = 0.0
        for t in qt:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            sc += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(toks) / avgdl))
        scores.append(sc)
    return scores


def _passages_list(passages) -> list[dict]:
    out = []
    for p in passages or []:
        if isinstance(p, dict):
            out.append({"id": p.get("id", len(out)), "source": str(p.get("source", "")),
                        "text": str(p.get("text", ""))})
        else:
            out.append({"id": len(out), "source": "", "text": str(p)})
    return out


def split_claims(text: str) -> list[dict]:
    """Sentence-ish claims with stable ids. Splits on . ! ? and newlines,
    keeps internal abbreviations intact enough for our purposes (no NLP)."""
    text = str(text or "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    claims = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        kind = "numeric" if _NUM_RE.search(s) else "assertion"
        claims.append({"id": len(claims), "text": s, "kind": kind})
    return claims


def support_check(claim: str, passages, *, min_score: float = 0.0) -> dict:
    """Check ONE claim against the passages. Returns a diagnosis dict.

    Gate: content-word coverage >= 0.60, at least one lexical overlap with
    the evidence (bm25 score > min_score, default 0.0), every stated number
    present in the evidence, and no unknown capitalized/identifier-like
    tokens. Rationale: BM25's absolute scale depends on corpus size, so an
    absolute floor cannot be used across N=1..many passage sets — coverage
    plus the number/entity checks carry the decision, BM25 is reported for
    diagnostics. Semantic flips (e.g. "north-west" vs "south-east") are NOT
    caught lexically; the fixture labels two such cases and the suite reports
    them as measured misses (the model-verifier pass in boost.py is the
    layered answer to that class)."""
    ps = _passages_list(passages)
    result = {"supported": False, "score": 0.0, "coverage": 0.0,
              "best_id": None, "best_source": None,
              "number_conflict": None, "entity_missing": None,
              "missing_terms": [], "reasons": []}
    if not ps:
        result["reasons"].append("no_passages")
        return result
    ct = content_tokens(claim)
    if not ct:
        # numbers only, e.g. "42." — judge purely on the number check below
        ct = tok(claim)
    corpus_terms = set()
    for p in ps:
        corpus_terms.update(tok(p["text"]))
    missing = [t for t in ct if t not in corpus_terms]
    cover = 1.0 - (len(missing) / len(ct)) if ct else 0.0
    result["coverage"] = round(cover, 4)
    result["missing_terms"] = missing[:12]

    docs = [tok(p["text"]) for p in ps]
    scores = bm25_scores(claim, docs)
    if scores:
        bi = max(range(len(scores)), key=lambda i: scores[i])
        result["score"] = round(scores[bi], 4)
        result["best_id"] = ps[bi]["id"]
        result["best_source"] = ps[bi]["source"]

    corpus_numbers = set()
    for p in ps:
        corpus_numbers.update(extract_numbers(p["text"]))
    nums = extract_numbers(claim)
    nums_missing = [n for n in nums if n not in corpus_numbers]
    if nums:
        result["number_conflict"] = bool(nums_missing)
        if nums_missing:
            result["reasons"].append("numbers_not_in_evidence:" + ",".join(nums_missing))

    idents = set(_IDENT_RE.findall(str(claim)))
    idents = {i for i in idents if not _NUM_RE.fullmatch(i)}
    corpus_idents = set()
    for p in ps:
        corpus_idents.update(_IDENT_RE.findall(str(p["text"])))
    ident_missing = []
    for i in sorted(idents):
        if i in corpus_idents:
            continue
        # Capitalized sentence starters and words that appear lowercase in the
        # evidence are not entities; only genuinely unknown names count.
        if i.lower() in corpus_terms or i.lower() in _STOP or i.lower() in _STARTERS:
            continue
        ident_missing.append(i)
    if idents:
        result["entity_missing"] = bool(ident_missing)
        if ident_missing:
            result["reasons"].append("entities_not_in_evidence:" + ",".join(ident_missing[:5]))

    ok = (cover >= 0.60) and (result["score"] > min_score) \
        and not nums_missing and not ident_missing
    if not ok:
        if cover < 0.60:
            result["reasons"].append("low_term_coverage")
        if result["score"] <= min_score:
            result["reasons"].append("no_lexical_overlap")
    result["supported"] = bool(ok)
    return result


def ground_answer(answer: str, passages, *, policy: str = "flag",
                  min_score: float = 0.0) -> dict:
    """Split -> check -> gate. Never edits the text under "flag" (the default);
    "withhold" removes exactly the flagged sentences."""
    claims = split_claims(answer)
    details = []
    unsupported = 0
    for c in claims:
        d = support_check(c["text"], passages, min_score=min_score)
        details.append({"id": c["id"], "text": c["text"], "kind": c["kind"],
                        **d})
        if not d["supported"]:
            unsupported += 1
    total = len(claims)
    coverage = (1.0 - unsupported / total) if total else 1.0
    if unsupported == 0:
        gate = "pass"
    elif coverage < 0.5:
        gate = "refuse"
    else:
        gate = "flag"
    out_text = str(answer or "")
    original = None
    if policy == "withhold" and unsupported:
        keep = [d["text"] for d in details if d["supported"]]
        if keep:
            out_text = " ".join(keep)
            original = str(answer or "")
        else:
            out_text = REFUSAL_TEXT
            original = str(answer or "")
    return {"claims": details, "total": total, "unsupported": unsupported,
            "coverage": round(coverage, 4), "gate": gate,
            "policy": policy, "answer": out_text, "answer_original": original,
            "min_score": min_score}


def gate_decision(passages) -> dict:
    """Pre-generation gate: refuse WITHOUT a model call when there is no
    evidence to condition on."""
    n = len(_passages_list(passages))
    if n == 0:
        return {"refuse": True, "reason": "no evidence retrieved"}
    return {"refuse": False, "reason": None, "passages": n}


def evidence_block(passages, max_chars: int = 2400) -> str:
    ps = _passages_list(passages)
    lines = []
    for i, p in enumerate(ps, 1):
        t = " ".join(p["text"].split())
        if len(t) > 400:
            t = t[:400].rstrip() + "..."
        src = p["source"] or ("passage %d" % i)
        lines.append("[%d] %s\n%s" % (i, src, t))
    block = "\n\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars].rstrip() + "\n[truncated]"
    return block


def evidence_messages(messages: list[dict], passages, *,
                      max_chars: int = 2400) -> list[dict]:
    """Inject the evidence-only instructions + numbered passages as the system
    message (merged into an existing system message when present)."""
    block = evidence_block(passages, max_chars=max_chars)
    instr = ("Answer ONLY from the numbered evidence below. Cite sources as [n]. "
             "If the evidence does not contain the answer, reply exactly: "
             + REFUSAL_TEXT + "\n\nEVIDENCE:\n" + block)
    msgs = [dict(m) if isinstance(m, dict) else {"role": "user", "content": str(m)}
            for m in (messages or [])]
    if msgs and str(msgs[0].get("role", "")) == "system":
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = str(msgs[0].get("content", "")) + "\n" + instr
    else:
        msgs.insert(0, {"role": "system", "content": instr})
    return msgs
