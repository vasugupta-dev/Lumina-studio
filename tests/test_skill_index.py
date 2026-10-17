"""Compact skill retrieval index (app/server/skill_index.py).

Covers, in order:
  * the hand-labeled 102-query fixture over the 30 builtin skills
  * the "make this text shorter" paraphrase (closest rewrite builtin)
  * confidence calibration: near-duplicates, gibberish, 20 ambiguous pairs
  * tokenizer parity against serve._toks and ranking parity against a
    brute-force re-implementation of serve._rag_score (3-doc corpus plus the
    real 30-skill corpus)
  * incremental add() parity, API failure shapes, fuzzy layer behavior
  * serve.py wiring: search API, skill_fetch agent tool, chat auto-fetch
    (opt-in), agent-loop auto-fetch, index invalidation on skill edits
  * 10k-doc synthetic scale: build time, p50/p95 latency, tracemalloc peak

Every number printed is measured here; gates (top-1 >= 0.85, top-5 >= 0.95,
p95 < 250 ms, peak < 200 MB) are asserted, never fudged. The fixture is read
from tests/fixtures/skill_queries.json (skills embedded there), so this suite
does not need serve.py for the fixture; serve is imported only for the
tokenizer/ranking parity checks.

Run:  python3 tests/test_skill_index.py
Exit codes: 0 = pass, 1 = fail.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import tracemalloc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

from skill_index import BM25, SkillIndex, tok  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "skill_queries.json")
RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:200]) if not cond and extra else ""))


def num(x, nd=4):
    return ("%." + str(nd) + "f") % x


def load_fixture() -> dict:
    doc = json.load(open(FIXTURE))
    if isinstance(doc, list):  # tolerate a bare list of entries
        doc = {"skills": [], "queries": doc}
    return doc


def pct(sorted_ms, q):
    """Nearest-rank percentile over an already sorted list (ms)."""
    if not sorted_ms:
        return 0.0
    i = min(len(sorted_ms) - 1, max(0, int(math.ceil(q * len(sorted_ms))) - 1))
    return sorted_ms[i]


def synth_stream(n, m, seed=7):
    """Seeded synthetic corpus + held-out marker queries.

    Doc i = 8..16 words drawn from a 300-word seeded pool plus the unique
    marker token "m%07d" % i. Query j targets doc (j * 97 + 13) % n and
    contains that doc's marker plus two of its pool words, so the target is
    known exactly and cannot be confused with another doc.
    """
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    pool = ["".join(rng.choice(alpha) for _ in range(7)) for _ in range(300)]
    targets: dict[int, list[int]] = {}
    for j in range(m):
        targets.setdefault((j * 97 + 13) % n, []).append(j)
    queries = [None] * m

    def docs():
        for i in range(n):
            words = [pool[rng.randrange(300)] for _ in range(rng.randint(8, 16))]
            marker = "m%07d" % i
            for j in targets.get(i, ()):
                queries[j] = (marker + " " + " ".join(words[:2]), i)
            yield {"id": "sk%06d" % i, "desc": " ".join(words), "prompt": marker}

    return docs(), queries


# ---------------------------------------------------------------------------
# fixture: hand-labeled queries over the 30 builtins
# ---------------------------------------------------------------------------

def fixture_case() -> SkillIndex:
    doc = load_fixture()
    skills, queries = doc["skills"], doc["queries"]
    ids = {s["id"] for s in skills}
    check("fixture: 30 builtin skills embedded", len(skills) == 30, len(skills))
    check("fixture: ~100 labeled entries", 90 <= len(queries) <= 130, len(queries))
    check("fixture: every label names an embedded skill",
          all(q["want"] in ids for q in queries),
          [q for q in queries if q["want"] not in ids][:3])
    check("fixture: all 30 skills are covered",
          ids == {q["want"] for q in queries},
          sorted(ids - {q["want"] for q in queries}))
    si = SkillIndex(skills)
    top1 = top5 = 0
    hard1 = hard5 = hard_n = 0
    misses = []
    for e in queries:
        r = si.search(e["q"], k=5)
        got = [h["id"] for h in r["skills"]]
        ok1 = bool(got) and got[0] == e["want"]
        ok5 = e["want"] in got
        top1 += ok1
        top5 += ok5
        if e.get("hard"):
            hard_n += 1
            hard1 += ok1
            hard5 += ok5
        if not ok1:
            misses.append((e["q"], e["want"], got[:3], bool(e.get("hard"))))
    n = len(queries)
    r1, r5 = top1 / n, top5 / n
    print("fixture accuracy: top1 %d/%d = %s (gate 0.85) | top5 %d/%d = %s "
          "(gate 0.95)" % (top1, n, num(r1), top5, n, num(r5)))
    print("fixture hard subset: top1 %d/%d = %s | top5 %d/%d = %s"
          % (hard1, hard_n, num(hard1 / max(1, hard_n)), hard5, hard_n,
             num(hard5 / max(1, hard_n))))
    for q, want, got, hard in misses:
        print("  top1 miss%s: %-58s want=%s got=%s"
              % (" (hard)" if hard else "", q[:58], want, got))
    check("fixture: top-1 rate >= 0.85", r1 >= 0.85, num(r1))
    check("fixture: top-5 rate >= 0.95", r5 >= 0.95, num(r5))
    check("fixture: all top-1 misses are flagged hard",
          all(hard for _q, _w, _g, hard in misses),
          [m for m in misses if not m[3]][:2])
    return si


def shorter_case(si: SkillIndex) -> None:
    # Naming: the builtin library has no skill literally named "shorten". The
    # closest are "rewrite-tight" ("Cut wordiness without losing meaning.")
    # and "summarize" ("Condense text into a tight summary with key points.");
    # "make this text shorter" is asserted against rewrite-tight in the top 3
    # rather than against a name that does not exist.
    r = si.search("make this text shorter", k=3)
    got = [h["id"] for h in r["skills"]]
    rank = got.index("rewrite-tight") + 1 if "rewrite-tight" in got else 0
    print("shorter: top3=%s scores=%s confident=%s"
          % (got, [round(h["score"], 3) for h in r["skills"]], r["confident"]))
    check("shorter: rewrite-tight inside top-3", 1 <= rank <= 3,
          {"got": got, "rank": rank})
    check("shorter: no crash on a 4-word query with hits", bool(got))


def near_duplicate_case() -> None:
    skills = [
        {"id": "tighten-a", "category": "writing",
         "desc": "Make a text shorter while keeping the meaning.",
         "prompt": "Rewrite the passage so it is shorter and clear."},
        {"id": "tighten-b", "category": "writing",
         "desc": "Make a text shorter while keeping the meaning.",
         "prompt": "Rewrite the passage so it is shorter and clear."},
    ]
    si = SkillIndex(skills)
    r = si.search("make this text shorter", k=5)
    print("near-duplicate: confident=%s reason=%s top1_minus_top2=%s hits=%d"
          % (r["confident"], r["reason"], num(r["top1_minus_top2"], 6),
             len(r["skills"])))
    check("near-duplicate: exactly two hits", len(r["skills"]) == 2,
          [h["id"] for h in r["skills"]])
    check("near-duplicate: not confident", r["confident"] is False)
    check("near-duplicate: reason margin", r["reason"] == "margin", r["reason"])
    check("near-duplicate: tie really is a tie", r["top1_minus_top2"] == 0.0,
          r["top1_minus_top2"])


def gibberish_case(si: SkillIndex) -> None:
    r = si.search("zxqv plorb", k=5)
    check("gibberish: not confident", r["confident"] is False, r)
    check("gibberish: reason no_hits", r["reason"] == "no_hits", r["reason"])
    check("gibberish: no skills returned", r["skills"] == [], r["skills"])


PAIR_DESCS = [
    ("Make a text shorter while keeping its meaning.", "make this text shorter"),
    ("Fix the tone of a passage.", "fix the tone of this passage"),
    ("Compress a file before sending it.", "compress this file before sending"),
    ("Resize photos for the web.", "resize these photos for the web"),
    ("Convert rows into a table.", "convert these rows into a table"),
    ("Review code for bugs.", "review code for bugs"),
    ("Clean up the raw notes from a meeting.",
     "clean up the raw notes from this meeting"),
    ("Summarize the main points.", "summarize the main points"),
    ("Explain the query step by step.", "explain the query step by step"),
    ("Write release notes from commits.", "write release notes from commits"),
    ("Structure the symptoms into a report.", "structure the symptoms into a report"),
    ("Translate the text into another language.",
     "translate the text into another language"),
    ("Extract action items from the text.", "extract action items from the text"),
    ("Draft a short email for the team.", "draft a short email for the team"),
    ("Trace the code line by line.", "trace the code line by line"),
    ("Validate the yaml before shipping.", "validate the yaml before shipping"),
    ("Turn a decision into a record.", "turn a decision into a record"),
    ("Plan the milestones for a feature.", "plan the milestones for a feature"),
    ("Improve the prompt with constraints.", "improve the prompt with constraints"),
    ("Proofread the text and list changes.", "proofread the text and list changes"),
]


def ambiguous_pairs_case() -> None:
    """Calibration gate: 20 near-identical pairs must produce ZERO confident
    picks. Each pair shares one description word-for-word (only the ids
    differ), so the runner-up is genuinely as good as the winner and a
    confident answer would be a false claim of certainty."""
    skills = []
    for i, (desc, _q) in enumerate(PAIR_DESCS, 1):
        for side in ("left", "right"):
            skills.append({"id": "pair-%02d-%s" % (i, side), "category": "test",
                           "desc": desc, "prompt": desc})
    si = SkillIndex(skills)
    confident = 0
    with_hits = 0
    max_margin = 0.0
    for i, (_desc, q) in enumerate(PAIR_DESCS, 1):
        r = si.search(q, k=5)
        got = {h["id"] for h in r["skills"]}
        if got:
            with_hits += 1
        if r["confident"]:
            confident += 1
            print("  CONFIDENT on ambiguous pair %d: %s -> %s" % (i, q, sorted(got)))
        top2 = r["skills"][:2]
        if len(top2) == 2:
            m = (top2[0]["score"] - top2[1]["score"]) / max(top2[0]["score"], 1e-9)
            max_margin = max(max_margin, m)
    print("ambiguous pairs: %d/%d queries had hits, %d confident picks, "
          "max observed margin %s" % (with_hits, len(PAIR_DESCS), confident,
                                      num(max_margin, 4)))
    check("ambiguous pairs: every query hits both pair members",
          with_hits == len(PAIR_DESCS), with_hits)
    check("ambiguous pairs: zero confident picks", confident == 0, confident)
    check("ambiguous pairs: max margin stays below 0.15", max_margin < 0.15,
          num(max_margin, 4))


# ---------------------------------------------------------------------------
# calibration mechanics
# ---------------------------------------------------------------------------

def calibration_case(si: SkillIndex) -> None:
    one = si.search("docstring", k=5)   # df=1 term -> single hit
    check("calibration: single hit is confident", one["confident"] is True
          and one["reason"] is None and len(one["skills"]) == 1,
          {"conf": one["confident"], "n": len(one["skills"])})
    check("calibration: single hit margin is the full score",
          one["top1_minus_top2"] == one["skills"][0]["score"],
          one["top1_minus_top2"])
    hi = si.search("docstring", k=5, min_score=1e9)
    check("calibration: score floor blocks confidence",
          hi["confident"] is False and hi["reason"] == "margin", hi["reason"])
    # margin gate: a corpus with a clear-but-not-overwhelming winner
    gate = SkillIndex([{"id": "strong-doc", "desc": "widget gadget"},
                       {"id": "weak-doc", "desc": "widget gizmo"}])
    loose = gate.search("widget gadget", k=2, min_score=0.0)
    tight = gate.search("widget gadget", k=2, min_score=0.0, min_margin=0.99)
    check("calibration: margin gate blocks a close-ish pair",
          loose["confident"] is True and tight["confident"] is False
          and tight["reason"] == "margin",
          {"loose": loose["confident"], "tight": tight["confident"],
           "reason": tight["reason"]})
    # k truncation must not fake confidence: the tie corpus with k=1
    tie = SkillIndex([{"id": "tie-left", "desc": "Make a text shorter."},
                      {"id": "tie-right", "desc": "Make a text shorter."}])
    r = tie.search("make this text shorter", k=1)
    check("calibration: confidence uses all candidates, not just k",
          len(r["skills"]) == 1 and r["confident"] is False, r["confident"])
    empty = SkillIndex([])
    e = empty.search("anything")
    check("calibration: empty index is no_hits",
          e["confident"] is False and e["reason"] == "no_hits"
          and e["skills"] == [], e)
    check("calibration: every returned score is > 0",
          all(h["score"] > 0 for h in si.search("report text", k=5)["skills"]))


# ---------------------------------------------------------------------------
# fuzzy layer
# ---------------------------------------------------------------------------

def fuzzy_case() -> None:
    # Doc 0 wins the doc-id tie-break; doc 1 has the trigram-closer word
    # ("textile" vs the query token "text"), so only the fuzzy layer flips it.
    skills = [
        {"id": "trim-b", "desc": "shrink image sizes for export"},
        {"id": "trim-a", "desc": "shrink textile sizes for export"},
    ]
    si = SkillIndex(skills)
    on = si.search("shrink text", k=2)
    off = si.search("shrink text", k=2, fuzzy=False)
    capped = si.search("shrink text", k=2, max_fuzzy_candidates=0)
    print("fuzzy: on=%s off=%s capped=%s"
          % ([h["id"] for h in on["skills"]], [h["id"] for h in off["skills"]],
             [h["id"] for h in capped["skills"]]))
    check("fuzzy: lexical tie is real",
          off["skills"][0]["score"] == off["skills"][1]["score"])
    check("fuzzy: trigram layer flips the tie",
          on["skills"][0]["id"] == "trim-a", [h["id"] for h in on["skills"]])
    check("fuzzy: contribution is annotated",
          "fuzzy" in on["skills"][0] and on["skills"][0]["fuzzy"] > 0,
          on["skills"][0])
    check("fuzzy: no annotation when disabled",
          all("fuzzy" not in h for h in off["skills"]))
    check("fuzzy: max_fuzzy_candidates=0 disables the layer",
          capped["skills"][0]["id"] == "trim-b",
          [h["id"] for h in capped["skills"]])
    check("fuzzy: flipping a tie is not confidence",
          on["confident"] is False, on["confident"])


# ---------------------------------------------------------------------------
# parity with serve
# ---------------------------------------------------------------------------

def tokenizer_case() -> None:
    import serve  # convention: importing serve in tests is fine
    strings = [
        "", "UPPER lower MiXeD", "café au Lait, très Crème!", "naïve résumé",
        "hello, world! 42 times", "  spaces\tand\nnewlines  ", "already-lower",
        "123 456 7890", "snake_case kebab-case dot.case",
        "ünïcödé wörds ÅÄÖ àôû", "привет текст drops cyrillic",
        "MiXeD-ÜnïCoDe 9",
    ]
    bad = [(s, tok(s), serve._toks(s)) for s in strings if tok(s) != serve._toks(s)]
    check("tokenizer: identical to serve._toks on %d strings" % len(strings),
          not bad, bad[:2])
    check("tokenizer: latin-1 supplement kept",
          tok("café") == ["café"] and serve._toks("café") == ["café"])
    check("tokenizer: digits kept, punctuation dropped",
          tok("42, 99-bottles!") == ["42", "99", "bottles"])
    check("tokenizer: BM25 constants match serve's denominator",
          BM25 == {"k1": 1.2, "b": 0.75})


def _serve_formula_guard() -> tuple[bool, str]:
    """serve._rag_score is the reference; fail loudly if it drifted."""
    src = open(os.path.join(ROOT, "app", "server", "serve.py")).read()
    body = 'f + 1.2 * (0.25 + 0.75 * len(c["tokens"]) / avgdl)'
    idf = "math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))"
    return (body in src and idf in src), (body, idf)


def _brute_parity(si: SkillIndex, queries: list[str]) -> tuple[float, int, list]:
    """Brute-force re-implementation of serve._rag_score over the texts the
    index retained, compared against SkillIndex. Valid only when every
    retained text is the exact indexed text (checked by callers via the
    display cap). Returns (worst abs diff, doc-scores checked, problems)."""
    import serve
    texts = list(si._texts)
    docs = [{"id": si._ids[i], "tokens": serve._toks(texts[i])}
            for i in range(len(texts))]
    N = len(docs)
    avgdl = sum(len(d["tokens"]) for d in docs) / N
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d["tokens"]):
            df[t] = df.get(t, 0) + 1
    worst = 0.0
    checked = 0
    problems = []
    for q in queries:
        qt = serve._toks(q)
        expected = {}
        for d in docs:
            tf: dict[str, int] = {}
            for t in d["tokens"]:
                tf[t] = tf.get(t, 0) + 1
            sc = 0.0
            for t in qt:  # duplicates counted, exactly like serve
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (N - df.get(t, 0) + 0.5)
                               / (df.get(t, 0) + 0.5))
                denom = f + 1.2 * (0.25 + 0.75 * len(d["tokens"]) / avgdl)
                sc += idf * f * 2.2 / denom
            expected[d["id"]] = sc
        positive = {k: v for k, v in expected.items() if v > 0}
        got = {h["id"]: h for h in si.search(q, k=len(docs), fuzzy=False)["skills"]}
        if set(positive) != set(got):
            problems.append((q, sorted(positive), sorted(got)))
        for k, v in got.items():
            worst = max(worst, abs(v["bm25"] - expected[k]),
                        abs(v["score"] - expected[k]))
            checked += 1
    return worst, checked, problems


def parity_case() -> None:
    ok, needles = _serve_formula_guard()
    check("parity: serve._rag_score formula has not drifted", ok, needles)
    skills = [
        {"id": "alpha-doc", "desc": "alpha bananas citrus"},
        {"id": "beta-doc", "desc": "beta bananas durian"},
        {"id": "gamma-doc", "desc": "gamma citrus durian endive"},
    ]
    si = SkillIndex(skills)
    # The retained per-doc text is capped at DISPLAY_CHARS; these corpus texts
    # are far below the cap, so the retained copy is the exact indexed text and
    # the brute force tokenizes the same bytes the index did.
    check("parity: 3-doc corpus texts sit under the display cap",
          max(len(t) for t in si._texts) < 160, [len(t) for t in si._texts])
    worst, checked, problems = _brute_parity(
        si, ["alpha bananas", "durian", "citrus endive",
             "gamma citrus durian", "zeta missing", "bananas durian"])
    check("parity: 3-doc corpus candidate sets match brute force",
          not problems, problems[:2])
    print("parity: %d doc-scores (3-doc corpus), worst abs diff %s"
          % (checked, worst))
    check("parity: 3-doc corpus scores equal within 1e-6", worst < 1e-6, worst)
    # Same claim on the real builtin corpus. Field caps are tightened so every
    # retained text is complete (no 160-char truncation), which keeps the
    # "brute force sees the indexed bytes" precondition true.
    fx = load_fixture()
    si30 = SkillIndex(fx["skills"], doc_field_chars={"id": 24, "desc": 80,
                                                     "category": 24, "prompt": 0})
    check("parity: 30-skill corpus texts sit under the display cap",
          max(len(t) for t in si30._texts) <= 160,
          max(len(t) for t in si30._texts))
    worst30, checked30, problems30 = _brute_parity(si30, [
        "explain this sql query step by step",
        "write a regex that matches email addresses",
        "make this text shorter",
        "review this api design for naming and versioning",
        "update the changelog from these changes",
        "my yaml file blows up in the parser, make it valid",
    ])
    check("parity: 30-skill corpus candidate sets match brute force",
          not problems30, problems30[:2])
    print("parity: %d doc-scores (30-skill corpus), worst abs diff %s"
          % (checked30, worst30))
    check("parity: 30-skill corpus scores equal within 1e-6", worst30 < 1e-6,
          worst30)
    with_fuzzy = {h["id"]: h for h in si.search("citrus endive", k=3)["skills"]}
    same = all(abs(h["score"] - h["bm25"]) < 1e-12 or "fuzzy" in h
               for h in with_fuzzy.values())
    check("parity: any fuzzy-driven score change is annotated", same, with_fuzzy)


# ---------------------------------------------------------------------------
# incremental build + API shapes
# ---------------------------------------------------------------------------

def api_case(si: SkillIndex) -> None:
    doc = load_fixture()
    half = len(doc["skills"]) // 2
    inc = SkillIndex()
    for s in doc["skills"][:half]:
        inc.add(s)
    big = SkillIndex(doc["skills"][:half])
    q = "write a regex that matches email addresses"
    check("api: incremental add == build on the same prefix",
          [h["id"] for h in inc.search(q)["skills"]]
          == [h["id"] for h in big.search(q)["skills"]])
    check("api: postings stay sorted by doc id",
          all(list(ids) == sorted(set(ids)) for ids, _tf in inc._postings.values()))
    check("api: build reports docs/terms/build_s",
          set(inc.build(doc["skills"])) == {"docs", "terms", "build_s"})
    b0 = inc.approx_bytes()
    inc.add({"id": "extra-skill", "desc": "An extra skill for size checks."})
    check("api: approx_bytes grows with a doc", inc.approx_bytes() > b0,
          (b0, inc.approx_bytes()))
    check("api: approx_bytes is a positive int",
          isinstance(b0, int) and b0 > 0, b0)
    check("api: non-string query returns an error",
          "error" in si.search(123))
    check("api: non-numeric thresholds return an error",
          "error" in si.search("text", min_score="high"))
    check("api: string build input returns an error",
          "error" in SkillIndex().build("not-a-list"))
    check("api: None build returns an error", "error" in SkillIndex().build(None))
    try:
        bare = SkillIndex([{"id": "only-id"}, {}, {"desc": "desc only"},
                           None, 42])
        raised = None
    except Exception as exc:  # pragma: no cover - only on a real bug
        bare, raised = None, exc
    check("api: missing/odd entries are handled gracefully",
          raised is None and bare._n == 3, (raised, getattr(bare, "_n", None)))
    check("api: id-only skill is searchable by its id token",
          bare is not None
          and [h["id"] for h in bare.search("only id", k=1)["skills"]]
          == ["only-id"])
    one = [{"id": "capped", "prompt": "zebraword"}]
    check("api: doc_field_chars really caps the indexed text",
          SkillIndex(one).search("zebraword", k=3)["skills"] != []
          and SkillIndex(one, doc_field_chars={"prompt": 3})
          .search("zebraword", k=3)["skills"] == [])
    check("api: tags list is indexed",
          bool(SkillIndex([{"id": "tag-skill", "tags": ["aardvark", "zebra"]}])
               .search("aardvark", k=1)["skills"]))
    long_doc = SkillIndex([{"id": "long-doc", "desc": "d" * 240,
                            "prompt": "lorem ipsum"}])
    check("api: retained doc text is capped at 160 chars",
          len(long_doc._texts[0]) == 160, len(long_doc._texts[0]))
    check("api: lexical index still covers text beyond the display cap",
          [h["id"] for h in long_doc.search("lorem", k=1)["skills"]]
          == ["long-doc"])


def determinism_case(si: SkillIndex) -> None:
    doc = load_fixture()
    a = SkillIndex(doc["skills"])
    b = SkillIndex(doc["skills"])
    queries = ["make this text shorter", "write a regex that matches email addresses",
               "turn the commit history into published release notes",
               "what is the time complexity of this algorithm",
               "pull the fields out of this text as json"]
    same = True
    for q in queries:
        ra = [(h["id"], h["score"]) for h in a.search(q, k=5)["skills"]]
        rb = [(h["id"], h["score"]) for h in b.search(q, k=5)["skills"]]
        same = same and ra == rb
    check("determinism: two independent builds agree exactly", same)
    check("determinism: repeated search on one index is stable",
          [(h["id"], h["score"]) for h in si.search(queries[0], k=5)["skills"]]
          == [(h["id"], h["score"]) for h in si.search(queries[0], k=5)["skills"]])


# ---------------------------------------------------------------------------
# scale: 10k docs, 200 held-out queries
# ---------------------------------------------------------------------------

def scale_case(n=10000, m=200) -> None:
    docs, queries = synth_stream(n, m, seed=7)
    si = SkillIndex()
    tracemalloc.start()
    t0 = time.perf_counter()
    for d in docs:
        si.add(d)
    build_s = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    ms = []
    top1 = top5 = 0
    for q, target in queries:
        t = time.perf_counter()
        hits = si.search(q, k=5)["skills"]
        ms.append((time.perf_counter() - t) * 1000.0)
        got = [h["id"] for h in hits]
        want = "sk%06d" % target
        top1 += bool(got) and got[0] == want
        top5 += want in got
    ms.sort()
    p50, p95 = pct(ms, 0.50), pct(ms, 0.95)
    apx = si.approx_bytes()
    tracemalloc.start()
    for q, _t in queries[:50]:
        si.search(q, k=5)
    _cur2, peak_q = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_q_mb = peak_q / (1024 * 1024)
    print("scale(%d docs): build %s s, terms %d, avg dl %s, approx %s MB"
          % (n, num(build_s, 4), len(si._postings),
             num(si._tokens / max(1, si._n), 2), num(apx / (1024 * 1024), 2)))
    print("scale(%d queries): p50 %s ms, p95 %s ms, max %s ms (gate p95 < 250 ms)"
          % (m, num(p50, 3), num(p95, 3), num(ms[-1], 3)))
    print("scale: tracemalloc peak build %s MB (gate < 200 MB), "
          "query-pass peak %s MB" % (num(peak_mb, 2), num(peak_q_mb, 2)))
    print("scale: top1 %d/%d = %s, top5 %d/%d = %s"
          % (top1, m, num(top1 / m, 4), top5, m, num(top5 / m, 4)))
    check("scale: all docs indexed", si._n == n, si._n)
    check("scale: marker queries resolve top-1", top1 == m, (top1, m))
    check("scale: marker queries resolve top-5", top5 == m, (top5, m))
    check("scale: p95 latency < 250 ms", p95 < 250.0, num(p95, 3))
    check("scale: tracemalloc peak < 200 MB", peak_mb < 200.0, num(peak_mb, 2))


def serve_wiring_case():
    """The serve.py integration: search API, agent tool, auto-fetch, cache
    invalidation. Uses the tiny fixture model for the chat/agent paths."""
    import serve

    n = len(serve._all_skills())
    r = serve.api_skills_search("write unit tests for a function", 3)
    check("serve: search confident on a clear task",
          r.get("confident") is True and r["skills"][0]["id"] == "unit-tests",
          r.get("skills"))
    check("serve: search reports corpus size", r.get("docs") == n,
          (r.get("docs"), n))
    check("serve: empty query errors", "error" in serve.api_skills_search("", 3))
    check("serve: gibberish search is not confident",
          serve.api_skills_search("zxqv plorb", 3).get("confident") is False)

    a = serve._skill_index()
    b = serve._skill_index()
    check("serve: index singleton reused", a is b)
    serve._skill_index_invalidate()
    check("serve: invalidate forces rebuild", serve._skill_index() is not a)

    # user-skill lifecycle: save -> search finds it -> delete -> gone
    upath = serve.USER_SKILLS_PATH
    snap = None
    had = os.path.exists(upath)
    if had:
        snap = open(upath, "rb").read()
    sid = "zz-skill-index-test"
    try:
        s = serve.api_skills_save({"id": sid,
                                   "desc": "quantum llama wrangling manual",
                                   "prompt": "Perform quantum llama wrangling.\n{input}"})
        check("serve: user skill saved", s.get("saved") == sid, s)
        r2 = serve.api_skills_search("quantum llama wrangling", 3)
        check("serve: user skill findable after save",
              r2.get("confident") is True and r2["skills"][0]["id"] == sid,
              r2.get("skills"))
        d = serve.api_skills_delete({"id": sid})
        check("serve: user skill deleted", d.get("deleted") == sid, d)
        r3 = serve.api_skills_search("quantum llama wrangling", 3)
        check("serve: user skill gone after delete",
              sid not in [h["id"] for h in r3.get("skills", [])], r3.get("skills"))
    finally:
        if had and snap is not None:
            open(upath, "wb").write(snap)
        elif not had and os.path.exists(upath):
            os.remove(upath)
        serve._skill_index_invalidate()

    t = serve._agent_tool_call({"tool": "skill_fetch",
                                "args": {"query": "write unit tests for a function"}})
    check("serve: skill_fetch agent tool returns the skill body",
          t.get("confident") is True and "unit-tests" in str(t.get("out", "")),
          str(t.get("out"))[:80])
    t2 = serve._agent_tool_call({"tool": "skill_fetch",
                                 "args": {"query": "zzzz nonsense plorb"}})
    check("serve: skill_fetch honest when nothing matches",
          t2.get("confident") is False and "out" in t2, t2)
    check("serve: skill_fetch listed as a base tool spec",
          any(s["tool"] == "skill_fetch" for s in serve._agent_base_tool_specs()))
    check("serve: skill_fetch listed in builtin tools",
          any(s["tool"] == "skill_fetch"
              for s in serve.api_builtin_tools()["open_tools"]))
    check("serve: skill_fetch has an SDK description",
          "skill_fetch" in serve._AGENT_TOOL_DESCRIPTIONS)

    # chat + agent auto-fetch (opt-in) with the tiny fixture model
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from _fixtures import build_tiny_chat  # noqa: E402
    build_tiny_chat()
    lr = serve.api_load("/tmp/tiny-chat")
    check("serve: tiny model loaded for auto-fetch checks",
          not lr.get("error"), lr)
    try:
        r4 = serve.api_chat([{"role": "user",
                              "content": "write unit tests for my function"}],
                            8, 0.0, skill_auto=True)
        sa = r4.get("skill_auto") or {}
        check("serve: chat auto-fetch injects a confident skill",
              sa.get("injected") is True and sa.get("id") == "unit-tests", sa)
        r5 = serve.api_chat([{"role": "user",
                              "content": "write unit tests for my function"}],
                            8, 0.0)
        check("serve: chat auto-fetch is opt-in",
              "skill_auto" not in r5 and "error" not in r5, r5.get("error"))
        r6 = serve.api_chat([{"role": "user",
                              "content": "zzzz nonsense plorb glorp"}],
                            8, 0.0, skill_auto=True)
        check("serve: chat auto-fetch never injects on no match",
              (r6.get("skill_auto") or {}).get("injected") is not True
              or not r6.get("skill_auto"), r6.get("skill_auto"))
        ab = serve._agent_builtin({"messages": [{"role": "user",
                                                 "content": "write unit tests for my function"}],
                                   "max_steps": 1, "max_tokens": 4,
                                   "skill_auto": True})
        check("serve: agent loop auto-fetch recorded",
              (ab.get("skill_auto") or {}).get("injected") is True,
              ab.get("skill_auto") or ab.get("error"))
    finally:
        serve.api_unload()


def main() -> int:
    t0 = time.time()
    si = fixture_case()
    shorter_case(si)
    near_duplicate_case()
    gibberish_case(si)
    ambiguous_pairs_case()
    calibration_case(si)
    fuzzy_case()
    tokenizer_case()
    parity_case()
    api_case(si)
    determinism_case(si)
    serve_wiring_case()
    scale_case()
    ok = sum(1 for _, c in RESULTS if c)
    print("\nSKILL-INDEX: %d/%d passed (%.1fs)"
          % (ok, len(RESULTS), time.time() - t0))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
