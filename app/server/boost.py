"""Capability boost for small local models (app/server/boost.py).

Lumina runs small on-phone models, so a single greedy sample is often wrong for
reasons that are cheap to fix without touching the weights:

  * ``self_consistency`` -- sample n times, majority-vote the normalized answer
  * ``decompose``        -- let the model split a question into sub-questions
  * ``tool_checks``      -- recompute every "a OP b = c" claim in the answer
  * ``verify``           -- one strict flag pass over the answer (+ evidence)

This module is stdlib-only, duck-typed and side-effect free. The caller passes a
``chat_fn(messages, max_tokens=..., temperature=..., **kw) -> {"text": str}`` and
gets plain dicts back. It never imports ``serve``/``agent_sdk``, never loads
weights, never sleeps and holds no global state: same ``chat_fn`` -> same output.

Honesty labels
--------------
* ``boost_answer()["real_model_lift"]`` is deliberately a constant string. Lift
  only means something when measured against real weights on a held-out set; a
  scripted/deterministic harness cannot measure it, and this module refuses to
  imply that it did.
* ``tool_checks`` audits binary ``a OP b = c`` claims with OP in ``* + -`` and
  integer/float literals only. Chained expressions (``1+2+3=6``), variables and
  ``/`` are skipped, never guessed: ``checked`` counts only what was recomputed.
* ``verify`` reports ``"unchecked"`` whenever the model will not emit parseable
  flags -- never ``"consistent"`` by default.
* ``self_consistency`` with ``n == 1`` is inert: one sample, agreement 1.0, and
  ``aggregate`` becomes ``"single"`` so no majority vote is ever claimed.

``trace[i]["seconds"]`` is the *duration* of stage ``i`` (repo convention, see
``serve.py``); durations are >= 0, so their prefix sums are monotonic and can be
read as cumulative elapsed time.
"""
from __future__ import annotations

import json
import re
import time

__all__ = [
    "answer_key",
    "self_consistency",
    "decompose",
    "verify",
    "tool_checks",
    "boost_answer",
    "REAL_MODEL_LIFT",
]

REAL_MODEL_LIFT = "not measured (scripted/deterministic harness - requires real weights)"

# ---------------------------------------------------------------------------
# chat_fn plumbing (duck-typed, never raises on a malformed client)
# ---------------------------------------------------------------------------


def _call_chat(chat_fn, messages, *, max_tokens, temperature, seed=None):
    """Normalize one model call to ``{"text": str}`` or ``{"error": str}``."""
    if not callable(chat_fn):
        return {"error": "chat_fn is not callable (%s)" % type(chat_fn).__name__}
    kw = {"max_tokens": max_tokens, "temperature": temperature}
    if seed is not None:
        kw["seed"] = seed
    try:
        raw = chat_fn(messages, **kw)
    except TypeError as exc:
        if "seed" not in kw:
            return {"error": "chat_fn raised TypeError: %s" % exc}
        kw.pop("seed")  # client without a seed kwarg (still a legal contract)
        try:
            raw = chat_fn(messages, **kw)
        except Exception as exc2:  # noqa: BLE001 - surfaced, not swallowed
            return {"error": "chat_fn raised %s: %s" % (type(exc2).__name__, exc2)}
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        return {"error": "chat_fn raised %s: %s" % (type(exc).__name__, exc)}
    if not isinstance(raw, dict):
        return {"error": "chat_fn returned %s, expected dict" % type(raw).__name__}
    if raw.get("error"):
        return {"error": str(raw.get("error"))}
    txt = raw.get("text")
    if txt is None:
        return {"error": "chat_fn returned no text"}
    return {"text": str(txt)}


def _last_user_text(messages) -> str:
    if not isinstance(messages, (list, tuple)):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _clip(text, limit):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ---------------------------------------------------------------------------
# answer normalization / majority vote
# ---------------------------------------------------------------------------

# numbers first (commas inside, optional fraction), then punctuation is dropped.
_NUM_TOKEN = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?|\.\d+)(?![\w.])")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _canon_number(token: str) -> str:
    body = token.replace(",", "")
    if "." in body:
        body = body.rstrip("0").rstrip(".")  # 42.0 -> 42 ; 0.70 -> 0.7
        if body in ("", "-"):
            body = "0"
    return body


def answer_key(text) -> str:
    """Normalization used for majority voting (empty-safe, never raises).

    lowercased; numbers canonicalized (``"1,000" == "1000"``, ``"42.0" == "42"``);
    punctuation dropped; whitespace collapsed.
    """
    if text is None:
        return ""
    s = str(text).lower()
    # numbers are canonicalized first and shielded from the punctuation pass,
    # otherwise "3.14" would lose its decimal point.
    out, pos = [], 0
    for m in _NUM_TOKEN.finditer(s):
        out.append(_PUNCT.sub(" ", s[pos:m.start()]))
        out.append(_canon_number(m.group(1)))
        pos = m.end()
    out.append(_PUNCT.sub(" ", s[pos:]))
    return " ".join("".join(out).split())


def _vote(samples, n, aggregate):
    """Return ``(final_text, agreement, final_key)`` or ``(None, 0.0, None)``."""
    ok = [s for s in samples if not s.get("error")]
    if not ok:
        return None, 0.0, None
    counts, first, order = {}, {}, {}
    for i, s in enumerate(ok):
        k = s["key"]
        counts[k] = counts.get(k, 0) + 1
        first.setdefault(k, s["text"])
        order.setdefault(k, i)
    if aggregate == "first":
        k = ok[0]["key"]
    else:  # majority: higher count, then longer normalized text, then first-seen
        k = min(counts, key=lambda c: (-counts[c], -len(c), order[c]))
    return first[k], counts[k] / float(n), k


def self_consistency(chat_fn, messages: list[dict], *, n: int = 3,
                     temperature: float = 0.7, max_tokens: int = 192,
                     aggregate: str = "majority", seed: int | None = None) -> dict:
    """Sample ``n`` answers and vote.

    Returns ``{"samples", "final", "agreement", "n", "aggregate"}`` where each
    sample is ``{"text", "key"}`` (plus ``"error"`` when that sample failed).
    ``final`` is the exact text of the first winning sample. ``agreement`` is
    ``winning_count / n``. With ``seed`` given, sample ``i`` is called with
    ``seed + i`` so repeats differ despite the fixed seed; a client that does not
    accept a seed kwarg is retried without it. If every sample failed, the
    result carries a top-level ``"error"`` (samples included).
    """
    agg = aggregate
    try:
        n_i = int(n)
        if n_i == 1:
            agg = "single"
    except (TypeError, ValueError):
        return {"error": "n must be an int >= 1", "samples": [], "final": "",
                "agreement": 0.0, "n": 0, "aggregate": str(aggregate)}
    if n_i < 1:
        return {"error": "n must be >= 1", "samples": [], "final": "",
                "agreement": 0.0, "n": n_i, "aggregate": agg}
    if agg not in ("majority", "first", "single"):
        return {"error": "unknown aggregate: %r" % aggregate, "samples": [],
                "final": "", "agreement": 0.0, "n": n_i, "aggregate": agg}
    samples = []
    for i in range(n_i):
        s_seed = None if seed is None else int(seed) + i
        r = _call_chat(chat_fn, messages, max_tokens=max_tokens,
                       temperature=temperature, seed=s_seed)
        if "error" in r:
            samples.append({"text": "", "key": "", "error": r["error"]})
        else:
            samples.append({"text": r["text"], "key": answer_key(r["text"])})
    out = {"samples": samples, "final": "", "agreement": 0.0, "n": n_i,
           "aggregate": agg}
    if agg == "single":
        s = samples[0]
        if "error" in s:
            out["error"] = "single sample failed: %s" % s["error"]
        else:
            out["final"], out["agreement"] = s["text"], 1.0
        return out
    final, agreement, _key = _vote(samples, n_i, agg)
    if final is None:
        errs = [s.get("error", "") for s in samples]
        out["error"] = "all %d samples failed: %s" % (n_i, errs[0] if errs else "")
        return out
    out["final"], out["agreement"] = final, agreement
    return out


# ---------------------------------------------------------------------------
# arithmetic audit (hand-rolled matcher, never eval)
# ---------------------------------------------------------------------------

_EXPR = re.compile(
    r"(?<![\d.+\-*])"
    r"(?P<a>-?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
    r"\s*(?P<op>[+\-*])\s*"
    r"(?P<b>-?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
    r"\s*=\s*"
    r"(?P<c>-?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
    r"(?![\d.+\-*/=])"
)
_TOL = 1e-6


def _lit(token):
    """Literal -> number. ``int`` when written without a decimal point."""
    body = token.replace(",", "")
    return float(body) if "." in body else int(body)


def tool_checks(answer: str) -> dict:
    """Recompute inline ``a OP b = c`` claims (OP in ``* + -``).

    Never uses ``eval``: the expression is matched by regex and converted with
    ``int``/``float``. Integers are checked exactly, floats with 1e-6 tolerance.
    Unparseable/chained/variable expressions are skipped, not guessed.
    """
    text = "" if answer is None else str(answer)
    mismatches = []
    checked = 0
    for m in _EXPR.finditer(text):
        try:
            a, b, stated = _lit(m.group("a")), _lit(m.group("b")), _lit(m.group("c"))
        except (TypeError, ValueError):
            continue  # unparseable literal -> skip, never guess
        op = m.group("op")
        if isinstance(a, int) and isinstance(b, int):
            computed = a * b if op == "*" else (a + b if op == "+" else a - b)
        else:
            fa, fb = float(a), float(b)
            computed = fa * fb if op == "*" else (fa + fb if op == "+" else fa - fb)
        checked += 1
        if isinstance(computed, int) and isinstance(stated, int):
            same = computed == stated
        else:
            same = abs(float(computed) - float(stated)) <= _TOL
        if not same:
            mismatches.append({"expr": m.group(0).strip(), "stated": stated,
                               "computed": computed})
    return {"checked": checked, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_SYS = (
    "Break the user's question into at most {max_parts} independent sub-questions. "
    "Reply with numbered lines only, exactly like:\n"
    "1. <sub-question>\n"
    "2. <sub-question>\n"
    "No preamble, no explanation, no numbering beyond the sub-questions."
)
_NUM_LINE = re.compile(r"^\s*(?:[-*]\s*)?(\d{1,2})\s*[.)]\s*(.+?)\s*$")
_NUM_INLINE = re.compile(r"(?:^|\s)(\d{1,2})\s*[.)]\s*")


def _parse_parts(raw, max_parts):
    text = str(raw)
    parts = []
    for line in text.splitlines():
        m = _NUM_LINE.match(line)
        if m and m.group(2).strip():
            parts.append(m.group(2).strip())
        if len(parts) >= max_parts:
            return parts[:max_parts]
    # the model may answer on one line ("1) x 2) y"): prefer whichever reading
    # actually splits into more parts.
    chunks = _NUM_INLINE.split(text)
    inline = []
    for i in range(1, len(chunks) - 1, 2):
        if chunks[i + 1].strip():
            inline.append(chunks[i + 1].strip())
        if len(inline) >= max_parts:
            break
    if len(inline) > len(parts):
        return inline
    return parts[:max_parts] if parts else inline


def decompose(chat_fn, question: str, *, max_parts: int = 4) -> dict:
    """Ask the model for numbered sub-questions. Returns ``{"parts", "raw"}``.

    ``{"error": ...}`` when the question is empty, the model call failed, or no
    numbered part could be parsed. ``raw`` is always present.
    """
    q = "" if question is None else str(question).strip()
    if not q:
        return {"error": "empty question", "parts": [], "raw": ""}
    try:
        mp = int(max_parts)
    except (TypeError, ValueError):
        return {"error": "max_parts must be an int >= 1", "parts": [], "raw": ""}
    if mp < 1:
        return {"error": "max_parts must be >= 1", "parts": [], "raw": ""}
    messages = [{"role": "system",
                 "content": _DECOMPOSE_SYS.format(max_parts=mp)},
                {"role": "user", "content": q}]
    r = _call_chat(chat_fn, messages, max_tokens=192, temperature=0.2)
    if "error" in r:
        return {"error": r["error"], "parts": [], "raw": ""}
    parts = _parse_parts(r["text"], mp)
    if not parts:
        return {"error": "no parts parsed", "parts": [], "raw": r["text"]}
    return {"parts": parts, "raw": r["text"]}


# ---------------------------------------------------------------------------
# self-verification
# ---------------------------------------------------------------------------

_VERIFY_SYS = (
    "You are a strict fact-checking auditor. Read the ANSWER and list every claim "
    "that is unsupported, internally inconsistent, contradicted by the evidence, or "
    "numerically wrong. Reply with ONE fenced JSON block and nothing else:\n"
    "```json\n"
    '[{"claim": "<the claim>", "issue": "<why it is a problem>"}]\n'
    "```\n"
    "If every claim holds, reply with an empty array:\n"
    "```json\n"
    "[]\n"
    "```"
)
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.S)
_NO_ISSUE = ("none", "none.", "no issues", "no issue", "no conflicts", "n/a",
             "ok", "no problems found")
_FLAG_LIMIT = 400  # per field, clipped with a visible "..." when it bites


def _first_fenced_json(text):
    for m in _FENCE.finditer(text or ""):
        body = m.group(2).strip()
        if not body:
            continue
        try:
            return json.loads(body)
        except (TypeError, ValueError):
            continue
    return None


def _evidence_block(evidence):
    rows = []
    for i, ev in enumerate(list(evidence or [])[:4], 1):
        if isinstance(ev, dict):
            txt = (ev.get("text") or ev.get("content") or ev.get("passage")
                   or json.dumps(ev, sort_keys=True))
        else:
            txt = ev
        rows.append("[%d] %s" % (i, _clip(txt, 400)))
    if not rows:
        return ""
    return "EVIDENCE PASSAGES (each truncated to 400 chars):\n" + "\n".join(rows)


def verify(chat_fn, answer: str, evidence: list[dict] | None = None) -> dict:
    """Strict flag pass over ``answer`` (optional ``evidence`` passages).

    Returns ``{"flags": [{"claim", "issue"}, ...], "verdict": ...}`` with verdict
    ``"conflict"`` (flags found), ``"consistent"`` (parseable, empty array) or
    ``"unchecked"`` (no parseable fenced JSON / model call failed). At most the
    first 4 evidence passages are sent, each truncated to 400 chars.
    """
    ans = "" if answer is None else str(answer)
    ev = _evidence_block(evidence)
    user = "ANSWER:\n" + ans
    if ev:
        user += "\n\n" + ev
    r = _call_chat(chat_fn, [{"role": "system", "content": _VERIFY_SYS},
                             {"role": "user", "content": user}],
                   max_tokens=256, temperature=0.0)
    if "error" in r:
        return {"flags": [], "verdict": "unchecked", "error": r["error"]}
    parsed = _first_fenced_json(r["text"])
    if not isinstance(parsed, list):
        return {"flags": [], "verdict": "unchecked"}
    flags = []
    for item in parsed:
        if isinstance(item, dict):
            claim = _clip(item.get("claim", ""), _FLAG_LIMIT)
            issue = _clip(item.get("issue", ""), _FLAG_LIMIT)
        elif isinstance(item, str):
            claim, issue = _clip(item, _FLAG_LIMIT), ""
        else:
            continue
        if not claim and not issue:
            continue
        if claim.lower().strip() in _NO_ISSUE and not issue:
            continue  # small models often answer "none" instead of "[]"
        flags.append({"claim": claim, "issue": issue})
    return {"flags": flags, "verdict": "conflict" if flags else "consistent"}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

_DECOMPOSE_GUIDE = (
    "Answer the user's message as one coherent answer that explicitly covers "
    "each sub-question below. Do not just list them back:\n{items}"
)


def boost_answer(chat_fn, messages: list[dict], *, evidence=None, n: int = 3,
                 temperature: float = 0.7, do_decompose: bool = False,
                 do_verify: bool = True, ground_fn=None) -> dict:
    """Run the boost pipeline and return the answer plus a stage trace.

    Stage order: ``decompose`` (if asked) -> ``consistency`` -> ``tool_checks``
    -> ``verify`` (if asked) -> ``grounding`` (if ``ground_fn`` given). Each
    trace entry is ``{"stage", "seconds", "out"}``; ``seconds`` is that stage's
    duration in seconds (>= 0, rounded to 4 decimals), so prefix sums give
    monotonic elapsed time. ``decompose`` parts are injected as a leading system
    message; a failed decompose is recorded in the trace and the pipeline
    continues on the original messages. ``n == 1`` is inert (no majority claim).
    """
    t_prev = time.perf_counter()
    trace = []
    final = ""
    agreement = 0.0
    tc = {"checked": 0, "mismatches": []}
    verified = None
    grounding = None
    hard_error = None
    msgs = list(messages or [])

    def stage(name, out):
        nonlocal t_prev
        now = time.perf_counter()
        trace.append({"stage": name, "seconds": round(max(now - t_prev, 0.0), 4),
                      "out": out})
        t_prev = now

    if do_decompose:
        d = decompose(chat_fn, _last_user_text(msgs))
        if d.get("parts"):
            items = "\n".join("%d. %s" % (i, p)
                              for i, p in enumerate(d["parts"], 1))
            msgs = [{"role": "system", "content": _DECOMPOSE_GUIDE.format(items=items)}] + msgs
            stage("decompose", {"parts": d["parts"]})
        else:
            stage("decompose", {"error": d.get("error", "no parts parsed")})

    sc = self_consistency(chat_fn, msgs, n=n, temperature=temperature)
    final = sc.get("final", "") or ""
    agreement = float(sc.get("agreement", 0.0))
    errs = sum(1 for s in sc.get("samples", []) if s.get("error"))
    stage("consistency", {"n": sc.get("n", n), "aggregate": sc.get("aggregate"),
                          "agreement": round(agreement, 4), "errors": errs})
    if "error" in sc:
        hard_error = sc["error"]

    tc = tool_checks(final)
    stage("tool_checks", {"checked": tc["checked"],
                          "mismatches": len(tc["mismatches"])})

    if do_verify:
        if hard_error:
            stage("verify", {"skipped": "no answer"})
        else:
            verified = verify(chat_fn, final, evidence)
            stage("verify", {"verdict": verified["verdict"],
                             "flags": len(verified["flags"])})

    if ground_fn is not None:
        if hard_error:
            grounding = {"skipped": "no answer"}
            stage("grounding", {"skipped": "no answer"})
        else:
            try:
                grounding = ground_fn(final, evidence)
                stage("grounding", {"ok": True})
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                grounding = {"error": "%s: %s" % (type(exc).__name__, exc)}
                stage("grounding", {"error": grounding["error"]})

    out = {"final": final, "trace": trace, "agreement": agreement,
           "tool_checks": tc, "real_model_lift": REAL_MODEL_LIFT}
    if verified is not None:
        out["verified"] = verified
    if ground_fn is not None:
        out["grounding"] = grounding
    if hard_error:
        out["error"] = hard_error
    return out
