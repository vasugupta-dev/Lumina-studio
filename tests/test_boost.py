"""Boost pipeline suite (app/server/boost.py) -- phone-light, no model loaded.

Everything here is a scripted duck-typed chat_fn, so the suite is deterministic
and mills through in well under a second. Covers:

  * answer_key normalization (punctuation, commas, trailing .0, empty-safe)
  * arithmetic audit (correct/mismatch/commas/floats/tolerance/skips, no eval)
  * self-consistency voting (majority, agreement, deterministic tie-break,
    partial and total sample failure, seed plumbing, inert n=1)
  * decomposition parsing (numbered lines, parentheses form, garbage -> error)
  * verification (fenced JSON flags, empty array, non-JSON -> unchecked,
    evidence truncation to 4 passages x 400 chars)
  * boost_answer pipeline (stage order, trace timing, grounding, n=1, hard error)
  * static honesty: stdlib-only imports, no serve/agent_sdk import, no eval()

Run:  python3 tests/test_boost.py
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

import boost  # noqa: E402

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:200]) if not cond and extra else ""))


# ---------------------------------------------------------------------------
# scripted duck-typed chat clients
# ---------------------------------------------------------------------------

class Scripted:
    """Returns queued replies; records every call. Strings -> {"text": s}."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, max_tokens=192, temperature=0.7, **kw):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "kw": kw})
        if not self.replies:
            return {"error": "script exhausted"}
        r = self.replies.pop(0)
        return r if isinstance(r, dict) else {"text": r}


def sys_text(messages):
    return " ".join(str(m.get("content", "")) for m in messages
                    if m.get("role") == "system")


def stage_names(trace):
    return [e["stage"] for e in trace]


# ---------------------------------------------------------------------------
# 1. answer_key
# ---------------------------------------------------------------------------

def answer_key_case():
    ak = boost.answer_key
    check("answer_key: case + punctuation stripped",
          ak("  Yes!! ") == "yes" and ak("Yes.") == "yes", repr(ak("  Yes!! ")))
    check("answer_key: thousands commas dropped",
          ak("1,000") == "1000" and ak("1,000") == ak("1000"), repr(ak("1,000")))
    check("answer_key: trailing .0 dropped",
          ak("42.0") == "42" and ak("0.0") == "0", repr(ak("42.0")))
    check("answer_key: whitespace collapsed",
          ak("the\t answer\nis") == "the answer is", repr(ak("the\t answer\nis")))
    check("answer_key: empty-safe",
          ak("") == "" and ak(None) == "" and ak("!!!--") == "",
          (repr(ak("")), repr(ak(None)), repr(ak("!!!--"))))
    check("answer_key: non-number precision kept",
          ak("3.14") == "3.14" and ak("1,234.50") == "1234.5",
          (repr(ak("3.14")), repr(ak("1,234.50"))))


# ---------------------------------------------------------------------------
# 2. tool_checks
# ---------------------------------------------------------------------------

def tool_checks_case():
    r = boost.tool_checks("17*23=391")
    check("tool_checks: correct product -> 1 checked, 0 mismatches",
          r["checked"] == 1 and r["mismatches"] == [], r)
    r = boost.tool_checks("17*23=400")
    check("tool_checks: wrong product reported",
          r["checked"] == 1 and len(r["mismatches"]) == 1
          and r["mismatches"][0]["computed"] == 391, r)
    check("tool_checks: mismatch carries expr + stated",
          r["mismatches"][0]["expr"] == "17*23=400"
          and r["mismatches"][0]["stated"] == 400, r["mismatches"][0])
    r = boost.tool_checks("10 + 5 = 16")
    check("tool_checks: spaced sum mismatch computed=15",
          r["checked"] == 1 and r["mismatches"][0]["computed"] == 15, r)
    r = boost.tool_checks("nothing numeric to audit here, just prose.")
    check("tool_checks: prose -> 0 checked", r["checked"] == 0
          and r["mismatches"] == [], r)
    r = boost.tool_checks("1,000-1=999")
    check("tool_checks: comma thousands handled",
          r["checked"] == 1 and r["mismatches"] == [], r)
    r = boost.tool_checks("3.5*2=7.0 and 0.1+0.2=0.3")
    check("tool_checks: float tolerance 1e-6",
          r["checked"] == 2 and r["mismatches"] == [], r)
    r = boost.tool_checks("minus: -5-3=-8")
    check("tool_checks: negative literals", r["checked"] == 1
          and r["mismatches"] == [], r)
    r = boost.tool_checks("1+2+3=6")
    check("tool_checks: chained expression skipped, not guessed",
          r["checked"] == 0, r)
    r = boost.tool_checks("__import__('os')*2=4")
    check("tool_checks: call-like text never matched",
          r["checked"] == 0 and r["mismatches"] == [], r)
    src = open(boost.__file__, encoding="utf-8").read()
    check("tool_checks: module source contains no eval(", "eval(" not in src)


# ---------------------------------------------------------------------------
# 3. self_consistency
# ---------------------------------------------------------------------------

def self_consistency_case():
    ch = Scripted(["B", "A", "B"])
    r = boost.self_consistency(ch, [{"role": "user", "content": "q"}], n=3)
    check("self_consistency: majority final", r["final"] == "B", r)
    check("self_consistency: agreement = 2/3",
          abs(r["agreement"] - 0.6666667) < 1e-6, r["agreement"])
    check("self_consistency: three samples recorded",
          len(r["samples"]) == 3 and [s["key"] for s in r["samples"]]
          == ["b", "a", "b"], r["samples"])
    check("self_consistency: n + aggregate echoed",
          r["n"] == 3 and r["aggregate"] == "majority", r)
    check("self_consistency: n samples actually requested",
          len(ch.calls) == 3 and all(c["max_tokens"] == 192 for c in ch.calls)
          and all(abs(c["temperature"] - 0.7) < 1e-9 for c in ch.calls),
          ch.calls)

    r = boost.self_consistency(Scripted(["B", "A"]), [], n=2)
    check("self_consistency: 1-1 tie -> first-seen wins (documented rule)",
          r["final"] == "B" and abs(r["agreement"] - 0.5) < 1e-9, r)
    r = boost.self_consistency(Scripted(["no", "it is 42"]), [], n=2)
    check("self_consistency: 1-1 tie -> longer normalized text wins",
          r["final"] == "it is 42", r)

    r = boost.self_consistency(Scripted([{"error": "boom"}, "B", "B"]), [], n=3)
    check("self_consistency: errored sample tolerated and agreement /n",
          r["final"] == "B" and abs(r["agreement"] - 0.6666667) < 1e-6
          and r["samples"][0].get("error") == "boom" and "error" not in r, r)

    r = boost.self_consistency(Scripted([{"error": "offline"}] * 3), [], n=3)
    check("self_consistency: all-error -> top-level error + samples kept",
          bool(r.get("error")) and len(r["samples"]) == 3
          and r["final"] == "" and r["agreement"] == 0.0, r)

    ch = Scripted(["x", "y"])
    r = boost.self_consistency(ch, [], n=2, seed=7)
    check("self_consistency: seed plumbed per sample (seed, seed+1)",
          [c["kw"].get("seed") for c in ch.calls] == [7, 8], ch.calls)

    def no_seed_kwarg(messages, max_tokens=192, temperature=0.7):
        return {"text": "ok"}

    r = boost.self_consistency(no_seed_kwarg, [], n=1, seed=1)
    check("self_consistency: client without seed kwarg retried safely",
          r["final"] == "ok", r)

    r = boost.self_consistency(Scripted(["solo"]), [], n=1)
    check("self_consistency: n=1 inert (1 sample, agreement 1.0, no vote)",
          r["final"] == "solo" and r["agreement"] == 1.0
          and r["aggregate"] == "single" and len(r["samples"]) == 1, r)

    check("self_consistency: n<1 refused",
          "error" in boost.self_consistency(Scripted(["x"]), [], n=0))
    check("self_consistency: garbage n refused as a dict, never raised",
          "error" in boost.self_consistency(Scripted(["x"]), [], n="three"))
    check("self_consistency: non-callable client -> error dict",
          "error" in boost.self_consistency(None, [], n=1))
    check("self_consistency: unknown aggregate refused",
          "error" in boost.self_consistency(Scripted(["x"]), [], n=2,
                                           aggregate="mean"))
    check("self_consistency: aggregate='first' honoured",
          boost.self_consistency(Scripted(["B", "A"]), [], n=2,
                                 aggregate="first")["final"] == "B")


# ---------------------------------------------------------------------------
# 4. decompose
# ---------------------------------------------------------------------------

def decompose_case():
    ch = Scripted(["1. first part\n2. second part\n"])
    r = boost.decompose(ch, "explain X")
    check("decompose: numbered lines parsed",
          r["parts"] == ["first part", "second part"] and r["raw"], r)
    check("decompose: raw preserved verbatim",
          r["raw"] == "1. first part\n2. second part\n", repr(r["raw"]))
    check("decompose: prompt asks for numbered parts",
          "numbered" in sys_text(ch.calls[0]["messages"]).lower()
          and ch.calls[0]["max_tokens"] == 192, ch.calls[0])

    r = boost.decompose(Scripted(["1) paren form\n2) second"]), "q")
    check("decompose: '1)' form parsed",
          r["parts"] == ["paren form", "second"], r)

    r = boost.decompose(Scripted(["1) a 2) b 3) c"]), "q", max_parts=3)
    check("decompose: single-line fallback parsed", r["parts"] == ["a", "b", "c"], r)

    ch = Scripted(["1. a\n2. b\n3. c\n4. d\n5. e"])
    r = boost.decompose(ch, "q", max_parts=4)
    check("decompose: max_parts bounds the parts", len(r["parts"]) == 4, r)

    r = boost.decompose(Scripted(["I cannot help with that."]), "q")
    check("decompose: garbage -> error with raw kept",
          r.get("error") == "no parts parsed" and r["parts"] == []
          and r["raw"] == "I cannot help with that.", r)
    r = boost.decompose(Scripted([{"error": "offline"}]), "q")
    check("decompose: chat error surfaced", r.get("error") == "offline"
          and r["raw"] == "", r)
    r = boost.decompose(Scripted(["1. x"]), "   ")
    check("decompose: empty question refused without a model call",
          "error" in r, r)
    check("decompose: garbage max_parts refused as a dict",
          "error" in boost.decompose(Scripted(["1. x"]), "q", max_parts="two"))
    check("decompose: non-callable client -> error dict",
          "error" in boost.decompose(None, "q"))


# ---------------------------------------------------------------------------
# 5. verify
# ---------------------------------------------------------------------------

def verify_case():
    flags = '[{"claim": "17*23=400", "issue": "product is wrong"}]'
    ch = Scripted(["checking...\n```json\n" + flags + "\n```\ntrailing prose"])
    r = boost.verify(ch, "17*23=400")
    check("verify: fenced JSON flags -> conflict",
          r["verdict"] == "conflict" and r["flags"] == [
              {"claim": "17*23=400", "issue": "product is wrong"}], r)
    check("verify: strict prompt sent",
          "fenced JSON" in sys_text(ch.calls[0]["messages"]), ch.calls[0])

    r = boost.verify(Scripted(["```json\n[]\n```"]), "all good")
    check("verify: empty array -> consistent, no flags",
          r["verdict"] == "consistent" and r["flags"] == [], r)
    r = boost.verify(Scripted(["No problems found, honestly."]), "x")
    check("verify: non-JSON -> unchecked (never a false consistent)",
          r["verdict"] == "unchecked" and r["flags"] == [], r)
    r = boost.verify(Scripted(['```json\n{"claim": "x"}\n```']), "x")
    check("verify: non-array JSON -> unchecked", r["verdict"] == "unchecked", r)
    r = boost.verify(Scripted([{"error": "offline"}]), "x")
    check("verify: chat error -> unchecked + error recorded",
          r["verdict"] == "unchecked" and r.get("error") == "offline", r)
    r = boost.verify(Scripted(["```json\n[\"none\"]\n```"]), "x")
    check("verify: 'none' sentinel not treated as a conflict",
          r["verdict"] == "consistent", r)

    long_passage = "A" * 900 + " TAILMARK1"
    ev = [{"text": long_passage}] + [{"text": "p%d" % i} for i in range(2, 8)]
    ch = Scripted(["```json\n[]\n```"])
    boost.verify(ch, "answer", ev)
    prompt = ch.calls[0]["messages"][-1]["content"]
    check("verify: evidence included, capped at 4 passages",
          "[1]" in prompt and "[4]" in prompt and "[5]" not in prompt, prompt[:200])
    check("verify: passages truncated to 400 chars",
          "TAILMARK1" not in prompt and "A" * 400 in prompt)
    check("verify: evidence note only when evidence given",
          "EVIDENCE" in prompt
          and "EVIDENCE" not in Scripted(["```json\n[]\n```"]).__call__(
              [{"role": "system", "content": ""}, {"role": "user", "content": ""}])["text"])


# ---------------------------------------------------------------------------
# 6. boost_answer
# ---------------------------------------------------------------------------

def pipeline_client(verify_reply='```json\n[{"claim": "x", "issue": "y"}]\n```'):
    """Scripted pipeline client: dispatch on the stage's system prompt."""
    calls = []

    def chat_fn(messages, max_tokens=192, temperature=0.7, **kw):
        calls.append({"sys": sys_text(messages), "temperature": temperature,
                      "max_tokens": max_tokens, "kw": kw})
        s = sys_text(messages)
        if "independent sub-questions" in s:
            return {"text": "1. part one\n2. part two"}
        if "fact-checking auditor" in s:
            return {"text": verify_reply}
        return {"text": "17*23=391"}

    return chat_fn, calls


def boost_answer_case():
    ground = {"passages": [{"doc": "d1"}], "note": "stub grounding"}
    ch, calls = pipeline_client()
    seen = {}

    def ground_fn(final, evidence):
        seen["args"] = (final, evidence)
        return ground

    r = boost.boost_answer(ch, [{"role": "user", "content": "what is 17*23?"}],
                           evidence=[{"text": "17*23=391"}], n=3,
                           do_decompose=True, do_verify=True, ground_fn=ground_fn)
    check("boost_answer: stage order decompose->consistency->tool_checks->verify->grounding",
          stage_names(r["trace"]) == ["decompose", "consistency", "tool_checks",
                                      "verify", "grounding"], stage_names(r["trace"]))
    check("boost_answer: final is the majority text",
          r["final"] == "17*23=391" and r["agreement"] == 1.0, r["final"])
    check("boost_answer: tool_checks run on the final answer",
          r["tool_checks"] == {"checked": 1, "mismatches": []}, r["tool_checks"])
    check("boost_answer: verified block from the flag pass",
          r["verified"]["verdict"] == "conflict"
          and r["verified"]["flags"][0]["claim"] == "x", r.get("verified"))
    check("boost_answer: grounding passed through verbatim",
          r["grounding"] == ground and seen["args"] == ("17*23=391",
                                                       [{"text": "17*23=391"}]),
          r.get("grounding"))
    check("boost_answer: real_model_lift present, labelled unmeasured",
          isinstance(r.get("real_model_lift"), str)
          and "not measured" in r["real_model_lift"], r.get("real_model_lift"))
    check("boost_answer: decompose parts injected as guidance",
          any("part one" in c["sys"] and "part two" in c["sys"]
              for c in calls), calls[:1])
    secs = [e["seconds"] for e in r["trace"]]
    prefix = [sum(secs[:i + 1]) for i in range(len(secs))]
    check("boost_answer: trace seconds are non-negative floats",
          all(isinstance(s, float) and s >= 0.0 for s in secs), secs)
    check("boost_answer: prefix-summed trace time is monotonic",
          prefix == sorted(prefix), prefix)
    check("boost_answer: consistent temperature / n samples",
          sum(1 for c in calls if "fact-checking" not in c["sys"]
              and "sub-questions" not in c["sys"]) == 3
          and all(abs(c["temperature"] - 0.7) < 1e-9 for c in calls
                  if "fact-checking" not in c["sys"]
                  and "sub-questions" not in c["sys"]), calls)

    ch2, _ = pipeline_client()
    r2 = boost.boost_answer(ch2, [{"role": "user", "content": "what is 17*23?"}],
                            n=3, do_decompose=False, do_verify=True)
    check("boost_answer: no decompose -> stage order without it",
          stage_names(r2["trace"]) == ["consistency", "tool_checks", "verify"],
          stage_names(r2["trace"]))
    check("boost_answer: no ground_fn -> no grounding key",
          "grounding" not in r2 and "decompose" not in stage_names(r2["trace"]), r2)

    r3 = boost.boost_answer(*pipeline_client(verify_reply="```json\n[]\n```"))
    check("boost_answer: verify consistent path",
          r3["verified"]["verdict"] == "consistent"
          and stage_names(r3["trace"]) == ["consistency", "tool_checks", "verify"],
          r3.get("verified"))

    ch4, _ = pipeline_client()
    r4 = boost.boost_answer(ch4, [{"role": "user", "content": "q"}], n=2,
                            do_decompose=False, do_verify=False)
    check("boost_answer: do_verify=False -> no verify stage, no verified block",
          "verify" not in stage_names(r4["trace"]) and "verified" not in r4, r4)

    ch5, _ = pipeline_client()
    r5 = boost.boost_answer(ch5, [{"role": "user", "content": "q"}], n=1,
                            do_decompose=False, do_verify=False)
    check("boost_answer: n=1 inert (agreement 1.0, single sample final)",
          r5["final"] == "17*23=391" and r5["agreement"] == 1.0
          and r5["trace"][0]["out"]["aggregate"] == "single", r5["trace"][0])

    def dead_chat(messages, max_tokens=192, temperature=0.7, **kw):
        return {"error": "offline"}

    r6 = boost.boost_answer(dead_chat, [{"role": "user", "content": "q"}], n=2,
                            do_verify=True, ground_fn=lambda a, e: {"never": True})
    step = {e["stage"]: e["out"] for e in r6["trace"]}
    check("boost_answer: dead client -> error surfaced, verify/grounding skipped",
          bool(r6.get("error")) and r6["final"] == ""
          and step["verify"].get("skipped") == "no answer"
          and step["grounding"].get("skipped") == "no answer"
          and r6["grounding"].get("skipped") == "no answer", r6)

    def bad_ground(final, evidence):
        raise RuntimeError("no corpus")

    r7 = boost.boost_answer(*pipeline_client(), ground_fn=bad_ground, n=1)
    check("boost_answer: ground_fn failure recorded, not raised",
          r7["grounding"].get("error", "").startswith("RuntimeError")
          and r7["trace"][-1]["out"].get("error", "").startswith("RuntimeError"), r7)

    a = boost.boost_answer(*pipeline_client(), do_decompose=True, n=3,
                           ground_fn=ground)
    b = boost.boost_answer(*pipeline_client(), do_decompose=True, n=3,
                           ground_fn=ground)

    def strip(r):
        return {k: v for k, v in r.items() if k != "trace"} | {
            "trace": [(e["stage"], e["out"]) for e in r["trace"]]}

    check("boost_answer: deterministic for identical chat_fn", strip(a) == strip(b),
          (strip(a), strip(b)))


# ---------------------------------------------------------------------------
# 7. static honesty: standalone, stdlib-only, no eval
# ---------------------------------------------------------------------------

def static_case():
    src = open(boost.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    tops = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            tops.add((node.module or "").split(".")[0])
    check("static: top-level imports are stdlib-only",
          tops <= {"__future__", "json", "re", "time"}, sorted(tops))
    check("static: boost never imports serve or agent_sdk",
          not ({"serve", "agent_sdk"} & tops)
          and "import serve" not in src and "import agent_sdk" not in src,
          sorted(tops))
    check("static: public API complete",
          all(callable(getattr(boost, n, None)) for n in
              ("answer_key", "self_consistency", "decompose", "verify",
               "tool_checks", "boost_answer")))
    P = inspect.Parameter
    want = {
        "answer_key": [("text", False, P.empty)],
        "self_consistency": [("chat_fn", False, P.empty), ("messages", False, P.empty),
                             ("n", True, 3), ("temperature", True, 0.7),
                             ("max_tokens", True, 192), ("aggregate", True, "majority"),
                             ("seed", True, None)],
        "decompose": [("chat_fn", False, P.empty), ("question", False, P.empty),
                      ("max_parts", True, 4)],
        "verify": [("chat_fn", False, P.empty), ("answer", False, P.empty),
                   ("evidence", False, None)],
        "tool_checks": [("answer", False, P.empty)],
        "boost_answer": [("chat_fn", False, P.empty), ("messages", False, P.empty),
                         ("evidence", True, None), ("n", True, 3),
                         ("temperature", True, 0.7), ("do_decompose", True, False),
                         ("do_verify", True, True), ("ground_fn", True, None)],
    }
    got = {k: [(p.name, p.kind == P.KEYWORD_ONLY, p.default)
               for p in inspect.signature(getattr(boost, k)).parameters.values()]
           for k in want}
    check("static: public signatures match the contracted shape", got == want,
          {k: got[k] for k in want if got[k] != want[k]})


def main() -> int:
    t0 = time.time()
    for case in (answer_key_case, tool_checks_case, self_consistency_case,
                 decompose_case, verify_case, boost_answer_case, static_case):
        case()
    ok = sum(1 for _, c in RESULTS if c)
    print("\nBOOST: %d/%d passed (%.1fs)" % (ok, len(RESULTS), time.time() - t0))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
