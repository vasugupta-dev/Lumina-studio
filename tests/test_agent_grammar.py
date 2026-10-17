"""Grammar-locked tool calls (app/server/agent_grammar.py) + agent-loop wiring.

Inspired by Atomic Agent (atomicagent.io): grammar-constrained tool calls,
bounded reasoning tails, parallel read-only batches. Lumina has no logits mask,
so this is a schema layer: tolerant parse, validate+repair, one bounded retry,
parallel read-only execution.

Gates fixed before measuring:
  G1 tolerant parse of a trailing-comma fenced block
  G2 bounded retry: a bad block is retried once, then a valid call runs
  G3 parallel read-only batch: 4 x 0.15 s stubs finish < 0.45 s, order kept
  G4 grammar stats are reported and consistent
  G5 grammar="off" keeps the legacy single-call path working
  G6 the grammar block is injected once (byte-stable prefix for the KV cache)

Run:  python3 tests/test_agent_grammar.py
Exit codes: 0 = pass, 1 = fail, 3 = honest skip.
"""
from __future__ import annotations

import os
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

import agent_grammar as ag  # noqa: E402

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:200]) if not cond and extra else ""))


def main() -> int:
    t0 = time.time()

    # ---- G1 tolerant parsing -------------------------------------------
    p1 = ag.parse_calls('reasoning\n```tool\n[{"tool":"fs_list","args":{"path":"~"}},]\n```')
    check("G1 parse: trailing comma tolerated",
          p1.get("calls", [{}])[0].get("tool") == "fs_list", p1)
    p2 = ag.parse_calls('```tool\n{"tool":"fetch_url","args":{"url":"http://x"}}\n```')
    check("parse: single object accepted", len(p2.get("calls", [])) == 1, p2)
    p3 = ag.parse_calls("no tool block here")
    check("parse: no block -> empty calls (final)", p3.get("calls") == [], p3)
    p4 = ag.parse_calls('```tool\n{not json}\n```')
    check("parse: unparseable -> error", "error" in p4, p4)
    p5 = ag.parse_calls('```tool\n[{"tool":"x"}],\n```')
    check("parse: not a list of calls -> error", "error" in p5, p5)

    # ---- validation + repair -------------------------------------------
    v = ag.validate_call({"tool": "fs_read", "args": {}}, {"fs_read"})
    check("validate: missing required arg rejected", v["ok"] is False, v)
    v2 = ag.validate_call({"tool": "nope", "args": {}}, {"fs_read"})
    check("validate: unknown tool rejected", v2["ok"] is False, v2)
    v3 = ag.validate_call({"tool": "media_generate",
                           "args": {"capability": "image", "prompt": "x"}},
                          None)
    check("validate: complete required args accepted", v3["ok"] is True, v3)
    rep = ag.repair_call({"tool": "media_generate",
                          "args": {"capability": "image", "prompt": "x",
                                   "width": "256", "bogus": 1}})
    a = rep["call"]["args"]
    check("repair: numeric string coerced", a.get("width") == 256, a)
    check("repair: default filled", a.get("steps") == 20, a)
    check("repair: unknown key dropped", "bogus" not in a, a)
    check("repair: required args kept", a.get("prompt") == "x", a)

    # ---- grammar block + bounded tail ----------------------------------
    gb = ag.grammar_block([{"tool": "fs_list"}, {"tool": "media_generate"},
                          {"tool": "mcp:x::y"}])
    check("grammar block: declares typed args",
          "fs_list(path:string)" in gb and "media_generate(" in gb, gb[:120])
    check("grammar block: omits schema-less tools", "mcp:x::y" not in gb, gb[:120])
    bt = ag.bounded_tail("A" * 3000, 100)
    check("bounded tail: length bounded", len(bt) <= 200, len(bt))
    check("bounded tail: keeps head and tail",
          bt.startswith("AAAA") and bt.endswith("AAAA"), bt[:20])

    # ---- batch planning + parallel execution ---------------------------
    check("plan: two read-only calls are parallel",
          ag.plan_batch([{"tool": "fs_list", "args": {}},
                         {"tool": "fetch_url", "args": {}}])["parallel"] is True)
    check("plan: gated tool forces serial",
          ag.plan_batch([{"tool": "fs_list", "args": {}},
                         {"tool": "fs_write", "args": {}}])["parallel"] is False)
    check("plan: single call is not a batch",
          ag.plan_batch([{"tool": "fs_list", "args": {}}])["parallel"] is False)

    lock = threading.Lock()
    state = {"active": 0, "max": 0, "order": []}

    def slow_runner(call):
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            state["order"].append(call["args"].get("i"))
        time.sleep(0.15)
        with lock:
            state["active"] -= 1
        return {"out": "ok-" + str(call["args"].get("i"))}

    calls = [{"tool": "fs_list", "args": {"i": i}} for i in range(4)]
    b = ag.execute_batch(calls, slow_runner, max_workers=4)
    check("G3 parallel: 4x0.15s finish < 0.45s", b["seconds"] < 0.45,
          round(b["seconds"], 3))
    check("G3 parallel: order preserved", state["order"] == [0, 1, 2, 3],
          state["order"])
    check("G3 parallel: actually concurrent", state["max"] >= 2, state["max"])

    # ---- agent-loop integration (scripted model) -----------------------
    import serve

    real_chat = serve.api_chat
    real_tool = serve._agent_tool_call
    try:
        script = [
            {"text": 'thinking\n```tool\n{not json}\n```'},          # bad block
            {"text": '```tool\n{"tool":"fs_list","args":{"path":"~"}}\n```'},
            {"text": "All done, here is the answer."},
        ]
        seen = {"n": 0}

        def fake_chat(messages, *a, **k):
            i = seen["n"]
            seen["n"] += 1
            seen.setdefault("msgs", []).append(messages)
            return script[min(i, len(script) - 1)]

        tools_seen = []

        def fake_tool(call):
            tools_seen.append(call.get("tool"))
            return {"out": "listed ~"}

        serve.api_chat = fake_chat
        serve._agent_tool_call = fake_tool
        out = serve._agent_builtin({"messages": [{"role": "user",
                                                  "content": "list my files"}],
                                    "max_steps": 4})
        g = out.get("grammar") or {}
        check("G2 bounded retry: bad block retried once",
              g.get("retries") == 1 and g.get("refused", 0) >= 1, g)
        check("G2 bounded retry: valid call executed after retry",
              "fs_list" in tools_seen, tools_seen)
        check("loop: reaches a final answer", "All done" in str(out.get("text", "")),
              str(out.get("text"))[:60])
        check("G4 stats: reported and consistent",
              g.get("valid_first", 0) >= 1
              and (g.get("refused", 0) + g.get("repaired", 0)
                   + g.get("valid_first", 0)) >= 1, g)
        check("G6 prefix: grammar block injected exactly once",
              sum(str(m.get("content", "")).count("Tool-call grammar")
                  for m in seen.get("msgs", [])[0]) == 1
              and all(sum(str(m.get("content", "")).count("Tool-call grammar")
                          for m in mm) == 1 for mm in seen.get("msgs", [])),
              [sum(str(m.get("content", "")).count("Tool-call grammar")
                   for m in mm) for mm in seen.get("msgs", [])])

        # parallel batch through the loop
        script2 = [
            {"text": '```tool\n[{"tool":"fs_list","args":{"path":"~"}},'
                     '{"tool":"fetch_url","args":{"url":"http://x"}}]\n```'},
            {"text": "done"},
        ]
        seen["n"] = 0
        serve.api_chat = lambda messages, *a, **k: script2[min(seen.__setitem__(
            "n", seen["n"] + 1) or seen["n"] - 1, len(script2) - 1)]
        out2 = serve._agent_builtin({"messages": [{"role": "user",
                                                   "content": "go"}],
                                     "max_steps": 3})
        g2 = out2.get("grammar") or {}
        check("loop: parallel read-only batch reported",
              g2.get("parallel_batches", 0) >= 1 and g2.get("parallel_calls", 0) >= 2,
              g2)

        # legacy path when grammar is off
        seen["n"] = 0
        serve.api_chat = lambda messages, *a, **k: (
            {"text": '```tool\n{"tool":"fs_list","args":{"path":"~"}}\n```'}
            if seen["n"] == 0 and seen.__setitem__("n", 1) is None
            else {"text": "finished"})
        out3 = serve._agent_builtin({"messages": [{"role": "user", "content": "x"}],
                                     "max_steps": 2, "grammar": "off"})
        check("G5 grammar off: legacy path still works",
              "finished" in str(out3.get("text", "")) and "grammar" not in out3,
              str(out3.get("text"))[:60])
    finally:
        serve.api_chat = real_chat
        serve._agent_tool_call = real_tool

    # ---- atomic registry (typed blocks) ---------------------------------
    import serve
    real_tool2 = serve._agent_tool_call
    try:
        serve._agent_tool_call = lambda call: {"out": "atom-ok"}
        atoms = serve.api_atoms_list()
        ids = {a["id"] for a in atoms["atoms"]}
        check("atoms: typed tools listed",
              "fs_list" in ids and "media_generate" in ids, sorted(ids)[:6])
        fs = next(a for a in atoms["atoms"] if a["id"] == "fs_list")
        check("atoms: contract declared",
              fs["schema"].get("path", {}).get("required") is True, fs)
        check("atoms: skills listed", atoms["skills"] >= 30, atoms["skills"])
        r = serve.api_atoms_run({"id": "fs_list", "args": {}})
        check("atoms/run: missing required arg reported",
              bool(r.get("validation_errors")), r)
        r2 = serve.api_atoms_run({"id": "media_generate",
                                  "args": {"capability": "image", "prompt": "x",
                                           "width": "256", "bogus": 1}})
        check("atoms/run: repair applied",
              "coerced 'width' to number" in str(r2.get("repairs")),
              r2.get("repairs"))
        r3 = serve.api_atoms_run({"id": "summarize", "args": {"input": "hello"}})
        check("atoms/run: skill renders its prompt",
              "hello" in str(r3.get("rendered")), str(r3)[:80])
        r4 = serve.api_atoms_run({"id": "nope-nope", "args": {}})
        check("atoms/run: unknown atom errors honestly", "error" in r4, r4)
    finally:
        serve._agent_tool_call = real_tool2

    ok = sum(1 for _, c in RESULTS if c)
    print("\nAGENT-GRAMMAR: %d/%d passed (%.1fs)"
          % (ok, len(RESULTS), time.time() - t0))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
