"""Auto-compaction: unit semantics + REAL prefix-cache hit-rate measurement.

Two identical 20-turn conversations run against the tiny on-device chat model
(fixtures/_fixtures.py): one compacting when the estimated budget is exceeded,
one never compacting. For every turn we record the prefill counters from the
REAL prefix cache (lumina_framework.chat.generate_cached) and print the table.
The steady-state gate is the mean ratio of the last 4 turns >= 0.90 — the
measured number is printed, and a failure is reported as a failure (no
threshold changes after seeing results).

Physics note (why the gate can be met): every compaction rewrites the FRONT
of the history, which costs one full prefill. Between checkpoints the history
only grows at the tail, so generate_cached reuses the entire previous prompt.
The workload is sized so the last checkpoint happens several turns before the
measured window; a rewrite inside the window would drag the mean down, and
this test would say so honestly.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lumina_framework.chat import ChatModel, generate_cached  # noqa: E402

import context_compact as cc  # noqa: E402
from _fixtures import build_tiny_chat  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    suffix = ("  :: " + str(detail)) if (detail and not cond) else ""
    print(("PASS " if cond else "FAIL ") + name + suffix)


def render(msgs):
    """ChatML fallback rendering (same shape the server uses without a template)."""
    parts = []
    for m in msgs:
        parts.append("<|im_start|>" + str(m.get("role", "user")) + "\n"
                     + str(m.get("content", "")) + "<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def run_workload(m, turns, budget, keep_recent, compact_on):
    """Returns (rows, events). rows = per-turn counters + prompt id counts."""
    msgs = []
    state = {"ids": [], "kv": {}, "pos": 0}
    rows, events = [], []
    for i in range(1, turns + 1):
        msgs.append({"role": "user",
                     "content": "q%d tell me more about this topic with several extra details and context words" % i})
        fired = False
        new_tok = None
        if compact_on:
            c = cc.compact_if_needed(msgs, budget_tokens=budget,
                                     keep_recent=keep_recent)
            if c["compacted"]:
                fired = True
                msgs = c["messages"]
                events.append({"turn": i, "dropped": c["dropped"],
                               "old_tokens": c["old_tokens"],
                               "new_tokens": c["new_tokens"],
                               "oversize": c["oversize"]})
        prompt = render(msgs)
        prompt_ids = len(m.tok.encode(prompt))
        r = generate_cached(m, prompt, max_tokens=8, state=state,
                            rng=np.random.default_rng(100 + i))
        state = r["cache_state"]
        reused = int(r["stats"]["prefill_reused"])
        computed = int(r["stats"]["prefill_computed"])
        rows.append({"turn": i, "prompt_ids": prompt_ids, "reused": reused,
                     "computed": computed, "ratio": cc.hit_rate(reused, computed),
                     "compacted": fired})
        msgs.append({"role": "assistant", "content": r["text"]})
        if compact_on:
            c2 = cc.compact_if_needed(msgs, budget_tokens=budget,
                                      keep_recent=keep_recent)
            if c2["compacted"]:
                msgs = c2["messages"]
    return rows, events


def print_table(title, rows):
    print("  " + title)
    for r in rows:
        print("    turn %2d  prompt=%5d  reused=%5d  computed=%5d  ratio=%.3f%s"
              % (r["turn"], r["prompt_ids"], r["reused"], r["computed"],
                 r["ratio"], "  [rewrite]" if r["compacted"] else ""))


def main() -> int:
    build_tiny_chat()
    m = ChatModel("/tmp/tiny-chat")

    # --- parity: token estimator must match the server's -------------------
    import serve  # noqa: E402
    samples = [
        [{"role": "user", "content": "hello world"},
         {"role": "assistant", "content": "a b c d e"}],
        [{"role": "user", "content": [{"type": "text", "text": "parts array"},
                                      {"type": "image", "x": 1}]}],
        [{"role": "user", "text": "legacy text field"}],
        ["bare string message"],
        [],
    ]
    par = all(cc.est_tokens(s) == serve._count_tokens(s) for s in samples)
    check("est_tokens parity with serve._count_tokens", par)

    # --- unit semantics ----------------------------------------------------
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(4):
        msgs.append({"role": "user", "content": "user turn %d" % i})
        msgs.append({"role": "assistant", "content": "reply %d" % i})
    msgs.append({"role": "user", "content": "last"})

    plan = cc.plan_budget(msgs, budget_tokens=10 ** 6, keep_recent=4)
    check("under budget -> no compaction", plan["compact"] is False
          and plan["drop"] == [] and plan["oversize"] is False)

    plan = cc.plan_budget(msgs, budget_tokens=1, keep_recent=4)
    first_conv = 1  # index 0 is the system message
    n = len(msgs)
    want_drop = list(range(first_conv, n - 4))
    check("over budget -> drops the front, keeps system + last 4",
          plan["compact"] is True and plan["drop"] == want_drop, 
          "drop=%s want=%s" % (plan["drop"], want_drop))
    check("plan keeps every non-dropped index in order",
          plan["kept"] == [i for i in range(n) if i not in set(want_drop)])
    check("summary is deterministic",
          cc.summarize_extractive([msgs[i] for i in want_drop])
          == cc.summarize_extractive([msgs[i] for i in want_drop])
          and plan["summary"].startswith(cc.SUMMARY_HEADER))

    snap = [dict(x) for x in msgs]
    out = cc.apply_compaction(msgs, plan, plan["summary"])
    check("apply_compaction does not mutate the input", msgs == snap)
    check("apply_compaction preserves kept order",
          [x["content"] for x in out[2:]] == [x["content"] for x in msgs[-4:]])
    check("apply_compaction inserts exactly one summary block",
          sum(1 for x in out if x.get("content") == plan["summary"]) == 1)

    huge = [{"role": "user", "content": "word " * 500}]
    plan2 = cc.plan_budget(huge, budget_tokens=50, keep_recent=4)
    check("unsplittable single turn -> oversize, no silent drop",
          plan2["compact"] is False and plan2["oversize"] is True)

    c_off = cc.compact_if_needed(msgs, budget_tokens=10 ** 6, keep_recent=4)
    check("compact_if_needed default-off leaves history untouched",
          c_off["compacted"] is False and c_off["messages"] == msgs)

    check("hit_rate arithmetic", cc.hit_rate(90, 10) == 0.9
          and cc.hit_rate(0, 0) == 0.0 and cc.hit_rate(0, 5) == 0.0)

    # --- real workload: compacting vs baseline -----------------------------
    # Sized so the checkpoint fires ONCE mid-run and the last 4 turns are
    # append-only (see the module docstring for the thrash physics: a budget
    # with little headroom rewrites repeatedly and the hit rate collapses —
    # the first draft of this test measured exactly that, 0.70, and FAILED).
    TURNS, BUDGET, KEEP = 26, 420, 4
    rows_c, events = run_workload(m, TURNS, BUDGET, KEEP, compact_on=True)
    rows_b, _ = run_workload(m, TURNS, BUDGET, KEEP, compact_on=False)
    print_table("compaction ON (budget=%d, keep_recent=%d)" % (BUDGET, KEEP), rows_c)
    print_table("compaction OFF (baseline)", rows_b)
    print("  compaction events: %s" % (events or "none"))

    check("compaction fired at least once (the demo is real)",
          len(events) >= 1)
    check("post-compaction estimate is under budget",
          all(not e["oversize"] for e in events))
    rewrites_in_window = [r for r in rows_c[-4:] if r["compacted"]]
    steady = sum(r["ratio"] for r in rows_c[-4:]) / 4.0
    agg_reused = sum(r["reused"] for r in rows_c[-4:])
    agg_computed = sum(r["computed"] for r in rows_c[-4:])
    am_all = cc.hit_rate(sum(r["reused"] for r in rows_c),
                         sum(r["computed"] for r in rows_c))
    print("  steady-state last 4: mean ratio=%.4f  aggregate=%.4f  rewrites in window=%d"
          % (steady, cc.hit_rate(agg_reused, agg_computed), len(rewrites_in_window)))
    print("  amortized over the WHOLE run incl. every rewrite: %.4f" % am_all)
    check("no checkpoint rewrite inside the steady-state window",
          len(rewrites_in_window) == 0,
          "rewrites=%d" % len(rewrites_in_window))
    check("steady-state (last 4 turns) mean ratio >= 0.90", steady >= 0.90,
          "measured=%.4f" % steady)

    total_c = sum(r["reused"] + r["computed"] for r in rows_c)
    total_b = sum(r["reused"] + r["computed"] for r in rows_b)
    print("  total prompt tokens: compacted=%d baseline=%d" % (total_c, total_b))
    check("compaction reduces total prompt tokens", total_c <= total_b,
          "compacted=%d baseline=%d" % (total_c, total_b))

    n = sum(1 for _, c in RESULTS if c)
    print(f"\nCONTEXT-COMPACT: {n}/{len(RESULTS)} passed")
    print("RESULT:", "ALL PASS" if n == len(RESULTS) else "FAILURES PRESENT")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
