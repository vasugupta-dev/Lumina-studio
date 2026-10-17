"""Context auto-compaction with prefix-cache-stable checkpoints.

Why it is shaped this way: the prefix KV cache (lumina_framework.chat
generate_cached) reuses the longest common prefix between the previous prompt
and the new one. ANY edit of the front of the history invalidates that prefix,
so compaction must be:

  * rare      — fires only when the estimated token budget is exceeded;
  * one-shot  — older turns collapse into ONE deterministic extractive
                summary block ("checkpoint"), never re-summarized per turn;
  * append-only afterwards — between checkpoints the history only grows at
                the tail, so the cache keeps hitting turn after turn.

Every checkpoint rewrite costs one full prefill; the steady-state hit rate is
measured in tests/test_context_compact.py against a real tiny model, and the
number reported is the measured one (never assumed).

Token estimation is parity-locked with serve._count_tokens (same formula:
whitespace words + 4 per message) — the two call sites must agree or the
budget accounting drifts.
"""
from __future__ import annotations

SUMMARY_HEADER = "[compacted history]"


def _msg_text(m) -> str:
    """Parity copy of serve._msg_text (content may be a string or OpenAI parts)."""
    if not isinstance(m, dict):
        return ""
    t = m.get("content", None)
    if isinstance(t, list):
        return "".join(str(p.get("text", "")) for p in t
                       if isinstance(p, dict) and p.get("type") == "text")
    if t:
        return str(t)
    return str(m.get("text", "") or "")


def est_tokens(messages: list[dict]) -> int:
    """Parity copy of serve._count_tokens."""
    return sum(len(_msg_text(m).split()) + 4 for m in messages)


def _safe(m) -> dict:
    return m if isinstance(m, dict) else {"role": "user", "content": str(m)}


def _is_system(m: dict) -> bool:
    return str(m.get("role", "")) in ("system", "developer")


def summarize_extractive(dropped: list[dict], *, max_chars: int = 600) -> str:
    """Deterministic summary of dropped history — NO model call.

    One compact bullet per dropped message: user turns keep the first 120
    chars, other roles 80. Byte-identical for identical input (tested).
    """
    lines = []
    for raw in dropped:
        m = _safe(raw)
        role = str(m.get("role", "user"))
        t = " ".join(_msg_text(m).split())
        if not t:
            continue
        cap = 120 if role == "user" else 80
        if len(t) > cap:
            t = t[:cap].rstrip() + "..."
        lines.append("- " + role + ": " + t)
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " ..."
    return SUMMARY_HEADER + ("\n" + body if body else "")


def plan_budget(messages: list[dict], *, budget_tokens: int,
                keep_recent: int = 4) -> dict:
    """Decide what to compact. Only the FRONT of the conversation is droppable;
    system/developer messages and the most recent `keep_recent` conversation
    messages are always kept. No message content is modified here.
    """
    msgs = [_safe(m) for m in messages]
    n = len(msgs)
    old = est_tokens(msgs)
    sys_idx = {i for i, m in enumerate(msgs) if _is_system(m)}
    conv_idx = [i for i in range(n) if i not in sys_idx]
    droppable = conv_idx[:-keep_recent] if len(conv_idx) > keep_recent else []
    if old <= budget_tokens:
        return {"compact": False, "drop": [], "kept": list(range(n)),
                "old_tokens": old, "new_tokens": old, "oversize": False,
                "summary": ""}
    if not droppable:
        # Nothing left to drop and still over budget: an unsplittable turn.
        return {"compact": False, "drop": [], "kept": list(range(n)),
                "old_tokens": old, "new_tokens": old, "oversize": True,
                "summary": ""}
    dropped = [msgs[i] for i in droppable]
    summary = summarize_extractive(dropped)
    kept = [i for i in range(n) if i not in set(droppable)]
    new = est_tokens([msgs[i] for i in kept]) + len(summary.split()) + 4
    return {"compact": True, "drop": droppable, "kept": kept,
            "old_tokens": old, "new_tokens": new,
            "oversize": new > budget_tokens, "summary": summary}


def apply_compaction(messages: list[dict], plan: dict, summary: str) -> list[dict]:
    """Return a NEW message list with dropped turns replaced by one summary
    block inserted where the dropped turns started. The input list is never
    mutated (tested)."""
    msgs = [_safe(m) for m in messages]
    if not plan.get("compact"):
        return list(msgs)
    drop = set(plan.get("drop", []))
    out: list[dict] = []
    inserted = False
    for i, m in enumerate(msgs):
        if i in drop:
            if not inserted:
                out.append({"role": "system", "content": summary})
                inserted = True
            continue
        out.append(m)
    return out


def compact_if_needed(messages: list[dict], *, budget_tokens: int,
                      keep_recent: int = 4) -> dict:
    """One-call compaction. Returns the (possibly new) message list plus
    honest counters. When the plan says no, the input is returned as a
    shallow copy — callers can rely on 'compacted' to know."""
    plan = plan_budget(messages, budget_tokens=budget_tokens,
                       keep_recent=keep_recent)
    if not plan["compact"]:
        return {"messages": list(messages), "compacted": False, "dropped": 0,
                "summary_chars": 0, "oversize": plan["oversize"],
                "old_tokens": plan["old_tokens"],
                "new_tokens": plan["old_tokens"]}
    summary = plan["summary"]
    msgs = apply_compaction(messages, plan, summary)
    new_tokens = est_tokens(msgs)
    return {"messages": msgs, "compacted": True, "dropped": len(plan["drop"]),
            "summary_chars": len(summary), "oversize": new_tokens > budget_tokens,
            "old_tokens": plan["old_tokens"], "new_tokens": new_tokens}


def hit_rate(reused: int, computed: int) -> float:
    """reused / (reused + computed); 0.0 when there was nothing to prefill."""
    total = int(reused) + int(computed)
    return (float(reused) / total) if total > 0 else 0.0
