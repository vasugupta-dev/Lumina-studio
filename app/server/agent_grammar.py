"""Grammar-locked tool calls for Lumina's builtin agent loop.

Inspiration (borrowed architecture, not code): Atomic Agent (atomicagent.io)
forces structurally-valid tool calls from tiny local models with GBNF
grammar-constrained decoding, keeps reasoning on-budget with bounded tails, and
runs independent read-only calls in parallel; Atomic Agents (aurelio-labs)
gives every block a typed input/output contract.

Honest scope: Lumina's framework exposes no logits mask, so this is NOT true
GBNF. It is a schema layer around the emitted tool JSON:

  * typed contracts for Lumina's own tools (TOOL_SCHEMAS) — like the LEGO
    blocks' typed I/O
  * tolerant parsing of the fenced block (trailing commas, single call or array)
  * validation + repair (type coercion, default filling, unknown-key dropping)
  * one bounded retry with a correction tail when a call is unusable
  * parallel execution for independent read-only calls

Everything is measured and reported by the caller; nothing is hidden.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

_CALL_RE = re.compile(r"```tool\s*(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

# (type, required, default)
TOOL_SCHEMAS: dict[str, dict[str, tuple]] = {
    "fs_list": {"path": ("string", True)},
    "fs_read": {"path": ("string", True)},
    "fs_write": {"path": ("string", True), "text": ("string", True)},
    "fs_delete": {"path": ("string", True)},
    "fs_search": {"path": ("string", True), "query": ("string", True)},
    "exec": {"cmd": ("string", True), "cwd": ("string", False),
             "timeout": ("number", False, 15)},
    "fetch_url": {"url": ("string", True)},
    "app_open": {"app": ("string", True), "args": ("array", False, [])},
    "app_list": {},
    "skill_fetch": {"query": ("string", True), "k": ("number", False, 3)},
    "media_generate": {"capability": ("string", True), "prompt": ("string", True),
                       "workflow_id": ("string", False),
                       "width": ("number", False, 1024),
                       "height": ("number", False, 1024),
                       "steps": ("number", False, 20),
                       "seed": ("number", False), "policy": ("string", False)},
    "media_status": {},
    "browser_navigate": {"url": ("string", True)},
    "browser_click": {"selector": ("string", True)},
    "browser_fill": {"selector": ("string", True), "value": ("string", True)},
    "browser_type_human": {"selector": ("string", True),
                           "value": ("string", True)},
    "browser_press": {"key": ("string", True)},
    "browser_select": {"selector": ("string", True),
                       "label_or_value": ("string", True)},
    "browser_text": {}, "browser_screenshot": {},
    "browser_scroll": {"dy": ("number", False, 600)},
    "browser_wait_for": {"selector": ("string", True),
                         "timeout_ms": ("number", False, 8000)},
    "browser_back": {}, "browser_tabs": {}, "browser_close": {},
}

# read-only tools may run concurrently; everything else is serial and gated
READ_ONLY: frozenset = frozenset({
    "fs_list", "fs_read", "fs_search", "fetch_url", "browser_text",
    "browser_screenshot", "browser_tabs", "browser_wait_for", "app_list",
    "media_status", "skill_fetch",
})

_TYPES = ("string", "number", "array", "object", "boolean")


def schema_for(tool: str) -> dict:
    """Declared contract for a tool ({} for MCP/unknown tools)."""
    if tool in TOOL_SCHEMAS:
        return TOOL_SCHEMAS[tool]
    if tool.startswith("browser_"):
        return TOOL_SCHEMAS.get(tool, {})
    return {}


def grammar_block(specs: list[dict] | None = None, max_tools: int = 24) -> str:
    """System-prompt block: exact JSON shape + typed args for each tool."""
    lines = ["Tool-call grammar (grammar-locked):",
             'Emit tool calls as a fenced block containing ONLY a JSON array:',
             "```tool",
             '[{"tool": "<id>", "args": {...}}]',
             "```",
             "Rules: tool must be from the list below; args must match the "
             "declared types; 0..15 calls per block; no prose inside the block."]
    for s in (specs or [])[:max_tools]:
        t = str(s.get("tool", ""))
        sch = schema_for(t)
        if not sch:
            continue
        parts = []
        for k, spec in sch.items():
            ty = spec[0]
            req = "" if len(spec) > 1 and spec[1] else "?"
            parts.append("%s:%s%s" % (k, ty, req))
        lines.append("- %s(%s)" % (t, ", ".join(parts)))
    return "\n".join(lines)


def _loads_tolerant(s: str):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", s))
    except Exception:  # noqa: BLE001
        return None


def parse_calls(text: str) -> dict:
    """Extract the last fenced tool block; tolerate trailing commas and a
    single object as well as an array. No block -> {"calls": []} (final)."""
    if not isinstance(text, str):
        return {"error": "text is not a string", "calls": []}
    blocks = _CALL_RE.findall(text)
    if not blocks:
        return {"calls": []}
    data = _loads_tolerant(blocks[-1].strip())
    if data is None:
        return {"error": "unparseable tool JSON", "calls": []}
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return {"error": "tool block must be an object or array", "calls": []}
    calls = []
    for c in data:
        if isinstance(c, dict) and "tool" in c:
            args = c.get("args")
            calls.append({"tool": str(c.get("tool", "")),
                          "args": args if isinstance(args, dict) else {}})
        else:
            return {"error": "tool call missing 'tool'", "calls": []}
    return {"calls": calls}


def validate_call(call: dict, known: set[str] | None = None) -> dict:
    """Check a call against its declared schema. Unknown tools pass (MCP)."""
    tool = str(call.get("tool", ""))
    if known is not None and tool not in known:
        return {"ok": False, "errors": ["unknown tool: " + tool[:60]],
                "warnings": [], "call": call}
    sch = schema_for(tool)
    errors, warnings = [], []
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    for k, spec in sch.items():
        ty, req = spec[0], (len(spec) > 1 and spec[1])
        if req and k not in args:
            errors.append("missing required arg '%s'" % k)
    for k in args:
        if k not in sch:
            warnings.append("unknown arg '%s'" % k)
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "call": call}


def repair_call(call: dict) -> dict:
    """Coerce numeric strings, drop unknown keys, fill defaults."""
    tool = str(call.get("tool", ""))
    sch = schema_for(tool)
    args = dict(call.get("args") or {})
    repairs = []
    for k in list(args):
        if k not in sch:
            repairs.append("dropped unknown arg '%s'" % k)
            del args[k]
    for k, spec in sch.items():
        ty = spec[0]
        default = spec[2] if len(spec) > 2 else None
        if k not in args:
            if default is not None:
                args[k] = default
                repairs.append("filled default '%s'" % k)
            continue
        v = args[k]
        if ty == "number" and isinstance(v, str):
            try:
                args[k] = float(v) if "." in v else int(v)
                repairs.append("coerced '%s' to number" % k)
            except ValueError:
                pass
        elif ty == "string" and not isinstance(v, str):
            args[k] = str(v)
            repairs.append("coerced '%s' to string" % k)
    return {"call": {"tool": tool, "args": args}, "repairs": repairs}


def bounded_tail(text: str, max_chars: int = 1600) -> str:
    """Atomic-Agent-style bounded reasoning tail: keep the head and the tail."""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text if isinstance(text, str) else ""
    head = max_chars // 4
    tail = max_chars - head
    return text[:head] + "\n[... reasoning tail bounded ...]\n" + text[-tail:]


def plan_batch(calls: list[dict]) -> dict:
    """Parallel only when 2+ calls and every tool is read-only."""
    tools = [str(c.get("tool", "")) for c in calls]
    read_only = [t for t in tools if t in READ_ONLY]
    gated = [t for t in tools if t not in READ_ONLY]
    return {"parallel": len(calls) > 1 and not gated,
            "read_only": read_only, "gated": gated}


def execute_batch(calls: list[dict], runner, max_workers: int = 4) -> dict:
    """Run independent read-only calls concurrently, preserving order."""
    import time
    n = len(calls)
    workers = max(1, min(int(max_workers), n))
    t0 = time.perf_counter()
    if workers <= 1:
        outs = [runner(c) for c in calls]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(runner, calls))
    dt = time.perf_counter() - t0
    return {"outs": outs, "seconds": dt, "calls": n, "workers": workers}


def correction_message(errors: list[str], warnings: list[str]) -> str:
    bits = []
    if errors:
        bits.append("Problems: " + "; ".join(errors[:4]))
    if warnings:
        bits.append("Warnings: " + "; ".join(warnings[:4]))
    return ("Your tool block was not valid. " + " ".join(bits)
            + " Re-emit ONLY a JSON array of tool calls using the declared "
              "schemas (no prose).")
