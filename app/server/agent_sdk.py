"""OpenAI Agents SDK bridge for Lumina Studio's agent layer (optional).

The server stays stdlib-only at runtime: `agents` is NEVER imported at module
load. It is imported once, lazily, either by warmup() (background thread) or
by the first SDK run, and every path degrades honestly when it is unavailable
(state "missing") or fails to import (state "error").

Two model routes, one tool dispatcher:

  * "cloud"  — any OpenAI-compatible endpoint with native function calling
    (OpenAIChatCompletionsModel + AsyncOpenAI). Tools are SDK function tools.
  * "local"  — Lumina's own loaded model through serve.api_chat, bridged with
    a duck-typed chat-completions client that speaks Lumina's existing fenced
    tool protocol (the same ```tool JSON blocks the builtin loop uses).

No tool ever runs here: everything funnels through the injected `tool_fn`
(serve._agent_tool_call), so operator gating, the path jail and the ops audit
log stay exactly where they already are.

Verified against openai-agents 0.24.0 / openai 3.29.0 on this device; the
module itself imports nothing outside the stdlib until the SDK is warmed up.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import threading
import time
import uuid

SDK_PACKAGE = "openai-agents"
DISABLE_ENV = "LUMINA_AGENTS_DISABLE"  # set to "1" to force "missing" (tests/air-gap)

# The one true fenced-tool pattern (serve.py's builtin loop uses this too).
FENCE_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.S)

_STATE = {"state": "cold", "version": None, "import_s": None,
          "error": None, "tried_at": None}
_LOCK = threading.Lock()
_MODULES: dict = {}


# ---------------------------------------------------------------------------
# availability / warmup
# ---------------------------------------------------------------------------

def _find_spec() -> bool:
    try:
        return importlib.util.find_spec("agents") is not None
    except Exception:
        return False


def status(refresh: bool = False) -> dict:
    """Report SDK state. Never imports `agents` unless refresh=True."""
    if refresh:
        _import_sdk()
    with _LOCK:
        st = dict(_STATE)
    st["package"] = SDK_PACKAGE
    st["disabled"] = os.environ.get(DISABLE_ENV) == "1"
    st["installed"] = _find_spec()
    if st["state"] == "cold" and st["disabled"]:
        st["state"] = "missing"
        st["error"] = "disabled via " + DISABLE_ENV
    st["ready"] = st["state"] == "ready"
    st["hint"] = "" if st["ready"] else (
        "pip install openai-agents" if not st["installed"] else
        "POST /api/agent/sdk/warmup" if st["state"] in ("cold", "importing") else "")
    return st


def warmup(background: bool = True) -> dict:
    """Import the SDK once. background=True never blocks the caller."""
    st = status()
    if st["state"] in ("ready", "missing"):
        return st
    if background:
        with _LOCK:
            if _STATE["state"] == "cold":
                _STATE["state"] = "importing"
                threading.Thread(target=_import_sdk, name="agents-warmup",
                                 daemon=True).start()
        return status()
    return _import_sdk()


def _import_sdk() -> dict:
    global _MODULES
    with _LOCK:
        if _STATE["state"] == "ready":
            return dict(_STATE)
        _STATE["state"] = "importing"
        _STATE["tried_at"] = time.time()
    t0 = time.perf_counter()
    try:
        if os.environ.get(DISABLE_ENV) == "1":
            raise ImportError("disabled via " + DISABLE_ENV)
        if not _find_spec():
            raise ImportError(SDK_PACKAGE + " is not installed")
        import agents as _agents
        from agents import (Agent, FunctionTool, OpenAIChatCompletionsModel,
                            RunConfig, Runner, set_tracing_disabled)
        from agents.mcp import MCPServerStdio
        set_tracing_disabled(True)  # never phone home
        mods = {"agents": _agents, "Agent": Agent, "FunctionTool": FunctionTool,
                "OpenAIChatCompletionsModel": OpenAIChatCompletionsModel,
                "RunConfig": RunConfig, "Runner": Runner,
                "MCPServerStdio": MCPServerStdio}
        with _LOCK:
            _MODULES = mods
            _STATE.update(state="ready", error=None,
                          version=str(getattr(_agents, "__version__", "?")),
                          import_s=round(time.perf_counter() - t0, 2))
    except Exception as e:
        with _LOCK:
            _MODULES = {}
            _STATE.update(state="missing" if isinstance(e, ImportError) else "error",
                          version=None, import_s=round(time.perf_counter() - t0, 2),
                          error=f"{type(e).__name__}: {e}")
    return status()


def _require() -> dict:
    with _LOCK:
        mods = dict(_MODULES)
    if not mods:
        _import_sdk()
        with _LOCK:
            mods = dict(_MODULES)
    if not mods:
        raise RuntimeError("openai-agents unavailable: "
                           + str(_STATE.get("error") or _STATE.get("state")))
    return mods


# ---------------------------------------------------------------------------
# fenced-tool protocol (same wire format as the builtin loop)
# ---------------------------------------------------------------------------

def fence_parse(text: str) -> tuple[list[dict], str]:
    """Parse ALL ```tool fences into OpenAI tool_calls wire dicts.

    Returns (tool_calls, remaining_text). Raises ValueError for malformed
    fences — callers decide whether that is an error or plain text.
    """
    calls = []
    for m in FENCE_RE.finditer(text or ""):
        d = json.loads(m.group(1))  # JSONDecodeError is a ValueError
        if not isinstance(d, dict) or not d.get("tool"):
            raise ValueError('tool fence must be {"tool": ..., "args": {...}}')
        calls.append({"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                      "function": {"name": str(d["tool"]),
                                   "arguments": json.dumps(d.get("args") or {},
                                                           ensure_ascii=False)}})
    return calls, FENCE_RE.sub("", text or "").strip()


def fence_render(tool: str, args: dict | None = None) -> str:
    """Render one call back into the fence form the local model emits."""
    return ("```tool\n"
            + json.dumps({"tool": str(tool), "args": args or {}}, ensure_ascii=False)
            + "\n```")


def tool_result_text(out, tool: str) -> str:
    """Result formatting identical to the builtin loop's feedback text."""
    if isinstance(out, dict) and out.get("error"):
        result = "ERROR: " + str(out["error"])
    elif isinstance(out, dict):
        result = str(out.get("out") or out.get("text") or out.get("data_url") or out)
    else:
        result = str(out)
    if str(tool).startswith("browser_screenshot"):
        result = "screenshot saved: " + str((out or {}).get("url", "")
                                            if isinstance(out, dict) else "")
    return result[:4000]


def sanitize_tool_name(name: str) -> str:
    """SDK tool names must be simple identifiers; 'mcp:a::b' -> 'mcp_a_b'."""
    s = re.sub(r"[^0-9a-zA-Z_-]+", "_", str(name)).strip("_")
    return s or "tool"


# ---------------------------------------------------------------------------
# tools + MCP mirrors
# ---------------------------------------------------------------------------

def build_tools(tool_specs, tool_fn):
    """FunctionTools wrapping the injected dispatcher (one per spec).

    Returns (tools, name_mapping). Sanitized names map back to the original
    tool ids so `mcp:server::tool` specs still dispatch through the normal
    operator-gated path.
    """
    mods = _require()
    tools, seen, mapping = [], set(), {}
    for s in tool_specs or []:
        orig = str(s.get("tool", ""))
        if not orig:
            continue
        name = base = sanitize_tool_name(orig)
        i = 2
        while name in seen:  # collision policy: our side de-dupes first
            name = f"{base}_{i}"
            i += 1
        seen.add(name)
        mapping[name] = orig

        async def _invoke(ctx, args_json: str, _orig=orig) -> str:
            try:
                args = json.loads(args_json or "{}")
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {"value": args}
            return tool_result_text(tool_fn({"tool": _orig, "args": args}), _orig)

        tools.append(mods["FunctionTool"](
            name=name,
            description=(str(s.get("description") or "") or f"Lumina tool: {orig}")[:1000],
            params_json_schema={"type": "object", "additionalProperties": True},
            on_invoke_tool=_invoke,
            strict_json_schema=False,
        ))
    return tools, mapping


def build_mcp_servers(entries):
    """Native MCPServerStdio mirrors of mcp_registry entries.

    Returns (servers, skipped_reasons). Servers must be entered as async
    context managers (run_sync does that).
    """
    mods = _require()
    servers, skipped = [], []
    for e in entries or []:
        eid = str(e.get("id") or "mcp")
        cmd = str(e.get("command") or "")
        if not cmd:
            skipped.append(eid + ": no stdio command configured")
            continue
        params = {"command": cmd}
        if e.get("args"):
            params["args"] = [str(a) for a in e["args"]]
        if isinstance(e.get("env"), dict) and e["env"]:
            params["env"] = {str(k): str(v) for k, v in e["env"].items()}
        servers.append(mods["MCPServerStdio"](params=params,
                                              cache_tools_list=True, name=eid))
    return servers, skipped


# ---------------------------------------------------------------------------
# model routes
# ---------------------------------------------------------------------------

def model_target(cloud_cfg: dict | None, local_model_id: str | None) -> dict:
    """Pick the run target. Local-first (privacy-first), cloud second."""
    c = cloud_cfg or {}
    if local_model_id:
        return {"kind": "local", "model_id": str(local_model_id),
                "reason": "local model loaded: " + str(local_model_id)[:80]}
    if c.get("enabled") and str(c.get("model") or "").strip() \
            and str(c.get("base_url") or "").strip():
        return {"kind": "cloud", "base_url": str(c["base_url"]),
                "model": str(c["model"]),
                "api_key": str(c.get("api_key") or "sk-local"),
                "reason": "cloud model configured: " + str(c["model"])[:60]}
    return {"kind": "none",
            "reason": "no local model loaded and no cloud model configured"}


def _to_lumina_messages(msgs) -> list[dict]:
    """OpenAI wire messages -> Lumina chat messages, re-rendering tool turns
    into the fenced protocol the local model already understands."""
    out = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(str(p.get("text", "")) for p in content
                              if isinstance(p, dict))
        content = str(content or "")
        if role in ("system", "developer"):
            if content:
                out.append({"role": "system", "content": content})
            continue
        if role == "assistant":
            rendered = []
            for tc in (m.get("tool_calls") or []):
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                rendered.append(fence_render(fn.get("name") or "tool",
                                             args if isinstance(args, dict) else {}))
            text = ((content + "\n") if content else "") + "\n".join(rendered)
            if text.strip():
                out.append({"role": "assistant", "content": text.strip()})
            continue
        if role == "tool":
            out.append({"role": "user",
                        "content": "Tool result:\n" + content[:4000]
                        + "\nContinue. If you have the answer, reply "
                          "without a tool block."})
            continue
        out.append({"role": "user", "content": content})
    return out or [{"role": "user", "content": ""}]


def lumina_chat_client(chat_fn, model_id: str = "local",
                       base_url: str = "http://127.0.0.1/v1",
                       default_max_tokens: int = 256,
                       default_temperature: float = 0.0):
    """Duck-typed AsyncOpenAI: chat.completions.create -> Lumina api_chat.

    Only the attributes OpenAIChatCompletionsModel reads are provided
    (base_url + chat.completions.create). The fenced protocol is translated
    both ways so the SDK's tool dispatch keeps working on local models.
    """
    from openai.types.chat import ChatCompletion

    class _Completions:
        async def create(self, **kw):
            msgs = _to_lumina_messages(kw.get("messages") or [])
            # The SDK may send Omit sentinels for unset params — coerce safely.
            try:
                mt = int(kw.get("max_tokens"))
            except (TypeError, ValueError):
                mt = int(default_max_tokens)
            try:
                temp = float(kw.get("temperature"))
            except (TypeError, ValueError):
                temp = float(default_temperature)
            r = await asyncio.to_thread(chat_fn, msgs, mt, temp)
            if not isinstance(r, dict):
                raise RuntimeError("local model returned no result")
            if r.get("error"):
                raise RuntimeError("local model error: " + str(r["error"]))
            text = str(r.get("text") or "")
            try:
                calls, rest = fence_parse(text)
            except ValueError:
                calls, rest = [], text  # malformed fence: show it, don't crash
            return ChatCompletion.model_validate({
                "id": "chatcmpl-lumina",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": str(model_id),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": rest or None,
                                "tool_calls": calls or None},
                    "finish_reason": "tool_calls" if calls else "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            })

    class _Chat:
        completions = _Completions()

    class _Client:
        pass

    c = _Client()
    c.base_url = base_url
    c.chat = _Chat()
    return c


def build_instructions(base: str, tool_specs, target_kind: str) -> str:
    parts = [(base or "").strip()
             or "You are Lumina's agent, running on the user's machine."]
    if target_kind == "local":
        parts.append("Tools are executed by the runtime. To call one, reply "
                     "with ONLY a fenced block:\n```tool\n"
                     '{"tool": "<name>", "args": {...}}\n```\nAvailable tools:')
        for s in (tool_specs or [])[:60]:
            d = str(s.get("description") or "").strip()[:110]
            parts.append("- " + str(s.get("tool")) + (": " + d if d else ""))
        parts.append("After each tool result: call another tool, or give the "
                     "final answer with no tool block.")
    return "\n".join(parts)[:4000]


def _to_sdk_input(messages) -> list[dict]:
    inp = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        t = str(c or "")
        if t:
            inp.append({"role": role, "content": t})
    return inp or [{"role": "user", "content": ""}]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_sync(*, target: dict, tool_specs: list, tool_fn, chat_fn=None,
             messages=None, instructions_base: str = "", max_steps: int = 3,
             max_tokens: int = 256, temperature: float = 0.0,
             mcp_entries=None) -> dict:
    """Run one SDK agent turn. Blocking; safe from a server handler thread.

    Never raises: returns {"fallback": True, ...} when the bridge cannot run
    (caller decides), or {"error": ..., "steps": [...]} for run failures.
    """
    st = status()
    if st["state"] != "ready":
        return {"engine": "sdk", "fallback": True, "sdk": st,
                "engine_reason": "OpenAI Agents SDK not ready: " + str(st["state"]),
                "error": "OpenAI Agents SDK not ready (" + str(st["state"]) + ")"}
    if not target or target.get("kind") == "none":
        msg = str((target or {}).get("reason") or "no model target")
        return {"engine": "sdk", "fallback": True, "sdk": st,
                "engine_reason": msg, "error": msg}
    try:
        return asyncio.run(_run_async(
            target=target, tool_specs=tool_specs, tool_fn=tool_fn,
            chat_fn=chat_fn, messages=messages, instructions_base=instructions_base,
            max_steps=max(1, min(int(max_steps or 3), 5)), max_tokens=int(max_tokens),
            temperature=float(temperature), mcp_entries=mcp_entries))
    except Exception as e:  # ModelBehaviorError, MaxTurnsExceeded, transport...
        return {"engine": "sdk", "fallback": False, "steps": [],
                "error": f"{type(e).__name__}: {e}"}


async def _run_async(*, target, tool_specs, tool_fn, chat_fn, messages,
                     instructions_base, max_steps, max_tokens, temperature,
                     mcp_entries):
    import contextlib

    mods = _require()
    tools, name_map = build_tools(tool_specs, tool_fn)
    servers, skipped = build_mcp_servers(mcp_entries or [])
    client = None
    if target.get("kind") == "cloud":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=str(target.get("base_url")),
                             api_key=str(target.get("api_key") or "sk-local"),
                             max_retries=0, timeout=120)
        model = mods["OpenAIChatCompletionsModel"](
            model=str(target.get("model") or "gpt-4o-mini"), openai_client=client)
    else:
        lc = lumina_chat_client(chat_fn, str(target.get("model_id") or "local"),
                                default_max_tokens=max_tokens,
                                default_temperature=temperature)
        model = mods["OpenAIChatCompletionsModel"](
            model=str(target.get("model_id") or "lumina-local"), openai_client=lc)
    try:
        run_cfg = mods["RunConfig"](tracing_disabled=True, workflow_name="lumina-agent",
                                    tool_name_collision_policy="error")
    except TypeError:  # older/newer signature: minimal is always valid
        run_cfg = mods["RunConfig"](tracing_disabled=True)
    agent_kw = {}
    if servers:
        agent_kw["mcp_servers"] = servers
    agent = mods["Agent"](name="lumina",
                          instructions=build_instructions(instructions_base,
                                                          tool_specs,
                                                          str(target.get("kind"))),
                          model=model, tools=tools, **agent_kw)
    try:
        async with contextlib.AsyncExitStack() as stack:
            for s in servers:
                await stack.enter_async_context(s)
            result = await mods["Runner"].run(agent, _to_sdk_input(messages),
                                              max_turns=max_steps, run_config=run_cfg)
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

    steps = []
    try:
        for item in list(getattr(result, "new_items", []) or []):
            nm = type(item).__name__
            if "ToolCallOutput" in nm:
                txt = str(getattr(item, "output", ""))
                for rec in reversed(steps):
                    if rec.get("type") == "tool" and "result" not in rec:
                        rec["result"] = txt[:600]
                        break
                else:
                    steps.append({"type": "tool", "call": {"tool": "?"},
                                  "result": txt[:600]})
            elif "ToolCall" in nm:
                raw = getattr(item, "raw_item", None)
                name = getattr(raw, "name", None)
                if name is None and isinstance(raw, dict):
                    name = raw.get("name")
                args = getattr(raw, "arguments", None)
                if args is None and isinstance(raw, dict):
                    args = raw.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                steps.append({"type": "tool",
                              "call": {"tool": name_map.get(str(name), str(name)),
                                       "args": args if isinstance(args, dict) else {}},
                              "external": str(name) not in name_map})
    except Exception:
        pass
    final_text = ""
    try:
        final_text = str(result.final_output or "")
    except Exception:
        pass
    if final_text:
        steps.append({"type": "final", "text": final_text[:8000]})
    out = {"engine": "sdk", "fallback": False, "text": final_text, "steps": steps,
           "sdk_version": status().get("version"),
           "model_target": {k: target.get(k) for k in ("kind", "model", "model_id")}}
    usage = None
    try:
        u = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if u is not None:
            usage = {"requests": getattr(u, "requests", None),
                     "input_tokens": getattr(u, "input_tokens", None),
                     "output_tokens": getattr(u, "output_tokens", None)}
    except Exception:
        pass
    if usage:
        out["usage"] = usage
    if skipped:
        out["mcp_skipped"] = skipped
    return out
