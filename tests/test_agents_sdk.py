"""OpenAI Agents SDK bridge (app/server/agent_sdk.py).

Phone-light by design: no model is loaded, nothing heavier than loopback HTTP
plus one generated stdlib MCP server subprocess. Covers:

  * status/warmup state machine (status must never import `agents`)
  * the fenced-tool codec (parse/render/result-text parity)
  * Route A — cloud target against a deterministic stub OpenAI endpoint
  * Route B — local target against a scripted chat_fn, incl. malformed fences
  * native MCP mirror — a Lumina-generated stdio server driven by the SDK
  * dispatcher semantics — explicit "sdk" never silently falls back
  * builtin-loop regression — the refactor kept the old behavior intact

Exit codes: 0 = pass, 1 = fail, 3 = honest skip (openai-agents missing /
LUMINA_AGENTS_DISABLE=1).

Run:  python3 tests/test_agents_sdk.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

import agent_sdk  # noqa: E402
import custom_mcp as CM  # noqa: E402

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:160]) if not cond and extra else ""))


# ---------------------------------------------------------------------------
# deterministic stub OpenAI-compatible endpoint (Route A)
# ---------------------------------------------------------------------------

class _Stub:
    def __init__(self, tool_name="lum_whoami", tool_args="{}"):
        self.calls = []
        self.tool_name = tool_name
        self.tool_args = tool_args
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                b = json.loads(self.rfile.read(n) or b"{}")
                outer.calls.append({
                    "tools": [t.get("function", {}).get("name")
                              for t in (b.get("tools") or [])],
                    "msgs": [(m.get("role"), str(m.get("content"))[:80])
                             for m in b.get("messages", [])]})
                has_result = any(m.get("role") == "tool"
                                 for m in b.get("messages", []))
                if b.get("tools") and not has_result:
                    msg = {"role": "assistant", "content": None,
                           "tool_calls": [{"id": "call_stub1", "type": "function",
                                           "function": {"name": outer.tool_name,
                                                        "arguments": outer.tool_args}}]}
                    fin = "tool_calls"
                else:
                    msg = {"role": "assistant", "content": "final answer: ok"}
                    fin = "stop"
                out = {"id": "chatcmpl-stub", "object": "chat.completion",
                       "created": 1, "model": b.get("model", "m"),
                       "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                                 "total_tokens": 10}}
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.port

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _cloud_target(stub):
    return {"kind": "cloud", "base_url": stub.base_url(), "model": "stub-model",
            "api_key": "test"}


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def status_case():
    st = agent_sdk.status()
    check("status: package + install reported",
          st["package"] == "openai-agents" and st["installed"] is True
          and st["disabled"] is False)
    check("status: cold state without importing",
          st["state"] == "cold" and st["version"] is None)
    check("status: hint points at warmup", "warmup" in str(st.get("hint", "")))


def fence_case():
    text = ('here\n```tool\n{"tool": "exec", "args": {"cmd": "ls"}}\n```\nend')
    calls, rest = agent_sdk.fence_parse(text)
    check("fence: one call parsed", len(calls) == 1
          and calls[0]["function"]["name"] == "exec"
          and json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls"},
          calls)
    check("fence: text stripped, ends kept",
          "```" not in rest and "here" in rest and "end" in rest, rest)
    rt = agent_sdk.fence_render("mcp:srv::echo", {"text": "hi"})
    c2, _ = agent_sdk.fence_parse(rt)
    check("fence: render -> parse roundtrip", len(c2) == 1
          and c2[0]["function"]["name"] == "mcp:srv::echo"
          and json.loads(c2[0]["function"]["arguments"]) == {"text": "hi"})
    try:
        agent_sdk.fence_parse('```tool\n{"tool": }\n```')
        bad = False
    except ValueError:
        bad = True
    check("fence: malformed raises ValueError", bad)
    check("fence: tool_result_text parity",
          agent_sdk.tool_result_text({"out": "x"}, "exec") == "x"
          and agent_sdk.tool_result_text({"error": "boom"}, "exec").startswith("ERROR: boom")
          and agent_sdk.tool_result_text({"url": "u"}, "browser_screenshot")
          == "screenshot saved: u")
    check("fence: mcp tool names sanitized",
          agent_sdk.sanitize_tool_name("mcp:a::b") == "mcp_a_b")


def warmup_case():
    r = agent_sdk.warmup(background=False)
    check("warmup: ready", r["state"] == "ready", r.get("error"))
    check("warmup: version recorded",
          str(r.get("version", "")).startswith("0.2"), r.get("version"))
    check("warmup: import time measured",
          isinstance(r.get("import_s"), float) and r["import_s"] > 0,
          r.get("import_s"))
    check("warmup: idempotent",
          agent_sdk.warmup(background=False)["state"] == "ready")


def stub_route_case():
    stub = _Stub()
    seen = []

    def tool_fn(call):
        seen.append(call)
        return {"out": "whoami:ok"}

    try:
        r = agent_sdk.run_sync(
            target=_cloud_target(stub),
            tool_specs=[{"tool": "lum_whoami", "description": "return whoami"}],
            tool_fn=tool_fn, messages=[{"role": "user", "content": "who am i"}],
            max_steps=3)
    finally:
        stub.close()
    check("route A: no fallback", r.get("fallback") is False, r.get("error"))
    check("route A: exactly two model calls", len(stub.calls) == 2,
          len(stub.calls))
    check("route A: tools offered to model",
          "lum_whoami" in (stub.calls[0]["tools"] or []) if stub.calls else False,
          stub.calls[:1])
    check("route A: tool dispatched through tool_fn",
          bool(seen) and seen[0]["tool"] == "lum_whoami", seen[:1])
    check("route A: tool result fed back",
          any(m[0] == "tool" for m in stub.calls[1]["msgs"]) if len(stub.calls) > 1 else False)
    check("route A: final text", r.get("text") == "final answer: ok", r.get("text"))
    steps = r.get("steps") or []
    check("route A: steps tool -> final",
          [s.get("type") for s in steps] == ["tool", "final"],
          [s.get("type") for s in steps])
    check("route A: tool step carries result",
          bool(steps) and "whoami:ok" in str(steps[0].get("result")), steps[:1])


def local_bridge_case():
    chat_calls = []
    seen = []

    def fake_chat(messages, max_tokens, temperature, **kw):
        chat_calls.append([(m.get("role"), str(m.get("content"))[:90]) for m in messages])
        if not any("Tool result:" in str(m.get("content")) for m in messages):
            return {"text": '```tool\n{"tool": "lum_echo", "args": {"text": "hi"}}\n```'}
        return {"text": "final from local model"}

    def tool_fn(call):
        seen.append(call)
        return {"out": "echo:" + str(call["args"].get("text", ""))}

    r = agent_sdk.run_sync(
        target={"kind": "local", "model_id": "fake-local"},
        tool_specs=[{"tool": "lum_echo"}], tool_fn=tool_fn, chat_fn=fake_chat,
        messages=[{"role": "user", "content": "echo hi"}], max_steps=3)
    check("route B: no fallback", r.get("fallback") is False, r.get("error"))
    check("route B: two local model calls", len(chat_calls) == 2, len(chat_calls))
    check("route B: tool dispatched via tool_fn",
          bool(seen) and seen[0]["tool"] == "lum_echo", seen[:1])
    check("route B: tool result fed back as a message",
          len(chat_calls) > 1 and any("Tool result:" in c for _, c in chat_calls[1]),
          chat_calls[1:] )
    check("route B: assistant turn re-rendered with the fence",
          len(chat_calls) > 1 and any(r_ == "assistant" and "```tool" in c
                                      for r_, c in chat_calls[1]))
    check("route B: final text", r.get("text") == "final from local model",
          r.get("text"))
    check("route B: steps end with final",
          [s.get("type") for s in (r.get("steps") or [])][-1:] == ["final"],
          r.get("steps"))

    def bad_chat(messages, max_tokens, temperature, **kw):
        return {"text": "```tool\n{oops\n```"}

    r2 = agent_sdk.run_sync(target={"kind": "local", "model_id": "fake-local"},
                            tool_specs=[], tool_fn=lambda c: {}, chat_fn=bad_chat,
                            messages=[{"role": "user", "content": "x"}], max_steps=2)
    check("route B: malformed fence surfaces as text, no crash",
          r2.get("fallback") is False and "oops" in str(r2.get("text", "")),
          r2)


def mcp_case():
    sid = "test-sdk-mcp-%d" % os.getpid()
    saved = CM.save_custom({"id": sid, "name": "sdk bridge test",
                            "tools": [{"name": "echo",
                                       "description": "echo args back",
                                       "handler": {"kind": "echo"}}]})
    check("mcp: custom stdio server generated", "saved" in saved, saved)
    entries = [e for e in CM.entries() if e["id"] == "custom/" + sid]
    check("mcp: registry entry present", len(entries) == 1)
    stub = _Stub(tool_name="echo", tool_args='{"text": "ping"}')
    try:
        r = agent_sdk.run_sync(target=_cloud_target(stub), tool_specs=[],
                               tool_fn=lambda c: {"out": "unused"},
                               messages=[{"role": "user", "content": "call echo"}],
                               max_steps=3, mcp_entries=entries)
    finally:
        stub.close()
        CM.delete_custom(sid)
    check("mcp: no fallback", r.get("fallback") is False, r.get("error"))
    check("mcp: native tool offered to the model",
          bool(stub.calls) and "echo" in (stub.calls[0]["tools"] or []),
          stub.calls[:1])
    check("mcp: tool result fed back contains ping",
          len(stub.calls) > 1 and any("ping" in c for m, c in stub.calls[1]["msgs"]
                                      if m == "tool"),
          stub.calls[1:] if len(stub.calls) > 1 else [])
    check("mcp: run completed", r.get("text") == "final answer: ok", r.get("text"))


def serve_regression_case():
    import serve
    scratch = os.path.join(tempfile.gettempdir(), "opencode", "agentsdk_test")
    os.makedirs(scratch, exist_ok=True)
    target = os.path.join(scratch, "agent_wrote.txt")
    serve.CFG["full_access"] = True
    calls = {"n": 0}
    real_chat = serve.api_chat

    def fake_chat(messages, max_tokens, temperature, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": '```tool\n{"tool": "fs_write", "args": '
                            '{"path": "%s", "text": "sdk suite"}}\n```' % target}
        return {"text": "done"}

    serve.api_chat = fake_chat
    try:
        r = serve.api_agent({"messages": [{"role": "user", "content": "go"}],
                             "max_steps": 3})
    finally:
        serve.api_chat = real_chat
        serve.CFG["full_access"] = False
    check("builtin regression: engine label", r.get("engine") == "builtin", r)
    check("builtin regression: tool -> final",
          [s.get("type") for s in r["steps"]] == ["tool", "final"],
          [s.get("type") for s in r["steps"]])
    check("builtin regression: fs_write tool wrote the file",
          os.path.exists(target) and open(target).read() == "sdk suite")


def fallback_case():
    os.environ["LUMINA_AGENTS_DISABLE"] = "1"
    try:
        importlib.reload(agent_sdk)
        st = agent_sdk.status()
        check("fallback: disabled reported as missing",
              st["state"] == "missing" and st["disabled"] is True, st)
        r = agent_sdk.run_sync(target={"kind": "local", "model_id": "x"},
                               tool_specs=[], tool_fn=lambda c: {})
        check("fallback: run_sync refuses with fallback flag",
              r.get("fallback") is True and "not ready" in str(r.get("error")), r)
        import serve
        r2 = serve.api_agent({"engine": "sdk",
                              "messages": [{"role": "user", "content": "x"}]})
        check("dispatcher: explicit sdk never silently falls back",
              r2.get("engine") == "sdk" and r2.get("fallback") is True
              and "error" in r2, r2)
        r3 = serve.api_agent({"engine": "auto",
                              "messages": [{"role": "user", "content": "x"}]})
        check("dispatcher: auto falls back to builtin with a reason",
              r3.get("engine") == "builtin" and bool(r3.get("engine_reason")), r3)
    finally:
        os.environ.pop("LUMINA_AGENTS_DISABLE", None)
        importlib.reload(agent_sdk)


def timing_guard():
    import serve
    check("guard: suite loaded no model", serve.CHAT["model"] is None)


def main() -> int:
    t0 = time.time()
    if os.environ.get("LUMINA_AGENTS_DISABLE") == "1":
        print("SKIP: LUMINA_AGENTS_DISABLE=1 (air-gap mode)")
        return 3
    if not agent_sdk.status()["installed"]:
        print("SKIP: openai-agents not installed (pip install openai-agents)")
        return 3
    status_case()
    fence_case()
    warmup_case()
    stub_route_case()
    local_bridge_case()
    mcp_case()
    serve_regression_case()
    fallback_case()
    timing_guard()
    ok = sum(1 for _, c in RESULTS if c)
    print("\nAGENTS-SDK: %d/%d passed (%.1fs)"
          % (ok, len(RESULTS), time.time() - t0))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
