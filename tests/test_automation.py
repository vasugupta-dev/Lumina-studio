"""Automation suite: operator mode, filesystem, custom MCP builder, custom
skills, agent tool dispatch, workflows — and one real browser form fill.

Everything here is phone-light: tiny files in /tmp/opencode, one short
headless Chromium page, one generated MCP server round-trip. Heavy browsing
and whole-machine scans are deliberately NOT part of the test suite.

Run:  python3 tests/test_automation.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

import sysops  # noqa: E402
import custom_mcp as CM  # noqa: E402

RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


SCRATCH = os.path.join(tempfile.gettempdir(), "opencode", "automation_test")


def clean_scratch():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. operator policy
# ---------------------------------------------------------------------------

def policy_case():
    m = sysops.mode({}, None, "127.0.0.1")
    check("policy: loopback can enable without token", m["can_enable"] is True)
    m = sysops.mode({}, None, "0.0.0.0")
    check("policy: LAN bind without token refused", m["can_enable"] is False)
    m = sysops.mode({}, "tok", "0.0.0.0")
    check("policy: LAN bind with token allowed", m["can_enable"] is True)
    cfg = {}
    r = sysops.set_mode(True, cfg, None, "0.0.0.0")
    check("policy: set_mode refuses LAN without token", "error" in r)
    r = sysops.set_mode(True, cfg, None, "127.0.0.1")
    check("policy: set_mode enables on loopback", r["mode"]["full_access"] is True)
    r = sysops.set_mode(False, cfg, None, "127.0.0.1")
    check("policy: set_mode disables", r["mode"]["full_access"] is False)


# ---------------------------------------------------------------------------
# 2. filesystem ops
# ---------------------------------------------------------------------------

def fs_case():
    clean_scratch()
    sub = os.path.join(SCRATCH, "sub")
    check("fs: mkdir", "made" in sysops.fs_mkdir(sub))
    check("fs: write", sysops.fs_write(os.path.join(sub, "a.txt"), "hello\n")["wrote"] == 6)
    sysops.fs_write(os.path.join(sub, "a.txt"), "more\n", append=True)
    r = sysops.fs_read(os.path.join(sub, "a.txt"))
    check("fs: append + read", r["text"] == "hello\nmore\n")
    r = sysops.fs_list(SCRATCH)
    check("fs: list finds sub", [i["name"] for i in r["items"]] == ["sub"])
    r = sysops.fs_search(SCRATCH, query="more")
    check("fs: content search", len(r["hits"]) == 1 and r["hits"][0]["name"] == "a.txt")
    r = sysops.fs_search(SCRATCH, glob="*.txt")
    check("fs: glob search", len(r["hits"]) == 1)
    r = sysops.fs_move(os.path.join(sub, "a.txt"), os.path.join(SCRATCH, "b.txt"))
    check("fs: move", os.path.exists(os.path.join(SCRATCH, "b.txt")))
    check("fs: stat size", sysops.fs_stat(os.path.join(SCRATCH, "b.txt"))["size"] == 11)
    check("fs: delete", "deleted" in sysops.fs_delete(os.path.join(SCRATCH, "b.txt")))
    # symlink cannot escape realpath accounting: the audit shows the resolved target
    link = os.path.join(SCRATCH, "link")
    try:
        os.symlink("/etc/hostname", link)
        r = sysops.fs_read(link)
        check("fs: symlink resolves to real target (realpath)",
              r.get("path") == "/etc/hostname" or "error" in r)
    except OSError:
        check("fs: symlink case skipped (no symlink support)", True)
    # audit recorded the ops
    check("fs: audit trail written", len(sysops.audit_tail(50)) >= 6)


# ---------------------------------------------------------------------------
# 3. custom MCP: generate + real JSON-RPC round-trip
# ---------------------------------------------------------------------------

def custom_mcp_case():
    sid = "test-mcp-%d" % os.getpid()
    r = CM.save_custom({
        "id": sid, "name": "Test MCP", "desc": "suite",
        "tools": [CM.ECHO_TOOL,
                  {"name": "clock", "description": "now",
                   "handler": {"kind": "datetime"}}]})
    check("mcp: generated file saved", "saved" in r and os.path.exists(r["file"]))
    src = open(r["file"]).read()
    check("mcp: file is valid python with DEF", "DEF = {" in src and "tools/call" in src)
    import mcp_registry as MR
    ent = next(e for e in CM.entries() if e["id"] == "custom/" + sid)
    st = MR.MANAGER.start(ent)
    check("mcp: server starts as a subprocess", "started" in st)
    init = MR.MANAGER._rpc(ent["id"], "initialize", {"protocolVersion": "2024-11-05"})
    check("mcp: initialize handshake", init.get("serverInfo", {}).get("name") == sid)
    tools = MR.MANAGER.list_tools(ent["id"])
    check("mcp: tools/list returns both tools",
          sorted(t["name"] for t in tools.get("tools", [])) == ["clock", "echo"])
    call = MR.MANAGER.call(ent["id"], "echo", {"text": "ping"})
    txt = call.get("content", [{}])[0].get("text", "")
    check("mcp: tools/call echo round-trip", '"ping"' in txt)
    now = MR.MANAGER.call(ent["id"], "clock", {})
    check("mcp: datetime handler returns a clock string",
          len(now.get("content", [{}])[0].get("text", "")) > 10)
    MR.MANAGER.stop(ent["id"])
    check("mcp: stop frees the process", MR.MANAGER.status()["running"] == [])
    d = CM.delete_custom(sid)
    check("mcp: delete removes index + file", "deleted" in d
          and not os.path.exists(CM.server_path(sid)))
    bad = CM.save_custom({"id": "BAD ID!", "tools": [CM.ECHO_TOOL]})
    check("mcp: invalid id rejected", "error" in bad)
    bad = CM.save_custom({"id": "ok-id", "tools": [{"name": "x",
                                                    "handler": {"kind": "command"}}]})
    check("mcp: handler missing required field rejected", "error" in bad)
    bad = CM.save_custom({"id": "ok-id", "tools": [{"name": "x",
                                                    "handler": {"kind": "nope"}}]})
    check("mcp: unknown handler kind rejected", "error" in bad)


# ---------------------------------------------------------------------------
# 4. serve-level: skills CRUD, agent dispatch, workflow, gates
# ---------------------------------------------------------------------------

def serve_case():
    import serve
    # skills CRUD through the real API functions
    r = serve.api_skills_save({"id": "test-skill-%d" % os.getpid(),
                               "desc": "suite skill",
                               "prompt": "do it: {input}"})
    check("serve: user skill saved", "saved" in r)
    lst = serve.api_skills_list()
    check("serve: list merges builtin + user",
          lst["user"] == 1 and lst["builtin"] == len(serve.SKILLS))
    check("serve: builtin id protected",
          "error" in serve.api_skills_save({"id": "summarize", "prompt": "x {input}"}))
    check("serve: delete user skill", "deleted" in serve.api_skills_delete(
        {"id": "test-skill-%d" % os.getpid()}))

    # gates: operator OFF refuses fs/browser, allowlist still works
    serve.CFG["full_access"] = False
    check("serve: fs refused with operator off", "error" in serve.api_fs({"op": "list", "path": "~"}))
    check("serve: exec allowlist active with operator off",
          "error" in serve.api_exec("env"))
    check("serve: allowlisted exec still works", "out" in serve.api_exec("echo hi"))

    # agent loop with a scripted model: tool call -> final answer
    clean_scratch()
    target = os.path.join(SCRATCH, "agent_wrote.txt")
    serve.CFG["full_access"] = True
    calls = {"n": 0}
    real_chat = serve.api_chat

    def fake_chat(messages, max_tokens, temperature, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": '```tool\n{"tool": "fs_write", "args": '
                            '{"path": "%s", "text": "agent wrote this"}}\n```' % target}
        if calls["n"] == 2:
            return {"text": '```tool\n{"tool": "exec", "args": {"cmd": "uname -s"}}\n```'}
        return {"text": "all done", "tokens_per_second": 1.0}
    serve.api_chat = fake_chat
    try:
        r = serve.api_agent({"messages": [{"role": "user", "content": "go"}],
                             "max_steps": 4})
    finally:
        serve.api_chat = real_chat
    kinds = [s.get("type") for s in r["steps"]]
    check("agent: tool -> tool -> final sequence",
          kinds[:3] == ["tool", "tool", "final"])
    check("agent: fs_write tool wrote the file",
          os.path.exists(target) and open(target).read() == "agent wrote this")
    check("agent: operator exec ran unrestricted (uname)",
          "Linux" in str(r["steps"][1].get("result", "")))
    # operator OFF: same exec path falls back to the allowlist
    serve.CFG["full_access"] = False
    r3 = serve._agent_tool_call({"tool": "exec", "args": {"cmd": "env"}})
    check("agent: operator-off exec of non-allowlisted binary refused",
          "error" in r3 and "allowlist" in str(r3["error"]))

    # unknown tool -> clean error, never a crash
    check("agent: unknown tool refused", "error" in serve._agent_tool_call({"tool": "nope"}))

    # workflow: with operator OFF an fs step stops the run cleanly
    wf_off = serve.api_workflow_run({"steps": [
        {"kind": "fs", "id": "w", "op": "write",
         "path": os.path.join(SCRATCH, "nope.txt"), "text": "x"},
        {"kind": "echo", "text": "unreachable"}]})
    check("workflow: fs step refused with operator off, run stops",
          wf_off["ok"] is False and wf_off["stopped_at"] == "w")

    # workflow: fs + echo + branch, with vars resolution (operator ON)
    serve.CFG["full_access"] = True
    wf = serve.api_workflow_run({"steps": [
        {"kind": "fs", "id": "w", "op": "write",
         "path": os.path.join(SCRATCH, "wf.txt"), "text": "wf-hello"},
        {"kind": "fs", "id": "r", "op": "read", "path": os.path.join(SCRATCH, "wf.txt")},
        {"kind": "branch", "id": "b", "cond": "true",
         "then": [{"kind": "echo", "text": "took ${r.text}"}]},
        {"kind": "echo", "text": "after"}]})
    check("workflow: fs steps ok", wf["trace"][0]["status"] == "ok"
          and wf["trace"][1]["status"] == "ok")
    check("workflow: vars carry step output forward",
          "wf-hello" in str(wf["vars"].get("r")))
    wf2 = serve.api_workflow_run({"steps": [
        {"kind": "exec", "id": "bad", "cmd": "definitely_not_a_real_cmd_zz",
         "retry": 1},
        {"kind": "echo", "text": "unreachable"}]})
    check("workflow: hard error stops the run", wf2["ok"] is False
          and wf2["stopped_at"] == "bad")
    check("workflow: retry attempted twice",
          len([t for t in wf2["trace"] if t.get("step") == "bad"]) == 2)

    # custom MCP through the serve API (list + registered entries)
    r = serve.api_custom_mcp({"op": "list"})
    check("serve: custom MCP list works", "servers" in r and "example" in r)


# ---------------------------------------------------------------------------
# 5. browser driver: one real Chromium, one real form fill
# ---------------------------------------------------------------------------

def browser_case():
    try:
        import browser_driver as B
    except Exception as e:
        check("browser: driver importable (%s)" % str(e)[:60], False)
        return
    clean_scratch()
    page = os.path.join(SCRATCH, "form.html")
    with open(page, "w") as f:
        f.write("""<!doctype html><html><title>t</title><body>
<form onsubmit="event.preventDefault();document.getElementById('o').textContent=
 'got:'+document.getElementById('u').value">
<input id="u"><button id="b" type="submit">go</button></form>
<p id="o">none</p></body></html>""")
    r = B.execute("navigate", {"url": "file://" + page})
    check("browser: navigate local page", r.get("title") == "t")
    B.execute("fill", {"selector": "#u", "value": "lumina-user"})
    B.execute("type_human", {"selector": "#u", "value": "!", "delay_ms": 10})
    B.execute("click", {"selector": "#b", "settle_ms": 200})
    txt = B.execute("text", {}).get("text", "")
    check("browser: form submitted with typed value", "got:lumina-user!" in txt)
    shot = B.execute("screenshot", {})
    check("browser: screenshot captured", shot.get("bytes", 0) > 1000
          and os.path.exists(shot.get("path", "")))
    check("browser: close releases chromium", B.close().get("closed") is True)


def main() -> int:
    policy_case()
    fs_case()
    custom_mcp_case()
    serve_case()
    browser_case()
    n = len(RESULTS)
    bad = [x for x in RESULTS if not x[1]]
    print("\nAUTOMATION: %d/%d passed" % (n - len(bad), n))
    print("RESULT: " + ("ALL PASS" if not bad else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
