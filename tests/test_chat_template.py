"""Chat templates: stdlib renderer, byte-identical to Jinja2.

Differential-tested against the real Jinja2 on Qwen2.5-0.5B's actual
tokenizer_config.json template (plain, system, tools, tool calls, tool
responses, no-generation-prompt). jinja2 is a TEST dependency only — the
server stays stdlib-only. Any unsupported construct raises TemplateError
instead of rendering wrong text.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lumina_framework import chat_template as CT  # noqa: E402

try:
    import jinja2  # test-only oracle
except ImportError:
    jinja2 = None

QWEN_TC = "/tmp/Qwen2.5-0.5B/tokenizer_config.json"
RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + extra[:160]) if not cond and extra else ""))


def ensure_qwen_tc() -> bool:
    """Restore the real Qwen2.5-0.5B tokenizer_config.json from the local HF
    cache if /tmp was cleared. The differential test is meaningless without
    the actual template, so a missing fixture is a SKIP, never a fake pass."""
    if os.path.exists(QWEN_TC):
        return True
    import glob
    import shutil
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--*--Qwen2.5-0.5B*/snapshots/*/tokenizer_config.json"))
    if not hits:
        return False
    os.makedirs(os.path.dirname(QWEN_TC), exist_ok=True)
    shutil.copy(hits[0], QWEN_TC)
    return True


def main() -> int:
    if not ensure_qwen_tc():
        print("SKIP: Qwen2.5-0.5B tokenizer_config.json unavailable (no /tmp "
              "copy, no HF cache) — real-template differential cannot run")
        return 3
    tpl = json.load(open(QWEN_TC))["chat_template"]
    cases = [
        ("plain", {"messages": [{"role": "user", "content": "hi"}]}, {}),
        ("sys+user+asst", {"messages": [{"role": "system", "content": "be nice"},
                                        {"role": "user", "content": "hi"},
                                        {"role": "assistant", "content": "yo"}]}, {}),
        ("tools", {"messages": [{"role": "user", "content": "run it"}],
                   "tools": [{"type": "function", "function": {"name": "exec",
                               "description": "run", "parameters": {"x": 1}}}]}, {}),
        ("tool_call", {"messages": [{"role": "user", "content": "run ls"},
                                    {"role": "assistant", "content": None,
                                     "tool_calls": [{"function": {"name": "exec",
                                                                  "arguments": '{"cmd": "ls"}'}}]},
                                    {"role": "tool", "content": "out"},
                                    {"role": "tool", "content": "out2"}]}, {}),
        ("no-gen", {"messages": [{"role": "user", "content": "hi"}]},
         {"add_generation_prompt": False}),
    ]
    if jinja2 is None:
        print("SKIP differential (jinja2 missing); structural checks only")
        for name, ctx, kw in cases:
            try:
                out = CT.render(tpl, dict(ctx, add_generation_prompt=kw.get("add_generation_prompt", True)))
                check(name + " renders", "<|im_start|>" in out)
            except Exception as e:
                check(name + " renders", False, str(e))
    else:
        J = jinja2.Template(tpl)
        for name, ctx, kw in cases:
            full = dict(ctx)
            full.setdefault("add_generation_prompt", True)
            full.update(kw)
            try:
                mine = CT.render(tpl, dict(full))
            except Exception as e:
                check("jinja-identical:" + name, False, "ours raised: " + str(e))
                continue
            check("jinja-identical:" + name, J.render(**full) == mine,
                  "jinja=" + repr(J.render(**full)[:120]) + " ours=" + repr(mine[:120]))
    # unsupported constructs must raise, never silently misrender
    for name, bad in [("unknown-filter", "{{ x | nosuchfilter }}"),
                      ("unknown-stmt", "{% macro m() %}x{% endmacro %}"),
                      ("unclosed", "{% if x %}A")]:
        try:
            CT.render(bad, {"x": 1})
            check("raises:" + name, False, "no error raised")
        except CT.TemplateError:
            check("raises:" + name, True)
        except Exception as e:
            check("raises:" + name, False, type(e).__name__ + str(e)[:80])
    n = sum(1 for _, c in RESULTS if c)
    print(f"\nCHAT-TEMPLATE: {n}/{len(RESULTS)} passed")
    print("RESULT:", "ALL PASS" if n == len(RESULTS) else "FAILURES PRESENT")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
