"""Run the full Lumina Framework reference suite.

  python3 tests/run_all.py

Prints a per-file summary and a final verdict. Every check is either PASS or FAIL;
"skipped" is reported honestly.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

SUITES = [
    ("framework invariants", "test_runtime.py"),
    ("cross-format equivalence", "test_cross_format.py"),
    ("format-codec frontier", "test_codec_frontier.py"),
    ("format-framework integration", "test_format_framework.py"),
    ("block-GPTQ parity", "test_gptq_block_parity.py"),
    ("chat fast-path parity", "test_chat_fast_parity.py"),
    ("torch MoE parity", "test_moe_torch_parity.py"),
    ("qwen4-exp kernels", "test_qwen_next.py"),
    ("chat templates", "test_chat_template.py"),
    ("prefix KV cache", "test_prefix_cache.py"),
    ("automation (ops/browser/mcp/skills)", "test_automation.py"),
    ("OpenAI Agents SDK bridge", "test_agents_sdk.py"),
    ("context compaction + prefix cache", "test_context_compact.py"),
    ("evidence grounding (claim checks)", "test_grounding.py"),
    ("small-model boost toolbox", "test_boost.py"),
    ("skill retrieval index", "test_skill_index.py"),
    ("media generation adapters", "test_media.py"),
    ("model residency manager", "test_residency.py"),
    ("compiled dotprod kernels (TOPS)", "test_tops_kernel.py"),
    ("grammar-locked agent calls", "test_agent_grammar.py"),
    ("sqlite FTS memory tier", "test_memory_sqlite.py"),
]


def main() -> int:
    # fixtures must exist before the cross-format suite
    fix = os.path.join(ROOT, "artifacts", "fixtures", "model.safetensors")
    if not os.path.exists(fix):
        print("fixtures missing — generating...\n")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_fixtures.py")],
                       check=True, cwd=ROOT)
    failures = []
    skips = []
    for label, fname in SUITES:
        path = os.path.join(HERE, fname)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           cwd=ROOT)
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        skip = r.returncode == 3  # suite convention: 3 = honest skip
        status = "PASS" if ok else ("SKIP" if skip else "FAIL")
        print(f"{status}  {label:<28} ({dt:.1f}s)  {fname}")
        if skip:
            last = next((l for l in reversed(r.stdout.strip().splitlines()) if l.strip()), "")
            print(f"      {last[:150]}")
            skips.append(label)
        elif not ok:
            failures.append((label, r.stdout + "\n" + r.stderr))
        elif "--verbose" in sys.argv or os.environ.get("LUMINA_TEST_VERBOSE"):
            print(r.stdout)
    for label, out in failures:
        print(f"\n=== {label} output ===")
        print(out[-3000:])
    print(f"\n{'=' * 60}")
    if failures:
        extra = f" ({len(skips)} skipped)" if skips else ""
        print(f"RESULT: {len(failures)} of {len(SUITES)} suites FAILED{extra}")
        return 1
    if skips:
        print(f"RESULT: all runnable suites passed — {len(skips)} SKIPPED: "
              + ", ".join(skips))
        return 0
    print(f"RESULT: all {len(SUITES)} suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
