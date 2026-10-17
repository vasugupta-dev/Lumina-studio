"""Scale benchmark for the compact skill retrieval index (skill_index.py).

Builds a seeded synthetic skill corpus (same generator family as the test
suite: 8-16 words from a 300-word seeded pool plus a unique marker token per
doc, queries = held-out marker combos with a known target doc), then measures:

  * build_s                  - untraced wall time to index every doc
  * approx_bytes             - SkillIndex.approx_bytes() (structure estimate)
  * tracemalloc_peak_mb      - peak of a traced rebuild (the RAM question)
  * p50_ms / p95_ms          - per-query search latency over --queries queries
  * accuracy_top1/top5       - marker queries must resolve to their target

Methodology notes (honest numbers):
  * latency is measured with tracemalloc OFF; tracemalloc inflates allocation
    cost several-fold and would misstate the search latency.
  * the peak comes from a second, traced build, so build_s is not inflated by
    tracing.
  * 1M docs is a projection from this measurement, not a measurement.

Run:  python3 scripts/bench_skill_index.py --n 100000 --queries 200
Writes: artifacts/skill_index_bench.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

from skill_index import SkillIndex  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "skill_index_bench.json")


def synth_stream(n, m, seed=7):
    """Seeded streaming corpus + held-out marker queries (see test suite)."""
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    pool = ["".join(rng.choice(alpha) for _ in range(7)) for _ in range(300)]
    targets: dict[int, list[int]] = {}
    for j in range(m):
        targets.setdefault((j * 97 + 13) % n, []).append(j)
    queries = [None] * m

    def docs():
        for i in range(n):
            words = [pool[rng.randrange(300)] for _ in range(rng.randint(8, 16))]
            marker = "m%07d" % i
            for j in targets.get(i, ()):
                queries[j] = (marker + " " + " ".join(words[:2]), i)
            yield {"id": "sk%06d" % i, "desc": " ".join(words), "prompt": marker}

    return docs(), queries


def percentile(sorted_ms, q):
    if not sorted_ms:
        return 0.0
    i = min(len(sorted_ms) - 1, max(0, int(math.ceil(q * len(sorted_ms))) - 1))
    return sorted_ms[i]


def device_info() -> dict:
    """uname-ish platform description (no hostname, no secrets)."""
    try:
        un = os.uname()
        return {"system": un.sysname, "machine": un.machine,
                "release": un.release, "python": platform.python_version(),
                "cpu_count": os.cpu_count()}
    except AttributeError:  # non-unix fallback
        return {"system": platform.system(), "machine": platform.machine(),
                "release": platform.release(),
                "python": platform.python_version(), "cpu_count": os.cpu_count()}


def run(n: int, m: int) -> dict:
    # pass 1: untraced build (honest build_s)
    docs, queries = synth_stream(n, m, seed=7)
    si = SkillIndex()
    t0 = time.perf_counter()
    for d in docs:
        si.add(d)
    build_s = time.perf_counter() - t0
    apx = si.approx_bytes()

    # pass 2: traced rebuild for the peak (memory, not time)
    docs2, _q2 = synth_stream(n, m, seed=7)
    si2 = SkillIndex()
    tracemalloc.start()
    for d in docs2:
        si2.add(d)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    del si2  # keep resident memory to one index on the phone

    # pass 3: timed queries (tracemalloc off)
    ms = []
    top1 = top5 = 0
    for q, target in queries:
        t = time.perf_counter()
        hits = si.search(q, k=5)["skills"]
        ms.append((time.perf_counter() - t) * 1000.0)
        got = [h["id"] for h in hits]
        want = "sk%06d" % target
        top1 += bool(got) and got[0] == want
        top5 += want in got
    ms.sort()
    return {
        "n": n,
        "build_s": round(build_s, 4),
        "approx_bytes": apx,
        "tracemalloc_peak_mb": round(peak_mb, 3),
        "p50_ms": round(percentile(ms, 0.50), 4),
        "p95_ms": round(percentile(ms, 0.95), 4),
        "accuracy_top1": round(top1 / m, 4),
        "accuracy_top5": round(top5 / m, 4),
        "device": device_info(),
        "note": "1M is a projection, not measured; measured on the phone (aarch64)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000,
                    help="synthetic docs (default 100000)")
    ap.add_argument("--queries", type=int, default=200,
                    help="held-out marker queries (default 200)")
    args = ap.parse_args()
    out = run(args.n, args.queries)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("skill-index bench (MEASURED on this device, seed 7)")
    print("  n=%d docs, %d held-out marker queries" % (out["n"], args.queries))
    print("  build_s (untraced)          %s" % out["build_s"])
    print("  approx_bytes                %d" % out["approx_bytes"])
    print("  tracemalloc_peak_mb         %s" % out["tracemalloc_peak_mb"])
    print("  p50_ms / p95_ms             %s / %s" % (out["p50_ms"], out["p95_ms"]))
    print("  accuracy top1 / top5        %s / %s"
          % (out["accuracy_top1"], out["accuracy_top5"]))
    print("  device                      %s" % out["device"])
    print("  wrote                       %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
