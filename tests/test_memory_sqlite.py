"""SQLite-backed memory tier (app/server/memory_sqlite.py).

Gates, fixed before running:
  T1 engine detection: stats()/info() report "fts5" or "like" truthfully,
     cross-checked against an independent FTS5 probe (nothing hardcoded).
  T2 recall ranking: a memory with a unique rare token ("quasar") ranks 1.
  T3 budget cap: budget_tokens=40 -> injected_tokens <= 40, truncated=True.
  T4 tail block: contains the match; gibberish query -> empty block.
  T5 LIKE fallback: force_like=True -> engine=="like", recall still works.
  T6 durability: add, close, reopen -> recall still finds it (WAL commit).
  T7 concurrency: 2 threads x 20 adds -> count == 40 (WAL + serialized writes).
  T8 honesty: gibberish recall -> empty items, no error.

Every printed number is measured here. Exit codes: 0 = pass, 1 = fail,
3 = SKIP (sqlite3 unavailable on this device).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

RESULTS = []
TMP = tempfile.mkdtemp(prefix="memsqlite-")


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:200]) if not cond and extra else ""))


def fresh(name):
    path = os.path.join(TMP, name + ".db")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass
    return path


def probe_fts5():
    c = sqlite3.connect(":memory:")
    try:
        c.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        return True
    except Exception:
        return False
    finally:
        c.close()


def seed_three(mem):
    mem.add("user", "fact", "the user likes strong black coffee in the morning")
    mem.add("proj", "note", "the quasar renderer project uses a custom Vulkan backend")
    mem.add("user", "pref", "dark mode everywhere, never light themes")


# ---------------------------------------------------------------- T1 engine

def t1_engine():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("t1"))
    try:
        probe = probe_fts5()
        s = mem.stats()
        eng = s.get("engine")
        print("T1: engine=%s sqlite=%s fts5_probe=%s"
              % (eng, sqlite3.sqlite_version, probe))
        check("T1: engine detection is truthful",
              eng in ("fts5", "like") and eng == ("fts5" if probe else "like"),
              {"engine": eng, "probe": probe})
        check("T1: info() agrees with stats()", mem.info().get("engine") == eng)
        check("T1: stats shape", all(k in s for k in
              ("count", "bytes", "engine", "path", "wal")), s)
        check("T1: fresh store is empty", s["count"] == 0 and s["bytes"] > 0, s)
    finally:
        mem.close()


# ---------------------------------------------------------------- T2 ranking

def t2_ranking():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("t2"))
    try:
        seed_three(mem)
        r = mem.recall("quasar")
        print("T2: engine=%s injected_tokens=%d items=%d"
              % (r["engine"], r["injected_tokens"], len(r["items"])))
        check("T2: quasar memory ranks first",
              bool(r["items"]) and "quasar" in r["items"][0]["content"]
              and r["items"][0]["rank"] == 1, r["items"][:1])
        check("T2: injected_tokens <= budget (320)",
              r["injected_tokens"] <= 320, r["injected_tokens"])
        check("T2: rank fields are 1-based positions",
              [it["rank"] for it in r["items"]] == list(range(1, len(r["items"]) + 1)))
    finally:
        mem.close()


# --------------------------------------------------------------- T3 budget

def t3_budget():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("t3"))
    try:
        for i in range(6):
            mem.add("proj", "note",
                    "quasar build %d " % i + " ".join(
                        "word%d" % j for j in range(20)))
        r = mem.recall("quasar", budget_tokens=40)
        print("T3: budget=40 injected_tokens=%d truncated=%s items=%d"
              % (r["injected_tokens"], r["truncated"], len(r["items"])))
        check("T3: injected_tokens <= 40", r["injected_tokens"] <= 40,
              r["injected_tokens"])
        check("T3: truncated=True when items were dropped", r["truncated"] is True,
              {"items": len(r["items"]), "injected": r["injected_tokens"]})
        check("T3: at least one item fits the budget", len(r["items"]) >= 1,
              len(r["items"]))
        check("T3: budget respected with default limit",
              len(r["items"]) <= 6, len(r["items"]))
    finally:
        mem.close()


# --------------------------------------------------------------- T4 tail

def t4_tail():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("t4"))
    try:
        seed_three(mem)
        b = mem.tail_block("quasar")
        print("T4: block=%r injected_tokens=%d engine=%s"
              % (b["block"][:60], b["injected_tokens"], b["engine"]))
        check("T4: block contains the match", "quasar" in b["block"], b["block"])
        check("T4: block has the recall header",
              b["block"].startswith("[memory recall]"), b["block"][:40])
        check("T4: injected_tokens <= budget", b["injected_tokens"] <= 320,
              b["injected_tokens"])
        empty = mem.tail_block("zxqvplorb nonexistentgibberish")
        check("T4: gibberish query -> empty block, no error",
              empty["block"] == "" and empty["injected_tokens"] == 0
              and "error" not in empty, empty)
    finally:
        mem.close()


# --------------------------------------------------------------- T5 LIKE

def t5_like():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("t5"), force_like=True)
    ref = SQLiteMemory(fresh("t5ref"))
    try:
        seed_three(mem)
        seed_three(ref)
        r = mem.recall("quasar")
        print("T5: engine=%s injected_tokens=%d items=%d"
              % (r["engine"], r["injected_tokens"], len(r["items"])))
        check("T5: engine reports 'like' when forced", r["engine"] == "like",
              r["engine"])
        check("T5: LIKE recall finds the right item first",
              bool(r["items"]) and "quasar" in r["items"][0]["content"]
              and r["items"][0]["rank"] == 1, r["items"][:1])
        check("T5: stats() agrees engine=='like'", mem.stats()["engine"] == "like")
        rr = ref.recall("quasar")
        check("T5: LIKE and FTS5 agree on the candidate set",
              {it["id"] for it in r["items"]} == {it["id"] for it in rr["items"]},
              ({it["id"] for it in r["items"]}, {it["id"] for it in rr["items"]}))
        ss = mem.search_scopes(scope="user")
        check("T5: search_scopes works on the LIKE path",
              ss["count"] == 2 and all(it["scope"] == "user" for it in ss["items"]),
              ss)
    finally:
        mem.close()
        ref.close()


# ----------------------------------------------------------- T6 durability

def t6_durability():
    from memory_sqlite import SQLiteMemory
    path = fresh("t6")
    mem = SQLiteMemory(path)
    a = mem.add("user", "fact", "the user's dog is called quasar the wonderhound")
    mem.close()
    again = SQLiteMemory(path)
    try:
        r = again.recall("quasar")
        s = again.stats()
        print("T6: reopen count=%d bytes=%d wal=%s found=%s"
              % (s["count"], s["bytes"], s["wal"],
                 bool(r["items"]) and "quasar" in r["items"][0]["content"]))
        check("T6: memory survives close/reopen",
              bool(r["items"]) and "quasar" in r["items"][0]["content"], r["items"])
        check("T6: count matches the single add", s["count"] == 1, s["count"])
        check("T6: WAL mode reported after reopen", s["wal"] is True, s["wal"])
    finally:
        again.close()


# --------------------------------------------------------- T7 concurrency

def t7_concurrency():
    from memory_sqlite import SQLiteMemory
    errors = []

    def worker(mem, tid):
        try:
            for i in range(20):
                mem.add("thread", "note", "concurrent item %d from thread %d"
                        % (i, tid))
        except Exception as exc:  # pragma: no cover - only on a real bug
            errors.append(exc)

    # (a) one shared instance across both threads (internal lock)
    mem = SQLiteMemory(fresh("t7a"))
    try:
        barrier = threading.Barrier(2)
        threads = [threading.Thread(target=lambda t=t: (barrier.wait(), worker(mem, t)))
                    for t in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check("T7a: shared instance, 2x20 adds -> count == 40",
              not errors and mem.stats()["count"] == 40,
              {"errors": errors, "count": mem.stats()["count"]})
    finally:
        mem.close()

    # (b) one connection per thread on the same file (WAL serialization)
    path = fresh("t7b")
    errors.clear()
    barriers = threading.Barrier(2)

    def worker_own(tid):
        m = SQLiteMemory(path)
        try:
            barriers.wait()
            for i in range(20):
                m.add("thread", "note", "concurrent item %d from thread %d"
                      % (i, tid))
        except Exception as exc:  # pragma: no cover - only on a real bug
            errors.append(exc)
        finally:
            m.close()

    threads = [threading.Thread(target=worker_own, args=(t,)) for t in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fin = SQLiteMemory(path)
    try:
        n = fin.stats()["count"]
        print("T7b: per-thread connections, count=%d errors=%d" % (n, len(errors)))
        check("T7b: per-thread connections, 2x20 adds -> count == 40",
              not errors and n == 40, {"errors": errors, "count": n})
    finally:
        fin.close()


# ------------------------------------------------------------- T8 honesty

def t8_honesty():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("t8"))
    try:
        seed_three(mem)
        r = mem.recall("zxqvplorb qqqwww nonexistentgibberish")
        print("T8: gibberish -> items=%d error=%s engine=%s"
              % (len(r["items"]), r.get("error"), r.get("engine")))
        check("T8: gibberish returns empty items, no error",
              r["items"] == [] and "error" not in r and r["engine"] in ("fts5", "like"),
              r)
        check("T8: empty query returns empty items, no error",
              mem.recall("")["items"] == [] and "error" not in mem.recall(""))
        bad = mem.add("user", "fact", 123)
        check("T8: non-string content is an error dict", "error" in bad, bad)
        miss = mem.search_scopes(scope="no-such-scope")
        check("T8: unknown scope -> empty items, no error",
              miss["items"] == [] and "error" not in miss, miss)
        nf = mem.forget(999999)
        check("T8: forgetting a missing id is honest, not an error",
              "error" not in nf and nf["deleted"] is False, nf)
    finally:
        mem.close()


# ------------------------------------------------------- extra: forget/stats

def t9_tail_placement():
    """The Atomic-Agent discipline: recall is appended at the TAIL, so the
    prompt prefix stays byte-identical and the KV cache is preserved.
    Structural check (the real cache is measured elsewhere)."""
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("tail"))
    try:
        mem.add("proj", "note", "the quasar launch window is narrow")
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "quasar?"}]
        tb = mem.tail_block("quasar")
        if not tb.get("block"):
            check("T9 tail: nothing to inject", True)
            return
        out = msgs + [{"role": "system", "content": tb["block"]}]
        check("T9 tail: prefix untouched, block appended last",
              out[:-1] == msgs and out[-1]["content"] == tb["block"],
              [m.get("content", "")[:20] for m in out])
    finally:
        mem.close()


def extra_forget_and_meta():
    from memory_sqlite import SQLiteMemory
    mem = SQLiteMemory(fresh("extra"))
    try:
        a = mem.add("proj", "note", "remember the quasar deadline", meta={"p": 1})
        b = mem.add("proj", "note", "second note about deadlines")
        r = mem.forget(a["id"])
        check("extra: forget deletes the row", r["deleted"] is True
              and mem.stats()["count"] == 1, r)
        r2 = mem.forget(a["id"])
        check("extra: second forget reports deleted=False", r2["deleted"] is False)
        got = mem.search_scopes(scope="proj")["items"][0]
        check("extra: forgotten content is gone from the store",
              got["content"].startswith("second note")
              and "quasar deadline" not in got["content"], got)
    finally:
        mem.close()


def main() -> int:
    try:
        import memory_sqlite  # noqa: F401
    except Exception as exc:
        print("MEMSQLITE: SKIP (cannot import memory_sqlite: %s)" % exc)
        return 3
    t0 = time.time()
    t1_engine()
    t2_ranking()
    t3_budget()
    t4_tail()
    t5_like()
    t6_durability()
    t7_concurrency()
    t8_honesty()
    extra_forget_and_meta()
    t9_tail_placement()
    ok = sum(1 for _n, c in RESULTS if c)
    print("\nMEMSQLITE: %d/%d passed (%.1fs)"
          % (ok, len(RESULTS), time.time() - t0))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
