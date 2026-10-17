"""Model-residency suite: measured numbers only, no promises.

Runs against the REAL `serve` module and the tiny fixture chat model.  Every
number printed here is read from the device (/proc/self/statm, /proc/meminfo,
ru_maxrss).  Assertions are about STATE (loaded flags, arrays released, policy
round-trips, call ordering) plus ONE physics bound that was measured on this
phone: a closed fp16 allocation >= 128 MB returns to the OS within 5 MB.

What is deliberately NOT asserted: that unloading the TINY model lowers RSS.
It does not (allocator arena retention) - freed_mb is printed as measured.

Deferred work is printed as SKIP, never as a failure.

  python3 tests/test_residency.py    -> exit 0 all pass, 1 otherwise
"""
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serve  # noqa: E402
import residency  # noqa: E402
from _fixtures import build_tiny_chat  # noqa: E402

RESULTS = []
ENV_KEY = "LUMINA_RESIDENCY_MEASURE"
CFG_PATH = os.path.join(ROOT, "app", "server", "studio_config.json")


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    suffix = ("  :: " + str(detail)) if (detail and not cond) else ""
    print(("PASS " if cond else "FAIL ") + name + suffix)


def section(title):
    print("\n-- " + title + " " + "-" * max(0, 58 - len(title)))


def num(x):
    return "None" if x is None else "%.1f" % x


def main() -> int:
    t0 = time.perf_counter()
    orig_env = os.environ.pop(ENV_KEY, None)
    orig_media = json.loads(json.dumps(serve.CFG.get("media")))  # deep copy
    orig_load = serve.api_load
    orig_quick_bench = serve._quick_bench
    # The 8-token bench + load_bench.json write are not what this suite tests;
    # skipping them keeps the suite well under its 30 s budget.
    serve._quick_bench = lambda model_id: None
    print("note: serve._quick_bench patched to None for this run (no bench write)")

    try:
        # ------------------------------------------------------------------
        section("measure()")
        m = residency.measure()
        print("  source=%s rss_mb=%s peak_rss_mb=%s available_mb=%s total_mb=%s page_size=%s"
              % (m["source"], num(m["rss_mb"]), num(m["peak_rss_mb"]),
                 num(m["available_mb"]), num(m["total_mb"]), m["page_size"]))
        check("measure() reads /proc/self/statm",
              m["source"] == "proc_self_statm", m["source"])
        check("measure() rss_mb is a number > 0",
              isinstance(m["rss_mb"], (int, float)) and m["rss_mb"] > 0)
        check("measure() available_mb is a number > 0",
              isinstance(m["available_mb"], (int, float)) and m["available_mb"] > 0)
        raw_peak = residency._rusage_peak_mb()
        print("  raw ru_maxrss=%s vs statm rss=%s (the kernel peak counter LAGS "
              "the live RSS on this device; measure() reports a peak floor)"
              % (num(raw_peak), num(m["rss_mb"])))
        check("peak_rss_mb is a valid floor of the current RSS (>= rss, > 0)",
              m["peak_rss_mb"] is not None and m["peak_rss_mb"] >= m["rss_mb"]
              and m["peak_rss_mb"] > 0,
              "peak=%s rss=%s" % (num(m["peak_rss_mb"]), num(m["rss_mb"])))
        check("raw ru_maxrss fallback is readable and numeric",
              isinstance(raw_peak, (int, float)) and raw_peak > 0, str(raw_peak))

        os.environ[ENV_KEY] = "rusage"
        m_r = residency.measure()
        print("  forced rusage: source=%s rss_mb=%s peak_rss_mb=%s"
              % (m_r["source"], num(m_r["rss_mb"]), num(m_r["peak_rss_mb"])))
        check("LUMINA_RESIDENCY_MEASURE=rusage forces the ru_maxrss source",
              m_r["source"] == "ru_maxrss", m_r["source"])
        check("rusage values are numbers (peak semantics)",
              isinstance(m_r["rss_mb"], (int, float)) and m_r["rss_mb"] > 0
              and m_r["rss_mb"] == m_r["peak_rss_mb"])
        del os.environ[ENV_KEY]

        # unreadable sources -> source "none", None values, never raises
        keep = (residency._STATM, residency._MEMINFO, residency._rusage_peak_mb)
        residency._STATM = "/nonexistent/statm"
        residency._MEMINFO = "/nonexistent/meminfo"
        residency._rusage_peak_mb = lambda: None
        os.environ[ENV_KEY] = "rusage"
        try:
            m_none = residency.measure()
        finally:
            residency._STATM, residency._MEMINFO, residency._rusage_peak_mb = keep
            del os.environ[ENV_KEY]
        print("  unreadable: source=%s rss_mb=%s peak_rss_mb=%s"
              % (m_none["source"], m_none["rss_mb"], m_none["peak_rss_mb"]))
        check("unreadable sources -> source 'none' with None values",
              m_none["source"] == "none" and m_none["rss_mb"] is None
              and m_none["peak_rss_mb"] is None, m_none["source"])

        # ------------------------------------------------------------------
        section("resident() with nothing loaded")
        serve.api_unload()
        r0 = residency.resident()
        print("  loaded=%s model_id=%s weights_mb=%s policy=%s measure_source=%s"
              % (r0["loaded"], r0["model_id"], r0["weights_mb"], r0["policy"],
                 r0["measure_source"]))
        view = r0["partial_residency"]
        print("  partial_residency=%s" % view)
        check("nothing loaded -> loaded False, model_id None, weights_mb None",
              r0["loaded"] is False and r0["model_id"] is None
              and r0["weights_mb"] is None)
        check("partial residency honestly unsupported (reason non-empty)",
              view["supported"] is False and isinstance(view["reason"], str)
              and len(view["reason"]) > 0)
        check("resident() reports policy and statm source",
              r0["policy"] in residency.POLICIES
              and r0["measure_source"] == "proc_self_statm")

        # ------------------------------------------------------------------
        section("policy round-trip")
        p_set = residency.set_policy("unload")
        p_get = residency.policy()
        print("  set_policy('unload')=%s  policy()=%s" % (p_set, p_get))
        check("set_policy persists into serve.CFG['media']['policy']",
              p_set.get("policy") == "unload" and p_set.get("saved") is True
              and isinstance(serve.CFG.get("media"), dict)
              and serve.CFG["media"].get("policy") == "unload")
        check("policy() reads the stored value", p_get["policy"] == "unload")
        on_disk = json.load(open(CFG_PATH)).get("media", {}).get("policy")
        check("set_policy wrote the config file", on_disk == "unload", on_disk)
        p_bad = residency.set_policy("bogus")
        print("  set_policy('bogus')=%s" % p_bad)
        check("invalid policy -> error, stored value untouched",
              "error" in p_bad
              and serve.CFG["media"]["policy"] == "unload")

        # the default path: ensure_headroom(need) with no policy argument must
        # read the STORED policy (this is what media.py relies on).
        r_def = residency.ensure_headroom(1)
        print("  ensure_headroom(1) with no policy arg -> policy_used=%s (stored 'unload')"
              % r_def.get("policy_used"))
        check("default policy comes from CFG (stored 'unload' -> not_needed)",
              r_def.get("policy_used") == "not_needed", str(r_def.get("policy_used")))
        residency.set_policy("off")
        r_off = residency.ensure_headroom(1)
        print("  stored policy 'off' -> policy_used=%s" % r_off.get("policy_used"))
        check("stored 'off' -> ensure_headroom never touches the model",
              r_off.get("policy_used") == "off" and r_off.get("unloaded_model") is None)
        serve.CFG["media"]["policy"] = "bogus"  # hand-edited/invalid on disk
        r_inv = residency.ensure_headroom(1)
        print("  stored invalid value -> error=%s policy_used=%s"
              % (r_inv.get("error"), r_inv.get("policy_used")))
        check("stored invalid policy -> honest error dict, no unload",
              "error" in r_inv and r_inv.get("policy_used") == "invalid")
        residency.set_policy("unload")  # back to the value the next checks expect

        # ------------------------------------------------------------------
        section("tiny model residency")
        mdir = build_tiny_chat()
        rss_before_load = residency.measure()["rss_mb"]
        lr = serve.api_load(mdir)
        rss_after_load = residency.measure()["rss_mb"]
        r1 = residency.resident()
        print("  api_load=%s" % {k: lr.get(k) for k in ("loaded", "params_m", "layers")})
        print("  rss_before_load=%s rss_after_load=%s delta=%s peak_rss_mb=%s"
              % (num(rss_before_load), num(rss_after_load),
                 num(None if None in (rss_before_load, rss_after_load)
                     else round(rss_after_load - rss_before_load, 3)),
                 num(r1["peak_rss_mb"])))
        print("  weights_mb=%s (arrays)  t tensors=%d  params_by_layer=%s"
              % (r1["weights_mb"], len(serve.CHAT["model"].t),
                 r1["params_by_layer"]))
        check("api_load returned no error and a model is resident",
              "error" not in lr and r1["loaded"] is True,
              str(lr)[:200])
        check("resident() reports model_id and weights_mb > 0",
              r1["model_id"] == mdir and isinstance(r1["weights_mb"], (int, float))
              and r1["weights_mb"] > 0, str(r1["weights_mb"]))
        check("resident() counts params per layer",
              isinstance(r1["params_by_layer"], dict)
              and len(r1["params_by_layer"]) == serve.CHAT["model"].n_layer,
              str(r1["params_by_layer"]))

        # ------------------------------------------------------------------
        section("absurd need: refuse BEFORE unloading")
        model_obj = serve.CHAT["model"]
        r_abs = residency.ensure_headroom(50_000, "unload")
        print("  need=50000 policy=unload -> policy_used=%s available_before=%s"
              % (r_abs.get("policy_used"), r_abs.get("available_mb_before")))
        print("  error=%s" % r_abs.get("error"))
        avail_txt = "%g" % r_abs["available_mb_before"]
        check("absurd need -> error mentioning the real available MB",
              "error" in r_abs and avail_txt in str(r_abs["error"]), str(r_abs.get("error"))[:160])
        check("absurd need -> policy_used 'insufficient'",
              r_abs.get("policy_used") == "insufficient", str(r_abs.get("policy_used")))
        check("absurd need -> model NOT touched (same object, not None)",
              serve.CHAT["model"] is model_obj and model_obj is not None)
        check("absurd need -> no unload recorded / nothing reloaded",
              r_abs.get("unloaded_model") is None and r_abs.get("reloaded") is False)

        # ------------------------------------------------------------------
        section("not-needed guard and the unload path")
        r_need = residency.ensure_headroom(1, "unload")
        print("  need=1 policy=unload (RAM covers it) -> policy_used=%s note=%s"
              % (r_need.get("policy_used"), r_need.get("note")))
        check("plenty of RAM -> policy_used 'not_needed', model still resident",
              r_need.get("policy_used") == "not_needed"
              and r_need.get("unloaded_model") is None
              and residency.resident()["loaded"] is True)
        r_unl = residency.ensure_headroom(1, "unload", force=True)
        print("  forced unload -> policy_used=%s unloaded_model=%s array_mb=%s"
              % (r_unl.get("policy_used"), r_unl.get("unloaded_model"),
                 r_unl.get("array_mb")))
        print("  rss_before_mb=%s rss_after_mb=%s freed_mb=%s malloc_trim_released_mb=%s"
              % (num(r_unl.get("rss_before_mb")), num(r_unl.get("rss_after_mb")),
                 num(r_unl.get("freed_mb")), num(r_unl.get("malloc_trim_released_mb"))))
        print("  available_mb_before=%s available_mb_after=%s note=%s"
              % (num(r_unl.get("available_mb_before")),
                 num(r_unl.get("available_mb_after")), r_unl.get("note")))
        check("forced unload -> recorded model_id and array_mb > 0",
              r_unl.get("unloaded_model") == mdir
              and isinstance(r_unl.get("array_mb"), (int, float))
              and r_unl["array_mb"] > 0, str(r_unl.get("array_mb")))
        check("forced unload -> error None and policy_used is the real policy",
              r_unl.get("error") is None and r_unl.get("policy_used") == "unload")
        check("forced unload -> resident() says not loaded",
              residency.resident()["loaded"] is False)
        check("freed_mb is a measured number (may be ~0 on small models)",
              isinstance(r_unl.get("freed_mb"), (int, float)),
              str(r_unl.get("freed_mb")))
        serve.api_unload()

        # ------------------------------------------------------------------
        section("reload pairing (recorder on serve.api_load)")
        reload_calls = []

        def fake_load(mid):
            reload_calls.append(mid)
            return {"loaded": mid, "params_m": 0.1, "layers": 2}

        lr2 = serve.api_load(mdir)  # real load first: the test needs a resident model
        check("fixture re-loaded for the reload test",
              "error" not in lr2 and serve.CHAT["model"] is not None)
        serve.api_load = fake_load
        try:
            r_ref = residency.ensure_headroom(50_000, "reload_after")
            print("  refusal with reload_after -> policy_used=%s unloaded=%s reloaded=%s"
                  % (r_ref.get("policy_used"), r_ref.get("unloaded_model"),
                     r_ref.get("reloaded")))
            check("refusal does not unload and does not reload (no churn)",
                  r_ref.get("policy_used") == "insufficient"
                  and r_ref.get("unloaded_model") is None
                  and r_ref.get("reloaded") is False and reload_calls == [])

            r_rl = residency.ensure_headroom(1, "reload_after", force=True)
            fin = residency.finish(r_rl.get("policy_used"), r_rl.get("unloaded_model"))
            print("  ensure_headroom(1,'reload_after',force) -> unloaded_model=%s reloaded=%s"
                  % (r_rl.get("unloaded_model"), r_rl.get("reloaded")))
            print("  finish()=%s  reload_calls=%s" % (fin, reload_calls))
            check("finish() reloads after a reload_after unload",
                  fin.get("reloaded") is True and fin.get("reload_error") is None)
            check("reload used the unloaded model id", reload_calls == [mdir],
                  str(reload_calls))

            err_txt = "not enough free memory: needs 2499 MB, have 1705 MB"

            def fake_load_err(mid):
                reload_calls.append(mid)
                return {"error": err_txt}

            serve.api_load = fake_load_err
            fin_err = residency.finish("reload_after", mdir)
            print("  finish() with failing api_load -> %s" % fin_err)
            check("reload error surfaced verbatim",
                  fin_err.get("reloaded") is False
                  and fin_err.get("reload_error") == err_txt,
                  str(fin_err.get("reload_error")))
            fin_off = residency.finish("off", mdir)
            check("finish() is a no-op for policy 'off'/'not_needed'",
                  fin_off.get("reloaded") is False
                  and fin_off.get("reload_error") is None
                  and residency.finish("not_needed", None).get("reloaded") is False)
            r_noid = residency.reload_model(None)
            check("reload_model(None) -> honest error",
                  r_noid.get("error") == "no model id to reload", str(r_noid))
        finally:
            serve.api_load = orig_load
        serve.api_unload()

        # ------------------------------------------------------------------
        section("large-allocation reclaim proof (>= 128 MB fp16)")
        gc.collect()
        base = residency.measure()["rss_mb"]
        big = np.full(64 * 1024 * 1024, 1.0, dtype=np.float16)  # 134.2 MB
        rss_alloc = residency.measure()["rss_mb"]
        big_bytes = big.nbytes
        del big
        gc.collect()
        rss_free = residency.measure()["rss_mb"]
        print("  allocation=%.1f MB  rss_baseline=%s rss_alloc=%s rss_free=%s"
              % (big_bytes / 1e6, num(base), num(rss_alloc), num(rss_free)))
        print("  growth=%s MB  retained_after_free=%s MB"
              % (num(round(rss_alloc - base, 3)), num(round(rss_free - base, 3))))
        check("allocation really committed (RSS grew >= 100 MB)",
              (rss_alloc - base) >= 100.0,
              "growth=%.1f MB" % (rss_alloc - base))
        check("closed 128 MB fp16 allocation returned to the OS (<= 5 MB left)",
              (rss_free - base) <= 5.0,
              "retained=%.1f MB" % (rss_free - base))

        # ------------------------------------------------------------------
        section("deferred")
        print("SKIP partial residency: loader materializes every tensor eagerly "
              "(chat.py:118-143); whole-file reads (adapters/safetensors.py:87-91, "
              "adapters/lumina.py:95-99); needs an mmap accessor rewrite "
              "(deferred, documented in STATUS.md)")
        print("SKIP resident()['partial_residency'] is reported as unsupported with the "
              "same reason; keeping a subset of layers resident would require an mmap "
              "weight accessor this loader does not have (no failure counted).")
        check("deferred partial residency is honestly reported, not faked",
              residency.resident()["partial_residency"]["supported"] is False)

        # ------------------------------------------------------------------
        section("cleanup")
        serve.api_unload()
        if orig_media is None:
            serve.CFG.pop("media", None)
        else:
            serve.CFG["media"] = orig_media
        serve._save_cfg()
        if orig_env is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = orig_env
        fin = residency.resident()
        disk_media = json.load(open(CFG_PATH)).get("media")
        print("  loaded=%s  policy()=%s  cfg media restored to %s  disk media=%s"
              % (fin["loaded"], residency.policy()["policy"], orig_media, disk_media))
        want = orig_media.get("policy") if isinstance(orig_media, dict) else None
        check("no model left loaded", fin["loaded"] is False)
        check("config restored (CFG and file)",
              serve.CFG.get("media") == orig_media
              and (disk_media.get("policy") if isinstance(disk_media, dict) else None) == want)
    finally:
        serve.api_load = orig_load
        serve._quick_bench = orig_quick_bench
        if orig_env is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = orig_env
        if orig_media is None:
            serve.CFG.pop("media", None)
        else:
            serve.CFG["media"] = orig_media
        serve._save_cfg()

    n = sum(1 for _, c in RESULTS if c)
    print("\nwall=%.1fs" % (time.perf_counter() - t0))
    print(f"RESIDENCY: {n}/{len(RESULTS)} passed")
    print("RESULT:", "ALL PASS" if n == len(RESULTS) else "FAILURES PRESENT")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
