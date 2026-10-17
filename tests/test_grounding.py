"""Grounding gate: claim attribution vs a fixed fixture, measured honestly.

The fixture (tests/fixtures/grounding_cases.json) labels, for each answer,
which claim indices are fabricated relative to the used passages. The suite
prints the unsupported-claim DETECTION RATE and the FALSE-REFUSAL RATE and
enforces the gates (>= 0.90 and <= 0.10). Two cases are known lexical limits
(direction/antonym swaps with high term overlap) — they are expected misses
and are reported; semantic flips of that class need the model-verifier pass
(boost.verify), not the lexical scorer.
"""
import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

import grounding  # noqa: E402

RESULTS = []
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "grounding_cases.json")


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    suffix = ("  :: " + str(detail)) if (detail and not cond) else ""
    print(("PASS " if cond else "FAIL ") + name + suffix)


def main() -> int:
    fx = json.load(open(FIXTURE))
    by_id = {p["id"]: p for p in fx["passages"]}

    # --- tokenizer / bm25 parity with the server ---------------------------
    import serve  # noqa: E402
    samples = ["Hello WORLD 42", "café naïve", "north-west", "", "Tern-9",
               "Δ unicode Γ", "a_b-c", "  spaced  words ", "UPPER", "下"]
    check("tok parity with serve._toks",
          all(grounding.tok(s) == serve._toks(s) for s in samples))

    saved = (serve.RAG_PATH, serve.RAG["chunks"], serve.RAG["df"])
    try:
        serve.RAG_PATH = "/nonexistent/rag_parity.json"
        docs = [p["text"] for p in fx["passages"]]
        serve.RAG["chunks"] = [{"id": i, "source": p["source"], "text": p["text"],
                                "tokens": serve._toks(p["text"])}
                               for i, p in enumerate(fx["passages"])]
        df = {}
        for c in serve.RAG["chunks"]:
            for t in set(c["tokens"]):
                df[t] = df.get(t, 0) + 1
        serve.RAG["df"] = df
        ok_par = True
        for q in ["Kestrel rover drill", "battery charge minutes", "depth meters"]:
            hits = serve._rag_score(q, 10)
            mine = grounding.bm25_scores(q, [serve._toks(t) for t in docs])
            for h in hits:
                if abs(h["bm25"] - round(mine[h["id"]], 4)) > 1e-9:
                    ok_par = False
        check("bm25 score parity with serve._rag_score (shared corpus)", ok_par)
    finally:
        serve.RAG_PATH, serve.RAG["chunks"], serve.RAG["df"] = saved

    # --- numbers / claims --------------------------------------------------
    check("extract_numbers canonicalizes",
          grounding.extract_numbers("1,000 and 42.0 and 3.14") == ["1000", "42", "3.14"])
    check("extract_numbers normalizes number words",
          grounding.extract_numbers("six lanes") == ["6"]
          and grounding.extract_numbers("four lanes") == ["4"])
    cls = grounding.split_claims("First claim here. Second has 42 units! Third?")
    check("split_claims: boundaries, ids, kinds",
          [c["id"] for c in cls] == [0, 1, 2]
          and cls[0]["kind"] == "assertion" and cls[1]["kind"] == "numeric")

    # --- support_check -----------------------------------------------------
    kestrel = [by_id["kestrel"]]
    ok_d = grounding.support_check("Its summit reaches 3,310 meters above sea level.", kestrel)
    check("grounded claim is supported", ok_d["supported"] is True, ok_d)
    n_d = grounding.support_check("Its summit reaches 4,500 meters above sea level.", kestrel)
    check("fabricated number -> number_conflict + unsupported",
          n_d["supported"] is False and n_d["number_conflict"] is True and
          any(r.startswith("numbers_not_in_evidence") for r in n_d["reasons"]))
    e_d = grounding.support_check("The drill was built by Zorblax Industries.", [by_id["rover"]])
    check("fabricated entity -> entity_missing + unsupported",
          e_d["supported"] is False and e_d["entity_missing"] is True)
    z_d = grounding.support_check("Anything at all.", [])
    check("no passages -> unsupported with reason",
          z_d["supported"] is False and "no_passages" in z_d["reasons"])

    # --- gate / evidence injection -----------------------------------------
    check("gate_decision refuses without evidence",
          grounding.gate_decision([])["refuse"] is True
          and grounding.gate_decision(kestrel)["refuse"] is False)
    blk = grounding.evidence_block(kestrel)
    check("evidence_block numbers sources", "[1]" in blk and "geo notes" in blk)
    msgs = grounding.evidence_messages(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}], kestrel)
    check("evidence_messages merges an existing system message",
          msgs[0]["role"] == "system" and "base" in msgs[0]["content"]
          and "EVIDENCE" in msgs[0]["content"] and len(msgs) == 2)
    msgs2 = grounding.evidence_messages([{"role": "user", "content": "q"}], kestrel)
    check("evidence_messages inserts when no system message",
          msgs2[0]["role"] == "system" and len(msgs2) == 2)

    # --- ground_answer policies --------------------------------------------
    ans = "Kestrel Peak is a dormant volcano. Its summit reaches 4,500 meters above sea level."
    g_flag = grounding.ground_answer(ans, kestrel)
    check("flag policy: text untouched, claims annotated",
          g_flag["answer"] == ans and g_flag["unsupported"] == 1
          and g_flag["claims"][0]["supported"] and not g_flag["claims"][1]["supported"])
    g_with = grounding.ground_answer(ans, kestrel, policy="withhold")
    check("withhold removes exactly the flagged sentence",
          "4,500" not in g_with["answer"]
          and "Kestrel Peak is a dormant volcano." in g_with["answer"]
          and g_with["answer_original"] == ans)
    g_all = grounding.ground_answer("Zorblax built the drill.", [by_id["rover"]],
                                    policy="withhold")
    check("withhold with nothing supportable -> REFUSAL_TEXT",
          g_all["answer"] == grounding.REFUSAL_TEXT and g_all["gate"] == "refuse")
    det = json.dumps(grounding.ground_answer(ans, kestrel), sort_keys=True)
    check("ground_answer is deterministic",
          det == json.dumps(grounding.ground_answer(ans, kestrel), sort_keys=True))

    # --- fixture metrics (the actual gate) ----------------------------------
    bad_idx = []
    for case in fx["cases"]:
        n_claims = len(grounding.split_claims(case["answer"]))
        for i in case["unsupported"]:
            if i >= n_claims:
                bad_idx.append(case["id"])
    check("fixture integrity: every labeled claim index exists", not bad_idx, bad_idx)

    det_hit = det_tot = 0
    frr_hit = frr_tot = 0
    ref_ok = ref_tot = 0
    misses, false_refusals = [], []
    for case in fx["cases"]:
        passages = [by_id[i] for i in case["use"]]
        if case.get("refusal") and not case["use"]:
            ref_tot += 1
            if grounding.gate_decision(passages)["refuse"]:
                ref_ok += 1
            continue
        g = grounding.ground_answer(case["answer"], passages)
        if case.get("refusal"):
            ref_tot += 1
            if g["gate"] == "refuse":
                ref_ok += 1
            continue
        for i in case["unsupported"]:
            det_tot += 1
            if not g["claims"][i]["supported"]:
                det_hit += 1
            else:
                misses.append((case["id"], i, g["claims"][i]["text"]))
        for i, c in enumerate(g["claims"]):
            if i in case["unsupported"]:
                continue
            frr_tot += 1
            if not c["supported"]:
                frr_hit += 1
                false_refusals.append((case["id"], i, c["text"],
                                       ",".join(c.get("reasons", []))))

    det_rate = det_hit / det_tot if det_tot else 0.0
    frr_rate = frr_hit / frr_tot if frr_tot else 1.0
    print("  planted unsupported claims: %d   detected: %d   detection rate: %.4f"
          % (det_tot, det_hit, det_rate))
    print("  supportable claims: %d   flagged: %d   false-refusal rate: %.4f"
          % (frr_tot, frr_hit, frr_rate))
    print("  refusal paths: %d/%d" % (ref_ok, ref_tot))
    for m in misses:
        print("  MISS   %s claim %d: %s" % m)
    for f in false_refusals:
        print("  FRR    %s claim %d: %s (%s)" % f)

    check("detection rate >= 0.90", det_rate >= 0.90, "measured=%.4f" % det_rate)
    check("false-refusal rate <= 0.10", frr_rate <= 0.10, "measured=%.4f" % frr_rate)
    check("every refusal path refuses", ref_ok == ref_tot and ref_tot > 0)

    # --- serve wiring: grounding + boost + compaction + cache meter ---------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _fixtures import build_tiny_chat  # noqa: E402
    build_tiny_chat()
    lr = serve.api_load("/tmp/tiny-chat")
    check("tiny model loads through serve.api_load", "loaded" in lr, lr)

    saved2 = (serve.RAG_PATH, serve.RAG["chunks"], serve.RAG["df"],
              dict(serve.CACHE_METER), dict(serve.CFG))
    try:
        serve.RAG_PATH = "/nonexistent/rag_serve.json"
        serve.RAG["chunks"] = [{"id": i, "source": p["source"], "text": p["text"],
                                "tokens": serve._toks(p["text"])}
                               for i, p in enumerate(fx["passages"])]
        df = {}
        for c in serve.RAG["chunks"]:
            for t in set(c["tokens"]):
                df[t] = df.get(t, 0) + 1
        serve.RAG["df"] = df

        before = dict(serve.CACHE_METER)
        r = serve.api_chat([{"role": "user", "content": "Zqx vump blorf?"}], 8, 0.0,
                           ground={"k": 3})
        after = dict(serve.CACHE_METER)
        check("api_chat refuses a no-evidence question before generating",
              r.get("refused") is True and r.get("text") == grounding.REFUSAL_TEXT
              and (r.get("grounding") or {}).get("gate") == "refuse", r)
        check("refused turn does not touch the cache meter",
              before["turns"] == after["turns"] and before["reused"] == after["reused"]
              and before["computed"] == after["computed"])

        r2 = serve.api_chat([{"role": "user",
                              "content": "What does the Halcyon cell store per kilogram?"}],
                            8, 0.0, ground={"k": 3})
        check("grounded api_chat attaches a grounding verdict",
              isinstance(r2.get("grounding"), dict)
              and r2["grounding"].get("gate") in ("pass", "flag", "refuse")
              and isinstance(r2["grounding"].get("claims"), list), r2.get("grounding"))
        after2 = dict(serve.CACHE_METER)
        check("grounded turn increments the prefill meter",
              after2["turns"] == after["turns"] + 1)

        serve.CFG["compact_budget_tokens"] = 120
        serve.CFG["compact_keep_recent"] = 2
        hist = []
        for i in range(12):
            hist.append({"role": "user",
                         "content": "question %d with several extra words for length" % i})
            hist.append({"role": "assistant", "content": "answer %d" % i})
        hist.append({"role": "user", "content": "final question"})
        c_before = serve.CACHE_METER["compactions"]
        rc = serve.api_chat(hist, 8, 0.0, compact=True)
        check("api_chat compact=True fires one checkpoint and still answers",
              serve.CACHE_METER["compactions"] == c_before + 1
              and isinstance(rc.get("text"), str), rc if "error" in rc else None)
        hist2 = [dict(m) for m in hist]
        serve.api_chat(hist2, 8, 0.0)
        check("api_chat default keeps auto-compaction off",
              serve.CACHE_METER["compactions"] == c_before + 1 and hist2 == hist)

        gcheck = serve.api_ground_check({"answer": "Bluefern Lake reaches a depth of 320 meters.",
                                         "passages": [by_id["bluefern"]]})
        check("api_ground_check flags a fabricated number",
              gcheck.get("unsupported") == 1
              and gcheck["claims"][0]["number_conflict"] is True, gcheck)
        gq = serve.api_ground_check({"answer": "Bluefern Lake reaches a depth of 320 meters.",
                                     "query": "Bluefern Lake depth meters"})
        check("api_ground_check query mode retrieves evidence first",
              isinstance(gq.get("passages"), int) and gq["passages"] >= 1
              and gq.get("unsupported") == 1, gq)

        rb = serve.api_boost({"question": "17*23=391?", "n": 1, "do_verify": False,
                              "max_tokens": 16})
        check("api_boost returns a trace + honest real_model_lift label",
              isinstance(rb.get("trace"), list) and rb.get("trace")
              and str(rb.get("real_model_lift", "")).startswith("not measured"),
              rb)

        st = serve.api_cache_stats()
        check("api_cache_stats reports the measured hit rate",
              st["turns"] >= 2 and st["target"] == 0.90
              and 0.0 <= st["hit_rate"] <= 1.0 and st["last"] is not None, st)
    finally:
        (serve.RAG_PATH, serve.RAG["chunks"], serve.RAG["df"],
         _meter, _cfg) = saved2
        serve.CACHE_METER.clear()
        serve.CACHE_METER.update(_meter)
        serve.CFG.clear()
        serve.CFG.update(_cfg)
        try:
            serve.api_unload()
        except Exception:
            pass

    n = sum(1 for _, c in RESULTS if c)
    print(f"\nGROUNDING: {n}/{len(RESULTS)} passed")
    print("RESULT:", "ALL PASS" if n == len(RESULTS) else "FAILURES PRESENT")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
