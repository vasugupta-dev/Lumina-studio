"""Lumina local server — stdlib only. Serves the web UI + JSON API."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WEB = os.path.normpath(os.path.join(HERE, "..", "web"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)  # sysops / browser_driver / custom_mcp live here

MIME = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
        ".json": "application/json", ".svg": "image/svg+xml",
        ".png": "image/png", ".webp": "image/webp",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
        ".flac": "audio/flac", ".txt": "text/plain",
        ".webmanifest": "application/manifest+json",
        ".ico": "image/x-icon"}


def api_status() -> dict:
    from lumina_framework.adapters import available_formats
    sp = os.path.join(ROOT, "STATUS.md")
    status = open(sp).read() if os.path.exists(sp) else ""
    out = {"formats": available_formats(), "status": status}
    try:
        out["operator"] = _operator_mode()  # defined later in module
    except Exception:
        pass
    return out


def api_targets() -> dict:
    """Aspirational speed targets (NOT measurements) + the measured record.

    The target cards are goals for the model classes named; every number in
    "measured" comes from artifacts/ or a live bench, with its source.
    """
    return {
        "models": [
            {"id": "qwen3.8-flash-next", "label": "Qwen 3.8 Flash Next",
             "layers": 48, "experts": 512, "d_model": 2560, "top_k": 10,
             "target_tps": "50-70 raw",
             "total_params_b": 180, "active_params_b": 6,
             "arch": "qwen4_exp (hybrid Gated DeltaNet + QSA, researched)",
             "kind": "aspirational-target",
             "note": "goal for this model class, not a measurement"},
            {"id": "glm-5.3-flash", "label": "GLM 5.3 Flash",
             "layers": 45, "experts": 288, "d_model": 4096, "top_k": 8,
             "target_tps": "45-50 w/ MTP",
             "kind": "aspirational-target",
             "note": "goal for this model class, not a measurement"},
            {"id": "glm-5.3", "label": "GLM 5.3",
             "layers": 45, "experts": 288, "d_model": 6144, "top_k": 8,
             "target_tps": "45-50 w/ MTP",
             "kind": "aspirational-target",
             "note": "goal for this model class, not a measurement"},
        ],
        "measured": _measured_speeds(),
    }


_MEASURED_CACHE: dict = {}


def _measured_speeds() -> dict:
    """Measured proxy numbers from artifacts/bench_runtime.json (cached)."""
    if _MEASURED_CACHE.get("data") is not None:
        return _MEASURED_CACHE["data"]
    out = {"label": "MEASURED RESULT (proxy rig, CPU-only)",
           "source": "artifacts/bench_runtime.json", "cases": []}
    try:
        doc = json.load(open(os.path.join(ROOT, "artifacts", "bench_runtime.json")))
        for c in doc.get("cases", [])[:6]:
            out["cases"].append({
                "label": c.get("label"),
                "warm_tok_per_s": round(float(c.get("warm_tok_per_s", 0)), 1),
                "tokens": c.get("tokens")})
    except Exception as e:
        out["error"] = "artifacts unreadable: " + str(e)[:100]
    _MEASURED_CACHE["data"] = out
    return out


def api_hardware() -> dict:
    """Every compute resource visible to the server + the Qwen3.8 recipe."""
    from lumina_framework.cuda_opt import hardware_concurrency, recommend_threads
    from lumina_framework.hardware import qwen38_laptop_recipe
    return {"inventory": hardware_concurrency(),
            "thread_policy": recommend_threads(),
            "qwen38_recipe": qwen38_laptop_recipe()}


def api_bench_tiny() -> dict:
    """Tiny proxy bench only (small, fast, phone-safe)."""
    import time
    import numpy as np
    from lumina_framework.runtime import RuntimeFabric, StubModel
    from lumina_framework.expert_memory import ExpertMemoryEngine
    from lumina_framework.memory import TIER_RAM
    m = StubModel(n_layer=1, n_expert=4, d_model=128, d_ff=160,
                  top_k=1, seed=6)
    bank = {}
    for lyr in m.layers:
        for e in lyr.experts:
            bank[e.name] = e
        bank[lyr.shared.name] = lyr.shared
    eng = ExpertMemoryEngine(bank, ram_budget=256 * 1024 * 1024,
                             vram_budget=0)
    eng.pin_hot_experts(10, target=TIER_RAM)
    rt = RuntimeFabric(m, eng, prefetch=False)
    t0 = time.perf_counter()
    res = rt.decode(np.array([1, 2, 3, 4]), 100, target=TIER_RAM)
    from lumina_framework.cuda_opt import hardware_concurrency
    inv = hardware_concurrency()
    return {"tokens_per_second": round(res["tokens_per_second"], 1),
            "seconds": round(time.perf_counter() - t0, 2),
            "cpu_count": inv.get("cpu_count"),
            "blas": inv.get("blas"),
            "note": "tiny proxy, CPU-only, not a target-model claim"}


LOG_PATH = os.path.join(HERE, "requests.jsonl")
_LOG_LOCK = __import__("threading").Lock()
_LOG_RING: list = []


def t_log(path: str, method: str) -> None:
    """Append-only request log (path only, never bodies/keys). Capped ring
    + jsonl file; /api/logs reads it back."""
    import datetime
    line = {"t": datetime.datetime.now().isoformat(timespec="seconds"),
            "method": method, "path": path}
    try:
        with _LOG_LOCK:
            _LOG_RING.append(line)
            if len(_LOG_RING) > 500:
                del _LOG_RING[:250]
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:
        pass


def _auth_token() -> str | None:
    return os.environ.get("LUMINA_TOKEN") or AUTH_FLAG.get("token")


AUTH_FLAG: dict = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "Lumina/0.1"

    def _authorized(self) -> bool:
        tok = _auth_token()
        if not tok:
            return True
        got = self.headers.get("Authorization", "")
        want = "Bearer " + tok
        return hmac.compare_digest(got, want)

    def _guard(self, fn, *a, **k):
        """Never die mid-response: a handler crash returns JSON, not a hang."""
        try:
            return self._json(fn(*a, **k))
        except Exception as e:
            try:
                return self._json({"error": ("handler: " + str(e))[:300]}, 500)
            except Exception:
                pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        qs = parse_qs(u.query)
        if p.startswith("/api/browser/shot/"):
            return self._serve_shot(p, qs)
        if p.startswith(("/api/", "/v1/")):
            if not self._authorized():
                return self._json({"error": {"message": "invalid api token",
                                             "type": "auth_error"}}, 401)
            t_log(p, "GET")
        if p == "/api/status":
            return self._json(api_status())
        if p == "/api/targets":
            return self._json(api_targets())
        if p == "/api/bench-tiny":
            return self._guard(api_bench_tiny)
        if p == "/api/hardware":
            return self._guard(api_hardware)
        if p == "/v1/models":
            return self._json(api_oai_models())
        if p == "/api/doctor":
            return self._guard(api_doctor, (qs.get("id") or [""])[0])
        if p == "/api/plan":
            return self._guard(api_plan, (qs.get("id") or [""])[0])
        if p == "/api/logs":
            return self._json(api_logs())
        if p == "/api/rag/list":
            return self._json(api_rag_list())
        if p == "/api/rag/query":
            return self._guard(api_rag_query, (qs.get("q") or [""])[0],
                               int((qs.get("k") or ["5"])[0]))
        if p == "/api/cache/stats":
            return self._json(api_cache_stats())
        if p == "/api/arena/elo":
            return self._json(api_elo_get())
        if p == "/api/catalog":
            return self._guard(api_catalog)
        if p == "/api/mcp/registry":
            return self._guard(api_mcp_registry, (qs.get("q") or [""])[0],
                               (qs.get("cat") or [""])[0])
        if p == "/api/mcp/status":
            return self._json(api_mcp_status())
        if p == "/api/skills":
            return self._json(api_skills_list())
        if p == "/api/skills/search":
            return self._guard(api_skills_search, (qs.get("q") or [""])[0],
                               int((qs.get("k") or ["5"])[0]))
        if p == "/api/media/status":
            import media as _M
            return self._guard(_M.api_media_status)
        if p == "/api/media/backends":
            import media as _M
            return self._guard(_M.api_media_backends)
        if p == "/api/media/workflows":
            import media as _M
            return self._guard(_M.api_media_workflows)
        if p == "/api/media/jobs":
            import media as _M
            _mid = (qs.get("id") or [""])[0]
            return self._guard(_M.api_media_jobs,
                               {"id": _mid} if _mid else None)
        if p == "/api/media/file":
            return self._serve_media((qs.get("name") or [""])[0], qs)
        if p == "/api/residency":
            import residency as _RES
            out = _RES.resident()
            out.update(_RES.policy())
            return self._json(out)
        if p == "/api/tops":
            from lumina_framework import fused as _FUSED
            return self._json(_FUSED.tops_status())
        if p == "/api/workflows":
            return self._json(api_workflow_examples())
        if p == "/api/browser/status":
            import browser_driver
            return self._json(browser_driver.status())
        if p == "/api/apps":
            return self._guard(api_apps)
        if p == "/api/custom_mcp":
            return self._guard(api_custom_mcp, {})
        if p == "/api/builtin-tools":
            return self._guard(api_builtin_tools)
        if p == "/api/atoms":
            return self._guard(api_atoms_list)
        if p == "/api/agent/sdk":
            return self._json(api_agent_sdk_status())
        if p == "/api/library":
            return self._json(api_library())
        if p == "/api/dl-progress":
            return self._json(DL.get((qs.get("repo") or [""])[0],
                                      {"state": "none"}))
        if p == "/api/hf-search":
            return self._json(api_hf_search((qs.get("q") or [""])[0]))
        if p == "/api/config":
            return self._json(_public_cfg())
        if p == "/api/fetch-url":
            return self._json(api_fetch_url((qs.get("url") or [""])[0]))
        if p == "/api/loaded":
            return self._json({"loaded": CHAT["model_id"]})
        if p == "/api/graph/stats":
            return self._guard(api_graph_stats)
        if p == "/api/graph/nodes":
            return self._json(api_graph_nodes((qs.get("q") or [""])[0]))
        if p == "/api/graph/node":
            return self._json(api_graph_node((qs.get("id") or [""])[0]))
        if p == "/api/claims":
            return self._guard(api_claims)
        if p == "/api/memory/list":
            return self._json(api_memory_list())
        if p == "/api/memory/sqlite":
            return self._guard(api_sqlmem_status)
        if p == "/api/memory/stats":
            return self._json(api_memory_stats())
        rel = "index.html" if p in ("/", "") else p.split("?", 1)[0].lstrip("/")
        if not rel:
            rel = "index.html"
        fp = os.path.normpath(os.path.join(WEB, rel))
        if not fp.startswith(WEB) or not os.path.isfile(fp):
            return self._json({"error": "not found"}, 404)
        ext = os.path.splitext(fp)[1]
        with open(fp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Type", MIME.get(ext, "text/plain"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_shot(self, p: str, qs: dict):
        """Screenshots for <img> tags: header token OR ?t= query token
        (img tags cannot carry headers; scope is read-only PNGs)."""
        name = os.path.basename(p)
        tok = _auth_token()
        if tok and not self._authorized() and (qs.get("t") or [""])[0] != tok:
            return self._json({"error": "invalid screenshot token"}, 401)
        t_log("/api/browser/shot", "GET")
        import browser_driver
        fp = os.path.join(browser_driver.SHOTS, name)
        if not (name.startswith("shot_") and name.endswith(".png")
                and os.path.isfile(fp)):
            return self._json({"error": "no such screenshot"}, 404)
        with open(fp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_media(self, name: str, qs: dict):
        """Generated media files for <img>/<video>/<audio> tags: header token
        OR ?t= query token, same rule as screenshots. Path resolution and the
        traversal jail live in media.api_media_file (basename-only)."""
        tok = _auth_token()
        if tok and not self._authorized() and (qs.get("t") or [""])[0] != tok:
            return self._json({"error": "invalid media token"}, 401)
        import media as _M
        r = _M.api_media_file(name)
        if "error" in r:
            return self._json(r, 404)
        try:
            with open(r["path"], "rb") as f:
                body = f.read()
        except OSError:
            return self._json({"error": "no such media file"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", r.get("mime") or "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
        except Exception:
            n = 0
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_POST(self):
        p = urlparse(self.path).path
        if p.startswith(("/api/", "/v1/")):
            if not self._authorized():
                return self._json({"error": {"message": "invalid api token",
                                             "type": "auth_error"}}, 401)
            t_log(p, "POST")
        t0 = time.perf_counter()
        b = self._body()
        if p == "/api/download":
            return self._json(api_dl_start(str(b.get("repo", ""))))
        if p == "/api/load":
            return self._guard(api_load, str(b.get("id", "")))
        if p == "/api/unload":
            return self._json(api_unload())
        if p == "/api/chat":
            return self._guard(api_chat, b.get("messages", []),
                               int(b.get("max_tokens", CFG.get("max_tokens", 64))),
                               float(b.get("temperature", CFG.get("temperature", 0.0))),
                               b.get("spec", False),
                               ground=b.get("ground"), compact=bool(b.get("compact")),
                               skill_auto=bool(b.get("skill_auto")))
        if p == "/api/exec":
            return self._json(api_exec(str(b.get("cmd", ""))))
        if p == "/api/config":
            for k in ("system", "temperature", "max_tokens", "fun"):
                if k in b:
                    CFG[k] = b[k]
            if "compact_budget_tokens" in b:
                try:
                    CFG["compact_budget_tokens"] = max(200, min(int(b["compact_budget_tokens"]), 8000))
                except (TypeError, ValueError):
                    pass
            if "compact_keep_recent" in b:
                try:
                    CFG["compact_keep_recent"] = max(1, min(int(b["compact_keep_recent"]), 32))
                except (TypeError, ValueError):
                    pass
            if isinstance(b.get("media"), dict):
                m = CFG.get("media")
                if not isinstance(m, dict):
                    CFG["media"] = m = {}
                for k in ("comfy_url", "a1111_url", "dir"):
                    if k in b["media"]:
                        m[k] = str(b["media"][k])[:300]
                if "max_output_mb" in b["media"]:
                    try:
                        m["max_output_mb"] = max(8, min(
                            int(b["media"]["max_output_mb"]), 8192))
                    except (TypeError, ValueError):
                        pass
                for k in ("tts_cmd", "stt_cmd", "ffmpeg_cmd"):
                    if isinstance(b["media"].get(k), list):
                        m[k] = [str(x)[:200] for x in b["media"][k][:12]]
                if "policy" in b["media"]:
                    import residency as _RES
                    rp = _RES.set_policy(str(b["media"]["policy"]))
                    if "error" in rp:
                        return self._json(rp, 400)
            if "agent_engine" in b:
                v = str(b["agent_engine"]).lower()
                if v not in ("builtin", "sdk", "auto"):
                    return self._json({"error": "agent_engine must be "
                                                "builtin|sdk|auto"}, 400)
                CFG["agent_engine"] = v
            if "agent_grammar" in b:
                CFG["agent_grammar"] = (
                    "off" if str(b["agent_grammar"]).lower() in ("0", "off",
                                                                "false")
                    else "on")
            if "sqlmem_enabled" in b:
                CFG["sqlmem_enabled"] = bool(b["sqlmem_enabled"])
            if "sqlmem_budget_tokens" in b:
                try:
                    CFG["sqlmem_budget_tokens"] = max(32, min(
                        int(b["sqlmem_budget_tokens"]), 2048))
                except (TypeError, ValueError):
                    pass
            if "cwd" in b:
                import sysops as _SY
                CFG["cwd"] = _SY._safe(str(b["cwd"]) or "~")[:300]
            if isinstance(b.get("cloud"), dict):
                _set_cloud_cfg(b["cloud"])
            if "full_access" in b:
                import sysops as _SY
                r = _SY.set_mode(bool(b["full_access"]), CFG, _auth_token(),
                                 AUTH_FLAG.get("host"))
                if "error" in r:
                    return self._json(r, 403)
            _save_cfg()
            return self._json(_public_cfg())
        if p == "/api/cloud-test":
            return self._json(api_cloud_test())
        if p == "/api/memory/archive":
            return self._json(api_memory_archive(b.get("thread", {})))
        if p == "/api/memory/restore":
            return self._json(api_memory_restore(str(b.get("id", ""))))
        if p == "/api/convert":
            return self._guard(api_convert, b)
        if p == "/api/inspect":
            return self._guard(api_inspect, str(b.get("path", "")))
        if p == "/api/memory/forget":
            return self._json(api_memory_forget(str(b.get("id", ""))))
        if p == "/v1/chat/completions":
            return self._guard_oai(b)
        if p == "/v1/messages":
            return self._guard_anthropic(b)
        if p == "/v1/embeddings":
            return self._guard(api_embeddings, b)
        if p == "/api/mcp/start":
            return self._guard(api_mcp_start, str(b.get("id", "")))
        if p == "/api/mcp/stop":
            return self._guard(api_mcp_stop, str(b.get("id", "")))
        if p == "/api/mcp/tools":
            return self._guard(api_mcp_tools, str(b.get("id", "")))
        if p == "/api/mcp/call":
            return self._guard(api_mcp_call, b)
        if p == "/api/skills/run":
            return self._guard(api_skills_run, b)
        if p == "/api/workflow/run":
            return self._guard(api_workflow_run, b)
        if p == "/api/memory3":
            return self._guard(api_memory3_op, b)
        if p == "/api/memory/sqlite":
            return self._guard(api_sqlmem_op, b)
        if p == "/api/fs":
            return self._guard(api_fs, b)
        if p == "/api/browser":
            return self._guard(api_browser, b)
        if p == "/api/apps":
            return self._guard(api_apps, b)
        if p == "/api/custom_mcp":
            return self._guard(api_custom_mcp, b)
        if p == "/api/atoms/run":
            return self._guard(api_atoms_run, b)
        if p == "/api/skills/save":
            return self._guard(api_skills_save, b)
        if p == "/api/skills/delete":
            return self._guard(api_skills_delete, b)
        if p == "/v1/systemone":
            return self._guard(api_systemone, b)
        if p == "/api/agent":
            return self._guard(api_agent, b)
        if p == "/api/agent/sdk/warmup":
            return self._json(api_agent_sdk_warmup(b))
        if p == "/api/rag/add":
            return self._json(api_rag_add(str(b.get("text", "")),
                                          str(b.get("source", "paste"))))
        if p == "/api/rag/clear":
            return self._json(api_rag_clear())
        if p == "/api/ground/check":
            return self._guard(api_ground_check, b)
        if p == "/api/boost":
            return self._guard(api_boost, b)
        if p == "/api/media/generate":
            import media as _M
            return self._guard(_M.api_media_generate, b)
        if p == "/api/residency":
            import residency as _RES
            if "policy" in b:
                r = _RES.set_policy(str(b.get("policy", "")))
                return self._json(r, 200 if "error" not in r else 400)
            out = _RES.resident()
            out.update(_RES.policy())
            return self._json(out)
        if p == "/api/train-draft":
            return self._guard(api_train_draft, str(b.get("id", "")))
        if p == "/api/arena/elo":
            return self._json(api_elo_vote(b))
        return self._json({"error": "unknown endpoint"}, 404)

    def _guard_anthropic(self, b: dict):
        try:
            return self._anthropic_chat(b)
        except Exception as e:
            try:
                return self._json({"type": "error",
                                   "error": {"type": "server_error",
                                             "message": str(e)[:300]}}, 500)
            except Exception:
                pass


    def _anthropic_chat(self, b: dict):
        """Anthropic Messages API: /v1/messages. Wraps api_chat; non-streaming
        (streaming Anthropic events are a documented TODO, honestly)."""
        import time as _t
        system = str(b.get("system", "") or "")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in b.get("messages", []):
            if isinstance(m, dict):
                c = m.get("content", "")
                if isinstance(c, list):
                    c = "".join(str(p.get("text", "")) for p in c
                                if isinstance(p, dict))
                msgs.append({"role": m.get("role", "user"), "content": str(c)})
        mtok = int(b.get("max_tokens", CFG.get("max_tokens", 64)))
        temp = float(b.get("temperature", CFG.get("temperature", 0.0)))
        stops = b.get("stop_sequences", [])
        if b.get("stream"):
            # Anthropic SSE: message_start → content_block_start →
            # content_block_delta* → content_block_stop → message_delta/stop
            self._sse_begin()
            mid = "msg_lumina_" + str(int(_t.time()))
            cmodel = CHAT.get("model_id") or "lumina"
            self._sse_send({"type": "message_start",
                            "message": {"id": mid, "type": "message",
                                        "role": "assistant", "model": cmodel,
                                        "content": [], "stop_reason": None}})
            self._sse_send({"type": "content_block_start", "index": 0,
                            "content_block": {"type": "text", "text": ""}})
            sent = [""]
            closed = [False]
            is_cloud = str(CHAT.get("model_id") or "").startswith("cloud/")

            def send_delta(d):
                if closed[0] or not d:
                    return
                sent[0] += d
                if not self._sse_send({"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "text_delta",
                                                 "text": d}}):
                    closed[0] = True
            if is_cloud:
                kw = dict(on_delta=send_delta)
            else:
                def on_token(tid):
                    if closed[0]:
                        return
                    try:
                        toks = on_token.acc
                    except AttributeError:
                        on_token.acc = []
                        toks = on_token.acc
                    toks.append(tid)
                    try:
                        text = CHAT["model"].tok.decode(toks)
                    except Exception:
                        return
                    d = text[len(sent[0]):]
                    if d:
                        send_delta(d)
                kw = dict(on_token=on_token)
            r = api_chat(msgs, mtok, temp, **kw)
            if "error" in r:
                self._sse_send({"type": "error",
                                "error": {"type": "server_error",
                                          "message": r["error"][:200]}})
            else:
                tail = r.get("text", "")[len(sent[0]):]
                if tail:
                    send_delta(tail)
            self._sse_send({"type": "content_block_stop", "index": 0})
            self._sse_send({"type": "message_delta",
                            "delta": {"stop_reason": "end_turn",
                                      "stop_sequence": None}})
            self._sse_send({"type": "message_stop"})
            return None
        r = api_chat(msgs, mtok, temp)
        if "error" in r:
            return self._json({"type": "error",
                               "error": {"type": "server_error",
                                         "message": r["error"][:200]}}, 500)
        text = _apply_stops(r.get("text", ""), stops)
        created = int(_t.time())
        return self._json({"id": "msg_lumina_" + str(created),
                           "type": "message", "role": "assistant",
                           "model": CHAT.get("model_id") or "lumina",
                           "content": [{"type": "text", "text": text}],
                           "stop_reason": "end_turn",
                           "usage": {"input_tokens": _count_tokens(msgs),
                                     "output_tokens": _count_tokens([{"content": text}])}})




    def _guard_oai(self, b: dict):
        """OpenAI-compatible chat endpoint (streaming SSE or full JSON)."""
        try:
            return self._oai_chat(b)
        except Exception as e:
            try:
                return self._json({"error": {"message": ("handler: " + str(e))[:300],
                                             "type": "server_error"}}, 500)
            except Exception:
                pass

    def _sse_begin(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # stream ends at EOF: HTTP/1.0 + no keep-alive so every client sees
        # the connection close after the last event (keep-alive without
        # Content-Length left readers waiting forever — found in testing)
        self.close_connection = True
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse_send(self, obj) -> bool:
        try:
            line = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(line)
            self.wfile.flush()
            return True
        except Exception:
            return False

    def _oai_chat(self, b: dict):
        import time as _t
        import numpy as _np
        model_id = str(b.get("model", "") or CHAT.get("model_id") or "")
        if model_id and model_id != CHAT.get("model_id"):
            lr = api_load(model_id)
            if "error" in lr:
                return self._json({"error": {"message": lr["error"][:200],
                                             "type": "model_not_found"}}, 404)
        if b.get("tools"):
            return self._json({"error": {"message": "tools: use /api/agent (tool loop); v1 tools land with it",
                                         "type": "invalid_request_error"}}, 400)
        stream = bool(b.get("stream", False))
        msgs = b.get("messages", [])
        temp = float(b.get("temperature", CFG.get("temperature", 0.0)))
        mtok = b.get("max_completion_tokens", b.get("max_tokens", CFG.get("max_tokens", 64)))
        samplers = {k: b[k] for k in ("top_p", "top_k", "min_p", "typical_p",
                                      "repeat_penalty", "repetition_penalty") if k in b}
        seed = b.get("seed", None)
        try:
            seed = int(seed) if seed is not None else None
        except Exception:
            seed = None
        stops = b.get("stop", [])
        if isinstance(stops, str):
            stops = [stops]
        created = int(_t.time())
        cmodel = CHAT.get("model_id") or model_id or "lumina"
        if not stream:
            r = api_chat(msgs, int(mtok), temp, samplers=samplers, seed=seed)
            if "error" in r:
                return self._json({"error": {"message": r["error"][:200],
                                             "type": "server_error"}}, 500)
            text = _apply_stops(r.get("text", ""), stops)
            if isinstance(b.get("response_format"), dict) and b["response_format"].get("type") == "json_object":
                text, ok = _ensure_json(msgs, text, temp, samplers, seed)
                if not ok:
                    return self._json({"error": {"message": "json_object requested but output failed validation after repair",
                                                 "type": "server_error"}}, 500)
            pt = _count_tokens(msgs)
            ct = _count_tokens([{"content": text}])
            return self._json({
                "id": "chatcmpl-lum-" + str(created), "object": "chat.completion",
                "created": created, "model": cmodel,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                          "total_tokens": pt + ct}})
        # streaming: real per-token SSE from the decode loop
        if CHAT.get("model") is None and not str(CHAT.get("model_id") or "").startswith("cloud/"):
            return self._json({"error": {"message": "no model loaded", "type": "server_error"}}, 500)
        if isinstance(b.get("response_format"), dict) and b["response_format"].get("type") == "json_object":
            return self._json({"error": {"message": "json_object + stream: use non-streaming for validated JSON",
                                         "type": "invalid_request_error"}}, 400)
        self._sse_begin()
        sent = [""]
        closed = [False]
        is_cloud = str(CHAT.get("model_id") or "").startswith("cloud/")

        def send_delta(d):
            if closed[0] or not d:
                return
            sent[0] += d
            if not self._sse_send({"id": "chatcmpl-lum-" + str(created),
                                   "object": "chat.completion.chunk",
                                   "created": created, "model": cmodel,
                                   "choices": [{"index": 0,
                                                "delta": {"role": "assistant",
                                                          "content": d},
                                                "finish_reason": None}]}):
                closed[0] = True

        if is_cloud:
            r = api_chat(msgs, int(mtok), temp, samplers=samplers, seed=seed,
                         on_delta=send_delta)
            if "error" in r:
                self._sse_send({"error": {"message": r["error"][:200]}})
            else:
                tail = r.get("text", "")[len(sent[0]):]
                if tail:
                    send_delta(tail)
                self._sse_send({"id": "chatcmpl-lum-" + str(created),
                                "object": "chat.completion.chunk",
                                "created": created, "model": cmodel,
                                "choices": [{"index": 0, "delta": {},
                                             "finish_reason": "stop"}]})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
            except Exception:
                pass
            return None

        def on_token(tid):
            if closed[0]:
                return
            try:
                toks = on_token.acc
            except AttributeError:
                on_token.acc = []
                toks = on_token.acc
            toks.append(tid)
            try:
                text = CHAT["model"].tok.decode(toks)
            except Exception:
                return
            d = text[len(sent[0]):]
            if d:
                sent[0] = text
                if not self._sse_send({"id": "chatcmpl-lum-" + str(created),
                                       "object": "chat.completion.chunk",
                                       "created": created, "model": cmodel,
                                       "choices": [{"index": 0,
                                                    "delta": {"role": "assistant",
                                                              "content": d},
                                                    "finish_reason": None}]}):
                    closed[0] = True
        try:
            r = api_chat(msgs, int(mtok), temp, samplers=samplers, seed=seed,
                         on_token=on_token)
        except Exception as e:
            self._sse_send({"error": {"message": str(e)[:200]}})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
            except Exception:
                pass
            return None
        if "error" in r:
            self._sse_send({"error": {"message": r["error"][:200]}})
        else:
            try:
                full = CHAT["model"].tok.decode(on_token.acc) if hasattr(on_token, "acc") else r.get("text", "")
            except Exception:
                full = r.get("text", "")
            tail = full[len(sent[0]):]
            if tail:
                self._sse_send({"id": "chatcmpl-lum-" + str(created),
                                "object": "chat.completion.chunk",
                                "created": created, "model": cmodel,
                                "choices": [{"index": 0, "delta": {"content": tail},
                                             "finish_reason": None}]})
            self._sse_send({"id": "chatcmpl-lum-" + str(created),
                            "object": "chat.completion.chunk",
                            "created": created, "model": cmodel,
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}]})
        try:
            self.wfile.write(b"data: [DONE]\n\n")
        except Exception:
            pass
        return None

    def log_message(self, *a):
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--token", default="", help="API bearer token (or LUMINA_TOKEN)")
    a = ap.parse_args(argv)
    if a.token:
        AUTH_FLAG["token"] = a.token
    try:
        from lumina_framework.cuda_opt import apply_thread_policy
        pol = apply_thread_policy()
        print(f"threads: BLAS={pol['blas_threads']} pool={pol['pool_workers']}")
    except Exception as e:
        print(f"threads: default ({e})")
    AUTH_FLAG["host"] = a.host
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Lumina Studio on http://{a.host}:{a.port}")
    print("operator mode:", _operator_mode())
    srv.serve_forever()
    return 0





# ---------------------------------------------------------------------------
# Studio: library, downloads, chat backend, config
# ---------------------------------------------------------------------------

MODELS_DIRS = [d for d in
               (os.environ.get("LUMINA_MODELS", "/tmp").split(":"))
               if os.path.isdir(d)]
MODELS_DIRS.append(os.path.join(ROOT, "artifacts"))
CHAT = {"model": None, "model_id": None, "tok": None,
        "prefix_state": None}
# one generation at a time: the fast path mutates model.t (lazy convert),
# load/unload swaps the model — concurrent chats would race those writes.
import threading as _threading
_CHAT_LOCK = _threading.Lock()
DL = {}
CFG_PATH = os.path.join(HERE, "studio_config.json")
CFG = {"system": "You are a helpful assistant.",
       "temperature": 0.0, "max_tokens": 64, "fun": False,
       "agent_engine": "builtin",
       "compact_budget_tokens": 900, "compact_keep_recent": 4,
       "agent_grammar": "on",
       "sqlmem_enabled": False, "sqlmem_budget_tokens": 320,
       "media": {"comfy_url": "", "a1111_url": "", "dir": "outputs/media",
                 "max_output_mb": 512, "tts_cmd": [], "stt_cmd": [],
                 "ffmpeg_cmd": ["ffmpeg"], "policy": "off"},
       "cloud": {"enabled": False, "provider": "custom",
                 "base_url": "", "api_key": "", "model": ""}}


def _load_cfg():
    try:
        CFG.update(json.load(open(CFG_PATH)))
    except Exception:
        pass


def _save_cfg():
    try:
        json.dump(CFG, open(CFG_PATH, "w"), indent=1)
    except Exception:
        pass


def _cloud_cfg() -> dict:
    c = CFG.get("cloud")
    return c if isinstance(c, dict) else {}


def _public_cfg() -> dict:
    out = dict(CFG)
    c = dict(_cloud_cfg())
    key = c.get("api_key", "")
    c["api_key"] = ""
    c["key_saved"] = bool(key)
    out["cloud"] = c
    try:
        out["operator"] = _operator_mode()
    except Exception:
        pass
    return out


def _set_cloud_cfg(d: dict) -> None:
    c = _cloud_cfg()
    if not isinstance(CFG.get("cloud"), dict):
        CFG["cloud"] = c = {}
    if "enabled" in d:
        c["enabled"] = bool(d["enabled"])
    if "provider" in d and str(d["provider"]) in ("custom", "openai", "openrouter"):
        c["provider"] = str(d["provider"])
    if "base_url" in d:
        u = str(d["base_url"]).strip().rstrip("/")
        if len(u) <= 200 and (u == "" or u.startswith(("http://", "https://"))):
            c["base_url"] = u
    if "model" in d:
        m = str(d["model"]).strip()
        if len(m) <= 120:
            c["model"] = m
    if "api_key" in d and str(d["api_key"]):
        # only overwrite when a new key is actually supplied
        if len(str(d["api_key"])) <= 500:
            c["api_key"] = str(d["api_key"])


def _cloud_chat(messages: list[dict], max_tokens: int, temperature: float,
                on_delta=None) -> dict:
    import time
    import urllib.request
    c = _cloud_cfg()
    base = (c.get("base_url") or "").rstrip("/")
    model = (c.get("model") or "").strip()
    key = c.get("api_key") or ""
    if not c.get("enabled"):
        return {"error": "cloud model not enabled — configure it in Settings → Cloud models"}
    if not base or not model or not key:
        return {"error": "cloud model incomplete — set base URL, model and API key first"}
    oai = [{"role": "system", "content": str(CFG.get("system") or "You are a helpful assistant.")}]
    for m in messages[-20:]:
        t = _msg_text(m)[:6000]
        if not t:
            continue
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        oai.append({"role": "assistant" if role == "assistant" else "user", "content": t})
    body = json.dumps({"model": model, "messages": oai,
                       "max_tokens": max(1, min(int(max_tokens), 1024)),
                       "temperature": max(0.0, min(float(temperature), 2.0))}).encode()
    if on_delta is None:
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key,
                                              "HTTP-Referer": "http://localhost/",
                                              "X-Title": "Lumina Studio"})
        t0 = time.perf_counter()
        try:
            raw = urllib.request.urlopen(req, timeout=90).read(200000)
            data = json.loads(raw)
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            return {"error": "cloud request failed: " + str(e)[:150]}
        wall = max(time.perf_counter() - t0, 0.01)
        toks = max(len(str(text)) // 4, 1)
        return {"text": str(text), "tokens_per_second": round(toks / wall, 1),
                "model": "cloud/" + model, "wall_s": round(wall, 1)}
    # streaming: parse upstream SSE and forward text deltas as they land
    sbody = dict(json.loads(body.decode() if isinstance(body, bytes) else body))
    sbody["stream"] = True
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(sbody).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key,
                                          "HTTP-Referer": "http://localhost/",
                                          "X-Title": "Lumina Studio"})
    t0 = time.perf_counter()
    text_parts = []
    try:
        resp = urllib.request.urlopen(req, timeout=90)
        for raw_line in resp:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
                d_txt = (d.get("choices") or [{}])[0].get("delta", {}).get("content", "")
                if d_txt:
                    text_parts.append(d_txt)
                    try:
                        on_delta(d_txt)
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception as e:
        if not text_parts:
            return {"error": "cloud request failed: " + str(e)[:150]}
    text = "".join(text_parts)
    wall = max(time.perf_counter() - t0, 0.01)
    toks = max(len(text) // 4, 1)
    return {"text": text, "tokens_per_second": round(toks / wall, 1),
            "model": "cloud/" + model, "wall_s": round(wall, 1)}


def api_cloud_test() -> dict:
    r = _cloud_chat([{"role": "user", "content": "Reply with exactly: ok"}], 8, 0.0)
    if "error" in r:
        return {"ok": False, "error": r["error"]}
    return {"ok": True, "text": str(r.get("text", ""))[:200]}


_load_cfg()


_LIB_CACHE = {"t": 0.0, "data": None}

def _lib_invalidate() -> None:
    _LIB_CACHE["t"] = 0.0
    _LIB_CACHE["data"] = None

def api_library() -> dict:
    import time as _t
    if _LIB_CACHE["data"] is not None and _t.time() - _LIB_CACHE["t"] < 60:
        return _LIB_CACHE["data"]
    from lumina_framework.adapters import detect_format
    items = []
    seen = set()
    SKIP_DIRS = {".cache", "__pycache__", ".git", "node_modules"}
    for d in MODELS_DIRS:
        if not os.path.isdir(d):
            continue
        dir_sizes: dict[str, int] = {}   # root-of-hf-dir -> accumulated bytes
        # single pass: one stat per file, dir sizes accumulated inline
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    fsize = os.path.getsize(fp)
                except OSError:
                    continue
                if fp in seen:
                    continue
                seen.add(fp)
                if fn == "config.json":
                    items.append({"id": os.path.basename(root), "path": root,
                                  "format": "hf-dir",
                                  "size_mb": None})  # filled after the walk
                    dir_sizes[root] = 0
                elif fn.endswith((".safetensors", ".gguf", ".lumina", ".exl3")):
                    try:
                        fmt = detect_format(fp) or "unknown"
                    except Exception:
                        fmt = "unknown"
                    items.append({"id": fn, "path": fp, "format": fmt,
                                  "size_mb": round(fsize / 1e6, 1)})
            if len(items) > 200:
                break
        # hf-dir sizes from a single targeted walk each (config dirs are few)
        for it in items:
            if it["format"] == "hf-dir" and it["size_mb"] is None:
                tot = 0
                for root2, dirs2, files2 in os.walk(it["path"]):
                    dirs2[:] = [x for x in dirs2 if x not in SKIP_DIRS]
                    for fn2 in files2:
                        try:
                            tot += os.path.getsize(os.path.join(root2, fn2))
                        except OSError:
                            pass
                it["size_mb"] = round(tot / 1e6, 1)
    c = _cloud_cfg()
    if c.get("enabled") and (c.get("model") or "").strip():
        items.insert(0, {"id": "cloud/" + c["model"].strip(), "path": "cloud",
                         "format": "cloud", "size_mb": 0})
    out = {"items": items, "dirs": MODELS_DIRS}
    _LIB_CACHE["data"] = out
    _LIB_CACHE["t"] = __import__("time").time()
    return out


def _dir_mb(d: str) -> float:
    tot = 0
    for root, _, files in os.walk(d):
        for fn in files:
            try:
                tot += os.path.getsize(os.path.join(root, fn))
            except Exception:
                pass
    return tot / 1e6


def api_dl_start(repo: str) -> dict:
    import threading
    from huggingface_hub import snapshot_download
    dest = os.path.join(MODELS_DIRS[0], repo.split("/")[-1])
    DL[repo] = {"state": "queued", "pct": 0.0}
    threading.Thread(target=_dl_run, args=(repo, dest), daemon=True).start()
    return {"repo": repo, "dest": dest, "state": "queued"}


def _dl_run(repo: str, dest: str) -> None:
    from huggingface_hub import snapshot_download
    DL[repo] = {"state": "downloading", "pct": 0.0}
    try:
        snapshot_download(repo, local_dir=dest)
        DL[repo] = {"state": "done", "pct": 100.0, "dest": dest}
        _lib_invalidate()
    except Exception as e:
        DL[repo] = {"state": "error", "error": str(e)[:200]}


def api_hf_search(q: str) -> dict:
    import urllib.request
    url = "https://huggingface.co/api/models?search=" + urllib.parse.quote(q) + "&limit=12"
    try:
        data = json.load(urllib.request.urlopen(url, timeout=20))
    except Exception as e:
        return {"items": [], "error": str(e)[:150]}
    return {"items": [{"id": m.get("modelId"), "likes": m.get("likes", 0),
                       "downloads": m.get("downloads", 0),
                       "tags": (m.get("tags", []) or [])[:4]} for m in data]}


def _resolve_model_dir(model_id: str) -> str | None:
    """Fast path probe: find the HF dir for a library id without a full scan."""
    if model_id.startswith("cloud/"):
        return None
    import glob as _g
    for d in MODELS_DIRS:
        cand = os.path.join(d, model_id)
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            return cand
        # one level deep (e.g. /tmp/opencode/chatfix)
        for hit in _g.glob(os.path.join(d, "*", model_id)):
            if os.path.isdir(hit) and os.path.isfile(os.path.join(hit, "config.json")):
                return hit
        cand = os.path.join(d, model_id + ".gguf")
        if os.path.isfile(cand):
            return os.path.dirname(cand)
        cand = os.path.join(d, model_id)
        if os.path.isfile(cand):
            return os.path.dirname(cand)
    return None

def _mem_available_mb() -> int | None:
    """Free RAM in MB, or None if unreadable."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _resolve_lumina_file(model_id: str) -> str | None:
    """Find a .lumina file for a library id (bare name or full path)."""
    cand = model_id if os.path.isfile(model_id) else None
    if cand is None:
        for d in MODELS_DIRS:
            for root, dirs, files in os.walk(d):
                dirs[:] = [x for x in dirs
                           if x not in (".cache", "__pycache__", ".git", "node_modules")]
                if os.path.basename(model_id) in files:
                    cand = os.path.join(root, os.path.basename(model_id))
                    break
                for fn in files:
                    if fn == model_id or fn.rsplit(".", 1)[0] == model_id:
                        if fn.endswith(".lumina"):
                            cand = os.path.join(root, fn)
                            break
                if cand:
                    break
            if cand:
                break
    if cand and cand.endswith(".lumina") and os.path.isfile(cand):
        return cand
    return None


def _lumina_layer_count(lum_path: str) -> int | None:
    """Layers evident from the tensor names inside the .lumina header."""
    import re as _re
    import struct as _st
    try:
        with open(lum_path, "rb") as f:
            _m, _v, hlen = _st.unpack("<IIQ", f.read(16))
            hdr = json.loads(f.read(hlen))
        layers = set()
        for name in hdr.get("tensors", {}):
            m = _re.search(r"(?:model\.)?layers\.(\d+)\.", name)
            if m:
                layers.add(int(m.group(1)))
        return max(layers) + 1 if layers else None
    except Exception:
        return None


def _lumina_recorded_source(lum_path: str) -> str | None:
    try:
        import struct as _st
        with open(lum_path, "rb") as f:
            _m, _v, hlen = _st.unpack("<IIQ", f.read(16))
            hdr = json.loads(f.read(hlen))
        src = hdr.get("source_dir")
        if src and os.path.isfile(os.path.join(src, "config.json")):
            return src
    except Exception:
        pass
    return None


def _st_tensor_names(st_path: str) -> set[str] | None:
    """Tensor names from a safetensors header (header bytes only, cheap)."""
    import struct as _st
    try:
        with open(st_path, "rb") as f:
            (n,) = _st.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        return {k for k in hdr if k != "__metadata__"}
    except Exception:
        return None


def _resolve_tokenizer_dir(lumina_path: str) -> str | None:
    """The HF dir holding config.json + tokenizer for a converted file.

    1. Provenance recorded at convert time (header "source_dir").
    2. Otherwise the dir whose safetensors tensor NAME SET exactly matches
       the names inside the file, with config layers agreeing and a real
       tokenizer present (vocab.json + merges.txt).
    3. Otherwise None (honest error) — never a blind alphabetical pick.
    """
    rec = _lumina_recorded_source(lumina_path)
    if rec is not None:
        return rec
    import glob as _g
    import struct as _st
    try:
        with open(lumina_path, "rb") as f:
            _m, _v, hlen = _st.unpack("<IIQ", f.read(16))
            hdr = json.loads(f.read(hlen))
        want_names = set(hdr.get("tensors", {}))
    except Exception:
        return None
    if not want_names:
        return None
    try:
        want_layers = _lumina_layer_count(lumina_path)
    except Exception:
        want_layers = None
    d = os.path.dirname(os.path.abspath(lumina_path))
    stem = os.path.basename(lumina_path).rsplit(".", 1)[0]
    cands = [d, os.path.join(d, stem)]
    try:
        for entry in sorted(os.listdir(d)):
            cand = os.path.join(d, entry)
            if os.path.isdir(cand) and cand not in cands:
                cands.append(cand)
    except OSError:
        pass
    for cand in cands:
        if not os.path.isfile(os.path.join(cand, "config.json")):
            continue
        if not (os.path.isfile(os.path.join(cand, "vocab.json"))
                and os.path.isfile(os.path.join(cand, "merges.txt"))):
            continue
        st_files = [os.path.join(cand, "model.safetensors")]
        st_files += sorted(_g.glob(os.path.join(cand, "model-*.safetensors")))
        have: set[str] = set()
        for stf in st_files:
            names = _st_tensor_names(stf)
            if names:
                have |= names
        if have != want_names:
            continue
        if want_layers is not None:
            try:
                cfg = json.load(open(os.path.join(cand, "config.json")))
            except Exception:
                continue
            if int(cfg.get("num_hidden_layers", -1)) != want_layers:
                continue
        return cand
    return None


def _api_load_lumina(model_id: str, lum_path: str) -> dict:
    """Load a converted .lumina file for chat: weights dequantize through the
    format adapter into the framework contract, config + tokenizer come from
    the original HF dir. Same ChatModel math as dense — the format only
    changes how many bits survived."""
    from lumina_framework.adapters import detect_format, load_model
    from lumina_framework.chat import ChatModel
    from lumina_framework.tokenizer import BPETokenizer
    if detect_format(lum_path) != "lumina":
        return {"error": "not a lumina file: " + os.path.basename(lum_path)}
    base = _resolve_tokenizer_dir(lum_path)
    if base is None:
        return {"error": ("quantized chat needs the original HF dir "
                           "(config.json + tokenizer) next to the .lumina file")}
    try:
        cfg = json.load(open(os.path.join(base, "config.json")))
    except Exception as e:
        return {"error": "unreadable config.json: " + str(e)[:120]}
    try:
        tok = BPETokenizer(base)
    except Exception as e:
        return {"error": "unreadable tokenizer: " + str(e)[:120]}
    try:
        m = load_model(lum_path)
    except Exception as e:
        return {"error": "lumina load failed: " + str(e)[:150]}
    if m.spec.arch == "qwen4_exp":
        from lumina_framework.adapters.qwen4_exp import QWEN4_UNSUPPORTED
        return {"error": ("quantized Qwen3.8-Flash-Next execution needs: "
                          + ", ".join(QWEN4_UNSUPPORTED)
                          + ". Refusing instead of decoding garbage.")}
    n_params = sum(int(v.size) for v in m.tensors.tensors.values())
    working_mb = n_params * 2 / 1e6  # fp16 working copy
    avail = _mem_available_mb()
    if avail is not None and working_mb * 2.5 > avail:
        return {"error": (
            f"not enough free memory: dequantized working set is about "
            f"{working_mb * 2.5:.0f}MB, only {avail}MB is free. Load a "
            "smaller model.")}
    api_unload()
    moe_map: dict = {}
    try:
        from lumina_framework.adapters.lumina import lumina_header
        import re as _re
        for pfx, mm in (lumina_header(lum_path).get("virtual_moe") or {}).items():
            mt = _re.search(r"layers\.(\d+)\.", str(pfx))
            if mt:
                moe_map[int(mt.group(1))] = mm
    except Exception:
        moe_map = {}
    try:
        CHAT["model"] = ChatModel.from_tensors(
            {k: v for k, v in m.tensors.tensors.items()}, cfg, tok,
            moe_map=moe_map or None)
        CHAT["prefix_state"] = None
    except Exception as e:
        return {"error": "quantized model build failed: " + str(e)[:150]}
    CHAT["model_id"] = model_id
    wire_bpp = round(8.0 * os.path.getsize(lum_path) / max(n_params, 1), 3)
    return {"loaded": model_id, "params_m": round(n_params / 1e6, 1),
            "layers": CHAT["model"].n_layer, "format": "lumina",
            "wire_bpp": wire_bpp, "tokenizer_from": base,
            "virtual_moe_layers": sorted(CHAT["model"].moe.keys())}


def api_load(model_id: str) -> dict:
    if model_id.startswith("cloud/"):
        c = _cloud_cfg()
        if not c.get("enabled") or not (c.get("model") or "").strip():
            return {"error": "cloud model not configured — see Settings → Cloud models"}
        api_unload()
        CHAT["model"] = None
        CHAT["model_id"] = model_id
        return {"loaded": model_id, "cloud": True}
    from lumina_framework.chat import ChatModel
    from lumina_framework.adapters import detect_format as _df
    _probe = None
    try:
        _probe = _df(model_id) or _df(os.path.join(model_id, "model.safetensors"))
    except Exception:
        pass
    if _probe == "qwen4_exp":
        from lumina_framework.adapters.qwen4_exp import QWEN4_UNSUPPORTED
        return {"error": ("Qwen3.8-Flash-Next execution needs: "
                          + ", ".join(QWEN4_UNSUPPORTED)
                          + ". No verified kernels here yet — refusing instead of "
                            "decoding garbage. See /api/hardware + STATUS.md for the recipe.")}
    lum_path = None
    if model_id.endswith(".lumina") or os.path.isfile(model_id):
        lum_path = _resolve_lumina_file(model_id)
    if lum_path is not None:
        return _api_load_lumina(model_id, lum_path)
    mdir = _resolve_model_dir(model_id)
    hit = None
    if mdir is None:
        lib = api_library()["items"]
        hit = next((x for x in lib if x["id"] == model_id or x["path"] == model_id), None)
        if hit is None:
            return {"error": "not in library: " + model_id}
        if str(hit["path"]).endswith(".lumina"):
            lum2 = _resolve_lumina_file(hit["path"])
            if lum2 is not None:
                return _api_load_lumina(model_id, lum2)
            return {"error": "lumina file not readable: " + str(hit["path"])[:120]}
        mdir = hit["path"]
        if not os.path.isdir(mdir):
            mdir = os.path.dirname(mdir)
    path = mdir
    if not os.path.isfile(os.path.join(mdir, "config.json")):
        return {"error": "chat needs an HF dir (config.json); convert others first"}
    # memory guard: refuse loads that would OOM the device (honest, adaptive).
    model_mb = _dir_mb(mdir)
    avail = _mem_available_mb()
    if avail is not None and model_mb * 2.5 > avail:
        return {"error": (
            f"not enough free memory: this model needs about {model_mb * 2.5:.0f}MB "
            f"to load safely, only {avail}MB is free. Close other apps/tabs, or "
            "load a smaller or quantized model.")}
    api_unload()
    with _CHAT_LOCK:
        CHAT["model"] = ChatModel(mdir)
        CHAT["prefix_state"] = None
    CHAT["model_id"] = model_id
    npar = sum(v.size for v in CHAT["model"].t.values())
    tps = _quick_bench(model_id)
    if tps:
        bench = _bench_load()
        bench[model_id] = tps
        _bench_save(bench)
    return {"loaded": model_id, "params_m": round(npar / 1e6, 1),
            "layers": CHAT["model"].n_layer,
            "measured_tps": tps}


def api_unload() -> dict:
    with _CHAT_LOCK:
        CHAT["model"] = None
        CHAT["model_id"] = None
        CHAT["prefix_state"] = None
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    return {"unloaded": True}


def _msg_text(m: dict) -> str:
    if not isinstance(m, dict):
        return ""
    t = m.get("content", None)
    if isinstance(t, list):
        # OpenAI parts array: [{type: "text", text: ...}, ...]
        return "".join(str(p.get("text", "")) for p in t
                       if isinstance(p, dict) and p.get("type") == "text")
    if t:
        return str(t)
    return str(m.get("text", "") or "")


def _chatml(messages: list[dict], tok) -> str:
    """Fallback prompt (plain ChatML) when a model ships no template."""
    parts = []
    for m in messages:
        parts.append("<|im_start|>" + (m.get("role", "user") if isinstance(m, dict) else "user") + "\n" + _msg_text(m) + "<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


_TEMPLATE_CACHE: dict = {}
RENDER_LOG: dict = {"origin": ""}


def _model_template(model_dir: str) -> tuple[str | None, str]:
    """Find a chat template: embedded chat_template, chat_template.jinja
    file, else none. Returns (source, origin). Cached per dir."""
    if model_dir in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[model_dir]
    out: tuple[str | None, str] = (None, "chatml-fallback")
    try:
        cj = os.path.join(model_dir, "tokenizer_config.json")
        cfg = json.load(open(cj))
        if cfg.get("chat_template"):
            out = (cfg["chat_template"], "tokenizer_config.json:chat_template")
        else:
            jf = os.path.join(model_dir, "chat_template.jinja")
            if os.path.isfile(jf):
                out = (open(jf).read(), "chat_template.jinja")
    except Exception:
        pass
    _TEMPLATE_CACHE[model_dir] = out
    return out


def render_prompt(messages: list[dict], tok) -> str:
    """Model-aware prompt rendering: the model's own Jinja template when it
    ships one (stdlib renderer, byte-identical to Jinja2 on Qwen2.5's real
    template), plain ChatML fallback otherwise. Adds generation prompt."""
    model_dir = getattr(tok, "dir", "")
    src, origin = _model_template(model_dir) if model_dir else (None, "")
    RENDER_LOG["origin"] = origin
    if src:
        try:
            from lumina_framework.chat_template import render
            RENDER_LOG["origin"] = origin
            norm = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                d = {"role": m.get("role", "user")}
                t = _msg_text(m)
                d["content"] = t
                if isinstance(m.get("tool_calls"), list):
                    d["tool_calls"] = m["tool_calls"]
                norm.append(d)
            return render(src, {"messages": norm, "tools": None,
                                "add_generation_prompt": True,
                                "bos_token": "", "eos_token": ""})
        except Exception:
            pass
    return _chatml(messages, tok)


CACHE_METER: dict = {"turns": 0, "reused": 0, "computed": 0,
                     "compactions": 0, "last": None}


def api_cache_stats() -> dict:
    """Honest prefix-cache accounting: counters come straight from the real
    generate_cached prefill counters (no estimation)."""
    reused = int(CACHE_METER["reused"])
    computed = int(CACHE_METER["computed"])
    total = reused + computed
    return {"turns": int(CACHE_METER["turns"]), "reused": reused,
            "computed": computed,
            "hit_rate": round(reused / total, 4) if total else 0.0,
            "compactions": int(CACHE_METER["compactions"]),
            "last": CACHE_METER["last"], "target": 0.90,
            "note": "prefill counters reported by the real prefix KV cache; "
                    "hit_rate = reused / (reused + computed) across metered turns"}


FUN_LINES = [
    "Lumina here. Zero training wheels, all signal.",
    "Quantized down to the good bits.",
    "Ask me anything. Politely daring answers included.",
]


def api_chat(messages: list[dict], max_tokens: int, temperature: float,
           spec=False, samplers: dict | None = None,
           on_token=None, seed: int | None = None,
           on_delta=None,
           ground: dict | None = None, compact: bool = False,
           skill_auto: bool = False) -> dict:
    import time
    samplers = samplers or {}
    # Opt-in auto-compaction: collapse the history front into one checkpoint so
    # the prefix cache can keep hitting (measured in test_context_compact.py;
    # steady-state reuse gate >= 0.90 on the scripted workload).
    if compact:
        import context_compact as _CC
        _c = _CC.compact_if_needed(
            messages,
            budget_tokens=int(CFG.get("compact_budget_tokens", 900)),
            keep_recent=int(CFG.get("compact_keep_recent", 4)))
        if _c["compacted"]:
            CACHE_METER["compactions"] += 1
        messages = _c["messages"]
    # Opt-in auto skill fetch: one calibrated BM25 query against the skill
    # index; the winning skill's instructions are injected only when the
    # index is confident (margin gate), never guessed. Bounded to 1200
    # chars so a small model's context is not flooded.
    _skill_note = None
    if skill_auto:
        _sq = ""
        for _m in reversed(messages):
            if isinstance(_m, dict) and _m.get("role") == "user":
                _sq = _msg_text(_m) or ""
                break
        if _sq.strip():
            _sr = api_skills_search(_sq, 3)
            if "error" not in _sr:
                _hits = _sr.get("skills") or []
                _top = _hits[0] if _hits else None
                _skill_note = {"confident": bool(_sr.get("confident")),
                               "reason": _sr.get("reason"),
                               "id": _top.get("id") if _top else None,
                               "score": round(_top["score"], 3) if _top else None,
                               "candidates": [h["id"] for h in _hits[:3]]}
                if _sr.get("confident") and _top:
                    _sk = next((x for x in _all_skills()
                                if x.get("id") == _top["id"]), None)
                    if _sk:
                        messages = [{"role": "system", "content":
                                     "[auto skill: " + str(_sk.get("id")) + "]\n"
                                     + str(_sk.get("prompt", ""))[:1200]}] + messages
                        _skill_note["injected"] = True
    # Opt-in evidence grounding: retrieve once, then gate BEFORE any model
    # call — a question with no supporting evidence is refused outright
    # instead of being answered from thin air.
    _G = None
    ground_ctx = None
    if ground:
        import grounding as _G
        _q = ""
        for _m in reversed(messages):
            if isinstance(_m, dict) and _m.get("role") == "user":
                _q = _msg_text(_m) or ""
                break
        _k = max(1, min(int((ground or {}).get("k", 3)), 10))
        _ps = _rag_score(_q, _k) if _q.strip() else []
        _gd = _G.gate_decision(_ps)
        _policy = ("withhold" if str((ground or {}).get("policy", "flag")) == "withhold"
                   else "flag")
        ground_ctx = {"passages": _ps, "policy": _policy, "gate": _gd}
        if _gd["refuse"]:
            return {"text": _G.REFUSAL_TEXT, "tokens": [], "refused": True,
                    "model": CHAT.get("model_id"),
                    "grounding": {"gate": "refuse", "reason": _gd["reason"],
                                  "passages": 0, "claims": [], "total": 0,
                                  "unsupported": 0, "coverage": 1.0,
                                  "policy": _policy}}
        messages = _G.evidence_messages(messages, _ps)
    if (CHAT.get("model_id") or "").startswith("cloud/"):
        r = _cloud_chat(messages, max_tokens, temperature, on_delta=on_delta)
        if ground_ctx is not None and isinstance(r.get("text"), str):
            _gr = _G.ground_answer(r["text"], ground_ctx["passages"],
                                   policy=ground_ctx["policy"])
            r["grounding"] = _gr
            if ground_ctx["policy"] == "withhold" and _gr["unsupported"]:
                r["answer_original"] = r["text"]
                r["text"] = _gr["answer"]
        if _skill_note is not None:
            r["skill_auto"] = _skill_note
        if CFG.get("fun") and "text" in r:
            import random
            r["flair"] = random.choice(FUN_LINES)
        return r
    if CHAT["model"] is None:
        return {"error": "no model loaded — load one in My Models first"}
    from lumina_framework.chat import generate
    import numpy as np
    model = CHAT["model"]
    # learn stable facts from this turn (conservative patterns, no model call)
    for _m in reversed(messages):
        if isinstance(_m, dict) and _m.get("role") == "user":
            for _fact in api_memory3_autoextract(_msg_text(_m) or ""):
                _mem3_remember("core", _fact, False)
                if CFG.get("sqlmem_enabled"):
                    _sm2 = _sqlmem()
                    if _sm2 is not None:
                        try:
                            _sm2.add("core", "fact", _fact)
                        except Exception:  # noqa: BLE001
                            pass
            break
    mem_block = _mem3_context_block()
    if mem_block:
        msgs2 = list(messages)
        if msgs2 and msgs2[0].get("role") == "system":
            msgs2[0] = dict(msgs2[0], content=str(msgs2[0].get("content", "")) + "\n" + mem_block)
        else:
            msgs2.insert(0, {"role": "system", "content": mem_block})
        messages = msgs2
    # SQLite FTS memory tier (Atomic-Agent-inspired): recall is injected in the
    # TAIL so the byte-stable prefix keeps its KV cache (the JSON tier above
    # prepends, which is why the two tiers behave differently).
    if CFG.get("sqlmem_enabled"):
        _sm = _sqlmem()
        if _sm is not None:
            _sq = ""
            for _m3 in reversed(messages):
                if isinstance(_m3, dict) and _m3.get("role") == "user":
                    _sq = _msg_text(_m3) or ""
                    break
            if _sq.strip():
                try:
                    _tb = _sm.tail_block(
                        _sq, limit=6,
                        budget_tokens=int(CFG.get("sqlmem_budget_tokens", 320)))
                except Exception:
                    _tb = {"block": ""}
                if _tb.get("block"):
                    messages = messages + [{"role": "system",
                                            "content": str(_tb["block"])}]
                    _SQLMEM_STATS["injections"] += 1
                    _SQLMEM_STATS["tokens"] += int(_tb.get("injected_tokens") or 0)
                    r_sql = {"sqlmem": {"engine": _tb.get("engine"),
                                        "injected_tokens": _tb.get("injected_tokens")}}
                else:
                    r_sql = {"sqlmem": {"engine": _tb.get("engine"),
                                        "injected_tokens": 0}}
                _SQLMEM_LAST["ctx"] = r_sql
    prompt = render_prompt(messages, model.tok)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    mode = str(spec) if isinstance(spec, str) else ("pld" if spec else "")
    use_spec = bool(mode) and float(temperature) <= 0
    gen_kw = dict(max_tokens=max(1, min(int(max_tokens), 512)),
                  budget_s=120.0, temperature=float(temperature),
                  top_p=float(samplers.get("top_p", 1.0)),
                  top_k=int(samplers.get("top_k", 0)),
                  min_p=float(samplers.get("min_p", 0.0)),
                  typical_p=float(samplers.get("typical_p", 1.0)),
                  repeat_penalty=float(samplers.get("repeat_penalty", samplers.get("repetition_penalty", 1.0))),
                  rng=rng, on_token=on_token)
    with _CHAT_LOCK:
        if use_spec and mode == "eagle":
            r = _api_chat_eagle(model, prompt, gen_kw["max_tokens"])
        elif use_spec and mode == "mtp":
            r = _api_chat_mtp(model, prompt, gen_kw["max_tokens"])
        elif use_spec:
            from lumina_framework.chat import generate_spec
            r = generate_spec(model, prompt, max_tokens=gen_kw["max_tokens"],
                              budget_s=120.0, rng=rng)
            r["speculative"] = "pld"
        else:
            from lumina_framework.chat import generate_cached
            if CHAT.get("prefix_state") is None:
                CHAT["prefix_state"] = {"ids": [], "kv": {}, "pos": 0}
            gkw = dict(gen_kw)
            state = gkw.pop("state", None)
            r = generate_cached(model, prompt, state=CHAT["prefix_state"], **gkw)
            CHAT["prefix_state"] = r["cache_state"]
            del r["cache_state"]
    r["model"] = CHAT["model_id"]
    r["wall_s"] = round(time.perf_counter() - t0, 1)
    r["prompt_template"] = RENDER_LOG.get("origin") or "chatml-fallback"
    if isinstance(r.get("stats"), dict) and "prefill_reused" in r["stats"]:
        _ru = int(r["stats"]["prefill_reused"])
        _co = int(r["stats"]["prefill_computed"])
        r["prefill_reused"] = _ru
        r["prefill_computed"] = _co
        CACHE_METER["turns"] += 1
        CACHE_METER["reused"] += _ru
        CACHE_METER["computed"] += _co
        CACHE_METER["last"] = {"reused": _ru, "computed": _co,
                               "hit_rate": round(_ru / (_ru + _co), 4)
                               if (_ru + _co) else 0.0,
                               "ts": round(time.time(), 3)}
    if ground_ctx is not None and isinstance(r.get("text"), str):
        _gr = _G.ground_answer(r["text"], ground_ctx["passages"],
                               policy=ground_ctx["policy"])
        r["grounding"] = _gr
        if ground_ctx["policy"] == "withhold" and _gr["unsupported"]:
            r["answer_original"] = r["text"]
            r["text"] = _gr["answer"]
    if _skill_note is not None:
        r["skill_auto"] = _skill_note
    if CFG.get("fun"):
        import random
        r["flair"] = random.choice(FUN_LINES)
    return r


def api_oai_models() -> dict:
    """OpenAI-style model list: loaded first, then library, then cloud."""
    data = []
    if CHAT.get("model_id"):
        data.append({"id": CHAT["model_id"], "object": "model",
                     "owned_by": "lumina", "loaded": True})
    try:
        lib = api_library()["items"]
    except Exception:
        lib = []
    for m in lib:
        if m["id"] == CHAT.get("model_id"):
            continue
        data.append({"id": m["id"], "object": "model", "owned_by": "lumina",
                     "format": m.get("format"), "size_mb": m.get("size_mb")})
    return {"object": "list", "data": data}


def _count_tokens(messages: list[dict]) -> int:
    return sum(len(_msg_text(m).split()) + 4 for m in messages)


def _apply_stops(text: str, stops: list) -> str:
    for s in stops or []:
        s = str(s)
        if s and s in text:
            text = text.split(s)[0]
    return text


def _ensure_json(messages: list[dict], text: str, temp: float,
                 samplers: dict, seed) -> tuple[str, bool]:
    """response_format=json_object: validate; one repair retry; honest flag."""
    try:
        json.loads(text)
        return text, True
    except Exception:
        pass
    fixed = messages + [{"role": "system",
                         "content": "Your previous reply was not valid JSON. "
                                    "Reply with ONLY a valid JSON object, no prose."}]
    r = api_chat(fixed, 128, temp, samplers=samplers, seed=seed)
    if "error" in r:
        return text, False
    try:
        json.loads(r.get("text", ""))
        return r["text"], True
    except Exception:
        return text, False


def api_fetch_url(url: str) -> dict:
    import re
    import urllib.request
    if not url.startswith(("http://", "https://")):
        return {"error": "http(s) only"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Lumina/0.1"})
        raw = urllib.request.urlopen(req, timeout=20).read(120000).decode("utf-8", "ignore")
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return {"text": text[:6000], "chars": len(text)}
    except Exception as e:
        return {"error": str(e)[:150]}


# Read-only introspection ONLY. python3 was removed: `python3 -c "exec(...)"`
# bypassed every chaining filter (no ;|$`><\ needed) = arbitrary code exec.
# If scripting is ever needed, run it outside this server.
EXEC_ALLOW = {"ls", "cat", "pwd", "echo", "uname", "df", "du",
              "head", "tail", "wc", "lsb_release", "whoami", "date"}


def _operator_mode() -> dict:
    """Operator-mode summary (loopback toggle free, LAN needs a token)."""
    import sysops
    return sysops.mode(CFG, _auth_token(), AUTH_FLAG.get("host"))


def api_exec(cmd: str, cwd: str | None = None, timeout: int = 15) -> dict:
    """Read-only allowlist by default; ANY command in operator mode (audited).

    Operator mode is the user's explicit switch — see sysops.mode(). When it
    is on, exec runs the real shell the way the user's own terminal would,
    and every invocation lands in ops_audit.jsonl.
    """
    import shlex
    import subprocess
    if _operator_mode().get("full_access"):
        import sysops
        return sysops.run_any(cmd, cwd=cwd or CFG.get("cwd"),
                              timeout=max(int(timeout), 1))
    try:
        parts = shlex.split(cmd.strip())
    except Exception:
        return {"error": "cannot parse"}
    if not parts or parts[0] not in EXEC_ALLOW:
        return {"error": "not allowlisted: " + (parts[0] if parts else "?")}
    if any(c in cmd for c in [";", "&", "|", "$", "`", ">", "<", "\\", "\n"]):
        return {"error": "no chaining/redirection"}
    try:
        p = subprocess.run(parts[:12], capture_output=True, text=True,
                           timeout=15, cwd="/tmp")
        return {"out": (p.stdout + p.stderr)[:4000], "code": p.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "timeout after 15s"}
    except Exception as e:
        return {"error": str(e)[:150]}




# ---------------------------------------------------------------------------
# Code graph (graphify-inbuilt): static import/reference graph of the repo
# ---------------------------------------------------------------------------

def _build_code_graph() -> dict:
    """Nodes = source files, edges = import/reference relations.
    Static, deterministic, no external services."""
    import re as _re
    roots = [os.path.join(ROOT, "src"), os.path.join(ROOT, "app"),
             os.path.join(ROOT, "scripts")]
    nodes, edges = {}, []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if not fn.endswith((".py", ".js", ".html", ".css")):
                    continue
                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, ROOT)
                try:
                    txt = open(fp, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                kind = {"py": "python", "js": "js", "html": "html", "css": "css"}[fn.rsplit(".", 1)[1]]
                nodes[rel] = {"id": rel, "kind": kind,
                              "lines": txt.count("\n") + 1,
                              "bytes": len(txt.encode("utf-8", "ignore"))}
    ids = set(nodes)
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if not fn.endswith((".py", ".js")):
                    continue
                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, ROOT)
                try:
                    txt = open(fp, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                src_dir = os.path.dirname(rel)
                for m in _re.finditer(r'(?:from|import)\s+([\\w.]+)', txt):
                    mod = m.group(1).split(".")[0]
                    cand = os.path.normpath(os.path.join(src_dir, mod + ".py"))
                    cand2 = os.path.normpath(os.path.join(src_dir, mod, "__init__.py"))
                    for c in (cand, cand2):
                        if c in ids and c != rel:
                            edges.append({"src": rel, "dst": c, "kind": "import"})
                for m in _re.finditer(r'lumina_([a-z_]+)', txt):
                    c = os.path.normpath(os.path.join("src", "lumina_framework", "lumina_" + m.group(1) + ".py"))
                    if c in ids and c != rel:
                        edges.append({"src": rel, "dst": c, "kind": "import"})
                if fn.endswith(".js"):
                    for m in _re.finditer(r'[\'"]([\\w./-]+\\.(?:js|css))[\'"]', txt):
                        c = os.path.normpath(os.path.join(src_dir, m.group(1)))
                        if c in ids and c != rel:
                            edges.append({"src": rel, "dst": c, "kind": "load"})
    # dedupe edges
    seen = set()
    uniq = []
    for e in edges:
        k = (e["src"], e["dst"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return {"nodes": nodes, "edges": uniq}


_GRAPH = {"built": False, "data": {"nodes": {}, "edges": []}}


def _graph() -> dict:
    if not _GRAPH["built"]:
        try:
            _GRAPH["data"] = _build_code_graph()
        except Exception:
            _GRAPH["data"] = {"nodes": {}, "edges": []}
        _GRAPH["built"] = True
    return _GRAPH["data"]


def api_graph_stats() -> dict:
    g = _graph()
    kinds = {}
    for n in g["nodes"].values():
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    deg = {}
    for e in g["edges"]:
        deg[e["src"]] = deg.get(e["src"], 0) + 1
        deg[e["dst"]] = deg.get(e["dst"], 0) + 1
    hubs = sorted(deg.items(), key=lambda kv: -kv[1])[:8]
    return {"nodes": len(g["nodes"]), "edges": len(g["edges"]),
            "by_kind": kinds, "hubs": [{"id": k, "degree": v} for k, v in hubs]}


def api_graph_nodes(q: str) -> dict:
    g = _graph()
    q = (q or "").lower()
    hits = [n for n in g["nodes"].values()
            if not q or q in n["id"].lower()]
    hits.sort(key=lambda n: n["id"])
    return {"items": hits[:80]}


def api_graph_node(node_id: str) -> dict:
    g = _graph()
    if node_id not in g["nodes"]:
        return {"error": "no such node: " + node_id[:120]}
    outs = [e["dst"] for e in g["edges"] if e["src"] == node_id]
    ins = [e["src"] for e in g["edges"] if e["dst"] == node_id]
    return {"node": g["nodes"][node_id], "imports": sorted(set(outs)),
            "imported_by": sorted(set(ins))}


# ---------------------------------------------------------------------------
# Claims board (ruflo-inbuilt): every quality claim + its honest label
# ---------------------------------------------------------------------------

_LABELS = ("Implemented and verified", "Implemented, partially verified",
          "Compatible in principle, not tested", "Deferred with",
          "Configured, awaiting", "not a claim")


def api_claims() -> dict:
    sp = os.path.join(ROOT, "STATUS.md")
    claims = []
    try:
        text = open(sp, encoding="utf-8", errors="ignore").read()
    except Exception:
        return {"items": [], "error": "STATUS.md unreadable"}
    section = "Overview"
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if "Label:" in line or any(l in line for l in _LABELS[:4]):
            label = None
            for l in _LABELS:
                if l in line:
                    label = l
                    break
            body = line.strip().lstrip("- ").strip()
            claims.append({"section": section, "text": body[:400],
                           "label": label or "Unlabeled",
                           "verified": label == "Implemented and verified"})
    return {"items": claims}


# ---------------------------------------------------------------------------
# Real format-framework endpoints: conversion + honest inspection
# ---------------------------------------------------------------------------

CONVERT_MAX_BYTES = 50_000_000  # floor; raised adaptively below


def _convert_cap_bytes() -> int:
    """On-device conversion cap: at most ~1/8 of free RAM for the working
    copies (weights + quantized payloads + headroom), never below the floor.
    Static caps either OOM phones or pointlessly refuse on big machines."""
    avail = _mem_available_mb()
    if avail is None:
        return CONVERT_MAX_BYTES
    return max(CONVERT_MAX_BYTES, int(avail * 1e6 // 8))


def _lib_path(p: str, must_exist: bool) -> str | None:
    """Resolve p only if it lives inside a configured models dir.
    realpath (not abspath) — a symlink inside a models dir must not
    smuggle reads/writes outside the jail."""
    real_dirs = [os.path.realpath(d) for d in MODELS_DIRS]
    rp = os.path.realpath(str(p))
    if not any(rp == d or rp.startswith(d + os.sep) for d in real_dirs):
        return None
    if must_exist and not os.path.isfile(rp):
        return None
    return rp


def _lumina_convert_module():
    import importlib.util
    sp = os.path.join(ROOT, "scripts", "lumina_convert.py")
    spec = importlib.util.spec_from_file_location("lumina_convert_srv", sp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def api_convert(body: dict) -> dict:
    """Run the REAL Lumina conversion pipeline on a library file.

    Loads through the format-agnostic adapters, calibrates with real forward
    passes, quantizes with the doctrine codecs, writes a .lumina file, and
    returns honest byte accounting. Phone-safe: path-jail + 50MB cap.
    """
    import time as _t
    import numpy as np
    inp = _lib_path(body.get("input", ""), must_exist=True)
    outp = _lib_path(body.get("output", ""), must_exist=False)
    rung = str(body.get("rung", "int4"))
    if inp is None:
        return {"error": "input must be an existing file inside a models dir ("
                + ", ".join(MODELS_DIRS) + ")"}
    if outp is None:
        return {"error": "output must be inside a models dir"}
    if rung not in ("int4", "int3", "int2", "binary"):
        return {"error": "rung must be int4/int3/int2/binary"}
    if os.path.getsize(inp) > _convert_cap_bytes():
        return {"error": "input too large for on-device conversion "
                f"({os.path.getsize(inp)/1e6:.0f}MB > {_convert_cap_bytes()/1e6:.0f}MB cap) — use the CLI"}
    if os.path.abspath(outp) == os.path.abspath(inp):
        return {"error": "output must differ from input"}
    from lumina_framework.adapters import detect_format, load_model
    fmt = detect_format(inp)
    if fmt is None:
        return {"error": "cannot detect format of " + os.path.basename(inp)}
    t0 = _t.perf_counter()
    try:
        m = load_model(inp)
        arrays = {k: v for k, v in m.tensors.tensors.items()}
        if not arrays:
            return {"error": (f"{fmt} loaded header-only (no weight bytes) — "
                              "conversion needs full weights on a bigger machine")}
        moe_meta = None
        moe_report = None
        moe_router_names: set[str] = set()
        mo = body.get("moe") or {}
        if mo:
            from lumina_framework.virtual_moe import apply_to_arrays
            arrays, moe_meta, moe_report = apply_to_arrays(
                arrays, {}, int(mo.get("experts", 8)),
                int(mo.get("top_k", 2)))
            moe_router_names = {k for k in arrays
                                if k.endswith("mlp.gate.weight")
                                and k not in m.tensors.tensors}
        n_params = sum(int(v.size) for v in arrays.values())
        fp32_mb = sum(v.size for v in arrays.values()) * 4 / 1e6
        LC = _lumina_convert_module()
        acts = LC._collect_acts(m)
        payloads, stats = LC.convert_arrays(arrays, acts, rung)
        for rn in moe_router_names:
            if rn in arrays:
                # virtual-MoE routers stay fp32: routing noise is quality loss
                a = np.ascontiguousarray(arrays[rn], dtype=np.float32)
                payloads[rn] = ("raw", a)
                for st in stats:
                    if st.get("name") == rn:
                        st.update({"kind": "fp32-router", "bytes": a.size * 4,
                                   "bpp": 32.0, "escaped": True})
        os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
        src_dir = os.path.dirname(os.path.abspath(inp))
        LC.write_lumina(outp, payloads, stats, m.spec.name, rung,
                        source_dir=src_dir, moe_meta=moe_meta)
    except Exception as e:
        return {"error": "conversion failed: " + str(e)[:200]}
    sz = os.path.getsize(outp)
    _lib_invalidate()
    rels = [x["rel_weight_err"] for x in stats if "rel_weight_err" in x]
    n_q = sum(1 for x in stats if not x.get("escaped") and not x.get("skipped"))
    out = {"ok": True, "input_format": fmt, "output": outp,
            "bytes": sz, "mb": round(sz / 1e6, 3),
            "effective_bpp": round(8.0 * sz / max(n_params, 1), 3),
            "params_m": round(n_params / 1e6, 2), "fp32_mb": round(fp32_mb, 2),
            "tensors": len(arrays), "quantized": n_q,
            "escaped": sum(1 for x in stats if x.get("escaped")),
            "calibrated": len(acts),
            "mean_relW": round(float(sum(rels) / len(rels)), 4) if rels else None,
            "max_relW": round(float(max(rels)), 4) if rels else None,
            "seconds": round(_t.perf_counter() - t0, 1)}
    if moe_report is not None:
        out["virtual_moe"] = {
            "experts": {k: v["experts"] for k, v in moe_report["layers"].items()},
            "rel_func_err": {k: round(v["rel_func_err"], 4)
                             for k, v in moe_report["layers"].items()},
            "signatures": moe_report["signatures"]}
    return out


def api_inspect(path: str) -> dict:
    """Honest inspection of any library model file: format + true bytes."""
    ap = _lib_path(path, must_exist=True)
    if ap is None:
        return {"error": "path must be an existing file inside a models dir"}
    from lumina_framework.adapters import detect_format, load_model
    fmt = detect_format(ap)
    if fmt is None:
        return {"error": "cannot detect format"}
    out = {"path": ap, "format": fmt,
           "bytes": os.path.getsize(ap), "mb": round(os.path.getsize(ap) / 1e6, 3)}
    if ap.endswith(".lumina"):
        try:
            import struct as _st
            with open(ap, "rb") as f:
                magic, ver, hlen = _st.unpack("<IIQ", f.read(16))
                header = json.loads(f.read(hlen))
            out["rung"] = header.get("rung")
            mats = header.get("matrices", [])
            out["matrices"] = len(mats)
            q = [x for x in mats if not x.get("escaped") and not x.get("skipped")]
            rels = [x["rel_weight_err"] for x in q if "rel_weight_err" in x]
            if rels:
                out["recorded_mean_relW"] = round(float(sum(rels) / len(rels)), 4)
            out["model"] = header.get("name")
            if header.get("source_dir"):
                out["source_dir"] = header.get("source_dir")
        except Exception as e:
            out["note"] = "header unreadable: " + str(e)[:120]
        return out
    try:
        m = load_model(ap)
        arrays = m.tensors.tensors
        out["tensors"] = len(arrays)
        out["params_m"] = round(sum(v.size for v in arrays.values()) / 1e6, 2)
        out["fp32_mb"] = round(sum(v.size for v in arrays.values()) * 4 / 1e6, 2)
    except Exception as e:
        out["note"] = "deep load failed: " + str(e)[:120]
    return out


# ---------------------------------------------------------------------------
# Tiered memory: T1 = browser localStorage (client), T2 = server archive
# ---------------------------------------------------------------------------

MEM_PATH = os.path.join(HERE, "memory_archive.json")


def _load_archive() -> dict:
    try:
        return json.load(open(MEM_PATH))
    except Exception:
        return {"threads": {}}


def _save_archive(d: dict) -> None:
    try:
        json.dump(d, open(MEM_PATH, "w"), indent=1)
    except Exception:
        pass


def api_memory_stats() -> dict:
    a = _load_archive()
    n = len(a.get("threads", {}))
    return {"tiers": [
        {"id": "t0", "name": "T0 live", "where": "page memory",
         "holds": "current thread, in-flight state"},
        {"id": "t1", "name": "T1 recent", "where": "browser localStorage",
         "holds": "threads, tasks, docs + app config"},
        {"id": "t2", "name": "T2 archive", "where": "server file (memory_archive.json)",
         "holds": f"{n} archived thread(s)"}],
        "archived": n}


def api_memory_list() -> dict:
    a = _load_archive()
    items = []
    for tid, t in a.get("threads", {}).items():
        items.append({"id": tid, "title": str(t.get("title", ""))[:80],
                      "msgs": len(t.get("msgs", [])),
                      "at": t.get("at", 0)})
    items.sort(key=lambda x: -x["at"])
    return {"items": items[:100]}


def api_memory_archive(thread: dict) -> dict:
    if not isinstance(thread, dict) or not thread.get("msgs"):
        return {"error": "nothing to archive"}
    a = _load_archive()
    tid = str(int(thread.get("id") or abs(hash(json.dumps(thread)[:200])) % 10**12))
    t = dict(thread)
    t["at"] = t.get("at") or time.time()
    a.setdefault("threads", {})[tid] = t
    _save_archive(a)
    return {"archived": tid}


def api_memory_restore(tid: str) -> dict:
    a = _load_archive()
    t = a.get("threads", {}).get(tid)
    if t is None:
        return {"error": "no archived thread " + tid[:60]}
    return {"thread": t}


def _eagle_head_for(model):
    """Load a trained EAGLE head matching the loaded model, or explain."""
    import glob as _g
    cands = []
    base = None
    mid = CHAT.get("model_id") or ""
    for d in MODELS_DIRS:
        hit = os.path.join(d, mid, "eagle_head.npz")
        if os.path.isfile(hit):
            cands.append(hit)
        hit2 = os.path.join(d, os.path.basename(mid) + "_eagle_head.npz")
        if os.path.isfile(hit2):
            cands.append(hit2)
    for d in MODELS_DIRS:
        for hit in _g.glob(os.path.join(d, "*eagle_head.npz")):
            if hit not in cands:
                cands.append(hit)
    if not cands:
        return None, ("no trained EAGLE head found — run "
                      "python3 scripts/train_eagle.py --model-dir <dir> first")
    from lumina_framework.spec import DraftHead
    import numpy as _np
    Dm = model.t["model.embed_tokens.weight"].shape[1]
    for npz in cands:
        try:
            z = _np.load(npz)
            meta = __import__("json").load(open(npz.replace(".npz", ".json")))
            if int(meta.get("d_model", -1)) != Dm:
                continue
            head = DraftHead(Dm, d_draft=int(meta.get("d_draft", Dm)), seed=0)
            for k in ("fc", "q", "k", "v", "o", "gate", "up", "down", "back", "n"):
                setattr(head, k, _np.asarray(z[k], dtype=_np.float32))
            return head, None
        except Exception:
            continue
    return None, "EAGLE heads found but none match d_model=%d" % Dm


def _mtp_stack_for(model):
    import glob as _g
    cands = []
    mid = CHAT.get("model_id") or ""
    for d in MODELS_DIRS:
        hit = os.path.join(d, mid, "mtp_stack.npz")
        if os.path.isfile(hit):
            cands.append(hit)
        for hit2 in _g.glob(os.path.join(d, "*mtp_stack.npz")):
            if hit2 not in cands:
                cands.append(hit2)
    if not cands:
        return None, ("no trained MTP stack found — run "
                      "python3 scripts/train_eagle.py --model-dir <dir> first")
    from lumina_framework.spec import MTPStack
    import numpy as _np
    Dm = model.t["model.embed_tokens.weight"].shape[1]
    for npz in cands:
        try:
            z = _np.load(npz)
            meta = __import__("json").load(open(npz.replace(".npz", ".json")))
            if int(meta.get("d_model", -1)) != Dm:
                continue
            st = MTPStack(depth=int(meta.get("depth", 1)),
                          d_model=Dm, d_mtp=int(meta.get("d_mtp", Dm)), seed=0)
            for dd in range(st.depth):
                for k in ("proj", "blk_q", "blk_o", "back"):
                    getattr(st, k)[dd] = _np.asarray(z[f"d{dd}_{k}"],
                                                     dtype=_np.float32)
            return st, None
        except Exception:
            continue
    return None, "MTP stacks found but none match d_model=%d" % Dm


def _api_chat_eagle(model, prompt: str, max_tokens: int) -> dict:
    from lumina_framework import eagle as EG
    if model.__class__.__name__ != "ChatModel":
        return {"error": "eagle tree verify needs the dense ChatModel math "
                         "(attn_bias/pos_override); quantized path uses spec=true (PLD)"}
    head, err = _eagle_head_for(model)
    if head is None:
        return {"error": err}
    return EG.generate_eagle(model, prompt, head,
                             max_tokens=max(1, min(int(max_tokens), 128)),
                             max_nodes=15, branch=(2, 2, 1))


def _api_chat_mtp(model, prompt: str, max_tokens: int) -> dict:
    from lumina_framework import eagle as EG
    if model.__class__.__name__ != "ChatModel":
        return {"error": "mtp verify needs the dense ChatModel math; "
                         "quantized path uses spec=true (PLD)"}
    stack, err = _mtp_stack_for(model)
    if stack is None:
        return {"error": err}
    return EG.generate_mtp(model, prompt, stack,
                           max_tokens=max(1, min(int(max_tokens), 128)))


def api_memory_forget(tid: str) -> dict:
    a = _load_archive()
    if tid in a.get("threads", {}):
        del a["threads"][tid]
        _save_archive(a)
        return {"forgotten": tid}
    return {"error": "no archived thread " + tid[:60]}


# ---------------------------------------------------------------------------
# doctor + plan: read-only model verification and RAM placement planning
# (Colibri-inspired; header reads only — no weight bytes, phone-safe)
# ---------------------------------------------------------------------------

def _model_headers(path: str) -> dict:
    """Header-only facts for any library model file/dir. Zero weight bytes."""
    import glob
    out: dict = {"path": path, "exists": os.path.exists(path)}
    try:
        if os.path.isdir(path):
            kvs: dict = {}
            gg = sorted(glob.glob(os.path.join(path, "*.gguf")))
            st = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            out["bytes"] = sum(os.path.getsize(f) for f in (gg + st))
            if gg:
                out["format"] = "gguf"
                from lumina_framework.adapters.gguf import GGUF_MAGIC
                with open(gg[0], "rb") as f:
                    import struct as _st
                    magic, _ver, nt, nkv = _st.unpack("<IIQQ", f.read(24))
                    out["gguf_magic_ok"] = (magic == GGUF_MAGIC)
                    out["tensors"] = nt
            elif st:
                out["format"] = "safetensors"
                out["shards"] = len(st)
                names: set[str] = set()
                for sh in st:
                    try:
                        import struct as _st2
                        with open(sh, "rb") as f:
                            (n,) = _st2.unpack("<Q", f.read(8))
                            hdr = json.loads(f.read(n))
                        names |= {k for k in hdr if k != "__metadata__"}
                        out["first_shard_header_ok"] = True
                    except Exception:
                        pass
                out["named_tensors"] = len(names)
                cfg = os.path.join(path, "config.json")
                out["has_config"] = os.path.isfile(cfg)
                out["has_tokenizer"] = (os.path.isfile(os.path.join(path, "vocab.json"))
                                        or os.path.isfile(os.path.join(path, "tokenizer.json")))
            else:
                out["format"] = "unknown-dir"
            return out
        if path.endswith(".gguf"):
            out["format"] = "gguf"
            out["bytes"] = os.path.getsize(path)
            from lumina_framework.adapters.gguf import GGUF_MAGIC
            with open(path, "rb") as f:
                import struct as _st
                magic = _st.unpack("<I", f.read(4))[0]
                out["gguf_magic_ok"] = (magic == GGUF_MAGIC)
            return out
        if path.endswith(".lumina"):
            out["format"] = "lumina"
            out["bytes"] = os.path.getsize(path)
            from lumina_framework.adapters.lumina import lumina_header
            hdr = lumina_header(path)
            mats = hdr.get("matrices", [])
            out["rung"] = hdr.get("rung")
            out["matrices"] = len(mats)
            out["has_source_dir"] = bool(hdr.get("source_dir"))
            out["virtual_moe"] = bool(hdr.get("virtual_moe"))
            n_params = 0
            for m in mats:
                sh = m.get("shape")
                if sh and len(sh) == 2:
                    n_params += int(sh[0]) * int(sh[1])
                elif "bytes" in m:
                    n_params += int(m["bytes"]) / 4
            out["params_m"] = round(n_params / 1e6, 1)
            return out
        if path.endswith(".safetensors"):
            out["format"] = "safetensors"
            out["bytes"] = os.path.getsize(path)
            try:
                import struct as _st2
                with open(path, "rb") as f:
                    (n,) = _st2.unpack("<Q", f.read(8))
                    hdr = json.loads(f.read(n))
                out["named_tensors"] = len([k for k in hdr if k != "__metadata__"])
            except Exception:
                pass
            return out
    except Exception as e:
        out["error"] = str(e)[:150]
    return out


def api_doctor(model_id: str) -> dict:
    """coli doctor --deep analog: verify a model is loadable BEFORE loading.
    Header reads only. Every check pass/fail is reported, nothing assumed."""
    lib = api_library()["items"]
    hit = next((x for x in lib if x["id"] == model_id), None)
    if hit is None:
        return {"error": "not in library: " + model_id[:120]}
    facts = _model_headers(hit["path"])
    checks = []
    def chk(name, ok, note=""):
        checks.append({"check": name, "ok": bool(ok), "note": note[:160]})
    if facts.get("format") == "lumina":
        chk("lumina magic + header readable", "rung" in facts)
        chk("matrices present", facts.get("matrices", 0) > 0)
        chk("provenance recorded (source_dir)",
            facts.get("has_source_dir"), "convert-time tokenizer resolution")
    elif facts.get("format") == "gguf":
        chk("gguf magic", facts.get("gguf_magic_ok"))
        chk("tensors present", facts.get("tensors", 0) > 0)
    elif facts.get("format") == "safetensors":
        chk("safetensors header readable", facts.get("named_tensors", 0) > 0)
        if os.path.isdir(facts["path"]):
            chk("config.json present", facts.get("has_config"))
            chk("tokenizer present", facts.get("has_tokenizer"))
    else:
        chk("format recognized", False, str(facts.get("format")))
    # RAM plan viability (plan is computed separately; doctor checks the floor)
    plan = api_plan(model_id)
    if "error" not in plan:
        chk("RAM placement plan computable", True,
            f"min {plan['min_ram_mb']}MB, resident {plan['resident_ram_mb']}MB")
        avail = plan.get("available_mb")
        if avail is not None:
            chk("machine meets the min RAM", avail >= plan["min_ram_mb"],
                f"{avail}MB free vs {plan['min_ram_mb']}MB min")
    return {"model": model_id, "facts": facts, "checks": checks,
            "ok": all(c["ok"] for c in checks)}


def api_plan(model_id: str) -> dict:
    """coli plan analog: where the model will live. Working-set estimate from
    header bytes; honest about being an estimate, not a measurement."""
    lib = api_library()["items"]
    hit = next((x for x in lib if x["id"] == model_id), None)
    if hit is None:
        return {"error": "not in library: " + model_id[:120]}
    facts = _model_headers(hit["path"])
    mb = facts.get("bytes", 0) / 1e6
    if facts.get("format") == "lumina" and facts.get("params_m"):
        # fp16 working copy of the dequantized weights + activations
        working = facts["params_m"] * 2.0
        min_ram = working * 2.5
    else:
        working = mb * (2.0 if facts.get("format") == "gguf" else 1.0)
        min_ram = max(working * 2.5, 300)
    avail = _mem_available_mb()
    plan = {"model": model_id, "format": facts.get("format"),
            "weights_mb": round(mb, 1),
            "working_set_mb": round(working, 1),
            "min_ram_mb": int(min_ram),
            "resident_ram_mb": int(min_ram),
            "available_mb": avail,
            "fits_now": (avail is None or avail >= min_ram),
            "note": "estimate from header bytes; measure on load for truth"}
    if avail is not None and avail < min_ram:
        plan["advice"] = ("needs streaming: load a smaller model, or run on a "
                           "bigger machine (the streamed 4-bit runtime handles "
                           "this on Lumina-format files)")
    return plan


def api_logs() -> dict:
    import copy
    with _LOG_LOCK:
        ring = copy.deepcopy(_LOG_RING)
    return {"items": ring[-100:], "log_file": LOG_PATH}


# ---------------------------------------------------------------------------
# System One: decisions with probabilities (Colibri's /v1/systemone analog)
# ---------------------------------------------------------------------------

def _option_logprob(model, prompt: str, option: str) -> float:
    """Mean log-prob of the option's tokens given the prompt. One forward."""
    import numpy as np
    from lumina_framework.chat import rmsnorm
    ids = model.tok.encode(prompt)
    emb = model.t["model.embed_tokens.weight"].astype(np.float32)
    cache: dict = {}
    h = emb[np.array(ids)]
    for lyr in range(model.n_layer):
        h = model._layer(h, lyr, cache, 0)
    head = model.head.T.astype(np.float32)
    total, n = 0.0, 0
    pos = len(ids)
    opt_ids = model.tok.encode(option)
    for t in opt_ids:
        lg = (h[-1:] @ head)[0]
        mx = lg.max()
        lp = float(lg[t] - mx - np.log(np.exp(lg - mx).sum()))
        total += lp
        n += 1
        h = emb[[t]]
        for lyr in range(model.n_layer):
            h = model._layer(h, lyr, cache, pos)
        pos += 1
    return total / max(n, 1)


def api_systemone(body: dict) -> dict:
    """A decision with a probability: score each option instead of generating.
    The model reads the state once; every allowed option gets a probability
    and a confidence (1 - normalised entropy). Nothing generated."""
    if CHAT["model"] is None or (CHAT.get("model_id") or "").startswith("cloud/"):
        return {"error": "systemone needs a local model loaded (scoring path)"}
    model = CHAT["model"]
    state = str(body.get("state", ""))[:8000]
    questions = body.get("questions") or {}
    if not state or not questions:
        return {"error": "state and questions are required"}
    answers = {}
    for name, q in questions.items():
        qtype = str(q.get("type", "choice"))
        instr = str(q.get("instructions", ""))[:500]
        crit = q.get("criteria") or {}
        if qtype == "choice":
            labels = [str(c) for c in crit.keys()]
            if not labels:
                answers[name] = {"error": "choice needs >=1 criterion"}
                continue
            base = (f"State: {state}\n\nQuestion: {instr}\n"
                    f"Answer with exactly one of: {', '.join(labels)}.\nAnswer:")
            lps = {}
            for lab in labels:
                lps[lab] = _option_logprob(model, base, lab)
            mx = max(lps.values())
            import math
            ws = {k: math.exp(v - mx) for k, v in lps.items()}
            tot = sum(ws.values())
            probs = {k: ws[k] / tot for k in ws}
            ent = -sum(p * math.log(p + 1e-12) for p in probs.values())
            nent = ent / math.log(len(probs)) if len(probs) > 1 else 0.0
            best = max(probs, key=probs.get)
            answers[name] = {"answer": best, "probabilities": probs,
                             "confidence": round(1.0 - nent, 4),
                             "logprobs": {k: round(v, 3) for k, v in lps.items()}}
        elif qtype == "noul":
            base = (f"State: {state}\n\nQuestion: {instr}\n"
                    "Answer yes or no.\nAnswer:")
            ly = _option_logprob(model, base, "yes")
            ln_ = _option_logprob(model, base, "no")
            import math
            wy, wn = math.exp(ly - max(ly, ln_)), math.exp(ln_ - max(ly, ln_))
            p_yes = wy / (wy + wn)
            answers[name] = {"answer": "yes" if p_yes >= 0.5 else "no",
                             "p_yes": round(p_yes, 4),
                             "confidence": round(abs(p_yes - 0.5) * 2, 4)}
        else:
            answers[name] = {"error": "type must be choice or noul"}
    return {"answers": answers, "generated": 0,
            "note": "scores options; no generation, so answers stay in your list"}

# ---------------------------------------------------------------------------
# RAG: chunk store + BM25 + embedding scoring (numpy-only, phone-light)
# ---------------------------------------------------------------------------

RAG_PATH = os.path.join(HERE, "rag_store.json")
RAG: dict = {"chunks": [], "df": {}, "lock": __import__("threading").Lock()}


def _rag_load():
    try:
        d = json.load(open(RAG_PATH))
        RAG["chunks"] = d.get("chunks", [])
        RAG["df"] = d.get("df", {})
    except Exception:
        pass


def _rag_save():
    try:
        json.dump({"chunks": RAG["chunks"], "df": RAG["df"]},
                  open(RAG_PATH, "w"))
    except Exception:
        pass


def _chunk(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


_WORDS_RE = __import__("re").compile(r"[a-z0-9\u00c0-\u024f]+")


def _toks(s: str) -> list[str]:
    return _WORDS_RE.findall(s.lower())


def api_rag_add(text: str, source: str) -> dict:
    if not text.strip():
        return {"error": "empty text"}
    chunks = _chunk(text[:200000])
    vecs = [_embed_text(c) for c in chunks] if CHAT["model"] is not None else [None] * len(chunks)
    with RAG["lock"]:
        _rag_load()
        for i, c in enumerate(chunks):
            entry = {"id": len(RAG["chunks"]), "source": source[:120],
                     "text": c, "tokens": _toks(c)}
            if vecs[i] is not None:
                entry["vec"] = [round(float(x), 5) for x in vecs[i]]
            RAG["chunks"].append(entry)
            for t in set(_toks(c)):
                RAG["df"][t] = RAG["df"].get(t, 0) + 1
        _rag_save()
    return {"added": len(chunks), "total_chunks": len(RAG["chunks"]),
            "embedded": sum(1 for v in vecs if v is not None)}


def api_rag_list() -> dict:
    with RAG["lock"]:
        _rag_load()
        return {"chunks": len(RAG["chunks"]),
                "sources": sorted({c["source"] for c in RAG["chunks"]})}


def api_rag_clear() -> dict:
    with RAG["lock"]:
        RAG["chunks"] = []
        RAG["df"] = {}
        _rag_save()
    return {"cleared": True}


def _rag_score(query: str, k: int) -> list[dict]:
    import math
    with RAG["lock"]:
        _rag_load()
        chunks = list(RAG["chunks"])
        df = dict(RAG["df"])
    if not chunks:
        return []
    N = len(chunks)
    qt = _toks(query)
    if not qt:
        return []
    # BM25
    avgdl = sum(len(c["tokens"]) for c in chunks) / N
    scores = []
    for ci, c in enumerate(chunks):
        tf: dict = {}
        for t in c["tokens"]:
            tf[t] = tf.get(t, 0) + 1
        sc = 0.0
        for t in qt:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            sc += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(c["tokens"]) / avgdl))
        scores.append(sc)
    top = sorted(range(N), key=lambda i: -scores[i])[: max(k * 3, k)]
    hits = []
    for i in top:
        if scores[i] <= 0:
            continue
        h = {"id": chunks[i]["id"], "source": chunks[i]["source"],
             "text": chunks[i]["text"], "bm25": round(scores[i], 4)}
        if "vec" in chunks[i]:
            h["vec"] = chunks[i]["vec"]
        hits.append(h)
    return hits


def api_rag_query(q: str, k: int) -> dict:
    """Hybrid rank: BM25 (always) + cosine on stored embedding vectors
    (when chunks were added AND the query runs with a model loaded).
    50/50 blend, each score min-max normalized over the candidate set."""
    import numpy as np
    hits = _rag_score(q, k * 3)
    if not hits:
        return {"query": q, "hits": [], "note": "no chunks matched"}
    qv = _embed_text(q)
    mode = "bm25"
    if qv is not None and any("vec" in h for h in hits):
        mode = "bm25+embedding"
        b = np.array([h.get("bm25", 0.0) for h in hits], dtype=np.float32)
        V = np.array([h["vec"] if "vec" in h else None for h in hits],
                     dtype=object)
        # cosine only against chunks that have vectors; others get 0
        cos = np.zeros(len(hits), dtype=np.float32)
        for i, v in enumerate(V):
            if isinstance(v, (list, np.ndarray)) and len(v) == len(qv):
                vv = np.asarray(v, dtype=np.float32)
                n = float(np.linalg.norm(vv) * np.linalg.norm(qv))
                cos[i] = float(vv @ qv) / n if n > 1e-9 else 0.0
        def nrm(x):
            mn, mx = float(x.min()), float(x.max())
            return (x - mn) / (mx - mn) if mx > mn + 1e-9 else np.zeros_like(x)
        hyb = 0.5 * nrm(b) + 0.5 * nrm(cos)
        for i, h in enumerate(hits):
            h["cosine"] = round(float(cos[i]), 4)
            h["score"] = round(float(hyb[i]), 4)
        hits.sort(key=lambda h: -h["score"])
    hits = hits[:k]
    for h in hits:
        h.setdefault("score", h.get("bm25", 0.0))
    return {"query": q, "hits": hits, "mode": mode,
            "note": "embedding vectors are input-embedding mean pools from "
                    "the loaded model (labeled honestly, not a trained "
                    "sentence embedder)"}


def api_ground_check(body: dict) -> dict:
    """POST /api/ground/check {answer, passages|query, k, policy} — pure
    scorer, never calls a model. With 'passages' it checks the supplied
    evidence directly; with 'query' it retrieves from the RAG store first
    (same BM25 as /api/rag/query)."""
    import grounding as _G
    b = body or {}
    answer = str(b.get("answer", ""))[:20000]
    if not answer.strip():
        return {"error": "answer required"}
    ps = b.get("passages")
    if ps is None:
        q = str(b.get("query", ""))[:400]
        k = max(1, min(int(b.get("k", 3)), 10))
        ps = _rag_score(q, k) if q.strip() else []
        note = "checked against %d retrieved passage(s) from the RAG store" % len(ps)
    else:
        note = "checked against supplied passages"
    if not isinstance(ps, list):
        return {"error": "passages must be a list"}
    ps = [p for p in ps[:8] if isinstance(p, (dict, str))]
    policy = "withhold" if str(b.get("policy", "flag")) == "withhold" else "flag"
    g = _G.ground_answer(answer, ps, policy=policy)
    g["passages"] = len(_G._passages_list(ps))
    g["note"] = note
    return g


def api_boost(body: dict) -> dict:
    """POST /api/boost {messages|question, n, temperature, max_tokens,
    do_decompose, do_verify, evidence, ground} — capability scaffolding for a
    small model: self-consistency voting, deterministic arithmetic checks, an
    optional verifier pass and optional evidence grounding. The mechanics are
    verified by tests/test_boost.py; the real-model LIFT is not measurable
    without real weights and the response says so."""
    import boost as _B
    b = body or {}
    msgs = b.get("messages")
    if not isinstance(msgs, list) or not msgs:
        q = str(b.get("question", ""))[:4000]
        if not q.strip():
            return {"error": "messages or question required"}
        msgs = [{"role": "user", "content": q}]
    n = max(1, min(int(b.get("n", 3)), 9))
    temp = float(b.get("temperature", 0.7))
    mt = max(8, min(int(b.get("max_tokens", 192)), 512))
    do_decompose = bool(b.get("do_decompose", False))
    do_verify = bool(b.get("do_verify", True))
    evidence = b.get("evidence")
    ground_fn = None
    if b.get("ground"):
        import grounding as _G
        if evidence is None:
            _q = ""
            for _m in reversed(msgs):
                if isinstance(_m, dict) and _m.get("role") == "user":
                    _q = _msg_text(_m) or ""
                    break
            evidence = _rag_score(_q, 3) if _q.strip() else []
        _policy = ("withhold" if str((b.get("ground") or {}).get("policy", "flag")) == "withhold"
                   else "flag")

        def ground_fn(text, evidence=None, _G=_G, _policy=_policy):
            return _G.ground_answer(text, evidence or [], policy=_policy)

    def chat_fn(messages, max_tokens=mt, temperature=temp, **kw):
        return api_chat(messages, int(max_tokens), float(temperature))

    out = _B.boost_answer(chat_fn, msgs, evidence=evidence, n=n,
                          temperature=temp, do_decompose=do_decompose,
                          do_verify=do_verify, ground_fn=ground_fn)
    out["model"] = CHAT.get("model_id")
    return out


# ---------------------------------------------------------------------------
# Agent: model-driven tool loop over the allowlisted exec + fetch tools
# ---------------------------------------------------------------------------

AGENT_OPERATOR_TOOLS = ("fs_list", "fs_read", "fs_write", "fs_delete",
                       "fs_search", "app_open", "app_list")


def _agent_tool_call(call: dict) -> dict:
    """Execute one agent tool call. Open tools work by default; operator
    tools refuse cleanly when operator mode is off; mcp: tools start lazily."""
    import browser_driver
    tool = str(call.get("tool", ""))
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    if tool == "exec":
        return api_exec(str(args.get("cmd", "")), args.get("cwd"),
                        int(args.get("timeout", 15)))
    if tool == "fetch_url":
        return api_fetch_url(str(args.get("url", "")))
    if tool.startswith("browser_"):
        return api_browser({"op": tool[len("browser_"):], **args})
    if tool in ("fs_list", "fs_read", "fs_write", "fs_delete", "fs_search"):
        fsb = {"op": tool[len("fs_"):], "path": args.get("path", "~")}
        fsb.update({k: v for k, v in args.items() if k != "path"})
        return api_fs(fsb)
    if tool == "app_open":
        return api_apps({"op": "launch", "app": args.get("app", ""),
                         "args": args.get("args") or []})
    if tool == "app_list":
        return api_apps({"op": "list"})
    if tool == "skill_fetch":
        q = str(args.get("query", ""))
        r = api_skills_search(q, int(args.get("k", 3) or 3))
        if "error" in r:
            return r
        hits = r.get("skills") or []
        if not hits:
            return {"out": "no skill matches '" + q[:60] + "'",
                    "confident": False, "reason": r.get("reason")}
        if r.get("confident"):
            by_id = {str(s.get("id", "")): s for s in _all_skills()}
            sk = by_id.get(hits[0]["id"]) or {}
            text = ("skill " + str(hits[0]["id"]) + " (score "
                    + str(round(hits[0]["score"], 2)) + "):\n"
                    + str(sk.get("prompt", ""))[:1200])
        else:
            text = ("no confident skill match for '" + q[:60]
                    + "'; candidates: " + ", ".join(h["id"] for h in hits[:5]))
        return {"out": text,
                "skills": [{"id": h["id"], "score": round(h["score"], 2)}
                           for h in hits[:5]],
                "confident": bool(r.get("confident")), "reason": r.get("reason")}
    if tool == "media_generate":
        import media as _M
        return _M.api_media_generate(args)
    if tool == "media_status":
        import media as _M
        return _M.api_media_status()
    if tool.startswith("mcp:"):
        rest = tool[4:]
        srv, _, mtool = rest.partition("::")
        return api_mcp_call({"id": srv, "tool": mtool, "args": args})
    return {"error": "unknown tool: " + tool[:80]}


_AGENT_TOOL_DESCRIPTIONS = {
    "exec": "Run a shell command on the user's machine (read-only allowlist; "
            "operator mode widens it).",
    "fetch_url": "Fetch an http/https URL and return readable text.",
    "fs_list": "List a directory.",
    "fs_read": "Read a text file.",
    "fs_write": "Write a text file (creates or replaces).",
    "fs_delete": "Delete a file (operator mode only).",
    "fs_search": "Search files by name and content.",
    "app_open": "Launch a desktop application.",
    "app_list": "List launchable applications.",
    "skill_fetch": "Find the best-matching skill for a task and read its "
                   "instructions. Args: query, k?.",
    "media_generate": "Generate an image/video/audio through a configured "
                      "backend (ComfyUI workflow, A1111, or a CLI tool). Args: "
                      "capability (image|video|audio_tts|audio_stt|music|"
                      "upscale|any-workflow), prompt, workflow_id?, width?, "
                      "height?, steps?, seed?, policy? (LLM residency: "
                      "off|unload|reload_after).",
    "media_status": "List media backends (configured/reachable), workflows "
                    "and recent jobs.",
}


def _agent_base_tool_specs() -> list[dict]:
    """Tool specs for GET /api/agent list_tools (shape unchanged)."""
    import browser_driver
    op_on = bool(_operator_mode().get("full_access"))
    avail = [{"tool": "exec",
              "gate": "allowlist" if not op_on else "operator",
              "args": {"cmd": "shell command"}},
             {"tool": "fetch_url", "args": {"url": "http..."}},
             {"tool": "skill_fetch", "gate": "open",
              "args": {"query": "task description"}}]
    for t in browser_driver.tool_list():
        avail.append({"tool": t["name"], "gate": "operator",
                      "args": t["args"]})
    avail += [{"tool": "fs_list", "gate": "operator", "args": {"path": "~"}},
              {"tool": "media_status", "gate": "open", "args": {}},
              {"tool": "media_generate", "gate": "open",
               "args": {"capability": "image", "prompt": "..."}},
              {"tool": "fs_read", "gate": "operator", "args": {"path": "..."}},
              {"tool": "fs_write", "gate": "operator",
               "args": {"path": "...", "text": "..."}},
              {"tool": "fs_search", "gate": "operator",
               "args": {"path": "~", "query": "..."}},
              {"tool": "app_open", "gate": "operator",
               "args": {"app": "firefox"}},
              {"tool": "app_list", "gate": "operator", "args": {}}]
    return avail


def _agent_tool_specs_sdk() -> list[dict]:
    """Same specs plus human descriptions for the SDK's function schemas."""
    out = []
    for s in _agent_base_tool_specs():
        d = dict(s)
        name = str(s.get("tool", ""))
        d["description"] = _AGENT_TOOL_DESCRIPTIONS.get(
            name, ("Browser: " + name[len("browser_"):].replace("_", " "))
            if name.startswith("browser_") else "Lumina tool " + name)
        out.append(d)
    return out


def api_agent(body: dict) -> dict:
    """Agent entry point. `engine`: builtin (default) | sdk | auto.

    builtin — the original fenced-tool loop.
    sdk     — the OpenAI Agents SDK bridge; explicit "sdk" never falls back.
    auto    — sdk when the bridge is ready and a model target exists,
              otherwise builtin with the reason reported.
    """
    eng = str(body.get("engine") or CFG.get("agent_engine") or "builtin").lower()
    if eng not in ("builtin", "sdk", "auto"):
        eng = "builtin"
    if eng in ("sdk", "auto"):
        r = api_agent_sdk(body)
        if not r.get("fallback"):
            return r
        if eng == "sdk":
            return r
        out = _agent_builtin(body)
        out["engine"] = "builtin"
        out["engine_reason"] = str(r.get("error") or r.get("engine_reason")
                                   or "sdk unavailable")
        return out
    out = _agent_builtin(body)
    out["engine"] = "builtin"
    return out


def api_agent_sdk(body: dict) -> dict:
    """Run the agent through the OpenAI Agents SDK (optional dependency).

    Returns {"fallback": true, ...} whenever the bridge cannot run, so the
    dispatcher decides what to do; tools still execute through
    _agent_tool_call, keeping operator gating and the ops audit intact.
    """
    import agent_sdk
    st = agent_sdk.status()
    if st.get("state") == "cold":
        agent_sdk.warmup(background=True)
        st = agent_sdk.status()
    if st.get("state") != "ready":
        hint = (" " + str(st.get("hint"))) if st.get("hint") else ""
        return {"engine": "sdk", "fallback": True, "sdk": st,
                "error": ("OpenAI Agents SDK not ready (" + str(st.get("state"))
                          + ")." + hint)}
    local_id = CHAT.get("model_id") if CHAT.get("model") is not None else None
    target = agent_sdk.model_target(_cloud_cfg(), local_id)
    if target.get("kind") == "none":
        return {"engine": "sdk", "fallback": True, "sdk": st,
                "error": target.get("reason", "no model target")}
    specs = _agent_tool_specs_sdk()
    mcp_entries = []
    want = [str(x) for x in (body.get("mcp_servers") or [])][:4]
    if want:
        try:
            import mcp_registry
            all_e = {str(e.get("id")): e for e in mcp_registry.build_registry()}
        except Exception:
            all_e = {}
        for rid in want:
            e = all_e.get(rid)
            if not e:
                continue
            mcp_entries.append(e)
            try:
                for t in api_mcp_tools(rid).get("tools", [])[:20]:
                    specs.append({"tool": "mcp:" + rid + "::" + str(t.get("name")),
                                  "description": "MCP tool from " + rid})
            except Exception:
                pass
    sys_ins = "\n".join(_msg_text(m) for m in (body.get("messages") or [])
                        if isinstance(m, dict) and m.get("role") == "system")
    r = agent_sdk.run_sync(
        target=target, tool_specs=specs, tool_fn=_agent_tool_call,
        chat_fn=api_chat, messages=body.get("messages") or [],
        instructions_base=sys_ins[:2000],
        max_steps=min(int(body.get("max_steps", 3)), 5),
        max_tokens=int(body.get("max_tokens", 256)),
        temperature=float(body.get("temperature", 0.0)),
        mcp_entries=mcp_entries)
    r.setdefault("engine", "sdk")
    r["fallback"] = bool(r.get("fallback"))
    r["sdk"] = {k: st.get(k) for k in ("state", "version", "import_s")}
    return r


def api_agent_sdk_status() -> dict:
    import agent_sdk
    return agent_sdk.status()


def api_agent_sdk_warmup(body: dict | None = None) -> dict:
    import agent_sdk
    bg = True
    if isinstance(body, dict) and "background" in body:
        bg = bool(body["background"])
    return agent_sdk.warmup(background=bg)


def _agent_builtin(body: dict) -> dict:
    """Run the model; if it emits a fenced ```tool JSON call, execute the
    tool and feed the result back, up to max_steps. Each step is a real
    api_chat call — every tool use reported, nothing hidden. Tool set:
    exec, fetch_url, mcp:<server>::<tool>, browser_*, fs_*, app_* (the
    operator-gated ones refuse cleanly while operator mode is off)."""
    import agent_sdk
    import agent_grammar
    messages = list(body.get("messages") or [])
    if body.get("list_tools"):
        avail = _agent_base_tool_specs()
        try:
            run = MR_STATUS()
            for rid in run[:20]:
                tl = api_mcp_tools(rid).get("tools", [])
                for t in tl[:20]:
                    avail.append({"tool": "mcp:" + rid + "::" + str(t.get("name"))})
        except Exception:
            pass
        return {"available_tools": avail, "operator": _operator_mode(),
                "format": "emit a fenced block: ```tool\n{\"tool\": \"...\", \"args\": {...}}\n```"}
    # Automatic skill fetch (small-model boost): one calibrated index query
    # on the user's task; the winning skill is injected only when the index
    # is confident, and recorded in the result either way.
    skill_note = None
    if body.get("skill_auto", True):
        _aq = ""
        for _m in reversed(messages):
            if isinstance(_m, dict) and _m.get("role") == "user":
                _aq = _msg_text(_m) or ""
                break
        if _aq.strip():
            _ar = api_skills_search(_aq, 3)
            if "error" not in _ar and (_ar.get("skills") or []):
                _at = _ar["skills"][0]
                skill_note = {"confident": bool(_ar.get("confident")),
                              "reason": _ar.get("reason"), "id": _at["id"],
                              "score": round(_at["score"], 3)}
                if _ar.get("confident"):
                    _ask = next((x for x in _all_skills()
                                 if x.get("id") == _at["id"]), None)
                    if _ask:
                        messages = [{"role": "system", "content":
                                     "[auto skill: " + str(_ask.get("id")) + "]\n"
                                     + str(_ask.get("prompt", ""))[:1200]}] + messages
                        skill_note["injected"] = True

    def _fin(d: dict) -> dict:
        if skill_note:
            d["skill_auto"] = skill_note
        return d

    max_steps = min(int(body.get("max_steps", 3)), 5)
    use_grammar = str(body.get("grammar",
                               CFG.get("agent_grammar", "on"))).lower() != "off"
    steps = []
    gstats = {"valid_first": 0, "repaired": 0, "refused": 0, "retries": 0,
              "parallel_batches": 0, "parallel_calls": 0, "batch_seconds": 0.0}
    specs = _agent_base_tool_specs()
    known = {str(s.get("tool", "")) for s in specs}
    if use_grammar:
        # Grammar block goes into the system message ONCE, before the loop:
        # the prefix stays byte-stable across steps (Atomic-Agent KV-cache
        # discipline), so every step reuses the cached prefill.
        _gb = agent_grammar.grammar_block(specs)
        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0],
                               content=str(messages[0].get("content", ""))
                               + "\n" + _gb)
        else:
            messages = [{"role": "system", "content": _gb}] + messages
    retries = 0
    for _ in range(max_steps):
        r = api_chat(messages, int(body.get("max_tokens", 96)),
                     float(body.get("temperature", 0.0)),
                     samplers=body.get("samplers") or {})
        if "error" in r:
            return _fin({"steps": steps, "error": r["error"],
                         **({"grammar": gstats} if use_grammar else {})})
        text = r.get("text", "")
        parsed = (agent_grammar.parse_calls(text)
                  if use_grammar else {"calls": [{"tool": "", "args": {}}]})
        if use_grammar and "error" in parsed:
            gstats["refused"] += 1
            steps.append({"type": "bad-call",
                          "text": agent_grammar.bounded_tail(text, 400),
                          "error": parsed["error"]})
            if retries < 1:                      # bounded retry (Atomic Agent)
                retries += 1
                gstats["retries"] += 1
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": agent_grammar.correction_message(
                        [parsed["error"]], [])}]
                continue
            return _fin({"steps": steps, "text": text,
                         "error": "invalid tool block after 1 retry",
                         "grammar": gstats})
        calls = parsed["calls"]
        if use_grammar and not calls:
            steps.append({"type": "final", "text": text,
                          "tokens_per_second": r.get("tokens_per_second")})
            return _fin({"steps": steps, "text": text, "grammar": gstats})
        if not use_grammar:
            m = agent_sdk.FENCE_RE.search(text)
            if not m:
                steps.append({"type": "final", "text": text,
                              "tokens_per_second": r.get("tokens_per_second")})
                return _fin({"steps": steps, "text": text})
            try:
                call = json.loads(m.group(1))
            except Exception:
                steps.append({"type": "bad-call", "text": text[:200]})
                return _fin({"steps": steps, "text": text,
                             "error": "model emitted unparseable tool JSON"})
            calls = [call]
        fixed = []
        call_repairs = []
        for _c in calls:
            _v = agent_grammar.validate_call(_c, known)
            if not _v["ok"]:
                gstats["refused"] += 1
                steps.append({"type": "invalid-call", "call": _c,
                              "errors": _v["errors"]})
                continue
            _rep = agent_grammar.repair_call(_c)
            if _rep["repairs"] or _v["warnings"]:
                gstats["repaired"] += 1
            fixed.append(_rep["call"])
            call_repairs.append(_rep["repairs"])
        if not fixed:
            if retries < 1:
                retries += 1
                gstats["retries"] += 1
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user",
                     "content": "No usable tool call. Re-emit valid calls or "
                                "answer directly."}]
                continue
            return _fin({"steps": steps, "text": text,
                         "error": "no valid tool call after retry",
                         "grammar": gstats})
        if len(fixed) == 1:
            gstats["valid_first"] += 1
        _plan = agent_grammar.plan_batch(fixed)
        if _plan["parallel"] and use_grammar:
            _b = agent_grammar.execute_batch(fixed, _agent_tool_call)
            gstats["parallel_batches"] += 1
            gstats["parallel_calls"] += _b["calls"]
            gstats["batch_seconds"] += round(_b["seconds"], 4)
            _outs = _b["outs"]
            _results = [str((_o.get("out") or _o.get("text")
                             or _o.get("data_url") or _o))[:2000]
                        for _o in _outs]
            steps.append({"type": "tool-batch", "calls": fixed,
                          "results": _results, "seconds": round(_b["seconds"], 3),
                          "workers": _b["workers"],
                          "repairs": [r for r in call_repairs if r]})
        else:
            _results = []
            for _i, _c in enumerate(fixed):
                _o = _agent_tool_call(_c)
                if "error" in _o and _o.get("error"):
                    _results.append("ERROR: " + str(_o["error"]))
                elif str(_c.get("tool", "")).startswith("browser_screenshot"):
                    _results.append("screenshot saved: " + str(_o.get("url", "")))
                else:
                    _results.append(str(_o.get("out") or _o.get("text")
                                        or _o.get("data_url") or _o)[:4000])
                steps.append({"type": "tool", "call": _c,
                              "result": str(_results[-1])[:600],
                              "out_keys": sorted(_o.keys())[:12],
                              "repairs": call_repairs[_i]})
        _combined = "\n".join("call %d -> %s" % (i + 1, x)
                              for i, x in enumerate(_results))
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user",
                                "content": "Tool results:\n" + _combined[:8000]
                                + "\nContinue. If you have the answer, reply "
                                  "without a tool block."}]
    _out = _fin({"steps": steps,
                 "error": "max_steps reached without a final answer"})
    if use_grammar:
        _out["grammar"] = gstats
    return _out
# ---------------------------------------------------------------------------
# Arena ELO (server-persistent, K=32)
# ---------------------------------------------------------------------------

ELO_PATH = os.path.join(HERE, "arena_elo.json")


def _elo_load() -> dict:
    try:
        return json.load(open(ELO_PATH))
    except Exception:
        return {"ratings": {}, "matches": 0}


def api_elo_get() -> dict:
    d = _elo_load()
    ratings = sorted(d["ratings"].items(), key=lambda kv: -kv[1])
    return {"ratings": [{"model": k, "elo": round(v)} for k, v in ratings],
            "matches": d["matches"]}


def api_elo_vote(body: dict) -> dict:
    a = str(body.get("a", ""))
    b = str(body.get("b", ""))
    outcome = str(body.get("outcome", ""))  # "a" | "b" | "tie"
    if not (a and b and outcome in ("a", "b", "tie")):
        return {"error": "need a, b and outcome in a|b|tie"}
    d = _elo_load()
    ra = d["ratings"].get(a, 1200.0)
    rb = d["ratings"].get(b, 1200.0)
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    sa = 1.0 if outcome == "a" else (0.5 if outcome == "tie" else 0.0)
    K = 32.0
    d["ratings"][a] = ra + K * (sa - ea)
    d["ratings"][b] = rb + K * ((1.0 - sa) - (1.0 - ea))
    d["matches"] = d.get("matches", 0) + 1
    try:
        json.dump(d, open(ELO_PATH, "w"), indent=1)
    except Exception:
        pass
    return api_elo_get()


# ---------------------------------------------------------------------------
# Draft-head training endpoint (phone-guarded: refuses heavy models)
# ---------------------------------------------------------------------------

def api_train_draft(model_id: str) -> dict:
    """Train EAGLE + MTP heads for a LOADED dense model via scripts/train_eagle.
    Phone-safe guard: refuses models whose weights exceed the RAM budget.
    Empty id = the currently loaded model."""
    if CHAT["model"] is None:
        return {"error": "load a model first"}
    model_id = model_id or (CHAT.get("model_id") or "")
    model = CHAT["model"]
    n_params = sum(v.size for v in model.t.values())
    working_mb = n_params * 2 / 1e6
    avail = _mem_available_mb()
    if avail is not None and working_mb * 2 > avail:
        return {"error": f"too heavy for on-device training "
                f"(~{working_mb*2:.0f}MB needed, {avail}MB free)"}
    mdir = None
    for d in MODELS_DIRS:
        cand = os.path.join(d, model_id)
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "model.safetensors")):
            mdir = cand
            break
    if mdir is None:
        return {"error": "training needs the original HF dir on disk"}
    import subprocess as _sp
    import sys as _sys
    r = _sp.run([_sys.executable,
                 os.path.join(ROOT, "scripts", "train_eagle.py"),
                 "--model-dir", mdir, "--steps", "160", "--iters", "120"],
                capture_output=True, text=True, timeout=600, cwd=ROOT)
    tail = (r.stdout or "").strip().splitlines()[-3:]
    ok = r.returncode == 0
    return {"ok": ok, "log": tail,
            "heads": ["eagle_head.npz", "mtp_stack.npz"] if ok else []}



# ---------------------------------------------------------------------------
# Model library: a curated catalog with per-machine fit checks + measured
# decode speed (Ollama-style library, honest fit labels)
# ---------------------------------------------------------------------------

CATALOG = [
    {"id": "Qwen2.5-0.5B-Instruct", "repo": "Qwen/Qwen2.5-0.5B-Instruct",
     "params_b": 0.49, "arch": "qwen2", "license": "qwen-2.5",
     "tokenizer": "native (vocab.json + merges.txt)",
     "blurb": "Smallest Qwen2.5 instruct — the reference small chat model."},
    {"id": "Qwen2.5-0.5B", "repo": "Qwen/Qwen2.5-0.5B",
     "params_b": 0.49, "arch": "qwen2", "license": "qwen-2.5",
     "tokenizer": "native",
     "blurb": "Base twin of the instruct model — the one this repo's tests run on."},
    {"id": "Qwen2.5-1.5B-Instruct", "repo": "Qwen/Qwen2.5-1.5B-Instruct",
     "params_b": 1.54, "arch": "qwen2", "license": "qwen-2.5",
     "tokenizer": "native",
     "blurb": "Strong small chat model; fp16 wants a laptop, not a phone."},
    {"id": "SmolLM2-135M-Instruct", "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
     "params_b": 0.135, "arch": "llama", "license": "apache-2.0",
     "tokenizer": "partial (tokenizer.json; byte-fallback)",
     "blurb": "Tiny instruct model — instant loads, toy quality."},
    {"id": "SmolLM2-360M-Instruct", "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
     "params_b": 0.36, "arch": "llama", "license": "apache-2.0",
     "tokenizer": "partial",
     "blurb": "Small instruct step-up; still phone-friendly in fp16."},
    {"id": "TinyLlama-1.1B-Chat", "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     "params_b": 1.1, "arch": "llama", "license": "apache-2.0",
     "tokenizer": "partial (sentencepiece; byte-fallback)",
     "blurb": "Classic 1.1B chat llama; convert to Lumina int4 for phones."},
]

BENCH_PATH = os.path.join(HERE, "load_bench.json")


def _bench_load() -> dict:
    try:
        return json.load(open(BENCH_PATH))
    except Exception:
        return {}


def _bench_save(d: dict) -> None:
    try:
        json.dump(d, open(BENCH_PATH, "w"), indent=1)
    except Exception:
        pass


def api_catalog() -> dict:
    """Curated library with per-machine fit and measured decode speed.
    Fit = fp16 working set (params*2 bytes * 2.5 headroom) vs free RAM:
    'fits-fp16' | 'needs-conversion' (int4 path) | 'too-big'."""
    avail = _mem_available_mb()
    bench = _bench_load()
    lib_ids = set()
    try:
        lib_ids = {x["id"] for x in api_library()["items"]}
    except Exception:
        pass
    out = []
    for e in CATALOG:
        fp16_mb = e["params_b"] * 1000 * 2.0
        min_ram = fp16_mb * 2.5
        int4_mb = e["params_b"] * 1000 * 0.6
        if avail is None:
            fit = "unknown"
        elif avail >= min_ram:
            fit = "fits-fp16"
        elif avail >= int4_mb * 2.5:
            fit = "needs-conversion"
        else:
            fit = "too-big"
        out.append({**e, "fp16_mb": round(fp16_mb), "min_ram_mb": round(min_ram),
                    "fit": fit, "downloaded": e["id"] in lib_ids,
                    "measured_tps": bench.get(e["id"])})
    return {"catalog": out, "available_mb": avail,
            "note": "fit is an fp16 estimate; conversion to Lumina int4 "
                    "shrinks weights ~4x — convert in Work after download"}


def _quick_bench(model_id: str) -> float | None:
    """8-token decode on the loaded model; honest, budget-guarded."""
    try:
        import numpy as np
        from lumina_framework.chat import generate
        model = CHAT["model"]
        if model is None:
            return None
        r = generate(model, "bench", max_tokens=8, budget_s=30.0,
                     rng=np.random.default_rng(0))
        return round(r.get("tokens_per_second", 0.0), 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Embeddings API (input-embedding mean pool — labeled honestly, not a
# trained sentence embedder) + RAG hybrid scoring
# ---------------------------------------------------------------------------

def _embed_text(text: str) -> np.ndarray | None:
    """Mean-pool the loaded model's embedding rows over BPE ids."""
    if CHAT["model"] is None:
        return None
    try:
        import numpy as np
        model = CHAT["model"]
        ids = model.tok.encode(text[:4000])
        if not ids:
            return None
        emb = np.asarray(model.t["model.embed_tokens.weight"],
                         dtype=np.float32)
        ids = [i % emb.shape[0] for i in ids]
        v = emb[ids].mean(axis=0)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-9 else None
    except Exception:
        return None


def api_embeddings(body: dict) -> dict:
    """POST /v1/embeddings — OpenAI-shaped. Uses the LOADED model's input
    embedding matrix, mean-pooled. Honest label: this is the model's token
    embedding average, not a contrastively trained sentence embedder."""
    inp = body.get("input", "")
    if isinstance(inp, list):
        inp = " ".join(str(x) for x in inp)
    if not str(inp).strip():
        return {"error": {"message": "input required", "type": "invalid_request_error"}}
    if CHAT["model"] is None:
        return {"error": {"message": "no model loaded — embeddings use the "
                                      "loaded model's embedding matrix",
                          "type": "server_error"}}
    v = _embed_text(str(inp))
    if v is None:
        return {"error": {"message": "embedding failed", "type": "server_error"}}
    import base64 as _b64
    return {"object": "list",
            "data": [{"object": "embedding", "index": 0,
                      "embedding": [round(float(x), 6) for x in v]}],
            "model": CHAT.get("model_id") or "lumina",
            "note": "input-embedding mean pool (not a trained sentence embedder)"}



# ---------------------------------------------------------------------------
# MCP endpoints (registry is lazy: nothing starts until asked)
# ---------------------------------------------------------------------------

_MCP: dict = {"reg": None, "by_id": {}}


def _mcp_registry():
    if _MCP["reg"] is None:
        import mcp_registry as MR
        reg = MR.build_registry()
        try:  # user-authored servers ride the same lazy machinery
            import custom_mcp as CM
            reg = reg + CM.entries()
        except Exception:
            pass
        _MCP["reg"] = reg
        _MCP["by_id"] = {e["id"]: e for e in reg}
    return _MCP


def api_mcp_registry(q: str, cat: str) -> dict:
    reg = _mcp_registry()["reg"]
    q = (q or "").lower()
    cat = (cat or "").lower()
    hits = [e for e in reg
            if (not q or q in e["id"].lower() or q in e["desc"].lower()
                or q in e["name"].lower())
            and (not cat or cat == e["category"])]
    curated = sum(1 for e in reg if e["variant"] is None
                  and e.get("source") != "custom")
    custom = sum(1 for e in reg if e.get("source") == "custom")
    cats = {}
    for e in reg:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    return {"total": len(reg), "curated_servers": curated,
            "preconfigured_variants": len(reg) - curated - custom,
            "custom_servers": custom,
            "categories": cats, "items": hits[:80],
            "note": ("variants are preconfigured instances of curated "
                     "servers (scopes/paths/DSNs), not distinct software; "
                     "every server is LAZY — nothing runs until started, and "
                     "starting requires its runtime (npx/uvx/...) installed "
                     "here; missing runtimes return the install hint"),
            "running": sorted(MR_STATUS())}


def MR_STATUS():
    from mcp_registry import MANAGER
    return MANAGER.status()["running"]


def api_mcp_status() -> dict:
    from mcp_registry import MANAGER
    return MANAGER.status()


def api_mcp_start(entry_id: str) -> dict:
    ent = _mcp_registry()["by_id"].get(entry_id)
    if ent is None:
        return {"error": "unknown mcp id: " + entry_id[:80]}
    from mcp_registry import MANAGER
    r = MANAGER.start(ent)
    if "error" not in r:
        tools = MANAGER.list_tools(entry_id)
        r["tools"] = tools.get("tools", tools.get("error", []))
    return r


def api_mcp_stop(entry_id: str) -> dict:
    from mcp_registry import MANAGER
    return MANAGER.stop(entry_id)


def api_mcp_tools(entry_id: str) -> dict:
    from mcp_registry import MANAGER
    return MANAGER.list_tools(entry_id)


def api_mcp_call(body: dict) -> dict:
    entry_id = str(body.get("id", ""))
    tool = str(body.get("tool", ""))
    args = body.get("args") or {}
    if not (entry_id and tool):
        return {"error": "need id and tool"}
    from mcp_registry import MANAGER
    st = MANAGER.status()
    if entry_id not in st["running"]:
        ent = _mcp_registry()["by_id"].get(entry_id)
        if ent is None:
            return {"error": "unknown mcp id"}
        r = MANAGER.start(ent)
        if "error" in r:
            return r
    return MANAGER.call(entry_id, tool, args)


# ---------------------------------------------------------------------------
# Skill library: reusable prompt workflows
# ---------------------------------------------------------------------------

SKILLS: list[dict] = [
    {"id": "summarize", "category": "writing", "icon": "doc",
     "desc": "Condense text into a tight summary with key points.",
     "prompt": "Summarize the following text in {n} bullet points, then one sentence.\n\n---\n{input}\n---",
     "vars": {"n": "5"}},
    {"id": "extract-json", "category": "data", "icon": "data",
     "desc": "Pull structured data out of messy text as JSON.",
     "prompt": "Extract the key facts from the text below as a JSON object. Only JSON, no prose.\n\n---\n{input}\n---"},
    {"id": "code-review", "category": "dev", "icon": "code",
     "desc": "Review code for bugs, edge cases, and clarity.",
     "prompt": "Review this code. List: (1) bugs, (2) edge cases, (3) clarity issues. Be specific, cite lines.\n\n```\n{input}\n```"},
    {"id": "unit-tests", "category": "dev", "icon": "code",
     "desc": "Write unit tests for a function.",
     "prompt": "Write unit tests for the following code. Cover the happy path, boundaries, and failure modes. Output only code.\n\n```\n{input}\n```"},
    {"id": "commit-message", "category": "dev", "icon": "code",
     "desc": "Turn a diff into a clean commit message.",
     "prompt": "Write a one-line commit message (imperative, under 72 chars) for this diff, then 3 bullet details if needed.\n\n```\n{input}\n```"},
    {"id": "release-notes", "category": "dev", "icon": "doc",
     "desc": "Generate release notes from commit history.",
     "prompt": "Write release notes from these commits: group by Added/Changed/Fixed.\n\n---\n{input}\n---"},
    {"id": "sql-explain", "category": "data", "icon": "data",
     "desc": "Explain a SQL query step by step.",
     "prompt": "Explain this SQL query step by step, then note performance risks.\n\n```sql\n{input}\n```"},
    {"id": "csv-insights", "category": "data", "icon": "data",
     "desc": "Surface insights from CSV rows.",
     "prompt": "Here are CSV rows. List the columns, data quality issues, and 5 insights with evidence.\n\n---\n{input}\n---"},
    {"id": "regex-explain", "category": "dev", "icon": "code",
     "desc": "Decode a regular expression.",
     "prompt": "Explain this regex token by token, then give 2 example matches and 2 non-matches.\n\n```\n{input}\n```"},
    {"id": "regex-gen", "category": "dev", "icon": "code",
     "desc": "Write a regex from a plain-language spec.",
     "prompt": "Write a regex that matches: {input}\nGive the pattern, a breakdown, and 3 test cases."},
    {"id": "translate", "category": "writing", "icon": "globe",
     "desc": "Translate text, keeping tone.",
     "prompt": "Translate the following into {lang}, keeping the tone.\n\n---\n{input}\n---",
     "vars": {"lang": "English"}},
    {"id": "rewrite-tight", "category": "writing", "icon": "pen",
     "desc": "Cut wordiness without losing meaning.",
     "prompt": "Rewrite the text below to be 30% shorter with the same meaning. No filler.\n\n---\n{input}\n---"},
    {"id": "grammar", "category": "writing", "icon": "pen",
     "desc": "Fix grammar and clarity, minimally.",
     "prompt": "Fix grammar and clarity in this text. Change as little as possible. Output the corrected text only.\n\n---\n{input}\n---"},
    {"id": "proofread", "category": "writing", "icon": "pen",
     "desc": "Deep proofread with change list.",
     "prompt": "Proofread the text. Output the corrected text, then a numbered list of every change and why.\n\n---\n{input}\n---"},
    {"id": "plan-features", "category": "thinking", "icon": "layers",
     "desc": "Break a feature request into a build plan.",
     "prompt": "Break this feature request into a concrete build plan: milestones, tasks, risks.\n\n---\n{input}\n---"},
    {"id": "decision-log", "category": "thinking", "icon": "clock",
     "desc": "Turn a decision into a documented record.",
     "prompt": "Write a decision record for: {input}\nSections: context, options considered, decision, consequences."},
    {"id": "meeting-notes", "category": "writing", "icon": "doc",
     "desc": "Structure raw meeting notes.",
     "prompt": "Structure these raw notes: decisions, action items (owner + verb), open questions.\n\n---\n{input}\n---"},
    {"id": "api-review", "category": "dev", "icon": "globe",
     "desc": "Review an API design.",
     "prompt": "Review this API design: naming, consistency, error handling, versioning, security.\n\n---\n{input}\n---"},
    {"id": "docstring", "category": "dev", "icon": "doc",
     "desc": "Write a docstring for code.",
     "prompt": "Write a clear docstring for this code: purpose, args, returns, raises, one example.\n\n```\n{input}\n```"},
    {"id": "bug-report", "category": "dev", "icon": "bolt",
     "desc": "Structure a bug from symptoms.",
     "prompt": "Turn these symptoms into a structured bug report: expected, actual, steps, suspected cause, severity.\n\n---\n{input}\n---"},
    {"id": "markdown-table", "category": "data", "icon": "grid",
     "desc": "Convert raw data into a Markdown table.",
     "prompt": "Convert the data below into a clean Markdown table. Keep it faithful.\n\n---\n{input}\n---"},
    {"id": "cron-gen", "category": "dev", "icon": "clock",
     "desc": "Write a cron expression from a schedule.",
     "prompt": "Write a cron expression (and a human sentence explaining it) for: {input}"},
    {"id": "yaml-fix", "category": "dev", "icon": "code",
     "desc": "Validate and fix YAML.",
     "prompt": "Find and fix errors in this YAML. Output the corrected YAML only, then a list of changes.\n\n```yaml\n{input}\n```"},
    {"id": "python-trace", "category": "dev", "icon": "term",
     "desc": "Trace code execution step by step.",
     "prompt": "Trace this code with the given input, line by line, then state the output.\n\n```\n{input}\n```"},
    {"id": "complexity-audit", "category": "dev", "icon": "flask",
     "desc": "Audit complexity and hot spots.",
     "prompt": "Identify the computational complexity and likely hot spots in this code.\n\n```\n{input}\n```"},
    {"id": "prompt-improve", "category": "thinking", "icon": "pen",
     "desc": "Strengthen a prompt.",
     "prompt": "Improve this prompt: make it specific, add constraints and output format.\n\n---\n{input}\n---"},
    {"id": "simplify", "category": "thinking", "icon": "layers",
     "desc": "Explain something simply, then precisely.",
     "prompt": "Explain this in one simple sentence anyone gets, then precisely for an expert.\n\n---\n{input}\n---"},
    {"id": "changelog", "category": "dev", "icon": "doc",
     "desc": "Maintain a changelog from changes.",
     "prompt": "Update this changelog (Keep a Changelog format) from the changes below.\n\n---\n{input}\n---"},
    {"id": "email-draft", "category": "writing", "icon": "chat",
     "desc": "Draft a short professional email.",
     "prompt": "Draft a short professional email for: {input}\nTone: direct, warm, no filler."},
    {"id": "todo-extract", "category": "thinking", "icon": "clock",
     "desc": "Pull action items from any text.",
     "prompt": "Extract every action item from the text. Each: verb-first, with owner if present.\n\n---\n{input}\n---"},
]


def api_skills_list() -> dict:
    return {"skills": _all_skills(),
            "builtin": len(SKILLS), "user": len(_user_skills()),
            "note": ("click a skill, paste input, the prompt renders into the "
                     "composer; user skills are editable/deletable")}


def api_skills_run(body: dict) -> dict:
    """Render a skill with input and run it through the loaded model.
    Returns the prompt too, so the UI can show exactly what was sent."""
    sid = str(body.get("skill", ""))
    skill = next((x for x in _all_skills() if x["id"] == sid), None)
    if skill is None:
        return {"error": "unknown skill: " + sid[:60]}
    inp = str(body.get("input", ""))
    if not inp.strip():
        return {"error": "skill needs input text"}
    prompt = skill["prompt"].replace("{input}", inp[:6000])
    for k, v in (skill.get("vars") or {}).items():
        prompt = prompt.replace("{" + k + "}", str(body.get("vars", {}).get(k, v)))
    r = api_chat([{"role": "user", "content": prompt}],
                 int(body.get("max_tokens", 128)), 0.0)
    r["skill"] = sid
    r["prompt"] = prompt
    return r



# ---------------------------------------------------------------------------
# Workflow execution engine (ruflo-style): typed step DAGs with retries,
# condition branches, and honest per-step reporting. Every step records
# seconds, status and output; nothing is claimed that did not run.
# ---------------------------------------------------------------------------

WORKFLOW_STEP_KINDS = ("chat", "skill", "exec", "fetch_url", "mcp",
                       "browser", "fs", "app", "branch", "set", "echo")

WORKFLOW_EXAMPLES = [
    {"id": "research-and-summarize",
     "name": "Research + summarize",
     "steps": [
         {"kind": "fetch_url", "id": "fetch", "url": "https://example.com"},
         {"kind": "skill", "id": "sum", "skill": "summarize", "input": "${fetch.text}"},
         {"kind": "branch", "id": "ok", "cond": "${sum.ok}", "then": [{"kind": "echo", "text": "done"}]}]},
    {"id": "code-review-flow",
     "name": "Code review pipeline",
     "steps": [
         {"kind": "skill", "id": "review", "skill": "code-review", "input": "${input.code}"},
         {"kind": "chat", "id": "plan", "prompt": "Given review:\n${review.text}\nWrite fix tasks."}]},
]


def api_workflow_examples() -> dict:
    return {"kinds": WORKFLOW_STEP_KINDS, "examples": WORKFLOW_EXAMPLES,
            "vars": "text templates ${step.output} resolve from earlier steps; "
                    "${input.x} from the request input; branch.cond must be "
                    "'true' to continue its then-steps"}


def _wf_resolve(template, ctx: dict) -> str:
    import re as _re
    out = str(template)
    for m in set(_re.findall(r"\$\{([\w.]+)\}", out)):
        cur = ctx
        ok = True
        for part in m.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        out = out.replace("${" + m + "}", str(cur)[:4000] if ok else "")
    return out


def _wf_run_step(step: dict, ctx: dict, trace: list, depth: int) -> None:
    import time as _t
    if depth > 8:
        trace.append({"step": step.get("id"), "status": "error",
                      "error": "nesting depth limit"})
        return
    kind = str(step.get("kind", ""))
    t0 = _t.perf_counter()
    out: dict = {}
    status = "ok"
    if kind == "chat":
        r = api_chat([{"role": "user", "content": _wf_resolve(step.get("prompt", ""), ctx)}],
                     int(step.get("max_tokens", 96)), 0.0)
        out = r if "error" not in r else {"error": r["error"]}
        if "error" in out:
            status = "error"
    elif kind == "skill":
        r = api_skills_run({"skill": step.get("skill", ""),
                             "input": _wf_resolve(step.get("input", ""), ctx)})
        out = r if "error" not in r else {"error": r["error"]}
        if "error" in out:
            status = "error"
    elif kind == "exec":
        r = api_exec(_wf_resolve(step.get("cmd", ""), ctx),
                     step.get("cwd"), int(step.get("timeout", 60)))
        out = r
        if "error" in r:
            status = "error"
        elif r.get("code", 0) != 0:
            if step.get("allow_fail"):
                status = "ok"
                out = {**r, "note": "non-zero exit allowed (allow_fail)"}
            else:
                status = "error"
                out = {**r, "error": "exit code %s: %s"
                       % (r.get("code"), str(r.get("out", ""))[-160:])}
    elif kind == "fetch_url":
        r = api_fetch_url(_wf_resolve(step.get("url", ""), ctx))
        out = r
        status = "ok" if "text" in r else "error"
    elif kind == "mcp":
        r = api_mcp_call({"id": step.get("id2", step.get("mcp", "")),
                          "tool": step.get("tool", ""),
                          "args": step.get("args", {})})
        out = r
        status = "ok" if "error" not in r else "error"
    elif kind == "branch":
        cond = str(_wf_resolve(str(step.get("cond", "")), ctx)).strip().lower()
        if cond in ("true", "1", "yes", "ok"):
            for sub in step.get("then", [])[:12]:
                _wf_run_step(sub, ctx, trace, depth + 1)
        else:
            for sub in step.get("else", [])[:12]:
                _wf_run_step(sub, ctx, trace, depth + 1)
        trace.append({"step": step.get("id"), "kind": kind,
                      "cond": cond, "took": True,
                      "status": "ok",
                      "seconds": round(_t.perf_counter() - t0, 3)})
        return
    elif kind == "browser":
        args = {k: _wf_resolve(v, ctx) for k, v in (step.get("args") or {}).items()}
        out = api_browser({"op": step.get("op", "text"), **args})
        status = "ok" if "error" not in out else "error"
    elif kind == "fs":
        fb = {k: (_wf_resolve(v, ctx) if isinstance(v, str) else v)
              for k, v in step.items() if k not in ("kind", "id", "retry")}
        out = api_fs(fb)
        status = "ok" if "error" not in out else "error"
    elif kind == "app":
        out = api_apps({"op": "launch", "app": _wf_resolve(step.get("app", ""), ctx),
                        "args": step.get("args") or []})
        status = "ok" if "error" not in out else "error"
    elif kind == "set":
        for k, v in (step.get("vars") or {}).items():
            ctx[k] = _wf_resolve(v, ctx)
    elif kind == "echo":
        ctx.setdefault("echo", []).append(_wf_resolve(step.get("text", ""), ctx))
    else:
        status = "error"
        out = {"error": "unknown step kind: " + kind}
    rec = {"step": step.get("id"), "kind": kind, "status": status,
           "seconds": round(_t.perf_counter() - t0, 3)}
    if status == "ok" and step.get("id"):
        ctx[str(step["id"])] = out
    if status == "error":
        rec["error"] = str(out.get("error", ""))[:200]
    trace.append(rec)


def api_workflow_run(body: dict) -> dict:
    """POST /api/workflow/run {steps: [...], input: {...}} — ruflo-style
    typed step execution. Retries: step.retry (<=2) reruns failed steps.
    Stops at first hard error unless step.continue_on_error."""
    import time as _t
    steps = body.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return {"error": "steps[] required (see /api/workflow examples)"}
    ctx = {"input": body.get("input") or {}}
    trace: list = []
    t0 = _t.perf_counter()
    for step in steps[:40]:
        if not isinstance(step, dict):
            trace.append({"step": None, "status": "error", "error": "step not an object"})
            continue
        retries = min(int(step.get("retry", 0)), 2)
        for attempt in range(retries + 1):
            before = len(trace)
            _wf_run_step(step, ctx, trace, 0)
            rec = trace[-1] if len(trace) > before else {}
            if rec.get("status") != "error" or attempt == retries:
                if rec.get("status") == "error" and not step.get("continue_on_error"):
                    return {"trace": trace, "stopped_at": step.get("id"),
                            "input_echo": ctx.get("input"),
                            "seconds": round(_t.perf_counter() - t0, 3),
                            "ok": False}
                break
    return {"trace": trace, "vars": {k: (v if isinstance(v, (int, float, bool)) else
                                          str(v)[:200])
                                     for k, v in ctx.items() if k != "input"},
            "seconds": round(_t.perf_counter() - t0, 3), "ok": True}




# ---------------------------------------------------------------------------
# Computer access: filesystem + browser + app launch (operator-gated) and
# custom MCP / custom skill authoring. Every operator action is audited.
# ---------------------------------------------------------------------------

def _op_gate() -> dict | None:
    """None when operator mode is on, else the refusal the caller returns."""
    m = _operator_mode()
    if m.get("full_access"):
        return None
    return {"error": ("operator mode is off — enable it in Settings > Computer "
                      "(loopback free; LAN bind needs an API token)"),
            "operator": m}


def api_fs(body: dict) -> dict:
    """Whole-computer filesystem: list/read/write/delete/move/mkdir/search."""
    g = _op_gate()
    if g:
        return g
    import sysops
    return sysops.fs_op(body)


def api_browser(body: dict) -> dict:
    """Drive a real Chromium (navigate/click/fill/type like a person)."""
    g = _op_gate()
    if g:
        return g
    import browser_driver
    return browser_driver.browser_op(body)


def api_apps(body: dict | None = None) -> dict:
    """Launch desktop apps / list what exists / read the audit trail."""
    import sysops
    b = body or {}
    op = str(b.get("op", "list"))
    if op == "list":
        return {"apps": sysops.app_list(), "operator": _operator_mode(),
                "launch_allowlist": sorted(sysops.LAUNCH_ALLOW)}
    if op == "launch":
        g = _op_gate()
        if g:
            return g
        return sysops.launch_app(str(b.get("app", "")), b.get("args") or [])
    if op == "audit":
        return {"events": sysops.audit_tail(int(b.get("n", 50)))}
    if op == "run":
        g = _op_gate()
        if g:
            return g
        return sysops.run_any(str(b.get("cmd", "")), b.get("cwd"),
                              int(b.get("timeout", 60)))
    return {"error": "op must be list|launch|audit|run"}


def api_builtin_tools() -> dict:
    """Agent tools Lumina always has, plus operator-only ones with their gate."""
    import browser_driver
    op_on = bool(_operator_mode().get("full_access"))
    base = [{"tool": "exec", "gate": "allowlist" if not op_on else "operator",
             "args": {"cmd": "shell command"}},
            {"tool": "fetch_url", "gate": "open", "args": {"url": "http..."}},
            {"tool": "skill_fetch", "gate": "open",
             "args": {"query": "task description"}},
            {"tool": "media_status", "gate": "open", "args": {}},
            {"tool": "media_generate", "gate": "open",
             "args": {"capability": "image", "prompt": "..."}},
            {"tool": "mcp:<server>::<tool>", "gate": "lazy mcp start", "args": {}}]
    op_tools = [{"tool": t["name"], "gate": "operator", "args": t["args"]}
                for t in browser_driver.tool_list()]
    op_tools += [{"tool": "fs_list", "gate": "operator", "args": {"path": "~"}},
                 {"tool": "fs_read", "gate": "operator", "args": {"path": "..."}},
                 {"tool": "fs_write", "gate": "operator",
                  "args": {"path": "...", "text": "..."}},
                 {"tool": "fs_search", "gate": "operator",
                  "args": {"path": "~", "query": "..."}},
                 {"tool": "app_open", "gate": "operator", "args": {"app": "firefox"}},
                 {"tool": "app_list", "gate": "operator", "args": {}}]
    return {"open_tools": base, "operator_tools": op_tools,
            "operator": _operator_mode(),
            "browser": browser_driver.status()}


def api_atoms_list() -> dict:
    """Typed atomic units (Atomic Agents-inspired LEGO blocks): every tool and
    skill with its declared input contract."""
    import agent_grammar as _AG
    tools = []
    for s in _agent_base_tool_specs():
        t = str(s.get("tool", ""))
        sch = _AG.schema_for(t)
        tools.append({
            "id": t, "kind": "tool", "gate": s.get("gate", "open"),
            "schema": {k: {"type": v[0],
                           "required": bool(v[1]) if len(v) > 1 else False,
                           **({"default": v[2]} if len(v) > 2 else {})}
                       for k, v in sch.items()},
            "description": _AGENT_TOOL_DESCRIPTIONS.get(t, "")})
    skills = [{"id": x.get("id"), "kind": "skill", "desc": x.get("desc", ""),
               "vars": x.get("vars") or {}} for x in _all_skills()]
    return {"atoms": tools + skills, "tools": len(tools),
            "skills": len(skills),
            "note": ("typed blocks: each unit declares its input contract "
                     "(Atomic Agents-inspired); MCP tools stay permissive")}


def api_atoms_run(body: dict) -> dict:
    """Run one atom by id with schema validation + repair (honest)."""
    import agent_grammar as _AG
    body = body if isinstance(body, dict) else {}
    aid = str(body.get("id", ""))
    args = body.get("args") if isinstance(body.get("args"), dict) else {}
    if not aid:
        return {"error": "id required"}
    known_tools = {str(s.get("tool", "")) for s in _agent_base_tool_specs()}
    is_tool = (aid in known_tools or aid.startswith("browser_")
               or aid.startswith("mcp:"))
    if not is_tool and not any(x.get("id") == aid for x in _all_skills()):
        return {"error": "unknown atom: " + aid[:60]}
    v = _AG.validate_call({"tool": aid, "args": args}, None)
    rep = _AG.repair_call({"tool": aid, "args": args})
    call = rep["call"]
    if is_tool:
        out = _agent_tool_call(call)
        return {"atom": aid, "kind": "tool", "out": out,
                "repairs": rep["repairs"], "validation_errors": v["errors"]}
    sk = next(x for x in _all_skills() if x.get("id") == aid)
    prompt = str(sk.get("prompt", "")).replace("{input}",
                                                str(args.get("input", "")))
    return {"atom": aid, "kind": "skill", "rendered": prompt[:4000],
            "repairs": rep["repairs"], "validation_errors": v["errors"]}


def _mcp_invalidate() -> None:
    """Custom MCPs changed: rebuild the lazy registry (nothing is running)."""
    _MCP["reg"] = None
    _MCP["by_id"] = {}


def api_custom_mcp(body: dict) -> dict:
    """Author your own MCP servers: list|save|delete|test (real JSON-RPC)."""
    import custom_mcp as CM
    op = str(body.get("op", "list"))
    if op == "list":
        out = CM.list_custom()
        out["example"] = CM.example()
        out["registered"] = [e for e in _mcp_registry()["reg"]
                             if e.get("source") == "custom"]
        return out
    if op == "save":
        r = CM.save_custom(body.get("server") or {})
        if "error" not in r:
            _mcp_invalidate()
        return r
    if op == "delete":
        r = CM.delete_custom(str(body.get("id", "")))
        if "error" not in r:
            _mcp_invalidate()
        return r
    if op == "test":
        sid = "custom/" + str(body.get("id", "")).strip().lower()
        _mcp_invalidate()
        entry = _mcp_registry()["by_id"].get(sid)
        if entry is None:
            return {"error": "no such custom MCP: " + sid}
        from mcp_registry import MANAGER
        MANAGER.stop(sid)
        st = MANAGER.start(entry)
        if "error" in st:
            return {"error": st["error"], "entry": sid}
        init = MANAGER._rpc(sid, "initialize",
                            {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "lumina", "version": "0.1"}})
        tools = MANAGER.list_tools(sid)
        sample = None
        names = [t.get("name") for t in tools.get("tools", [])]
        if names:
            pick = "echo" if "echo" in names else names[0]
            sample = MANAGER.call(sid, pick,
                                  body.get("args") or {"text": "ping",
                                                       "path": "/", "url": "https://example.com"})
        return {"entry": sid, "started": st, "initialize": init,
                "tools": tools, "sample_call": sample,
                "note": "real stdio JSON-RPC round-trip against the generated file"}
    return {"error": "op must be list|save|delete|test"}


USER_SKILLS_PATH = os.path.join(HERE, "user_skills.json")


def _user_skills() -> list[dict]:
    try:
        d = json.load(open(USER_SKILLS_PATH))
        return [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []
    except Exception:
        return []


def _all_skills() -> list[dict]:
    return SKILLS + _user_skills()


# -- skill retrieval index (compact BM25; see skill_index.py) ----------------
_SKILL_INDEX: dict = {"idx": None, "stamp": None}


def _skill_index():
    """Singleton BM25 index over builtin + user skills. Rebuilt only when
    the skill set changes (mtime of user_skills.json or explicit
    invalidation); building is single-digit ms at this corpus size."""
    import skill_index as _SI
    skills = _all_skills()
    try:
        _ust = os.path.getmtime(USER_SKILLS_PATH)
    except OSError:
        _ust = 0.0
    stamp = (len(SKILLS), _ust)
    if _SKILL_INDEX["idx"] is None or _SKILL_INDEX["stamp"] != stamp:
        idx = _SI.SkillIndex()
        idx.build(skills)
        _SKILL_INDEX["idx"] = idx
        _SKILL_INDEX["stamp"] = stamp
    return _SKILL_INDEX["idx"]


def _skill_index_invalidate() -> None:
    _SKILL_INDEX["idx"] = None


def api_skills_search(q: str = "", k: int = 5) -> dict:
    """Rank skills for a free-text task query, with calibrated confidence
    (margin gate; see tests/test_skill_index.py for measured accuracy).
    Used by the /api/skills/search route, the skill_fetch agent tool and
    the opt-in chat auto-fetch."""
    q = str(q or "").strip()[:300]
    if not q:
        return {"error": "query required (q=...)"}
    try:
        k = max(1, min(int(k), 20))
    except (TypeError, ValueError):
        k = 5
    idx = _skill_index()
    r = idx.search(q, k)
    if "error" in r:
        return r
    r["docs"] = len(_all_skills())
    r["index_bytes"] = idx.approx_bytes()
    return r


def api_skills_save(body: dict) -> dict:
    """Create/update a user skill (built-ins stay read-only)."""
    import re as _re
    sid = str(body.get("id", "")).strip().lower()
    if not _re.match(r"^[a-z0-9][a-z0-9_\-]{0,40}$", sid):
        return {"error": "id must be a slug: lowercase letters/digits/_/-"}
    if any(x["id"] == sid for x in SKILLS):
        return {"error": "that id is a built-in skill (pick another id)"}
    prompt = str(body.get("prompt", ""))
    if not prompt.strip():
        return {"error": "prompt required (use {input} for the text)"}
    skill = {"id": sid, "category": str(body.get("category", "custom"))[:40],
             "icon": str(body.get("icon", "sparkles"))[:40],
             "desc": str(body.get("desc", ""))[:200] or "custom skill",
             "prompt": prompt[:8000], "user": True}
    if isinstance(body.get("vars"), dict):
        skill["vars"] = {str(k)[:40]: str(v)[:200]
                         for k, v in list(body["vars"].items())[:8]}
    skills = [x for x in _user_skills() if x.get("id") != sid] + [skill]
    try:
        json.dump(skills, open(USER_SKILLS_PATH, "w"), indent=1)
    except Exception as e:
        return {"error": str(e)[:150]}
    _skill_index_invalidate()
    return {"saved": sid, "user_skills": len(skills),
            "has_input_var": "{input}" in prompt}


def api_skills_delete(body: dict) -> dict:
    sid = str(body.get("id", "")).strip().lower()
    before = _user_skills()
    after = [x for x in before if x.get("id") != sid]
    if len(after) == len(before):
        return {"error": "not a user skill: " + sid}
    try:
        json.dump(after, open(USER_SKILLS_PATH, "w"), indent=1)
    except Exception as e:
        return {"error": str(e)[:150]}
    _skill_index_invalidate()
    return {"deleted": sid, "user_skills": len(after)}



# ---------------------------------------------------------------------------
# Anthropic-compatible /v1/messages
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Memory plugin (top-tier pattern, modified for Lumina): Letta/MemGPT-style
# tiered long-term memory with retrieval, auto fact extraction, and prompt
# injection. Core tier = always in the system prompt; archival tier =
# BM25 + embedding retrieval on demand. The model manages memory through
# the /api/memory3 tools; chat auto-injects the core tier.
# ---------------------------------------------------------------------------

MEM3_PATH = os.path.join(HERE, "memory_core.json")
SQLMEM_PATH = os.path.join(HERE, "memory_sqlite.db")
_SQLMEM: dict = {"m": None}
_SQLMEM_STATS: dict = {"injections": 0, "tokens": 0}
_SQLMEM_LAST: dict = {"ctx": None}


def _sqlmem():
    """Lazy singleton for the SQLite FTS memory tier (stdlib-only)."""
    import memory_sqlite as _MS
    if _SQLMEM["m"] is None:
        try:
            _SQLMEM["m"] = _MS.SQLiteMemory(SQLMEM_PATH)
        except Exception:
            _SQLMEM["m"] = None
    return _SQLMEM["m"]


def _sqlmem_invalidate() -> None:
    _m = _SQLMEM["m"]
    if _m is not None:
        try:
            _m.close()
        except Exception:  # noqa: BLE001
            pass
    _SQLMEM["m"] = None


def api_sqlmem_status() -> dict:
    _m = _sqlmem()
    if _m is None:
        return {"available": False, "path": SQLMEM_PATH,
                "stats": None, "injections": _SQLMEM_STATS}
    try:
        return {"available": True, "path": SQLMEM_PATH, "stats": _m.stats(),
                "injections": _SQLMEM_STATS, "last_context": _SQLMEM_LAST["ctx"]}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:120],
                "injections": _SQLMEM_STATS}


def api_sqlmem_op(body: dict) -> dict:
    """add|recall|search|forget|stats|context — same honesty contract."""
    body = body if isinstance(body, dict) else {}
    op = str(body.get("op", "recall"))
    _m = _sqlmem()
    if _m is None:
        return {"error": "sqlite memory unavailable"}
    try:
        if op == "add":
            return _m.add(str(body.get("scope", "core")),
                          str(body.get("kind", "fact")),
                          str(body.get("content", "")),
                          body.get("meta"))
        if op == "recall":
            return _m.recall(str(body.get("q", "")),
                             int(body.get("limit", 6)),
                             int(body.get("budget_tokens", 320)))
        if op == "search":
            return _m.search_scopes(str(body.get("scope") or "") or None,
                                    str(body.get("kind") or "") or None,
                                    int(body.get("limit", 20)))
        if op == "forget":
            return _m.forget(int(body.get("id", 0)))
        if op == "stats":
            return _m.stats()
        if op == "context":
            return _m.tail_block(str(body.get("q", "")),
                                 int(body.get("limit", 6)),
                                 int(body.get("budget_tokens", 320)))
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:120])}
    return {"error": "unknown op: " + op[:40]}
_MEM3: dict = {"core": [], "archival": [], "lock": __import__("threading").Lock()}


def _mem3_load():
    try:
        d = json.load(open(MEM3_PATH))
        _MEM3["core"] = d.get("core", [])
        _MEM3["archival"] = d.get("archival", [])
    except Exception:
        pass


def _mem3_save():
    try:
        json.dump({"core": _MEM3["core"], "archival": _MEM3["archival"]},
                  open(MEM3_PATH, "w"), indent=1)
    except Exception:
        pass


def _mem3_remember(kind: str, text: str, pin: bool) -> dict:
    text = str(text).strip()[:1000]
    if not text:
        return {"error": "empty memory text"}
    with _MEM3["lock"]:
        _mem3_load()
        entry = {"kind": kind, "text": text, "at": time.time(),
                 "pin": bool(pin)}
        if kind == "core":
            _MEM3["core"] = [e for e in _MEM3["core"] if e["text"] != text]
            _MEM3["core"].append(entry)
            if len(_MEM3["core"]) > 64:  # core stays small: it's in every prompt
                _MEM3["core"] = _MEM3["core"][-64:]
        else:
            _MEM3["archival"].append(entry)
        _mem3_save()
    return {"remembered": kind, "text": text[:80]}


def _mem3_recall(query: str, k: int) -> list[dict]:
    """Retrieval over both tiers: BM25-style token overlap + embedding
    cosine when a model is loaded (vectors stored at remember-time)."""
    import numpy as np
    with _MEM3["lock"]:
        _mem3_load()
        pool = [dict(e) for e in _MEM3["core"]] + [dict(e) for e in _MEM3["archival"]]
    if not pool:
        return []
    qt = set(_toks(query))
    qv = _embed_text(query)
    scored = []
    for e in pool:
        et = set(_toks(e["text"]))
        overlap = len(qt & et)
        cos = 0.0
        if qv is not None and e.get("vec") is not None and len(e["vec"]) == len(qv):
            vv = np.asarray(e["vec"], np.float32)
            n = float(np.linalg.norm(vv) * np.linalg.norm(qv))
            cos = float(vv @ qv) / n if n > 1e-9 else 0.0
        s = overlap + 2.0 * cos + (1.0 if e.get("kind") == "core" else 0.0)
        scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    out = []
    for s, e in scored[:k]:
        if s <= 0 and not e.get("pin"):
            continue
        e["score"] = round(s, 3)
        out.append(e)
    return out


def _mem3_context_block(max_items: int = 8) -> str:
    """The injected system block: pinned core + recent archival."""
    with _MEM3["lock"]:
        _mem3_load()
        core = list(_MEM3["core"])
        arch = list(_MEM3["archival"])
    if not core and not arch:
        return ""
    lines = ["[long-term memory]"]
    for e in core[-max_items:]:
        lines.append("- " + e["text"][:160])
    for e in arch[-3:]:
        lines.append("- (recent) " + e["text"][:160])
    return "\n".join(lines)


def api_memory3_op(body: dict) -> dict:
    """POST /api/memory3 {op: remember|recall|forget|clear|context, ...}"""
    op = str(body.get("op", ""))
    if op == "remember":
        vec = None
        text = str(body.get("text", ""))
        if CHAT["model"] is not None:
            v = _embed_text(text)
            if v is not None:
                vec = [round(float(x), 5) for x in v]
        r = _mem3_remember(str(body.get("kind", "archival")), text,
                           bool(body.get("pin")))
        if vec is not None and "error" not in r:
            with _MEM3["lock"]:
                for tier in ("core", "archival"):
                    for e in _MEM3[tier]:
                        if e["text"] == text[:1000]:
                            e["vec"] = vec
                _mem3_save()
        return r
    if op == "recall":
        return {"hits": _mem3_recall(str(body.get("query", "")),
                                     int(body.get("k", 5)))}
    if op == "forget":
        with _MEM3["lock"]:
            _mem3_load()
            t = str(body.get("text", ""))
            n0 = len(_MEM3["core"]) + len(_MEM3["archival"])
            _MEM3["core"] = [e for e in _MEM3["core"] if e["text"] != t]
            _MEM3["archival"] = [e for e in _MEM3["archival"] if e["text"] != t]
            _mem3_save()
            return {"forgotten": n0 - (len(_MEM3["core"]) + len(_MEM3["archival"]))}
    if op == "clear":
        with _MEM3["lock"]:
            _MEM3["core"] = []
            _MEM3["archival"] = []
            _mem3_save()
        return {"cleared": True}
    if op == "context":
        return {"block": _mem3_context_block(),
                "core": len(_MEM3["core"]), "archival": len(_MEM3["archival"])}
    return {"error": "op must be remember|recall|forget|clear|context"}


def api_memory3_autoextract(user_msg: str) -> list[str]:
    """Heuristic fact extraction from user messages (no model call): picks
    statements about stable preferences/facts. Deliberately conservative —
    only 'my name is / i prefer / i always / remember that' patterns."""
    import re as _re
    out = []
    for pat in (r"(?:my name is|i am called|i'm called)\s+([\w\- ]{2,40})",
                r"(?:i prefer|i like|i always|i never)\s+([\w\-, ]{3,60})",
                r"remember (?:that )?([\w\-, ]{3,60})"):
        for m in _re.finditer(pat, user_msg, _re.I):
            fact = m.group(0).strip()[:100]
            if fact.lower() not in out:
                out.append(fact.lower())
    return out[:3]


if __name__ == "__main__":
    sys.exit(main())

