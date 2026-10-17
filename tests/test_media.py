"""Media generation orchestration (app/server/media.py) -- in-suite stub suite.

Phone-light by design: no model is loaded, no real media backend is touched.
Everything runs against a deterministic loopback ComfyUI-shaped stub plus a
real stdlib PNG, so substitution, job routing, output fetching, size caps and
the filename jail are all exercised end to end.

Covers:
  * import contract (media pulls neither serve nor residency at module scope)
  * workflow listing + placeholder scan (3 JSONs, exact placeholders)
  * render_workflow: nested substitution, int coercion, meta stripped
  * required/unknown-placeholder/unknown-id errors
  * backends_status: configured vs reachable, closed port, measured ffmpeg gap
  * pick_backend: preference order, declared caps, config requirement
  * cli_run: operator gate, argv template, rc!=0, missing binary, {out} jail
  * generate: comfy submit -> poll -> fetch -> files (+ a1111 branch)
  * failure modes: HTTP 500, timeout, residency refusal (nothing sent)
  * api_media_file jail + mime, job list/get, agent-facing "text"

Run:  python3 tests/test_media.py
Exit codes: 0 = pass, 1 = fail, 3 = honest skip (no writable /tmp/opencode).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import struct
import sys
import threading
import time
import types
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

import media  # noqa: E402

# Order matters: these two facts are only observable before anything else
# in this process imports serve / residency.
_SERVE_PULLED = "serve" in sys.modules
_RESIDENCY_PULLED = "residency" in sys.modules

import serve  # noqa: E402

MEDIA_TEST_DIR = "/tmp/opencode/media_test"
RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name
          + ((" :: " + str(extra)[:160]) if not cond and extra else ""))


def _make_png(w=8, h=8, rgba=(200, 30, 30, 255)):
    """A real, tiny, stdlib-built PNG (no PIL, no model)."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    row = bytes(rgba) * w
    raw = b"".join(b"\x00" + row for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


PNG_8 = _make_png()


def _multipart_filename(blob: bytes):
    m = re.search(rb'filename="([^"]*)"', blob)
    return m.group(1).decode("utf-8", "replace") if m else ""


class _ComfyStub:
    """Deterministic ComfyUI-shaped endpoint on 127.0.0.1:0."""

    def __init__(self):
        self.mode = "ok"            # "ok" | "500" | "pending"
        self.attempts = 0           # every POST /prompt, recorded or not
        self.prompts = []           # parsed POST /prompt bodies (ok mode)
        self.uploads = []
        self.history_calls = 0
        self.view_calls = []
        self.last_txt2img = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None, ctype="application/json"):
                body = raw if raw is not None else json.dumps(obj or {}).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except Exception:
                    pass

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/queue":
                    return self._send(200, {"queue_running": [],
                                            "queue_pending": []})
                if u.path == "/history/p1":
                    outer.history_calls += 1
                    if outer.mode == "pending" or outer.history_calls == 1:
                        return self._send(200, {})
                    return self._send(200, {"p1": {
                        "status": {"status_str": "success", "completed": True,
                                   "messages": []},
                        "outputs": {"9": {"images": [
                            {"filename": "0001.png", "subfolder": "",
                             "type": "output"}]}}}})
                if u.path == "/view":
                    outer.view_calls.append(parse_qs(u.query))
                    return self._send(200, raw=PNG_8, ctype="image/png")
                if u.path == "/sdapi/v1/sd-models":
                    return self._send(200, [])
                return self._send(404, {"error": "stub: no route " + u.path})

            def do_POST(self):
                p = urlparse(self.path).path
                n = int(self.headers.get("Content-Length", 0) or 0)
                blob = self.rfile.read(n) if n else b""
                if p == "/prompt":
                    outer.attempts += 1
                    if outer.mode == "500":
                        return self._send(500, {"error": "stub forced failure"})
                    try:
                        outer.prompts.append(json.loads(blob or b"{}"))
                    except Exception:
                        outer.prompts.append({"_unparsed": True})
                    return self._send(200, {"prompt_id": "p1", "number": 1})
                if p == "/upload/image":
                    ct = self.headers.get("Content-Type", "")
                    outer.uploads.append({
                        "content_type": ct, "boundary_seen": "boundary=" in ct,
                        "bytes": len(blob),
                        "filename": _multipart_filename(blob),
                        "has_image_field": b'name="image"' in blob,
                        "png_magic": b"\x89PNG\r\n\x1a\n" in blob})
                    return self._send(200, {"name": _multipart_filename(blob),
                                            "subfolder": "", "type": "input"})
                if p == "/sdapi/v1/txt2img":
                    outer.last_txt2img = json.loads(blob or b"{}")
                    return self._send(200, {
                        "images": [base64.b64encode(PNG_8).decode()],
                        "info": "{}"})
                return self._send(404, {"error": "stub: no route " + p})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True)
        self.thread.start()

    def base_url(self):
        return "http://127.0.0.1:%d" % self.port

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _cfg_on(stub):
    serve.CFG["media"] = {"comfy_url": stub.base_url(), "a1111_url": "",
                          "dir": MEDIA_TEST_DIR, "max_output_mb": 8,
                          "tts_cmd": [], "stt_cmd": [],
                          "ffmpeg_cmd": ["ffmpeg"], "policy": "off"}


def _cfg_set(**kw):
    m = dict(serve.CFG.get("media") or {})
    m.update(kw)
    serve.CFG["media"] = m


def _install_fake_residency(mode="ok"):
    """A stand-in for the residency module (another agent owns the real one)."""
    fake = types.ModuleType("residency")
    fake.CALLS = []

    def ensure_headroom(need_mb, policy=None, reason="", _mode=mode):
        fake.CALLS.append({"need_mb": int(need_mb), "policy": policy,
                           "reason": reason})
        if _mode == "refuse":
            return {"error": "insufficient RAM: need %dMB, have 100MB"
                             % int(need_mb),
                    "requested_mb": int(need_mb), "available_mb_before": 100.0,
                    "freed_mb": 0.0, "policy_used": policy or "off"}
        return {"freed_mb": 0.0, "array_mb": 0.0, "policy_used": policy or "off",
                "error": None, "requested_mb": int(need_mb),
                "available_mb_before": 9999.0}

    fake.ensure_headroom = ensure_headroom
    sys.modules["residency"] = fake
    return fake


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _gen(body):
    try:
        return media.api_media_generate(body)
    except Exception as e:  # a raise is a failure, reported as such
        return {"error": "RAISED %s: %s" % (type(e).__name__, e), "job": None}


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def import_case():
    check("import: media does not pull serve at module scope", not _SERVE_PULLED)
    check("import: media does not pull residency at module scope",
          not _RESIDENCY_PULLED)
    check("import: capabilities declared", media.CAPABILITIES ==
          ("image", "video", "audio_tts", "audio_stt", "music", "upscale",
           "any-workflow"), media.CAPABILITIES)
    check("import: backend table has the five declared backends",
          list(media.BACKENDS) == ["comfy", "a1111", "tts_cli", "stt_cli",
                                   "ffmpeg_assembly"], list(media.BACKENDS))
    check("import: need_mb defaults cover every capability",
          all(c in media.NEED_MB_DEFAULT for c in media.CAPABILITIES))


def workflows_case():
    lst = media.list_workflows()
    wfs = {w["id"]: w for w in lst["workflows"]}
    check("workflows: three found by id",
          sorted(wfs) == ["img2img_upscale", "sdxl_txt2img", "wan_txt2video"],
          sorted(wfs))
    check("workflows: declared capabilities",
          [wfs[i]["capability"] for i in sorted(wfs)]
          == ["upscale", "image", "video"],
          [(w["id"], w["capability"]) for w in lst["workflows"]])
    check("workflows: sdxl placeholders",
          wfs["sdxl_txt2img"]["placeholders"]
          == ["height", "negative", "prompt", "seed", "steps", "width"],
          wfs["sdxl_txt2img"]["placeholders"])
    check("workflows: img2img placeholders",
          wfs["img2img_upscale"]["placeholders"]
          == ["input_image", "negative", "prompt", "seed", "steps"],
          wfs["img2img_upscale"]["placeholders"])
    check("workflows: wan placeholders",
          wfs["wan_txt2video"]["placeholders"]
          == ["frames", "height", "negative", "prompt", "seed", "steps",
              "width"], wfs["wan_txt2video"]["placeholders"])
    check("workflows: required fields declared",
          wfs["sdxl_txt2img"]["required"] == ["prompt"]
          and wfs["img2img_upscale"]["required"] == ["prompt", "input_image"]
          and wfs["wan_txt2video"]["required"] == ["prompt"],
          [(w["id"], w["required"]) for w in lst["workflows"]])
    check("workflows: comfy hint + json file names",
          all(w["backend_hint"] == "comfy" and w["file"].endswith(".json")
              for w in lst["workflows"]))
    check("workflows: api wrapper shape", "workflows" in media.api_media_workflows())


def render_case():
    p = {"prompt": "a red cube", "negative": "blurry", "width": "256",
         "height": 256, "steps": "20", "seed": "7"}
    r = media.render_workflow("sdxl_txt2img", p)
    check("render: no error", "error" not in r, r.get("error"))
    wf = r.get("workflow") or {}
    check("render: nested CLIPTextEncode substituted",
          wf.get("2", {}).get("inputs", {}).get("text") == "a red cube")
    check("render: negative prompt substituted",
          wf.get("3", {}).get("inputs", {}).get("text") == "blurry")
    check("render: width/height coerced to int",
          wf.get("4", {}).get("inputs", {}).get("width") == 256
          and isinstance(wf.get("4", {}).get("inputs", {}).get("width"), int)
          and wf["4"]["inputs"]["height"] == 256
          and isinstance(wf["4"]["inputs"]["height"], int),
          wf.get("4"))
    check("render: seed/steps coerced to int",
          wf.get("5", {}).get("inputs", {}).get("seed") == 7
          and isinstance(wf["5"]["inputs"]["seed"], int)
          and wf["5"]["inputs"]["steps"] == 20, wf.get("5"))
    check("render: literal cfg/denoise untouched",
          wf.get("5", {}).get("inputs", {}).get("cfg") == 7.0
          and wf["5"]["inputs"]["denoise"] == 1.0)
    check("render: internal meta stripped from submitted graph",
          "_lumina" not in wf and r.get("id") == "sdxl_txt2img")
    check("render: used map reports coerced values",
          r.get("used", {}).get("width") == 256
          and r["used"]["steps"] == 20 and r["used"]["prompt"] == "a red cube",
          r.get("used"))
    check("render: unknown workflow id",
          media.render_workflow("nope", {}).get("error")
          == "unknown workflow: nope")
    check("render: missing required placeholder error",
          media.render_workflow("sdxl_txt2img", {}).get("error")
          == "missing required placeholders: prompt",
          media.render_workflow("sdxl_txt2img", {}))
    check("render: unknown placeholder error on the first one",
          str(media.render_workflow("sdxl_txt2img",
                                    {"prompt": "x"}).get("error")
              ).startswith("unknown placeholder: {{"),
          media.render_workflow("sdxl_txt2img", {"prompt": "x"}))
    check("render: prompt containing braces is not re-scanned",
          (media.render_workflow("sdxl_txt2img", dict(p, prompt="a {{cube}}"))
           .get("workflow", {}).get("2", {}).get("inputs", {}).get("text")
           == "a {{cube}}"))
    img = media.render_workflow("img2img_upscale",
                                {"prompt": "p", "negative": "",
                                 "steps": "10", "seed": "1",
                                 "input_image": "/tmp/in.png"})
    check("render: img2img load input + denoise 0.4 survive",
          img["workflow"]["2"]["inputs"]["image"] == "/tmp/in.png"
          and img["workflow"]["6"]["inputs"]["denoise"] == 0.4
          and img["workflow"]["6"]["inputs"]["latent_image"] == ["5", 0],
          img.get("error"))
    check("render: img2img missing input_image is required-error",
          "input_image" in str(media.render_workflow(
              "img2img_upscale", {"prompt": "p"}).get("error")))
    vid = media.render_workflow("wan_txt2video",
                                {"prompt": "p", "negative": "", "width": 512,
                                 "height": 512, "steps": "10", "seed": "3",
                                 "frames": "25"})
    check("render: wan frames coerced into video latent node",
          vid["workflow"]["6"]["inputs"]["length"] == 25
          and isinstance(vid["workflow"]["6"]["inputs"]["length"], int)
          and vid["workflow"]["9"]["class_type"] == "SaveWEBM",
          vid.get("error"))


def backends_case(stub):
    st = media.backends_status()
    by = {b["id"]: b for b in st["backends"]}
    check("backends: all five entries reported",
          list(by) == list(media.BACKENDS), list(by))
    comfy = by["comfy"]
    check("backends: comfy configured + reachable against stub",
          comfy["configured"] is True and comfy["reachable"] is True
          and isinstance(comfy["probe_ms"], (int, float)),
          comfy)
    check("backends: comfy reachable listed as available",
          "comfy" in st["available"] and st["policy"] == "off", st["available"])
    check("backends: a1111 unconfigured",
          by["a1111"]["configured"] is False
          and by["a1111"]["reachable"] is False
          and "a1111_url" in by["a1111"]["note"], by["a1111"])
    check("backends: tts/stt unconfigured with setting hints",
          by["tts_cli"]["configured"] is False
          and "tts_cmd" in by["tts_cli"]["note"]
          and by["stt_cli"]["configured"] is False
          and "stt_cmd" in by["stt_cli"]["note"],
          [by["tts_cli"], by["stt_cli"]])
    check("backends: ffmpeg_assembly reports the measured absence",
          by["ffmpeg_assembly"]["configured"] is False
          and by["ffmpeg_assembly"]["error"]
          == "ffmpeg not found (measured on this device)",
          by["ffmpeg_assembly"])

    closed = _closed_port()
    _cfg_set(comfy_url="http://127.0.0.1:%d" % closed)
    st2 = media.backends_status()
    c2 = {b["id"]: b for b in st2["backends"]}["comfy"]
    check("backends: closed port -> unreachable with error, no raise",
          c2["configured"] is True and c2["reachable"] is False
          and isinstance(c2.get("error"), str) and c2["error"],
          c2)
    check("backends: closed port drops comfy from available",
          "comfy" not in st2["available"], st2["available"])
    _cfg_set(comfy_url=stub.base_url())

    s = media.api_media_status()
    check("status: keys + device note",
          {"backends", "jobs", "policy", "device_note"} <= set(s)
          and isinstance(s["jobs"], int)
          and "no local diffusion engine" in s["device_note"]
          and "no CUDA" in s["device_note"], s.get("device_note"))
    check("status: compact backend rows keep honesty flags",
          all({"id", "caps", "kind", "configured", "reachable"} <= set(b)
              for b in s["backends"])
          and any(b["id"] == "comfy" and b["reachable"] for b in s["backends"]))
    check("status: api_media_backends is backends_status",
          {"backends", "policy", "available"} <= set(media.api_media_backends()))


def pick_case():
    pick = media.pick_backend("image")
    check("pick: comfy preferred over a1111 for image",
          pick.get("backend") == "comfy" and pick.get("kind") == "http"
          and pick.get("need_mb") == 2048
          and pick.get("need_source") == "declared_default", pick)
    check("pick: explicit need_mb is reported as request-sourced",
          media.pick_backend("image", need_mb=99).get("need_source")
          == "request"
          and media.pick_backend("image", need_mb=99).get("need_mb") == 99)
    check("pick: explicit unconfigured backend refuses",
          "not configured" in str(media.pick_backend("image",
                                                     backend="a1111").get("error"))
          and "a1111_url" in str(media.pick_backend("image",
                                                    backend="a1111").get("error")),
          media.pick_backend("image", backend="a1111"))
    bad = media.pick_backend("video", backend="a1111")
    check("pick: undeclared capability names the declaration",
          bad.get("error", "").startswith(
              "backend 'a1111' does not declare capability 'video'")
          and "image" in bad["error"] and "upscale" in bad["error"], bad)
    check("pick: unknown backend id",
          media.pick_backend("image", backend="nope").get("error", "")
          .startswith("unknown backend: nope"))
    none = media.pick_backend("audio_tts")
    check("pick: no configured backend for a capability",
          none.get("error") == "no backend provides capability 'audio_tts'. "
                               "Configure one in Settings > Media.", none)
    check("pick: video resolves to comfy (declared caps, not guessing)",
          media.pick_backend("video").get("backend") == "comfy"
          and media.pick_backend("video").get("need_mb") == 6144)


def generate_case(stub, fake):
    n_before = stub.attempts
    r = _gen({"capability": "image", "prompt": "a red cube",
              "workflow_id": "sdxl_txt2img", "width": 256, "height": 256})
    j = r.get("job") or {}
    files = j.get("files") or []
    check("generate: job done", j.get("status") == "done" and "error" not in r,
          r.get("error") or j.get("error"))
    check("generate: one file with real bytes", len(files) == 1
          and files[0]["name"] == "0001.png" and files[0]["bytes"] == len(PNG_8)
          and files[0]["bytes"] > 0, files)
    path = files[0]["path"] if files else ""
    data = open(path, "rb").read() if os.path.isfile(path) else b""
    check("generate: file written under CFG dir",
          bool(files) and path.startswith(MEDIA_TEST_DIR)
          and os.path.isfile(path) and data == PNG_8, path)
    check("generate: png magic + mime + size + url",
          data[:8] == b"\x89PNG\r\n\x1a\n"
          and files[0]["mime"] == "image/png"
          and (files[0]["w"], files[0]["h"]) == (8, 8)
          and files[0]["url"] == "/api/media/file?name=0001.png",
          files[:1])
    check("generate: submit happened once and was recorded",
          stub.attempts == n_before + 1 and len(stub.prompts) == 1,
          (stub.attempts, len(stub.prompts)))
    sent = (stub.prompts[-1] if stub.prompts else {})
    body = sent.get("prompt") or {}
    check("generate: rendered prompt/width/height reached the backend",
          body.get("2", {}).get("inputs", {}).get("text") == "a red cube"
          and body.get("4", {}).get("inputs", {}).get("width") == 256
          and isinstance(body["4"]["inputs"]["width"], int)
          and body.get("4", {}).get("inputs", {}).get("height") == 256,
          json.dumps(body)[:200])
    check("generate: client_id names the job",
          str(sent.get("client_id", "")).startswith("lumina-mj-"),
          sent.get("client_id"))
    check("generate: output fetched from /view with the reported name",
          bool(stub.view_calls)
          and stub.view_calls[-1].get("filename") == ["0001.png"], stub.view_calls)
    check("generate: residency consulted with declared need + policy",
          bool(fake.CALLS) and fake.CALLS[-1]["need_mb"] == 2048
          and fake.CALLS[-1]["policy"] == "off"
          and fake.CALLS[-1]["reason"] == "media job", fake.CALLS[-1:])
    check("generate: residency verdict attached to the job result",
          isinstance(r.get("residency"), dict)
          and r["residency"].get("error") is None
          and j.get("residency", {}).get("policy_used") == "off",
          r.get("residency"))
    text = str(r.get("text") or "")
    check("generate: agent-facing text is populated and specific",
          text.startswith("job %s done: 1 file 0001.png" % j.get("id"))
          and "(8x8," in text and "via comfy" in text, text)
    check("generate: job seconds measured",
          isinstance(j.get("seconds"), float) and j["seconds"] >= 0.0
          and j.get("started") and j.get("finished"), j.get("seconds"))
    check("generate: comfy history was polled until outputs appeared",
          stub.history_calls >= 2)
    check("generate: job id format", str(j.get("id", "")).startswith("mj-"), j.get("id"))


def a1111_case(stub):
    _cfg_set(a1111_url=stub.base_url())
    check("a1111: explicit configured backend accepted",
          media.pick_backend("image", backend="a1111").get("backend")
          == "a1111")
    r = _gen({"capability": "image", "prompt": "a1111 prompt",
              "backend": "a1111", "width": 64, "height": 64, "steps": 8})
    j = r.get("job") or {}
    files = j.get("files") or []
    check("a1111: own payload built and images decoded to files",
          j.get("status") == "done" and len(files) == 1
          and files[0]["name"].endswith("_0001.png")
          and files[0]["bytes"] == len(PNG_8)
          and os.path.isfile(files[0]["path"])
          and files[0]["mime"] == "image/png", r.get("error") or files)
    check("a1111: request carried prompt + size",
          (stub.last_txt2img or {}).get("prompt") == "a1111 prompt"
          and stub.last_txt2img.get("width") == 64
          and stub.last_txt2img.get("height") == 64, stub.last_txt2img)
    check("a1111: no workflow_id required", j.get("workflow_id") is None)
    _cfg_set(a1111_url="")


def helpers_case(stub):
    code, body = media._http_json("GET", stub.base_url() + "/queue")
    check("http: _http_json returns (status, dict) on success",
          code == 200 and isinstance(body, dict)
          and "queue_pending" in body, (code, body))
    code2, body2 = media._http_json("GET", "http://127.0.0.1:%d/x"
                                    % _closed_port(), None, 1.0)
    check("http: _http_json never raises on a dead port",
          code2 is None and isinstance(body2, str) and bool(body2),
          (code2, body2))
    q = media.comfy_queue(stub.base_url())
    check("comfy: queue probe shape", q.get("queue_running") == []
          and q.get("queue_pending") == [], q)
    bad = media.comfy_submit("http://127.0.0.1:%d" % _closed_port(),
                             {"1": {}}, "cid")
    check("comfy: submit to a dead port is an error string", "error" in bad
          and isinstance(bad["error"], str), bad)
    check("comfy: view on a dead port returns b'' (no raise)",
          media.comfy_view("http://127.0.0.1:%d" % _closed_port(),
                           "x.png") == b"")
    up_path = os.path.join(MEDIA_TEST_DIR, "upload_src.png")
    open(up_path, "wb").write(PNG_8)
    up = media.comfy_upload_image(stub.base_url(), up_path, name="in.png")
    last = stub.uploads[-1] if stub.uploads else {}
    check("comfy: multipart upload hand-rolled correctly",
          up.get("name") == "in.png" and last.get("boundary_seen") is True
          and last.get("has_image_field") is True
          and last.get("filename") == "in.png"
          and last.get("png_magic") is True and last.get("bytes", 0) > 0,
          (up, last))
    miss = media.comfy_upload_image(stub.base_url(),
                                    os.path.join(MEDIA_TEST_DIR, "nope.png"))
    check("comfy: upload of a missing file errors without raising",
          "error" in miss and "cannot read image" in miss["error"], miss)


def jail_case():
    ok_path = os.path.join(MEDIA_TEST_DIR, "jail_ok.png")
    open(ok_path, "wb").write(PNG_8)
    bad = ["../../etc/passwd", "..%2f..%2fetc%2fpasswd", "/etc/passwd",
           "sub/evil.png", "..", ".", "a\\b.png"]
    results = [media.api_media_file(n) for n in bad]
    check("jail: all traversal/split forms refused",
          all("error" in x and "path" not in x for x in results),
          [(n, x) for n, x in zip(bad, results) if "error" not in x])
    check("jail: refusals never echo filesystem bytes",
          all("etc" not in str(x.get("error", "")) for x in results)
          and all(x.get("error") == "invalid media name"
                  for x in results[:4]), [x.get("error") for x in results])
    check("jail: .lumina -> unsupported media type",
          media.api_media_file(".lumina").get("error")
          == "unsupported media type")
    check("jail: unknown extension refused",
          media.api_media_file("notes.xyzzy").get("error")
          == "unsupported media type")
    check("jail: absent file is an error, not a path",
          "error" in media.api_media_file("missing_file.png")
          and "path" not in media.api_media_file("missing_file.png"))
    good = media.api_media_file("jail_ok.png")
    check("jail: legit basename resolves with mime",
          good.get("name") == "jail_ok.png" and good.get("mime") == "image/png"
          and os.path.realpath(good.get("path", ""))
          == os.path.realpath(ok_path) and good.get("path"), good)
    check("jail: token_ok=False refuses",
          "error" in media.api_media_file("jail_ok.png", token_ok=False))
    gen = media.api_media_file("0001.png")
    check("jail: generated output is servable",
          gen.get("mime") == "image/png" and gen.get("name") == "0001.png"
          and os.path.isfile(gen.get("path", "")), gen)


def cli_case():
    orig_op = serve._operator_mode
    try:
        r0 = media.cli_run("tts_cmd", {}, "x.wav")
        check("cli: empty template refused",
              "no tts_cmd configured" in str(r0.get("error")), r0)
        _cfg_set(tts_cmd=["cp", "/dev/null", "{out}"])
        serve._operator_mode = lambda: {"full_access": False}
        r1 = media.cli_run("tts_cmd", {}, "x.wav")
        check("cli: operator mode required (exact guidance)",
              r1.get("error") == "CLI media backends need operator mode: "
                                 "Settings > Security", r1)
        serve._operator_mode = lambda: {"full_access": True}
        r2 = media.cli_run("tts_cmd", {}, "x.wav")
        check("cli: argv template runs, {out} forced into media dir",
              r2.get("rc") == 0
              and os.path.dirname(r2.get("out", ""))
              == os.path.dirname(os.path.join(MEDIA_TEST_DIR, "x.wav"))
              and os.path.basename(r2["out"]) == "x.wav"
              and os.path.isfile(r2["out"]) and r2["seconds"] >= 0.0, r2)
        r3 = media.cli_run("tts_cmd", {}, "../../escape.wav")
        check("cli: out_name cannot escape media dir",
              os.path.dirname(r3.get("out", "")) == os.path.dirname(
                  os.path.join(MEDIA_TEST_DIR, "escape.wav"))
              and os.path.basename(r3["out"]) == "escape.wav", r3)
        _cfg_set(tts_cmd=["ls", "/definitely-not-here-lumina-xyz"])
        r4 = media.cli_run("tts_cmd", {}, "y.wav")
        check("cli: rc!=0 surfaces stderr_tail",
              r4.get("rc") not in (0, None) and r4.get("stderr_tail")
              and "cli rc=" in str(r4.get("error"))
              and "y.wav" not in str(r4.get("error")), r4)
        _cfg_set(tts_cmd=["no-such-binary-lumina-xyz", "{out}"])
        r5 = media.cli_run("tts_cmd", {}, "z.wav")
        check("cli: missing binary reported",
              "cli binary not found" in str(r5.get("error")), r5)
        _cfg_set(tts_cmd=["cp", "{voice}", "{out}"])
        r6 = media.cli_run("tts_cmd", {}, "v.wav")
        check("cli: unfilled template field reported, not executed",
              "template field error in tts_cmd" in str(r6.get("error"))
              and "voice" in str(r6.get("error")), r6)
        _cfg_set(tts_cmd=[])
    finally:
        serve._operator_mode = orig_op


def size_cap_case():
    _cfg_set(max_output_mb=1)
    big = b"x" * (1024 * 1024 + 1)
    r = media._save_outputs("mj-cap", [("big.bin", big)])
    check("cap: over-limit output refused before writing",
          "error" in r and "max_output_mb=1" in r["error"]
          and not os.path.exists(os.path.join(MEDIA_TEST_DIR, "big.bin")), r)
    ok = media._save_outputs("mj-cap", [("small.bin", b"ok")])
    check("cap: under-limit output still written",
          ok.get("files", [{}])[0].get("name") == "small.bin"
          and os.path.getsize(ok["files"][0]["path"]) == 2, ok)
    _cfg_set(max_output_mb=8)


def fail_cases(stub, fake):
    stub.mode = "500"
    n = stub.attempts
    r = _gen({"capability": "image", "prompt": "boom",
              "workflow_id": "sdxl_txt2img", "width": 64, "height": 64})
    j = r.get("job") or {}
    check("fail: HTTP 500 becomes a job error, no raise",
          j.get("status") == "error" and "HTTP 500" in str(j.get("error"))
          and j.get("files") == [] and stub.attempts == n + 1,
          (j.get("status"), j.get("error")))
    check("fail: agent text reports the error honestly",
          "error" in str(r.get("text", "")) and str(j.get("id")) in str(r["text"]),
          r.get("text"))
    check("fail: error text does not leak the whole body",
          len(str(j.get("error"))) < 300, j.get("error"))
    stub.mode = "ok"

    fake_refuse = _install_fake_residency("refuse")
    attempts_before = stub.attempts
    hist_before = stub.history_calls
    r2 = _gen({"capability": "image", "prompt": "should never run",
               "workflow_id": "sdxl_txt2img", "width": 64, "height": 64})
    check("residency: refusal returns error + residency + job None",
          r2.get("job") is None and "insufficient RAM" in str(r2.get("error"))
          and isinstance(r2.get("residency"), dict)
          and r2["residency"].get("error"), r2)
    check("residency: nothing was sent to the backend on refusal",
          stub.attempts == attempts_before
          and stub.history_calls == hist_before, (stub.attempts,
                                                  stub.history_calls))
    check("residency: refusal recorded the need it was denied",
          fake_refuse.CALLS and fake_refuse.CALLS[-1]["need_mb"] == 2048
          and fake_refuse.CALLS[-1]["reason"] == "media job",
          fake_refuse.CALLS[-1:])
    _install_fake_residency("ok")

    nb = media.pick_backend("audio_stt")
    check("fail: stt without a template names the capability",
          "no backend provides capability 'audio_stt'" in str(nb.get("error")),
          nb)
    r3 = _gen({"capability": "hologram", "prompt": "x"})
    check("fail: unknown capability lists the valid ones",
          r3.get("job") is None and "image" in r3.get("error", "")
          and "audio_tts" in r3.get("error", ""), r3)
    r4 = _gen({"capability": "image", "prompt": "x", "backend": "a1111"})
    check("fail: unconfigured backend refuses before any job exists",
          r4.get("job") is None and "not configured" in str(r4.get("error")), r4)


def timeout_case(stub):
    stub.mode = "pending"
    ren = media.render_workflow("sdxl_txt2img",
                                {"prompt": "p", "negative": "", "width": 64,
                                 "height": 64, "steps": 1, "seed": 0})
    job = media.new_job("comfy", "image", {"prompt": "p"}, "sdxl_txt2img")
    media.job_update(job["id"], workflow=ren["workflow"])
    r = media.run_job(job["id"], timeout_s=0.4)
    j = r.get("job") or {}
    check("timeout: poll gives up and reports a timeout",
          j.get("status") == "error" and "timeout" in str(j.get("error")),
          j.get("error"))
    check("timeout: history excerpt kept for diagnosis",
          isinstance(j.get("history_excerpt"), str)
          and bool(j.get("history_excerpt")), j.get("history_excerpt"))
    stub.mode = "ok"


def jobs_case(stub):
    lst = media.jobs_list(limit=5)
    ids = [j["id"] for j in lst["jobs"]]
    seqs = [int(i.split("-")[1]) for i in ids]
    check("jobs: newest-first listing",
          seqs == sorted(seqs, reverse=True) and len(ids) <= 5
          and lst["count"] == len(ids) and lst["total"] >= lst["count"], lst)
    one = media.api_media_jobs({"id": ids[0]})
    check("jobs: get by id returns the same job",
          (one.get("job") or {}).get("id") == ids[0], one)
    check("jobs: unknown id error",
          media.job_get("mj-999999").get("error") == "no such job: mj-999999")
    check("jobs: job_update on unknown id errors",
          "error" in media.job_update("mj-999999", status="done"))
    check("jobs: api_media_jobs default lists", "jobs" in media.api_media_jobs())
    check("jobs: count in status reflects the registry",
          media.api_media_status()["jobs"] == media.jobs_list()["total"])
    check("jobs: files carry only relative urls",
          all(str(f.get("url", "")).startswith("/api/media/file?name=")
              for j in lst["jobs"] for f in (j.get("files") or [])))


def guard_case():
    check("guard: no model was loaded by this suite", serve.CHAT["model"] is None)
    d = media.media_dir()
    check("guard: media_dir honours the CFG override",
          os.path.basename(d) == "media_test" and os.path.isfile(
              os.path.join(d, "0001.png")), d)
    check("guard: workflow dir contains the three templates",
          sorted(os.listdir(media.WORKFLOW_DIR))
          == ["img2img_upscale.json", "sdxl_txt2img.json",
              "wan_txt2video.json"], sorted(os.listdir(media.WORKFLOW_DIR)))


def run_case(label, fn, *a):
    """Run one case; a raised exception is a FAIL, never a traceback."""
    try:
        fn(*a)
    except Exception as e:
        check("case %s: no exception" % label, False,
              "%s: %s" % (type(e).__name__, e))


def serve_wiring_case():
    """serve.py integration: tool listings, tool calls, cfg exposure, and
    the media -> residency.finish pairing for reload_after."""
    import serve
    check("serve-wiring: media tools in builtin tools",
          all(any(s["tool"] == t for s in serve.api_builtin_tools()["open_tools"])
              for t in ("media_status", "media_generate")))
    check("serve-wiring: media tools in base specs",
          all(any(s["tool"] == t for s in serve._agent_base_tool_specs())
              for t in ("media_status", "media_generate")))
    check("serve-wiring: media tool descriptions present",
          "media_generate" in serve._AGENT_TOOL_DESCRIPTIONS
          and "media_status" in serve._AGENT_TOOL_DESCRIPTIONS)
    st = serve._agent_tool_call({"tool": "media_status"})
    check("serve-wiring: media_status tool returns a backend table",
          isinstance(st, dict) and "backends" in st, st)
    saved = serve.CFG.get("media")
    try:
        serve.CFG["media"] = {"comfy_url": "", "a1111_url": "",
                              "dir": MEDIA_TEST_DIR}
        g = serve._agent_tool_call({"tool": "media_generate",
                                    "args": {"capability": "image",
                                             "prompt": "x"}})
        check("serve-wiring: media_generate with no backend errors honestly",
              "error" in g and "backend" in str(g.get("error", "")).lower()
              and g.get("job") is None, g)
    finally:
        if saved is None:
            serve.CFG.pop("media", None)
        else:
            serve.CFG["media"] = saved
    cfg = serve._public_cfg().get("media")
    check("serve-wiring: public cfg exposes the media block",
          isinstance(cfg, dict) and "policy" in cfg and "comfy_url" in cfg, cfg)

    saved_mod = sys.modules.get("residency")
    calls = []

    def _fin(pol, mid):
        calls.append((pol, mid))
        return {"reloaded": True, "reload_error": None}

    fake = types.ModuleType("residency")
    fake.finish = _fin
    sys.modules["residency"] = fake
    try:
        rr = {"policy_used": "reload_after", "unloaded_model": "/tmp/tiny-chat"}
        media._residency_finish(rr)
        check("serve-wiring: reload_after finishes with a reload",
              rr.get("reloaded") is True
              and calls == [("reload_after", "/tmp/tiny-chat")], (rr, calls))
        rr2 = {"policy_used": "off", "unloaded_model": None}
        media._residency_finish(rr2)
        check("serve-wiring: no churn when nothing was unloaded",
              len(calls) == 1 and "reload_error" not in rr2, rr2)
        fake.finish = lambda pol, mid: {"reloaded": False,
                                        "reload_error": "not enough free memory"}
        rr3 = {"policy_used": "unload", "unloaded_model": "m"}
        media._residency_finish(rr3)
        check("serve-wiring: reload failure is surfaced",
              rr3.get("reload_error") == "not enough free memory", rr3)
    finally:
        if saved_mod is None:
            sys.modules.pop("residency", None)
        else:
            sys.modules["residency"] = saved_mod


def main() -> int:
    t0 = time.time()
    try:
        os.makedirs(MEDIA_TEST_DIR, exist_ok=True)
    except Exception as e:
        print("SKIP: cannot create %s (%s)" % (MEDIA_TEST_DIR, e))
        return 3
    if not os.access(MEDIA_TEST_DIR, os.W_OK):
        print("SKIP: %s is not writable" % MEDIA_TEST_DIR)
        return 3
    shutil.rmtree(MEDIA_TEST_DIR, ignore_errors=True)
    os.makedirs(MEDIA_TEST_DIR, exist_ok=True)
    saved_media_cfg = serve.CFG.get("media")
    saved_residency = sys.modules.get("residency")
    orig_op = serve._operator_mode
    stub = None
    try:
        stub = _ComfyStub()
        _cfg_on(stub)
        fake = _install_fake_residency("ok")
        run_case("import", import_case)
        run_case("workflows", workflows_case)
        run_case("render", render_case)
        run_case("backends", backends_case, stub)
        run_case("pick", pick_case)
        run_case("generate", generate_case, stub, fake)
        run_case("a1111", a1111_case, stub)
        run_case("helpers", helpers_case, stub)
        run_case("jail", jail_case)
        run_case("cli", cli_case)
        run_case("size-cap", size_cap_case)
        run_case("timeout", timeout_case, stub)
        run_case("fail", fail_cases, stub, fake)
        run_case("jobs", jobs_case, stub)
        run_case("guard", guard_case)
        run_case("serve-wiring", serve_wiring_case)
    finally:
        serve._operator_mode = orig_op
        if saved_media_cfg is None:
            serve.CFG.pop("media", None)
        else:
            serve.CFG["media"] = saved_media_cfg
        if saved_residency is None:
            sys.modules.pop("residency", None)
        else:
            sys.modules["residency"] = saved_residency
        if stub is not None:
            stub.close()
        shutil.rmtree(MEDIA_TEST_DIR, ignore_errors=True)
    ok = sum(1 for _, c in RESULTS if c)
    print("\nMEDIA: %d/%d passed (%.1fs)" % (ok, len(RESULTS), time.time() - t0))
    if ok != len(RESULTS):
        print("RESULT: FAIL")
        for name, passed in RESULTS:
            if not passed:
                print("  failed: " + name)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
