"""Media generation orchestration: declared capabilities -> jobs -> files.

Honest scope on this device: no local diffusion engine (CPU-only, no CUDA) and
no bundled ffmpeg, so every backend is either a remote HTTP service the user
configured (ComfyUI, A1111) or a CLI tool the user installed. Nothing is
inferred: a backend counts only when CFG["media"] names it AND a probe (http)
or PATH lookup (cli) backs that up.

Dict in / dict out; every failure is {"error": str}, never an exception.
``serve`` and ``residency`` are imported lazily so importing this module never
loads a model and the suite can stub residency in sys.modules.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKFLOW_DIR = os.path.join(HERE, "media_workflows")

CAPABILITIES = ("image", "video", "audio_tts", "audio_stt", "music",
                "upscale", "any-workflow")
# Declared capabilities per backend; never inferred from what answers.
BACKENDS = {"comfy": {"caps": ["image", "video", "upscale", "any-workflow"],
                      "kind": "http"},
            "a1111": {"caps": ["image", "upscale"], "kind": "http"},
            "tts_cli": {"caps": ["audio_tts", "music"], "kind": "cli"},
            "stt_cli": {"caps": ["audio_stt"], "kind": "cli"},
            "ffmpeg_assembly": {"caps": ["video"], "kind": "cli"}}
NEED_MB_DEFAULT = {"image": 2048, "video": 6144, "audio_tts": 256,
                   "audio_stt": 1024, "music": 2048, "upscale": 2048,
                   "any-workflow": 4096}
_CFG_KEY = {"comfy": "comfy_url", "a1111": "a1111_url", "tts_cli": "tts_cmd",
            "stt_cli": "stt_cmd", "ffmpeg_assembly": "ffmpeg_cmd"}
_CLI_CFG = {"tts_cli": "tts_cmd", "stt_cli": "stt_cmd",
            "ffmpeg_assembly": "ffmpeg_cmd"}
_CLI_EXT = {"audio_tts": ".wav", "audio_stt": ".txt", "music": ".wav",
            "video": ".mp4", "image": ".png", "upscale": ".png"}
_PROBE = {"comfy": "/queue", "a1111": "/sdapi/v1/sd-models"}
MIME_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4",
            ".webm": "video/webm", ".wav": "audio/wav", ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg", ".flac": "audio/flac", ".txt": "text/plain"}
INT_KEYS = ("seed", "width", "height", "steps", "frames", "batch")
_PH = re.compile(r"\{\{([A-Za-z0-9_.-]+)\}\}")
FRAMES_DEFAULT = 49  # Wan-style clip length; declared default, not a guess


# ---------------------------------------------------------------------------
# config, paths
# ---------------------------------------------------------------------------
def _cfg() -> dict:
    """The media sub-dict of serve.CFG, or {} when absent (never raises)."""
    try:
        import serve
        m = serve.CFG.get("media")
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _policy() -> str:
    """Residency policy the user declared for media work ("off" if unset)."""
    return str(_cfg().get("policy") or "off")


def media_dir() -> str:
    """Output directory (CFG["media"]["dir"] or ROOT/outputs/media). Created."""
    d = str(_cfg().get("dir") or os.path.join("outputs", "media"))
    if not os.path.isabs(d):
        d = os.path.join(ROOT, d)
    os.makedirs(d, exist_ok=True)
    return d


def _max_output_bytes() -> int:
    try:
        mb = int(_cfg().get("max_output_mb") or 512)
    except (TypeError, ValueError):
        mb = 512
    return max(1, mb) * 1024 * 1024


def safe_name(name: str) -> str:
    """Basename-ish, path-free, boring filename (never escapes a directory)."""
    base = os.path.basename(str(name or "").strip())
    keep = "".join(c for c in base if c.isalnum() or c in "._-")
    return keep.strip(".")[:96] or "out.bin"


def _mime_for(name: str) -> str:
    return MIME_EXT.get(os.path.splitext(str(name))[1].lower(), "")


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------
def _workflow_path(workflow_id: str):
    wid = str(workflow_id or "").strip()
    if wid.endswith(".json"):
        wid = wid[:-5]
    if not wid or not re.fullmatch(r"[A-Za-z0-9_.-]+", wid) or ".." in wid:
        return None
    p = os.path.join(WORKFLOW_DIR, wid + ".json")
    return p if os.path.isfile(p) else None


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _placeholders(obj) -> list:
    """Ordered unique {{name}} tokens by deep scan (values only, not keys)."""
    found = []
    for s in _strings(obj):
        for m in _PH.findall(s):
            if m not in found:
                found.append(m)
    return found


def list_workflows() -> dict:
    """Every workflow JSON on disk with its declared placeholders."""
    try:
        names = sorted(n for n in os.listdir(WORKFLOW_DIR)
                       if n.endswith(".json"))
    except Exception:
        names = []
    out = []
    for fn in names:
        wid = fn[:-5]
        ent = {"id": wid, "file": fn, "capability": "any-workflow",
               "description": "", "placeholders": [], "required": [],
               "backend_hint": ""}
        try:
            doc = json.load(open(os.path.join(WORKFLOW_DIR, fn)))
        except Exception as e:
            ent["error"] = "cannot read: %s" % str(e)[:120]
            out.append(ent)
            continue
        meta = doc.pop("_lumina", {}) if isinstance(doc, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        ent.update({"id": str(meta.get("id") or wid),
                    "capability": str(meta.get("capability") or "any-workflow"),
                    "description": str(meta.get("description") or ""),
                    "required": [str(r) for r in (meta.get("required") or [])],
                    "backend_hint": str(meta.get("backend_hint") or ""),
                    "placeholders": sorted(set(_placeholders(doc)))})
        out.append(ent)
    return {"workflows": out}


def _coerce(name: str, value):
    if name not in INT_KEYS:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value


def _substitute(obj, used: dict):
    if isinstance(obj, str):
        m = _PH.fullmatch(obj)
        if m and m.group(1) in used:
            return used[m.group(1)]  # whole-token: keep the value's type
        return _PH.sub(lambda mm: str(used.get(mm.group(1), mm.group(0))), obj)
    if isinstance(obj, dict):
        return {k: _substitute(v, used) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, used) for v in obj]
    return obj


def render_workflow(workflow_id: str, params: dict) -> dict:
    """Fill a workflow template; unknown token -> error on the first one.

    Leftover detection runs against the template, not the filled text, so a
    user prompt containing "{{...}}" cannot false-positive.
    """
    params = params if isinstance(params, dict) else {}
    path = _workflow_path(workflow_id)
    if path is None:
        return {"error": "unknown workflow: %s" % workflow_id}
    try:
        doc = json.load(open(path))
    except Exception as e:
        return {"error": "cannot read workflow %s: %s"
                         % (workflow_id, str(e)[:150])}
    if not isinstance(doc, dict):
        return {"error": "workflow %s is not an object" % workflow_id}
    meta = doc.pop("_lumina", {})
    required = [str(r) for r in ((meta or {}).get("required") or [])]
    missing = [r for r in required if r not in params]
    if missing:
        return {"error": "missing required placeholders: " + ", ".join(missing)}
    used = {}
    for name in _placeholders(doc):
        if name not in params:
            return {"error": "unknown placeholder: {{%s}}" % name}
        used[name] = _coerce(name, params[name])
    return {"workflow": _substitute(doc, used), "id": str(workflow_id),
            "used": used}


def _default_workflow_for(capability: str):
    for w in list_workflows()["workflows"]:
        if w.get("capability") == capability:
            return w.get("id")
    return None


# ---------------------------------------------------------------------------
# backend status + selection
# ---------------------------------------------------------------------------
def _http_json(method: str, url: str, payload: dict | None = None,
               timeout: float = 10.0) -> tuple:
    """urllib wrapper: (status_code|None, parsed_dict|str). Never raises."""
    try:
        data, headers = None, {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, code = r.read(), int(getattr(r, "status", 0) or 0)
        txt = raw.decode("utf-8", "replace")
        try:
            return code, (json.loads(txt) if txt.strip() else {})
        except Exception:
            return code, txt[:2000]
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:2000]
        except Exception:
            body = str(e)
        return int(e.code), body
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:160])


def _http_bytes(url: str, timeout: float = 60.0) -> bytes:
    """Raw GET -> bytes. Empty bytes means the fetch failed or was empty."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return b""


def _backend_entry(bid: str) -> dict:
    """Config + live probe for one backend. Probes never raise."""
    spec = BACKENDS[bid]
    cfg = _cfg()
    ent = {"id": bid, "caps": list(spec["caps"]), "kind": spec["kind"],
           "configured": False, "reachable": False, "probe_ms": None, "note": ""}
    key = _CFG_KEY[bid]
    if spec["kind"] == "http":
        url = str(cfg.get(key) or "").strip()
        ent["configured"] = bool(url)
        if not ent["configured"]:
            ent["note"] = "set %s in Settings > Media (a base URL)" % key
            return ent
        t0 = time.perf_counter()
        code, body = _http_json("GET", url.rstrip("/") + _PROBE[bid], None, 2.0)
        ent["probe_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        ent["reachable"] = code is not None and 200 <= code < 300
        if ent["reachable"]:
            ent["note"] = "GET %s answered %s" % (_PROBE[bid], code)
        else:
            ent["error"] = ("HTTP %s: %s" % (code, str(body)[:120]) if code
                            is not None else str(body)[:160])
            ent["note"] = "configured, probe failed"
        return ent
    tpl = cfg.get(key)  # cli: the template is the config, the binary the proof
    tpl = tpl if isinstance(tpl, list) else []
    if bid == "ffmpeg_assembly":
        found = shutil.which(str(tpl[0]) if tpl else "ffmpeg")
        ent["configured"] = ent["reachable"] = found is not None
        if not found:
            ent["error"] = "ffmpeg not found (measured on this device)"
            ent["note"] = ("install ffmpeg, or set ffmpeg_cmd to a binary path "
                           "in Settings > Media")
        else:
            ent["note"] = "binary: " + found
        return ent
    ent["configured"] = bool(tpl)
    if not ent["configured"]:
        ent["note"] = "set %s (an argv template) in Settings > Media" % key
        return ent
    exe = str(tpl[0]) if tpl else ""
    found = shutil.which(exe) if exe else None
    ent["reachable"] = found is not None
    if not ent["reachable"]:
        ent["error"] = "binary not found on PATH: %s" % (exe or "(empty)")
    ent["note"] = ("template: %s" % " ".join(str(x) for x in tpl))[:160]
    return ent


def backends_status() -> dict:
    """Declared config + live probe per backend."""
    entries = [_backend_entry(b) for b in BACKENDS]
    avail = [e["id"] for e in entries if e["configured"]
             and (e["reachable"] or BACKENDS[e["id"]]["kind"] != "http")]
    return {"backends": entries, "policy": _policy(), "available": avail}


def pick_backend(capability: str, backend: str | None = None,
                 need_mb: int | None = None) -> dict:
    """Choose a backend: declared caps only, config required, comfy first."""
    cap = str(capability or "")
    if need_mb is None:
        need, source = NEED_MB_DEFAULT.get(cap, 2048), "declared_default"
    else:
        try:
            need, source = int(need_mb), "request"
        except (TypeError, ValueError):
            return {"error": "need_mb must be an integer"}
    if backend:
        bid = str(backend)
        spec = BACKENDS.get(bid)
        if spec is None:
            return {"error": "unknown backend: %s (known: %s)"
                             % (bid, ", ".join(BACKENDS))}
        if cap not in spec["caps"]:
            return {"error": "backend '%s' does not declare capability '%s' "
                             "(declares: %s)"
                             % (bid, cap, ", ".join(spec["caps"]))}
        ent = _backend_entry(bid)
        if not ent["configured"]:
            extra = ("; " + ent["error"]) if ent.get("error") else ""
            return {"error": "backend '%s' is not configured (%s empty in "
                             "Settings > Media)%s"
                             % (bid, _CFG_KEY[bid], extra)}
        return {"backend": bid, "caps": list(spec["caps"]), "kind": spec["kind"],
                "need_mb": need, "need_source": source}
    for bid, spec in BACKENDS.items():  # insertion order = preference order
        if cap in spec["caps"] and _backend_entry(bid)["configured"]:
            return {"backend": bid, "caps": list(spec["caps"]),
                    "kind": spec["kind"], "need_mb": need,
                    "need_source": source}
    return {"error": "no backend provides capability '%s'. Configure one in "
                     "Settings > Media." % cap}


# ---------------------------------------------------------------------------
# backend calls (comfy / a1111 / cli)
# ---------------------------------------------------------------------------
def comfy_submit(base_url: str, workflow: dict, client_id: str) -> dict:
    code, body = _http_json("POST", base_url.rstrip("/") + "/prompt",
                            {"prompt": workflow, "client_id": client_id}, 15.0)
    if code is None:
        return {"error": "comfy /prompt: " + str(body)}
    if not (200 <= code < 300):
        return {"error": "comfy /prompt HTTP %s: %s" % (code, str(body)[:200])}
    if isinstance(body, dict) and body.get("error"):
        return {"error": "comfy /prompt: " + str(body["error"])[:200]}
    if not (isinstance(body, dict) and body.get("prompt_id")):
        return {"error": "comfy /prompt: no prompt_id in reply"}
    return {"prompt_id": str(body["prompt_id"]), "status": code}


def comfy_history(base_url: str, prompt_id: str) -> dict:
    pid = urllib.parse.quote(str(prompt_id), safe="")
    code, body = _http_json("GET", base_url.rstrip("/") + "/history/" + pid,
                            None, 10.0)
    if code is None:
        return {"error": "comfy /history: " + str(body)}
    if not (200 <= code < 300):
        return {"error": "comfy /history HTTP %s: %s" % (code, str(body)[:200])}
    return body if isinstance(body, dict) else {"error": "history not an object"}


def comfy_view(base_url: str, filename: str, subfolder: str = "",
               type_: str = "output") -> bytes:
    q = urllib.parse.urlencode({"filename": str(filename),
                                "subfolder": str(subfolder or ""),
                                "type": str(type_ or "output")})
    return _http_bytes(base_url.rstrip("/") + "/view?" + q)


def comfy_queue(base_url: str) -> dict:
    code, body = _http_json("GET", base_url.rstrip("/") + "/queue", None, 2.0)
    if code is None:
        return {"error": "comfy /queue: " + str(body)}
    if not (200 <= code < 300):
        return {"error": "comfy /queue HTTP %s: %s" % (code, str(body)[:200])}
    return body if isinstance(body, dict) else {"error": "queue not an object"}


def comfy_upload_image(base_url: str, path: str, name: str = None) -> dict:
    """POST /upload/image with a hand-rolled multipart body (no deps)."""
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except Exception as e:
        return {"error": "cannot read image: %s" % str(e)[:150]}
    fn = safe_name(name or os.path.basename(str(path)))
    boundary = "----lumina" + uuid.uuid4().hex
    body = b"".join([
        ("--%s\r\n" % boundary).encode(),
        ('Content-Disposition: form-data; name="image"; filename="%s"\r\n'
         % fn).encode(),
        b"Content-Type: application/octet-stream\r\n\r\n", blob,
        ("\r\n--%s--\r\n" % boundary).encode()])
    req = urllib.request.Request(
        base_url.rstrip("/") + "/upload/image", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                 "Content-Length": str(len(body))})
    try:
        with urllib.request.urlopen(req, timeout=30.0) as r:
            raw, code = r.read(), int(getattr(r, "status", 0) or 0)
    except urllib.error.HTTPError as e:
        return {"error": "comfy upload HTTP %s" % e.code}
    except Exception as e:
        return {"error": "comfy upload: %s: %s"
                         % (type(e).__name__, str(e)[:150])}
    if not (200 <= code < 300):
        return {"error": "comfy upload HTTP %s" % code}
    try:
        info = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return {"error": "comfy upload: non-JSON reply"}
    if not isinstance(info, dict):
        return {"error": "comfy upload: unexpected reply"}
    if info.get("error"):
        return {"error": "comfy upload: " + str(info["error"])[:200]}
    info.setdefault("name", fn)
    return info


def a1111_txt2img(base_url: str, params: dict) -> dict:
    code, body = _http_json("POST", base_url.rstrip("/") + "/sdapi/v1/txt2img",
                            params if isinstance(params, dict) else {}, 120.0)
    if code is None:
        return {"error": "a1111 txt2img: " + str(body)}
    if not (200 <= code < 300):
        return {"error": "a1111 txt2img HTTP %s: %s" % (code, str(body)[:200])}
    if not isinstance(body, dict):
        return {"error": "a1111 txt2img: non-dict reply"}
    images = []
    for i, b64 in enumerate(body.get("images") or []):
        try:
            images.append(base64.b64decode(str(b64)))
        except Exception:
            return {"error": "a1111 txt2img: bad base64 image %d" % (i + 1)}
    if not images:
        return {"error": "a1111 txt2img: no images in reply"}
    return {"images": images}


def cli_run(cfg_key: str, subs: dict, out_name: str,
            timeout: float = 120.0) -> dict:
    """Run an argv template from CFG["media"][cfg_key]. Operator mode required.

    Deliberately NOT routed through serve.api_exec: python3 is excluded there,
    and these templates are user-configured argv lists run with shell=False.
    """
    tpl = _cfg().get(cfg_key)
    if not isinstance(tpl, list) or not tpl:
        return {"error": "no %s configured (Settings > Media)" % cfg_key}
    try:
        import serve
        op = serve._operator_mode()
    except Exception as e:
        return {"error": "operator mode check failed: %s" % str(e)[:150]}
    if not (isinstance(op, dict) and op.get("full_access")):
        return {"error": "CLI media backends need operator mode: "
                         "Settings > Security"}
    out_path = os.path.join(media_dir(), safe_name(out_name))
    vals = dict(subs or {})
    vals["out"] = out_path
    argv = []
    for el in tpl:
        try:
            argv.append(str(el).format(**vals))
        except (KeyError, IndexError, ValueError) as e:
            return {"error": "template field error in %s: %s" % (cfg_key, e)}
    if not shutil.which(argv[0]):
        return {"error": "cli binary not found: %s" % argv[0]}
    t0 = time.perf_counter()
    try:
        p = subprocess.run(argv, capture_output=True, timeout=float(timeout))
    except subprocess.TimeoutExpired:
        return {"error": "cli timeout after %.0fs: %s"
                         % (float(timeout), argv[0])}
    except Exception as e:
        return {"error": "cli run failed: %s: %s"
                         % (type(e).__name__, str(e)[:150])}
    secs = round(time.perf_counter() - t0, 3)
    tail = (p.stderr or b"").decode("utf-8", "replace")[-300:]
    if p.returncode != 0:
        return {"error": "cli rc=%d: %s" % (p.returncode,
                                            tail or "(no stderr)"),
                "rc": p.returncode, "stderr_tail": tail, "seconds": secs,
                "cmd": argv[0]}
    return {"out": out_path, "seconds": secs, "rc": 0, "stderr_tail": tail}


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_SEQ = itertools.count(1)


def _job_copy(job: dict) -> dict:
    out = dict(job)
    out["files"] = [dict(f) for f in (job.get("files") or [])]
    out["params"] = dict(job.get("params") or {})
    return out


def new_job(backend, capability, params, workflow_id) -> dict:
    with _JOBS_LOCK:
        jid = "mj-%d" % next(_JOB_SEQ)
    job = {"id": jid, "status": "queued", "backend": str(backend),
           "capability": str(capability), "workflow_id": workflow_id,
           "params": dict(params or {}), "created": time.time(), "started": None,
           "finished": None, "seconds": None, "files": [], "error": None,
           "residency": None}
    with _JOBS_LOCK:
        JOBS[jid] = job
        return _job_copy(job)


def job_get(job_id) -> dict:
    with _JOBS_LOCK:
        job = JOBS.get(str(job_id))
        if job is None:
            return {"error": "no such job: %s" % job_id}
        return {"job": _job_copy(job)}


def jobs_list(limit=20) -> dict:
    try:
        lim = max(1, int(limit))
    except (TypeError, ValueError):
        lim = 20
    with _JOBS_LOCK:
        js = [_job_copy(j) for j in JOBS.values()]
    js.sort(key=lambda j: (j.get("created") or 0.0, int(
        str(j.get("id", "mj-0")).split("-")[-1] or 0)), reverse=True)
    return {"jobs": js[:lim], "count": len(js[:lim]), "total": len(js)}


def job_update(job_id, **kw) -> dict:
    with _JOBS_LOCK:
        job = JOBS.get(str(job_id))
        if job is None:
            return {"error": "no such job: %s" % job_id}
        job.update(kw)
        if job.get("status") in ("done", "error"):
            if not job.get("finished"):
                job["finished"] = time.time()
            if job.get("started"):
                job["seconds"] = round(job["finished"] - job["started"], 3)
        return {"job": _job_copy(job)}


def _png_size(data: bytes):
    """(width, height) from a PNG IHDR, or None (only PNG is parsed)."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" \
            and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return (w, h) if w > 0 and h > 0 else None
    return None


def _save_outputs(jid: str, blobs: list) -> dict:
    """Write (name, bytes) pairs into media_dir with a total-size cap."""
    out_dir, max_total = media_dir(), _max_output_bytes()
    total = sum(len(b) for _, b in blobs)
    if total > max_total:
        return {"error": "refused: %d bytes > max_output_mb=%d"
                         % (total, max_total // (1024 * 1024))}
    files = []
    for name, data in blobs:
        path = os.path.join(out_dir, safe_name(name))
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:
            return {"error": "cannot write output: %s" % str(e)[:150],
                    "files": files}
        ent = {"name": os.path.basename(path), "path": path,
               "url": "/api/media/file?name="
                      + urllib.parse.quote(os.path.basename(path)),
               "bytes": len(data), "mime": _mime_for(path), "w": None, "h": None}
        wh = _png_size(data)
        if wh:
            ent["w"], ent["h"] = wh
        files.append(ent)
    return {"files": files}


def _comfy_outputs(entry: dict) -> list:
    items = []
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return items
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for key in ("images", "gifs", "videos"):
            for it in (node.get(key) or []):
                if isinstance(it, dict) and it.get("filename"):
                    items.append({"filename": str(it["filename"]),
                                  "subfolder": str(it.get("subfolder") or ""),
                                  "type": str(it.get("type") or "output"),
                                  "size": it.get("size")})
    return items


def _run_comfy(jid: str, workflow, timeout_s: float) -> dict:
    base = str(_cfg().get("comfy_url") or "").strip()
    if not base:
        return {"error": "comfy_url not configured (Settings > Media)"}
    if not isinstance(workflow, dict) or not workflow:
        return {"error": "no rendered workflow on job %s" % jid}
    sub = comfy_submit(base, workflow, client_id="lumina-" + jid)
    if sub.get("error"):
        return {"error": sub["error"]}
    pid = sub["prompt_id"]
    deadline, entry, hist = time.perf_counter() + float(timeout_s), None, {}
    while True:
        hist = comfy_history(base, pid)
        if isinstance(hist, dict) and not hist.get("error"):
            cand = hist.get(pid)
            if isinstance(cand, dict):
                st = cand.get("status") or {}
                if st.get("status_str") == "error":
                    return {"error": "comfy node error: " + str(st)[:300],
                            "history_excerpt": str(hist)[:300]}
                if cand.get("outputs") or st.get("completed"):
                    entry = cand
                    break
        if time.perf_counter() > deadline:
            return {"error": "comfy timeout after %.1fs waiting for %s"
                             % (float(timeout_s), pid),
                    "history_excerpt": str(hist)[:300]}
        time.sleep(0.4)
    items = _comfy_outputs(entry or {})
    if not items:
        return {"error": "comfy finished with no output files",
                "history_excerpt": str(hist)[:300]}
    max_total = _max_output_bytes()
    sizes = [int(i.get("size") or 0) for i in items]
    if all(sizes) and sum(sizes) > max_total:
        return {"error": "refused before download: outputs report %d bytes > "
                         "max_output_mb=%d" % (sum(sizes),
                                               max_total // (1024 * 1024))}
    blobs = []
    for it in items:
        data = comfy_view(base, it["filename"], it["subfolder"], it["type"])
        if not data:
            continue  # unfetchable/empty: never claim a file we do not have
        blobs.append((it["filename"], data))
    if not blobs:
        return {"error": "comfy outputs could not be fetched",
                "history_excerpt": str(hist)[:300]}
    return _save_outputs(jid, blobs)


def _run_a1111(jid: str, params: dict) -> dict:
    base = str(_cfg().get("a1111_url") or "").strip()
    if not base:
        return {"error": "a1111_url not configured (Settings > Media)"}
    r = a1111_txt2img(base, {"prompt": str(params.get("prompt") or ""),
                             "negative_prompt": str(params.get("negative") or ""),
                             "width": int(params.get("width") or 1024),
                             "height": int(params.get("height") or 1024),
                             "steps": int(params.get("steps") or 20),
                             "seed": int(params.get("seed") or 0)})
    if r.get("error"):
        return {"error": r["error"]}
    blobs = [("%s_%04d.png" % (jid, i), data)
             for i, data in enumerate(r["images"], 1)]
    return _save_outputs(jid, blobs)


def _run_cli(jid: str, backend: str, capability: str, params: dict) -> dict:
    cfg_key = _CLI_CFG.get(backend)
    if not cfg_key:
        return {"error": "no cli template mapped for backend %s" % backend}
    prompt = str(params.get("prompt") or "")
    r = cli_run(cfg_key, {"prompt": prompt, "text": prompt,
                          "voice": params.get("voice", ""),
                          "in": params.get("input_image", ""),
                          "seed": params.get("seed", 0), "job": jid},
                jid + _CLI_EXT.get(capability, ".bin"))
    if r.get("error"):
        return {"error": r["error"], "rc": r.get("rc"),
                "stderr_tail": r.get("stderr_tail")}
    path = r["out"]
    if not os.path.isfile(path):
        return {"error": "cli produced no output file: "
                         + os.path.basename(path)}
    name = os.path.basename(path)
    return {"files": [{"name": name, "path": path,
                       "url": "/api/media/file?name="
                              + urllib.parse.quote(name),
                       "bytes": os.path.getsize(path), "mime": _mime_for(name),
                       "w": None, "h": None}]}


def run_job(job_id, timeout_s: float = 600.0) -> dict:
    """Execute a queued job. Returns {"job": ...} (plus "error" on failure)."""
    jid = str(job_id)
    with _JOBS_LOCK:
        snap = _job_copy(JOBS[jid]) if jid in JOBS else None
    if snap is None:
        return {"error": "no such job: %s" % jid}
    job_update(jid, status="running", started=time.time(), error=None)
    backend = snap["backend"]
    if backend == "comfy":
        res = _run_comfy(jid, snap.get("workflow"), timeout_s)
    elif backend == "a1111":
        res = _run_a1111(jid, snap["params"])
    elif (BACKENDS.get(backend) or {}).get("kind") == "cli":
        res = _run_cli(jid, backend, snap["capability"], snap["params"])
    else:
        res = {"error": "unknown backend for job: %s" % backend}
    if res.get("error"):
        kw = {"status": "error", "error": res["error"], "files": []}
        for k in ("history_excerpt", "rc", "stderr_tail"):
            if res.get(k) is not None and res.get(k) != "":
                kw[k] = res[k]
        r = job_update(jid, **kw)
        return {"job": r["job"], "error": res["error"]}
    r = job_update(jid, status="done", files=res["files"], error=None)
    return {"job": r["job"], "files": res["files"]}


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------
def api_media_status() -> dict:
    bs = backends_status()
    with _JOBS_LOCK:
        n = len(JOBS)
    compact = [{"id": e["id"], "caps": e["caps"], "kind": e["kind"],
                "configured": e["configured"], "reachable": e["reachable"]}
               for e in bs["backends"]]
    return {"backends": compact, "jobs": n, "policy": bs["policy"],
            "available": bs["available"],
            "device_note": "no local diffusion engine on this device "
                           "(CPU-only, no CUDA) \u2014 backends are remote "
                           "or CLI"}


def api_media_backends() -> dict:
    return backends_status()


def api_media_workflows() -> dict:
    return list_workflows()


def api_media_jobs(body=None) -> dict:
    body = body if isinstance(body, dict) else {}
    if body.get("id"):
        return job_get(body["id"])
    try:
        lim = int(body.get("limit", 20))
    except (TypeError, ValueError):
        lim = 20
    return jobs_list(lim)


def api_media_file(name: str, token_ok: bool = True) -> dict:
    """Resolve one media file. Strict jail: basenames in media_dir only."""
    if not token_ok:
        return {"error": "invalid api token"}
    if not isinstance(name, str) or not name:
        return {"error": "media name required"}
    if name != os.path.basename(name) or os.path.isabs(name):
        return {"error": "invalid media name"}
    if name in (".", "..") or ".." in name or "/" in name or "\\" in name \
            or "\x00" in name:
        return {"error": "invalid media name"}
    mime = _mime_for(name)
    if not mime:
        return {"error": "unsupported media type"}
    base = os.path.realpath(media_dir())
    full = os.path.realpath(os.path.join(base, name))
    if os.path.dirname(full) != base:
        return {"error": "invalid media name"}
    if not os.path.isfile(full):
        return {"error": "media file not found"}
    return {"path": full, "mime": mime, "name": name}


def _job_text(job: dict, backend: str) -> str:
    if job.get("status") == "error":
        return "job %s error: %s via %s" % (job["id"], job.get("error"),
                                            backend)
    files = job.get("files") or []
    parts = []
    for f in files[:3]:
        bits = []
        if f.get("w") and f.get("h"):
            bits.append("%dx%d" % (f["w"], f["h"]))
        if job.get("seconds") is not None:
            bits.append("%.1fs" % job["seconds"])
        parts.append("%s%s" % (f["name"],
                               (" (" + ", ".join(bits) + ")") if bits else ""))
    more = "" if len(files) <= 3 else " (+%d more)" % (len(files) - 3)
    return "job %s done: %d file%s %s via %s" % (
        job["id"], len(files), "" if len(files) == 1 else "s",
        " ".join(parts) + more, backend)


def _residency_finish(rr: dict) -> None:
    """After a media job: honor the reload_after policy via
    residency.finish(). Never raises; merges the reload outcome into the
    residency verdict dict in place so the job record tells the truth.
    Only called when a model was actually unloaded."""
    if str(rr.get("policy_used") or "") not in ("unload", "reload_after"):
        return
    if not rr.get("unloaded_model"):
        return
    try:
        import residency as _R
        f = _R.finish(str(rr.get("policy_used") or ""),
                      rr.get("unloaded_model"))
        if isinstance(f, dict):
            if "reloaded" in f:
                rr["reloaded"] = bool(f["reloaded"])
            if f.get("reload_error"):
                rr["reload_error"] = str(f["reload_error"])
    except Exception as e:
        rr["reload_error"] = "%s: %s" % (type(e).__name__, str(e)[:150])


def api_media_generate(body: dict) -> dict:
    """Validate -> pick backend -> residency headroom -> job -> run/wait."""
    body = body if isinstance(body, dict) else {}
    cap = str(body.get("capability") or "")
    if cap not in CAPABILITIES:
        return {"error": "unknown capability '%s'; valid: %s"
                         % (cap or "(empty)", ", ".join(CAPABILITIES)),
                "job": None}
    pick = pick_backend(cap, body.get("backend"), body.get("need_mb"))
    if pick.get("error"):
        return {"error": pick["error"], "job": None}
    policy = str(body.get("policy") or _policy() or "off")
    try:
        import residency as _R
        rr = _R.ensure_headroom(int(pick["need_mb"]), policy,
                                reason="media job")
    except Exception as e:
        return {"error": "residency check unavailable (%s: %s)"
                         % (type(e).__name__, str(e)[:150]),
                "residency": None, "job": None}
    if not isinstance(rr, dict):
        return {"error": "residency check returned no verdict",
                "residency": None, "job": None}
    if rr.get("error"):
        return {"error": str(rr["error"]), "residency": rr, "job": None}
    params = {"prompt": str(body.get("prompt") or ""),
              "negative": str(body.get("negative") or ""),
              "width": int(body.get("width") or 1024),
              "height": int(body.get("height") or 1024),
              "steps": int(body.get("steps") or 20),
              "seed": int(body.get("seed") if body.get("seed") is not None
                          else 0),
              "frames": int(body.get("frames") or FRAMES_DEFAULT), "batch": 1}
    if body.get("input_image"):
        params["input_image"] = str(body["input_image"])
    wid, rendered = body.get("workflow_id") or None, None
    if pick["backend"] == "comfy":
        if not wid:
            if cap == "any-workflow":
                return {"error": "capability 'any-workflow' needs an explicit "
                                 "workflow_id", "job": None}
            wid = _default_workflow_for(cap)
        if not wid:
            return {"error": "no workflow for capability '%s' in %s"
                             % (cap, WORKFLOW_DIR), "job": None}
        ren = render_workflow(str(wid), params)
        if ren.get("error"):
            return {"error": ren["error"], "job": None}
        rendered = ren["workflow"]
    job = new_job(pick["backend"], cap, params, wid)
    job_update(job["id"], residency=rr)
    if rendered is not None:
        job_update(job["id"], workflow=rendered)
    if not body.get("wait", True):
        return {"job": job_get(job["id"]).get("job"), "residency": rr,
                "text": "job %s queued via %s" % (job["id"], pick["backend"])}
    run = None
    try:
        run = run_job(job["id"])
    finally:
        _residency_finish(rr)
    j = run["job"]
    j["residency"] = rr
    err = run.get("error") or j.get("error")
    if err or j.get("status") != "done":
        return {"job": j, "residency": rr, "error": err,
                "text": _job_text(j, pick["backend"])}
    if not j.get("files"):
        j = dict(j, status="error", error="job produced no files")
        return {"job": j, "residency": rr, "error": j["error"],
                "text": _job_text(j, pick["backend"])}
    return {"job": j, "residency": rr, "text": _job_text(j, pick["backend"])}
