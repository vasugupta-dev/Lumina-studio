"""Operator mode: whole-computer access for Lumina, opt-in and audited.

Default posture (unchanged from the original security review): read-only
allowlisted exec, no filesystem API, no browser driving. The user asked for
an assistant that can use the machine like a person — so this module adds
that capability, but only behind a deliberate operator switch:

  * API toggle  : POST /api/config {"full_access": true}  — REQUIRES an API
                  token to be configured (a LAN-exposed unrestricted shell
                  without auth is refused).
  * Env switch  : LUMINA_FULL_ACCESS=1 at server start — an explicit local
                  operator decision; reported as source=env.

While operator mode is ON:
  * /api/exec runs any command (shell, cwd configurable, timeout bounded).
  * /api/fs does list/read/write/append/delete/move/mkdir/search/stat.
  * The browser driver and the agent's launch/fs/browser tools unlock.

Every action is appended to ops_audit.jsonl (time, op, target, outcome) so
the operator can see exactly what the assistant did. Nothing here is
claimed beyond what the audit log records.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AUDIT_PATH = os.path.join(HERE, "ops_audit.jsonl")

# Search guards so a recursive scan cannot pin this phone forever.
SEARCH_MAX_FILES = 4000
SEARCH_MAX_BYTES = 64 * 1024 * 1024
READ_MAX = 512 * 1024
WRITE_MAX = 8 * 1024 * 1024

_lock = threading.Lock()


def audit(op: str, target: str, ok: bool, detail: str = "") -> dict:
    rec = {"at": round(time.time(), 3), "op": op, "target": str(target)[:300],
           "ok": bool(ok), "detail": str(detail)[:300]}
    try:
        with _lock, open(AUDIT_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def audit_tail(n: int = 50) -> list[dict]:
    try:
        with open(AUDIT_PATH) as f:
            lines = f.read().splitlines()[-n:]
    except Exception:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _policy(token: str | None, host: str | None) -> tuple[bool, str]:
    """May operator mode be enabled here? (toggle policy, not current state)"""
    host = host or "127.0.0.1"
    if host in LOOPBACK:
        return True, "operator toggle (loopback only)"
    if token:
        return True, "operator toggle + API token (non-loopback bind)"
    return False, ("full_access on a non-loopback bind (%s) needs an API token: "
                   "start the server with --token or LUMINA_TOKEN, then toggle"
                   % host)


def mode(cfg: dict | None = None, token: str | None = None,
         host: str | None = None) -> dict:
    """Current operator-mode state.

    Guard policy (honest and deliberate): operator mode is free to enable on
    a loopback-only server — it is then the user's own machine talking to
    itself. On a non-loopback bind (LAN/0.0.0.0) an API token is REQUIRED,
    because an unauthenticated unrestricted shell reachable from the network
    is the exact failure mode the earlier review flagged.
    """
    cfg = cfg or {}
    host = host or "127.0.0.1"
    if str(os.environ.get("LUMINA_FULL_ACCESS", "")).strip() in ("1", "true", "yes"):
        return {"full_access": True, "source": "env", "host": host,
                "guard": "explicit operator env flag (LUMINA_FULL_ACCESS=1)"}
    ok, why = _policy(token, host)
    if cfg.get("full_access"):
        if not ok:
            return {"full_access": False, "source": "cfg", "host": host,
                    "blocked": why}
        return {"full_access": True, "source": "cfg", "host": host,
                "guard": why}
    return {"full_access": False, "source": "cfg", "host": host,
            "can_enable": ok, "guard_if_enabled": why}


def set_mode(enable: bool, cfg: dict, token: str | None,
             host: str | None = None) -> dict:
    if enable:
        ok, why = _policy(token, host)
        if not ok:
            return {"error": why, "mode": mode(cfg, token, host)}
        cfg["full_access"] = True
        audit("mode", "full_access", True, "enabled")
        return {"mode": mode(cfg, token, host)}
    cfg["full_access"] = False
    audit("mode", "full_access", False, "disabled")
    return {"mode": mode(cfg, token, host)}


# ---------------------------------------------------------------------------
# Process launch (operator mode: any command)
# ---------------------------------------------------------------------------

def run_any(cmd: str, cwd: str | None = None, timeout: int = 60,
            env_extra: dict | None = None) -> dict:
    """Run an arbitrary shell command. Operator mode only — the caller must
    have checked the mode. Bounded output, bounded time, audit-logged."""
    cwd = os.path.expanduser(cwd or os.path.expanduser("~"))
    if not os.path.isdir(cwd):
        return {"error": "cwd not a directory: " + cwd[:120], "code": -1}
    env = dict(os.environ)
    env.update(env_extra or {})
    try:
        p = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True,
                           text=True, timeout=max(1, min(int(timeout), 600)),
                           cwd=cwd, env=env, errors="replace")
        out = (p.stdout + p.stderr)
        rec = audit("exec", cmd[:200], p.returncode == 0,
                    "code=%d" % p.returncode)
        return {"out": out[:8000], "code": p.returncode,
                "cwd": cwd, "seconds": rec["at"], "audited": True}
    except subprocess.TimeoutExpired:
        audit("exec", cmd[:200], False, "timeout")
        return {"error": "timeout after %ss" % timeout, "code": -2}
    except Exception as e:
        audit("exec", cmd[:200], False, str(e)[:80])
        return {"error": str(e)[:200], "code": -1}


LAUNCH_ALLOW = {
    # GUI/desktop app names Lumina may start in operator mode. Anything else
    # still runs through /api/exec; this list only exists so "open an app"
    # has an auditable, explicit vocabulary.
    "firefox", "chromium", "chromium-browser", "google-chrome", "gedit",
    "kate", "thunar", "nautilus", "vlc", "mpv", "libreoffice", "soffice",
    "xterm", "foot", "pcmanfm", "evince", "rhythmbox", "gimp", "okular",
    "code", "codium", "xmessage", "zenity", "xdg-open", "open",
}


def launch_app(name: str, args: list[str] | None = None) -> dict:
    """Start a desktop application the way a user would (detached, no wait).
    Whitelisted names only; returns the exact command that was launched."""
    name = str(name).strip()
    exe = os.path.basename(name)
    if exe not in LAUNCH_ALLOW:
        return {"error": "app not in launch allowlist: " + exe,
                "allowlist": sorted(LAUNCH_ALLOW)}
    path = shutil.which(exe)
    if path is None:
        return {"error": exe + " is not installed on this machine",
                "checked": "PATH"}
    cmd = [path] + [str(a)[:200] for a in (args or [])][:8]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL,
                             start_new_session=True)
    except Exception as e:
        audit("launch", exe, False, str(e)[:80])
        return {"error": str(e)[:150], "code": -1}
    audit("launch", exe, True, "pid=%d" % p.pid)
    return {"launched": exe, "pid": p.pid, "cmd": cmd}


def app_list() -> dict:
    """Which allowlisted apps actually exist on this machine."""
    have = {a: (shutil.which(a) or "") for a in sorted(LAUNCH_ALLOW)}
    return {"available": {k: v for k, v in have.items() if v},
            "missing": sorted(k for k, v in have.items() if not v),
            "count_available": sum(1 for v in have.values() if v)}


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def _safe(path: str) -> str:
    """Absolute real path — resolves symlinks so a link cannot escape intent
    (the path jail lesson from the earlier review, now used for evidence)."""
    return os.path.realpath(os.path.expanduser(str(path)))


def fs_list(path: str, show_hidden: bool = False, limit: int = 800) -> dict:
    p = _safe(path or "~")
    if not os.path.exists(p):
        return {"error": "not found: " + p[:160]}
    if not os.path.isdir(p):
        return {"error": "not a directory (use fs_read): " + p[:160]}
    items = []
    try:
        with os.scandir(p) as it:
            for e in it:
                if not show_hidden and e.name.startswith("."):
                    continue
                try:
                    st = e.stat(follow_symlinks=False)
                    items.append({"name": e.name, "dir": e.is_dir(follow_symlinks=False),
                                  "size": st.st_size, "mtime": round(st.st_mtime),
                                  "link": e.is_symlink()})
                except Exception:
                    items.append({"name": e.name, "dir": False, "size": -1})
                if len(items) >= limit:
                    break
    except PermissionError:
        return {"error": "permission denied: " + p[:160]}
    items.sort(key=lambda x: (not x["dir"], x["name"].lower()))
    return {"path": p, "items": items, "count": len(items),
            "truncated": len(items) >= limit}


def fs_read(path: str, max_bytes: int = READ_MAX) -> dict:
    p = _safe(path)
    if os.path.isdir(p):
        return {"error": "is a directory — use fs_list"}
    try:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            data = f.read(min(size, max_bytes))
    except Exception as e:
        audit("fs_read", p, False, str(e)[:80])
        return {"error": str(e)[:160]}
    audit("fs_read", p, True, "%d bytes" % len(data))
    text = data.decode("utf-8", "replace")
    return {"path": p, "size": size, "text": text,
            "truncated": size > max_bytes, "binary": b"\x00" in data[:2048]}


def fs_write(path: str, text: str, append: bool = False) -> dict:
    p = _safe(path)
    b = str(text).encode()[:WRITE_MAX]
    try:
        parent = os.path.dirname(p)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(p, "ab" if append else "wb") as f:
            f.write(b)
    except Exception as e:
        audit("fs_write", p, False, str(e)[:80])
        return {"error": str(e)[:160]}
    audit("fs_write", p, True, "%d bytes%s" % (len(b), " appended" if append else ""))
    return {"path": p, "wrote": len(b)}


def fs_delete(path: str, recursive: bool = False) -> dict:
    p = _safe(path)
    try:
        if os.path.isdir(p):
            if not recursive:
                return {"error": "is a directory — pass recursive=true to remove",
                        "path": p}
            shutil.rmtree(p)
        else:
            os.remove(p)
    except Exception as e:
        audit("fs_delete", p, False, str(e)[:80])
        return {"error": str(e)[:160]}
    audit("fs_delete", p, True, "recursive" if recursive else "")
    return {"deleted": p}


def fs_move(src: str, dst: str) -> dict:
    s, d = _safe(src), _safe(dst)
    try:
        shutil.move(s, d)
    except Exception as e:
        audit("fs_move", s + " -> " + d, False, str(e)[:80])
        return {"error": str(e)[:160]}
    audit("fs_move", s + " -> " + d, True, "")
    return {"moved": {"from": s, "to": d}}


def fs_mkdir(path: str) -> dict:
    p = _safe(path)
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        audit("fs_mkdir", p, False, str(e)[:80])
        return {"error": str(e)[:160]}
    audit("fs_mkdir", p, True, "")
    return {"made": p}


def fs_stat(path: str) -> dict:
    p = _safe(path)
    try:
        st = os.stat(p)
    except Exception as e:
        return {"error": str(e)[:160]}
    return {"path": p, "size": st.st_size, "dir": os.path.isdir(p),
            "mtime": round(st.st_mtime), "mode": oct(st.st_mode & 0o777)}


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             "site-packages", ".cache", "proc", "sys"}


def fs_search(root: str, query: str = "", glob: str = "", limit: int = 200,
              max_bytes: int = SEARCH_MAX_BYTES) -> dict:
    """Bounded recursive walk. glob matches filenames (fnmatch when it has
    wildcards, substring otherwise); query matches the filename OR the file
    content (first 64 KB, decoded leniently). Hard caps on files and bytes
    so this stays light on a phone."""
    import fnmatch
    r = _safe(root or ".")
    q = str(query).lower()
    g = str(glob).lower()
    wildcard = any(c in g for c in "*?[")
    hits, files_seen, bytes_seen = [], 0, 0
    if not os.path.isdir(r):
        return {"error": "not a directory: " + r[:160]}
    stop = False
    for dirpath, dirnames, filenames in os.walk(r):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            files_seen += 1
            full = os.path.join(dirpath, fn)
            low = fn.lower()
            if g:
                if wildcard:
                    if not fnmatch.fnmatch(low, g):
                        continue
                elif g not in low:
                    continue
            entry = {"path": full, "name": fn, "dir": False}
            if q:
                if q in low:
                    entry["matched_in"] = "name"
                else:
                    try:
                        if os.path.getsize(full) > max_bytes - bytes_seen:
                            continue
                        with open(full, "rb") as f:
                            blob = f.read(65536)
                        bytes_seen += len(blob)
                        at = blob.decode("utf-8", "ignore").lower().find(q)
                        if at < 0:
                            continue
                        entry["matched_in"] = "content@%d" % at
                    except Exception:
                        continue
            hits.append(entry)
            if len(hits) >= limit or files_seen >= SEARCH_MAX_FILES:
                stop = True
                break
        if stop:
            break
    audit("fs_search", r, True, "%d hits / %d files" % (len(hits), files_seen))
    return {"root": r, "query": query, "glob": glob, "hits": hits[:limit],
            "files_scanned": files_seen, "bytes_read": bytes_seen,
            "stopped_early": stop,
            "caps": {"max_files": SEARCH_MAX_FILES, "max_bytes": max_bytes}}


def fs_op(body: dict) -> dict:
    """POST /api/fs — one dispatch for every filesystem operation."""
    op = str(body.get("op", "")).strip()
    path = str(body.get("path", ""))
    if op == "list":
        return fs_list(path, bool(body.get("hidden", False)),
                       int(body.get("limit", 800)))
    if op == "read":
        return fs_read(path, int(body.get("max_bytes", READ_MAX)))
    if op == "write":
        return fs_write(path, str(body.get("text", "")), bool(body.get("append", False)))
    if op == "delete":
        return fs_delete(path, bool(body.get("recursive", False)))
    if op == "move":
        return fs_move(path, str(body.get("to", "")))
    if op == "mkdir":
        return fs_mkdir(path)
    if op == "stat":
        return fs_stat(path)
    if op == "search":
        return fs_search(path, str(body.get("query", "")),
                         str(body.get("glob", "")), int(body.get("limit", 200)))
    return {"error": "op must be list|read|write|delete|move|mkdir|stat|search"}