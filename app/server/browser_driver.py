"""Browser driver: Lumina drives a real Chromium the way a person would.

Playwright's sync API is not thread-safe, and the HTTP server is threaded —
so the browser lives in ONE dedicated worker thread that owns the Playwright
objects, and request threads talk to it over a command queue. Nothing is
launched until the first browser command arrives (lazy), and one browser is
reused across commands until closed.

Capabilities (all real, all measured in tests against a live local page):
  navigate, back, click, fill, type_human (per-key delay), press, hover,
  select, scroll, text, html, url, title, wait_for, screenshot, tabs,
  new_tab, close_tab, eval_js, cookies, close.

Everything is gated by operator mode in serve.py — the driver itself only
executes what it is told.
"""
from __future__ import annotations

import base64
import os
import queue
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "shots")

_TOOL_DEFS = [
    ("browser_navigate", {"url": "https://..."}),
    ("browser_click", {"selector": "css or text= ..."}),
    ("browser_fill", {"selector": "...", "value": "text to type into the field"}),
    ("browser_type_human", {"selector": "...", "value": "...", "delay_ms": 35,
                            "clear": "true to replace existing text first"}),
    ("browser_press", {"key": "Enter|Tab|Control+A|..."}),
    ("browser_text", {}),
    ("browser_screenshot", {}),
    ("browser_scroll", {"dy": 600}),
    ("browser_wait_for", {"selector": "...", "timeout_ms": 8000}),
    ("browser_select", {"selector": "select", "label_or_value": "..."}),
    ("browser_back", {}),
    ("browser_tabs", {}),
    ("browser_close", {}),
]


def tool_list() -> list[dict]:
    return [{"name": n, "description": "browser driver: " + n,
             "args": a} for n, a in _TOOL_DEFS]


class _BrowserWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="lumina-browser")
        self.q: queue.Queue = queue.Queue()
        self.pw = None
        self.browser = None
        self.context = None
        self.page = None
        self.errors: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    def _ensure(self):
        if self.page is not None and not self.page.is_closed():
            return
        from playwright.sync_api import sync_playwright
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=("Mozilla/5.0 (Linux; Android 13; Lumina) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"))
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            op, args, box = item
            try:
                box["result"] = self._dispatch(op, args or {})
            except Exception as e:
                self.errors.append(str(e)[:200])
                box["result"] = {"error": str(e)[:300]}
            finally:
                box["event"].set()

    # -- ops ---------------------------------------------------------------
    def _dispatch(self, op: str, a: dict) -> dict:
        if op == "close":
            return self._close()
        self._ensure()
        p = self.page
        if op == "navigate":
            url = str(a.get("url", ""))
            if not url.startswith(("http://", "https://", "file://", "data:")):
                return {"error": "http(s)/file/data URLs only"}
            resp = p.goto(url, wait_until=str(a.get("wait_until", "domcontentloaded")),
                          timeout=int(a.get("timeout_ms", 20000)))
            return {"url": p.url, "title": p.title(),
                    "status": resp.status if resp else None}
        if op == "back":
            p.go_back(timeout=int(a.get("timeout_ms", 15000)))
            return {"url": p.url, "title": p.title()}
        if op == "click":
            sel = str(a.get("selector", ""))
            p.click(sel, timeout=int(a.get("timeout_ms", 15000)))
            self._settle(a)
            return {"clicked": sel, "url": p.url}
        if op == "fill":
            sel, val = str(a.get("selector", "")), str(a.get("value", ""))
            p.fill(sel, val, timeout=int(a.get("timeout_ms", 15000)))
            return {"filled": sel, "chars": len(val), "url": p.url}
        if op == "type_human":
            sel, val = str(a.get("selector", "")), str(a.get("value", ""))
            delay = max(0, min(int(a.get("delay_ms", 35)), 500))
            loc = p.locator(sel).first
            try:
                loc.click(timeout=int(a.get("timeout_ms", 15000)))
            except Exception:
                pass
            if a.get("clear"):
                # only when asked: select-all + type, the way a person replaces
                try:
                    p.fill(sel, "", timeout=3000)
                except Exception:
                    pass
            loc.type(val, delay=delay)
            return {"typed": sel, "chars": len(val), "per_key_ms": delay,
                    "cleared_first": bool(a.get("clear")), "url": p.url}
        if op == "press":
            key = str(a.get("key", "Enter"))
            sel = a.get("selector")
            if sel:
                p.press(str(sel), key)
            else:
                p.keyboard.press(key)
            self._settle(a)
            return {"pressed": key, "url": p.url, "title": p.title()}
        if op == "hover":
            sel = str(a.get("selector", ""))
            p.hover(sel)
            return {"hovered": sel}
        if op == "select":
            sel = str(a.get("selector", ""))
            by = str(a.get("label_or_value", ""))
            try:
                p.select_option(sel, label=by)
            except Exception:
                p.select_option(sel, value=by)
            return {"selected": sel, "value": by}
        if op == "scroll":
            dy = int(a.get("dy", 600))
            p.mouse.wheel(0, dy)
            return {"scrolled": dy}
        if op == "text":
            body = p.inner_text("body")[:20000]
            return {"url": p.url, "title": p.title(), "text": body,
                    "chars": len(body)}
        if op == "html":
            return {"url": p.url, "html": p.content()[:20000]}
        if op == "url":
            return {"url": p.url, "title": p.title()}
        if op == "wait_for":
            sel = str(a.get("selector", "body"))
            p.wait_for_selector(sel, timeout=int(a.get("timeout_ms", 8000)))
            return {"appeared": sel, "url": p.url}
        if op == "screenshot":
            os.makedirs(SHOTS, exist_ok=True)
            name = "shot_%d.png" % int(time.time() * 1000)
            path = os.path.join(SHOTS, name)
            full = bool(a.get("full_page"))
            p.screenshot(path=path, full_page=full)
            size = os.path.getsize(path)
            out = {"path": path, "name": name, "bytes": size,
                   "url": f"/api/browser/shot/{name}", "full_page": full}
            if size < 400_000 and a.get("inline", True):
                with open(path, "rb") as f:
                    out["data_url"] = ("data:image/png;base64,"
                                       + base64.b64encode(f.read()).decode())
            return out
        if op == "tabs":
            return {"tabs": [{"i": i, "url": pg.url}
                             for i, pg in enumerate(self.context.pages)],
                    "active": self.context.pages.index(p)}
        if op == "new_tab":
            self.page = self.context.new_page()
            self.page.set_default_timeout(15000)
            if a.get("url"):
                self.page.goto(str(a["url"]), wait_until="domcontentloaded")
            return {"url": self.page.url}
        if op == "close_tab":
            if len(self.context.pages) > 1:
                p.close()
                self.page = self.context.pages[-1]
            return {"url": self.page.url}
        if op == "eval_js":
            v = p.evaluate(str(a.get("js", ""))[:4000])
            return {"value": v if isinstance(v, (int, float, str, bool, list, dict))
                    else str(v)[:2000]}
        return {"error": "unknown browser op: " + op}

    def _settle(self, a: dict):
        """Optional small wait so SPA navigation/dropdowns finish — a person
        does not click and instantly screenshot either."""
        ms = int(a.get("settle_ms", 0))
        if ms:
            try:
                self.page.wait_for_timeout(min(ms, 5000))
            except Exception:
                pass

    def _close(self):
        out = {"closed": True}
        try:
            if self.browser:
                self.browser.close()
        except Exception as e:
            out["warn"] = str(e)[:120]
        try:
            if self.pw:
                self.pw.stop()
        except Exception:
            pass
        self.pw = self.browser = self.context = self.page = None
        return out


_W: dict = {"worker": None, "lock": threading.Lock()}


def _worker() -> _BrowserWorker:
    with _W["lock"]:
        w = _W["worker"]
        if w is None or not w.is_alive():
            w = _BrowserWorker()
            w.start()
            _W["worker"] = w
        return w


def execute(op: str, args: dict | None = None, timeout: float = 90.0) -> dict:
    """Send one command to the browser worker and wait for its result."""
    w = _worker()
    box = {"event": threading.Event(), "result": None}
    w.q.put((str(op), args or {}, box))
    if not box["event"].wait(timeout):
        return {"error": "browser command timed out (%ss): %s" % (timeout, op)}
    return box["result"] if box["result"] is not None else {"error": "no result"}


def status() -> dict:
    w = _W.get("worker")
    live = bool(w and w.is_alive())
    return {"worker": live,
            "page_open": bool(w and w.page is not None and not w.page.is_closed())
            if live else False,
            "last_errors": (w.errors[-3:] if live and w.errors else []),
            "note": "chromium starts lazily on first browser command"}


def close() -> dict:
    w = _W.get("worker")
    if not w or not w.is_alive():
        return {"closed": True, "already": True}
    return execute("close")


def browser_op(body: dict) -> dict:
    """POST /api/browser {op, ...args} — thin dispatcher for the API."""
    op = str(body.get("op", "")).strip()
    if not op:
        return {"error": "op required", "ops": [n for n, _ in _TOOL_DEFS]}
    args = {k: v for k, v in body.items() if k != "op"}
    return execute(op, args)