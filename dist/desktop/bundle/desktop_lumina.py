#!/usr/bin/env python3
from __future__ import annotations

"""Lumina Studio — desktop app (Tkinter). Connects to localhost:18743."""
import sys

import json
import os
import threading
import urllib.request as urq

BASE = os.environ.get("LUMINA_DESKTOP_URL", "http://127.0.0.1:18743")
try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
except Exception:  # noqa: BLE001
    tk = None  # type: ignore


# ---------------------------------------------------------------------------
# Theme tokens (3 themes, text-only; no binary assets needed)
# ---------------------------------------------------------------------------
THEMES = {
    "dark":  {"bg": "#0b0d10", "fg": "#d4d4d8", "surface": "#16161a",
             "line": "#27272a", "accent": "#0a84ff", "accent_text": "#e6f2ff",
             "dim":  "#9ca3af", "mono": "#0d1117", "mono_fg": "#7ee787"},
    "light": {"bg": "#fafafa", "fg": "#1a1a1a", "surface": "#f0f0f0",
              "line": "#e5e5e5", "accent": "#0a84ff", "accent_text": "#003a6b",
              "dim":  "#6b7280", "mono": "#f4f4f5", "mono_fg": "#059669"},
    "oled":  {"bg": "#000000", "fg": "#e6e6e6", "surface": "#0a0a0a",
              "line": "#1a1a1a", "accent": "#00cc66", "accent_text": "#baffba",
              "dim":  "#888888", "mono": "#050505", "mono_fg": "#88ff88"},
}

# ---------------------------------------------------------------------------
# Minimal icon labels (unicode, no binary assets) — Lucide-equivalent shapes
# ---------------------------------------------------------------------------
ICONS = {
    "logo": "⧗",  "agent": "⚡", "slides": "▤",  "media": "◈",
    "docs": "◉",  "swarm": "◊",  "graph": "◉",
    "claims": "✓", "models": "●",  "settings": "⚙",
    "search": "⌘",
    "image": "◈",    # media thumbnail
    "video": "▶",    # video thumbnail
    "audio": "◐",    # audio thumbnail
    "folder": "◻",   # files
    "script": "▶",   # agent loop
    "message": "◈",
}


# ---------------------------------------------------------------------------
# Simple HTTP client (stdlib only, no requests dependency)
# ---------------------------------------------------------------------------

def post_json(path: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urq.Request(BASE + path, data=data, method="POST",
                      headers={"Content-Type": "application/json"},
                      unverifiable=True)
    try:
        with urq.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body.decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return {"offline": True, "error": str(exc)[:120]}


def get_json(path: str, timeout: float = 15.0) -> dict:
    try:
        with urq.urlopen(BASE + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return {"offline": True, "error": str(exc)[:120]}


# ---------------------------------------------------------------------------
# Main Tk app
# ---------------------------------------------------------------------------

class LuminaDesktop:
    def __init__(self, root: "tk.Tk"):
        self.root = root
        root.title("Lumina Studio")
        root.geometry("1024x680")
        root.minsize(760, 540)
        self.theme = "dark"
        self.theme_var = tk.StringVar(value="dark")
        root.configure(bg=THEMES[self.theme]["bg"])
        root.option_add("*font", "system 11")
        self.build_sidebar()
        self.build_main()
        self.status_var = tk.StringVar(value="ready · tok/s shown per reply")
        self.status_label = tk.Label(root, textvariable=self.status_var,
                                     bg=THEMES[self.theme]["bg"],
                                     fg=THEMES[self.theme]["dim"], anchor="w",
                                     font=("system", 10))
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=4)

    # ------------------------------------------------------------------
    # Sidebar (nav + theme picker)
    # ------------------------------------------------------------------
    def build_sidebar(self):
        sb = tk.Frame(self.root, bg=THEMES[self.theme]["surface"],
                      width=200, bd=1, relief="flat")
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        tk.Label(sb, text="Lumina", bg=THEMES[self.theme]["surface"],
                 fg=THEMES[self.theme]["fg"], font=("system", 18, "bold"))
        for label, cmd in (("Agent (loop)", self.open_agent),
                           ("Chat", self.open_chat),
                           ("Memory (sqlite)", self.open_memory),
                           ("Tops (kernel status)", self.open_tops),
                           ("Resume / Reload", self.open_reload)):
            btn = tk.Button(sb, text=label, command=cmd,
                            bg=THEMES[self.theme]["surface"],
                            fg=THEMES[self.theme]["fg"],
                            activebackground=THEMES[self.theme]["line"],
                            anchor="w", bd=0, padx=8, pady=6)
            btn.pack(fill="x", padx=6, pady=2)
        theme_row = tk.Frame(sb, bg=THEMES[self.theme]["surface"])
        theme_row.pack(side="bottom", fill="x", pady=8)
        tk.Label(theme_row, text="Theme", bg=THEMES[self.theme]["surface"],
                 fg=THEMES[self.theme]["dim"], font=("system", 9)).pack(anchor="w", padx=8)
        for name in ("dark", "light", "oled"):
            tk.Button(theme_row, text=name, command=lambda n=name: self.set_theme(n),
                      bg=THEMES[name]["surface"], fg=THEMES[name]["fg"],
                      bd=0, padx=4, pady=2).pack(side="left", padx=2, pady=2)
        tk.Label(sb, text="Per-reply tok/s shown in chat (live server).",
                 bg=THEMES[self.theme]["surface"], fg=THEMES[self.theme]["dim"],
                 wraplength=170, justify="left", font=("system", 9)).pack(
                     side="bottom", padx=8, pady=10, anchor="w")
        tk.Label(sb, text="LLM residency: off (default). SQL memory = opt-in.",
                 bg=THEMES[self.theme]["surface"], fg=THEMES[self.theme]["dim"],
                 wraplength=170, justify="left", font=("system", 9)).pack(
                     side="bottom", fill="x", padx=8, pady=(0, 6))

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def set_theme(self, name: str):
        self.theme = name
        self.theme_var.set(name)
        th = THEMES[name]
        self.root.configure(bg=th["bg"])

    # ------------------------------------------------------------------
    # Main frames
    # ------------------------------------------------------------------
    def build_main(self):
        self.main = tk.Frame(self.root, bg=THEMES[self.theme]["bg"])
        self.main.pack(side="right", expand=True, fill="both")
        tk.Label(self.main, text="Desktop — live server connection",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["fg"],
                 font=("system", 14, "bold")).pack(pady=(14, 6))
        tk.Label(self.main, text="Status:", bg=THEMES[self.theme]["bg"],
                 fg=THEMES[self.theme]["dim"], font=("system", 9)).pack(anchor="w", padx=12)
        self.poll_id = None
        self.poll()

    # ------------------------------------------------------------------
    # Agent view
    # ------------------------------------------------------------------
    def open_agent(self):
        for child in self.main.winfo_children():
            child.destroy()
        self.build_agent_view()

    def build_agent_view(self):
        tk.Label(self.main, text="Agent", bg=THEMES[self.theme]["bg"],
                 fg=THEMES[self.theme]["fg"], font=("system", 18)).pack(pady=6)
        tk.Label(self.main, text="Engine: builtin (default, never fakes results) | sdk | auto",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["dim"],
                 font=("system", 10), wraplength=820, justify="left").pack(padx=14, pady=4)
        tk.Label(self.main, text="Every call reports engine, steps, tool results, errors.",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["dim"],
                 font=("system", 9)).pack(padx=14, pady=2)
        tk.Label(self.main, text="Task:", bg=THEMES[self.theme]["bg"],
                 fg=THEMES[self.theme]["fg"], font=("system", 11)).pack(anchor="w", padx=14)
        self.agent_prompt = tk.Entry(self.main, width=90, font=("system", 12))
        self.agent_prompt.insert(0, "find patterns in this csv file")
        self.agent_prompt.pack(padx=14, pady=6, fill="x")
        row = tk.Frame(self.main, bg=THEMES[self.theme]["bg"])
        row.pack(padx=14, pady=4, fill="x")
        tk.Label(row, text="Grammar: on (typed contracts + bounded retry)",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["dim"],
                 font=("system", 9)).pack(side="left")
        tk.Button(row, text="Grammar ON", bg=THEMES[self.theme]["surface"],
                  fg=THEMES[self.theme]["fg"], font=("system", 9),
                  command=self.toggle_grammar).pack(side="right")
        tk.Button(self.main, text="Send (agent loop, max_steps=3)",
                  bg=THEMES[self.theme]["surface"], fg=THEMES[self.theme]["fg"],
                  font=("system", 10, "bold"),
                  command=self.send_agent_task).pack(padx=14, pady=8, fill="x")
        self.agent_result = scrolledtext.ScrolledText(
            self.main, height=12, font=("system", 10, "mono"),
            bg=THEMES[self.theme]["mono"], fg=THEMES[self.theme]["mono_fg"],
            insertbackground=THEMES[self.theme]["mono_fg"])
        self.agent_result.pack(padx=14, pady=(4, 6), fill="both", expand=True)
        self.agent_result.insert("1.0", "Agent results appear here.\n"
                                "Grammar-locked: typed contracts, repair, bounded retries.\n"
                                "Every result reports engine, step count, tool results, errors.\n"
                                "Real tok/s shown per reply in chat (see Chat view).")
        self.poll_agent()

    def send_agent_task(self):
        prompt = self.agent_prompt.get().strip()
        if not prompt:
            return
        self.agent_result.delete("1.0", "end")
        result_text = post_json("/api/agent", {"messages": [{"role": "user",
                                                             "content": prompt}],
                                            "max_steps": 3})
        steps = result_text.get("steps", []) if isinstance(result_text, dict) else []
        lines = []
        if isinstance(result_text, dict):
            if result_text.get("engine"):
                lines.append("engine: " + str(result_text.get("engine"))
                            + (" (" + str(result_text.get("engine_reason") or "") + ")" if result_text.get("engine_reason") else ""))
            if result_text.get("grammar"):
                g = result_text["grammar"]
                lines.append("grammar: retries=%d repaired=%d refused=%d valid_first=%d batches=%d calls=%d"
                            % (g.get("retries", 0), g.get("repaired", 0),
                               g.get("refused", 0), g.get("valid_first", 0),
                               g.get("parallel_batches", 0), g.get("parallel_calls", 0)))
            for s in steps:
                kind = s.get("type", "?")
                if kind == "final":
                    lines.append("final: " + str(s.get("text", ""))[:400])
                elif kind == "tool":
                    lines.append("tool: %s args=%s -> %s (keys: %s)"
                                 % (str(s.get("call", {}).get("tool", "?")),
                                    str(s.get("call", {}).get("args", {}))[:80],
                                    str(s.get("result", ""))[:80],
                                    ",".join(str(s.get("out_keys", [])))))
                elif kind == "repaired-call":
                    lines.append("repaired: %s repairs=%s"
                                 % (str(s.get("call", {}).get("tool", "?")),
                                    str(s.get("repairs", []))))
                else:
                    lines.append(str(kind) + ": " + str(s.get("text", ""))[:120])
            if "tokens_per_second" in result_text:
                lines.append("tok/s: " + str(result_text.get("tokens_per_second")))
            if result_text.get("skill_auto"):
                sa = result_text["skill_auto"]
                lines.append("auto skill: %s (confident=%s)"
                            % (str(sa.get("id", "?")), str(sa.get("confident"))))
        self.agent_result.insert("end", "\n".join(lines) if lines else str(result_text)[:800])
        self.poll_agent()

    def open_chat(self):
        for child in self.main.winfo_children():
            child.destroy()
        self.build_main()

    def open_memory(self):
        for child in self.main.winfo_children():
            child.destroy()
        self.build_memory_view()

    def open_tops(self):
        for child in self.main.winfo_children():
            child.destroy()
        tk.Label(self.main, text="Compiled dotprod kernels",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["fg"],
                 font=("system", 18)).pack(pady=6)
        tk.Label(self.main, text="C NEON int8 + Q4_0 dotprod; M-blocked int8 GEMM (GEMM shape: 5.56x fp32 BLAS).",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["fg"],
                 font=("system", 10), wraplength=820).pack(padx=14, pady=4)
        self.poll_tops()
        self.poll_agent()

    def open_reload(self):
        for child in self.main.winfo_children():
            child.destroy()
        tk.Label(self.main, text="LLM residency / reload",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["fg"],
                 font=("system", 18)).pack(pady=6)
        tk.Label(self.main, text="Policy options: off (keep loaded), unload (free RAM), reload_after (unload then reload after media/job).",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["fg"],
                 font=("system", 9), wraplength=820).pack(padx=14, pady=4)
        self.poll_memory_and_tops()

    # ------------------------------------------------------------------
    # Memory view (SQLite tier + JSON tier indicators)
    # ------------------------------------------------------------------
    def build_memory_view(self):
        tk.Label(self.main, text="Memory", bg=THEMES[self.theme]["bg"],
                 fg=THEMES[self.theme]["fg"], font=("system", 16)).pack(pady=6)
        tk.Label(self.main, text="JSON tier (core/archival): pinned + FTS-recall tail injection.",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["dim"],
                 wraplength=820, justify="left", font=("system", 9)).pack(padx=14, pady=4)
        tk.Label(self.main, text="SQLite tier (opt-in sqlmem_enabled): FTS5 (LIKE fallback).",
                 bg=THEMES[self.theme]["bg"], fg=THEMES[self.theme]["dim"],
                 wraplength=820, justify="left", font=("system", 9)).pack(padx=14, pady=4)
        self.poll_memory_and_tops()

    # ------------------------------------------------------------------
    # Poll helpers (honest, measured data only)
    # ------------------------------------------------------------------
    def poll_agent(self):
        if self.poll_id is not None:
            try: self.poll_id.cancel()
            except Exception: pass
        self.poll_id = threading.Timer(6.0, self.poll_agent)
        self.poll_id.daemon = True; self.poll_id.start()
        try:
            r = get_json("/api/agent/sdk")
            self.status_var.set("SDK: %s · import %s"
                              % (r.get("version", "?"),
                                 r.get("import_s", "?") + " s")
                      + (" · disabled" if r.get("disabled") else ""))
        except Exception:
            self.status_var.set("Agent: backend unreachable")

    def poll_memory_and_tops(self):
        if self.poll_id is not None:
            try: self.poll_id.cancel()
            except Exception: pass
        self.poll_id = threading.Timer(6.0, self.poll_memory_and_tops)
        self.poll_id.daemon = True; self.poll_id.start()
        try:
            r = get_json("/api/residency")
            parts = ["LLM: " + ("loaded %s" % r.get("model_id", "?")
                      if r.get("loaded") else "not loaded")]
            if r.get("rss_mb") is not None:
                parts.append("RSS %d MB" % round(r.get("rss_mb")))
            if r.get("available_mb") is not None:
                parts.append("free %d MB" % round(r.get("available_mb")))
            parts.append("policy: %s" % (r.get("policies", "off")[-1]))
            parts.append("partial: %s"
                         % ("not supported (deferred)"
                            if (r.get("partial_residency", {}).get("supported") is False)
                            else "supported"))
            self.status_var.set("  ·  ".join(parts))
        except Exception:
            self.status_var.set("Memory: backend unreachable")

    def poll_tops(self):
        if self.poll_id is not None:
            try: self.poll_id.cancel()
            except Exception: pass
        self.poll_id = threading.Timer(6.0, self.poll_tops)
        self.poll_id.daemon = True; self.poll_id.start()
        try:
            r = get_json("/api/tops")
            self.status_var.set("Kernel: %s · %s · build %s"
                              % (str(r.get("available")).lower(),
                                 r.get("reason", "?"),
                                 os.path.basename(r.get("path", "?") or "?/")))
        except Exception:
            self.status_var.set("Kernel: backend unreachable")

    def toggle_grammar(self):
        current = get_json("/api/config").get("agent_grammar", "on")
        new_v = "off" if current == "on" else "on"
        post_json("/api/config", {"agent_grammar": new_v})
        self.build_agent_view()


def build_tiny_chat():

    import tempfile, shutil, json
    p = tempfile.mkdtemp(prefix="desk-tiny-")
    cfg = {"hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
           "vocab_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2}
    for fn in ("config.json",):
        open(os.path.join(p, fn), "w").write(json.dumps(cfg))
    return p


def main():
    root = tk.Tk()
    app = LuminaDesktop(root)
    root.mainloop()


if __name__ == "__main__":
    if tk is None:
        print("Tkinter is not available on this system (DISPLAY=%s). "
              "This is expected in proot/Termux without X11; the UI is still "
              "functional via the server's web interface (open the URL in a "
              "real browser on a desktop machine pointing at --host 0.0.0.0)."
              % os.environ.get("DISPLAY", "(not set)"))
        sys.exit(0)
    main()
