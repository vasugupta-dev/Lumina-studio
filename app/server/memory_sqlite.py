"""SQLite-backed opt-in memory tier for Lumina Studio (stdlib only).

Architecture borrowed from Atomic Agent (atomicagent.io): a durable SQLite
store with an FTS5 full-text index, and prompt injection that happens at the
TAIL of the prompt (~320 tokens) so the byte-stable prefix (system prompt)
keeps its KV cache. The existing JSON tier (serve._mem3_context_block)
prepends to the system message, which invalidates the prefix cache on every
turn; this tier is the opt-in alternative that appends at the end instead.

Honesty contract (mirrors serve.py's memory3 API):
  * dict in, dict out; failures return {"error": str}, never raise.
  * recall() of a query with no matches returns empty items, NOT an error.
  * stats()/info() report the ACTIVE engine ("fts5" or "like") truthfully;
    nothing here pretends FTS5 exists when the sqlite build lacks it.

Thread safety: every public method takes an internal lock, so one instance
may be shared across threads. Writes also run inside BEGIN IMMEDIATE
transactions so separate connections to the same file serialize at the SQLite
level (WAL + busy timeout). WAL is used with synchronous=NORMAL: safe against
process crashes, only the last transaction can be lost on power failure.

Token accounting: an item costs len(content.split()) + 8 tokens (the +8
covers the "- [scope/kind] " prefix and rank metadata in the injected block).
injected_tokens is the sum of the costs of the items actually returned; the
"[memory recall]" header itself is not counted.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    scope TEXT,
    kind TEXT,
    content TEXT,
    meta TEXT,
    ts REAL
)
"""

# External-content FTS5: the index lives in memories_fts, the bytes stay in
# memories. Triggers keep the two in sync on INSERT/UPDATE/DELETE.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, scope, kind,
    content='memories', content_rowid='id'
)
"""

# Degraded path when the sqlite build has no FTS5: a plain shadow table
# carrying the source row's id/ts explicitly; recall() then ranks with LIKE.
_FTS_FALLBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories_fts(
    content TEXT,
    scope TEXT,
    kind TEXT,
    mid INTEGER,
    ts REAL
)
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, scope, kind)
    VALUES (new.id, new.content, new.scope, new.kind);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, scope, kind)
    VALUES ('delete', old.id, old.content, old.scope, old.kind);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, scope, kind)
    VALUES ('delete', old.id, old.content, old.scope, old.kind);
    INSERT INTO memories_fts(rowid, content, scope, kind)
    VALUES (new.id, new.content, new.scope, new.kind);
END;
"""

_FTS_FALLBACK_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(content, scope, kind, mid, ts)
    VALUES (new.content, new.scope, new.kind, new.id, new.ts);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE mid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE mid = new.id;
    INSERT INTO memories_fts(content, scope, kind, mid, ts)
    VALUES (new.content, new.scope, new.kind, new.id, new.ts);
END;
"""


def _fts5_available() -> bool:
    """Probe this sqlite build for FTS5 (honest engine detection)."""
    c = sqlite3.connect(":memory:")
    try:
        c.execute("CREATE VIRTUAL TABLE _probe_fts5 USING fts5(x)")
        return True
    except Exception:
        return False
    finally:
        c.close()


class SQLiteMemory:
    """Durable memory store: FTS5 (or LIKE fallback) + tail-of-prompt blocks.

    force_like=True (tests / degraded devices) skips FTS5 entirely and uses
    the plain-table LIKE path; engine reports "like" either way.
    """

    def __init__(self, path: str, force_like: bool = False):
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        self.path = path
        self._lock = threading.RLock()
        self._fts5_available = _fts5_available()
        self.fts = self._fts5_available and not force_like
        if path != ":memory:":
            d = os.path.dirname(os.path.abspath(path))
            os.makedirs(d, exist_ok=True)
        # check_same_thread=False: the internal lock serializes all access, so
        # the instance may be shared. timeout = busy-wait for concurrent
        # writers (WAL allows one writer; the loser waits, not fails).
        self.conn = sqlite3.connect(path, check_same_thread=False,
                                    timeout=30.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self._init_db()

    # ------------------------------------------------------------------ setup

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        # WAL is best-effort: some phone filesystems lack shared-memory
        # support. If it fails the store still works in rollback-journal
        # mode; stats() then reports wal=False truthfully.
        try:
            cur.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute(_SCHEMA)
        if self.fts:
            cur.execute(_FTS_SCHEMA)
            cur.executescript(_FTS_TRIGGERS)
        else:
            cur.execute(_FTS_FALLBACK_SCHEMA)
            cur.executescript(_FTS_FALLBACK_TRIGGERS)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _tokens(query: str) -> list[str]:
        return _TOKEN_RE.findall(query.lower())

    @staticmethod
    def _item_cost(content: str) -> int:
        return len(content.split()) + 8

    def _apply_budget(self, candidates: list[dict], limit: int,
                      budget_tokens: int) -> tuple[list[dict], int, bool]:
        """Accumulate whole items (best-first) until the next would exceed the
        token budget. Never returns more than `limit` items. truncated=True
        means at least one candidate was dropped because of the budget."""
        items: list[dict] = []
        used = 0
        truncated = False
        for i, cand in enumerate(candidates[:limit]):
            cost = self._item_cost(cand["content"])
            if used + cost <= budget_tokens:
                used += cost
                items.append({"id": cand["id"], "scope": cand["scope"],
                              "kind": cand["kind"], "content": cand["content"],
                              "rank": i + 1})
            else:
                truncated = True
                break
        return items, used, truncated

    # ------------------------------------------------------------------- API

    def add(self, scope: str, kind: str, content: str,
            meta=None) -> dict:
        """Insert + index one memory. Returns {"id": n, "bytes": len} where
        bytes is the UTF-8 length of the stored content."""
        if not isinstance(scope, str):
            return {"error": "scope must be a string"}
        if not isinstance(kind, str):
            return {"error": "kind must be a string"}
        if not isinstance(content, str) or not content.strip():
            return {"error": "content must be a non-empty string"}
        if meta is None:
            meta_sql = None
        elif isinstance(meta, str):
            meta_sql = meta
        elif isinstance(meta, (dict, list)):
            meta_sql = json.dumps(meta)
        else:
            return {"error": "meta must be a string, dict, list, or None"}
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                cur.execute(
                    "INSERT INTO memories(scope, kind, content, meta, ts)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (scope, kind, content, meta_sql, time.time()))
                new_id = int(cur.lastrowid)
                cur.execute("COMMIT")
            except Exception as exc:
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
                return {"error": "add failed: %s" % exc}
        return {"id": new_id, "bytes": len(content.encode("utf-8"))}

    def recall(self, query: str, limit: int = 6,
               budget_tokens: int = 320) -> dict:
        """Ranked recall. FTS5: bm25 via MATCH. LIKE fallback: occurrence
        count, then recency. Respects budget_tokens (whole items only) and
        never returns more than `limit` items. No match -> empty items, no
        error."""
        if not isinstance(query, str):
            return {"error": "query must be a string"}
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return {"error": "limit must be a positive int"}
        if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) \
                or budget_tokens < 0:
            return {"error": "budget_tokens must be a non-negative int"}
        tokens = self._tokens(query)
        if not tokens:
            return {"items": [], "engine": "fts5" if self.fts else "like",
                    "injected_tokens": 0, "truncated": False}
        with self._lock:
            try:
                if self.fts:
                    candidates = self._recall_fts(tokens, limit)
                else:
                    candidates = self._recall_like(tokens, limit)
            except Exception as exc:
                return {"error": "recall failed: %s" % exc}
        items, used, truncated = self._apply_budget(candidates, limit,
                                                     budget_tokens)
        return {"items": items, "engine": "fts5" if self.fts else "like",
                "injected_tokens": used, "truncated": truncated}

    def _recall_fts(self, tokens: list[str], limit: int) -> list[dict]:
        # Double-quoted phrases: AND across tokens, immune to FTS5 syntax
        # characters in user text.
        match = " ".join('"%s"' % t.replace('"', '""') for t in tokens)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT m.id, m.scope, m.kind, m.content"
            " FROM memories_fts f JOIN memories m ON m.id = f.rowid"
            " WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit))
        return [dict(r) for r in cur.fetchall()]

    def _recall_like(self, tokens: list[str], limit: int) -> list[dict]:
        # Candidate filter: any token in content/scope/kind (LIKE, escaped).
        # Ranking: total occurrence count in content, then recency, then id
        # for determinism. Fetches all candidates (LIKE is a scan anyway) so
        # the Python-side ranking sees the true best-first order.
        likes, params = [], []
        for t in tokens:
            esc = t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            p = "%" + esc + "%"
            likes.append("(content LIKE ? ESCAPE '\\' OR scope LIKE ? ESCAPE '\\'"
                         " OR kind LIKE ? ESCAPE '\\')")
            params += [p, p, p]
        sql = ("SELECT mid AS id, scope, kind, content, ts FROM memories_fts"
               " WHERE " + " OR ".join(likes))
        cur = self.conn.cursor()
        cur.execute(sql, params)
        scored = []
        for r in cur.fetchall():
            low = r["content"].lower()
            score = sum(low.count(t) for t in tokens)
            if score > 0:
                scored.append((score, r["ts"], r["id"],
                               {"id": r["id"], "scope": r["scope"],
                                "kind": r["kind"], "content": r["content"]}))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return [c for _s, _ts, _id, c in scored[:limit]]

    def search_scopes(self, scope=None, kind=None, limit: int = 20) -> dict:
        """Plain SQL filter over scope/kind (no ranking). Recent first."""
        if scope is not None and not isinstance(scope, str):
            return {"error": "scope must be a string or None"}
        if kind is not None and not isinstance(kind, str):
            return {"error": "kind must be a string or None"}
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return {"error": "limit must be a positive int"}
        where, params = [], []
        if scope is not None:
            where.append("scope = ?")
            params.append(scope)
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        sql = "SELECT id, scope, kind, content, meta, ts FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            try:
                cur = self.conn.cursor()
                cur.execute(sql, params)
                rows = [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                return {"error": "search failed: %s" % exc}
        for r in rows:
            r["meta"] = self._parse_meta(r.get("meta"))
        return {"items": rows, "count": len(rows), "limit": limit}

    @staticmethod
    def _parse_meta(meta):
        if meta is None:
            return None
        try:
            return json.loads(meta)
        except Exception:
            return meta

    def forget(self, memory_id) -> dict:
        """Delete one memory by id (FTS/plain index follows via triggers).
        Returns {"forgotten": id, "deleted": bool} — deleted=False when no
        such row exists (honest, not an error)."""
        if isinstance(memory_id, bool) or not isinstance(memory_id, int):
            return {"error": "memory_id must be an int"}
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                n = cur.rowcount
                cur.execute("COMMIT")
            except Exception as exc:
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
                return {"error": "forget failed: %s" % exc}
        return {"forgotten": memory_id, "deleted": n > 0}

    def stats(self) -> dict:
        """Measured store stats. engine is the ACTIVE engine, truthfully."""
        with self._lock:
            try:
                cur = self.conn.cursor()
                cur.execute("SELECT COUNT(*) FROM memories")
                count = int(cur.fetchone()[0])
                cur.execute("PRAGMA page_count")
                pages = int(cur.fetchone()[0])
                cur.execute("PRAGMA page_size")
                page_size = int(cur.fetchone()[0])
                cur.execute("PRAGMA journal_mode")
                mode = str(cur.fetchone()[0]).lower()
            except Exception as exc:
                return {"error": "stats failed: %s" % exc}
        return {"count": count, "bytes": pages * page_size,
                "engine": "fts5" if self.fts else "like",
                "path": self.path, "wal": mode == "wal"}

    def info(self) -> dict:
        """stats() plus build-level FTS5 availability (which may differ from
        the active engine when force_like=True)."""
        d = self.stats()
        if "error" in d:
            return d
        d["fts5_available"] = self._fts5_available
        d["forced_like"] = self._fts5_available and not self.fts
        return d

    def tail_block(self, query: str, limit: int = 6,
                   budget_tokens: int = 320) -> dict:
        """Atomic-Agent-style injection block for the TAIL of the prompt.

        The caller appends `block` at the END of the prompt (after the last
        message), NEVER at the front: the byte-stable prefix (system prompt)
        then keeps its KV cache across turns. Empty block when nothing
        matches. Uses only the recalled items; injected_tokens counts item
        costs (see module docstring)."""
        r = self.recall(query, limit=limit, budget_tokens=budget_tokens)
        if "error" in r:
            return r
        if not r["items"]:
            return {"block": "", "injected_tokens": 0, "engine": r["engine"]}
        lines = ["[memory recall]"]
        for it in r["items"]:
            lines.append("- [%s/%s] %s" % (it["scope"], it["kind"],
                                           it["content"]))
        return {"block": "\n".join(lines) + "\n",
                "injected_tokens": r["injected_tokens"],
                "engine": r["engine"]}

    def close(self) -> None:
        """Commit and close. Idempotent; safe to call from any thread."""
        with self._lock:
            if self.conn is not None:
                try:
                    self.conn.commit()
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
