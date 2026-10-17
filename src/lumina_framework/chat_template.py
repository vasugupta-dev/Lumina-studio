"""Minimal Jinja-subset renderer for model chat templates (stdlib only).

Supports exactly what real chat templates use (verified by differential test
against Jinja2 on Qwen2.5's real 2.4KB template): {{ }}, {% if/elif/else/endif
%}, {% for x in y %}/endfor (incl. tuple unpack), {% set %}, {# comments #},
whitespace control (-), literals, attribute/index access, ==/!=/</>/and/or/
not/in, `is defined`, loop (index/index0/first/last/length), and the filters
tojson/trim/join/length/list/first/last/default/string/int.

Anything else raises TemplateError naming the construct — never silently
renders wrong text. Undefined names behave Jinja-like (falsy, chainable).
"""
from __future__ import annotations

import json
import re


class TemplateError(Exception):
    pass


class _Undef:
    __slots__ = ()

    def __repr__(self):  # noqa: D105
        return "Undefined"

    def __bool__(self):
        return False

    def __eq__(self, o):
        return isinstance(o, _Undef) or o is None

    def __ne__(self, o):
        return not self.__eq__(o)

    def __getattr__(self, _):
        return self

    def __getitem__(self, _):
        return self

    def __iter__(self):
        return iter(())

    def __str__(self):
        return ""


UNDEF = _Undef()
_TOKEN = re.compile(r"(\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\})", re.S)


def _filters():
    return {
        "tojson": lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True),
        "trim": lambda v: str(v).strip(),
        "join": lambda v, d="": d.join(str(x) for x in v),
        "length": lambda v: len(v),
        "list": lambda v: list(v),
        "first": lambda v: next(iter(v)),
        "last": lambda v: list(v)[-1],
        "default": lambda v, d="": v if v not in (None, UNDEF, "") else d,
        "string": lambda v: str(v),
        "int": lambda v: int(v),
    }


class _Parser:
    def __init__(self, src: str):
        self.toks = self._lex(src)
        self.pos = 0

    def _lex(self, src):
        toks = []
        parts = _TOKEN.split(src)
        for p in parts:
            if not p:
                continue
            if p.startswith("{{") and p.endswith("}}"):
                inner = p[2:-2]
                ls, rs = inner.startswith("-"), inner.endswith("-")
                inner = inner.strip()
                if ls:
                    inner = inner[1:].strip()
                if rs:
                    inner = inner[:-1].strip()
                toks.append(("EXPR", inner, ls, rs))
            elif p.startswith("{%") and p.endswith("%}"):
                inner = p[2:-2]
                ls = inner.startswith("-")
                rs = inner.endswith("-")
                inner = inner.strip()
                if ls:
                    inner = inner[1:].strip()
                if rs:
                    inner = inner[:-1].strip()
                toks.append(("STMT", inner, ls, rs))
            elif p.startswith("{#"):
                continue
            else:
                toks.append(("TEXT", p, False, False))
        # Jinja '-' is lexical: strip TEXT around stipping tags up front.
        # (Skipped branches never emit, so branch-aware stripping is wrong.)
        for i, t in enumerate(toks):
            if t[0] != "TEXT":
                continue
            txt = t[1]
            if i > 0 and toks[i - 1][0] in ("EXPR", "STMT") and toks[i - 1][3]:
                txt = txt.lstrip(" \t\r\n")
            if i + 1 < len(toks) and toks[i + 1][0] in ("EXPR", "STMT") and toks[i + 1][2]:
                txt = txt.rstrip(" \t\r\n")
            toks[i] = ("TEXT", txt, False, False)
        return toks

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else ("END", "", False, False)

    def next(self):
        t = self.peek()
        self.pos += 1
        return t


class _Expr:
    """Recursive-descent expression parser over a token list."""

    OPS = {"or": 1, "and": 2, "not": 3, "in": 4, "==": 4, "!=": 4, "<": 5,
           ">": 5, "<=": 5, ">=": 5, "|": 6, "+": 7, "-": 7, "*": 8, "/": 8,
           "%": 8}

    def __init__(self, text: str):
        self.ts = self._tok(text)
        self.i = 0

    def _tok(self, s):
        out = []
        i = 0
        while i < len(s):
            c = s[i]
            if c in " \t\r\n":
                i += 1
                continue
            if c in "\"'":
                j = i + 1
                buf = []
                while j < len(s):
                    if s[j] == "\\" and j + 1 < len(s):
                        buf.append(s[j:j + 2])
                        j += 2
                        continue
                    if s[j] == c:
                        break
                    buf.append(s[j])
                    j += 1
                if j >= len(s):
                    raise TemplateError("unterminated string")
                out.append(("STR", eval(c + "".join(buf) + c,
                                        {"__builtins__": {}}, {})))
                i = j + 1
                continue
            if c.isdigit() or (c == "." and i + 1 < len(s) and s[i + 1].isdigit()):
                m = re.match(r"\d+(\.\d+)?", s[i:])
                out.append(("NUM", float(m.group(0)) if "." in m.group(0) else int(m.group(0))))
                i += len(m.group(0))
                continue
            if c.isalpha() or c == "_":
                m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
                out.append(("NAME", m.group(0)))
                i += len(m.group(0))
                continue
            two = s[i:i + 2]
            if two in ("==", "!=", "<=", ">="):
                out.append(("OP", two))
                i += 2
                continue
            if c in "<>+-*/%()[],.|":
                out.append(("OP", c))
                i += 1
                continue
            raise TemplateError("bad char in expression: " + c)
        # merge multi-word ops
        merged = []
        i = 0
        while i < len(out):
            if (out[i] == ("NAME", "is") and i + 2 < len(out)
                    and out[i + 1] == ("NAME", "not")
                    and out[i + 2][0] == "NAME"):
                merged.append(("OP", "is not"))
                merged.append(("NAME", out[i + 2][1]))
                i += 3
            elif out[i] == ("NAME", "is") and i + 1 < len(out) and out[i + 1][0] == "NAME":
                merged.append(("OP", "is"))
                merged.append(("NAME", out[i + 1][1]))
                i += 2
            elif out[i][0] == "NAME" and out[i][1] in ("and", "or", "not", "in"):
                merged.append(("OP", out[i][1]))
                i += 1
            else:
                merged.append(out[i])
                i += 1
        return merged

    def parse(self):
        v = self._or()
        if self.i != len(self.ts):
            raise TemplateError("trailing tokens in expression")
        return v

    def _peek(self):
        return self.ts[self.i] if self.i < len(self.ts) else (None, None)

    def _or(self):
        v = self._and()
        while self._peek() == ("OP", "or"):
            self.i += 1
            r = self._and()
            v = v or r
        return v

    def _and(self):
        v = self._cmp()
        while self._peek() == ("OP", "and"):
            self.i += 1
            r = self._cmp()
            v = v and r
        return v

    def _cmp(self):
        v = self._add()
        while True:
            p = self._peek()
            if p[0] == "OP" and p[1] in ("==", "!=", "<", ">", "<=", ">=", "in",
                                         "is", "is not"):
                self.i += 1
                if p[1] == "is" or p[1] == "is not":
                    t = self._peek()
                    if t[0] != "NAME" or t[1] != "defined":
                        raise TemplateError("only 'is defined' is supported")
                    self.i += 1
                    v = isinstance(v, _Undef)
                    if p[1] == "is":
                        v = not v
                    continue
                r = self._add()
                if p[1] == "==":
                    v = (v == r)
                elif p[1] == "!=":
                    v = (v != r)
                elif p[1] == "in":
                    try:
                        v = (v in r)
                    except Exception:
                        v = False
                elif p[1] == "<":
                    v = (v < r)
                elif p[1] == ">":
                    v = (v > r)
                elif p[1] == "<=":
                    v = (v <= r)
                else:
                    v = (v >= r)
            elif p == ("OP", "|"):
                self.i += 1
                f = self._peek()
                if f[0] != "NAME":
                    raise TemplateError("filter name expected")
                self.i += 1
                args = []
                if self._peek() == ("OP", "("):
                    self.i += 1
                    if self._peek() != ("OP", ")"):
                        args.append(self._or())
                        while self._peek() == ("OP", ","):
                            self.i += 1
                            args.append(self._or())
                    if self._peek() != ("OP", ")"):
                        raise TemplateError(") expected in filter call")
                    self.i += 1
                fn = _filters().get(f[1])
                if fn is None:
                    raise TemplateError("unknown filter: " + f[1])
                v = fn(v, *args)
            else:
                return v
        return v

    def _add(self):
        v = self._mul()
        while self._peek() in (("OP", "+"), ("OP", "-")):
            op = self.ts[self.i][1]
            self.i += 1
            r = self._mul()
            if isinstance(v, _Undef) or isinstance(r, _Undef):
                v = _Undef()
            elif op == "+":
                v = v + r
            else:
                v = v - r
        return v

    def _mul(self):
        v = self._unary()
        while self._peek() in (("OP", "*"), ("OP", "/"), ("OP", "%")):
            op = self.ts[self.i][1]
            self.i += 1
            r = self._unary()
            if op == "*":
                v = v * r
            elif op == "/":
                v = v / r
            else:
                v = v % r
        return v

    def _unary(self):
        if self._peek() == ("OP", "not"):
            self.i += 1
            return not self._unary()
        if self._peek() == ("OP", "-"):
            self.i += 1
            return -self._unary()
        return self._postfix()

    def _postfix(self):
        v = self._atom()
        while True:
            p = self._peek()
            if p == ("OP", "."):
                self.i += 1
                f = self._peek()
                if f[0] != "NAME":
                    raise TemplateError("attribute name expected")
                self.i += 1
                v = self._get(v, f[1])
            elif p == ("OP", "["):
                self.i += 1
                k = self._or()
                if self._peek() != ("OP", "]"):
                    raise TemplateError("] expected")
                self.i += 1
                v = self._get(v, k)
            else:
                return v

    def _atom(self):
        t, val = self._peek()
        if t == "STR":
            self.i += 1
            return val
        if t == "NUM":
            self.i += 1
            return val
        if t == "NAME":
            self.i += 1
            if val in ("True", "true"):
                return True
            if val in ("False", "false"):
                return False
            if val in ("None", "none", "null"):
                return None
            return _CTX.get(val, UNDEF)
        if t == "OP" and val == "(":
            self.i += 1
            v = self._or()
            if self._peek() != ("OP", ")"):
                raise TemplateError(") expected")
            self.i += 1
            return v
        if t == "OP" and val == "[":
            self.i += 1
            items = []
            if self._peek() != ("OP", "]"):
                items.append(self._or())
                while self._peek() == ("OP", ","):
                    self.i += 1
                    items.append(self._or())
            if self._peek() != ("OP", "]"):
                raise TemplateError("] expected")
            self.i += 1
            return items
        raise TemplateError("unexpected token in expression")

    @staticmethod
    def _get(v, k):
        if isinstance(v, _Undef):
            return UNDEF
        try:
            if isinstance(v, dict):
                return v.get(k, UNDEF)
            if isinstance(v, (list, tuple)) and isinstance(k, int):
                return v[k]
            return getattr(v, k)
        except Exception:
            return UNDEF


_CTX: dict = {}


class _Loop:
    def __init__(self, n: int, i: int):
        self.length = n
        self.index0 = i
        self.index = i + 1
        self.first = (i == 0)
        self.last = (i == n - 1)


def render(template: str, ctx: dict) -> str:
    """Render a chat template with context. Raises TemplateError on
    unsupported constructs."""
    global _CTX
    _CTX = dict(ctx)
    p = _Parser(template)
    out, _ = _block(p, set())
    # Jinja2 default keep_trailing_newline=False: one trailing newline goes
    if template.endswith("\n") and out.endswith("\n"):
        out = out[:-1]
    return out


def _block(p: _Parser, stops: set) -> tuple[str, str]:
    buf = []
    while True:
        kind, val, ls, rs = p.peek()
        if kind == "END":
            return "".join(buf), ""
        if kind == "STMT" and val.split()[0] in stops:
            return "".join(buf), val
        t = p.next()
        if kind == "TEXT":
            txt = val
            if ls:
                pass
            buf.append(txt)
        elif kind == "EXPR":
            if ls and buf:
                buf[-1] = buf[-1].rstrip(" \t\r\n")
            v = _Expr(val).parse() if val.strip() else ""
            buf.append("" if isinstance(v, _Undef) else str(v))
            if rs:
                # strip leading whitespace of next text
                nxt = p.peek()
                if nxt[0] == "TEXT":
                    p.next()
                    buf.append(nxt[1].lstrip(" \t\r\n"))
        elif kind == "STMT":
            if ls and buf:
                buf[-1] = buf[-1].rstrip(" \t\r\n")
            word = val.split()[0] if val.split() else ""
            if word == "if":
                buf.append(_if(p, val[2:].strip()))
            elif word == "for":
                buf.append(_for(p, val[3:].strip()))
            elif word == "set":
                _set(p, val[3:].strip())
            else:
                raise TemplateError("unsupported statement: " + word)
            if rs:
                nxt = p.peek()
                if nxt[0] == "TEXT":
                    p.next()
                    buf.append(nxt[1].lstrip(" \t\r\n"))
    return "".join(buf), ""


def _if(p: _Parser, cond: str) -> str:
    global _CTX
    taken = False
    out = ""
    if _Expr(cond).parse():
        out, stop = _block(p, {"elif", "else", "endif"})
        taken = True
    else:
        stop = _skip_to(p, {"elif", "else", "endif"})
    while stop.startswith("elif"):
        body = stop[4:].strip()
        p.next()  # consume the elif token before rendering/skipping
        if not taken and _Expr(body).parse():
            o, stop = _block(p, {"elif", "else", "endif"})
            out += o
            taken = True
        else:
            stop = _skip_to(p, {"elif", "else", "endif"})
    if stop.startswith("else"):
        p.next()  # consume else
        if not taken:
            o, stop = _block(p, {"endif"})
            out += o
            taken = True
        else:
            stop = _skip_to(p, {"endif"})
    _expect(p, "endif")
    return out


def _skip_to(p: _Parser, stops: set) -> str:
    """Advance past tokens until a stop-word at depth 0 WITHOUT consuming it
    (same contract as _block: the caller consumes via _expect)."""
    depth = 0
    while True:
        kind, val, ls, rs = p.peek()
        if kind == "END":
            raise TemplateError("unterminated block")
        if kind != "STMT":
            p.next()
            continue
        word = val.split()[0] if val.split() else ""
        if word in ("if", "for"):
            depth += 1
            p.next()
        elif word in ("endif", "endfor"):
            if depth == 0 and word in stops:
                return val
            depth -= 1
            p.next()
        elif depth == 0 and word in stops:
            return val
        else:
            p.next()
    return ""


def _expect(p: _Parser, word: str) -> None:
    kind, val, _, _ = p.next()
    if kind != "STMT" or (val.split()[0] if val.split() else "") != word:
        raise TemplateError(word + " expected")


def _for(p: _Parser, spec: str) -> str:
    global _CTX
    m = re.match(r"(.+?)\s+in\s+(.+)", spec, re.S)
    if not m:
        raise TemplateError("bad for: " + spec)
    targets = [t.strip() for t in m.group(1).split(",")]
    seq = _Expr(m.group(2)).parse()
    if isinstance(seq, _Undef):
        seq = []
    try:
        items = list(seq)
    except Exception:
        raise TemplateError("for over non-iterable")
    # capture loop body tokens
    start = p.pos
    depth = 0
    body_toks = []
    while True:
        kind, val, ls, rs = p.next()
        if kind == "END":
            raise TemplateError("unterminated for")
        if kind == "STMT":
            word = val.split()[0] if val.split() else ""
            if word == "for":
                depth += 1
            elif word == "endfor":
                if depth == 0:
                    break
                depth -= 1
        body_toks.append((kind, val, ls, rs))
    out = []
    old = dict(_CTX)
    try:
        for i, it in enumerate(items):
            _CTX["loop"] = _Loop(len(items), i)
            if len(targets) == 1:
                _CTX[targets[0]] = it
            else:
                try:
                    parts = list(it)
                except Exception:
                    raise TemplateError("cannot unpack loop item")
                if len(parts) != len(targets):
                    raise TemplateError("loop unpack arity")
                for name, pv in zip(targets, parts):
                    _CTX[name] = pv
            sub = _Parser.__new__(_Parser)
            sub.toks = body_toks
            sub.pos = 0
            buf, _ = _block(sub, set())
            out.append(buf)
    finally:
        _CTX.clear()
        _CTX.update(old)
    return "".join(out)


def _set(p: _Parser, spec: str) -> None:
    global _CTX
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)", spec, re.S)
    if not m:
        raise TemplateError("bad set: " + spec)
    _CTX[m.group(1)] = _Expr(m.group(2)).parse()
