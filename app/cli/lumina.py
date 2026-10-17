"""Lumina CLI — convert, benchmark, serve, inspect. stdlib only."""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def cmd_status(_a) -> int:
    from lumina_framework.adapters import available_formats
    print("formats:", ", ".join(available_formats()))
    sp = os.path.join(ROOT, "STATUS.md")
    if os.path.exists(sp):
        print(open(sp).read())
    return 0


def cmd_formats(_a) -> int:
    from lumina_framework.adapters import available_formats
    print(json.dumps({"formats": available_formats()}, indent=2))
    return 0


def cmd_convert(a) -> int:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.argv = ["lumina_convert.py", "--input", a.input, "--output", a.output,
                "--rung", a.rung, "--report", a.report or "/tmp/lumina_rep.json"]
    if a.no_keep_sensitive:
        sys.argv.append("--no-keep-sensitive")
    import lumina_convert as lc
    return lc.main()


def cmd_serve(a) -> int:
    sys.path.insert(0, os.path.join(ROOT, "app", "server"))
    import serve
    return serve.main(["--port", str(a.port)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lumina", description="Lumina Studio CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("formats").set_defaults(fn=cmd_formats)
    c = sub.add_parser("convert")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--rung", default="int4")
    c.add_argument("--report", default=None)
    c.add_argument("--no-keep-sensitive", action="store_true")
    c.set_defaults(fn=cmd_convert)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8130)
    s.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
