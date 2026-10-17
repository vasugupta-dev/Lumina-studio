# Lumina Studio App

One app, one UI, three entry points:

| Surface | Entry | Platforms |
|---|---|---|
| CLI | `app/cli/lumina.py` | mac / windows / linux / android (termux) |
| Web UI | `app/server/serve.py` + `app/web/` | any browser (desktop + android) |
| PWA | same web UI, "Add to Home screen" | android / any chromium |

The web UI is the primary polished surface and installs as a PWA — no wrapper
project and no build step. The old android/desktop wrapper stubs were removed.

Quick start (no build step, stdlib only):

```sh
python3 app/cli/lumina.py status
python3 app/server/serve.py --port 8130
# open http://localhost:8130
```
