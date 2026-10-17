# Lumina Studio

Local-first LLM workspace: honest quantization (the Lumina format), a
from-scratch inference framework, and a production server with OpenAI- and
Anthropic-compatible APIs. Python stdlib only at runtime — numpy for math,
no web frameworks, no model downloads you didn't ask for.

## Quick start

```bash
cd Lumina_Studio
python3 app/server/serve.py --port 8130        # http://127.0.0.1:8130
```

Optional, production:

```bash
python3 app/server/serve.py --token SECRET    # bearer auth on /api/* and /v1/*
export LUMINA_MODELS=/path/to/models          # extra model dirs (colon-separated)
```

Open the URL. Put models (HF dirs, `.safetensors`, `.gguf`, `.lumina`) in
`/tmp` or any `LUMINA_MODELS` dir, then load one in **My Models** and chat.

## What it does

- **Chat** against local HF-dir models (fp16), converted `.lumina` files
  (int4/int3/int2/1-bit, virtual-MoE), and real-format GGUF (Q4_0/Q8_0),
  or any OpenAI-compatible cloud endpoint configured in Settings.
- **Convert** models into the Lumina format with honest byte accounting
  (every byte counted, effective bpp reported, per-layer quality reported).
  Optional **virtual MoE**: dense MLPs become routed experts (top-k) with a
  trained router; per-layer error is reported, never hidden.
- **Speculative decoding**: PLD (`"spec": true`), trained EAGLE tree and MTP
  chain (`"spec": "eagle"` / `"mtp"`) — trained on-device from the Work view;
  greedy outputs provably identical to plain decoding.
- **Doctor + plan** (Work view): verify a model is loadable and see its RAM
  placement plan before loading anything. Header reads only.
- **RAG** (Docs view): server-side chunk store with BM25 ranking and sources.
- **Evidence grounding** (opt-in per message, "Ground" toggle): claim-level
  attribution against retrieved passages with number and entity checks;
  `flag` (default) or `withhold` policy; a no-evidence question is refused
  before any model call. Measured on a fixed 62-case fixture (see Claims).
- **Auto skill fetch** (opt-in "Skill" toggle / always on in the agent loop):
  a compact BM25 skill index picks the matching skill only when the margin
  gate says it is confident; otherwise it says so and injects nothing.
  `skill_fetch` is also a tool the agent can call explicitly.
- **Context compaction + cache meter**: optional append-only checkpoint
  compaction keeps the prefix cache hitting (measured steady-state reuse
  0.9135 on the scripted workload); every turn reports real prefill
  reused/computed tokens.
- **Media generation orchestration** (Media view): declared backends —
  ComfyUI workflows over HTTP (submit → poll → fetch), A1111, and CLI tools
  (TTS/STT/ffmpeg) — with placeholder-substituted workflow templates and an
  AI-callable `media_generate` tool. Lumina does not run diffusion on the
  phone; it orchestrates remote/CLI backends, honestly reporting which are
  configured and reachable.
- **Model residency manager**: before a media job the LLM can be unloaded
  (`unload`) or unloaded-and-reloaded (`reload_after`) so the media model
  gets the RAM; refusals are honest and leave the loaded model untouched.
  True partial (layer-wise) residency is deferred — the loader materializes
  every tensor eagerly; that decision is stated in the API and the docs.
- **Arena** with server-side ELO (K=32); **System One** decisions
  (`/v1/systemone`): scored choices with probabilities, nothing generated.
- **Agent** loop (`/api/agent`) with two engines: Lumina's builtin
  fenced-tool loop, or the official **OpenAI Agents SDK** (`openai-agents`,
  optional dependency, lazy one-time import) driving the same
  operator-gated tools. `engine`: `builtin` (default) | `sdk` | `auto`;
  every call is reported either way.
- **Computer automation**: operator mode (file ops, allowlisted shell, app
  launch) with an audit log, a human-like Playwright browser driver, custom
  MCP server authoring, and ordered workflow steps — all operator-gated.
- **Claims board**: every quality claim with its honest label (verified /
  deferred / not-a-claim) — read it before trusting any number.
- **Atomic agent layer** (inspired by Atomic Agent, Atomic Chat and Atomic
  Agents): grammar-locked tool calls (typed contracts, tolerant parse, repair,
  one bounded retry), parallel read-only tool batches, a typed atom registry
  (`GET /api/atoms`, `POST /api/atoms/run`), and an opt-in SQLite FTS memory
  tier whose recall is injected in the prompt **tail** so the prefix KV cache
  stays intact.
- **Performance feasibility** (`STATUS.md`): TOPS-vs-TFLOPS study measured
  on-device — integer dotprod wins per MAC (4-6x) and a C dotprod int8 GEMV is
  3-6x fp32 BLAS GEMV on the real decode shape; decode is bandwidth-bound, so
  weight *bytes* (fp32 28 GB -> int8 7 GB -> 4-bit 3.5 GB for 7B) are the
  lever; the phone's NPU/QNN libs exist but are not loadable from this shell.
  Shipped as an optional compiled kernel (`tops.py`, gcc build cached in
  `/tmp`, `LUMINA_TOPS_DISABLE` air-gap): int8 GEMV 7.2x and a 32-layer decode
  loop 5.3x the numba path, with automatic fallback.

## APIs

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; SSE streaming; `response_format: json_object`; `stop`; `seed`; samplers (`top_k`, `min_p`, `typical_p`, `repeat_penalty`) |
| `GET /v1/models` | loaded + library + cloud |
| `POST /v1/messages` | Anthropic-compatible (non-streaming) |
| `POST /v1/systemone` | scored decisions (choice / noul) |
| `POST /api/chat` | native; `spec`: `true\|"eagle"|"mtp"`; `ground: {k,policy}`; `skill_auto`; `compact` |
| `POST /api/ground/check` | claim-level grounding of an answer vs passages (pure, no model) |
| `POST /api/boost` | self-consistency / verify / tool-checks / decompose toolbox |
| `GET /api/skills/search` | BM25 skill retrieval with calibrated confidence |
| `GET /api/cache/stats` | prefix-cache meter (per-turn reused/computed tokens) |
| `GET /api/media/status`, `/backends`, `/workflows`, `/jobs`, `/file` | media backends, workflow templates, jobs, generated files (path-jailed) |
| `POST /api/media/generate` | run a media job (`capability`, `prompt`, `workflow_id?`, `policy?`) |
| `GET/POST /api/residency` | LLM residency state + policy (`off\|unload\|reload_after`) |
| `GET /api/tops` | compiled dotprod kernel status (available, build path, reason) |
| `GET /api/atoms` | typed atomic units (tools with declared contracts + skills) |
| `POST /api/atoms/run` | run one atom by id with schema validation + repair |
| `GET /api/memory/sqlite` | SQLite FTS memory tier status + stats + injection counters |
| `POST /api/memory/sqlite` | `op: add\|recall\|search\|forget\|stats\|context` |
| `POST /api/load` / `unload` | memory-guarded; refuses loads that would OOM |
| `GET /api/doctor`, `/api/plan` | pre-load verification + RAM plan |
| `POST /api/convert`, `/api/inspect` | real codec; path-jailed (realpath), RAM-capped |
| `POST /api/rag/add`, `GET /api/rag/query` | chunk store + BM25 |
| `POST /api/agent` | operator-gated tool loop (max 5 steps); `engine`: `builtin\|sdk\|auto`; `mcp_servers: [...]` to hand SDK runs native MCP mirrors |
| `GET /api/agent/sdk` | Agents SDK state (never imports it); install/warmup hint |
| `POST /api/agent/sdk/warmup` | one-time lazy import of `openai-agents` (background by default) |
| `GET /api/logs` | request log (paths only — never bodies or keys) |
| `GET /api/hardware` | CPU/RAM/BLAS inventory + roofline recipe |

Point any OpenAI client at `--base-url http://127.0.0.1:8130/v1`.

## Security posture

- `--token` / `LUMINA_TOKEN` gates every `/api/*` and `/v1/*` route
  (constant-time compare); static UI assets stay open.
- `exec` is a read-only allowlist (ls/cat/df/…) — `python3` is deliberately
  NOT allowlisted (it was an RCE bypass).
- Convert/inspect inputs are jailed to configured model dirs with
  `realpath` (symlink-escape resistant) and RAM-capped.
- The request log records paths only, never bodies or keys.
- Operator mode (file ops / shell / app launch) is off unless enabled in
  Settings; a non-loopback bind requires the token, and every operator action
  is appended to `app/server/ops_audit.jsonl`.
- The cloud API key is stored server-side only and never sent back.

## Tests

```bash
python3 tests/run_all.py        # 21 suites: formats, codecs, parity, templates, kernels, automation, Agents SDK, compaction, grounding, boost, skill index, media, residency, TOPS kernels, agent grammar, sqlite memory
```

`openai-agents` is optional: without it the server still runs the builtin
agent loop and every SDK path degrades honestly. `LUMINA_AGENTS_DISABLE=1`
forces that mode (air-gap switch); the test suite then reports SKIP, never a
fake pass.

Heavy-class verification (real 0.5B+ models, GPU paths) runs on Kaggle
notebooks, never on a phone — see `STATUS.md` for what is measured vs
deferred, with reasons.
