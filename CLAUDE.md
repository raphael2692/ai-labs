# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal AI lab: an LLM inference server (Unsloth Studio), a speech-to-text server (Whisper), and a
growing collection of Streamlit apps built on top of both. See `README.md` for the user-facing overview.

## Architecture

This is a **uv workspace** (`pyproject.toml` at root, `[tool.uv.workspace] members = ["servers/*", "apps/*"]`).
Every server and app is an independent package with its own `pyproject.toml`. This is deliberate, not
incidental — three things depend on it:

1. **Servers run on different machines than apps, and often different machines than each other.**
   `servers/whisper/` and `servers/unsloth/` are each self-contained: if you copy just one of those
   folders to another machine (no root `pyproject.toml` present), `uv` treats it as a standalone project
   with its own lock/venv. Never add a dependency from a server on anything under `apps/` or the other
   server.
2. **Apps never hardcode a server host.** They read `WHISPER_API_URL` / `UNSLOTH_API_BASE` /
   `UNSLOTH_STUDIO_AUTH_TOKEN` / `UNSLOTH_MODEL` via `ai_lab_common.settings.Settings` (pydantic-settings,
   `.env`-backed), and every app renders `ai_lab_common.sidebar.render_endpoint_sidebar()` so those values
   can be overridden per-session without editing `.env`. When adding a new app, do the same rather than
   constructing a `requests`/`OpenAI` client with a literal URL.
3. **`servers/unsloth` pulls torch/xformers/triton/unsloth into the *same* workspace lockfile as
   everything else.** It resolves today (`uv sync --all-packages` from root), but it's the one dependency
   in this repo big enough to break workspace-wide resolution if it's ever bumped carelessly. If it ever
   conflicts with other members' deps, the fix is to remove it from `[tool.uv.workspace] members` (make it
   a fully standalone project, never resolved together with the rest) rather than fighting the resolver.

### `apps/common` (package `ai-lab-common`)

Every app depends on this via `[tool.uv.sources] ai-lab-common = { workspace = true }`. It provides:

- `settings.py` — `Settings` / `get_settings()`. Env vars map to fields case-insensitively
  (`WHISPER_API_URL` -> `whisper_api_url`), loaded from `.env` at the process's cwd (i.e. the repo root,
  when apps are launched via `scripts/run_app.*` or `uv run --package <name> streamlit run apps/<x>/app.py`
  from the root).
- `sidebar.py` — `render_endpoint_sidebar(settings) -> Settings`, returns a **new** `Settings` with any
  sidebar overrides applied (via `settings.model_copy(update=...)`); doesn't mutate the original.
- `llm_client.py` — `get_llm_client(settings)`, returns an `openai.OpenAI` pointed at `unsloth_api_base`.
- `whisper_client.py` — `transcribe_stream(url, filename, bytes, content_type)`, a generator over the
  Whisper server's NDJSON stream (`{"type": "info" | "segment" | "done" | "error", ...}`); raises
  `WhisperTranscriptionError` on a non-200 response or a mid-stream `error` event.

Changes to `apps/common` take effect in every app immediately (workspace path dependency, no publish step).

### Adding a new app

Copy `apps/_template/` to `apps/<name>/`, rename `name` in its `pyproject.toml`, edit `app.py`. It already
wires up `ai-lab-common`. Run it with `./scripts/run_app.sh <name>` / `./scripts/run_app.ps1 <name>`.

### Servers

- `servers/whisper/whisper_server/` — FastAPI app (`main.py`) + env-driven `config.py`
  (`WHISPER_HOST/PORT/MODEL_SIZE/DEVICE/COMPUTE_TYPE`). `create_app()` loads the faster-whisper model at
  import time (module-level in `create_app`, not lazily) — expect server startup to block on model load.
  `POST /v1/audio/transcriptions` streams `application/x-ndjson`; `GET /health` is a plain readiness probe.
  Windows-only: `_fix_windows_cuda_dll_path()` patches `PATH`/DLL search dirs for the `nvidia-cublas`/
  `nvidia-cudnn` pip packages, since faster-whisper's CTranslate2 backend needs those DLLs discoverable and
  they don't land on `PATH` the way system CUDA installs do.
- `servers/unsloth/` — no Python package, just `start.sh`/`start.ps1` wrapping the `unsloth studio` CLI
  (installed via the `unsloth` pip dependency in its own `pyproject.toml`). The Unsloth Studio API key is
  generated manually through its own UI/CLI on first run — there's no scripted way to do this — and then
  copied into that server's `.env` as `UNSLOTH_STUDIO_AUTH_TOKEN`.

Both server `start.*` scripts follow the same pattern: `cd` to the script's own directory, copy
`.env.example` -> `.env` if missing, `uv sync`, then run. Running the server module directly (e.g.
`python -m whisper_server.main`) only works with the server's own directory as cwd — both
`servers/whisper` and `servers/unsloth` have `[tool.uv] package = false`, so they aren't installed into
site-packages; they're only importable when their own folder is on `sys.path` (i.e. it is cwd).

## Commands

```
uv sync                              # install root workspace project only (fast, minimal)
uv sync --all-packages               # install every server + app into the one shared .venv (local dev)
uv run --package <name> <cmd>        # run something inside one workspace member's dependency set
./scripts/run_app.sh <app_dir>       # run apps/<app_dir>/app.py via streamlit (Linux/macOS)
./scripts/run_app.ps1 <app_dir>      # same, Windows PowerShell
./servers/whisper/start.sh|start.ps1 # sync + run the Whisper server (run from that machine)
./servers/unsloth/start.sh|start.ps1 # sync + run Unsloth Studio (run from that machine)
```

There is no test suite, lint config, or CI in this repo yet.

## Config

Copy `.env.example` -> `.env` at whichever level you're running from (repo root for apps, or inside
`servers/whisper` / `servers/unsloth` for those servers — each has its own `.env.example`). Apps read the
root `.env`; servers read their own.
