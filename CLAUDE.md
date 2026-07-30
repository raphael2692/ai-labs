# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal AI lab: an LLM inference server (Unsloth, via `unsloth run`), a speech-to-text server (Whisper), and a
growing collection of Streamlit apps built on top of both. See `README.md` for the user-facing overview.

## Architecture

This is a **uv workspace** (`pyproject.toml` at root, `[tool.uv.workspace] members = ["servers/*", "apps/*"]`,
`exclude = ["servers/unsloth"]`). Every server and app is an independent package with its own
`pyproject.toml`. This is deliberate, not incidental:

1. **Servers run on different machines than apps, and often different machines than each other.**
   `servers/whisper/` is self-contained: copy just that folder to another machine (no root
   `pyproject.toml` present) and `uv` treats it as a standalone project with its own lock/venv. Never add a
   dependency from a server on anything under `apps/` or the other server.
2. **Apps never hardcode a server host.** They read `WHISPER_API_URL` / `UNSLOTH_API_BASE` /
   `UNSLOTH_STUDIO_AUTH_TOKEN` / `UNSLOTH_MODEL` via `ai_lab_common.settings.Settings` (pydantic-settings,
   `.env`-backed), and every app renders `ai_lab_common.sidebar.render_endpoint_sidebar()` so those values
   can be overridden per-session without editing `.env`. When adding a new app, do the same rather than
   constructing a `requests`/`OpenAI` client with a literal URL.
3. **`servers/unsloth` is deliberately *not* a uv/Python project at all** — no `pyproject.toml`, excluded
   from the workspace. Do not add one, and do not add a `unsloth` pip dependency anywhere. Unsloth Studio
   (the API server) and `pip install unsloth` ("Unsloth Core") are two unrelated products that share a
   name: Studio is installed by a standalone installer (`irm https://unsloth.ai/install.ps1 | iex` on
   Windows, `curl -fsSL https://unsloth.ai/install.sh | sh` on macOS/Linux) that manages its own isolated
   environment and puts a global `unsloth` CLI on PATH; Core is the fine-tuning library and does not
   provide the `unsloth run`/`unsloth studio` CLI. `servers/unsloth/start.*` call the global `unsloth`
   binary directly — never `uv run unsloth ...`.

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
- `servers/unsloth/` — no Python package (see point 3 above), just `start.sh`/`start.ps1` that check
  `unsloth` is on PATH, then run `unsloth run --model $UNSLOTH_MODEL -H $UNSLOTH_HOST -p $UNSLOTH_PORT -y`.
  `UNSLOTH_MODEL` here is a GGUF repo, optionally `repo:QUANT` (e.g.
  `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL`) — different from the `UNSLOTH_MODEL` apps use (the loaded
  model's *id*, as returned by `GET /v1/models`, sent in chat-completion requests). On first run the API
  key is auto-generated and printed to the console (not created through a UI step) — copy it into
  `UNSLOTH_STUDIO_AUTH_TOKEN` in both this server's `.env` and the repo-root `.env`. Binding to `0.0.0.0`
  disables Unsloth's server-side tools (web search/code exec) by default and would otherwise prompt
  interactively for confirmation; `-y` in the start scripts skips that prompt non-interactively.

`servers/whisper/start.*` `cd`s to the script's own directory, copies `.env.example` -> `.env` if missing,
`uv sync`s, then runs. Running the server module directly (e.g. `python -m whisper_server.main`) only
works with `servers/whisper` as cwd — it has `[tool.uv] package = false`, so it isn't installed into
site-packages; it's only importable when its own folder is on `sys.path` (i.e. it is cwd).
`servers/unsloth/start.*` don't touch uv at all (see point 3 above).

## Commands

```
uv sync                              # install root workspace project only (fast, minimal)
uv sync --all-packages               # install every server + app into the one shared .venv (local dev)
uv run --package <name> <cmd>        # run something inside one workspace member's dependency set
./scripts/run_app.sh <app_dir>       # run apps/<app_dir>/app.py via streamlit (Linux/macOS)
./scripts/run_app.ps1 <app_dir>      # same, Windows PowerShell
./servers/whisper/start.sh|start.ps1 # sync + run the Whisper server (run from that machine)
./servers/unsloth/start.sh|start.ps1 # run the globally-installed `unsloth` CLI (run from that machine)
```

There is no test suite, lint config, or CI in this repo yet.

## Config

Copy `.env.example` -> `.env` at whichever level you're running from (repo root for apps, or inside
`servers/whisper` / `servers/unsloth` for those servers — each has its own `.env.example`). Apps read the
root `.env`; servers read their own.
