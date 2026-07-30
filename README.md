# ai-lab

Personal AI lab: an LLM inference server (Unsloth, via `unsloth run`), a speech-to-text server (Whisper),
and a growing collection of Streamlit apps built on top of both.

## Layout

```
servers/
  unsloth/     # OpenAI/Anthropic-compatible LLM API server (unsloth run)
  whisper/     # GPU-backed speech-to-text HTTP API (faster-whisper)
apps/
  common/      # shared config + HTTP clients used by every app
  transcribe/  # transcribe a recording via the Whisper server
  chat/        # minimal chat UI against the Unsloth server
  _template/   # copy this to start a new app
scripts/
  run_app.sh / run_app.ps1   # run any apps/<name> from the repo root
```

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): every server and app
is its own package with its own `pyproject.toml`, so dependencies don't bleed between them and each
`servers/*` folder can be copied to a different machine on its own (see below). `servers/unsloth` is the
exception — it's driven by a globally-installed CLI rather than a Python project (see its README), so it
has no `pyproject.toml` and is excluded from the workspace.

## Servers live wherever they want

`servers/unsloth` and `servers/whisper` are each self-contained — copy just that folder to whatever
machine has the GPU you want to run it on, then `cd` in and run its `start.sh`/`start.ps1`. They don't need
the rest of this repo.

Apps never hardcode `localhost` for these — they read `WHISPER_API_URL` / `UNSLOTH_API_BASE` from `.env`
(or override per-session in the sidebar), so a given app can point at servers on any host.

1. **Whisper server** — see [servers/whisper/README.md](servers/whisper/README.md).
2. **Unsloth server** — see [servers/unsloth/README.md](servers/unsloth/README.md). Prints its API key to
   the console on first run — copy it into `.env`.

## Apps

All apps are Streamlit and share `apps/common` (`ai-lab-common`) for config/sidebar/HTTP clients — see
[apps/common/README.md](apps/common/README.md).

Copy `.env.example` to `.env` at the repo root and fill in your servers' addresses + the Unsloth API key,
then:

```
./scripts/run_app.sh transcribe     # Linux/macOS
./scripts/run_app.ps1 transcribe    # Windows
```

To add a new app, copy `apps/_template` — see [apps/_template/README.md](apps/_template/README.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
cp .env.example .env   # then edit it
```

`uv sync --all-packages` at the root installs every workspace member (the Whisper server *and* all apps)
into one shared `.venv`, which is convenient for local dev where everything runs on one machine. When
deploying the Whisper server to a different machine, run `uv sync` inside just that folder instead (see
its README). The Unsloth server needs no `uv sync` at all — see
[servers/unsloth/README.md](servers/unsloth/README.md).
