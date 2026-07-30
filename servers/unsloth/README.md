# Unsloth Server

Runs [Unsloth](https://unsloth.ai) as an OpenAI/Anthropic-compatible LLM API server
(`/v1/chat/completions`, `/v1/messages`, `/v1/models`), via `unsloth run` (a `llama-server` wrapper
that loads a GGUF model). Like `servers/whisper`, this folder is self-contained and meant to be copied to
whichever machine hosts your model.

## First-time setup

1. `uv sync` — installs `unsloth` and its (CUDA/torch-specific) dependencies. If this fails, follow
   Unsloth's own install instructions for your CUDA/torch version and adjust `pyproject.toml` accordingly
   — the exact wheel constraints vary by GPU/driver.
2. Copy `.env.example` to `.env` and set `UNSLOTH_MODEL` to your favorite model, e.g.
   `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL` (repo, optionally `:QUANT`).
3. Start the server (see below). On first run, `unsloth run` prints an endpoint URL and a
   freshly-generated `sk-unsloth-…` API key to the console — Unsloth only shows it once. Copy that value
   into this folder's `.env` as `UNSLOTH_STUDIO_AUTH_TOKEN`, and into the repo-root `.env` so client apps
   can use it too. (A key can also be created/revoked later from the Studio UI: avatar (bottom-left) ->
   Settings -> API.)

## Run

```
./start.sh      # Linux/macOS
./start.ps1     # Windows (PowerShell)
```

This loads `.env`, then runs:

```
unsloth run --model $UNSLOTH_MODEL -H $UNSLOTH_HOST -p $UNSLOTH_PORT -y
```

`-y` skips the interactive tool-access confirmation prompt that `unsloth run` shows when binding to a
non-loopback host (see "Server-side tools" below).

## Configuration

| Variable                    | Default   | Notes                                                    |
|-------------------------------|-----------|------------------------------------------------------------|
| `UNSLOTH_HOST`               | `0.0.0.0` | bind address                                              |
| `UNSLOTH_PORT`               | `8888`    | bind port                                                 |
| `UNSLOTH_MODEL`              | *(required)* | GGUF repo, optionally `repo:QUANT`                    |
| `UNSLOTH_STUDIO_AUTH_TOKEN`  | *(unset)* | API key from step 3 above                                 |

Point any app at `http://<this-machine-ip>:<UNSLOTH_PORT>/v1` with that auth token
(`Authorization: Bearer sk-unsloth-…`).

## Server-side tools (web search, code execution)

`unsloth run` can expose built-in tools (Python execution, web search, bash) to the model. The default
depends on the bind address:

- `127.0.0.1` (localhost-only) — tools **on** by default.
- `0.0.0.0` / any non-loopback address (our default, for LAN access from apps on other machines) — tools
  **off** by default, since a leaked API key on a network-exposed server would otherwise mean arbitrary
  code execution on the host.

Pass `--enable-tools` (and `-y` to skip the resulting confirmation prompt) to `unsloth run` in the start
script if you want them anyway.

## Useful checks

```
curl http://localhost:8888/v1/models -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
```

Returns the loaded model's exact id (needed by some clients' "Model ID" field).
