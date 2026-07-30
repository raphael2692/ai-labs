# Unsloth Server

Runs [Unsloth Studio](https://unsloth.ai) as an OpenAI-compatible LLM API server
(`/v1/chat/completions`, etc.). Like `servers/whisper`, this folder is self-contained and meant to be
copied to whichever GPU machine hosts your model.

## First-time setup

1. `uv sync` — installs `unsloth` and its (CUDA/torch-specific) dependencies. If this fails, follow
   Unsloth's own install instructions for your CUDA/torch version and adjust `pyproject.toml` accordingly
   — the exact wheel constraints vary by GPU/driver.
2. Copy `.env.example` to `.env` and set `UNSLOTH_MODEL` to your favorite model id/path.
3. Start the server once (see below) and generate an API key from the Unsloth Studio UI/CLI — there is no
   automated way to do this, it's a manual one-time step. Put the resulting key in `.env` as
   `UNSLOTH_STUDIO_AUTH_TOKEN` so client apps can pick it up.

## Run

```
./start.sh      # Linux/macOS
./start.ps1     # Windows (PowerShell)
```

This loads `.env`, then runs:

```
unsloth studio -H $UNSLOTH_HOST -p $UNSLOTH_PORT [--model $UNSLOTH_MODEL]
```

## Configuration

| Variable                    | Default   | Notes                                  |
|-------------------------------|-----------|------------------------------------------|
| `UNSLOTH_HOST`               | `0.0.0.0` | bind address                            |
| `UNSLOTH_PORT`               | `8888`    | bind port                               |
| `UNSLOTH_MODEL`              | *(unset)* | model id/path to preload on start       |
| `UNSLOTH_STUDIO_AUTH_TOKEN`  | *(unset)* | API key generated per step 3 above      |

Point any app at `http://<this-machine-ip>:<UNSLOTH_PORT>/v1` with that auth token.
