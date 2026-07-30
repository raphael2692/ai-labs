#!/usr/bin/env bash
# Starts the Unsloth LLM server (llama-server via `unsloth run`) on this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ] && [ -f .env.example ]; then
    echo "No .env found, copying .env.example -> .env"
    cp .env.example .env
fi

set -a
source .env
set +a

if [ -z "${UNSLOTH_MODEL:-}" ]; then
    echo "UNSLOTH_MODEL is not set. Edit .env, e.g.:"
    echo '  UNSLOTH_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL'
    exit 1
fi

UNSLOTH_HOST="${UNSLOTH_HOST:-0.0.0.0}"
UNSLOTH_PORT="${UNSLOTH_PORT:-8888}"

uv sync

# Binding to a non-loopback address disables server-side tools (web search, code
# exec) by default and prompts for confirmation; -y skips that prompt for this
# non-interactive script. Add --enable-tools yourself if you want them anyway.
uv run unsloth run --model "$UNSLOTH_MODEL" -H "$UNSLOTH_HOST" -p "$UNSLOTH_PORT" -y
