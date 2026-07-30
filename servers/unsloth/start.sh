#!/usr/bin/env bash
# Starts the Unsloth Studio LLM server on this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ] && [ -f .env.example ]; then
    echo "No .env found, copying .env.example -> .env"
    cp .env.example .env
fi

set -a
source .env
set +a

UNSLOTH_HOST="${UNSLOTH_HOST:-0.0.0.0}"
UNSLOTH_PORT="${UNSLOTH_PORT:-8888}"

uv sync

if [ -n "${UNSLOTH_MODEL:-}" ]; then
    uv run unsloth studio -H "$UNSLOTH_HOST" -p "$UNSLOTH_PORT" --model "$UNSLOTH_MODEL"
else
    echo "UNSLOTH_MODEL is not set in .env - starting without a preselected model."
    uv run unsloth studio -H "$UNSLOTH_HOST" -p "$UNSLOTH_PORT"
fi
