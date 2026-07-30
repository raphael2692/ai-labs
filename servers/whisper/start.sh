#!/usr/bin/env bash
# Starts the Whisper ASR server on this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ] && [ -f .env.example ]; then
    echo "No .env found, copying .env.example -> .env"
    cp .env.example .env
fi

uv sync
uv run python -m whisper_server.main
