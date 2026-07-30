# Starts the Whisper ASR server on this machine.
# Run from anywhere; this script always operates on its own folder.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Write-Host "No .env found, copying .env.example -> .env"
    Copy-Item ".env.example" ".env"
}

uv sync
uv run python -m whisper_server.main
