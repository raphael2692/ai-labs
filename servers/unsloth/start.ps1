# Starts the Unsloth Studio LLM server on this machine.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Write-Host "No .env found, copying .env.example -> .env"
    Copy-Item ".env.example" ".env"
}

# Load .env into the current process environment.
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        $name, $value = $matches[1].Trim(), $matches[2].Trim()
        if ($value) { [System.Environment]::SetEnvironmentVariable($name, $value) }
    }
}

$hostAddr = if ($env:UNSLOTH_HOST) { $env:UNSLOTH_HOST } else { "0.0.0.0" }
$port = if ($env:UNSLOTH_PORT) { $env:UNSLOTH_PORT } else { "8888" }

uv sync

if ($env:UNSLOTH_MODEL) {
    uv run unsloth studio -H $hostAddr -p $port --model $env:UNSLOTH_MODEL
} else {
    Write-Host "UNSLOTH_MODEL is not set in .env - starting without a preselected model."
    uv run unsloth studio -H $hostAddr -p $port
}
