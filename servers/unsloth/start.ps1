# Starts the Unsloth LLM server (llama-server via `unsloth run`) on this machine.
# Requires Unsloth Studio itself to be installed globally first - see README.md.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command unsloth -ErrorAction SilentlyContinue)) {
    Write-Host "'unsloth' was not found on PATH. Install Unsloth Studio first:"
    Write-Host "  irm https://unsloth.ai/install.ps1 | iex"
    Write-Host "then open a new terminal and re-run this script."
    exit 1
}

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

if (-not $env:UNSLOTH_MODEL) {
    Write-Host "UNSLOTH_MODEL is not set. Edit .env, e.g.:"
    Write-Host '  UNSLOTH_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL'
    exit 1
}

$hostAddr = if ($env:UNSLOTH_HOST) { $env:UNSLOTH_HOST } else { "0.0.0.0" }
$port = if ($env:UNSLOTH_PORT) { $env:UNSLOTH_PORT } else { "8888" }

# Binding to a non-loopback address disables server-side tools (web search, code
# exec) by default and prompts for confirmation; -y skips that prompt for this
# non-interactive script. Add --enable-tools yourself if you want them anyway.
unsloth run --model $env:UNSLOTH_MODEL -H $hostAddr -p $port -y
