# Runs one of the apps/ Streamlit apps from the repo root.
# Usage: ./scripts/run_app.ps1 meeting_minutes
param(
    [Parameter(Mandatory = $true)][string]$AppDir
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$appPath = "apps/$AppDir/app.py"
if (-not (Test-Path $appPath)) {
    Write-Host "No app.py found at $appPath"
    Write-Host "Available apps:"
    Get-ChildItem apps | Select-Object -ExpandProperty Name
    exit 1
}

$pyproject = Get-Content "apps/$AppDir/pyproject.toml" | Select-String '^name = "(.*)"' | Select-Object -First 1
$packageName = $pyproject.Matches[0].Groups[1].Value

uv run --package $packageName streamlit run $appPath
