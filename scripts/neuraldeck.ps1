# Start NeuralDeck (proxy + dashboard) from a checkout, creating the venv on
# first run. Any arguments are passed through to the CLI, e.g.
#   .\scripts\neuraldeck.ps1 doctor
#
# If PowerShell refuses to run this, allow local scripts for this session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = if ($env:NEURALDECK_VENV) { $env:NEURALDECK_VENV } else { Join-Path $Root "venv" }
$Py   = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "Creating venv at $Venv ..."
    py -3 -m venv $Venv
    & (Join-Path $Venv "Scripts\python.exe") -m pip install --upgrade pip | Out-Null
    & (Join-Path $Venv "Scripts\python.exe") -m pip install -r (Join-Path $Root "requirements.txt")
}

Set-Location $Root
& $Py -m neuraldeck @args
