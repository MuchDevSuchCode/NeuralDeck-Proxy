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

# Native commands run with "Continue": under "Stop", Windows PowerShell 5.1
# aborts on the first line a program writes to a redirected stderr.
function Invoke-Native ([scriptblock]$Block) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $prev }
}

# Checked on every run, not just the first: a setup interrupted half-way
# leaves a python.exe behind with no pip or no dependencies, and that is
# repaired here rather than failing later with an ImportError.
$ok = $false
if (Test-Path $Py) {
    Invoke-Native { & $Py -m pip --version *> $null }
    $ok = ($LASTEXITCODE -eq 0)
}
if (-not $ok) {
    Write-Host "Creating venv at $Venv ..."
    $venvArgs = @("-m", "venv")
    if (Test-Path (Join-Path $Venv "pyvenv.cfg")) { $venvArgs += "--clear" }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Native { & py -3 @venvArgs $Venv }
    } else {
        Invoke-Native { & python @venvArgs $Venv }
    }
    if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment at $Venv" }
}
Invoke-Native { & $Py -c "import fastapi, uvicorn, httpx, psutil, multipart" *> $null }
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies into $Venv ..."
    Invoke-Native { & $Py -m pip install --upgrade pip *> $null }
    Invoke-Native { & $Py -m pip install -r (Join-Path $Root "requirements.txt") }
    if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }
}

Set-Location $Root
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$env:PYTHONPATH" } else { $Root }
$cliArgs = $args                 # $args inside the block would be the block's own
Invoke-Native { & $Py -m neuraldeck @cliArgs }
exit $LASTEXITCODE
