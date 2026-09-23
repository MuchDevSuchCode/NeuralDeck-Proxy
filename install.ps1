<#
.SYNOPSIS
    NeuralDeck installer for Windows.

.DESCRIPTION
    Creates a virtual environment next to this script, installs NeuralDeck
    into it, prints what the machine looks like, and starts the dashboard.
    Nothing is installed system-wide and nothing touches the registry.

.PARAMETER NoLaunch
    Install only; do not start NeuralDeck afterwards.

.PARAMETER NoShortcut
    Skip creating the Start Menu shortcut.

.PARAMETER Dir
    Where to put the virtual environment (default: .\venv next to this file).

.EXAMPLE
    .\install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -NoLaunch
#>
[CmdletBinding()]
param(
    [switch]$NoLaunch,
    [switch]$NoShortcut,
    [string]$Dir
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = if ($Dir) { $Dir } else { Join-Path $Root "venv" }

function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "  ! $m"  -ForegroundColor Yellow }
function Die  ($m) { Write-Host "  x $m"  -ForegroundColor Red; exit 1 }

# ── python ────────────────────────────────────────────────────────────────
function Find-Python {
    $check = 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)'
    foreach ($c in @(@{cmd="py";args=@("-3")}, @{cmd="python";args=@()},
                     @{cmd="python3";args=@()})) {
        $exe = Get-Command $c.cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        & $c.cmd @($c.args + @("-c", $check)) 2>$null
        if ($LASTEXITCODE -eq 0) { return @{ cmd = $c.cmd; args = $c.args } }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Die "no Python 3.10+ found. Install it from https://python.org/downloads (tick 'Add python.exe to PATH') and run this again."
}
$ver = & $py.cmd @($py.args + @("--version"))
Say "using $ver"

# ── venv ──────────────────────────────────────────────────────────────────
$VPy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Say "creating the virtual environment at $Venv"
    & $py.cmd @($py.args + @("-m", "venv", $Venv))
    if ($LASTEXITCODE -ne 0) { Die "could not create the virtual environment" }
}
& $VPy -m pip install --upgrade pip *> $null

# ── the package ───────────────────────────────────────────────────────────
Say "installing NeuralDeck and its dependencies"
& $VPy -m pip install -e "$Root[hub]"
if ($LASTEXITCODE -ne 0) {
    Warn "editable install failed - falling back to requirements.txt"
    & $VPy -m pip install -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Die "dependency install failed. Are you online?" }
}

# ── Start Menu shortcut ───────────────────────────────────────────────────
if (-not $NoShortcut) {
    try {
        $programs = [Environment]::GetFolderPath("Programs")
        $lnk = Join-Path $programs "NeuralDeck.lnk"
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnk)
        # pythonw keeps the console window from appearing; the dashboard is
        # the interface, not the terminal
        $wpy = Join-Path $Venv "Scripts\pythonw.exe"
        $sc.TargetPath = if (Test-Path $wpy) { $wpy } else { $VPy }
        $sc.Arguments = "-m neuraldeck up --open"
        $sc.WorkingDirectory = $Root
        $sc.Description = "NeuralDeck - local LLM dashboard and proxy"
        $sc.Save()
        Say "Start Menu shortcut: $lnk"
    } catch {
        Warn "could not create the Start Menu shortcut: $($_.Exception.Message)"
    }
}

# ── what this machine looks like ──────────────────────────────────────────
Write-Host ""
& $VPy -m neuraldeck doctor
if ($LASTEXITCODE -ne 0) { Warn "doctor reported something missing (see above)" }
Write-Host ""

Say "installed."
Say "Windows Firewall may ask to allow Python on first run - allow it for private networks, or deny it to keep NeuralDeck on this machine only."
Say "settings live in the dashboard's Settings tab - point it at your models there."

if (-not $NoLaunch) {
    $port = & $VPy -c "from neuraldeck import config; print(config.DECK_PORT)"
    Say "starting NeuralDeck - the dashboard will open at http://localhost:$port"
    Say "press Ctrl-C to stop it"
    Write-Host ""
    & $VPy -m neuraldeck up --open
} else {
    Say "start it with:  $VPy -m neuraldeck"
}
