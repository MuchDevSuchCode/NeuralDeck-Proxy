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
# a relative -Dir is relative to where you ran this, and has to stay
# meaningful once it is written into the shortcut
$Venv = if ($Dir) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Dir)
} else { Join-Path $Root "venv" }

function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "  ! $m"  -ForegroundColor Yellow }
function Die  ($m) { Write-Host "  x $m"  -ForegroundColor Red; exit 1 }

# Windows PowerShell 5.1 turns anything a native program writes to a
# redirected stderr into an error record, and under "Stop" that aborts the
# script (pip's warnings, "py -3" with no Python 3). Run native commands
# with "Continue" and judge them by $LASTEXITCODE instead.
function Invoke-Native ([scriptblock]$Block) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $prev }
}

# ── python ────────────────────────────────────────────────────────────────
function Find-Python {
    $check = 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)'
    foreach ($c in @(@{cmd="py";args=@("-3")}, @{cmd="python";args=@()},
                     @{cmd="python3";args=@()})) {
        $exe = Get-Command $c.cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        Invoke-Native { & $c.cmd @($c.args + @("-c", $check)) 2>$null }
        if ($LASTEXITCODE -eq 0) { return @{ cmd = $c.cmd; args = $c.args } }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Die "no Python 3.10+ found. Install it from https://python.org/downloads (tick 'Add python.exe to PATH') and run this again."
}
$ver = Invoke-Native { & $py.cmd @($py.args + @("--version")) 2>&1 }
Say "using $ver"

# ── venv ──────────────────────────────────────────────────────────────────
# A venv left half-built by an interrupted run has a python.exe but no
# working pip; rebuild it rather than trusting that the file exists.
$VPy = Join-Path $Venv "Scripts\python.exe"
$healthy = $false
if (Test-Path $VPy) {
    Invoke-Native { & $VPy -m pip --version *> $null }
    $healthy = ($LASTEXITCODE -eq 0)
}
if (-not $healthy) {
    $venvArgs = @("-m", "venv")
    if (Test-Path (Join-Path $Venv "pyvenv.cfg")) {
        Say "repairing the virtual environment at $Venv"
        $venvArgs += "--clear"
    } else {
        Say "creating the virtual environment at $Venv"
    }
    Invoke-Native { & $py.cmd @($py.args + $venvArgs + @($Venv)) }
    if ($LASTEXITCODE -ne 0) { Die "could not create the virtual environment" }
}
Invoke-Native { & $VPy -m pip install --upgrade pip *> $null }
if ($LASTEXITCODE -ne 0) { Warn "could not upgrade pip; continuing" }

# ── the package ───────────────────────────────────────────────────────────
Say "installing NeuralDeck and its dependencies"
Invoke-Native { & $VPy -m pip install -e "$Root[hub]" }
if ($LASTEXITCODE -ne 0) {
    Warn "editable install failed - falling back to requirements.txt"
    Invoke-Native { & $VPy -m pip install -r (Join-Path $Root "requirements.txt") }
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
        # the interface, not the terminal. With no console, output goes to
        # deck.log in the log directory (see cli._ensure_streams). The
        # working directory is the checkout, so `-m neuraldeck` resolves
        # even without an editable install.
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
# from the checkout, so `-m neuraldeck` resolves without an editable install
Set-Location $Root
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$env:PYTHONPATH" } else { $Root }
Write-Host ""
Invoke-Native { & $VPy -m neuraldeck doctor }
if ($LASTEXITCODE -ne 0) { Warn "doctor reported something missing (see above)" }
Write-Host ""

Say "installed."
Say "Windows Firewall may ask to allow Python on first run - allow it for private networks, or deny it to keep NeuralDeck on this machine only."
Say "settings live in the dashboard's Settings tab - point it at your models there."

if (-not $NoLaunch) {
    $port = Invoke-Native { & $VPy -c "from neuraldeck import config; print(config.DECK_PORT)" 2>$null }
    if (-not $port) { $port = 8770 }
    Say "starting NeuralDeck - the dashboard will open at http://localhost:$port"
    Say "press Ctrl-C to stop it"
    Write-Host ""
    Invoke-Native { & $VPy -m neuraldeck up --open }
} else {
    Say "start it with:  $VPy -m neuraldeck"
}
