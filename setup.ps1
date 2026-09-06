<#
.SYNOPSIS
    Sets PolyRAG up from a fresh clone.

.DESCRIPTION
    Runs the same steps the README lists one by one, in order, and stops at the
    first one that fails. Every step is skippable and re-runnable: it checks
    whether the work is already done before doing it, so running this twice is
    safe and only fills in what is missing.

.EXAMPLE
    .\setup.ps1
    Install everything, then print how to start the servers.

.EXAMPLE
    .\setup.ps1 -SkipModels
    Set up the code but do not download the ~4.9 GB of binaries and GGUFs.

.EXAMPLE
    .\setup.ps1 -Start
    Install everything and start the servers and the backend when done.
#>

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$SkipFrontend,
    [switch]$Start
)

# Any failing command aborts the script rather than letting the next step run on
# a broken state and report a confusing error further down.
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$script:step = 0
function Step($text) {
    $script:step++
    Write-Host ''
    Write-Host "[$script:step] $text" -ForegroundColor Cyan
}
function Ok($text)   { Write-Host "    $text" -ForegroundColor DarkGray }
function Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }
function Have($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

Write-Host 'PolyRAG setup' -ForegroundColor White
Write-Host 'Federated multi-modal RAG - local, AMD/Vulkan, no CUDA and no Docker.' -ForegroundColor DarkGray

# --------------------------------------------------------------------------- #
Step 'Checking uv'
# --------------------------------------------------------------------------- #
if (Have 'uv') {
    Ok "found $(uv --version)"
} else {
    Ok 'not found - installing from astral.sh'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # The installer extends PATH for new shells only, so this one needs telling.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Have 'uv')) {
        throw 'uv installed but is not on PATH. Open a new terminal and run setup.ps1 again.'
    }
    Ok "installed $(uv --version)"
}

# --------------------------------------------------------------------------- #
Step 'Python environment and dependencies'
# --------------------------------------------------------------------------- #
# `uv sync` creates .venv, resolves against uv.lock and installs dev extras, so
# it replaces venv + pip + pip-sync in one call. Locked, so this reproduces the
# exact versions the project was tested against.
uv sync --dev
Ok "environment ready at .venv ($(uv run python --version))"

# --------------------------------------------------------------------------- #
Step 'Binaries and models'
# --------------------------------------------------------------------------- #
if ($SkipModels) {
    Warn 'skipped (-SkipModels). The backend will not start without them.'
} else {
    Ok 'about 4.9 GB on a first run; already-present files are skipped'
    uv run python scripts/fetch_runtimes.py
}

# --------------------------------------------------------------------------- #
Step 'GPU check'
# --------------------------------------------------------------------------- #
$llama = 'bin/llama-server/llama-server.exe'
if (Test-Path $llama) {
    # Vulkan device discovery is the one thing that cannot be inferred from the
    # config: if llama-server sees no GPU it will silently fall back to CPU and
    # every measurement in the README becomes unreachable.
    $devices = & $llama --list-devices 2>&1 | Out-String
    if ($devices -match 'Vulkan') {
        ($devices -split "`n" | Where-Object { $_ -match 'Vulkan|Device' } | Select-Object -First 4) |
            ForEach-Object { Ok $_.Trim() }
    } else {
        Warn 'no Vulkan device listed - llama-server would run on CPU.'
        Warn 'Update your GPU driver, then re-run: bin\llama-server\llama-server.exe --list-devices'
    }
} else {
    Warn 'llama-server not present yet, so the GPU was not checked.'
}

# --------------------------------------------------------------------------- #
Step 'Frontend'
# --------------------------------------------------------------------------- #
if ($SkipFrontend) {
    Warn 'skipped (-SkipFrontend)'
} elseif (-not (Have 'npm')) {
    Warn 'npm not found - install Node.js 20+ from https://nodejs.org and re-run.'
    Warn 'The backend and the API work without it; only the web UI needs npm.'
} else {
    npm install --prefix frontend --no-fund --no-audit
    Ok 'frontend dependencies installed'
}

# --------------------------------------------------------------------------- #
Step 'Verifying the install'
# --------------------------------------------------------------------------- #
# The unit tests need no GPU and no servers, which makes them the right check
# here: they prove the environment is usable before anything heavy is started.
uv run pytest tests/ -q

# --------------------------------------------------------------------------- #
if ($Start) {
    Step 'Starting the servers'
    uv run python scripts/start_servers.py
    Step 'Starting the backend'
    Ok 'http://127.0.0.1:8000/docs - Ctrl+C to stop'
    uv run uvicorn app.main:app --app-dir backend --port 8000
} else {
    Write-Host ''
    Write-Host 'Done. To run it:' -ForegroundColor Green
    Write-Host ''
    Write-Host '    uv run python scripts/start_servers.py                        # models + Qdrant'
    Write-Host '    uv run uvicorn app.main:app --app-dir backend --port 8000     # API'
    Write-Host '    npm --prefix frontend run dev                                 # web UI'
    Write-Host ''
    Write-Host '    uv run python scripts/load_demo.py --reset                    # optional demo corpus'
    Write-Host ''
}
