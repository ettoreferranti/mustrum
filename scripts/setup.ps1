# Mustrum setup for Windows: installs uv, Python 3.12, and Ollama (with the
# two models Mustrum needs) if they're not already present, then `uv sync`s
# the project. Safe to re-run - every step is skip-if-present.
#
# Run from PowerShell:
#   irm file-or-url-to-this-script | iex     (or just: .\scripts\setup.ps1)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LlmModel = "qwen3:30b"
$EmbedModel = "nomic-embed-text"

function Log($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# $ErrorActionPreference only governs cmdlets; native .exe failures signal
# via $LASTEXITCODE and are silently ignored otherwise, so check explicitly
# after each step that must succeed for the rest of setup to make sense.
function Assert-Success($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# --- uv --------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Log "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # the installer updates the persistent user PATH, but not this
    # already-running session
    $localBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path $localBin) { $env:Path = "$localBin;$env:Path" }
} else {
    Log "uv already installed ($(uv --version))"
}

# --- Python 3.12+ (managed by uv, no separate installer needed) ------------
Log "Ensuring Python 3.12 is available to uv..."
uv python install 3.12
Assert-Success "uv python install 3.12"

# --- Ollama ------------------------------------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Log "Installing Ollama..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama -e --silent `
            --accept-package-agreements --accept-source-agreements
    } else {
        Log "winget not found - downloading the Ollama installer instead (interactive)"
        $installer = Join-Path $env:TEMP "OllamaSetup.exe"
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
        Start-Process -FilePath $installer -Wait
    }
    # a fresh install needs a new PATH entry picked up; re-check next section
    $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
} else {
    Log "Ollama already installed"
}

# make sure a server is reachable before pulling models (external commands
# signal failure via $LASTEXITCODE, not a catchable exception)
function Test-OllamaServer {
    ollama list *> $null
    return $LASTEXITCODE -eq 0
}

$serverUp = Test-OllamaServer
if (-not $serverUp) {
    Log "Starting Ollama server..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 10; $i++) {
        Start-Sleep -Seconds 1
        if (Test-OllamaServer) { $serverUp = $true; break }
    }
    if (-not $serverUp) {
        Log "Couldn't reach the Ollama server yet - launch the Ollama app once from the Start menu, then re-run this script."
    }
}

Log "Pulling $LlmModel (generation)..."
ollama pull $LlmModel
Assert-Success "ollama pull $LlmModel"

Log "Pulling $EmbedModel (embeddings)..."
ollama pull $EmbedModel
Assert-Success "ollama pull $EmbedModel"

# --- project dependencies ---------------------------------------------------
Log "Installing Mustrum's Python dependencies (uv sync)..."
Push-Location $RepoRoot
try { uv sync; Assert-Success "uv sync" } finally { Pop-Location }

Log "Done. Try: uv run mustrum ui"
