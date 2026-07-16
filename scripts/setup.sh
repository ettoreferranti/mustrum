#!/usr/bin/env bash
# Mustrum setup for macOS and Linux: installs uv, Python 3.12, and Ollama
# (with the two models Mustrum needs) if they're not already present, then
# `uv sync`s the project. Safe to re-run — every step is skip-if-present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_MODEL="qwen3:30b"
EMBED_MODEL="nomic-embed-text"

log() { printf '\n==> %s\n' "$1"; }

# --- uv ----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # the installer adds ~/.local/bin to shell rc files, but not this
    # already-running shell
    export PATH="$HOME/.local/bin:$PATH"
else
    log "uv already installed ($(uv --version))"
fi

# --- Python 3.12+ (managed by uv, no system package manager involved) --
log "Ensuring Python 3.12 is available to uv..."
uv python install 3.12

# --- Ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    log "Ollama already installed ($(ollama --version 2>/dev/null || echo present))"
fi

# make sure a server is reachable before pulling models
if ! ollama list >/dev/null 2>&1; then
    log "Starting Ollama server..."
    if command -v systemctl >/dev/null 2>&1 && systemctl is-enabled --quiet ollama 2>/dev/null; then
        sudo systemctl start ollama
    else
        nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
        disown
    fi
    for _ in $(seq 1 10); do
        ollama list >/dev/null 2>&1 && break
        sleep 1
    done
fi

log "Pulling $LLM_MODEL (generation)..."
ollama pull "$LLM_MODEL"

log "Pulling $EMBED_MODEL (embeddings)..."
ollama pull "$EMBED_MODEL"

# --- project dependencies -------------------------------------------------
log "Installing Mustrum's Python dependencies (uv sync)..."
(cd "$REPO_ROOT" && uv sync)

log "Done. Try: uv run mustrum ui"
