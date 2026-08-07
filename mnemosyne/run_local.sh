#!/usr/bin/env bash
# This is the local-inference entry point.
# It runs the MemConflict and Mnemosyne benchmark using two local vLLM
# containers instead of OpenRouter.
# One container serves the generation model, for answers and the judge.
# The other serves the embedding model, for Mnemosyne recall.
# This script is the Windows and Git-Bash analog of run.sh.
# The file run.sh targets OpenRouter and local fastembed instead.
#
# Prereqs (the containers started for this run):
#   * vllm-gen   : serves the answer/judge LLM   on http://localhost:8000/v1
#   * vllm-embed : serves BAAI/bge-small-en-v1.5 on http://localhost:8001/v1
#   * .venv      : Windows venv with MemConflict requirements and mnemosyne[embeddings]
#
# Usage:
#   benchmark/run_local.sh python -u benchmark/eval_mnemosyne.py [args...]
#   MEMCONFLICT_JSON_MODE=1 benchmark/run_local.sh python benchmark/score_resumable.py [args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Python: Windows venv layout (.venv/Scripts), not POSIX (.venv/bin) --------
if [[ -f "$ROOT/.venv/Scripts/python.exe" ]]; then
  export PATH="$ROOT/.venv/Scripts:$PATH"
elif [[ -f "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
else
  echo "ERROR: no .venv found (expected .venv/Scripts/python.exe or .venv/bin/python)" >&2
  exit 1
fi

# --- Answer + judge LLM -> local vLLM generation server (gemma-4-e2b, FP8) -----
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"          # vLLM ignores this key. The SDK needs a non-empty value.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gemma-4-e2b}"             # --served-model-name on vllm-gen
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.2}"
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-1024}"
# This is a mimo-only reasoning knob. It is not relevant to gemma, but this script leaves it overridable.
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-}"

# --- Mnemosyne embeddings -> local vLLM embedding server (bge-small, 384-dim) --
# This uses the API path, not local fastembed. It posts to $MNEMOSYNE_EMBEDDING_API_URL/embeddings.
export MNEMOSYNE_EMBEDDINGS_VIA_API=1
export MNEMOSYNE_EMBEDDING_API_URL="${MNEMOSYNE_EMBEDDING_API_URL:-http://localhost:8001/v1}"
export MNEMOSYNE_EMBEDDING_API_KEY="${MNEMOSYNE_EMBEDDING_API_KEY:-local-vllm}"
export MNEMOSYNE_EMBEDDING_MODEL="${MNEMOSYNE_EMBEDDING_MODEL:-bge-small-en-v1.5}"
export MNEMOSYNE_EMBEDDING_DIM="${MNEMOSYNE_EMBEDDING_DIM:-384}"

# --- Mnemosyne internal LLM (fact extraction, conflict detection, consolidation)
# -> same local vLLM generation server. A feature must ask for this LLM before
# it runs, for example eval_mnemosyne.py --extract or
# MNEMOSYNE_LLM_CONFLICT_DETECTION. Setting this is harmless for the default
# baseline, because the baseline never calls it.
export MNEMOSYNE_LLM_ENABLED="${MNEMOSYNE_LLM_ENABLED:-true}"
export MNEMOSYNE_LLM_BASE_URL="${MNEMOSYNE_LLM_BASE_URL:-http://localhost:8000/v1}"
export MNEMOSYNE_LLM_API_KEY="${MNEMOSYNE_LLM_API_KEY:-local-vllm}"
export MNEMOSYNE_LLM_MODEL="${MNEMOSYNE_LLM_MODEL:-gemma-4-e2b}"
export MNEMOSYNE_LLM_MAX_TOKENS="${MNEMOSYNE_LLM_MAX_TOKENS:-512}"

# --- Isolated, disposable Mnemosyne home, so the benchmark never touches real state. ---
export HERMES_HOME="${HERMES_HOME:-$ROOT/.hermes}"
export MNEMOSYNE_DATA_DIR="${MNEMOSYNE_DATA_DIR:-$ROOT/.hermes/mnemosyne/data}"
mkdir -p "$MNEMOSYNE_DATA_DIR"

exec "$@"
