#!/usr/bin/env bash
# This script is the entry point for the MemConflict benchmark against Mnemosyne.
#
# This script requires OPENROUTER_API_KEY in the environment.
# All LLM traffic goes through OpenRouter: answer generation, the LLM judge, and Mnemosyne embeddings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# --- Answer + judge LLM (used by the MemConflict llm_request helper) --------
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="${OPENAI_MODEL:-xiaomi/mimo-v2.5}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.2}"
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-1024}"
# NOTE: xiaomi/mimo-v2.5 ignores the OpenRouter reasoning controls.
# Both effort:none and reasoning:{enabled:false} still emit about 100 to 170
# reasoning tokens per call. So there is no fast path. Each call takes about
# 5 to 10 seconds no matter what. This script therefore runs the model at its
# natural, default reasoning level, and gets throughput from parallelism instead.
# Leave this setting empty for default reasoning. Set it to, for example, "low"
# only if a future model honors that value.
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-}"

# --- Mnemosyne embeddings: native local model (Mnemosyne's default config) ----
# The model is BAAI/bge-small-en-v1.5 (384-dim), the exact model behind
# Mnemosyne's published LongMemEval numbers. It runs locally through
# fastembed and ONNX, at about 650 embeds per second on this machine.
# This script disables the HF Xet backend, so weights download over the
# classic CDN path (*.cdn.hf.co).
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export MNEMOSYNE_FASTEMBED_CACHE_DIR="${MNEMOSYNE_FASTEMBED_CACHE_DIR:-$ROOT/.fastembed_cache}"
export MNEMOSYNE_EMBEDDING_THREADS="${MNEMOSYNE_EMBEDDING_THREADS:-4}"
# This is an isolated, disposable Mnemosyne home. The benchmark never touches real state.
export HERMES_HOME="${HERMES_HOME:-$ROOT/.hermes}"
export MNEMOSYNE_DATA_DIR="${MNEMOSYNE_DATA_DIR:-$ROOT/.hermes/mnemosyne/data}"
mkdir -p "$HERMES_HOME/mnemosyne" "$MNEMOSYNE_FASTEMBED_CACHE_DIR"

exec "$@"
