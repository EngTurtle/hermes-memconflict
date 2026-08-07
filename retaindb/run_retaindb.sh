#!/usr/bin/env bash
# Entrypoint for running the MemConflict benchmark against RetainDB Local.
#
# This needs OPENROUTER_API_KEY in the environment. Unlike Mnemosyne and
# Hindsight, RetainDB Local has no internal LLM. Memory building and
# retrieval run entirely in the local Node server. So only one LLM role uses
# gpt-oss-120b here:
#   * The answer and judge LLM (MemConflict's llm_request helper), configured
#     through OPENAI_* (an OpenAI-compatible client pointed at OpenRouter).
#
# RetainDB embeddings run locally. The original run docs found the small
# local embedding model fastest. Default is 'hash' (zero-dependency,
# deterministic), or 'local-transformers' (Xenova/all-MiniLM-L6-v2,
# semantic). Select with RETAINDB_EMBEDDING_PROVIDER (see
# docs/BENCHMARK_MATRIX.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# --- Answer + judge LLM (MemConflict llm_request) ---------------------------
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="${OPENAI_MODEL:-openai/gpt-oss-120b}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.2}"
# gpt-oss-120b is a reasoning model. Reasoning tokens draw from the same
# completion budget. Give it headroom, or the answer text comes back empty.
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-2048}"
# 'low' keeps reasoning short (about 9 tokens), so answers are fast and not
# truncated. Leave it empty for the model's default reasoning.
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-low}"

# --- RetainDB Local embeddings: native local provider -----------------------
# 'hash' (default) needs nothing. 'local-transformers' downloads a MiniLM
# model on first use. HF's Xet backend is disabled, so weights come over the
# classic CDN.
export RETAINDB_EMBEDDING_PROVIDER="${RETAINDB_EMBEDDING_PROVIDER:-hash}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# node must be on PATH, so the RetainDB server manager can spawn dist/cli.js.
if ! command -v node >/dev/null 2>&1; then
  if [[ -x /opt/node22/bin/node ]]; then export PATH="/opt/node22/bin:$PATH"; fi
fi

exec "$@"
