#!/usr/bin/env bash
# Entrypoint for the MemConflict benchmark against Hindsight.
#
# The environment must set OPENROUTER_API_KEY. Both LLM roles use
# gpt-oss-120b through OpenRouter:
#   * Hindsight's INTERNAL LLM (fact extraction on retain, query understanding
#     on recall). Set it with HINDSIGHT_LLM_* (eval_hindsight.py reads these).
#   * The answer and judge LLM (MemConflict's llm_request helper). Set it with
#     OPENAI_* (an OpenAI-compatible client points at OpenRouter).
#
# Embeddings and the reranker run LOCALLY (Hindsight defaults:
# BAAI/bge-small-en-v1.5 and cross-encoder/ms-marco-MiniLM-L-6-v2). No
# embedding API call is needed.
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
# gpt-oss-120b is a reasoning model. Leave headroom so the answer is not truncated.
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-2048}"
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-}"

# --- Hindsight internal LLM (fact extraction + recall) ----------------------
export HINDSIGHT_LLM_PROVIDER="${HINDSIGHT_LLM_PROVIDER:-openai}"
export HINDSIGHT_LLM_BASE_URL="${HINDSIGHT_LLM_BASE_URL:-https://openrouter.ai/api/v1}"
export HINDSIGHT_LLM_MODEL="${HINDSIGHT_LLM_MODEL:-openai/gpt-oss-120b}"
export HINDSIGHT_LLM_API_KEY="${HINDSIGHT_LLM_API_KEY:-$OPENROUTER_API_KEY}"
export HINDSIGHT_LOG_LEVEL="${HINDSIGHT_LOG_LEVEL:-warning}"

# This setting is REQUIRED for gpt-oss-120b through OpenRouter. It forces
# schema-enforced structured output for fact extraction. Without it, Hindsight
# puts the JSON schema in the prompt. Then gpt-oss returns text that fails to
# parse ("Fact extraction failed: JSONDecodeError"). Each retain() call then
# returns a 500 error. With strict schema, OpenRouter constrains the model
# grammar to valid JSON. Extraction then succeeds.
export HINDSIGHT_API_LLM_STRICT_SCHEMA="${HINDSIGHT_API_LLM_STRICT_SCHEMA:-1}"

# These settings control gpt-oss-120b reliability and latency for fact
# extraction. gpt-oss is slow. On larger chunks it still sometimes emits
# malformed or wrong-shape JSON, even under strict schema. Dropping causal-link
# extraction simplifies the extraction schema and gives the model fewer fields
# to get wrong. It does not add LLM calls.
#
# NOTE: this script keeps Hindsight's default RETAIN_CHUNK_SIZE (3000) on
# purpose. A smaller chunk size multiplies the number of gpt-oss calls. These
# calls are rate-limited and slow, so retain becomes slower overall. With the
# default HINDSIGHT_API_FAIL_ON_EXTRACTION_ERRORS=false, one bad chunk among
# several does not fail the whole retain. Override these to defaults for a
# stronger or faster model.
export HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS="${HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS:-false}"
# This is the client-side HTTP timeout, in seconds, for one retain() call.
# eval_hindsight reads this value. A large session's synchronous extraction on
# gpt-oss-120b can exceed the client's 300-second default. Raise the timeout to
# avoid that.
export HINDSIGHT_CLIENT_TIMEOUT="${HINDSIGHT_CLIENT_TIMEOUT:-900}"

# This disables background auto-consolidation for the smoke test. Otherwise,
# after each retain, the worker runs its own gpt-oss "structured" consolidation
# LLM calls to merge facts into observations. On gpt-oss-120b these calls stall
# (observed [STUCK_STACK] type=consolidation age>600s). They also compete with
# foreground extraction and drag a single-persona run past 30 minutes. The
# smoke test recalls raw extracted facts, so consolidation is not needed to
# prove the wiring.
# IMPORTANT: consolidation is part of how Hindsight resolves conflicting facts
# into a current "observation". A real MemConflict study should re-enable it
# (HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=true) on a faster and steadier
# model.
export HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION="${HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION:-false}"

# --- Hindsight embeddings + reranker: NATIVE LOCAL models (defaults) --------
# Hindsight loads BAAI/bge-small-en-v1.5 (384-dim) through fastembed/torch. It
# also loads a local cross-encoder for reranking.
# This disables HF's Xet backend so weights download over the classic CDN path.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
# The embedded daemon inherits any HINDSIGHT_API_* variable set here. Each one
# overrides the daemon's default (see docs/BENCHMARK_MATRIX.md). This script
# leaves them at their defaults for the honest baseline.
mkdir -p "$HF_HOME"

exec "$@"
