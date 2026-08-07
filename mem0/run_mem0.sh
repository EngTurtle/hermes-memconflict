#!/usr/bin/env bash
# Entrypoint for running the MemConflict benchmark against mem0 (host path).
#
# This script requires OPENROUTER_API_KEY in the environment. mem0 is fully
# self-hosted: it owns an internal LLM, an embedder, and a vector store.
# Under the 2.0.14 pin, the internal LLM does fact extraction ONLY; the
# product removed the 0.1.x update-memory decision. Two LLM roles are
# involved, and BOTH use gpt-oss-120b for this validation smoke:
#   * The shared answer + judge LLM (MemConflict llm_request), via OPENAI_*.
#   * mem0's INTERNAL extraction/update LLM, via MEM0_LLM_*.
# Per the best-effort ruling, mem0's internal LLM points at the SAME serving
# model the harness uses to answer. A real Hermes deployment self-hosting
# mem0 would do the same. For offline v2 runs, point both roles at the local
# vLLM instead.
#
# The embedder and vector store run LOCALLY, offline. The smoke default is a
# local HuggingFace sentence-transformers model (all-MiniLM-L6-v2, dim 384)
# with an embedded qdrant store (on-disk path, NO server). For offline/
# Docker runs, point MEM0_EMBEDDER_* at the shared vllm-embed (bge-small-
# en-v1.5, dim 384) so the embedding surface matches Mnemosyne and Hindsight
# (see docker-compose.yml).
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
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-openai/gpt-oss-120b}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.2}"
# gpt-oss-120b is a reasoning model: reasoning tokens draw from the same
# completion budget. Give the model headroom, or the answer text comes back empty.
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-2048}"
# 'low' keeps reasoning short, so answers return fast without truncation.
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-low}"

# --- mem0 internal LLM (extraction + update-memory) -------------------------
# This defaults to the same OpenRouter gpt-oss-120b as the answer/judge role.
export MEM0_LLM_PROVIDER="${MEM0_LLM_PROVIDER:-openai}"
export MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-openai/gpt-oss-120b}"
export MEM0_LLM_BASE_URL="${MEM0_LLM_BASE_URL:-https://openrouter.ai/api/v1}"
export MEM0_LLM_API_KEY="${MEM0_LLM_API_KEY:-$OPENROUTER_API_KEY}"
export MEM0_LLM_TEMPERATURE="${MEM0_LLM_TEMPERATURE:-0.7}"
export MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-2048}"

# --- mem0 embedder: NATIVE LOCAL provider -----------------------------------
# 'huggingface' downloads a small MiniLM model on first use. This script
# disables HF's Xet backend, so weights come over the classic CDN. For
# offline/Docker runs, set MEM0_EMBEDDER_PROVIDER=openai plus
# MEM0_EMBEDDER_BASE_URL=http://vllm-embed:8000/v1 plus
# MEM0_EMBEDDER_MODEL=bge-small-en-v1.5 to match the other providers.
export MEM0_EMBEDDER_PROVIDER="${MEM0_EMBEDDER_PROVIDER:-huggingface}"
export MEM0_EMBEDDER_MODEL="${MEM0_EMBEDDER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export MEM0_EMBEDDER_DIMS="${MEM0_EMBEDDER_DIMS:-384}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# Disable mem0's anonymous PostHog telemetry. The egress proxy blocks it.
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"

exec "$@"
