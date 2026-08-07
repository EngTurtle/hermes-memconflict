#!/usr/bin/env bash
# Host entrypoint for running the MemConflict benchmark against self-hosted
# Supermemory (the `supermemory-server` binary).
#
# This requires OPENROUTER_API_KEY in the environment for the validation
# path. Unlike RetainDB, Supermemory HAS an internal LLM (memory extraction
# and summarization), so there are TWO LLM roles here, kept strictly
# separate for fairness:
#
#   1. The shared ANSWER + JUDGE LLM (MemConflict's llm_request), configured
#      through OPENAI_* (an OpenAI-compatible client pointed at OpenRouter).
#      This is the fairness-locked model, identical across every provider.
#   2. Supermemory's INTERNAL extraction LLM, configured through
#      SUPERMEMORY_LLM_*, which _supermemory_server.py maps onto the
#      spawned server's own OPENAI_*. For this smoke, both point at
#      gpt-oss-120b on OpenRouter, but they are independent knobs (the
#      offline run can drive extraction with a local model while
#      answer/judge stays on the shared server).
#
# Supermemory embeddings run LOCALLY by default (Xenova/bge-base-en-v1.5, no key).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

# shellcheck disable=SC1091
[[ -f "$ROOT/.venv/bin/activate" ]] && source "$ROOT/.venv/bin/activate"

# --- 1. Shared answer + judge LLM (MemConflict llm_request) ------------------
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-openai/gpt-oss-120b}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.2}"
# gpt-oss-120b is a reasoning model. Its reasoning tokens draw from the
# completion budget, so give it headroom, or the answer text comes back
# empty.
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-2048}"
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-low}"

# --- 2. Supermemory's INTERNAL extraction LLM (server-side) ------------------
# Defaults to the same OpenRouter gpt-oss-120b for the validation smoke.
# Point these at a local OpenAI-compatible endpoint (Ollama, vLLM, or LM
# Studio) for a fully offline run. The server reads them as its OPENAI_*
# through _supermemory_server.py.
export SUPERMEMORY_LLM_API_KEY="${SUPERMEMORY_LLM_API_KEY:-$OPENROUTER_API_KEY}"
export SUPERMEMORY_LLM_BASE_URL="${SUPERMEMORY_LLM_BASE_URL:-https://openrouter.ai/api/v1}"
# NOTE: an OpenRouter model id ("openai/gpt-oss-120b") works here. For a
# raw OpenAI-compatible local server, use its bare model id (for example
# "gpt-oss:120b").
export SUPERMEMORY_LLM_MODEL="${SUPERMEMORY_LLM_MODEL:-openai/gpt-oss-120b}"

# --- 3. Supermemory embeddings: NATIVE LOCAL provider -----------------------
# 'local' (default) needs no key. It downloads Xenova/bge-base-en-v1.5 on
# first use. HF Xet is disabled, so weights come over the classic CDN
# (this matches other providers).
export SUPERMEMORY_EMBEDDING_PROVIDER="${SUPERMEMORY_EMBEDDING_PROVIDER:-local}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# node and npm must be on PATH, so the `supermemory` launcher (local
# start) can spawn the server binary. The install script drops the
# wrapper in ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v node >/dev/null 2>&1; then
  if [[ -x /opt/node22/bin/node ]]; then export PATH="/opt/node22/bin:$PATH"; fi
fi

exec "$@"
