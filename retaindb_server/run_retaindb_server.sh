#!/usr/bin/env bash
# Host entrypoint for running the MemConflict benchmark against the RetainDB
# server edition (@retaindb/server) over OpenRouter (a cross-model host smoke).
#
# Requires OPENROUTER_API_KEY. This wires the shared answer and judge LLM
# (MemConflict's llm_request helper) to OpenRouter gpt-oss-120b through OPENAI_*.
#
# The RetainDB server also makes LLM calls, for extraction at ingest. This
# script configures only the answer and judge env for the Python adapter. A
# separate serve_local.sh call starts the server itself, with its own
# OPENAI_* env (extraction model = EXTRACTOR_MODEL). For the host smoke, both
# point at the same OpenRouter endpoint, so extraction and answer share
# gpt-oss-120b. This makes it a cross-model smoke (extraction and answer on
# gpt-oss-120b), carrying the usual OpenRouter caveat. The offline v3
# contract runs everything on vllm-gen qwen3.5-4b through the Docker path,
# with no edits here.
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
# completion budget, so give it headroom, or the answer text comes back empty.
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-2048}"
# 'low' keeps reasoning short, about 9 tokens, so answers stay fast without
# truncation. Leave this empty for the model's default reasoning.
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-low}"

# Attach-only: the adapter connects to a server that serve_local.sh already started.
export RETAINDB_SERVER_BASE_URL="${RETAINDB_SERVER_BASE_URL:-http://127.0.0.1:3000}"

exec "$@"
