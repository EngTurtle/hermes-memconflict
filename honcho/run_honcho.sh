#!/usr/bin/env bash
# Entrypoint for running the MemConflict benchmark against Honcho (host path).
#
# This script requires OPENROUTER_API_KEY in the environment. Honcho is fully
# self-hosted, but it is not one process: this path spawns a FastAPI API, a
# separate deriver worker, and a per-run PostgreSQL database, and it serves
# the embedder locally. Two LLM roles are involved, and BOTH use
# gpt-oss-20b for this validation smoke:
#   * The shared answer + judge LLM (MemConflict llm_request), via OPENAI_*.
#   * Honcho's INTERNAL models (deriver, dialectic, summary, dream, peer
#     card), via HONCHO_LLM_*, which _honcho_server.py maps onto the spawned
#     children's own namespaced variables.
# Per the best-effort ruling, Honcho's internal models point at the SAME
# serving model the harness uses to answer. A real Hermes deployment
# self-hosting Honcho would do the same. For offline runs, point both roles
# at the local vLLM instead.
#
# The embedder runs LOCALLY. The host default is the fastembed shim in
# _local_embed_server.py (BAAI/bge-small-en-v1.5, dim 384). Docker runs point
# HONCHO_EMBEDDER_BASE_URL at the shared vllm-embed instead, which serves the
# same model at the same width, so the embedding surface matches Mnemosyne,
# Hindsight, and mem0.
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
export OPENAI_MODEL="${OPENAI_MODEL:-openai/gpt-oss-20b}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.2}"
# gpt-oss-20b is a reasoning model: reasoning tokens draw from the same
# completion budget. Give the model headroom, or the answer text comes back empty.
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-2048}"
# 'low' keeps reasoning short, so answers return fast without truncation.
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-low}"

# --- Honcho internal LLM (deriver, dialectic, summary, dream, peer card) -----
export HONCHO_LLM_MODEL="${HONCHO_LLM_MODEL:-openai/gpt-oss-20b}"
export HONCHO_LLM_BASE_URL="${HONCHO_LLM_BASE_URL:-https://openrouter.ai/api/v1}"
export HONCHO_LLM_API_KEY="${HONCHO_LLM_API_KEY:-$OPENROUTER_API_KEY}"
# gpt-oss-20b spends the same budget on reasoning tokens in Honcho's internal
# roles. Below about 4096 the deriver's structured output truncates and the
# extraction call yields no conclusions.
export HONCHO_LLM_MAX_OUTPUT_TOKENS="${HONCHO_LLM_MAX_OUTPUT_TOKENS:-8192}"
# 'low' for the same reason MEMCONFLICT_REASONING_EFFORT is low on the answer
# role. At the model's default effort the deriver's observation call spent the
# whole 8192-token budget on reasoning and returned EMPTY content: measured
# 122s and 129s calls that Honcho logged as "Deriver generated zero
# observations", leaving the user representation empty for the whole run.
export HONCHO_LLM_THINKING_EFFORT="${HONCHO_LLM_THINKING_EFFORT:-low}"

# --- Honcho server (spawn: API + deriver + a per-run database) ---------------
export HONCHO_SERVER_DIR="${HONCHO_SERVER_DIR:-$ROOT/external/honcho}"
export HONCHO_PG_DSN="${HONCHO_PG_DSN:-postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres}"
export HONCHO_PG_CREATE_DB="${HONCHO_PG_CREATE_DB:-1}"
export HONCHO_DERIVER_WORKERS="${HONCHO_DERIVER_WORKERS:-4}"
export HONCHO_DERIVER_FLUSH="${HONCHO_DERIVER_FLUSH:-1}"

# --- Honcho embedder: the local fastembed shim ------------------------------
export HONCHO_EMBEDDER_MODEL="${HONCHO_EMBEDDER_MODEL:-bge-small-en-v1.5}"
export HONCHO_EMBEDDER_DIMS="${HONCHO_EMBEDDER_DIMS:-384}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# HONCHO_EMBEDDER_BASE_URL unset means the adapter and the smoke start the
# in-process shim themselves (HONCHO_EMBED_SHIM, default on). Set the URL to
# point at vllm-embed instead, and set HONCHO_EMBED_SHIM=0 so a missing URL
# fails the run rather than falling back to a per-process embedder.

# --- SDK HTTP timeout -------------------------------------------------------
# 30 is the Hermes plugin's own default (client.py:245). The plugin runs the
# dialectic on a background thread and consumes the answer one turn later, so
# 30 costs it nothing. This adapter calls the dialectic INLINE, per question
# (documented deviation 2 in eval_honcho.py), and one dialectic pass on a
# reasoning model runs well past 30s. At 30 the dialectic layer would return
# empty on every question and the run would measure the timeout, not Honcho.
export HONCHO_TIMEOUT="${HONCHO_TIMEOUT:-300}"

exec "$@"
