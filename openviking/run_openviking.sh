#!/usr/bin/env bash
# Entrypoint for running the MemConflict benchmark against OpenViking (host path).
#
# This script requires OPENROUTER_API_KEY in the environment. OpenViking is
# fully self-hosted: the pip package ships the server and its storage, and the
# adapter spawns one server process per run. Two LLM roles are involved, and
# BOTH use gpt-oss-20b for this validation smoke:
#   * The shared answer + judge LLM (MemConflict llm_request), via OPENAI_*.
#   * OpenViking's INTERNAL model (memory extraction at commit, and the
#     search/search intent analysis), via OPENVIKING_LLM_*, which
#     _openviking_server.py writes into the `vlm` block of ov.conf.
# Per the best-effort ruling, OpenViking's internal model points at the SAME
# serving model the harness uses to answer. For offline runs, point both roles
# at the local vLLM instead.
#
# The embedder runs LOCALLY. The host default is the fastembed shim in
# honcho/_local_embed_server.py (BAAI/bge-small-en-v1.5, dim 384), started as a
# SUBPROCESS so this provider imports nothing from another provider's folder.
# 384 dims is the v4-era host-smoke surface and is off-contract for v5: use it
# for smokes, not for a banked number. Docker runs point
# OPENVIKING_EMBEDDER_BASE_URL at the shared vllm-embed instead.
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

# --- OpenViking internal LLM (extraction, query planning) --------------------
export OPENVIKING_LLM_MODEL="${OPENVIKING_LLM_MODEL:-openai/gpt-oss-20b}"
export OPENVIKING_LLM_BASE_URL="${OPENVIKING_LLM_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENVIKING_LLM_API_KEY="${OPENVIKING_LLM_API_KEY:-$OPENROUTER_API_KEY}"
# Extraction returns structured tool calls. Below about 4096 the call truncates
# and the commit stores fewer memories without reporting an error. 8192 leaves
# room for gpt-oss-20b's reasoning tokens, which draw from the same budget
# (HONCHO_LLM_MAX_OUTPUT_TOKENS=8192 precedent).
export OPENVIKING_LLM_MAX_TOKENS="${OPENVIKING_LLM_MAX_TOKENS:-8192}"
# gpt-oss-20b at default reasoning effort burns the budget on reasoning
# (empty extraction responses) or outlives OpenRouter's keep-alive window
# (newline-padded body, no JSON). Same fix as HONCHO_LLM_THINKING_EFFORT=low.
if [[ -z "${OPENVIKING_LLM_EXTRA_BODY:-}" ]]; then
  export OPENVIKING_LLM_EXTRA_BODY='{"reasoning": {"effort": "low"}}'
fi
# The vendor default is 64. OpenRouter rate-limits well below that, and a 429
# costs a whole extraction retry.
export OPENVIKING_LLM_MAX_CONCURRENT="${OPENVIKING_LLM_MAX_CONCURRENT:-4}"

# --- HuggingFace hygiene for the fastembed shim ------------------------------
# HF's Xet backend is not reachable through the egress proxy (mem0 precedent).
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# --- OpenViking embedder: the local fastembed shim ---------------------------
EMBED_SHIM_PID=""
cleanup() {
  if [[ -n "$EMBED_SHIM_PID" ]]; then
    kill "$EMBED_SHIM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "${OPENVIKING_EMBEDDER_BASE_URL:-}" ]]; then
  EMBED_PORT="${OPENVIKING_EMBED_SHIM_PORT:-$(python -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')}"
  EMBED_LOG="$ROOT/openviking/.openviking_runs/embed_shim.log"
  mkdir -p "$(dirname "$EMBED_LOG")"
  python honcho/_local_embed_server.py --port "$EMBED_PORT" \
    --served_model bge-small-en-v1.5 >"$EMBED_LOG" 2>&1 &
  EMBED_SHIM_PID=$!

  # The first start downloads the ONNX weights, so the budget is 120s, not the
  # few seconds a warm cache needs.
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${EMBED_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$EMBED_SHIM_PID" 2>/dev/null; then
      echo "ERROR: embedding shim exited; see $EMBED_LOG" >&2
      exit 1
    fi
    sleep 1
  done
  if ! curl -sf "http://127.0.0.1:${EMBED_PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: embedding shim not healthy within 120s; see $EMBED_LOG" >&2
    exit 1
  fi

  export OPENVIKING_EMBEDDER_BASE_URL="http://127.0.0.1:${EMBED_PORT}/v1"
  export OPENVIKING_EMBEDDER_MODEL="${OPENVIKING_EMBEDDER_MODEL:-BAAI/bge-small-en-v1.5}"
  export OPENVIKING_EMBEDDER_DIMS="${OPENVIKING_EMBEDDER_DIMS:-384}"
  export OPENVIKING_EMBEDDER_API_KEY="${OPENVIKING_EMBEDDER_API_KEY:-local}"
  echo "[openviking] embedding shim at $OPENVIKING_EMBEDDER_BASE_URL (pid $EMBED_SHIM_PID)"
fi

# `exec` replaces this shell, which would discard the EXIT trap and leak the
# shim. So the shim owner waits for the command and exits with its status; a
# run that brings its own embedder execs as the other providers' scripts do.
if [[ -n "$EMBED_SHIM_PID" ]]; then
  "$@"
  exit $?
fi
exec "$@"
