#!/usr/bin/env bash
# This is the central Supermemory server for SHARDED runs, the shared "central DB".
#
# It boots ONE long-running supermemory-server. Many `supermemory` shard
# containers attach to it through SUPERMEMORY_SERVER_MODE=shared. This is the
# analog of the shared `hindsight-pg` Postgres service.
# Supermemory's server is already an HTTP service that owns an embedded graph
# engine. So the shards just point at it and isolate by containerTag namespace.
# There is no per-shard server and no DB to externalize.
#
# Persist SUPERMEMORY_DATA_DIR on a named volume, so the generated bearer key
# stays stable across restarts.
# The key is published to SUPERMEMORY_SHARED_DIR/api_key on a second shared
# volume that the shards mount read-only. See _serve_supermemory.py.
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[sm-server $(date -u +%H:%M:%S)] $*"; }

# --- This is the fixed port the shards dial (supermemory-server:PORT) -------
export PORT="${SUPERMEMORY_SERVER_PORT:-${PORT:-8787}}"
export SUPERMEMORY_PORT="$PORT"

# --- This is the persistent embedded store and shared key handoff directory --
export SUPERMEMORY_DATA_DIR="${SUPERMEMORY_DATA_DIR:-/data/supermemory}"
export SUPERMEMORY_SHARED_DIR="${SUPERMEMORY_SHARED_DIR:-/shared}"
mkdir -p "$SUPERMEMORY_DATA_DIR" "$SUPERMEMORY_SHARED_DIR"

# --- This sets the server binary and turns off telemetry --------------------
export PATH="/root/.local/bin:$PATH"
export SUPERMEMORY_SERVER_CMD="${SUPERMEMORY_SERVER_CMD:-supermemory-server}"
export SUPERMEMORY_DISABLE_TELEMETRY="${SUPERMEMORY_DISABLE_TELEMETRY:-1}"

# --- This is the internal extraction LLM, on the server side ----------------
# _supermemory_server.py maps SUPERMEMORY_LLM_* onto the server child's OPENAI_*.
# If the LLM trio is unset, this defaults to the shared answer endpoint and
# model, so a single config works.
# The fairness-locked answer/judge model lives in the SHARD containers, never here.
export SUPERMEMORY_LLM_API_KEY="${SUPERMEMORY_LLM_API_KEY:-${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}}"
export SUPERMEMORY_LLM_BASE_URL="${SUPERMEMORY_LLM_BASE_URL:-${OPENAI_BASE_URL:-}}"
export SUPERMEMORY_LLM_MODEL="${SUPERMEMORY_LLM_MODEL:-${OPENAI_MODEL:-}}"

# --- These are the embeddings. Local ONNX is the default. -------------------
export SUPERMEMORY_EMBEDDING_PROVIDER="${SUPERMEMORY_EMBEDDING_PROVIDER:-local}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-$SUPERMEMORY_DATA_DIR/.hf_cache}"
mkdir -p "$HF_HOME"

# --- This is ingest throughput. A shared server fed by N shards needs more
# than the vendor default of 2. Raise it so shards' submissions extract in
# parallel. The real ceiling is the extraction LLM's throughput. ------------
export SUPERMEMORY_INGEST_CONCURRENCY="${SUPERMEMORY_INGEST_CONCURRENCY:-10}"

log "starting central server on :$PORT data_dir=$SUPERMEMORY_DATA_DIR shared=$SUPERMEMORY_SHARED_DIR extraction=${SUPERMEMORY_LLM_MODEL:-default} embed=$SUPERMEMORY_EMBEDDING_PROVIDER ingest_concurrency=$SUPERMEMORY_INGEST_CONCURRENCY"
exec python3 -u "$ROOT/supermemory/_serve_supermemory.py"
