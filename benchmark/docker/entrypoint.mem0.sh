#!/usr/bin/env bash
# MemConflict x mem0 run-service entrypoint. This is a SCAFFOLD.
# The host smoke test passed. The full Docker run is still pending.
# This status matches how Hindsight and RetainDB started.
#
# This script follows the same stage shape as the RetainDB entrypoint.
# The stages are generate, score, and summarize. The script runs as one process.
# The mem0 adapter isolates each persona by a unique mem0 user_id.
# All personas share one disposable, per-run vector store.
# So this entrypoint does not shard by persona.
# mem0 is fully self-hosted. It owns an internal LLM, an embedder, and a vector store.
# The internal LLM does single-pass additive fact extraction.
# mem0ai 2.x removed the 0.1.x ADD/UPDATE/DELETE/NONE update phase.
# Two LLM roles exist:
#   * shared answer and judge LLM (MemConflict llm_request) -> OPENAI_* (vllm-gen)
#   * mem0 internal extraction LLM                          -> MEM0_LLM_* (vllm-gen)
# The best-effort ruling sets both roles to the same serving model by default.
# The embedder defaults to the shared vllm-embed (gte-modernbert-base, dim 768).
# This matches the embedding surface of Mnemosyne and Hindsight.
# The vector store is embedded qdrant, a local on-disk path with no server.
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[mem0 $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score/summarize
# invocation. The fairness contract requires byte-for-byte identical config for
# every provider. Set BENCH_PYTHON=python3 because the base image ships python3
# only, with no bare `python` command.
export BENCH_PYTHON=python3
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"

# This is the clock-sync arm (BENCH_CLOCKSYNC=1). It sources libfaketime helpers.
# The path is relative to THIS script, the same way as answer_env.sh above.
# Every function is a no-op unless BENCH_CLOCKSYNC=1. A default run stays unaffected.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"
# Run-contract helpers. These capture the serving envelope, build the manifest
# and run_contract_hash, and account for vLLM tokens.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"
# These are named launch presets (PRESET=<name>). Apply them before every
# `${VAR:-default}` block and before the vector-mode gate below.
# An unset PRESET is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset mem0

STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-mem0}"

# This sets the answer/judge LLM used by MemConflict llm_request.
# Compose supplies OPENAI_BASE_URL and OPENAI_MODEL.
# The SDK needs a non-empty API key even for vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"

# This is the mem0 internal LLM. It defaults to the shared answer LLM
# endpoint and model unless overridden.
# The adapter also falls back to OPENAI_* when MEM0_LLM_* is unset.
# Set these explicitly so the manifest records unambiguous values.
export MEM0_LLM_PROVIDER="${MEM0_LLM_PROVIDER:-openai}"
export MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-${OPENAI_MODEL:-qwen3.5-4b}}"
export MEM0_LLM_BASE_URL="${MEM0_LLM_BASE_URL:-${OPENAI_BASE_URL:-http://vllm-gen:8000/v1}}"
export MEM0_LLM_API_KEY="${MEM0_LLM_API_KEY:-${OPENAI_API_KEY}}"
export MEM0_LLM_TEMPERATURE="${MEM0_LLM_TEMPERATURE:-0.7}"
export MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-2048}"

# This is the mem0 embedder. It uses the shared vllm-embed (OpenAI-compatible)
# server by default. This matches the other providers' gte-modernbert-base,
# dim 768 surface.
export MEM0_EMBEDDER_PROVIDER="${MEM0_EMBEDDER_PROVIDER:-openai}"
export MEM0_EMBEDDER_MODEL="${MEM0_EMBEDDER_MODEL:-gte-modernbert-base}"
export MEM0_EMBEDDER_BASE_URL="${MEM0_EMBEDDER_BASE_URL:-http://vllm-embed:8000/v1}"
export MEM0_EMBEDDER_DIMS="${MEM0_EMBEDDER_DIMS:-768}"
# The OpenAI-compatible embedder client needs an API key even against vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"

# Disable mem0's anonymous PostHog telemetry.
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# --- Vector store: central shared qdrant (default) vs. per-run embedded ------
# This setting is the analog of entrypoint.hindsight.sh's HINDSIGHT_PG_MODE.
# mem0's embedded qdrant locks its on-disk path to ONE process.
# So sharded runs (N processes) must point at a central server.
# MEM0_VECTOR_MODE=server is the compose default. It targets the shared `qdrant`
# service. Each run or shard gets its own collection, named
# mem0_<sanitized RUN_TAG>. This is the qdrant analog of Hindsight's per-run
# database inside the shared postmaster. Shards never collide, and personas
# stay isolated by mem0 user_id.
# MEM0_VECTOR_MODE=embedded restores the per-process on-disk store.
# This matches host-smoke behavior and does not work across shards.
MEM0_VECTOR_MODE="${MEM0_VECTOR_MODE:-server}"
case "$MEM0_VECTOR_MODE" in
  server)
    export MEM0_QDRANT_HOST="${MEM0_QDRANT_HOST:-qdrant}"
    export MEM0_QDRANT_PORT="${MEM0_QDRANT_PORT:-6333}"
    # This derives the per-run or per-shard collection name from RUN_TAG.
    # Sharded runs pass RUN_TAG=<tag>_s<k>, so each shard gets its own collection.
    # An explicit MEM0_COLLECTION value always wins.
    if [ -z "${MEM0_COLLECTION:-}" ]; then
      _san="$(printf '%s' "${RUN_TAG:-default}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_]/_/g')"
      export MEM0_COLLECTION="mem0_${_san}"
      unset _san
    fi
    # Drop any inherited embedded path. This makes the adapter select server mode cleanly.
    unset MEM0_VECTOR_STORE_PATH 2>/dev/null || true
    log "vector store: shared qdrant ${MEM0_QDRANT_HOST}:${MEM0_QDRANT_PORT} collection=${MEM0_COLLECTION}"
    ;;
  embedded)
    # Unset the server knobs. Compose sets MEM0_QDRANT_HOST by default.
    # This makes the adapter fall back to a per-process embedded path.
    unset MEM0_QDRANT_HOST MEM0_QDRANT_PORT MEM0_QDRANT_URL 2>/dev/null || true
    log "vector store: per-run EMBEDDED qdrant (NOT shareable across shards)"
    ;;
  *)
    log "unknown MEM0_VECTOR_MODE=$MEM0_VECTOR_MODE (expected server|embedded)"; exit 2 ;;
esac

RESDIR="$ROOT/mem0/Results"
SCOREDIR="$ROOT/mem0/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/mem0_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]               && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ]  && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")
# RETAIN_GRANULARITY (batch|session|exchange) selects the ingestion arm.
# If omitted, the adapter defaults to 'batch', the MemConflict authors' 8-message cadence.
[ -n "${RETAIN_GRANULARITY:-}" ]         && CAPS+=(--retain_granularity "$RETAIN_GRANULARITY")
[ -n "${MEM0_ADD_BATCH_SIZE:-}" ]        && CAPS+=(--mem0_add_batch_size "$MEM0_ADD_BATCH_SIZE")
# This overrides the dataset path, for example to benchmark/probes/.
# The default dataset is Step4_4.
[ -n "${INPUT_JSONL:-}" ]                && CAPS+=(--input_jsonl_path "$INPUT_JSONL")

do_generate() {
  bench_answer_env
  # This sets the persona shard range. The sharded launcher supplies
  # START_IDX/END_IDX. Otherwise the range is the whole [0, NUM_PERSONAS).
  # Each shard writes its own RESULTS_FILE, keyed off RUN_TAG=<tag>_s<k>.
  # The files merge at the JSONL level before scoring, the same way as the
  # Hindsight sharded flow.
  local S="${START_IDX:-0}" E="${END_IDX:-$TOTAL}"
  # This resets to a fresh collection for the GENERATE stage in server mode.
  # In embedded mode this is a no-op, because mem0 wipes the store on init.
  # Set ALLOW_EXISTING_COLLECTION=1 to skip the reset and append or resume instead.
  local RESET=()
  [ "${ALLOW_EXISTING_COLLECTION:-0}" = "1" ] || RESET=(--reset_collection)
  # This captures the serving envelope, builds the manifest (run-contract hash),
  # and starts token accounting.
  # Run this OUTSIDE the libfaketime subshell below, so its provenance timestamps
  # stay real wall-clock time.
  # This step aborts under STRICT_RUN_CONTRACT=1 or BENCH_CLOCKSYNC=1 if the run
  # contract is incomplete.
  bench_generate_preamble "$ROOT/mem0" "$TAG"
  log "GENERATE personas=[$S,$E) top_k=$TOPK search_top_k=${MEM0_SEARCH_TOP_K:-20} search_threshold=${MEM0_SEARCH_THRESHOLD:-0.0} vector_mode=${MEM0_VECTOR_MODE} collection=${MEM0_COLLECTION:-embedded} internal_llm=${MEM0_LLM_MODEL} embed=${MEM0_EMBEDDER_MODEL} thinking=${MEMCONFLICT_ENABLE_THINKING} answer_max_tokens=${OPENAI_MAX_TOKENS} clocksync=${BENCH_CLOCKSYNC:-0} -> $RESULTS_FILE"
  # This is the clock-sync arm (BENCH_CLOCKSYNC=1). It runs the mem0 python
  # process under libfaketime, so its OS clock steps to each session's logical
  # date. benchmark/clock_sync.py drives this per session through the shared
  # driver.
  # Under mem0ai 2.x, libfaketime alone covers both the stored created_at value
  # and the extraction prompt's Observation/Current Date.
  # mem0 computes that date per add() call from the process clock, in
  # prompts.py _resolve_dates.
  # The 0.1.118 import-frozen-prompt monkeypatch no longer exists in this version.
  # Each shard runs as its own container through run_shards.sh, so one
  # container-local timestamp file is enough.
  # This fake clock applies only inside THIS subshell. The manifest step above
  # and score/summarize (STAGE=all) stay on the real clock.
  # All three clock-sync calls are no-ops unless BENCH_CLOCKSYNC=1.
  # So a default run execs the identical python process with no fake clock.
  (
    bench_clocksync_prepare "/tmp/clocksync/faketime.rc"
    bench_clocksync_probe
    bench_clocksync_preload
    exec python3 -u "$ROOT/mem0/eval_mem0.py" \
        --start_idx "$S" --end_idx "$E" --top_k "$TOPK" "${CAPS[@]}" "${RESET[@]}" \
        --output_jsonl_path "$RESULTS_FILE" \
        --output_json_path "$RESULTS_JSON"
  )
  # Close the token-accounting window. This shell's clock was never faked.
  bench_tokens_finish "$ROOT/mem0" "$TAG"
}

run_stage "$ROOT/mem0" "$TAG" do_generate
