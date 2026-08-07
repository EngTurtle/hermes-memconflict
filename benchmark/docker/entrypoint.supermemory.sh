#!/usr/bin/env bash
# MemConflict x Supermemory (self-hosted) run-service entrypoint.
#
# This script follows the same stage shape as the RetainDB entrypoint: generate,
# score, summarize. It also runs as one process.
# The adapter spawns and owns ONE disposable supermemory-server per run.
# It isolates personas by containerTag, so this entrypoint does not shard by
# persona. START_IDX/END_IDX split the dataset across containers instead.
#
# Supermemory has its own internal extraction LLM.
# The shared answer and judge model is still the only FAIRNESS-LOCKED LLM. It
# stays byte-identical across providers through answer_env.sh.
# The extraction LLM is Supermemory's own knob (SUPERMEMORY_LLM_*).
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[supermemory $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score/summarize
# invocation. The fairness contract requires byte-for-byte identical config for
# every provider. Set BENCH_PYTHON=python3 because the node:22 base image ships
# python3 only, with no bare `python` command.
export BENCH_PYTHON=python3
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"
# These are the clock-sync arm helpers (bench_clocksync_prepare/probe).
# They are no-ops unless BENCH_CLOCKSYNC=1.
# Only the spawned server child is preloaded with libfaketime. The adapter's
# _supermemory_server._env injects LD_PRELOAD into the child env.
# This shell never sets LD_PRELOAD itself.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"
# Run-contract helpers. These capture the serving envelope, build the manifest
# and run_contract_hash, and account for vLLM tokens.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"
# These are named launch presets (PRESET=<name>). Apply them before the
# server-mode selection below. This order lets a preset that sets
# SUPERMEMORY_SERVER_MODE=spawn plus BENCH_CLOCKSYNC=1 still reach the
# shared-mode clocksync guard. An unset PRESET is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset supermemory

STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-supermemory}"

# This sets the shared answer/judge LLM used by MemConflict llm_request.
# Compose supplies OPENAI_BASE_URL and OPENAI_MODEL.
# The SDK needs a non-empty API key even for a local vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"

# This is Supermemory's internal extraction LLM, on the server side.
# It defaults to the shared answer endpoint and model, so a single config works
# out of the box.
# Override the SUPERMEMORY_LLM_* trio to drive extraction with a different
# model, for example a local one.
export SUPERMEMORY_LLM_API_KEY="${SUPERMEMORY_LLM_API_KEY:-$OPENAI_API_KEY}"
export SUPERMEMORY_LLM_BASE_URL="${SUPERMEMORY_LLM_BASE_URL:-${OPENAI_BASE_URL:-}}"
export SUPERMEMORY_LLM_MODEL="${SUPERMEMORY_LLM_MODEL:-${OPENAI_MODEL:-}}"

# --- Server backend selection (shared central server vs. per-run spawned) ---
# SUPERMEMORY_SERVER_MODE=shared attaches to the central `supermemory-server`
# compose service, the shared "central DB". This is the analog of Hindsight's
# shared hindsight-pg.
# Many shard containers point at ONE server and isolate by containerTag
# namespace. This lets a full 30-persona run shard across containers without
# each container booting its own server. A shared embedded data dir cannot
# safely serve more than one server anyway.
# SUPERMEMORY_SERVER_MODE=spawn is the default. The adapter spawns its own
# disposable server per run, the standalone single-process path.
export PATH="/root/.local/bin:$PATH"
export SUPERMEMORY_SERVER_MODE="${SUPERMEMORY_SERVER_MODE:-spawn}"
case "$SUPERMEMORY_SERVER_MODE" in
  shared)
    # Clock-sync is incompatible with a shared central server.
    # One server process has ONE perceived clock, but each shard drives a
    # different persona/session timeline. A single shared server cannot sit at
    # N timeline points at once.
    # Clock-sync therefore requires spawn mode: one server per shard, with a
    # per-shard timestamp file.
    # run_shards.sh already forces spawn mode for this arm. This guard also
    # covers the direct-invocation path.
    if bench_clocksync_enabled; then
      log "FATAL: BENCH_CLOCKSYNC=1 requires SUPERMEMORY_SERVER_MODE=spawn."
      log "       A shared central server has a single perceived clock and cannot"
      log "       serve N shards at different logical session dates. Use spawn mode"
      log "       (per-shard disposable server), which run_shards.sh forces for the arm."
      exit 2
    fi
    # Only generate/all actually attach to the central server, because they
    # ingest and recall. score/summarize never touch it, so this skips the
    # attach setup for them entirely. They need no central server running.
    # Run those stages with `--no-deps`.
    case "${STAGE:-all}" in
      generate|all)
        # This derives the per-run containerTag namespace from RUN_TAG.
        # Shards of a run share ONE prefix on the shared store, so different
        # runs never collide.
        # Strip a trailing range-shard suffix (_s<k>) or per-persona suffix
        # (_p<i>). run_shards.sh PERSONA_CONTAINERS mode passes
        # RUN_TAG=<tag>_p<i>, so all containers of RUN_TAG=full_s0..N or
        # full_p0..29 land under namespace "full".
        # Without the _p strip, each persona container would get namespace
        # <tag>_p<i>. The adapter (eval_supermemory.py begin_persona) would then
        # mint containerTags <tag>_p<i>_p<personaID>. That stays collision-free,
        # but the doubled suffix breaks the documented <ns>_p<persona>
        # reclaim-by-prefix convention.
        # A RUN_TAG is required. On a persistent shared store, an empty RUN_TAG
        # would namespace to a fixed default. A re-run would then silently
        # re-ingest the same personas into the same tags, duplicating memories
        # and perturbing recall. Fail loudly instead. An explicit override wins.
        if [ -z "${SUPERMEMORY_CONTAINER_NAMESPACE:-}" ]; then
          _ns="$(printf '%s' "${RUN_TAG:-}" | sed -E 's/_(s|p)[0-9]+$//')"
          if [ -z "$_ns" ]; then
            log "FATAL: shared mode needs an explicit RUN_TAG. It becomes the per-run"
            log "       containerTag namespace on the PERSISTENT shared store; without it a"
            log "       re-run silently re-ingests into another run's tags. Set -e RUN_TAG=<name>"
            log "       (or -e SUPERMEMORY_CONTAINER_NAMESPACE=<ns>), and use a FRESH tag per run."
            exit 1
          fi
          export SUPERMEMORY_CONTAINER_NAMESPACE="$_ns"
        fi
        # This is an optional two-server topology. A shard attaches to either
        # central server, selected by SUPERMEMORY_ATTACH_URL and
        # SUPERMEMORY_KEY_FILE.
        # run_shards.sh sets these per shard when SUPERMEMORY_TWO_SERVERS=1.
        # Server B gets http://supermemory-server-b:8788 and /shared_b/api_key.
        # Both fall back to the single-server values below, so a one-server run
        # stays byte-unchanged.
        _sm_host="${SUPERMEMORY_SERVER_HOST:-supermemory-server}"
        _sm_port="${SUPERMEMORY_SERVER_PORT:-8787}"
        _sm_attach="${SUPERMEMORY_ATTACH_URL:-http://${_sm_host}:${_sm_port}}"
        export SUPERMEMORY_BASE_URL="${SUPERMEMORY_BASE_URL:-$_sm_attach}"
        # For the bearer key, prefer an explicit env value. Otherwise read the
        # file the central server published on the shared volume.
        # depends_on:service_healthy guarantees the file exists before a shard
        # starts. A short bounded retry covers a residual race.
        # SUPERMEMORY_KEY_FILE overrides the path. Server B publishes to
        # /shared_b.
        _sm_keyfile="${SUPERMEMORY_KEY_FILE:-${SUPERMEMORY_SHARED_DIR:-/shared}/api_key}"
        if [ -z "${SUPERMEMORY_API_KEY:-}" ]; then
          for _try in 1 2 3 4 5 6 7 8 9 10; do
            if [ -s "$_sm_keyfile" ]; then
              SUPERMEMORY_API_KEY="$(tr -d '[:space:]' < "$_sm_keyfile")"; break
            fi
            log "waiting for central server key at $_sm_keyfile (try $_try)"; sleep 3
          done
        fi
        export SUPERMEMORY_API_KEY
        [ -n "${SUPERMEMORY_API_KEY:-}" ] || { log "FATAL: no SUPERMEMORY_API_KEY (env or $_sm_keyfile) for shared mode"; exit 1; }
        # Do NOT set SUPERMEMORY_SERVER_CMD. Attach mode (base_url) skips spawning.
        unset SUPERMEMORY_SERVER_CMD 2>/dev/null || true
        log "storage=shared server=$SUPERMEMORY_BASE_URL namespace=$SUPERMEMORY_CONTAINER_NAMESPACE key=${SUPERMEMORY_API_KEY:0:6}..."
        ;;
      *)
        log "storage=shared (STAGE=$STAGE: no central-server attach needed — use --no-deps to skip it)"
        ;;
    esac
    ;;
  spawn)
    # The vendor wrapper reads PORT, SUPERMEMORY_DATA_DIR, OPENAI_*, and
    # SUPERMEMORY_EMBEDDING_* from the child env that _supermemory_server.py sets.
    export SUPERMEMORY_SERVER_CMD="${SUPERMEMORY_SERVER_CMD:-supermemory-server}"
    log "storage=spawn (per-run disposable server)"
    ;;
  *)
    log "unknown SUPERMEMORY_SERVER_MODE=$SUPERMEMORY_SERVER_MODE (expected shared|spawn)"; exit 2
    ;;
esac
export SUPERMEMORY_EMBEDDING_PROVIDER="${SUPERMEMORY_EMBEDDING_PROVIDER:-local}"
# Local embedding weights come over the classic HF CDN, with Xet disabled.
# This matches the other providers.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/app/.hf_cache}"
mkdir -p "$HF_HOME"

RESDIR="$ROOT/supermemory/Results"
SCOREDIR="$ROOT/supermemory/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/supermemory_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]              && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")
[ -n "${SUPERMEMORY_RETAIN_GRANULARITY:-}" ] && CAPS+=(--retain_granularity "$SUPERMEMORY_RETAIN_GRANULARITY")
# This is the FEATURED plugin-faithful path. The default minimal path uses
# documents plus search.
#   SUPERMEMORY_INGEST_ENDPOINT=conversations -> POST /v4/conversations (plugin ingest)
#   SUPERMEMORY_RECALL_ENDPOINT=profile        -> POST /v4/profile (plugin auto-recall)
[ -n "${SUPERMEMORY_INGEST_ENDPOINT:-}" ] && CAPS+=(--ingest_endpoint "$SUPERMEMORY_INGEST_ENDPOINT")
[ -n "${SUPERMEMORY_RECALL_ENDPOINT:-}" ] && CAPS+=(--recall_endpoint "$SUPERMEMORY_RECALL_ENDPOINT")
[ -n "${SUPERMEMORY_SEARCH_MODE:-}" ]  && CAPS+=(--search_mode "$SUPERMEMORY_SEARCH_MODE")
[ -n "${SUPERMEMORY_SEARCH_THRESHOLD:-}" ] && CAPS+=(--search_threshold "$SUPERMEMORY_SEARCH_THRESHOLD")
[ -n "${SUPERMEMORY_EMBEDDING_MODEL:-}" ] && CAPS+=(--embedding_model "$SUPERMEMORY_EMBEDDING_MODEL")
[ -n "${SUPERMEMORY_EMBEDDING_DIMENSIONS:-}" ] && CAPS+=(--embedding_dimensions "$SUPERMEMORY_EMBEDDING_DIMENSIONS")
[ -n "${SUPERMEMORY_LLM_MODEL:-}" ]    && CAPS+=(--llm_model "$SUPERMEMORY_LLM_MODEL")
[ "${SUPERMEMORY_DOCUMENTS_ARM:-0}" = "1" ] && CAPS+=(--documents_arm)
[ "${SUPERMEMORY_RERANK:-0}" = "1" ]   && CAPS+=(--rerank)
[ "${SUPERMEMORY_REWRITE_QUERY:-0}" = "1" ] && CAPS+=(--rewrite_query)
# This overrides the dataset path, for example to benchmark/probes/.
# The default dataset is Step4_4.
[ -n "${INPUT_JSONL:-}" ]               && CAPS+=(--input_jsonl_path "$INPUT_JSONL")

# This sets the persona range. Personas run serially in one process.
# START_IDX/END_IDX let several containers split the dataset, each with its
# own RUN_TAG and results file.
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-$TOTAL}"

do_generate() {
  bench_answer_env  # canonical ANSWER decoding (identical across providers)
  # This is the clock-sync arm. It seeds the per-shard timestamp file with real
  # time and verifies that the preload bends perceived time in THIS image,
  # before the adapter spawns the server.
  # The container-local path is per-shard automatically.
  # LD_PRELOAD is NOT set in this shell. Only the adapter injects it into the
  # spawned server child.
  # This step is a no-op unless BENCH_CLOCKSYNC=1.
  bench_clocksync_prepare /tmp/clocksync/faketime.rc
  bench_clocksync_probe
  # This captures the serving envelope, builds the manifest (run-contract hash),
  # and starts token accounting.
  # This step aborts under STRICT_RUN_CONTRACT=1 or BENCH_CLOCKSYNC=1 if the
  # contract is incomplete.
  # This shell is never libfaketime-preloaded. Only the spawned server child is.
  # So provenance timestamps here are real wall-clock time.
  bench_generate_preamble "$ROOT/supermemory" "$TAG"
  log "GENERATE personas=[$START_IDX,$END_IDX) top_k=$TOPK embed=$SUPERMEMORY_EMBEDDING_PROVIDER search_mode=${SUPERMEMORY_SEARCH_MODE:-hybrid} extraction=${SUPERMEMORY_LLM_MODEL:-default} answer_max_tokens=${OPENAI_MAX_TOKENS} -> $RESULTS_FILE"
  python3 -u "$ROOT/supermemory/eval_supermemory.py" \
      --start_idx "$START_IDX" --end_idx "$END_IDX" --top_k "$TOPK" "${CAPS[@]}" \
      --output_jsonl_path "$RESULTS_FILE" \
      --output_json_path "$RESULTS_JSON"
  bench_tokens_finish "$ROOT/supermemory" "$TAG"
}

run_stage "$ROOT/supermemory" "$TAG" do_generate
