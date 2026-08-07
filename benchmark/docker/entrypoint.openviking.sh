#!/usr/bin/env bash
# MemConflict x OpenViking run-service entrypoint. This is a SCAFFOLD (the
# adapter, openviking/eval_openviking.py, is under active development).
#
# This script follows the same stage shape as the other entrypoints:
# generate, score, summarize, one process. The OpenViking adapter isolates
# each persona by the X-OpenViking-User header, so this entrypoint does not
# shard by persona; START_IDX/END_IDX split the dataset across containers
# instead (the mem0/honcho model).
#
# OpenViking is a server product with self-contained storage: one pip
# distribution ships the `openviking-server` console script and its local
# content store plus vector index. Two backends exist:
#   spawn (default) -- openviking/_openviking_server.py writes ov.conf and
#     starts the server as a child process under a libfaketime LD_PRELOAD it
#     controls (supermemory's and honcho's spawn-mode pattern).
#   shared -- attach to an already-running server at OPENVIKING_ENDPOINT
#     (health check only). There is no compose service for it; a workspace
#     holds a one-process `.openviking.pid` lock, so this is an operator-run
#     server, not a stack service.
#
# Two LLM roles exist:
#   * shared answer and judge LLM (MemConflict llm_request) -> OPENAI_* (vllm-gen)
#   * OpenViking-internal chat model (`vlm` in ov.conf: memory extraction and
#     search intent analysis)                               -> OPENVIKING_LLM_* (vllm-gen)
# The best-effort ruling sets both roles to the same serving model by default.
# The embedder defaults to the shared vllm-embed (gte-modernbert-base, dim 768).
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[openviking $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score/summarize
# invocation. The fairness contract requires byte-for-byte identical config for
# every provider. Set BENCH_PYTHON=python3 because the base image ships python3
# only, with no bare `python` command.
export BENCH_PYTHON=python3
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"

# This is the clock-sync arm (BENCH_CLOCKSYNC=1). It sources libfaketime
# helpers. The path is relative to THIS script, the same way as answer_env.sh
# above. Only the spawned server child gets LD_PRELOAD, injected by the
# adapter's own child-env builder (openviking/_openviking_server.py), mirroring
# _honcho_server.py. This shell never sets LD_PRELOAD itself, so its own
# timeouts and the harness python keep real time.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"
# Run-contract helpers. These capture the serving envelope, build the manifest
# and run_contract_hash, and account for vLLM tokens.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"
# These are named launch presets (PRESET=<name>). Apply them before every
# `${VAR:-default}` block and before the server-mode gate below. An unset
# PRESET is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset openviking

# openviking/eval_openviking.py's _env(name, default) returns the default when
# the variable is unset OR empty, but _openviking_server.py int()/float()
# parses its own knobs (OPENVIKING_LLM_MAX_CONCURRENT, OPENVIKING_DRAIN_TIMEOUT_S,
# ...). docker-compose's `environment:` block always SETS every key it lists,
# so an unconfigured `${OPENVIKING_X:-}` entry reaches the container as a real,
# present, EMPTY string, not an absence -- the same trap CLAUDE.md documents
# for HINDSIGHT_API_*. This unsets any empty OPENVIKING_* variable before the
# defaulting block below runs, so a genuinely unset knob falls back to this
# script's own default instead of an empty string reaching the adapter.
unset_empty_env_with_prefix OPENVIKING_

STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-openviking}"

# This sets the answer/judge LLM used by MemConflict llm_request.
# Compose supplies OPENAI_BASE_URL and OPENAI_MODEL.
# The SDK needs a non-empty API key even for vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"

# --- OpenViking-internal chat model (`vlm` in ov.conf) ----------------------
# Extraction at POST /sessions/{sid}/commit and the search-intent analysis of
# POST /search/search both run on this model. It defaults to the shared answer
# LLM endpoint and model. Set explicitly so the manifest records unambiguous
# values (mem0 and honcho precedent).
export OPENVIKING_LLM_MODEL="${OPENVIKING_LLM_MODEL:-${OPENAI_MODEL:-qwen3.5-4b}}"
export OPENVIKING_LLM_BASE_URL="${OPENVIKING_LLM_BASE_URL:-${OPENAI_BASE_URL:-http://vllm-gen:8000/v1}}"
export OPENVIKING_LLM_API_KEY="${OPENVIKING_LLM_API_KEY:-${OPENAI_API_KEY}}"
export OPENVIKING_LLM_MAX_TOKENS="${OPENVIKING_LLM_MAX_TOKENS:-4096}"
# 0.0 is the vendor's own sample-config value for `vlm.temperature`.
export OPENVIKING_LLM_TEMPERATURE="${OPENVIKING_LLM_TEMPERATURE:-0.0}"
# `vlm.max_concurrent` ships at 64. One shard at 64 in-flight extraction calls
# starves the shared vllm-gen that also serves the answer role; 8 keeps the
# extraction load per shard bounded.
export OPENVIKING_LLM_MAX_CONCURRENT="${OPENVIKING_LLM_MAX_CONCURRENT:-8}"
export OPENVIKING_LLM_TIMEOUT="${OPENVIKING_LLM_TIMEOUT:-600}"
# JSON merged into vlm.extra_request_body; empty means the key is omitted.
# Needed only for reasoning models (docs/TROUBLESHOOTING.md "Provider:
# OpenViking"); local qwen3.5-4b leaves it empty.
export OPENVIKING_LLM_EXTRA_BODY="${OPENVIKING_LLM_EXTRA_BODY:-}"

# --- OpenViking embedder (`embedding.dense` in ov.conf) ---------------------
# The shared vllm-embed, the same retrieval-embedding surface as every other
# provider. _openviking_server.py raises at start() when the base URL is empty
# in spawn mode: an unreachable embedder surfaces only as a nonzero error_count
# inside POST /system/wait, and nowhere else in the run.
# Changing model or dimension invalidates an existing workspace, so a contract
# v4 workspace (384 dims) cannot be reused under v5.
export OPENVIKING_EMBEDDER_MODEL="${OPENVIKING_EMBEDDER_MODEL:-gte-modernbert-base}"
export OPENVIKING_EMBEDDER_BASE_URL="${OPENVIKING_EMBEDDER_BASE_URL:-http://vllm-embed:8000/v1}"
export OPENVIKING_EMBEDDER_API_KEY="${OPENVIKING_EMBEDDER_API_KEY:-local-vllm}"
export OPENVIKING_EMBEDDER_DIMS="${OPENVIKING_EMBEDDER_DIMS:-768}"
# token_usage.py resolves the embed /metrics URL from a fixed list of provider
# env names that predates openviking (token_usage.py:110-115). The documented
# BENCH_TOKENS_EMBED_URL override carries ours; without it the sidecar records
# "no metrics URL resolved" for vllm_embed.
export BENCH_TOKENS_EMBED_URL="${BENCH_TOKENS_EMBED_URL:-$OPENVIKING_EMBEDDER_BASE_URL}"

# --- Adapter-facing knobs (identity, recall arm, ingest cadence) ------------
# Dev auth mode on loopback: identity comes from the X-OpenViking-Account /
# X-OpenViking-User headers and no key is needed. An empty API key selects it.
export OPENVIKING_API_KEY="${OPENVIKING_API_KEY:-}"
export OPENVIKING_ACCOUNT="${OPENVIKING_ACCOUNT:-default}"
export OPENVIKING_AGENT="${OPENVIKING_AGENT:-hermes}"
# prefetch (featured, plugin-faithful: session-start profile/preferences/
# entities block plus the /search/search entries) | find (minimal,
# diagnostic: deterministic /search/find, no internal LLM call) | search
# (/search/search entries alone, an unplanned auxiliary reduction). Every
# mode passes the plugin's recall_limit selection whole; no top-K slice.
export OPENVIKING_RECALL_MODE="${OPENVIKING_RECALL_MODE:-prefetch}"
# Plugin recall knobs, at the plugin's own defaults (plugins/memory/openviking).
export OPENVIKING_RECALL_LIMIT="${OPENVIKING_RECALL_LIMIT:-6}"
export OPENVIKING_RECALL_SCORE_THRESHOLD="${OPENVIKING_RECALL_SCORE_THRESHOLD:-0.15}"
export OPENVIKING_RECALL_MAX_INJECTED_CHARS="${OPENVIKING_RECALL_MAX_INJECTED_CHARS:-4000}"
export OPENVIKING_PROFILE_TOKEN_BUDGET="${OPENVIKING_PROFILE_TOKEN_BUDGET:-6000}"
export OPENVIKING_RECALL_FULL_READ_LIMIT="${OPENVIKING_RECALL_FULL_READ_LIMIT:-2}"
export OPENVIKING_RECALL_PREFER_ABSTRACT="${OPENVIKING_RECALL_PREFER_ABSTRACT:-0}"
export OPENVIKING_RECALL_RESOURCES="${OPENVIKING_RECALL_RESOURCES:-0}"
# DEVIATION from the plugin's 4.0/3.0 second budget, recorded in
# docs/DECISIONS.md. The plugin joins prefetch on a background thread and drops
# whatever has not arrived; the adapter calls recall INLINE, so the plugin
# budget empties recall silently under benchmark serving latency. 60 is the
# plugin's own clamp maximum (honcho HONCHO_TIMEOUT=300 precedent).
export OPENVIKING_RECALL_TIMEOUT_SECONDS="${OPENVIKING_RECALL_TIMEOUT_SECONDS:-60}"
export OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS="${OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS:-30}"
# Vendor-exposed capability the plugin itself does not use: the plugin sends no
# created_at, so the plugin-faithful temporal path is BENCH_CLOCKSYNC.
export OPENVIKING_SEND_CREATED_AT="${OPENVIKING_SEND_CREATED_AT:-0}"
# Deviation from the plugin's fire-and-forget commit (honcho and supermemory
# precedent): poll the commit task, then POST /system/wait, so recall never
# runs against a persona whose memory state is unknown.
export OPENVIKING_DRAIN_TIMEOUT_S="${OPENVIKING_DRAIN_TIMEOUT_S:-1800}"
export OPENVIKING_DRAIN_POLL_S="${OPENVIKING_DRAIN_POLL_S:-1.0}"
# httpx per-request timeout for the ingest and drain calls. It bounds a commit
# POST, which extraction makes long. The recall path has its own two budgets
# above.
export OPENVIKING_HTTP_TIMEOUT="${OPENVIKING_HTTP_TIMEOUT:-600}"
# Spawn mode only. Empty -> the adapter's own defaults: OPENVIKING_RUN_DIR
# falls back to .openviking_runs/<sanitized RUN_TAG> next to the adapter,
# OPENVIKING_WORKSPACE to <run dir>/data, OPENVIKING_SERVER_BIN to the console
# script next to sys.executable.
export OPENVIKING_RUN_DIR="${OPENVIKING_RUN_DIR:-}"
export OPENVIKING_WORKSPACE="${OPENVIKING_WORKSPACE:-}"
export OPENVIKING_SERVER_BIN="${OPENVIKING_SERVER_BIN:-}"
# 0 asks the OS for an ephemeral port, so co-tenant shards never collide.
export OPENVIKING_SERVER_PORT="${OPENVIKING_SERVER_PORT:-0}"

# Ingest cadence: exchange (default, the plugin's sync_turn cadence -- one
# messages/batch POST per user+assistant exchange) | session (one POST per
# session, chunked at the server's 100-message batch cap). Both spellings are
# resolved to ONE value here, in bash, so the two manifest keys cannot
# disagree and this script's log matches what the adapter runs.
RETAIN_GRANULARITY="${OPENVIKING_RETAIN_GRANULARITY:-${RETAIN_GRANULARITY:-exchange}}"
export RETAIN_GRANULARITY
export OPENVIKING_RETAIN_GRANULARITY="$RETAIN_GRANULARITY"

# Per-run user prefix: every persona's OpenViking user id is
# "${OPENVIKING_USER_PREFIX}<persona tag>". An explicit override always wins.
# Derived here, in bash, the same way entrypoint.honcho.sh derives
# HONCHO_WORKSPACE_PREFIX, so an attached shared server keeps two waves apart.
if [ -z "${OPENVIKING_USER_PREFIX:-}" ]; then
  _san="$(printf '%s' "${RUN_TAG:-run}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-zA-Z0-9_]/_/g')"
  export OPENVIKING_USER_PREFIX="${_san}_"
  unset _san
fi

# --- Server backend selection (spawned vs. attached) ------------------------
export OPENVIKING_SERVER_MODE="${OPENVIKING_SERVER_MODE:-spawn}"
case "$OPENVIKING_SERVER_MODE" in
  spawn)
    # Unset any inherited endpoint, so the adapter's attach-vs-spawn branch
    # selects spawn cleanly even if OPENVIKING_ENDPOINT leaked in from a
    # compose default.
    unset OPENVIKING_ENDPOINT 2>/dev/null || true
    log "storage=spawn (per-run server + local workspace) port=${OPENVIKING_SERVER_PORT} workspace=${OPENVIKING_WORKSPACE:-<adapter default>}"
    ;;
  shared)
    # One attached server has one perceived clock, but each shard drives a
    # different persona/session timeline, so it cannot sit at N logical session
    # dates at once (the same guard as entrypoint.honcho.sh and
    # entrypoint.supermemory.sh).
    if bench_clocksync_enabled; then
      log "FATAL: BENCH_CLOCKSYNC=1 requires OPENVIKING_SERVER_MODE=spawn."
      log "       An attached server has a single perceived clock and cannot"
      log "       serve N shards at different logical session dates. Use spawn"
      log "       mode, which run_shards.sh sets for every openviking wave."
      exit 2
    fi
    case "$STAGE" in
      generate|all)
        if [ -z "${OPENVIKING_ENDPOINT:-}" ]; then
          log "FATAL: OPENVIKING_SERVER_MODE=shared needs OPENVIKING_ENDPOINT"
          exit 2
        fi
        log "storage=shared server=$OPENVIKING_ENDPOINT user_prefix=$OPENVIKING_USER_PREFIX"
        ;;
      *)
        log "storage=shared (STAGE=$STAGE: no server attach needed -- use --no-deps to skip the vLLM deps)"
        ;;
    esac
    ;;
  *)
    log "unknown OPENVIKING_SERVER_MODE=$OPENVIKING_SERVER_MODE (expected spawn|shared)"; exit 2
    ;;
esac

RESDIR="$ROOT/openviking/Results"
SCOREDIR="$ROOT/openviking/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/openviking_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]              && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")
CAPS+=(--retain_granularity "$RETAIN_GRANULARITY")
# Env fallback (mem0-style): the adapter reads OPENVIKING_RECALL_MODE directly
# if the flag is ever omitted, so passing the exported value here is a no-op
# when it already equals the adapter's own default.
[ -n "${OPENVIKING_RECALL_MODE:-}" ]    && CAPS+=(--recall_mode "$OPENVIKING_RECALL_MODE")
# This overrides the dataset path, for example to benchmark/probes/.
# The default dataset is Step4_4.
[ -n "${INPUT_JSONL:-}" ]               && CAPS+=(--input_jsonl_path "$INPUT_JSONL")

# This sets the persona range. Personas run serially in one process.
# START_IDX/END_IDX let several containers split the dataset, each with its
# own RUN_TAG and results file (the mem0/honcho sharding model).
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-$TOTAL}"

do_generate() {
  bench_answer_env  # canonical ANSWER decoding (identical across providers)
  # This is the clock-sync arm. It seeds the per-shard timestamp file with real
  # time and verifies that the preload bends perceived time in THIS image,
  # before the adapter spawns its server child. LD_PRELOAD is NOT set in this
  # shell; only the adapter's own child-env builder injects it (spawn mode
  # only). No-op unless BENCH_CLOCKSYNC=1.
  bench_clocksync_prepare /tmp/clocksync/faketime.rc
  bench_clocksync_probe
  # This captures the serving envelope, builds the manifest (run-contract hash),
  # and starts token accounting. Aborts under STRICT_RUN_CONTRACT=1 or
  # BENCH_CLOCKSYNC=1 if the run contract is incomplete. Runs OUTSIDE any
  # faked-clock scope, so provenance timestamps stay real wall-clock time.
  bench_generate_preamble "$ROOT/openviking" "$TAG"
  log "GENERATE personas=[$START_IDX,$END_IDX) top_k=$TOPK recall_mode=$OPENVIKING_RECALL_MODE retain_granularity=$RETAIN_GRANULARITY server_mode=$OPENVIKING_SERVER_MODE internal_llm=$OPENVIKING_LLM_MODEL embed=$OPENVIKING_EMBEDDER_MODEL send_created_at=$OPENVIKING_SEND_CREATED_AT answer_max_tokens=${OPENAI_MAX_TOKENS} clocksync=${BENCH_CLOCKSYNC:-0} -> $RESULTS_FILE"
  python3 -u "$ROOT/openviking/eval_openviking.py" \
      --start_idx "$START_IDX" --end_idx "$END_IDX" --top_k "$TOPK" "${CAPS[@]}" \
      --output_jsonl_path "$RESULTS_FILE" \
      --output_json_path "$RESULTS_JSON"
  bench_tokens_finish "$ROOT/openviking" "$TAG"
}

run_stage "$ROOT/openviking" "$TAG" do_generate
