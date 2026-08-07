#!/usr/bin/env bash
# MemConflict x Honcho run-service entrypoint. This is a SCAFFOLD (the
# adapter, honcho/eval_honcho.py, is under active development).
#
# This script follows the same stage shape as the other entrypoints:
# generate, score, summarize, one process. The Honcho adapter isolates each
# persona by a Honcho WORKSPACE, so this entrypoint does not shard by
# persona; START_IDX/END_IDX split the dataset across containers instead
# (the mem0/supermemory model).
#
# Honcho is a server product: an API process plus a separate deriver worker,
# both backed by Postgres. Two backends exist:
#   shared (default) -- attach to the compose honcho-api/honcho-deriver
#     services over REST (HONCHO_BASE_URL), the honcho-pg analog of
#     Hindsight's shared hindsight-pg.
#   spawn -- the adapter (honcho/_honcho_server.py) launches its own API,
#     deriver, and (under BENCH_CLOCKSYNC=1) in-container Postgres as child
#     processes, all under one libfaketime clock domain it controls
#     (supermemory's spawn-mode pattern). Required under clock-sync: one
#     shared server has one perceived clock and cannot serve N shards at
#     different logical session dates at once.
#
# Two LLM roles exist:
#   * shared answer and judge LLM (MemConflict llm_request) -> OPENAI_* (vllm-gen)
#   * Honcho-internal LLM (deriver/dialectic/summary/dream)  -> HONCHO_LLM_* (vllm-gen)
# The best-effort ruling sets both roles to the same serving model by default.
# The embedder defaults to the shared vllm-embed (gte-modernbert-base, dim 768).
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[honcho $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score/summarize
# invocation. The fairness contract requires byte-for-byte identical config for
# every provider. Set BENCH_PYTHON=python3 because the base image ships python3
# only, with no bare `python` command.
export BENCH_PYTHON=python3
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"

# This is the clock-sync arm (BENCH_CLOCKSYNC=1). It sources libfaketime helpers.
# The path is relative to THIS script, the same way as answer_env.sh above.
# Only the spawned server+deriver+postgres children get LD_PRELOAD, injected by
# the adapter's own child-env builder (honcho/_honcho_server.py), mirroring
# _supermemory_server.py._env(). This shell never sets LD_PRELOAD itself.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"
# Run-contract helpers. These capture the serving envelope, build the manifest
# and run_contract_hash, and account for vLLM tokens.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"
# These are named launch presets (PRESET=<name>). Apply them before every
# `${VAR:-default}` block and before the server-mode gate below. An unset
# PRESET is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset honcho

# honcho-ai's Honcho(...) client and honcho/_honcho_server.py's own
# int()/float() parsing (HONCHO_TIMEOUT, HONCHO_DRAIN_TIMEOUT_S, ...) read
# os.getenv(VAR, str(DEFAULT)), which substitutes DEFAULT only when VAR is
# fully UNSET. docker-compose's `environment:` block always SETS every key it
# lists, so an unconfigured `${HONCHO_X:-}` entry reaches the container as a
# real, present, EMPTY string, not an absence -- the same trap CLAUDE.md
# documents for HINDSIGHT_API_*. This unsets any empty HONCHO_* variable
# before the defaulting block below runs, so a genuinely unset knob falls
# back to this script's own default instead of an empty string reaching the
# adapter.
unset_empty_env_with_prefix HONCHO_

STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-honcho}"

# This sets the answer/judge LLM used by MemConflict llm_request.
# Compose supplies OPENAI_BASE_URL and OPENAI_MODEL.
# The SDK needs a non-empty API key even for vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"

# --- Honcho-internal LLM + embedder (deriver, dialectic, summary, dream) -----
# These default to the shared answer LLM endpoint and model. Set explicitly so
# the manifest records unambiguous values (mem0 precedent).
export HONCHO_LLM_MODEL="${HONCHO_LLM_MODEL:-${OPENAI_MODEL:-qwen3.5-4b}}"
export HONCHO_LLM_BASE_URL="${HONCHO_LLM_BASE_URL:-${OPENAI_BASE_URL:-http://vllm-gen:8000/v1}}"
export HONCHO_LLM_API_KEY="${HONCHO_LLM_API_KEY:-${OPENAI_API_KEY}}"
# Output budget for every internal role. Reasoning-effort passthrough stays
# empty for qwen3.5-4b (a non-reasoning-model default); a reasoning model at
# its own default effort can spend the whole budget reasoning and return
# empty content (observed: gpt-oss-20b's deriver call burned 8192 tokens on
# reasoning alone and emitted zero observations). unset_empty_env_with_prefix
# above strips this back to a true absence when left empty.
export HONCHO_LLM_MAX_OUTPUT_TOKENS="${HONCHO_LLM_MAX_OUTPUT_TOKENS:-8192}"
export HONCHO_LLM_THINKING_EFFORT="${HONCHO_LLM_THINKING_EFFORT:-}"
# DERIVER-ONLY overlay on the budget above, which otherwise feeds every role
# (deriver, summary, five dialectic levels, two dream specialists). The
# deriver is the role that repetition-loops on qwen3.5-4b: smoke hn_smkft_p0
# stored 18 of 79 documents pinned at the old 8192-token cap, mean 41,189
# chars, unique-sentence ratio 0.181, and those rows then overflowed the
# 32768-token window at recall. A real observation has a median length of 241
# chars. presence_penalty 1.5 is the Qwen card value the answer role already
# uses (answer_env.sh); vLLM's get_diff_sampling_param does not allowlist
# presence_penalty, so it reaches an internal call only per request. Empty
# sends no penalty.
export HONCHO_DERIVER_MAX_OUTPUT_TOKENS="${HONCHO_DERIVER_MAX_OUTPUT_TOKENS:-2048}"
export HONCHO_DERIVER_PRESENCE_PENALTY="${HONCHO_DERIVER_PRESENCE_PENALTY:-1.5}"
# The only count-scaling repetition penalty honcho's OpenAI backend forwards.
# presence_penalty saturates after one occurrence and left 5.2% of stored
# documents as repetition loops (smkft3); see honcho/_honcho_server.py.
export HONCHO_DERIVER_FREQUENCY_PENALTY="${HONCHO_DERIVER_FREQUENCY_PENALTY:-0.3}"
# Token budget for the assembled injection block (the plugin's own
# `contextTokens`). The plugin ships it UNSET = uncapped, which does not fit a
# 32768-token window: an uncapped hybrid block measured 254k tokens at persona
# 0 session 5. 8192 is half the prompt budget left after the shared
# 16384-token answer allowance. 0 restores the shipped uncapped behavior.
export HONCHO_CONTEXT_TOKENS="${HONCHO_CONTEXT_TOKENS:-8192}"
export HONCHO_EMBEDDER_MODEL="${HONCHO_EMBEDDER_MODEL:-gte-modernbert-base}"
export HONCHO_EMBEDDER_BASE_URL="${HONCHO_EMBEDDER_BASE_URL:-http://vllm-embed:8000/v1}"
export HONCHO_EMBEDDER_API_KEY="${HONCHO_EMBEDDER_API_KEY:-local-vllm}"
export HONCHO_EMBEDDER_DIMS="${HONCHO_EMBEDDER_DIMS:-768}"
# never stops the OpenAI client sending `dimensions=`. gte-modernbert-base has
# one fixed output width, so never is simply correct: vLLM returns 400 for a
# `dimensions=` request against this model.
export HONCHO_EMBEDDER_DIMENSIONS_MODE="${HONCHO_EMBEDDER_DIMENSIONS_MODE:-never}"
# The adapter auto-starts a local fastembed shim only when
# HONCHO_EMBEDDER_BASE_URL is empty (host smokes with no vllm-embed). Docker
# always sets HONCHO_EMBEDDER_BASE_URL above, so this is inert either way;
# set explicitly so the manifest records the intent.
export HONCHO_EMBED_SHIM="${HONCHO_EMBED_SHIM:-0}"
export HONCHO_DERIVER_FLUSH="${HONCHO_DERIVER_FLUSH:-1}"
export HONCHO_DERIVER_WORKERS="${HONCHO_DERIVER_WORKERS:-4}"
# Vendor polling defaults (idle-sleep interval / max interval). Startup
# jitter is lowered to 0.0 unconditionally: it exists so co-started peer
# instances do not poll in lockstep, and a dedicated single deriver has no
# peers to collide with -- that jitter is pure added latency on every drain.
export HONCHO_DERIVER_POLL_S="${HONCHO_DERIVER_POLL_S:-1.0}"
export HONCHO_DERIVER_POLL_MAX_S="${HONCHO_DERIVER_POLL_MAX_S:-2.0}"
export HONCHO_DERIVER_STARTUP_JITTER_S="${HONCHO_DERIVER_STARTUP_JITTER_S:-0.0}"
# Left unset by design (vendor default true): dreams idle-trigger after 60
# minutes, so they do not fire mid-run. See the README knob table for why this
# is not force-set to a literal value here.
export HONCHO_DREAM_ENABLED="${HONCHO_DREAM_ENABLED:-}"
# Featured arm only: manually trigger a dream after each session instead of
# waiting for the idle-based scheduler (dataset sessions are days apart, so
# the real 60-minute idle trigger never fires inside a benchmark run).
export HONCHO_DREAM_AFTER_SESSION="${HONCHO_DREAM_AFTER_SESSION:-0}"

# --- Adapter-facing knobs (isolation, recall arm, dialectic shape) ----------
export HONCHO_API_KEY="${HONCHO_API_KEY:-local}"
# CRITICAL: the adapter calls the dialectic inline over this SDK timeout. The
# plugin's own default (30s) silently empties the dialectic layer of hybrid
# recall on slow serving -- the call just times out and that section drops,
# with no error surfaced to the run. 300 matches the harness's own
# answer-path request timeout.
export HONCHO_TIMEOUT="${HONCHO_TIMEOUT:-300}"
# Spawn mode only. Empty -> the adapter's own defaults: HONCHO_SERVER_DIR
# falls back to its DEFAULT_SERVER_DIR, HONCHO_SERVER_PYTHON to
# <server_dir>/.venv/bin/python if present, HONCHO_RUN_DIR to
# .honcho_runs/<RUN_TAG> next to the adapter, HONCHO_PG_DB to
# honcho_<sanitized RUN_TAG>, HONCHO_PG_DROP_DB to true (drop-and-recreate on
# the HONCHO_PG_CREATE_DB=1 path).
export HONCHO_SERVER_PYTHON="${HONCHO_SERVER_PYTHON:-}"
export HONCHO_RUN_DIR="${HONCHO_RUN_DIR:-}"
export HONCHO_PG_DB="${HONCHO_PG_DB:-}"
export HONCHO_PG_DROP_DB="${HONCHO_PG_DROP_DB:-}"
export HONCHO_DB_SCHEMA="${HONCHO_DB_SCHEMA:-public}"
export HONCHO_USER_PEER_ID="${HONCHO_USER_PEER_ID:-user}"
export HONCHO_AI_PEER_ID="${HONCHO_AI_PEER_ID:-hermes}"
export HONCHO_OBSERVATION_MODE="${HONCHO_OBSERVATION_MODE:-directional}"
# hybrid | base | dialectic | search | conclusions. The minimal preset pairs
# conclusions with HONCHO_SUMMARY_ENABLED=0 and HONCHO_PEER_CARD_ENABLED=0,
# since that arm never reads either section.
export HONCHO_RECALL_MODE="${HONCHO_RECALL_MODE:-hybrid}"
export HONCHO_SUMMARY_ENABLED="${HONCHO_SUMMARY_ENABLED:-1}"
export HONCHO_PEER_CARD_ENABLED="${HONCHO_PEER_CARD_ENABLED:-1}"
export HONCHO_DIALECTIC_REASONING_LEVEL="${HONCHO_DIALECTIC_REASONING_LEVEL:-low}"
export HONCHO_DIALECTIC_DYNAMIC="${HONCHO_DIALECTIC_DYNAMIC:-1}"
export HONCHO_REASONING_LEVEL_CAP="${HONCHO_REASONING_LEVEL_CAP:-high}"
export HONCHO_DIALECTIC_MAX_CHARS="${HONCHO_DIALECTIC_MAX_CHARS:-600}"
export HONCHO_DIALECTIC_MAX_INPUT_CHARS="${HONCHO_DIALECTIC_MAX_INPUT_CHARS:-10000}"
export HONCHO_MESSAGE_MAX_CHARS="${HONCHO_MESSAGE_MAX_CHARS:-25000}"
export HONCHO_SEARCH_LIMIT="${HONCHO_SEARCH_LIMIT:-10}"
export HONCHO_SEND_CREATED_AT="${HONCHO_SEND_CREATED_AT:-0}"
export HONCHO_DRAIN_TIMEOUT_S="${HONCHO_DRAIN_TIMEOUT_S:-1800}"
export HONCHO_DRAIN_POLL_S="${HONCHO_DRAIN_POLL_S:-2.0}"

# Per-run workspace prefix (design decision 1): every persona's Honcho
# workspace is "${HONCHO_WORKSPACE_PREFIX}p<idx>_<sanitized persona id>". An
# explicit override always wins. Derived here, in bash, the same way
# entrypoint.mem0.sh derives MEM0_COLLECTION from RUN_TAG, so the value is
# fixed once and both this script's logs and the adapter read the identical
# string.
if [ -z "${HONCHO_WORKSPACE_PREFIX:-}" ]; then
  _san="$(printf '%s' "${RUN_TAG:-run}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-zA-Z0-9_-]/_/g')"
  export HONCHO_WORKSPACE_PREFIX="hermes_${_san}_"
  unset _san
fi

# --- Server backend selection (shared honcho-api/-deriver vs. spawned) ------
# HONCHO_SERVER_MODE=shared attaches to the central honcho-api service (the
# "central DB" analog of Hindsight's shared hindsight-pg / Supermemory's
# shared central server). HONCHO_SERVER_MODE=spawn launches a disposable
# API+deriver(+Postgres under clock-sync) inside THIS container instead, the
# standalone single-process path. The compose default is shared (set in
# docker-compose.yml's HONCHO_SERVER_MODE interpolation default); the
# fallback below only applies to a bare `docker run` with no compose env.
export HONCHO_SERVER_MODE="${HONCHO_SERVER_MODE:-shared}"
case "$HONCHO_SERVER_MODE" in
  shared)
    # Clock-sync needs one perceived clock per server. A shared central
    # honcho-api/-deriver pair has exactly one, but each shard drives a
    # different persona/session timeline, so it cannot sit at N logical dates
    # at once (same reasoning as Supermemory's shared+clocksync guard).
    if bench_clocksync_enabled; then
      log "FATAL: BENCH_CLOCKSYNC=1 requires HONCHO_SERVER_MODE=spawn."
      log "       A shared central honcho-api/-deriver pair has a single perceived"
      log "       clock and cannot serve N shards at different logical session"
      log "       dates. Use spawn mode (per-shard disposable server), which"
      log "       run_shards.sh forces for the honcho clock-sync preset."
      exit 2
    fi
    # score/summarize never touch honcho-api (they read the Results file the
    # generate stage already wrote), so this skips the attach setup for them.
    # Run those stages with `--no-deps` to skip bringing up honcho-api/-deriver.
    case "${STAGE:-all}" in
      generate|all)
        export HONCHO_BASE_URL="${HONCHO_BASE_URL:-http://honcho-api:8000}"
        log "storage=shared server=$HONCHO_BASE_URL workspace_prefix=$HONCHO_WORKSPACE_PREFIX"
        ;;
      *)
        log "storage=shared (STAGE=$STAGE: no honcho-api attach needed -- use --no-deps to skip it)"
        ;;
    esac
    ;;
  spawn)
    # honcho/_honcho_server.py reads HONCHO_SERVER_DIR to find the vendored,
    # uv-synced external/honcho checkout this image bakes at build time
    # (Dockerfile.honcho). No network uv sync happens at container start.
    export HONCHO_SERVER_DIR="${HONCHO_SERVER_DIR:-/app/external/honcho}"
    export HONCHO_SERVER_PORT="${HONCHO_SERVER_PORT:-0}"
    export HONCHO_PG_CREATE_DB="${HONCHO_PG_CREATE_DB:-1}"
    # Unset any inherited shared-mode URL, so the adapter's own attach-vs-spawn
    # branch selects spawn cleanly even if HONCHO_BASE_URL leaked in from the
    # compose default.
    unset HONCHO_BASE_URL 2>/dev/null || true
    log "storage=spawn (per-run disposable server) dir=$HONCHO_SERVER_DIR pg_create_db=$HONCHO_PG_CREATE_DB"
    ;;
  *)
    log "unknown HONCHO_SERVER_MODE=$HONCHO_SERVER_MODE (expected shared|spawn)"; exit 2
    ;;
esac

# --- Spawn mode: the in-container Postgres the adapter connects to ----------
# Dockerfile.honcho bakes PGDG postgresql-16 + pgvector and an empty
# /var/lib/bench-pg for this arm, but nothing ever started a postmaster, so
# honcho/_honcho_server.py provision() got "connection refused" on its default
# localhost:5432 DSN (first Docker smoke, 2026-08-01). Spawn mode has no
# honcho-pg to attach to: `--no-deps` is part of the arm, and under clock-sync
# a co-tenant central cluster cannot read this shard's faked clock anyway. So
# this starts a throwaway cluster in THIS container, the same way
# entrypoint.retaindb-server.sh does. Generate only: score and summarize read
# the Results file the generate stage already wrote.
BENCH_PG_BINDIR=/usr/lib/postgresql/16/bin
BENCH_PG_DATA=/var/lib/bench-pg/data

bench_honcho_local_pg() {
  [ "$HONCHO_SERVER_MODE" = "spawn" ] || return 0
  case "$STAGE" in generate|all) ;; *) return 0 ;; esac
  # An explicit HONCHO_PG_DSN names an external cluster (a host smoke against
  # a local Postgres, or the compose honcho-pg service). Leave it alone.
  if [ -n "${HONCHO_PG_DSN:-}" ]; then
    log "spawn: HONCHO_PG_DSN is set; no in-container Postgres started"
    return 0
  fi

  log "spawn: initdb local cluster at $BENCH_PG_DATA (postgresql-16 + pgvector)"
  rm -rf "$BENCH_PG_DATA"
  runuser -u postgres -- "$BENCH_PG_BINDIR/initdb" --no-locale -E UTF8 -D "$BENCH_PG_DATA" >/dev/null

  # Under clock-sync the postmaster joins the faked clock domain, so Postgres
  # NOW() and the spawned API/deriver children read one clock. libfaketime is
  # injected per command: this shell keeps real time for its own timeouts.
  # Durability is off, because the cluster dies with the container.
  local pre=()
  if bench_clocksync_enabled; then
    pre=(env LD_PRELOAD="$BENCH_LIBFAKETIME"
         FAKETIME_TIMESTAMP_FILE="$BENCH_CLOCKSYNC_FILE"
         FAKETIME_NO_CACHE=1 FAKETIME_DONT_FAKE_MONOTONIC=1 NO_FAKE_STAT=1)
    log "spawn: starting local postmaster inside the faked clock domain"
  else
    log "spawn: starting local postmaster"
  fi
  runuser -u postgres -- "${pre[@]}" "$BENCH_PG_BINDIR/pg_ctl" -D "$BENCH_PG_DATA" \
      -o "-c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off" \
      -w start

  # _honcho_server.py's default DSN authenticates as postgres/postgres. initdb
  # leaves that role with no password, so set one here instead of widening
  # pg_hba, in case a future image ships a scram default for host lines.
  runuser -u postgres -- "$BENCH_PG_BINDIR/psql" -h 127.0.0.1 -d postgres -v ON_ERROR_STOP=1 \
      -c "ALTER ROLE postgres PASSWORD 'postgres';" >/dev/null

  # 127.0.0.1, not localhost: psycopg tries ::1 first, and the postmaster
  # listens on IPv4 only. The adapter appends the run database to this DSN
  # (HONCHO_PG_CREATE_DB=1 -> honcho_<sanitized RUN_TAG>).
  export HONCHO_PG_DSN="postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres"
  log "spawn: local Postgres ready on 127.0.0.1:5432"
}

# --- Spawn mode: the truncating embed proxy in front of vllm-embed ----------
# vllm-embed answers 400 to any input above the served window. Honcho's
# representation path calls simple_batch_embed
# (external/honcho/src/embedding_client.py:251), which does no chunking, so
# ONE long observation fails the whole save for both observers
# (src/crud/representation.py:111). Measured on smoke hn_smkmin_p0b under the
# earlier 512-token embedder: 14 dropped saves against 11 completed deriver
# batches in persona 0, sessions 0-2. Honcho exposes no way to forward
# truncate_prompt_tokens, and external/ must not be edited, so the proxy adds
# that field on the way through. It sends truncate_prompt_tokens=-1, which
# tells vLLM to cut at the served model's own max_model_len, so the proxy
# self-adjusts to whatever vllm-embed serves. At the current 32768-token
# window it is a near-inert backstop, and it stays enabled.
# This is the same precedent as retaindb_server/embed_proxy.py, wired by
# entrypoint.retaindb-server.sh.
export HONCHO_EMBED_PROXY="${HONCHO_EMBED_PROXY:-1}"
export HONCHO_EMBED_PROXY_PORT="${HONCHO_EMBED_PROXY_PORT:-3198}"
BENCH_EMBED_PROXY_PID=""

bench_honcho_embed_proxy_stop() {
  [ -n "$BENCH_EMBED_PROXY_PID" ] || return 0
  kill "$BENCH_EMBED_PROXY_PID" 2>/dev/null || true
  BENCH_EMBED_PROXY_PID=""
}

bench_honcho_embed_proxy() {
  [ "$HONCHO_EMBED_PROXY" = "1" ] || return 0
  # Spawn mode only. In shared mode the embedder URL of the central
  # honcho-api/-deriver services is set in docker-compose.yml and never reads
  # this shell's HONCHO_EMBEDDER_BASE_URL, so a proxy started here would
  # listen with no client.
  [ "$HONCHO_SERVER_MODE" = "spawn" ] || return 0
  case "$STAGE" in generate|all) ;; *) return 0 ;; esac

  export HONCHO_EMBED_PROXY_UPSTREAM="${HONCHO_EMBED_PROXY_UPSTREAM:-$HONCHO_EMBEDDER_BASE_URL}"
  export HONCHO_EMBED_PROXY_UPSTREAM_API_KEY="${HONCHO_EMBED_PROXY_UPSTREAM_API_KEY:-$HONCHO_EMBEDDER_API_KEY}"
  log "starting embed_proxy.py on 127.0.0.1:$HONCHO_EMBED_PROXY_PORT -> $HONCHO_EMBED_PROXY_UPSTREAM"
  # No LD_PRELOAD here on purpose: the proxy keeps this shell's real clock.
  # It speaks plain HTTP to vllm-embed and never opens TLS, so a faked
  # dataset-year clock could not break it either way.
  python3 -u "$ROOT/honcho/embed_proxy.py" &
  BENCH_EMBED_PROXY_PID=$!
  # Kill it when the stage ends. Docker also reaps it when PID 1 exits, so
  # this cannot block container exit; the trap just avoids an orphan during
  # a multi-stage STAGE=all run.
  trap bench_honcho_embed_proxy_stop EXIT

  local deadline=$(( $(date +%s) + 60 ))
  until curl -fsS "http://127.0.0.1:${HONCHO_EMBED_PROXY_PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; do
    if ! kill -0 "$BENCH_EMBED_PROXY_PID" 2>/dev/null; then
      log "FATAL: embed_proxy exited during startup"; exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "FATAL: embed_proxy did not become healthy within 60s"
      bench_honcho_embed_proxy_stop; exit 1
    fi
    sleep 1
  done
  # Point the spawned API and deriver at the proxy instead of vllm-embed.
  # _honcho_server.py maps this onto the children's
  # EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL.
  export HONCHO_EMBEDDER_BASE_URL="http://127.0.0.1:${HONCHO_EMBED_PROXY_PORT}/v1"
  log "embed_proxy healthy; HONCHO_EMBEDDER_BASE_URL=$HONCHO_EMBEDDER_BASE_URL"
}

RESDIR="$ROOT/honcho/Results"
SCOREDIR="$ROOT/honcho/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/honcho_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]              && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")
# CLI flags mirror the design contract's four adapter-facing knobs, env
# fallback (mem0-style): the adapter reads the HONCHO_* var directly if a
# flag is ever omitted, so passing the exported value here is a no-op when
# it already equals the adapter's own default.
CAPS+=(--recall_mode "$HONCHO_RECALL_MODE")
CAPS+=(--observation_mode "$HONCHO_OBSERVATION_MODE")
CAPS+=(--search_limit "$HONCHO_SEARCH_LIMIT")
[ "${HONCHO_SEND_CREATED_AT:-0}" = "1" ] && CAPS+=(--send_created_at)
# This overrides the dataset path, for example to benchmark/probes/.
# The default dataset is Step4_4.
[ -n "${INPUT_JSONL:-}" ]               && CAPS+=(--input_jsonl_path "$INPUT_JSONL")

# This sets the persona range. Personas run serially in one process.
# START_IDX/END_IDX let several containers split the dataset, each with its
# own RUN_TAG and results file (the mem0/supermemory sharding model).
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-$TOTAL}"

do_generate() {
  bench_answer_env  # canonical ANSWER decoding (identical across providers)
  # This is the clock-sync arm. It seeds the per-shard timestamp file with real
  # time and verifies that the preload bends perceived time in THIS image,
  # before the adapter spawns its server+deriver(+postgres) children.
  # LD_PRELOAD is NOT set in this shell; only the adapter's own child-env
  # builder injects it (spawn mode only). No-op unless BENCH_CLOCKSYNC=1.
  bench_clocksync_prepare /tmp/clocksync/faketime.rc
  bench_clocksync_probe
  # Spawn mode only. This must run AFTER bench_clocksync_prepare, because the
  # postmaster reads BENCH_CLOCKSYNC_FILE that call creates, and BEFORE the
  # adapter, which connects to Postgres in its first setup step.
  bench_honcho_local_pg
  # This must run BEFORE the adapter, which spawns the API and deriver with
  # whatever HONCHO_EMBEDDER_BASE_URL holds at that moment.
  bench_honcho_embed_proxy
  # This captures the serving envelope, builds the manifest (run-contract hash),
  # and starts token accounting. Aborts under STRICT_RUN_CONTRACT=1 or
  # BENCH_CLOCKSYNC=1 if the run contract is incomplete. Runs OUTSIDE any
  # faked-clock scope, so provenance timestamps stay real wall-clock time.
  bench_generate_preamble "$ROOT/honcho" "$TAG"
  log "GENERATE personas=[$START_IDX,$END_IDX) top_k=$TOPK recall_mode=$HONCHO_RECALL_MODE observation_mode=$HONCHO_OBSERVATION_MODE server_mode=$HONCHO_SERVER_MODE internal_llm=$HONCHO_LLM_MODEL embed=$HONCHO_EMBEDDER_MODEL answer_max_tokens=${OPENAI_MAX_TOKENS} clocksync=${BENCH_CLOCKSYNC:-0} -> $RESULTS_FILE"
  python3 -u "$ROOT/honcho/eval_honcho.py" \
      --start_idx "$START_IDX" --end_idx "$END_IDX" --top_k "$TOPK" "${CAPS[@]}" \
      --output_jsonl_path "$RESULTS_FILE" \
      --output_json_path "$RESULTS_JSON"
  bench_tokens_finish "$ROOT/honcho" "$TAG"
}

run_stage "$ROOT/honcho" "$TAG" do_generate
