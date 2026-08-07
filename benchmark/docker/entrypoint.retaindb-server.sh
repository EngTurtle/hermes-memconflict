#!/usr/bin/env bash
# MemConflict and RetainDB SERVER edition run-service entrypoint. This is a
# scaffold.
#
# This script benchmarks @retaindb/server (Postgres/pgvector plus LLM
# extraction). The retaindb/ LOCAL entrypoint lets the Python adapter spawn a
# disposable node server. The SERVER edition works differently: this script
# manages its lifecycle. This script creates a per-run database on the
# shared hindsight-pg service. It then runs `prisma migrate deploy` for the
# server's own migrations, launches `node dist/index.js`, and waits for
# health. Finally, it runs the requested STAGE. The Python adapter only
# attaches to the running server.
#
# Sharding follows Hindsight's model. Each shard container runs its own
# in-container server. Each server has its own per-run database on the
# shared pg instance (RUN_TAG=full_s0, and so on). Each shard serves its
# personas one at a time, in one process, through START_IDX and END_IDX.
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[retaindb-server $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score and
# summarize calls. The fairness contract requires byte-for-byte identical
# config for every provider. BENCH_PYTHON=python3 is required because the
# node:20 base image ships python3 only, not a bare `python` command. The
# shared run_score/run_summarize helpers must call python3 for that reason.
export BENCH_PYTHON=python3
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"
# This sources the clock-sync arm helpers (bench_clocksync_prepare, probe,
# enabled, and BENCH_LIBFAKETIME). All of them are no-ops unless
# BENCH_CLOCKSYNC=1. Sourcing this file always happens and has no side
# effects by itself.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"
# This sources the run-contract helpers: serving-envelope capture, the
# manifest, run_contract_hash, and vLLM token accounting.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"
# This sources the named launch presets (PRESET=<name>). It runs before the
# empty-RETAINDB_ guard and before the clock-sync local-Postgres bring-up. A
# preset that sets BENCH_CLOCKSYNC=1 or DISABLE_SCHEDULER=false therefore
# reaches both. If PRESET is unset, this is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset retaindb_server

# This is the empty-var guard, and it mirrors entrypoint.hindsight.sh.
# Compose's `environment:` block sets every key it lists. So an
# unconfigured `${RETAINDB_SERVER_X:-}` reaches this script as a present
# empty string, not as an absent variable. This unsets any empty RETAINDB_*
# variable, so downstream `${VAR:-default}` fallbacks and the server's own
# `||` defaults apply cleanly. For example, an empty RETAINDB_API_KEY must
# read as "open".
unset_empty_env_with_prefix RETAINDB_

STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-retaindb_server}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-$TOTAL}"

# This is the answer/judge LLM used by MemConflict's llm_request. Compose
# sets OPENAI_BASE_URL and OPENAI_MODEL. The SDK needs a non-empty key even
# for vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"

# --- Clock-sync arm: per-shard INTERNAL Postgres in the faked clock domain ----
# When BENCH_CLOCKSYNC=1, and only for STAGE generate or all, this section
# starts a throwaway per-shard Postgres. This lets the node app and its
# database share one faked clock that tracks the dataset's logical session
# dates. This fixes the validUntil wall-clock leak: the app writes
# validUntil=new Date(), while the database filters on validUntil>NOW(). If
# both read the real wall clock, conflict recall on backdated data breaks. A
# shared central hindsight-pg instance cannot have its clock faked per shard,
# so this section starts a local cluster instead. This block runs before the
# create-db block below, and only redirects the Postgres connection to
# 127.0.0.1 through the RETAINDB_SERVER_PG_* exports. When BENCH_CLOCKSYNC is
# unset, the create-db flow, the DATABASE_URL composition, and the whole
# default shared-hindsight-pg path stay byte-identical.
#
# This script launches the postmaster inside the clock domain using
# per-command env only. It never exports LD_PRELOAD for the whole shell.
# The entrypoint's own `date +%s` health-wait loops and curl calls must stay
# on the real clock. The un-faked Python adapter is the only writer of the
# timestamp file (benchmark/clock_sync.py, once per session). It polls the
# lifecycle with real-clock timeouts. Postgres NOW() and node's Date() both
# read the same faked file live. So the scheduler's inactivity math runs in
# one consistent domain that advances at 1x speed.
BENCH_PG_BINDIR=/usr/lib/postgresql/18/bin
BENCH_PG_DATA=/var/lib/bench-pg/data
BENCH_PG_ROLE="${RETAINDB_SERVER_PG_USER:-${HINDSIGHT_PG_USER:-hindsight}}"
BENCH_PG_PW="${RETAINDB_SERVER_PG_PASSWORD:-${HINDSIGHT_PG_PASSWORD:-hindsight}}"

bench_clocksync_local_pg() {
  bench_clocksync_enabled || return 0
  case "$STAGE" in generate|all) ;; *) return 0 ;; esac

  bench_clocksync_prepare /tmp/clocksync/faketime.rc
  bench_clocksync_probe

  log "clocksync: initdb local cluster at $BENCH_PG_DATA"
  rm -rf "$BENCH_PG_DATA"
  runuser -u postgres -- "$BENCH_PG_BINDIR/initdb" --no-locale -E UTF8 -D "$BENCH_PG_DATA" >/dev/null

  # This starts the postmaster in the faked clock domain, using per-command
  # env. The -w flag waits for the postmaster to become ready. This data is
  # throwaway per shard, so this turns durability off (fsync and
  # synchronous_commit).
  log "clocksync: starting local postmaster inside the faked clock domain"
  runuser -u postgres -- env \
      LD_PRELOAD="$BENCH_LIBFAKETIME" \
      FAKETIME_TIMESTAMP_FILE="$BENCH_CLOCKSYNC_FILE" \
      FAKETIME_NO_CACHE=1 FAKETIME_DONT_FAKE_MONOTONIC=1 NO_FAKE_STAT=1 \
      "$BENCH_PG_BINDIR/pg_ctl" -D "$BENCH_PG_DATA" \
      -o "-c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off" \
      -w start

  # This creates the same role, with the same name and password, that the
  # shared path uses. So the create-db flow and DATABASE_URL compose the
  # same way against the local cluster. The role needs SUPERUSER so
  # hindsight_create_db.py can run CREATE DATABASE and CREATE EXTENSION
  # vector.
  log "clocksync: ensuring role ${BENCH_PG_ROLE}"
  runuser -u postgres -- "$BENCH_PG_BINDIR/psql" -h 127.0.0.1 -d postgres -v ON_ERROR_STOP=1 -c \
    "DO \$\$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='${BENCH_PG_ROLE}') THEN CREATE ROLE ${BENCH_PG_ROLE} LOGIN SUPERUSER PASSWORD '${BENCH_PG_PW}'; END IF; END \$\$;"

  # This redirects the create-db block below, and DATABASE_URL, at the local
  # cluster. This overrides compose's RETAINDB_SERVER_PG_HOST=hindsight-pg
  # default, with no edit to that block. The already-exists gate below has no
  # effect on this fresh cluster.
  export RETAINDB_SERVER_PG_HOST=127.0.0.1
  export RETAINDB_SERVER_PG_PORT=5432
  export RETAINDB_SERVER_PG_USER="$BENCH_PG_ROLE"
  export RETAINDB_SERVER_PG_PASSWORD="$BENCH_PG_PW"

  # This defeats the session-lifecycle scheduler's clock-rewind lockout. The
  # scheduler throttles on faked `Date.now() - lastRun <
  # SESSION_LIFECYCLE_INTERVAL_MS`. Its first tick runs at boot. At that
  # point the clock-sync file still reads +0, the real wall clock, and the
  # tick stamps lastRun there. Once the driver steps the clock back to a
  # dataset year, the diff becomes hugely negative. It falls below any
  # positive interval. So the lifecycle never runs again. A large negative
  # interval makes the throttle condition unsatisfiable in either clock
  # direction. So it also survives clock rewinds across per-shard persona
  # rollovers. With this setting, the lifecycle fires on every 60-second
  # tick. This line runs before the create-db block's `:-` default below, so
  # this value wins. This function returns early on non-clocksync runs, so
  # only the clock-sync arm reaches this line. Non-clocksync runs keep the
  # shipped throttle.
  export SESSION_LIFECYCLE_INTERVAL_MS="${SESSION_LIFECYCLE_INTERVAL_MS:--1000000000000000}"
  log "clocksync: local Postgres ready; server DB on 127.0.0.1:5432; lifecycle throttle disabled (rewind-proof)"
}
bench_clocksync_local_pg

# --- Per-run database on the SHARED hindsight-pg service ----------------------
# This reuses the common pgvector database in the infrastructure, per user
# direction. The name is "retaindb_" plus RUN_TAG, lowercased, with every
# character outside [a-z0-9_] mapped to "_" (an empty RUN_TAG becomes
# "retaindb_default"). Sharded runs pass a RUN_TAG like full_s0, so each
# shard gets its own database. This is fine, because RetainDB migrations run
# per database and personas never span shards. This reuses
# hindsight_create_db.py, a generic script that creates the database and the
# vector/pg_trgm extensions. RetainDB's own migration 0 also runs CREATE
# EXTENSION vector, so this is a belt-and-suspenders check.
_pg_host="${RETAINDB_SERVER_PG_HOST:-${HINDSIGHT_PG_HOST:-hindsight-pg}}"
_pg_port="${RETAINDB_SERVER_PG_PORT:-${HINDSIGHT_PG_PORT:-5432}}"
_pg_user="${RETAINDB_SERVER_PG_USER:-${HINDSIGHT_PG_USER:-hindsight}}"
_pg_password="${RETAINDB_SERVER_PG_PASSWORD:-${HINDSIGHT_PG_PASSWORD:-hindsight}}"
if [ -n "${RETAINDB_SERVER_PG_DB:-}" ]; then
  _pg_db="$RETAINDB_SERVER_PG_DB"
else
  bench_pg_db_name "retaindb_" "retaindb_default"
  _pg_db="$_BENCH_PG_DB"
fi

log "ensuring per-run database ${_pg_db} on ${_pg_host}:${_pg_port}"
if ! _create_out="$(HS_PG_HOST="$_pg_host" HS_PG_PORT="$_pg_port" HS_PG_USER="$_pg_user" \
    HS_PG_PASSWORD="$_pg_password" HS_PG_DB="$_pg_db" \
    python3 "$(dirname "${BASH_SOURCE[0]}")/hindsight_create_db.py" 2>&1)"; then
  printf '%s\n' "$_create_out"
  log "FATAL: per-run database creation failed for ${_pg_db}"
  exit 1
fi
printf '%s\n' "$_create_out"
# A generate stage, either bare `generate` or `all`, must start against a
# fresh per-run database. Both ingest data, so a relaunch under the same
# RUN_TAG would silently ingest into old memories. A score or summarize
# rerun legitimately reuses the database, so this gate applies only to
# generate and all. ALLOW_EXISTING_DB=1 overrides this gate on purpose.
case "$STAGE" in
  generate|all)
    if printf '%s' "$_create_out" | grep -q "already exists"; then
      if [ "${ALLOW_EXISTING_DB:-0}" = "1" ]; then
        log "WARN: per-run database ${_pg_db} already exists but ALLOW_EXISTING_DB=1 — reusing it (stale memories from a prior run under this RUN_TAG may remain)."
      else
        log "FATAL: per-run database ${_pg_db} already exists for a STAGE=${STAGE} run."
        log "       A prior run used RUN_TAG='${RUN_TAG:-}' (db ${_pg_db}). Reusing it would ingest"
        log "       into the old memories. DROP DATABASE ${_pg_db} on ${_pg_host} or use a NEW RUN_TAG."
        log "       To override intentionally: -e ALLOW_EXISTING_DB=1."
        exit 1
      fi
    fi
    ;;
esac
export DATABASE_URL="postgresql://${_pg_user}:${_pg_password}@${_pg_host}:${_pg_port}/${_pg_db}"

# --- RetainDB server best-effort config (documented in docs/DECISIONS.md) -----
# DISABLE_SCHEDULER=true is the MINIMAL arm and the default. It gives a
# deterministic baseline with no consolidation. DISABLE_SCHEDULER=false is
# the FEATURED arm. In that arm the server's 60-second scheduler runs
# runSessionLifecycle(). This function promotes cold SESSION memories to
# USER scope and writes a per-session summary. EXTRACTOR_MODEL is the variable the
# server actually reads. EXTRACTION_MODEL, from the vendor .env.example, is a
# dead variable with no effect. This defaults EXTRACTOR_MODEL to the answer
# model. OPENAI_BASE_URL and OPENAI_API_KEY, set by compose to vllm-gen,
# drive the server's extraction LLM directly, with no proxy on that path. A
# non-empty key is required, or extraction silently falls back to
# regex/pattern matching only.
export DISABLE_SCHEDULER="${DISABLE_SCHEDULER:-true}"
export EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-${OPENAI_MODEL:-qwen3.5-4b}}"

# --- Featured session-lifecycle tuning (only meaningful when scheduler is ON) --
# The scheduler ticks every 60 seconds. This is hardcoded in
# engine/scheduler.ts:110, is not configurable through env, and is the floor
# on per-session lifecycle latency. The server reads two other knobs at
# module load. This lowers both knobs so the lifecycle fires promptly.
#   * SESSION_INACTIVITY_THRESHOLD_MS (session-lifecycle.ts:19, default 2
#     hours). This sets how long since a session's last memory write, by
#     wall-clock createdAt, before the session counts as "cold". This lowers
#     it to 5 seconds, so a just-ingested session qualifies on the next tick.
#   * SESSION_LIFECYCLE_INTERVAL_MS (scheduler.ts:86, default 10 minutes)
#     throttles how often the lifecycle sub-task runs within the 60-second
#     ticks. This lowers it to 1 second, so the sub-task runs on every tick.
#     Both variables are env-overridable. This exports both unconditionally.
#     They have no effect when DISABLE_SCHEDULER=true. The server reads them
#     from process.env.
export SESSION_INACTIVITY_THRESHOLD_MS="${SESSION_INACTIVITY_THRESHOLD_MS:-5000}"
export SESSION_LIFECYCLE_INTERVAL_MS="${SESSION_LIFECYCLE_INTERVAL_MS:-1000}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/opt/retaindb_models}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PORT="${PORT:-3000}"
export RETAINDB_SERVER_BASE_URL="${RETAINDB_SERVER_BASE_URL:-http://127.0.0.1:${PORT}}"

# This is the node server log file. The server's stdout and stderr land
# here. A tail process also echoes them into container stdout, so
# `docker logs` is unchanged. The adapter reads this file for the scheduler's
# per-session completion marker,
# `[session-lifecycle] <sid>: promoted=N skipped=M summary=<uuid|skipped>`.
# This marker is the definitive release signal for the featured lifecycle
# barrier. It replaces the former fixed-grace guess (see
# eval_retaindb_server.py). This exports the path so the adapter can find it.
# On the MINIMAL arm (DISABLE_SCHEDULER=true) the file exists but the adapter
# does not read it.
export RETAINDB_SERVER_LOG="${RETAINDB_SERVER_LOG:-/tmp/retaindb_server_${TAG}.log}"

# --- Embedding mode: CONTRACT remote (default) vs OFF-CONTRACT local debug -----
# RETAINDB_EMBEDDING_MODE=remote is the Docker default and the contract
# config. In this mode embeddings come from the same embedder every other
# provider uses: the shared vllm-embed service (Alibaba-NLP/gte-modernbert-base,
# 768-dim). RetainDB's remote inference protocol is a bespoke
# {inputs}->{embeddings} shape, not OpenAI's shape. So embed_proxy.py
# translates requests to vllm's OpenAI /v1/embeddings endpoint. It then
# right-pads the result to EMBED_PROXY_PAD_DIM, which RetainDB's pinned
# vector(1024) schema requires. The pad takes the 768-dim vectors to 1024
# with zeros, which preserves both the L2 norm and the dot product, so
# ranking is unchanged.
# The source (engine/embeddings.ts,
# inference-client.ts) forces two traps here. First, the remote path silently
# falls back unless REMOTE_INFERENCE_REQUIRED=true. That fallback would call
# OpenAI embeddings on vllm-gen or OpenRouter. Neither serves that endpoint,
# so the result is garbage or an error. Second, INFERENCE_TIMEOUT_MS
# defaults to 2500 ms, too short for a batch request.
#
# RETAINDB_EMBEDDING_MODE=local is an off-contract debug knob only. It runs
# an in-process Xenova/bge-large-en-v1.5 model (1024-dim) on CPU, a
# different embedder. So its numbers are not comparable to the other
# providers. The host smoke test uses this mode because the host has no
# GPU.
RETAINDB_EMBEDDING_MODE="${RETAINDB_EMBEDDING_MODE:-remote}"
EMBED_PROXY_PORT="${EMBED_PROXY_PORT:-3199}"
EMBED_PROXY_PID=""
SERVER_LOG_TAIL_PID=""
case "$RETAINDB_EMBEDDING_MODE" in
  remote)
    export EMBEDDING_MODE="remote"
    export EMBEDDING_INFERENCE_BASE_URL="http://127.0.0.1:${EMBED_PROXY_PORT}"
    export REMOTE_INFERENCE_REQUIRED="true"           # fail loudly, never fall back to OpenAI silently
    export INFERENCE_TIMEOUT_MS="${INFERENCE_TIMEOUT_MS:-60000}"  # gives batch embed calls enough time (default 2500 ms is too short)
    export EMBED_PROXY_PORT
    export EMBED_PROXY_UPSTREAM_BASE_URL="${EMBED_PROXY_UPSTREAM_BASE_URL:-http://vllm-embed:8000/v1}"
    export EMBED_PROXY_UPSTREAM_MODEL="${EMBED_PROXY_UPSTREAM_MODEL:-gte-modernbert-base}"
    export EMBED_PROXY_UPSTREAM_API_KEY="${EMBED_PROXY_UPSTREAM_API_KEY:-local-vllm}"
    export EMBED_PROXY_PAD_DIM="${EMBED_PROXY_PAD_DIM:-1024}"
    log "embeddings=remote (CONTRACT) via proxy :$EMBED_PROXY_PORT -> $EMBED_PROXY_UPSTREAM_BASE_URL (model=$EMBED_PROXY_UPSTREAM_MODEL pad=$EMBED_PROXY_PAD_DIM)"
    ;;
  local)
    export EMBEDDING_MODE="local"
    log "embeddings=local (OFF-CONTRACT debug: in-process Xenova/bge-large-en-v1.5, CPU — numbers NOT comparable)"
    ;;
  *)
    log "unknown RETAINDB_EMBEDDING_MODE=$RETAINDB_EMBEDDING_MODE (expected remote|local)"; exit 2 ;;
esac

# ENCRYPTION_KEY is fresh-deploy fix #4 (see
# retaindb_server/server_patches/README.md). The server refuses to boot
# without a non-empty key of at least 32 characters. This key guards only
# agent-task connector credential encryption, not the memory path. So a
# documented, non-secret dev default is fine for the benchmark. Set the env
# var to override it for a real deploy.
export ENCRYPTION_KEY="${ENCRYPTION_KEY:-retaindb-benchmark-dev-encryption-key-0123456789}"

# RETAINDB_DISABLE_SEARCH_CACHE=true is patch 0004, and it bypasses both
# search caches. The semantic cache keys only on query-embedding similarity,
# with no project, user, or question_date scoping, which risks cross-tenant
# leakage. The exact-key cache omits question_date. MemConflict re-asks
# dynamic questions, repeating the same text at a later logical date. The
# benchmark compresses months into seconds, so both caches sit inside the
# 300-second TTL for those repeated questions. Both caches then return
# stale, temporally wrong results. This disables both caches for
# measurement correctness. Set
# the env var to override. The vendor server default is caches on.
export RETAINDB_DISABLE_SEARCH_CACHE="${RETAINDB_DISABLE_SEARCH_CACHE:-true}"

SERVER_PKG=/opt/retaindb_server_src/packages/server

start_embed_proxy() {
  [ "$RETAINDB_EMBEDDING_MODE" = "remote" ] || return 0
  log "starting embed_proxy.py on :$EMBED_PROXY_PORT"
  python3 -u "$ROOT/retaindb_server/embed_proxy.py" &
  EMBED_PROXY_PID=$!
  local deadline=$(( $(date +%s) + 60 ))
  until curl -fsS "http://127.0.0.1:${EMBED_PROXY_PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; do
    if ! kill -0 "$EMBED_PROXY_PID" 2>/dev/null; then
      log "FATAL: embed_proxy exited during startup"; exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "FATAL: embed_proxy did not become healthy within 60s"; kill "$EMBED_PROXY_PID" 2>/dev/null || true; exit 1
    fi
    sleep 1
  done
  log "embed_proxy healthy at http://127.0.0.1:${EMBED_PROXY_PORT}"
}

start_server() {
  start_embed_proxy
  cd "$SERVER_PKG"
  log "prisma migrate deploy -> ${_pg_db}"
  pnpm exec prisma migrate deploy --schema=./prisma/schema.prisma
  # This applies post-migrate SQL, fixes #3 and #7. seed.sql creates the
  # default org and owner user that the server's project auto-creation
  # references by foreign key. post_migrate.sql then drops the empty-table
  # ivfflat indexes, so retrieval uses exact KNN instead. An ivfflat index
  # on an empty table with probes=1 collapses recall. This step is idempotent
  # and runs through asyncpg, because the node:20 image has no psql. It runs
  # after migrate deploy and before the server starts.
  log "applying post-migrate SQL (seed + drop empty-table ivfflat indexes)"
  python3 "$ROOT/retaindb_server/server_patches/apply_seed.py"
  log "starting node dist/index.js on :${PORT} (EMBEDDING_MODE=$EMBEDDING_MODE DISABLE_SCHEDULER=$DISABLE_SCHEDULER EXTRACTOR_MODEL=$EXTRACTOR_MODEL SESSION_INACTIVITY_THRESHOLD_MS=$SESSION_INACTIVITY_THRESHOLD_MS SESSION_LIFECYCLE_INTERVAL_MS=$SESSION_LIFECYCLE_INTERVAL_MS)"
  # This redirects the server's stdout and stderr to a file,
  # $RETAINDB_SERVER_LOG. The adapter scans that file for the
  # session-lifecycle completion marker. A background `tail -F` also echoes
  # the file into container stdout, so `docker logs` still shows everything.
  # File redirection, not a pipe, is required for correct PID handling.
  # `node ... | tee f &` would set $! to the tee PID and silently break the
  # liveness and kill logic below. `node ... >>file 2>&1 &` keeps $! as the
  # node PID (the faketime branch's `env ...` execs into node and keeps the
  # same PID).
  : > "$RETAINDB_SERVER_LOG"
  tail -n +1 -F "$RETAINDB_SERVER_LOG" &
  SERVER_LOG_TAIL_PID=$!
  # The prisma migrate deploy and apply_seed calls above ran on the real
  # clock. That is harmless DDL and seed data on a throwaway cluster. On the
  # clock-sync arm, the node server itself must run inside the faked clock
  # domain. This way its scheduler's Date() matches the local Postgres
  # NOW(). This applies the fake clock through per-command env only, never
  # for the whole shell. embed_proxy.py always stays on the real clock.
  if bench_clocksync_enabled; then
    log "clocksync: launching node inside the faked clock domain"
    env LD_PRELOAD="$BENCH_LIBFAKETIME" \
        FAKETIME_TIMESTAMP_FILE="$BENCH_CLOCKSYNC_FILE" \
        FAKETIME_NO_CACHE=1 FAKETIME_DONT_FAKE_MONOTONIC=1 NO_FAKE_STAT=1 \
        node dist/index.js >> "$RETAINDB_SERVER_LOG" 2>&1 &
  else
    node dist/index.js >> "$RETAINDB_SERVER_LOG" 2>&1 &
  fi
  SERVER_PID=$!
  log "server log -> $RETAINDB_SERVER_LOG (tail pid $SERVER_LOG_TAIL_PID echoes it to container stdout)"
  # This waits for health. Server startup, plus the local BGE model load, can
  # take some time.
  local deadline=$(( $(date +%s) + 300 ))
  until curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FATAL: RetainDB server exited during startup"; exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "FATAL: RetainDB server did not become healthy within 300s"; kill "$SERVER_PID" 2>/dev/null || true; exit 1
    fi
    sleep 1
  done
  cd "$ROOT"
  log "server healthy at http://127.0.0.1:${PORT}"
}

stop_server() {
  # The RetainDB node server registers process.on('SIGTERM', ...) in
  # engine/compressor.ts. This overrides Node's default behavior of
  # terminating on SIGTERM. So a plain `kill` runs the server's handler, but
  # leaves the HTTP server and embedding worker running. The event loop
  # never exits. A bare `wait "$SERVER_PID"` would then block PID 1 forever.
  # The container would stay "running" after generate finished. The fix
  # sends SIGTERM, waits a bounded grace period, then sends SIGKILL, which
  # the process cannot trap. This lets `wait` return promptly, so the
  # container exits with the eval's status. All memories commit
  # synchronously, through write_mode=sync, so a forced kill loses no state.
  local pid
  for pid in "${SERVER_PID:-}" "${EMBED_PROXY_PID:-}" "${SERVER_LOG_TAIL_PID:-}"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null || true
  done
  local deadline=$(( $(date +%s) + 8 ))
  for pid in "${SERVER_PID:-}" "${EMBED_PROXY_PID:-}" "${SERVER_LOG_TAIL_PID:-}"; do
    [ -n "$pid" ] || continue
    while kill -0 "$pid" 2>/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  # On the clock-sync arm, this tears down the throwaway local cluster.
  # The immediate stop mode skips a checkpoint. That is fine, because
  # durability is already off and --rm discards the data directory. This
  # check guards on the arm, so it is a harmless no-op if the cluster never
  # started.
  if bench_clocksync_enabled; then
    runuser -u postgres -- "$BENCH_PG_BINDIR/pg_ctl" -D "$BENCH_PG_DATA" -m immediate stop 2>/dev/null || true
  fi
}
trap stop_server EXIT

RESDIR="$ROOT/retaindb_server/Results"
SCOREDIR="$ROOT/retaindb_server/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/retaindb_server_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]              && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")
[ -n "${RETAINDB_SERVER_PROFILE:-}" ]   && CAPS+=(--profile "$RETAINDB_SERVER_PROFILE")
# This lets INPUT_JSONL override the dataset, for example with a probe file
# in benchmark/probes/. The default dataset is Step4_4.
[ -n "${INPUT_JSONL:-}" ]               && CAPS+=(--input_jsonl_path "$INPUT_JSONL")

do_generate() {
  bench_answer_env
  # This captures the serving envelope and manifest, computes the
  # run_contract_hash, and starts token accounting. It aborts under
  # STRICT_RUN_CONTRACT=1 or BENCH_CLOCKSYNC=1 if the contract is incomplete.
  # This shell stays on the real clock. Only the postmaster and the node
  # server run inside the faked domain, so provenance timestamps are real.
  bench_generate_preamble "$ROOT/retaindb_server" "$TAG"
  log "GENERATE personas=[$START_IDX,$END_IDX) top_k=$TOPK embed=$EMBEDDING_MODE thinking=${MEMCONFLICT_ENABLE_THINKING} answer_max_tokens=${OPENAI_MAX_TOKENS} -> $RESULTS_FILE"
  python3 -u "$ROOT/retaindb_server/eval_retaindb_server.py" \
      --start_idx "$START_IDX" --end_idx "$END_IDX" --top_k "$TOPK" "${CAPS[@]}" \
      --base_url "$RETAINDB_SERVER_BASE_URL" \
      --output_jsonl_path "$RESULTS_FILE" \
      --output_json_path "$RESULTS_JSON"
  bench_tokens_finish "$ROOT/retaindb_server" "$TAG"
}

# Only the generate stage needs the live server. score and summarize only
# read files.
generate_with_server() { start_server; do_generate; }

run_stage "$ROOT/retaindb_server" "$TAG" generate_with_server
