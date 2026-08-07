#!/usr/bin/env bash
# MemConflict x Hindsight run-service entrypoint — SCAFFOLD.
#
# This mirrors the Mnemosyne entrypoint's stage shape (generate, score,
# summarize) but runs single-process. Hindsight isolates personas internally
# by bank and runs an embedded Postgres daemon, so there is no per-persona
# sharding here. The answer, judge, and Hindsight-internal LLM default to
# the shared vLLM, set by docker-compose. Point them at OpenRouter
# gpt-oss-120b for the verified path.
#
# This has NOT yet run in Docker. See Dockerfile.hindsight and README.md for
# the TODOs.
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[hindsight $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score and
# summarize calls. The fairness contract requires this file to be
# byte-for-byte identical for every provider. Previously this entrypoint set
# NO answer/judge decoding, so answers ran at the vLLM server defaults while
# Mnemosyne pinned temperature 0.2. That was a fairness bug, and this line
# fixes it. The path is relative to THIS script, so it resolves from any
# cwd.
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"
# This sources the clock-sync arm helpers (bench_clocksync_enabled,
# bench_clocksync_prepare, bench_clocksync_probe, and BENCH_LIBFAKETIME).
# Every one of them returns without acting unless BENCH_CLOCKSYNC=1, and
# sourcing the file by itself exports no LD_PRELOAD and writes no file. Only
# the HINDSIGHT_PG_MODE=pg0 branch below calls them, so the shared-pg arms
# behave exactly as before.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"
# This sources the run-contract helpers: serving-envelope capture, the
# manifest and run_contract_hash, and vLLM token accounting.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"
# This sources the named launch presets (PRESET=<name>) and applies them
# BEFORE the empty-var guard and the storage, embedding, and reranker
# selection below, so a preset value is a real, non-empty env var by the
# time any of them reads it. If PRESET is unset, this step is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset hindsight

# HindsightConfig.from_env() parses several HINDSIGHT_API_* variables with
# `int(os.getenv(VAR, str(DEFAULT)))`, or with a bare string or Literal
# assignment. os.getenv(VAR, default) substitutes `default` ONLY when VAR is
# fully UNSET. A variable that is SET to an empty string still returns an
# empty string, and int("") raises an error. docker-compose's `environment:`
# block always SETS every key it lists. An empty-default entry, such as
# `${HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET:-}`, resolves to a real,
# present, empty-string environment variable when the caller does not
# override it. So an unconfigured consolidation bound reaches the daemon as
# an empty string, not as an absence, and this can crash config parsing.
# This has happened before. This line unsets any empty HINDSIGHT_API_*
# variable here, so from_env() sees a true absence and falls back to the
# package default.
unset_empty_env_with_prefix HINDSIGHT_API_

# These reads sit ABOVE the storage selection below, because the
# HINDSIGHT_PG_MODE=pg0 branch gates on STAGE, derives its per-container HOME
# from TAG, and refuses a persona range wider than one. They stay after
# bench_apply_preset, so a preset value still wins. The shared branch reads
# only ${STAGE:-all} and ${RUN_TAG:-}, and STAGE gets the same "all" default
# here, so moving these lines changes nothing for it.
STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-hindsight}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-$TOTAL}"

# --- Storage backend selection (shared Postgres only) -------------------------
# HINDSIGHT_PG_MODE=shared is the only supported mode. It points the
# embedded Hindsight daemon at the shared hindsight-pg service instead of
# booting its own pg0 cluster. The embed manager
# (hindsight_embed/daemon_embed_manager.py) reads
# HINDSIGHT_EMBED_API_DATABASE_URL straight from the environment and maps it
# to the daemon's HINDSIGHT_API_DATABASE_URL. Because it is a postgresql://
# URL, not a pg0:// URL, parse_pg0_url() returns is_pg0=False, so NO
# embedded Postgres starts. One daemon per shard container still runs. They
# share the one database and are isolated by a globally unique bank_id.
# Migrations across the concurrent daemons serialize on Hindsight's
# per-schema advisory lock.
#
# This runs AFTER the empty-HINDSIGHT_API_* guard above, so the shared-mode
# pool caps it exports are not stripped. The URL variable has the
# HINDSIGHT_EMBED_ prefix and that guard never touches it.
#
# The team removed the legacy per-shard embedded pg0 mode
# (HINDSIGHT_PG_MODE=embedded) on 2026-07-22. Every real run uses shared;
# git history has the branch.
HINDSIGHT_PG_MODE="${HINDSIGHT_PG_MODE:-shared}"
case "$HINDSIGHT_PG_MODE" in
  shared)
    _pg_host="${HINDSIGHT_PG_HOST:-hindsight-pg}"
    _pg_port="${HINDSIGHT_PG_PORT:-5432}"
    _pg_user="${HINDSIGHT_PG_USER:-hindsight}"
    _pg_password="${HINDSIGHT_PG_PASSWORD:-hindsight}"
    # --- PER-RUN DATABASE ISOLATION (2026-07-20) -----------------------------
    # Each run gets its OWN database inside the shared server, derived from
    # RUN_TAG, so a run's daemon only ever sees its OWN banks.
    #
    # WHY, verified live on 2026-07-20: a consolidation-enabled (Arm B)
    # daemon's startup sweep enqueues consolidation ops for EVERY bank in its
    # database. With all runs sharing ONE database, that sweep reached
    # across runs. The team watched an Arm-B daemon enqueue consolidation
    # onto a consolidation-OFF run's LIVE bank, a cross-arm contamination
    # that is a fairness hazard, and onto orphan banks left by dead runs,
    # which wastes GPU time. Giving each run its own database fixes both
    # problems: a run only ever sees its own banks, and a fresh run starts
    # against a fresh catalog. bank_id isolation alone was NOT enough,
    # because the sweep runs per database, not per bank.
    #
    # The database name is "hindsight_" plus RUN_TAG, lowercased, with every
    # character outside [a-z0-9_] mapped to "_" (an empty RUN_TAG becomes
    # "hindsight_default"). Sharded runs pass a RUN_TAG like armB_s0 or
    # armB_s1, so with plain sanitization EACH SHARD gets its OWN database.
    # This is FINE: banks never span shards, and migrations run per database
    # under Hindsight's per-schema advisory lock, so each shard database
    # migrates independently with no cross-shard coordination.
    #
    # An explicit HINDSIGHT_PG_DB override WINS over the derived name
    # (compose defaults it to empty, so this derivation is the norm).
    # WARNING: overriding this re-introduces cross-run sharing. Two runs
    # pointed at the same HINDSIGHT_PG_DB land in one database and lose this
    # isolation.
    if [ -n "${HINDSIGHT_PG_DB:-}" ]; then
      _pg_db="$HINDSIGHT_PG_DB"
    else
      bench_pg_db_name "hindsight_" "hindsight_default"
      _pg_db="$_BENCH_PG_DB"
    fi
    # This idempotently creates the per-run database BEFORE the daemon URL is
    # exported. The image has no psql but ships asyncpg, a Hindsight
    # dependency. It connects to the always-present maintenance `postgres`
    # database and runs CREATE DATABASE, tolerating an already-exists error,
    # then runs CREATE EXTENSION vector/pg_trgm in the new database as
    # belt-and-suspenders over template1 inheritance (hindsight-pg-init.sql
    # seeds template1 so any new database inherits them). A connection
    # failure exits non-zero here, rather than letting the daemon boot a
    # confusing pg0 fallback. Compose health-gates hindsight-pg through
    # depends_on, so a short bounded retry here covers only a residual
    # startup race.
    log "shared-pg: ensuring per-run database ${_pg_db} on ${_pg_host}:${_pg_port}"
    # This captures the creator's output so the script can tell a FRESH
    # database ("created ...") from a REUSED one ("already exists ...").
    # hindsight_create_db.py is idempotent by design and tolerates
    # already-exists, which is good for score and summarize reruns, but is a
    # HAZARD for a fresh generate: relaunching a failed shard under the SAME
    # RUN_TAG would silently REUSE its old banks, and a consolidation-enabled
    # daemon's startup sweep then re-consolidates those orphaned banks,
    # wasting GPU time and polluting drain statistics. The `if !` form keeps
    # `set -e` from killing the script before it prints the diagnostics.
    if ! _create_out="$(HS_PG_HOST="$_pg_host" HS_PG_PORT="$_pg_port" HS_PG_USER="$_pg_user" \
        HS_PG_PASSWORD="$_pg_password" HS_PG_DB="$_pg_db" \
        python3 "$(dirname "${BASH_SOURCE[0]}")/hindsight_create_db.py" 2>&1)"; then
      printf '%s\n' "$_create_out"
      log "FATAL: per-run database creation failed for ${_pg_db}"
      exit 1
    fi
    printf '%s\n' "$_create_out"
    # This is a loud guard: a GENERATE stage, either bare `generate` or
    # `all`, both of which ingest, MUST start against a fresh per-run
    # database. If the database already exists, this aborts with an
    # actionable message unless the caller sets ALLOW_EXISTING_DB=1
    # explicitly. score and summarize reruns legitimately reuse the generate
    # run's database, so they are exempt; only generate and all are gated.
    case "${STAGE:-all}" in
      generate|all)
        if printf '%s' "$_create_out" | grep -q "already exists"; then
          if [ "${ALLOW_EXISTING_DB:-0}" = "1" ]; then
            log "WARN: per-run database ${_pg_db} already exists but ALLOW_EXISTING_DB=1 — reusing it (orphaned banks from a prior run under this RUN_TAG may be re-consolidated)."
          else
            log "FATAL: per-run database ${_pg_db} already exists for a STAGE=${STAGE:-all} run."
            log "       A prior run used RUN_TAG='${RUN_TAG:-}' (shard db ${_pg_db}). Reusing it would"
            log "       silently ingest into the old banks and let the startup consolidation sweep"
            log "       re-process orphaned banks. DROP the db (or DROP DATABASE ${_pg_db} on ${_pg_host})"
            log "       or relaunch with a NEW RUN_TAG. To override intentionally: -e ALLOW_EXISTING_DB=1."
            exit 1
          fi
        fi
        ;;
    esac
    export HINDSIGHT_EMBED_API_DATABASE_URL="postgresql://${_pg_user}:${_pg_password}@${_pg_host}:${_pg_port}/${_pg_db}"
    # This bounds each shard-daemon's asyncpg pool so N concurrent daemons
    # cannot exhaust the shared server's max_connections (=200). It respects
    # an explicit caller override, which compose passes through non-empty;
    # otherwise it defaults to 2/16.
    export HINDSIGHT_API_DB_POOL_MIN_SIZE="${HINDSIGHT_API_DB_POOL_MIN_SIZE:-2}"
    export HINDSIGHT_API_DB_POOL_MAX_SIZE="${HINDSIGHT_API_DB_POOL_MAX_SIZE:-16}"
    echo "[hindsight $(date -u +%H:%M:%S)] storage=shared pg ${_pg_host}:${_pg_port}/${_pg_db} pool=${HINDSIGHT_API_DB_POOL_MIN_SIZE}-${HINDSIGHT_API_DB_POOL_MAX_SIZE}/daemon"
    unset _pg_host _pg_port _pg_db _pg_user _pg_password _BENCH_PG_DB
    ;;
  pg0)
    # --- FEATURED ARM: per-container embedded pg0 in a faked clock domain ----
    # Hindsight 0.8.4 discards the caller's retain timestamp on the
    # update_mode="append" path: the append branch at orchestrator.py:824
    # merges the JSON array at :852 without event_date, and _build_contents
    # at :2296 then falls back to utcnow(). exchange_append is the only
    # granularity that appends, so the ftclk1_p0 smoke stamped every fact
    # 2026 against a 2022 dataset. Persona 0 measured update order
    # recognition 0.547 -> 0.305 and micro answer accuracy 0.475 -> 0.344.
    #
    # This mode gives the daemon its OWN embedded Postgres, so the daemon,
    # initdb, and the postmaster all read one faked clock — the mechanism
    # every other provider's clock-sync arm already uses. The shared
    # hindsight-pg service cannot carry a per-shard clock, and a faked daemon
    # writing 2022 rows into that co-tenant cluster would corrupt other runs.
    # Storage is therefore the only selector: pg0 without the fake reproduces
    # the wall-clock stamps, and the fake against shared damages co-tenants.
    # _preset_hindsight_featured_clocksync is the only setter of this value.
    # BENCH_CLOCKSYNC cannot select it, because hindsight_minimal_clocksync
    # sets that variable too and stays on shared.

    # HindsightEmbedded boots pg0 only when NO postgresql:// URL reaches the
    # daemon. A set HINDSIGHT_API_DATABASE_URL sends it to an external server
    # on the real clock and restores the stamps this arm exists to remove, so
    # this refuses instead of ignoring it. The empty-var guard above already
    # unset an empty one, so a value that survives to here was passed on
    # purpose.
    if [ -n "${HINDSIGHT_API_DATABASE_URL:-}" ]; then
      echo "FATAL: HINDSIGHT_PG_MODE=pg0 needs HINDSIGHT_API_DATABASE_URL unset (got '$HINDSIGHT_API_DATABASE_URL'); a URL points the daemon at an external server on the real clock" >&2
      exit 2
    fi
    # The embed manager maps HINDSIGHT_EMBED_API_DATABASE_URL onto the
    # daemon's HINDSIGHT_API_DATABASE_URL (daemon_embed_manager.py:476-480).
    # Compose sets that key for the shared arm, and the HINDSIGHT_API_ guard
    # above never matches the HINDSIGHT_EMBED_ prefix. Unsetting it is what
    # makes the manager fall back to its pg0:// default.
    unset HINDSIGHT_EMBED_API_DATABASE_URL

    case "$STAGE" in
      generate|all)
        # pg0 without the faked clock is the ftclk1_p0 defect with extra
        # steps: a private cluster still stamps mentioned_at from the wall
        # clock.
        if ! bench_clocksync_enabled; then
          echo "FATAL: HINDSIGHT_PG_MODE=pg0 needs BENCH_CLOCKSYNC=1 for a STAGE=$STAGE run; an unfaked pg0 stamps mentioned_at at wall-clock time, which is the defect this arm removes" >&2
          exit 2
        fi
        # One persona per container. The driver steps the faked clock to each
        # session date, so a second persona would rewind the clock over a
        # store that already holds the first persona's rows.
        if [ "$(( END_IDX - START_IDX ))" -ne 1 ]; then
          echo "FATAL: HINDSIGHT_PG_MODE=pg0 runs ONE persona per container (got START_IDX=$START_IDX END_IDX=$END_IDX); a persona rollover rewinds the faked clock over a live store" >&2
          exit 2
        fi
        # pg0 hardcodes its data directory to ~/.pg0/instances/<name>/data,
        # so $HOME is the only path lever. Compose mounts the SHARED
        # hindsight_state volume at /home/bench, and concurrent per-persona
        # containers cannot share one cluster directory. This moves HOME onto
        # the container's own filesystem, the choice
        # entrypoint.retaindb-server.sh makes for its cluster. pg0 extracts
        # its PostgreSQL installation under the new HOME on first start
        # (about 60 s, measured 2026-07-31 in memconflict-hindsight:latest).
        export HOME="/tmp/hs_home_${TAG}"
        if [ -d "$HOME/.pg0/instances" ]; then
          if [ "${ALLOW_EXISTING_PG0:-0}" = "1" ]; then
            log "WARN: pg0 instances already exist under $HOME/.pg0 but ALLOW_EXISTING_PG0=1 — reusing them (rows from a prior run under this RUN_TAG stay in the store)."
          else
            log "FATAL: pg0 instances already exist under $HOME/.pg0 for a STAGE=${STAGE} run."
            log "       A prior container used RUN_TAG='${RUN_TAG:-}'. Reusing that cluster would ingest"
            log "       into its old banks. Relaunch with a NEW RUN_TAG, or set -e ALLOW_EXISTING_PG0=1."
            exit 1
          fi
        fi
        mkdir -p "$HOME"
        # This seeds the timestamp file with "+0", real time, so the daemon,
        # initdb, and the postmaster all boot at the real clock. The driver
        # writes the first session date before session 1
        # (benchmark/eval_common.py calls clock_sync.set_clock per session).
        # The path is per container, never shared across shards.
        bench_clocksync_prepare /tmp/clocksync/faketime.rc
        bench_clocksync_probe
        # NO bench_clocksync_preload here, on purpose. This shell and the
        # adapter stay on the real clock: the adapter is the only writer of
        # the timestamp file and it polls the daemon with real-clock
        # timeouts. hindsight/eval_hindsight.py injects LD_PRELOAD and the
        # FAKETIME_* contract into os.environ immediately before
        # HindsightEmbedded(...), and the vendor spawns the daemon with
        # env=os.environ.copy(), so the daemon and the pg0 postmaster it
        # orphans inherit the fake.
        #
        # 0 disables the daemon's idle auto-exit: hindsight_api/daemon.py:58-59
        # returns before the checker loop when idle_timeout <= 0. That loop
        # compares time.time(), which libfaketime fakes, so a faked forward
        # jump of weeks between sessions would otherwise read as idleness and
        # kill the daemon mid-run. Do NOT declare this key in compose. The
        # manager int-parses it (daemon_embed_manager.py:495) and the
        # empty-var guard above matches HINDSIGHT_API_ only, so a compose-set
        # empty value would reach that parse and crash the boot.
        export HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT="${HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT:-0}"
        ;;
    esac
    # These match the shared branch's caps, so the daemon's connection
    # profile is the same in both arms and storage stays the only difference
    # under test. Here they bound one daemon against this container's own
    # cluster, not N daemons against a shared server.
    export HINDSIGHT_API_DB_POOL_MIN_SIZE="${HINDSIGHT_API_DB_POOL_MIN_SIZE:-2}"
    export HINDSIGHT_API_DB_POOL_MAX_SIZE="${HINDSIGHT_API_DB_POOL_MAX_SIZE:-16}"
    # write_manifest.py maps hindsight to temporal_capability=native, which
    # is true for the minimal arm only. This branch is the one that knows a
    # faked daemon plus its own postmaster carried logical time, so it
    # declares the mechanism. The score stage writes a manifest too, so this
    # export sits outside the generate gate.
    export BENCH_TEMPORAL_CAPABILITY="controlled_process_clock+postgres"
    echo "[hindsight $(date -u +%H:%M:%S)] storage=embedded pg0 home=${HOME} clocksync=${BENCH_CLOCKSYNC:-0} idle_timeout=${HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT:-unset} pool=${HINDSIGHT_API_DB_POOL_MIN_SIZE}-${HINDSIGHT_API_DB_POOL_MAX_SIZE}/daemon"
    ;;
  *)
    echo "FATAL: HINDSIGHT_PG_MODE=$HINDSIGHT_PG_MODE unsupported (local/embedded modes removed 2026-07-22; git history has them)" >&2
    exit 2
    ;;
esac

# --- Embedding source selection (shared vllm-embed only) ----------------------
# HINDSIGHT_EMBED_SOURCE=remote is the only supported mode. It points the
# daemon's embedding provider at the shared vllm-embed service over the
# compose network, instead of loading the model in-process. hindsight_api's
# factory (create_embeddings_from_env,
# hindsight_api/engine/embeddings.py:1583-1620) hard-dispatches on
# HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai and returns OpenAIEmbeddings
# WITHOUT ever constructing LocalSTEmbeddings, so the local bge weights are
# never loaded. This is the RAM win. The local ms-marco RERANKER, a
# cross-encoder vllm-embed cannot serve, stays local and keeps the torch and
# sentence-transformers runtime resident, so the reclaimed slice is the
# embedding model weights only, not the whole ML stack. vllm-embed serves the
# contract embedder (gte-modernbert-base, dim 768), so this is a
# serving-envelope change, not a model change. The dimension is
# auto-detected from a probe embedding at daemon startup.
#
# IMPORTANT: do NOT set HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS. Leaving it
# unset, the default None, omits the field from the request, and auto-detect
# yields 768. This runs AFTER the empty-HINDSIGHT_API_* guard, so these
# exports survive.
#
# The team removed the legacy per-daemon in-process sentence-transformers
# mode (HINDSIGHT_EMBED_SOURCE=local) on 2026-07-22. Every real run uses
# remote; git history has the branch.
HINDSIGHT_EMBED_SOURCE="${HINDSIGHT_EMBED_SOURCE:-remote}"
case "$HINDSIGHT_EMBED_SOURCE" in
  remote)
    export HINDSIGHT_API_EMBEDDINGS_PROVIDER="openai"
    export HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL="${HINDSIGHT_EMBED_BASE_URL:-http://vllm-embed:8000/v1}"
    export HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL="${HINDSIGHT_EMBED_MODEL:-gte-modernbert-base}"
    # vLLM ignores this key, but the OpenAI SDK requires a non-empty one.
    export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY="${HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY:-local-vllm}"
    echo "[hindsight $(date -u +%H:%M:%S)] embeddings=remote openai ${HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL} model=${HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL}"
    ;;
  *)
    echo "FATAL: HINDSIGHT_EMBED_SOURCE=$HINDSIGHT_EMBED_SOURCE unsupported (local/embedded modes removed 2026-07-22; git history has them)" >&2
    exit 2
    ;;
esac

# --- Reranker source selection (shared TEI cross-encoder only) ----------------
# HINDSIGHT_RERANK_SOURCE=remote is the only supported mode. It points the
# daemon's reranker at a shared HuggingFace text-embeddings-inference (TEI)
# server, which serves the same cross-encoder over its /rerank endpoint.
# hindsight_api's factory (create_cross_encoder_from_env,
# hindsight_api/engine/cross_encoder.py:1617-1642) hard-dispatches on
# HINDSIGHT_API_RERANKER_PROVIDER=tei and returns RemoteTEICrossEncoder
# WITHOUT constructing LocalSTCrossEncoder.
#
# WHY THIS WAS THE BIG RAM LEVER, kept for context even with local removed:
# torch and sentence-transformers, about 430MB resident, load ONLY when
# embeddings OR the reranker is local (config.py:2225 gates a
# sentence_transformers import on `provider == "local"`; every torch import
# in the package is lazy and provider-gated). The team verified that with
# BOTH providers remote, importing the full memory_engine leaves torch and
# sentence_transformers UNloaded, and the default DateparserQueryAnalyzer is
# pure Python. With both providers now always remote, the full per-daemon
# ML-stack win, about 0.6GB static and more under load, is unconditional.
#
# The TEI server does NOT start automatically. Before a run, bring it up
# first with `docker compose up -d hindsight-rerank` (it is not in
# hindsight's depends_on). This runs AFTER the empty-HINDSIGHT_API_* guard,
# so these exports survive.
#
# The team removed the legacy per-daemon local ms-marco cross-encoder mode
# (HINDSIGHT_RERANK_SOURCE=local) on 2026-07-22. Every real run uses remote;
# git history has the branch.
HINDSIGHT_RERANK_SOURCE="${HINDSIGHT_RERANK_SOURCE:-remote}"
case "$HINDSIGHT_RERANK_SOURCE" in
  remote)
    export HINDSIGHT_API_RERANKER_PROVIDER="tei"
    export HINDSIGHT_API_RERANKER_TEI_URL="${HINDSIGHT_RERANK_TEI_URL:-http://hindsight-rerank:80}"
    echo "[hindsight $(date -u +%H:%M:%S)] reranker=remote tei ${HINDSIGHT_API_RERANKER_TEI_URL}"
    ;;
  *)
    echo "FATAL: HINDSIGHT_RERANK_SOURCE=$HINDSIGHT_RERANK_SOURCE unsupported (local/embedded modes removed 2026-07-22; git history has them)" >&2
    exit 2
    ;;
esac

# This sets the answer and judge LLM for MemConflict's llm_request.
# OPENAI_BASE_URL and OPENAI_MODEL come from compose. The SDK needs a
# non-empty key even for vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"
# This sets the Hindsight-internal LLM for retain fact-extraction and
# recall. The provider, base, and model come from compose. The key mirrors
# the answer LLM's key.
export HINDSIGHT_LLM_PROVIDER="${HINDSIGHT_LLM_PROVIDER:-openai}"
export HINDSIGHT_LLM_API_KEY="${HINDSIGHT_LLM_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"

RESDIR="$ROOT/hindsight/Results"
SCOREDIR="$ROOT/hindsight/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/hindsight_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"
SCORES_FILE="${SCORES_FILE:-$SCOREDIR/hindsight_${TAG}_eval_scores.jsonl}"
CHECKPOINT="${CHECKPOINT:-$SCOREDIR/${TAG}_judged_checkpoint.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$SCOREDIR/summary_${TAG}.json}"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]              && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")

# These are the Arm B/C feature flags. If unset, this script passes nothing,
# so behavior is identical to today, Arm A.
[ "${PREFER_OBSERVATIONS:-0}" = "1" ]   && CAPS+=(--prefer_observations)
[ "${WAIT_CONSOLIDATION:-0}" = "1" ]    && CAPS+=(--wait_consolidation)
[ -n "${RETAIN_GRANULARITY:-}" ]        && CAPS+=(--retain_granularity "$RETAIN_GRANULARITY")
# RECALL_TYPES is a comma-separated recall() fact-type filter (world,
# experience, or observation). If UNSET, this script passes no flag, so
# there is no filter and all three types apply. That is exactly how arms A
# and B were run, so this variable is inert for them.
#
# ARM C sets RECALL_TYPES=observation because that is what the real Hermes
# plugin does. NousResearch/hermes-agent
# plugins/memory/hindsight/__init__.py at 977884e6 initializes
# `self._recall_types: list[str] = ["observation"]` and passes it as
# recall_kwargs["types"]. So a production Hermes agent is only ever handed
# the consolidated observation layer, never the raw world and experience
# facts underneath it. Arm C therefore also needs WAIT_CONSOLIDATION=1,
# because observations do not exist until consolidation drains, and an
# observation-only recall before then legitimately returns an empty result
# set.
#
# NOTE this variable is UNPREFIXED, like RETAIN_GRANULARITY and
# PREFER_OBSERVATIONS, so write_manifest.py's HINDSIGHT_ prefix match does
# not cover it. It must be listed in that script's _ENV_EXACT set, or the
# run manifest will not record which recall surface produced a Results
# file.
[ -n "${RECALL_TYPES:-}" ]              && CAPS+=(--recall_types "$RECALL_TYPES")
# STRICT_QUALITY_RUN=1, for the headline Arm-B/C only, makes the adapter
# ABORT nonzero on any silent-degradation path: a consolidation drain
# timeout or poll failure, an exchange_append fallback off the append path,
# or Append_Mode != "append". If unset, this script passes no flag, so
# today's tolerant log-and-continue behavior applies (exploratory arms). See
# the --strict_quality_run help in eval_hindsight.py.
[ "${STRICT_QUALITY_RUN:-0}" = "1" ]    && CAPS+=(--strict_quality_run)
# PLUGIN_NATIVE_RECALL=1, for FEATURED Arm C, emits every recalled item to
# the answer LLM and Retrieved_Memories instead of the top_k slice. The real
# Hermes hindsight plugin injects its full token-budgeted recall result, not
# a top-K slice. If unset, this script passes no flag, so the historical
# top_k slice applies (Minimal Arm A). This variable is UNPREFIXED, like
# RECALL_TYPES, so it must be in write_manifest.py's _ENV_EXACT set to be
# recorded.
[ "${PLUGIN_NATIVE_RECALL:-0}" = "1" ]  && CAPS+=(--plugin_native_k)
# This is a probe or alternate dataset override, for example
# benchmark/probes/. The default is Step4_4.
[ -n "${INPUT_JSONL:-}" ]               && CAPS+=(--input_jsonl_path "$INPUT_JSONL")
# This is the per-session consolidation drain-wait timeout, meaningful only
# with WAIT_CONSOLIDATION=1. If unset, the adapter default applies (450s,
# per the Arm-B timeout diagnosis). This is plumbed here because the
# drain-wait routinely exceeds the old 300s default under load; see
# docs/TROUBLESHOOTING.md.
[ -n "${CONSOLIDATION_WAIT_TIMEOUT_S:-}" ] && CAPS+=(--consolidation_wait_timeout_s "$CONSOLIDATION_WAIT_TIMEOUT_S")

# bench_hs_pg0_report prints the evidence that the pg0 store holds LOGICAL
# dates: the installed extensions, and the mentioned_at range per fact type.
# Every featured shard log then carries its own proof, so a repeat of the
# ftclk1_p0 wall-clock stamping shows up in the log instead of only in a
# score drop weeks later. It runs AFTER the adapter wrote its Results file,
# so an unreadable cluster must never discard a finished run: every step here
# is best effort and the function always returns 0.
bench_hs_pg0_report() {
  [ "$HINDSIGHT_PG_MODE" = "pg0" ] || return 0
  local psql="" p uri=""
  # psql ships inside pg0's extracted PostgreSQL installation and is NOT on
  # PATH (verified 2026-07-31 in memconflict-hindsight:latest:
  # $HOME/.pg0/installation/18.1.0/bin/psql).
  for p in "$HOME"/.pg0/installation/*/bin/psql; do
    if [ -x "$p" ]; then psql="$p"; break; fi
  done
  if [ -z "$psql" ]; then
    log "pg0-report: no psql under $HOME/.pg0/installation/*/bin — skipped"
    return 0
  fi
  # pg0 assigns the port itself, so the URI has to come from pg0. Its Python
  # SDK returns InstanceInfo(uri=...) per instance. The daemon names its
  # instance hindsight-embed-<profile> (daemon_embed_manager.py:133), so this
  # takes the first RUNNING instance instead of guessing that name.
  uri="$(python - <<'PY' 2>/dev/null
import pg0
for inst in pg0.list_instances():
    if inst.running and inst.uri:
        print(inst.uri)
        break
PY
)" || uri=""
  if [ -z "$uri" ]; then
    # Fallback for a stopped instance. User, password, and database are all
    # "hindsight" (hindsight_api/pg0.py:11-13); the port is pg0's default and
    # is a guess, because a running instance may have been auto-assigned
    # another one.
    uri="postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight"
    log "pg0-report: pg0 reports no running instance — trying $uri"
  fi
  log "pg0-report: extensions"
  "$psql" "$uri" -At -c \
    "SELECT extname, extversion FROM pg_extension ORDER BY extname;" 2>&1 \
    | sed 's/^/[pg0-report] /' || true
  log "pg0-report: mentioned_at range per fact type (dataset-year dates prove logical time reached the store)"
  "$psql" "$uri" -At -c \
    "SELECT fact_type, count(*), min(mentioned_at), max(mentioned_at) FROM memory_units GROUP BY fact_type ORDER BY fact_type;" 2>&1 \
    | sed 's/^/[pg0-report] /' || true
  return 0
}

do_generate() {
  # This sets the canonical ANSWER decoding, identical across providers, and
  # writes the best-effort manifest.
  bench_answer_env
  # This captures the serving envelope, the manifest with the run-contract
  # hash, and the token-accounting start. It aborts under
  # STRICT_RUN_CONTRACT=1 or BENCH_CLOCKSYNC=1 if the contract is
  # incomplete, and warns otherwise.
  bench_generate_preamble "$ROOT/hindsight" "$TAG"
  log "GENERATE personas=[$START_IDX,$END_IDX) top_k=$TOPK thinking=${MEMCONFLICT_ENABLE_THINKING} answer_max_tokens=${OPENAI_MAX_TOKENS} -> $RESULTS_FILE"
  python -u "$ROOT/hindsight/eval_hindsight.py" \
      --start_idx "$START_IDX" --end_idx "$END_IDX" --top_k "$TOPK" "${CAPS[@]}" \
      --output_jsonl_path "$RESULTS_FILE" \
      --output_json_path "$RESULTS_JSON"
  # This is a no-op on every arm except HINDSIGHT_PG_MODE=pg0.
  bench_hs_pg0_report || true
  bench_tokens_finish "$ROOT/hindsight" "$TAG"
}

# The shared judge env, score call, and score-stage manifest, written inside
# run_score AFTER bench_judge_env so it records the judge decoding, and the
# shared summarize call now both live in answer_env.sh's run_stage.
run_stage "$ROOT/hindsight" "$TAG" do_generate
