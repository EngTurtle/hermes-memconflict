#!/usr/bin/env bash
# This is the shared sharded-run launcher (Docker) for every provider. It fans
# a full run out across NUM_SHARDS detached `docker compose run` containers,
# each covering a contiguous persona range. This is the ONE code path that
# replaces the three per-provider run_full_docker.sh scripts (hindsight, mem0,
# supermemory). So the next wave (mem0, supermemory, retaindb_server) shares
# a single loop.
#
# Usage (from anywhere):
#   benchmark/docker/run_shards.sh <provider> <run_tag> [extra -e args...]
# Providers: hindsight | mem0 | supermemory | retaindb_server | honcho | openviking
# Examples:
#   benchmark/docker/run_shards.sh hindsight full                 # Arm A (10 shards)
#   benchmark/docker/run_shards.sh hindsight armB \
#     -e HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=true \
#     -e PREFER_OBSERVATIONS=1 -e WAIT_CONSOLIDATION=1 \
#     -e CONSOLIDATION_WAIT_TIMEOUT_S=450 -e STRICT_QUALITY_RUN=1   # Arm B (auto -> 5 shards)
#   benchmark/docker/run_shards.sh mem0 full                       # batch arm (default)
#   benchmark/docker/run_shards.sh mem0 sess -e RETAIN_GRANULARITY=session
#   benchmark/docker/run_shards.sh supermemory full                # hybrid, session ingest
#   SUPERMEMORY_TWO_SERVERS=1 SUPERMEMORY_INGEST_CONCURRENCY=15 NUM_SHARDS=12 \
#     benchmark/docker/run_shards.sh supermemory full2srv          # 12 shards, 6 per server
#   benchmark/docker/run_shards.sh retaindb_server full            # UNTESTED, see below
#   benchmark/docker/run_shards.sh honcho full                     # shared honcho-api/-deriver, hybrid recall
#   PRESET=honcho_minimal_clocksync benchmark/docker/run_shards.sh honcho minc  # per-shard spawn
#   benchmark/docker/run_shards.sh openviking full                # per-shard spawned server
#
# DRY_RUN=1 prints the docker command(s) it would run (pre-step plus every
# shard) instead of running them. Nothing launches. Use it to diff this
# launcher against the old per-provider scripts.
#
# Containers are named <prefix>_<tag>_s<k> (range mode) or <prefix>_<tag>_p<i>
# (per-persona mode, see PERSONA_CONTAINERS below). They stay KEPT on exit (no
# --rm) so logs survive. Remove them once the run is merged and scored. Each
# shard runs STAGE=generate only. Merge the shard result JSONL, then score and
# summarize in a single post-merge pass, for example (use `_p*` instead of
# `_s*` after a per-persona wave):
#   cat <provider>/Results/<file>_<tag>_s*.jsonl > <provider>/Results/<file>_<tag>.jsonl  # verify 30 lines
#   docker compose run -d --rm -e STAGE=score     -e RUN_TAG=<tag> -e NUM_PERSONAS=30 -e SCORE_WORKERS=16 <service>
#   docker compose run -d --rm -e STAGE=summarize -e RUN_TAG=<tag> <service>
#
# GIT_SHA: the containers have no .git, so write_manifest.py records
# head_sha:null unless GIT_SHA is exported before `docker compose run`
# (CLAUDE.md: the serving checkpoint is recoverable ONLY from the compose file
# at the run's SHA). This launcher exports GIT_SHA for EVERY provider. The old
# mem0/run_full_docker.sh did NOT export it, so its manifests recorded null.
# The shared path now stamps it, which is a strictly additive provenance fix,
# never a removal. This launcher also writes benchmark/.git_sha as the
# fallback for hand-launched containers; see stamp_git_sha.sh.
#
# PRESET=<name> is forwarded to every shard (benchmark/docker/presets.sh), as
# are STRICT_RUN_CONTRACT and the host-only serving image identity
# (BENCH_SERVING_IMAGE*), which a container cannot read for itself.
#
# TOKEN ACCOUNTING: this launcher snapshots vLLM's Prometheus counters before
# the shards launch, and again once every shard container has exited, into
# <provider>/Results/token_usage_<tag>.json (scope=run). That file is the only
# place provider-INTERNAL LLM spend (extraction, consolidation, memory agents)
# is visible; the harness never sees those calls directly. The end snapshot
# runs in a DETACHED waiter (this launcher returns immediately, as it always
# has), and accounting never affects a run's success. BENCH_TOKEN_ACCOUNTING=0
# disables it.
set -euo pipefail

PROVIDER="${1:?usage: run_shards.sh <provider> <run_tag> [extra -e args...]}"
shift
TAG="${1:?usage: run_shards.sh <provider> <run_tag> [extra -e args...]}"
shift || true

NUM_PERSONAS="${NUM_PERSONAS:-30}"

# A *_clocksync PRESET implies BENCH_CLOCKSYNC=1 for the LAUNCHER too. The
# per-provider branches below choose the run TOPOLOGY from BENCH_CLOCKSYNC
# (supermemory: per-shard spawned servers plus --no-deps; retaindb_server:
# per-shard internal Postgres; mem0: forward the flag). The entrypoints' own
# presets only run INSIDE the containers, which is too late to change the
# topology. Without this check, a
# `PRESET=supermemory_minimal_clocksync run_shards.sh supermemory <tag>` would
# launch shared-mode shards that each fail fatally on the shared+clocksync guard.
if [ -z "${BENCH_CLOCKSYNC:-}" ] && [ -n "${PRESET:-}" ]; then
  case "$PRESET" in
    *_clocksync)
      export BENCH_CLOCKSYNC=1
      echo "[run_shards] PRESET=$PRESET implies BENCH_CLOCKSYNC=1 (launcher topology)"
      ;;
  esac
fi

# Per-provider deltas: compose service name, container-name prefix, default
# shard count, any extra -e passthrough baked in, and any pre-step. Everything
# else is the shared loop below.
EXTRA_ENV=()          # extra -e args this provider always passes (before "$@")
PRESTEP=""            # provider-specific setup run once before the shard loop
NODEPS=0              # 1 = launch shards with --no-deps (clock-sync arms that
                      # must NOT drag a shared central service up via depends_on)
case "$PROVIDER" in
  hindsight)
    SERVICE=hindsight; PREFIX=hs
    # --- Clock-sync featured arm (HINDSIGHT_PG_MODE=pg0): per-container pg0 ---
    # The featured arm retains with update_mode="append", the one path whose
    # dates come from the DB clock, so it runs an embedded pg0 cluster inside
    # the container's faked clock domain instead of the shared hindsight-pg
    # service (one postmaster, co-tenant shards, one perceived clock). This
    # mirrors the retaindb_server per-shard-Postgres branch below.
    # `HINDSIGHT_PG_MODE=pg0` selects it; PRESET=hindsight_featured_clocksync is
    # accepted as the same request, because that preset sets the var INSIDE the
    # container, which is too late to change the launcher topology.
    _HS_PG0=0
    if [ "${HINDSIGHT_PG_MODE:-}" = "pg0" ] || [ "${PRESET:-}" = "hindsight_featured_clocksync" ]; then
      _HS_PG0=1
    fi
    if [ "$_HS_PG0" = "1" ]; then
      # BENCH_CLOCKSYNC is exported (not just forwarded) so PERSONA_CONTAINERS
      # below defaults to 1. The entrypoint's pg0 branch exits 2 on a container
      # covering more than one persona, so a range-mode launch would fail every
      # shard. The *_clocksync PRESET check above already exports it for the
      # preset launch; an explicit HINDSIGHT_PG_MODE=pg0 with no preset does not
      # reach that check. Re-exporting the same value 1 is a no-op.
      export BENCH_CLOCKSYNC=1
      EXTRA_ENV=(-e BENCH_CLOCKSYNC=1 -e HINDSIGHT_PG_MODE=pg0)
      PRESTEP="docker compose up -d --wait vllm-gen vllm-embed"
      # --no-deps keeps depends_on from dragging hindsight-pg up for this arm.
      # hindsight-rerank is NOT in depends_on and never starts on its own:
      # `docker compose up -d hindsight-rerank` before the wave.
      NODEPS=1
      # RAM-bound, not vllm-gen-bound: each container carries the ~1.9 GB daemon
      # plus its own 0.3-0.4 GB postmaster, about 2.5 GB per container. An
      # explicit NUM_SHARDS always wins.
      DEFAULT_SHARDS="${NUM_SHARDS:-4}"
    # The shard-count default depends on the arm. Consolidation arms default
    # to 5, because 10-shard concurrent consolidation saturates vllm-gen (see
    # docs/TROUBLESHOOTING.md). Every other arm defaults to 10. An explicit
    # NUM_SHARDS always wins.
    elif [ -z "${NUM_SHARDS:-}" ]; then
      if printf '%s\n' "$@" | grep -q 'HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=true'; then
        DEFAULT_SHARDS=5
      else
        DEFAULT_SHARDS=10
      fi
    else
      DEFAULT_SHARDS="$NUM_SHARDS"
    fi
    ;;
  mem0)
    SERVICE=mem0; PREFIX=mem0; DEFAULT_SHARDS=6
    # MAX_SESSIONS= clears the compose smoke cap (6), so a full run ingests
    # every session. A later -e MAX_SESSIONS=<n> in "$@" still wins (docker:
    # last -e wins).
    EXTRA_ENV=(-e MAX_SESSIONS=)
    # --- Clock-sync arm (BENCH_CLOCKSYNC=1): forward the flag into each shard ---
    # Unlike supermemory and retaindb, which need per-shard SPAWN because a
    # central server cannot serve N shards at different timeline points, mem0
    # needs NO special topology. Each shard is already its OWN container, and
    # fakes its OWN mem0 python through the entrypoint (container-local
    # /tmp/clocksync/faketime.rc). The shared central qdrant is NOT clock-faked
    # and needs no faking: mem0 recall is pure vector cosine similarity with no
    # recall-time temporal ranking. So a real-clock qdrant is correct, and every
    # shard keeps sharing it (no per-shard collision, no --no-deps; depends_on
    # brings it up normally). BENCH_CLOCKSYNC is not in the mem0 compose
    # environment block, so this forwards it explicitly.
    if [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; then
      EXTRA_ENV+=(-e BENCH_CLOCKSYNC=1)
    fi
    ;;
  supermemory)
    SERVICE=supermemory; PREFIX=sm; DEFAULT_SHARDS=6
    # Shards ATTACH to ONE central server (the hindsight-pg analog).
    EXTRA_ENV=(-e SUPERMEMORY_SERVER_MODE=shared)
    # Bring the central server up and WAIT for it to become healthy. Its
    # healthcheck gates on the published bearer key plus a live API, so shards
    # never race the key handoff.
    PRESTEP="docker compose up -d --wait supermemory-server"
    # --- Optional TWO-SERVER topology (SUPERMEMORY_TWO_SERVERS=1) --------------
    # The single central server gets connection-reset under multi-shard ingest
    # load, and is one memory pool for every shard. Splitting the shards across
    # two independent servers halves per-server connection concurrency and
    # doubles ingest throughput. Target profile: 12 shards, 6 per server. The
    # prestep brings BOTH servers up with --wait. The shard loop routes the
    # first half to server A and the second half to server B
    # (SUPERMEMORY_ATTACH_URL plus SUPERMEMORY_KEY_FILE per shard). Personas
    # still use the existing contiguous START_IDX/END_IDX split, so each
    # persona ingests AND recalls on ONE server (disjoint halves), with no
    # cross-server recall. Rows merge at the JSONL level, like Hindsight's
    # per-shard DBs. Host RAM is tight (about an 8 GiB vLLM floor in a 23.6 GiB
    # VM), so this caps each server at about 4 GiB: default
    # SUPERMEMORY_EMBEDDING_RAM_LIMIT=4gb (this reaches BOTH servers through
    # the shared var; an explicit value still wins). Leaving the flag unset
    # keeps today's single-server path untouched.
    SM_TWO="${SUPERMEMORY_TWO_SERVERS:-0}"
    if [ "$SM_TWO" = "1" ]; then
      DEFAULT_SHARDS=12
      export SUPERMEMORY_EMBEDDING_RAM_LIMIT="${SUPERMEMORY_EMBEDDING_RAM_LIMIT:-4gb}"
      PRESTEP="docker compose up -d --wait supermemory-server supermemory-server-b"
    fi
    # THE SERVER CAPS THROUGHPUT, NOT NUM_SHARDS. Every shard posts into the
    # ONE central server, whose ingest worker pool is
    # SUPERMEMORY_INGEST_CONCURRENCY (compose default 10). Raising NUM_SHARDS
    # past that just queues documents behind the same pool. Preferred profile
    # (user, 2026-07-23): 10 shard containers against a server pool of 15:
    #   SUPERMEMORY_INGEST_CONCURRENCY=15 NUM_SHARDS=10 run_shards.sh supermemory <tag>
    # Fewer containers than pool workers means the pool is never the
    # bottleneck (some headroom), and fewer concurrent client connections
    # reach the one server. That matters: at 15 shards, the server reset one
    # shard's connection mid-run (v4min, one shard lost). Do NOT push
    # NUM_SHARDS above the pool size. The extra containers only add
    # connection pressure for no throughput gain. The pre-step only STARTS an
    # already-running server; it does not recreate it. So a concurrency
    # change needs the server stopped and removed first. Also, because
    # per-run isolation works by containerTag namespace rather than a per-run
    # DB, re-running an existing RUN_TAG appends to the previous attempt's
    # namespaces. Wipe docker_supermemory_data AND docker_supermemory_shared
    # together, or use a fresh RUN_TAG.
    # --- Clock-sync arm (BENCH_CLOCKSYNC=1): per-shard SPAWN topology --------
    # A central shared server cannot serve N shards at different timeline
    # points. So each shard spawns its OWN server, with libfaketime injected
    # into the server child env (see supermemory/_supermemory_server.py).
    # Shards run --no-deps, so depends_on does not drag the central server up.
    if [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; then
      if [ "$SM_TWO" = "1" ]; then
        echo "run_shards.sh: BENCH_CLOCKSYNC=1 is incompatible with" \
             "SUPERMEMORY_TWO_SERVERS=1 (spawn mode has no central servers)" >&2
        exit 2
      fi
      # This spawns N servers, one per shard, and caps each server's embedding
      # RAM well below the shared-server default (host RAM has about an 8 GiB
      # vLLM floor). SUPERMEMORY_INGEST_CONCURRENCY is declared ONLY on the
      # two central supermemory-server services. So before this line it never
      # reached a spawned server, and spawn mode always ran at the vendor
      # default of 2. _supermemory_server.py builds the child env from
      # dict(os.environ), so forwarding the var here does reach the binary.
      # The value is PER SERVER: the run's total concurrent extraction load is
      # this value multiplied by NUM_SHARDS.
      EXTRA_ENV=(-e SUPERMEMORY_SERVER_MODE=spawn -e BENCH_CLOCKSYNC=1
                 -e SUPERMEMORY_EMBEDDING_RAM_LIMIT="${SUPERMEMORY_EMBEDDING_RAM_LIMIT:-2gb}"
                 -e SUPERMEMORY_INGEST_CONCURRENCY="${SUPERMEMORY_INGEST_CONCURRENCY:-2}")
      PRESTEP="docker compose up -d --wait vllm-gen"
      NODEPS=1
    fi
    ;;
  retaindb_server)
    # UNTESTED until retaindb_server's first full run. Sharding follows
    # hindsight's model: each shard is its own container, server, and db,
    # split by START_IDX/END_IDX with a distinct RUN_TAG, all on the SHARED
    # hindsight-pg service (the service's depends_on pulls it up, so no
    # pre-step is needed). The compose service is `retaindb-server`.
    SERVICE=retaindb-server; PREFIX=rdbs; DEFAULT_SHARDS=6
    # --- Clock-sync arm (BENCH_CLOCKSYNC=1): per-shard INTERNAL Postgres -----
    # The shared hindsight-pg cluster cannot be clock-faked per shard (one
    # postmaster, co-tenant providers). So each shard runs its own Postgres
    # inside the shard container's clock domain (see
    # entrypoint.retaindb-server.sh). --no-deps keeps depends_on from dragging
    # hindsight-pg up for this arm.
    if [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; then
      EXTRA_ENV=(-e BENCH_CLOCKSYNC=1)
      PRESTEP="docker compose up -d --wait vllm-gen vllm-embed"
      NODEPS=1
    fi
    ;;
  honcho)
    # Shards ATTACH to ONE central honcho-api/-deriver pair (the
    # hindsight-pg / supermemory-server analog: one shared Postgres-backed
    # server serves every shard, personas isolated by Honcho workspace).
    SERVICE=honcho; PREFIX=hc; DEFAULT_SHARDS=6
    EXTRA_ENV=(-e HONCHO_SERVER_MODE=shared)
    # Bring the central services up and WAIT for honcho-api's healthcheck.
    # honcho-api's own depends_on (compose file) already gates it behind
    # honcho-pg and the one-shot honcho-db-init (provision + vector-dimension
    # fix), so listing it here is enough to pull in the whole chain.
    PRESTEP="docker compose up -d --wait honcho-pg honcho-api honcho-deriver"
    # --- Clock-sync arm (BENCH_CLOCKSYNC=1): per-shard SPAWN topology --------
    # A shared honcho-api/-deriver pair has ONE perceived clock, but each
    # shard drives a different persona/session timeline, so it cannot sit at
    # N logical dates at once (same reasoning as Supermemory and
    # retaindb_server). Each shard instead spawns its OWN
    # server+deriver(+Postgres) inside its own container, under a libfaketime
    # LD_PRELOAD the adapter controls (honcho/_honcho_server.py). --no-deps
    # keeps depends_on from dragging the central honcho-pg/-api/-deriver up
    # for this arm.
    if [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; then
      EXTRA_ENV=(-e HONCHO_SERVER_MODE=spawn -e BENCH_CLOCKSYNC=1)
      PRESTEP="docker compose up -d --wait vllm-gen vllm-embed"
      NODEPS=1
    fi
    ;;
  openviking)
    # Every shard spawns its OWN server, always. OpenViking keeps its content
    # store and vector index in one local workspace directory, a workspace
    # holds a one-process `.openviking.pid` lock, and no compose service
    # exists to attach to, so there is no shared-backing-service topology to
    # choose (unlike hindsight-pg, qdrant, supermemory-server, or honcho-api).
    SERVICE=openviking; PREFIX=ovk; DEFAULT_SHARDS=6
    EXTRA_ENV=(-e OPENVIKING_SERVER_MODE=spawn)
    PRESTEP="docker compose up -d --wait vllm-gen vllm-embed"
    # --- Clock-sync arm (BENCH_CLOCKSYNC=1) ---------------------------------
    # The spawned server child runs under a libfaketime LD_PRELOAD the adapter
    # injects (openviking/_openviking_server.py). --no-deps keeps depends_on
    # from re-resolving the vLLM services the pre-step already waited on, the
    # same shard shape as honcho's spawn branch.
    if [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; then
      EXTRA_ENV+=(-e BENCH_CLOCKSYNC=1)
      NODEPS=1
    fi
    ;;
  *)
    echo "run_shards.sh: unknown provider '$PROVIDER'" \
         "(expected hindsight|mem0|supermemory|retaindb_server|honcho|openviking)" >&2
    exit 2
    ;;
esac
NUM_SHARDS="${NUM_SHARDS:-$DEFAULT_SHARDS}"

COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmark/docker
REPO_ROOT="$(cd "$COMPOSE_DIR/../.." && pwd)"
# This stamps the launching repo commit into every shard's manifest, on a
# best-effort basis: outside a git checkout this is empty, and the manifest
# records git_sha_env: null.
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
export GIT_SHA
# The same SHA also goes into the bind-mounted stamp file, so a container
# launched WITHOUT this launcher (a hand-run `docker compose run`, a re-score)
# still resolves a code SHA. This is an untracked run artifact, like the
# Results/Scores files. Never commit it.
if [ -n "$GIT_SHA" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  printf '%s\n' "$GIT_SHA" > "$REPO_ROOT/benchmark/.git_sha" 2>/dev/null || true
fi

# --- host-only serving identity ------------------------------------------------
# A provider container can read the served ALIAS (OPENAI_MODEL) but not the
# image that serves it, and the alias stayed IDENTICAL across contracts
# v2/v3/v4 while the checkpoint changed twice. This reads the running
# vllm-gen/-embed image and digest here and forwards them.
# capture_serving_envelope.py records them in the sidecar, and
# write_manifest.py lifts them into the required run contract.
_service_image() {   # $1 = compose service; prints "image<TAB>repo-digest"
  local cid img dig
  cid="$( (cd "$COMPOSE_DIR" && docker compose ps -q "$1" 2>/dev/null) | head -1 )"
  [ -n "$cid" ] || return 0
  # .RepoDigests lives on the IMAGE, not the container. Inspecting a container
  # with that template errors out; `|| true` swallows the error, and every
  # shard would get an empty BENCH_SERVING_IMAGE*. So this resolves the image
  # reference from the container first, then inspects that image for its digest.
  img="$(docker inspect --format '{{.Config.Image}}' "$cid" 2>/dev/null || true)"
  [ -n "$img" ] || return 0
  dig="$(docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' \
      "$img" 2>/dev/null || true)"
  printf '%s\t%s\n' "$img" "$dig"
}
_gen_img_line="$(_service_image vllm-gen || true)"
_embed_img_line="$(_service_image vllm-embed || true)"
export BENCH_SERVING_IMAGE="${BENCH_SERVING_IMAGE:-$(printf '%s' "$_gen_img_line" | cut -f1)}"
export BENCH_SERVING_IMAGE_DIGEST="${BENCH_SERVING_IMAGE_DIGEST:-$(printf '%s' "$_gen_img_line" | cut -f2)}"
export BENCH_SERVING_EMBED_IMAGE="${BENCH_SERVING_EMBED_IMAGE:-$(printf '%s' "$_embed_img_line" | cut -f1)}"
export BENCH_SERVING_EMBED_IMAGE_DIGEST="${BENCH_SERVING_EMBED_IMAGE_DIGEST:-$(printf '%s' "$_embed_img_line" | cut -f2)}"

# This env reaches every provider's shards, ahead of the provider-specific
# EXTRA_ENV and the caller's "$@" (docker: last -e wins, so both still
# override these).
COMMON_ENV=()
for _v in BENCH_SERVING_IMAGE BENCH_SERVING_IMAGE_DIGEST \
          BENCH_SERVING_EMBED_IMAGE BENCH_SERVING_EMBED_IMAGE_DIGEST; do
  # This forwards only what was actually resolved. An empty -e VAR= would
  # land in the manifest's env snapshot as a present-but-empty value, which
  # reads like a configured blank rather than "the launcher could not see
  # the image".
  [ -n "${!_v:-}" ] && COMMON_ENV+=(-e "$_v=${!_v}")
done
[ -n "${PRESET:-}" ]               && COMMON_ENV+=(-e PRESET="$PRESET")
[ -n "${STRICT_RUN_CONTRACT:-}" ]  && COMMON_ENV+=(-e STRICT_RUN_CONTRACT="$STRICT_RUN_CONTRACT")
echo "[run_shards] serving image=${BENCH_SERVING_IMAGE:-unknown}" \
     "digest=${BENCH_SERVING_IMAGE_DIGEST:-unknown} preset=${PRESET:-none}"

cd "$COMPOSE_DIR"

if [ -n "$PRESTEP" ]; then
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] $PRESTEP"
  else
    echo "[run_shards] pre-step: $PRESTEP"
    $PRESTEP
  fi
fi

# --- token accounting: start snapshot -----------------------------------------
# This runs host-side, against the PUBLISHED vLLM ports (8000 gen, 8001
# embed), because the launcher is not on the compose network. It needs a host
# python; the repo venv on Windows is .venv/Scripts/python.exe. This step is
# entirely best-effort: no python, no server, or a failed probe just means no
# run-level sidecar (per-shard sidecars, written by each entrypoint, still exist).
TOKENS_PY=""
for _cand in "${BENCH_HOST_PYTHON:-}" "$REPO_ROOT/.venv/Scripts/python.exe" \
             "$REPO_ROOT/.venv/bin/python" python3 python; do
  [ -n "$_cand" ] || continue
  if command -v "$_cand" >/dev/null 2>&1 || [ -x "$_cand" ]; then TOKENS_PY="$_cand"; break; fi
done
TOKENS_ENABLED=0
TOKENS_START_FILE="$REPO_ROOT/$PROVIDER/Results/.token_usage_${TAG}_start.json"
TOKENS_OUT_FILE="$REPO_ROOT/$PROVIDER/Results/token_usage_${TAG}.json"
TOKENS_GEN_URL="${BENCH_TOKENS_GEN_URL:-http://127.0.0.1:8000/metrics}"
TOKENS_EMBED_URL="${BENCH_TOKENS_EMBED_URL:-http://127.0.0.1:8001/metrics}"
if [ "${BENCH_TOKEN_ACCOUNTING:-1}" = "1" ] && [ -n "$TOKENS_PY" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p "$REPO_ROOT/$PROVIDER/Results"
  if "$TOKENS_PY" "$REPO_ROOT/benchmark/token_usage.py" snapshot \
        --gen_url "$TOKENS_GEN_URL" --embed_url "$TOKENS_EMBED_URL" \
        --out "$TOKENS_START_FILE"; then
    TOKENS_ENABLED=1
  else
    echo "[run_shards] WARN: token-usage start snapshot failed — no run-level" \
         "token sidecar for $TAG (accounting only, the run is unaffected)"
  fi
elif [ "${BENCH_TOKEN_ACCOUNTING:-1}" = "1" ] && [ -z "$TOKENS_PY" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "[run_shards] WARN: no host python found for token accounting" \
       "(set BENCH_HOST_PYTHON=/path/to/python)"
fi

# --- per-persona container mode (PERSONA_CONTAINERS=1) --------------------------
# This mode launches one container per PERSONA (name <prefix>_<tag>_p<i>,
# RUN_TAG=<tag>_p<i>, START_IDX=i END_IDX=i+1), instead of one per
# persona-RANGE. It defaults ON under BENCH_CLOCKSYNC=1 (which every
# *_clocksync PRESET implies, see the case above). Under clock manipulation, a
# container's process clock REWINDS at each persona rollover, and a rewinding
# clock must never act on a store that already holds an earlier persona's
# data (user-approved design change, 2026-07-27). One persona per container
# makes every store's timeline monotonic by construction. The per-run
# resource each entrypoint derives from RUN_TAG (hindsight/retaindb pg db
# "..._<tag>_p<i>", mem0 qdrant collection "mem0_<tag>_p<i>", supermemory
# spawn data dir) is then per-PERSONA. PERSONA_CONTAINERS=0 forces legacy
# range mode even under clocksync. PERSONA_CONTAINERS=1 enables this mode for
# any run.
#
# NUM_SHARDS is REUSED as the pool size. In this mode it is CONCURRENCY (the
# max number of simultaneously running persona containers), NOT a partition
# count. The per-provider defaults above (hindsight 10, consolidation 5,
# mem0 6, supermemory 6, retaindb_server 6) carry over as the concurrency.
# This is the reason a pool exists at all: 30 simultaneous containers, each
# carrying its own Postgres or spawned provider server, would exhaust the
# host (supermemory spawn alone caps embedding RAM at 2gb per server, against
# a host with about an 8 GiB vLLM floor).
#
# NUM_PERSONAS stays the FULL count in every container. The adapters only
# ever see --start_idx/--end_idx (set explicitly here), and
# preflight_rows.py prefers START_IDX+END_IDX over NUM_PERSONAS when both are
# set. So the full count is inert per-container, and stays correct in the
# manifest env snapshot. The post-merge score container derives its row gate
# from NUM_PERSONAS alone, exactly as after a range-mode wave.
#
# Because the pool must WAIT for free slots while this launcher stays
# non-blocking, the pool runs in a detached nohup'd SUPERVISOR (the same
# pattern as the range-mode token waiter at the bottom). This supervisor
# keeps up to NUM_SHARDS containers running, launches the next persona as
# soon as one exits (it polls every 5s, because `docker wait` on a list
# blocks until ALL exit, which would serialize the pool), logs every launch
# and exit to persona_pool_<tag>.log next to the token sidecar, NEVER stops
# the pool on a nonzero persona exit (it logs the exit and summarizes at the
# end), and runs the token-accounting `finish` (--scope run) itself when the
# last persona exits. Containers stay KEPT on exit, like range mode.
PERSONA_CONTAINERS="${PERSONA_CONTAINERS:-${BENCH_CLOCKSYNC:-0}}"
if [ "$PERSONA_CONTAINERS" = "1" ]; then
  if [ "${SM_TWO:-0}" = "1" ]; then
    echo "run_shards.sh: PERSONA_CONTAINERS=1 is incompatible with" \
         "SUPERMEMORY_TWO_SERVERS=1 (per-shard server routing is range-mode only)" >&2
    exit 2
  fi
  # This is a Postgres identifier guard. hindsight/retaindb_server derive a
  # per-run db "hindsight_<tag>_p<i>" / "retaindb_<tag>_p<i>", and Postgres
  # SILENTLY truncates identifiers at 63 bytes. A truncated-off _p<i> suffix
  # would collapse every persona into ONE db, which is the exact hazard this
  # mode prevents. The longest prefix is "hindsight_" (10) plus "_p29" (4),
  # so TAG must fit in 49 characters. Under HINDSIGHT_PG_MODE=pg0 no Postgres
  # identifier derives from RUN_TAG (the container owns a private pg0 cluster
  # with the vendor's own db name), but the cap stays, because shared-pg
  # hindsight and retaindb_server still derive one.
  case "$PROVIDER" in
    hindsight|retaindb_server)
      if [ "${#TAG}" -gt 49 ]; then
        echo "run_shards.sh: RUN_TAG '$TAG' too long (${#TAG} > 49 chars) for" \
             "per-persona mode: the per-run Postgres db name would exceed the" \
             "63-char identifier limit and silently truncate, collapsing personas" \
             "into one db. Use a shorter tag." >&2
        exit 2
      fi
      ;;
  esac
  if [ "${DRY_RUN:-0}" = "1" ]; then
    for ((i=0; i<NUM_PERSONAS; i++)); do
      cmd=(docker compose run -d)
      [ "$NODEPS" = "1" ] && cmd+=(--no-deps)
      cmd+=(--name "${PREFIX}_${TAG}_p${i}"
           -e RUN_TAG="${TAG}_p${i}" -e STAGE=generate
           -e START_IDX="$i" -e END_IDX="$((i+1))" -e NUM_PERSONAS="$NUM_PERSONAS"
           "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}" "$@" "$SERVICE")
      printf '[dry-run]'; printf ' %q' "${cmd[@]}"; printf '\n'
    done
    echo "[dry-run] persona-pool: $NUM_PERSONAS containers, concurrency $NUM_SHARDS" \
         "(detached supervisor keeps at most $NUM_SHARDS running at once)"
    exit 0
  fi
  _pool_log="$REPO_ROOT/$PROVIDER/Results/persona_pool_${TAG}.log"
  mkdir -p "$REPO_ROOT/$PROVIDER/Results"
  # This serializes the per-shard env args (COMMON_ENV, EXTRA_ENV, caller
  # "$@" -- the same order as range mode, minus the two-server ROUTE_ENV
  # rejected above), so the detached supervisor can eval them back into an argv.
  _POOL_ENVQ=""
  for _a in "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}" "$@"; do
    _POOL_ENVQ+=" $(printf '%q' "$_a")"
  done
  export _POOL_ENVQ _POOL_PREFIX="$PREFIX" _POOL_TAG="$TAG" _POOL_SERVICE="$SERVICE" \
         _POOL_NP="$NUM_PERSONAS" _POOL_K="$NUM_SHARDS" _POOL_NODEPS="$NODEPS" \
         _POOL_COMPOSE_DIR="$COMPOSE_DIR" _POOL_PROVIDER="$PROVIDER" \
         _POOL_REPO_ROOT="$REPO_ROOT" \
         _POOL_TOKENS_ENABLED="$TOKENS_ENABLED" _POOL_TOKENS_PY="$TOKENS_PY" \
         _POOL_TOKENS_START="$TOKENS_START_FILE" _POOL_TOKENS_OUT="$TOKENS_OUT_FILE" \
         _POOL_TOKENS_GEN="$TOKENS_GEN_URL" _POOL_TOKENS_EMBED="$TOKENS_EMBED_URL"
  nohup bash -c '
    set -u
    log() { echo "[pool $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
    cd "$_POOL_COMPOSE_DIR" || exit 1
    declare -a ENVARGS=()
    eval "ENVARGS=($_POOL_ENVQ)"
    declare -A RUNNING=()   # container name -> persona index
    declare -a FAILED=()
    next=0
    log "persona pool start: personas=$_POOL_NP concurrency=$_POOL_K service=$_POOL_SERVICE tag=$_POOL_TAG"
    while (( next < _POOL_NP )) || (( ${#RUNNING[@]} > 0 )); do
      # This reaps by polling, instead of using `docker wait <list>`, because
      # `docker wait` on a list blocks until ALL containers in it exit.
      for name in "${!RUNNING[@]}"; do
        st="$(docker inspect -f "{{.State.Running}}" "$name" 2>/dev/null || echo gone)"
        [ "$st" = "true" ] && continue
        ec="$(docker inspect -f "{{.State.ExitCode}}" "$name" 2>/dev/null || echo unknown)"
        i="${RUNNING[$name]}"
        unset "RUNNING[$name]"
        if [ "$ec" = "0" ]; then
          log "persona $i done: $name exited 0"
        else
          log "persona $i FAILED: $name exited $ec -- pool continues"
          FAILED+=("$i")
        fi
      done
      while (( ${#RUNNING[@]} < _POOL_K )) && (( next < _POOL_NP )); do
        name="${_POOL_PREFIX}_${_POOL_TAG}_p${next}"
        cmd=(docker compose run -d)
        [ "$_POOL_NODEPS" = "1" ] && cmd+=(--no-deps)
        cmd+=(--name "$name" -e RUN_TAG="${_POOL_TAG}_p${next}" -e STAGE=generate
             -e START_IDX="$next" -e END_IDX="$((next+1))" -e NUM_PERSONAS="$_POOL_NP"
             "${ENVARGS[@]}" "$_POOL_SERVICE")
        if out="$("${cmd[@]}" 2>&1)"; then
          RUNNING["$name"]="$next"
          log "launched $name (persona $next; slots ${#RUNNING[@]}/$_POOL_K)"
        else
          log "persona $next LAUNCH FAILED ($name -- stale container name from a prior attempt?): $out"
          log "pool continues; docker rm $name then relaunch that persona by hand"
          FAILED+=("$next")
        fi
        next=$((next+1))
      done
      if (( ${#RUNNING[@]} > 0 )); then sleep 5; fi
    done
    # The last persona has exited, so this closes the run-scope token-
    # accounting window here (the launcher returned long ago). This is
    # accounting only and never fails the run.
    if [ "$_POOL_TOKENS_ENABLED" = "1" ]; then
      "$_POOL_TOKENS_PY" "$_POOL_REPO_ROOT/benchmark/token_usage.py" finish \
          --start "$_POOL_TOKENS_START" --out "$_POOL_TOKENS_OUT" \
          --gen_url "$_POOL_TOKENS_GEN" --embed_url "$_POOL_TOKENS_EMBED" \
          --provider "$_POOL_PROVIDER" --run_tag "$_POOL_TAG" --scope run \
        || log "WARN: token-usage finish failed (accounting only, run unaffected)"
      rm -f "$_POOL_TOKENS_START"
    fi
    if (( ${#FAILED[@]} > 0 )); then
      log "SUMMARY: ${#FAILED[@]}/$_POOL_NP persona(s) FAILED: ${FAILED[*]}"
      log "relaunch each failed persona as its own container (free its name with docker rm first)"
      exit 1
    fi
    log "SUMMARY: all $_POOL_NP personas exited 0"
  ' > "$_pool_log" 2>&1 &
  disown || true
  echo "[run_shards] persona-pool: $NUM_PERSONAS personas, concurrency $NUM_SHARDS" \
       "-> containers ${PREFIX}_${TAG}_p<i> (supervisor log: $_pool_log)"
  exit 0
fi

# This is the two-server split point: shards [0, half) go to server A, and
# shards [half, NUM_SHARDS) go to server B.
SM_HALF=$(( NUM_SHARDS / 2 ))
LAUNCHED_NAMES=()
per=$(( (NUM_PERSONAS + NUM_SHARDS - 1) / NUM_SHARDS ))
for ((k=0; k<NUM_SHARDS; k++)); do
  S=$((k*per)); E=$((S+per)); ((E>NUM_PERSONAS)) && E=$NUM_PERSONAS
  ((S>=E)) && break
  # This is per-shard server routing, for the two-server topology only. It is
  # placed BEFORE "$@", so an explicit -e SUPERMEMORY_ATTACH_URL in the
  # caller's args still wins (docker: last -e wins).
  ROUTE_ENV=()
  if [ "${SM_TWO:-0}" = "1" ]; then
    if ((k < SM_HALF)); then
      ROUTE_ENV=(-e SUPERMEMORY_ATTACH_URL=http://supermemory-server:8787 -e SUPERMEMORY_KEY_FILE=/shared/api_key)
    else
      ROUTE_ENV=(-e SUPERMEMORY_ATTACH_URL=http://supermemory-server-b:8788 -e SUPERMEMORY_KEY_FILE=/shared_b/api_key)
    fi
  fi
  cmd=(docker compose run -d)
  [ "$NODEPS" = "1" ] && cmd+=(--no-deps)
  cmd+=(--name "${PREFIX}_${TAG}_s${k}"
       -e RUN_TAG="${TAG}_s${k}" -e STAGE=generate
       -e START_IDX="$S" -e END_IDX="$E" -e NUM_PERSONAS="$NUM_PERSONAS"
       "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}" "${ROUTE_ENV[@]}" "$@" "$SERVICE")
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[dry-run]'; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
    echo "launched ${PREFIX}_${TAG}_s${k}: personas [$S,$E)"
    LAUNCHED_NAMES+=("${PREFIX}_${TAG}_s${k}")
  fi
done

# --- token accounting: detached end snapshot ------------------------------------
# `docker wait` blocks until each named container exits. So this waiter closes
# the window at the moment the LAST shard finishes, without making this
# launcher block (it has always returned immediately, and long runs get
# monitored, not run in the foreground). The waiter runs under nohup, and its
# output goes to a log next to the sidecar. It can never fail the run.
if [ "$TOKENS_ENABLED" = "1" ] && [ "${#LAUNCHED_NAMES[@]}" -gt 0 ]; then
  _waiter_log="$REPO_ROOT/$PROVIDER/Results/token_usage_${TAG}.waiter.log"
  nohup bash -c '
    py="$1"; repo="$2"; start="$3"; out="$4"; gen="$5"; embed="$6"; provider="$7"; tag="$8"; shift 8
    docker wait "$@" >/dev/null 2>&1 || true
    "$py" "$repo/benchmark/token_usage.py" finish --start "$start" --out "$out" \
        --gen_url "$gen" --embed_url "$embed" --provider "$provider" \
        --run_tag "$tag" --scope run
    rm -f "$start"
  ' _ "$TOKENS_PY" "$REPO_ROOT" "$TOKENS_START_FILE" "$TOKENS_OUT_FILE" \
      "$TOKENS_GEN_URL" "$TOKENS_EMBED_URL" "$PROVIDER" "$TAG" \
      "${LAUNCHED_NAMES[@]}" > "$_waiter_log" 2>&1 &
  disown || true
  echo "[run_shards] token accounting: waiting on ${#LAUNCHED_NAMES[@]} shards ->" \
       "$TOKENS_OUT_FILE (waiter log: $_waiter_log)"
fi
