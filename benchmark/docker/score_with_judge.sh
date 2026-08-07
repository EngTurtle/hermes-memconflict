#!/usr/bin/env bash
# SUPERSEDED by benchmark/score_files.sh (2026-07-30). Kept because it produced
# four of the five _gj12 results.
#
# This drives scoring through each provider's Docker entrypoint, so the judge
# inherits generate-time infrastructure it never uses. That is why the hindsight
# leg failed here with zero questions judged. Scoring needs no provider
# infrastructure, so use score_files.sh, which takes result paths and judge
# server details and enters no entrypoint.
# Score already-generated Results files with a SECOND judge model, one provider
# at a time. This is operational tooling. It is not part of the harness.
#
# WHY THIS SCRIPT EXISTS
# The normal path scores with the same model that answered (CLAUDE.md harness
# contract). To re-judge a banked wave with a different judge, five things must
# change together, and getting any one wrong produces a number that looks valid:
#   1. OPENAI_BASE_URL + OPENAI_MODEL must point at the judge server, NOT the
#      answer server. `compose run` without --no-deps would start vllm-gen and
#      contend for the same 16GB card.
#   2. The score outputs must carry a tag that no qwen-judged file uses, or a
#      re-judge silently overwrites the original scores.
#   3. RESULTS_FILE must be an absolute CONTAINER path, and Git-Bash rewrites
#      those (see MSYS_NO_PATHCONV below).
#   4. NUM_PERSONAS must be set, because preflight_rows.py derives the expected
#      persona count from it and hindsight's entrypoint defaults it to 1.
#   5. The providers must run SEQUENTIALLY. One GPU, and the judge already
#      saturates it (94-97% util at 16 workers).
#
# USAGE (run from benchmark/docker/, detached):
#   nohup ./score_with_judge.sh > /tmp/score_gj12.log 2>&1 &
#   ./score_with_judge.sh mnemosyne hindsight       # a subset, same order rules
#
# A re-run is safe and resumes: score_resumable.py checkpoints per question, so
# an interrupted provider picks up where it stopped.
set -uo pipefail

# Git-Bash rewrites any argument that looks like a POSIX absolute path into a
# Windows path before exec. `-e RESULTS_FILE=/app/x` arrived inside the
# container as `C:/Program Files/Git/app/x` and the row gate failed with FILE
# NOT FOUND. This must be exported, not just prefixed, because it has to reach
# every docker invocation below.
export MSYS_NO_PATHCONV=1

COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$COMPOSE_DIR" || exit 1

# Defined here, ABOVE the GIT_SHA resolution below, because that block calls it
# on its fatal path. It used to live further down, past its first caller.
log() { echo "[score_judge $(date -u +%H:%M:%S)] $*"; }

# SINGLE-INSTANCE GUARD -- do not remove.
# On 2026-07-29 two copies of this script ran at once. The first launch was
# stopped with `pkill -f score_with_judge`, which reported success and killed
# NOTHING: under MSYS, pkill routinely fails to signal these processes, the same
# way plain `kill` failed on watch_vllm.sh and needed `kill -9 <pid>`. So the
# relaunch produced two drivers.
# The damage is not just duplicated work. Each driver runs SCORE_WORKERS judge
# threads, so the shared vllm-judge saw 64 concurrent requests against a 32-worker
# budget. KV hit 100%, vLLM began preempting and recomputing, and generation
# throughput HALVED, from 795-900 tok/s to 326-403, with the per-50-question
# interval degrading run-on-run (236s -> 297s). Worse, the two drivers walk the
# same provider list, so they eventually reach the same provider and write the
# same Scores/<tag>_judged_checkpoint.jsonl concurrently.
# Verify a kill with `ps -ef | grep [s]core_with_judge`, never by pkill's exit code.
# mkdir is atomic on every filesystem here, including MSYS/Windows.
LOCK_DIR="${SCORE_JUDGE_LOCK_DIR:-${TMPDIR:-/tmp}/score_with_judge.lock}"
if mkdir "$LOCK_DIR" 2>/dev/null; then
  echo $$ > "$LOCK_DIR/pid"
else
  _owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")"
  if [ -n "$_owner" ] && kill -0 "$_owner" 2>/dev/null; then
    log "FATAL: another scoring driver is already running (pid=$_owner)."
    log "       Two drivers would double the judge's concurrent load and can"
    log "       write the same checkpoint. To stop that one: kill -9 $_owner"
    log "       then rm -rf $LOCK_DIR"
    exit 3
  fi
  log "clearing stale lock from dead pid=${_owner:-unknown}"
  echo $$ > "$LOCK_DIR/pid"
fi
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

# The judge server. See the vllm-judge service in docker-compose.yml. Callers on
# the compose network reach it by service name; the published port is 8002.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://vllm-judge:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma-4-12b}"
# Suffix appended to the source RUN_TAG for every score artifact. It keeps a
# gemma-judged score file, checkpoint, summary, and manifest from colliding with
# the qwen-judged ones for the same run. CHANGE THIS if you swap judge models
# again, or the next judge overwrites this one's scores.
JUDGE_SUFFIX="${JUDGE_SUFFIX:-gj12}"

# Google's suggested Gemma 4 sampling, which is also the checkpoint's own
# generation_config.json. bench_judge_env reads these BENCH_JUDGE_* overrides;
# without them the judge would run the Qwen3.5 card's precise set (temp 0.6,
# top_k 20), which is the wrong card for this model.
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.95}"
JUDGE_TOP_K="${JUDGE_TOP_K:-64}"

# 32, not answer_env.sh's default 40 and not the 16 first guessed here.
# MEASURED 2026-07-29 on the same 122-question persona-0 slice, judged from an
# empty checkpoint each time so the work was identical:
#   kernel + workers           to 50    50->100 (steady)     full 122
#   unified, 16 workers        +370 s   288 s (10.4 q/min)     930 s
#   unified, 32 workers        +217 s   227 s (13.2 q/min)     629 s
#   split prefill/decode, 32   +103 s    75 s (40.0 q/min)     239 s
# The 3.03x jump on the last row came from the vllm-judge --attention-config
# use_prefill_decode_attention flag, NOT from anything in this file. Its
# evidence, including the check that it does not change verdicts, is on that
# flag in docker-compose.yml.
# Quote the STEADY-STATE column, 1.27x for 16->32 workers, not the 1.71x that
# time-to-50 suggests.
# Time-to-50 is inflated because a 32-worker first wave fills with twice as many
# requests, so early completions land sooner. That is a one-off ramp effect and
# it does not repeat across 3,750 questions.
# Why 32 and not more: the judge is KV-bound, not compute-bound. At 32 workers
# the server pins at Running 23-24 with Waiting 8-9 and KV at 98-99%, so ~24 is
# the concurrent-execution ceiling and the 8-deep queue only exists to stop the
# engine starving. More workers add queue latency, not parallelism, and queue
# time counts against MEMCONFLICT_REQUEST_TIMEOUT.
# Do NOT raise --max-num-batched-tokens to buy prefill throughput: 8192 cut the
# KV cache from 78,905 to 47,053 tokens (~23 concurrent sequences down to ~14),
# because the wider prefill batch inflates activation memory and vLLM takes it
# out of KV. That is the wrong trade when KV is the binding constraint.
SCORE_WORKERS="${SCORE_WORKERS:-32}"

# The container has no .git (the build bakes source), so a manifest falls back
# to the /app/benchmark/.git_sha stamp unless GIT_SHA is exported here. That
# stamp is written by an earlier run and goes STALE: the first launch of this
# script recorded code_sha=58b45071ff8c, left over from a Supermemory run, when
# HEAD was 37cea28. A score manifest that names the wrong commit is worse than
# one that names none.
#
# Use `cd` and NOT `git -C "$path"`. MSYS_NO_PATHCONV=1 is exported above, which
# stops Git-Bash rewriting POSIX paths in arguments, so git.exe receives a
# literal /c/Users/... and dies with "cannot change to ...: No such file or
# directory". It fails quietly into the empty string, which is how the stale
# stamp got used. `cd` never passes the path as an argument, so it is immune.
if [ -z "${GIT_SHA:-}" ]; then
  GIT_SHA="$(cd "$COMPOSE_DIR/../.." && git rev-parse --short HEAD 2>/dev/null)"
fi
export GIT_SHA
if [ -z "$GIT_SHA" ]; then
  log "FATAL: could not resolve GIT_SHA, so every manifest would record a stale"
  log "       or missing code_sha. Export GIT_SHA=<sha> explicitly and re-run."
  exit 2
fi

# service | provider dir under /app | source RUN_TAG | results basename
# Supermemory's banked minimal run is v4minc3, not v4minc: the first three
# attempts never completed 30 personas (BENCHMARK_MATRIX v4minc3 row).
PROVIDERS_ALL=(
  "mnemosyne|mnemosyne|v4minc|mnemosyne_results_v4minc.jsonl"
  "hindsight|hindsight|v4minc|hindsight_results_v4minc.jsonl"
  "mem0|mem0|v4minc|mem0_results_v4minc.jsonl"
  "retaindb-server|retaindb_server|v4minc|retaindb_server_results_v4minc.jsonl"
  "supermemory|supermemory|v4minc3|supermemory_results_v4minc3.jsonl"
)

# Pick the requested subset, preserving the order above.
SELECTED=()
if [ "$#" -gt 0 ]; then
  for want in "$@"; do
    for row in "${PROVIDERS_ALL[@]}"; do
      [ "${row%%|*}" = "$want" ] && SELECTED+=("$row")
    done
  done
  if [ "${#SELECTED[@]}" -ne "$#" ]; then
    log "FATAL: unknown provider in: $*"
    log "       known: mnemosyne hindsight mem0 retaindb-server supermemory"
    exit 2
  fi
else
  SELECTED=("${PROVIDERS_ALL[@]}")
fi

if ! curl -fsS http://localhost:8002/health >/dev/null 2>&1; then
  log "FATAL: vllm-judge is not answering on localhost:8002."
  log "       start it with: docker compose --profile judge up -d vllm-judge"
  exit 1
fi
log "judge=${JUDGE_MODEL} at ${JUDGE_BASE_URL} workers=${SCORE_WORKERS} suffix=${JUDGE_SUFFIX} git_sha=${GIT_SHA:-none}"

overall=0
for row in "${SELECTED[@]}"; do
  IFS='|' read -r service dir srctag results <<< "$row"
  tag="${srctag}_${JUDGE_SUFFIX}"
  results_path="/app/${dir}/Results/v4/${results}"

  # entrypoint.retaindb-server.sh creates its per-run Postgres database at the
  # TOP level, before the STAGE dispatch, and exits 1 if that fails. So a score
  # stage needs hindsight-pg even though it reads only a JSONL file. The
  # "already exists" gate applies to generate|all only, so a fresh empty DB here
  # is harmless.
  if [ "$service" = "retaindb-server" ]; then
    log "starting hindsight-pg (retaindb-server creates its per-run DB even at STAGE=score)"
    docker compose up -d hindsight-pg >/dev/null 2>&1 || log "WARN: hindsight-pg start reported an error"
  fi

  for stage in score summarize; do
    cname="j${JUDGE_SUFFIX}_${stage}_${dir}"
    docker rm -f "$cname" >/dev/null 2>&1
    log "START $service stage=$stage tag=$tag <- $results_path"
    docker compose run --rm --no-deps --name "$cname" \
      -e STAGE="$stage" \
      -e RUN_TAG="$tag" \
      -e NUM_PERSONAS=30 \
      -e START_IDX=0 -e END_IDX=30 \
      -e OPENAI_BASE_URL="$JUDGE_BASE_URL" \
      -e OPENAI_MODEL="$JUDGE_MODEL" \
      -e BENCH_JUDGE_TEMPERATURE="$JUDGE_TEMPERATURE" \
      -e BENCH_JUDGE_TOP_P="$JUDGE_TOP_P" \
      -e BENCH_JUDGE_TOP_K="$JUDGE_TOP_K" \
      -e BENCH_JUDGE_MIN_P=0 \
      -e BENCH_JUDGE_PRESENCE_PENALTY=0 \
      -e SCORE_WORKERS="$SCORE_WORKERS" \
      -e RESULTS_FILE="$results_path" \
      "$service"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      log "FAIL $service stage=$stage rc=$rc -- continuing with the next provider"
      overall=1
      break        # a failed score makes its summarize meaningless
    fi
    log "DONE $service stage=$stage"
  done
done

log "all done (overall rc=$overall)"
exit "$overall"
