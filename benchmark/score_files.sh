#!/usr/bin/env bash
# Score any provider's Results JSONL against any OpenAI-compatible judge.
#
# Judging is provider-agnostic: run_score() in benchmark/docker/answer_env.sh
# takes only a provider directory and a tag, and every tool it calls lives in
# benchmark/. The per-provider Docker entrypoints exist for the GENERATE stage,
# where the providers need different infrastructure. Routing a score through
# them makes the judge inherit that infrastructure: entrypoint.hindsight.sh
# creates a per-run Postgres database at top level with no STAGE gate, so
# STAGE=score dies on a missing hindsight-pg before it reads a single row.
# This script skips the entrypoints instead of gating each one.
#
# It sources answer_env.sh rather than re-declaring the judge decoding, so the
# fairness contract stays in ONE place. A second copy of those ten variables
# would drift and silently change what the numbers mean.
#
# usage:
#   benchmark/score_files.sh [options] <results.jsonl> [<results.jsonl> ...]
#
#   --base_url URL   judge endpoint      (default http://localhost:8002/v1)
#   --model NAME     judge model         (default gemma-4-12b)
#   --api_key KEY    judge key           (default local-vllm)
#   --suffix S       tag suffix          (default gj12)
#   --workers N      judge concurrency   (default 32)
#   --temperature T  judge sampling      (REQUIRED, no default)
#   --top_p P        judge sampling      (REQUIRED, no default)
#   --top_k K        judge sampling      (REQUIRED, no default)
#   --python PATH    interpreter         (default repo .venv, else python)
#   --stage S        score|summarize|both (default both)
#
# The three sampling options have NO defaults on purpose. bench_judge_env
# defaults to temperature 0.6 / top_k 20, which is the qwen3.5-4b contract. A
# gemma-4-12b judge runs 1.0 / 0.95 / 64. Inheriting the default silently
# judges one provider under different sampling than the rest of its arm and
# makes the numbers non-comparable. That happened on 2026-07-30: hindsight ran
# 61 questions at 0.6 against the wave's 1.0, and the checkpoint was deleted.
#
# example, gemma-4-12b judge:
#   benchmark/score_files.sh --temperature 1.0 --top_p 0.95 --top_k 64 \
#     hindsight/Results/v4/hindsight_results_v4minc.jsonl
set -uo pipefail

BASE_URL="http://localhost:8002/v1"
MODEL="gemma-4-12b"
API_KEY="local-vllm"
SUFFIX="gj12"
WORKERS="32"
PYBIN=""
STAGE_SEL="both"
TEMPERATURE=""
TOP_P=""
TOP_K=""
FILES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --base_url) BASE_URL="$2"; shift 2 ;;
    --model)    MODEL="$2";    shift 2 ;;
    --api_key)  API_KEY="$2";  shift 2 ;;
    --suffix)   SUFFIX="$2";   shift 2 ;;
    --workers)  WORKERS="$2";  shift 2 ;;
    --python)   PYBIN="$2";    shift 2 ;;
    --stage)    STAGE_SEL="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p)    TOP_P="$2";    shift 2 ;;
    --top_k)    TOP_K="$2";    shift 2 ;;
    -h|--help)  sed -n '18,32p' "$0"; exit 0 ;;
    -*)         echo "unknown option: $1" >&2; exit 2 ;;
    *)          FILES+=("$1"); shift ;;
  esac
done

[ ${#FILES[@]} -gt 0 ] || { echo "no result files given" >&2; exit 2; }

# Refuse to start rather than inherit bench_judge_env's qwen defaults. See the
# header note: a silent fallback produces scores that cannot be compared.
for _req in TEMPERATURE:--temperature TOP_P:--top_p TOP_K:--top_k; do
  _var="${_req%%:*}"
  if [ -z "${!_var}" ]; then
    echo "FATAL: ${_req#*:} is required -- it must match the judge model's card" >&2
    echo "  gemma-4-12b: --temperature 1.0 --top_p 0.95 --top_k 64" >&2
    echo "  qwen3.5-4b:  --temperature 0.6 --top_p 0.95 --top_k 20" >&2
    exit 2
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

log() { echo "[score_files $(date -u +%H:%M:%S)] $*"; }

# Two drivers against one judge halve throughput and can race the same
# checkpoint file. mkdir is atomic, so it works where a PID file does not.
# The lock is keyed on the suffix, so two different judge arms can run.
LOCK_DIR="${SCORE_FILES_LOCK_DIR:-${TMPDIR:-/tmp}/score_files_${SUFFIX}.lock}"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another score_files.sh holds $LOCK_DIR -- refusing to start" >&2
  echo "verify with: ps -ef | grep [s]core_files" >&2
  exit 3
fi
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

if [ -z "$PYBIN" ]; then
  if [ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PYBIN="$REPO_ROOT/.venv/Scripts/python.exe"
  elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYBIN="$REPO_ROOT/.venv/bin/python"
  else
    PYBIN="python"
  fi
fi

# answer_env.sh declares two variables and then only functions, so sourcing it
# has no side effect beyond BENCH_ROOT.
# shellcheck source=docker/answer_env.sh
. "$REPO_ROOT/benchmark/docker/answer_env.sh"

export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_MODEL="$MODEL"
export OPENAI_API_KEY="$API_KEY"
export SCORE_WORKERS="$WORKERS"
export BENCH_PYTHON="$PYBIN"
# bench_judge_env reads these and exports OPENAI_TEMPERATURE / MEMCONFLICT_TOP_P
# / MEMCONFLICT_TOP_K from them.
export BENCH_JUDGE_TEMPERATURE="$TEMPERATURE"
export BENCH_JUDGE_TOP_P="$TOP_P"
export BENCH_JUDGE_TOP_K="$TOP_K"
export BENCH_JUDGE_MIN_P="${BENCH_JUDGE_MIN_P:-0}"
export BENCH_JUDGE_PRESENCE_PENALTY="${BENCH_JUDGE_PRESENCE_PENALTY:-0}"

# write_manifest.py records the code version that produced a score. Resolve it
# here, because this script runs on the host where .git is present. Do NOT set
# MSYS_NO_PATHCONV globally to fix a Docker path: it breaks `git -C` on
# Git-Bash and the manifest then records a stale code_sha.
if [ -z "${GIT_SHA:-}" ]; then
  GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null)"
fi
[ -n "$GIT_SHA" ] || { echo "FATAL: cannot resolve GIT_SHA" >&2; exit 1; }
export GIT_SHA

log "judge=$MODEL at $BASE_URL workers=$WORKERS suffix=$SUFFIX git_sha=$GIT_SHA"
log "sampling temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
log "python=$PYBIN files=${#FILES[@]}"

rc_total=0
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    log "SKIP $f -- not found (a .7z needs extracting first)"
    rc_total=1
    continue
  fi

  # The provider directory is the parent of Results/, so Scores/ lands beside
  # it exactly where the Docker path writes it.
  abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
  case "$abs" in
    */Results/*) provider_dir="${abs%%/Results/*}" ;;
    *) log "SKIP $f -- not under a Results/ directory"; rc_total=1; continue ;;
  esac
  provider="$(basename "$provider_dir")"

  # Tag mirrors the Docker naming so a host-scored artifact is
  # indistinguishable from a container-scored one:
  # <provider>_results_<tag>.jsonl -> <tag>_<suffix>
  base="$(basename "$abs" .jsonl)"
  tag="${base#${provider}_results_}"
  [ "$tag" != "$base" ] || tag="$base"
  tag="${tag}_${SUFFIX}"

  # run_score reads RESULTS_FILE when set, so this pins the exact input and
  # lets the file live under any Results/v<N>/ folder.
  export RESULTS_FILE="$abs"
  unset SCORES_FILE CHECKPOINT SUMMARY_FILE

  log "START $provider tag=$tag <- $abs"
  rc=0
  case "$STAGE_SEL" in
    score)     STAGE=score     run_score     "$provider_dir" "$tag" || rc=$? ;;
    summarize) STAGE=summarize run_summarize "$provider_dir" "$tag" || rc=$? ;;
    both)      STAGE=score     run_score     "$provider_dir" "$tag" \
                 && STAGE=summarize run_summarize "$provider_dir" "$tag" || rc=$? ;;
    *) log "unknown --stage $STAGE_SEL (expected score|summarize|both)"; exit 2 ;;
  esac

  if [ "$rc" -ne 0 ]; then
    log "FAIL $provider rc=$rc -- continuing with the next file"
    rc_total=1
  else
    log "DONE $provider tag=$tag"
  fi
done

log "all files complete (rc=$rc_total)"
exit "$rc_total"
