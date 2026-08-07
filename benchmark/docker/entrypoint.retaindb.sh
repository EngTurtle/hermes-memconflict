#!/usr/bin/env bash
# MemConflict and RetainDB run-service entrypoint.
# A 1-persona smoke test verified this script in Docker with a local vLLM
# answer and judge model.
#
# This script uses the same stage shape as the Mnemosyne entrypoint: generate,
# then score, then summarize. This script runs in one process, not sharded.
# The adapter starts and owns one disposable @retaindb/local Node server per
# run. The adapter isolates personas by project, so there is no per-persona
# sharding here. RetainDB has no internal LLM. The answer and judge LLM is
# the only LLM in the pipeline. By default that is the shared vLLM server.
# An earlier host smoke test used OpenRouter gpt-oss-120b instead.
# Embeddings run in Node.
set -euo pipefail

ROOT=/app
cd "$ROOT"
log() { echo "[retaindb $(date -u +%H:%M:%S)] $*"; }

# This sources the shared answer/judge decoding config and the score and
# summarize calls. The fairness contract requires byte-for-byte identical
# config for every provider. Before this fix, this entrypoint set no
# answer/judge decoding. Answers then ran at the vLLM server defaults, while
# Mnemosyne pinned temperature to 0.2. That was a fairness bug. This restores
# the shared config. BENCH_PYTHON=python3 is required because the node:22
# base image ships python3 only, not a bare `python` command. The shared
# run_score/run_summarize helpers must call python3 for that reason.
export BENCH_PYTHON=python3
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"

STAGE="${STAGE:-all}"
TOTAL="${NUM_PERSONAS:-1}"
TOPK="${TOP_K:-5}"
TAG="${RUN_TAG:-retaindb}"

# This is the answer/judge LLM used by MemConflict's llm_request. Compose
# sets OPENAI_BASE_URL and OPENAI_MODEL. The SDK needs a non-empty key even
# for vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-local-vllm}}"
# node must be on PATH so the RetainDB server manager can spawn dist/cli.js.
# The node:22 base image provides node.
export RETAINDB_EMBEDDING_PROVIDER="${RETAINDB_EMBEDDING_PROVIDER:-hash}"
# Build time installs the @retaindb/local package outside /app/retaindb, into
# /opt/retaindb_pkg (see Dockerfile.retaindb). Compose bind-mounts the whole
# retaindb/ directory over /app/retaindb for live edits, and this mount would
# hide an in-tree node_modules. Point the server manager straight at the
# /opt install instead.
export RETAINDB_CLI_JS="${RETAINDB_CLI_JS:-/opt/retaindb_pkg/node_modules/@retaindb/local/dist/cli.js}"

RESDIR="$ROOT/retaindb/Results"
SCOREDIR="$ROOT/retaindb/Scores"
mkdir -p "$RESDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/retaindb_results_${TAG}.jsonl}"
RESULTS_JSON="${RESULTS_FILE%.jsonl}.json"
SCORES_FILE="${SCORES_FILE:-$SCOREDIR/retaindb_${TAG}_eval_scores.jsonl}"
CHECKPOINT="${CHECKPOINT:-$SCOREDIR/${TAG}_judged_checkpoint.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$SCOREDIR/summary_${TAG}.json}"

CAPS=()
[ -n "${MAX_SESSIONS:-}" ]              && CAPS+=(--max_sessions "$MAX_SESSIONS")
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAPS+=(--max_questions_per_session "$MAX_QUESTIONS_PER_SESSION")
[ -n "${RETAINDB_EMBEDDING_PROVIDER:-}" ] && CAPS+=(--embedding_provider "$RETAINDB_EMBEDDING_PROVIDER")
# RETAIN_GRANULARITY (session|message|exchange) selects the ingestion arm.
# 'exchange' is the plugin-faithful arm. If unset, the adapter default is
# 'session'.
[ -n "${RETAIN_GRANULARITY:-}" ]        && CAPS+=(--retain_granularity "$RETAIN_GRANULARITY")

# This sets the persona range. This entrypoint runs personas serially in one
# process, unlike entrypoint.mnemosyne.sh, which forks NUM_SHARDS workers. So
# a full 30-persona generate run takes about 30 times one persona's wall
# time. START_IDX and END_IDX let several containers split the dataset, each
# with its own RUN_TAG and its own results file. Merge the files with `cat`
# and check the merged file before scoring. These defaults reproduce the
# earlier hardcoded [0,TOTAL) range exactly.
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-$TOTAL}"

do_generate() {
  # This applies the canonical answer decoding, identical across providers,
  # and writes a best-effort manifest.
  bench_answer_env
  python3 "$ROOT/benchmark/write_manifest.py" --provider_dir "$ROOT/retaindb" \
      --run_tag "$TAG" --stage generate || echo "[retaindb] WARN: manifest write failed"
  log "GENERATE personas=[$START_IDX,$END_IDX) top_k=$TOPK embed=$RETAINDB_EMBEDDING_PROVIDER thinking=${MEMCONFLICT_ENABLE_THINKING} answer_max_tokens=${OPENAI_MAX_TOKENS} -> $RESULTS_FILE"
  python3 -u "$ROOT/retaindb/eval_retaindb.py" \
      --start_idx "$START_IDX" --end_idx "$END_IDX" --top_k "$TOPK" "${CAPS[@]}" \
      --output_jsonl_path "$RESULTS_FILE" \
      --output_json_path "$RESULTS_JSON"
}

# answer_env.sh's run_stage now holds the shared judge env, the score call,
# the score-stage manifest, and the shared summarize call. run_stage writes
# the manifest inside run_score, after bench_judge_env, so the manifest
# records the judge decoding.
run_stage "$ROOT/retaindb" "$TAG" do_generate
