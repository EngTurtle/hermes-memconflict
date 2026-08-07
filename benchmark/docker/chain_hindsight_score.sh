#!/usr/bin/env bash
# Wait for the running score_with_judge.sh wave to finish, then score the
# hindsight file that the wave skipped.
#
# Hindsight FAILED in the wave at 22:48 EDT 2026-07-29 with rc=1 and zero
# questions judged: score_with_judge.sh runs providers with --no-deps, which
# suppresses the compose `depends_on: hindsight-pg`, and
# entrypoint.hindsight.sh creates a per-run Postgres database at top level
# with no STAGE gate. score_files.sh bypasses the entrypoint entirely, so no
# database is involved.
#
# This waits instead of running now, because a second judge client halves
# generation throughput. Two drivers on 2026-07-29 took the judge from
# 795-900 tok/s to 326-403 tok/s.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

log() { echo "[chain $(date -u +%H:%M:%S)] $*"; }

TARGET="hindsight/Results/v4/hindsight_results_v4minc.jsonl"

log "waiting for the score_with_judge.sh wave to exit"
waited=0
while ps -ef | grep -q "[s]core_with_judge"; do
  sleep 60
  waited=$((waited + 60))
done
log "wave exited after ${waited}s of waiting; judge is free"

# The wave leaves its own container behind for a moment on exit. This lets the
# judge drain in-flight requests before a new client opens 32 more.
sleep 30

[ -f "$TARGET" ] || { log "FATAL: $TARGET not found"; exit 1; }

log "START hindsight score"
exec bash "$REPO_ROOT/benchmark/score_files.sh" \
  --base_url "${JUDGE_BASE_URL_HOST:-http://localhost:8002/v1}" \
  --model "${JUDGE_MODEL:-gemma-4-12b}" \
  --suffix "${JUDGE_SUFFIX:-gj12}" \
  --workers "${SCORE_WORKERS:-32}" \
  --temperature "${JUDGE_TEMPERATURE:-1.0}" \
  --top_p "${JUDGE_TOP_P:-0.95}" \
  --top_k "${JUDGE_TOP_K:-64}" \
  "$TARGET"
