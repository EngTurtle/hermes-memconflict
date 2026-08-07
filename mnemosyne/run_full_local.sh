#!/usr/bin/env bash
# This is the sharded local generation runner. It is the throughput path
# for a full local run.
#
# A single eval_mnemosyne.py process sends one LLM request at a time.
# So the vLLM server runs at batch size 1, and the GPU sits mostly idle.
# This runner spreads the personas across N processes that run at the
# same time. All of these processes drive the SAME local vLLM servers
# (gemma-4-e2b on :8000, bge-small on :8001). vLLM's continuous batching
# then combines the N in-flight requests into real GPU batches. This is
# a throughput gain the OpenRouter run could not use, because OpenRouter
# has no server-side batching.
#
# This script is the local analog of run_full.sh, which shards for the
# OpenRouter path. This script calls run_local.sh instead, and it also
# passes feature flags through to each shard.
#
# Usage:
#   benchmark/run_full_local.sh [NUM_PERSONAS] [NUM_SHARDS] [TOP_K]
# These feature flags use env vars. Every shard inherits them through run_local.sh's exec call:
#   EXTRACT=1                     -> pass --extract (LLM fact-extraction on ingest)
#   MNEMOSYNE_ENHANCED_RECALL=1   -> enhanced recall
# Examples:
#   benchmark/run_full_local.sh                 # 30 personas, 30 shards (one each)
#   EXTRACT=1 MNEMOSYNE_ENHANCED_RECALL=1 benchmark/run_full_local.sh
#
# NUM_SHARDS defaults to NUM_PERSONAS. This gives the test the highest
# logical granularity: one process per persona. A persona is the atomic
# unit, because its sessions must load in order into one isolated
# database. So one shard per persona is the most parallelism the test
# allows. There is nothing smaller to split.
#
# All shards drive the same two vLLM servers. vLLM's continuous batching
# combines the concurrent embedding, answer, and extraction requests into
# real GPU batches on its own. This script only supplies the concurrency
# and lets vLLM do the batching.
#
# Lower NUM_SHARDS only if the machine runs short on RAM or CPU with N
# Python processes. The vLLM side is not the limit here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL="${1:-30}"
SHARDS="${2:-$TOTAL}"   # default: one shard per persona (max logical parallelism)
TOPK="${3:-5}"

EXTRACT_FLAG=""
[ "${EXTRACT:-0}" = "1" ] && EXTRACT_FLAG="--extract"

RESDIR="$ROOT/mnemosyne/Results"
SHARDDIR="$RESDIR/shards"
mkdir -p "$SHARDDIR"
# This is the merged output path. Override it with RESULTS_FILE to keep
# separate runs from overwriting each other, for example a default
# baseline run versus an EXTRACT or enhanced-recall run. This also
# protects the committed mnemosyne_results.jsonl file.
MERGED="${RESULTS_FILE:-$RESDIR/mnemosyne_results.jsonl}"
# This is a fresh run. It drops stale shard fragments, so the merge does not pick up a prior run's files.
rm -f "$SHARDDIR"/shard_*.jsonl "$SHARDDIR"/shard_*.json "$SHARDDIR"/shard_*.log

echo "[run_full_local] personas=$TOTAL shards=$SHARDS top_k=$TOPK" \
     "extract=${EXTRACT:-0} enhanced_recall=${MNEMOSYNE_ENHANCED_RECALL:-0}" \
     "-> $MERGED"

# This computes ceiling division to size each shard.
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))

pids=()
for ((s=0; s<SHARDS; s++)); do
  START=$(( s * PER ))
  END=$(( START + PER ))
  (( END > TOTAL )) && END=$TOTAL
  (( START >= END )) && continue
  echo "[run_full_local] shard $s: personas [$START,$END)"
  # run_local.sh exports the local vLLM environment and then runs exec on
  # the command. It inherits any EXTRACT-derived flag and the
  # MNEMOSYNE_ENHANCED_RECALL env variable from this script.
  "$ROOT/mnemosyne/run_local.sh" python -u "$ROOT/mnemosyne/eval_mnemosyne.py" \
      --start_idx "$START" --end_idx "$END" --top_k "$TOPK" $EXTRACT_FLAG \
      --output_jsonl_path "$SHARDDIR/shard_${s}.jsonl" \
      --output_json_path "$SHARDDIR/shard_${s}.json" \
      > "$SHARDDIR/shard_${s}.log" 2>&1 &
  pids+=($!)
done

echo "[run_full_local] launched ${#pids[@]} shards; waiting..."
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; echo "[run_full_local] shard pid $p FAILED"; fi
done

# This merges the shard outputs, in shard order, into the MERGED results
# file. The MERGED path comes from RESULTS_FILE or the default, set
# earlier in this script.
: > "$MERGED"
for ((s=0; s<SHARDS; s++)); do
  [ -f "$SHARDDIR/shard_${s}.jsonl" ] && cat "$SHARDDIR/shard_${s}.jsonl" >> "$MERGED"
done
echo "[run_full_local] merged $(wc -l < "$MERGED") persona rows into $MERGED"
[ "$fail" -eq 0 ] && echo "[run_full_local] ALL SHARDS OK" || echo "[run_full_local] COMPLETED WITH FAILURES"
