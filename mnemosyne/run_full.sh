#!/usr/bin/env bash
# This script runs a full or bounded MemConflict-vs-Mnemosyne benchmark.
# It shards the run across personas, so the answer-generation LLM calls run
# at the same time instead of one after another.
#
# Usage:
#   benchmark/run_full.sh [NUM_PERSONAS] [NUM_SHARDS] [TOP_K]
# Defaults: all 30 personas, 6 shards, top_k=5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL="${1:-30}"
SHARDS="${2:-6}"
TOPK="${3:-5}"

RESDIR="$ROOT/mnemosyne/Results"
SHARDDIR="$RESDIR/shards"
mkdir -p "$SHARDDIR"

echo "[run_full] personas=$TOTAL shards=$SHARDS top_k=$TOPK"

# This computes ceiling division to size each shard.
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))

pids=()
for ((s=0; s<SHARDS; s++)); do
  START=$(( s * PER ))
  END=$(( START + PER ))
  (( END > TOTAL )) && END=$TOTAL
  (( START >= END )) && continue
  echo "[run_full] shard $s: personas [$START,$END)"
  MNEMOSYNE_EMBEDDING_THREADS=1 "$ROOT/mnemosyne/run.sh" python -u "$ROOT/mnemosyne/eval_mnemosyne.py" \
      --start_idx "$START" --end_idx "$END" --top_k "$TOPK" \
      --output_jsonl_path "$SHARDDIR/shard_${s}.jsonl" \
      --output_json_path "$SHARDDIR/shard_${s}.json" \
      > "$SHARDDIR/shard_${s}.log" 2>&1 &
  pids+=($!)
done

echo "[run_full] launched ${#pids[@]} shards; waiting..."
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; echo "[run_full] shard pid $p FAILED"; fi
done

# This merges the shard outputs, in shard order, into the canonical results file.
MERGED="$RESDIR/mnemosyne_results.jsonl"
: > "$MERGED"
for ((s=0; s<SHARDS; s++)); do
  [ -f "$SHARDDIR/shard_${s}.jsonl" ] && cat "$SHARDDIR/shard_${s}.jsonl" >> "$MERGED"
done
echo "[run_full] merged $(wc -l < "$MERGED") persona rows into $MERGED"
[ "$fail" -eq 0 ] && echo "[run_full] ALL SHARDS OK" || echo "[run_full] COMPLETED WITH FAILURES"
