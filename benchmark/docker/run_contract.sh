# shellcheck shell=bash
# Shared RUN-CONTRACT helpers for every provider entrypoint. Every entrypoint
# sources this file. Do not run it on its own.
#
# WHY THIS FILE EXISTS: three run-identity steps must happen identically for
# all five providers at generate time. The 2026-07-24 upstream review's
# "Make run identity fail closed" finding says these steps cannot stay
# per-provider copy-paste code. Copy-paste code is how the v4-minimal wave
# ended up with one provider's manifest recording head_sha:null, and none of
# the five recording a serving envelope. The steps are:
#
#   1. bench_capture_serving_envelope — probe the live vllm-gen/-embed servers
#      for the served alias, checkpoint root, and engine version. Write the
#      result into the sidecar file write_manifest.py already looks for.
#   2. bench_write_manifest — write the manifest and the run-contract hash.
#      Fail the stage when the strict gate is armed and a required contract
#      field is missing.
#   3. bench_tokens_start / bench_tokens_finish — snapshot vLLM's Prometheus
#      counters around the generate stage. This bills provider-internal LLM
#      spend (extraction, consolidation, memory agents) to the run, since the
#      harness itself never sees those calls.
#
# STRICT GATE: the gate arms when STRICT_RUN_CONTRACT=1, or when
# BENCH_CLOCKSYNC=1 (the clock-normalized wave is exactly the wave whose
# artifacts must be identifiable). Steps 1 and 2 abort the stage under the
# gate. Step 3 never aborts anything. Bad token accounting gets recorded as
# valid:false. It is not a reason to lose a run.
#
# API (each function takes the in-container provider dir, for example
# /app/mnemosyne, and the tag):
#   bench_run_contract_strict                            # exit status only
#   bench_capture_serving_envelope <provider_dir> <tag>
#   bench_write_manifest          <provider_dir> <tag> <stage>
#   bench_tokens_start            <provider_dir> <tag>
#   bench_tokens_finish           <provider_dir> <tag>
#   bench_generate_preamble       <provider_dir> <tag>   # runs steps 1, 2, and 3-start
#
# `log` resolves dynamically to the calling entrypoint's own log() function
# (the same trick answer_env.sh's run_stage relies on). This keeps each
# provider's own log prefix.

_BENCH_RC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$(cd "$_BENCH_RC_DIR/../.." && pwd)}"

bench_run_contract_strict() {
  [ "${STRICT_RUN_CONTRACT:-0}" = "1" ] || [ "${BENCH_CLOCKSYNC:-0}" = "1" ]
}

# --- step 1: serving envelope ---------------------------------------------------
bench_capture_serving_envelope() {
  local provider_dir="$1" tag="$2"
  local py="${BENCH_PYTHON:-python}"
  local strict_flag=()
  bench_run_contract_strict && strict_flag=(--strict)
  if "$py" "$BENCH_ROOT/benchmark/capture_serving_envelope.py" \
        --provider_dir "$provider_dir" --run_tag "$tag" "${strict_flag[@]}"; then
    return 0
  fi
  if bench_run_contract_strict; then
    log "FATAL: serving-envelope capture failed under the strict run contract."
    log "       A clock-normalized/strict run must record which checkpoint and"
    log "       engine served it (OPENAI_MODEL is only the served ALIAS, identical"
    log "       across contracts v2-v4). Bring vllm-gen up, then relaunch."
    exit 1
  fi
  log "WARN: serving-envelope capture failed (best-effort mode) — the manifest"
  log "      will record no serving envelope for this run."
  return 0
}

# --- step 2: manifest and run-contract hash -------------------------------------
# write_manifest.py exits 3 when the strict gate is armed and a required
# contract field is missing. It still writes the manifest first, so the
# failure stays diagnosable from the artifact. Any other nonzero exit code is
# an ordinary best-effort failure.
bench_write_manifest() {
  local provider_dir="$1" tag="$2" stage="$3"
  local py="${BENCH_PYTHON:-python}"
  local rc=0
  "$py" "$BENCH_ROOT/benchmark/write_manifest.py" \
      --provider_dir "$provider_dir" --run_tag "$tag" --stage "$stage" || rc=$?
  [ "$rc" -eq 0 ] && return 0
  if [ "$rc" -eq 3 ]; then
    log "FATAL: run contract incomplete (write_manifest exit 3) — refusing to"
    log "       generate artifacts that cannot be identified. See the FATAL lines"
    log "       above for the missing fields."
    exit 1
  fi
  log "WARN: manifest write failed (exit $rc)"
  return 0
}

# --- step 3: vLLM token accounting -----------------------------------------------
# Scope is per shard: each container's window covers the whole server. So do
# not sum these files across shards (run_shards.sh writes the wave-level
# file instead). This step is on by default. BENCH_TOKEN_ACCOUNTING=0 turns it
# off.
_bench_tokens_enabled() { [ "${BENCH_TOKEN_ACCOUNTING:-1}" = "1" ]; }
_bench_tokens_start_file() { echo "/tmp/bench_token_start_${1//\//_}.json"; }

bench_tokens_start() {
  _bench_tokens_enabled || return 0
  local provider_dir="$1" tag="$2"
  local py="${BENCH_PYTHON:-python}"
  local start_file; start_file="$(_bench_tokens_start_file "$tag")"
  "$py" "$BENCH_ROOT/benchmark/token_usage.py" snapshot --out "$start_file" \
      || log "WARN: token-usage start snapshot failed (accounting only)"
  return 0
}

bench_tokens_finish() {
  _bench_tokens_enabled || return 0
  local provider_dir="$1" tag="$2"
  local py="${BENCH_PYTHON:-python}"
  local start_file; start_file="$(_bench_tokens_start_file "$tag")"
  local provider; provider="$(basename "$provider_dir")"
  "$py" "$BENCH_ROOT/benchmark/token_usage.py" finish \
      --start "$start_file" --scope shard --provider "$provider" --run_tag "$tag" \
      --out "$provider_dir/Results/token_usage_${tag}.json" \
      || log "WARN: token-usage finish failed (accounting only)"
  return 0
}

# --- convenience: the whole generate-time preamble in one call ----------------
# Order matters here. The envelope sidecar must exist before write_manifest
# runs, because the manifest folds it in and the required contract's serving
# fields come from it. The token snapshot must run before any adapter work.
bench_generate_preamble() {
  local provider_dir="$1" tag="$2"
  bench_capture_serving_envelope "$provider_dir" "$tag"
  bench_write_manifest "$provider_dir" "$tag" generate
  bench_tokens_start "$provider_dir" "$tag"
}
