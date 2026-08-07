#!/usr/bin/env bash
# vllm-gen wedge watchdog. This is operational tooling. It is not part of the harness.
#
# WHY THIS WATCHDOG EXISTS
# vllm-gen sometimes wedges without any error. The symptom is about 100% GPU
# utilization at idle power (about 63W of a 300W limit). The metrics show
# `Avg generation throughput: 0.0 tokens/s` with Running>0. The /health check
# still returns 200. No log shows an error. Every normal health signal reports
# success. No other check can detect the wedge. During the contract-v2 wave,
# the wedge killed two consecutive runs before any persona completed.
# `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` is the suspected fix
# (vllm#40969, #48718). The root cause is not confirmed. This watchdog is the
# safety floor. The widened retry window in answer_env.sh helps too
# (RETRY_TIMES=20, 2-45s backoff, about 6 min). With this watchdog and the
# retry window, a wedge costs a run a 3-5 minute stall, not the whole run.
#
# DETECTION
# All four conditions must hold for CONSEC consecutive polls. One sample is not
# enough because a brief lull between requests can look the same.
#   1. GPU utilization > UTIL_MIN%          — the GPU is spinning
#   2. GPU power       < POWER_MAX W        — but it does no real work (the tell)
#   3. num_requests_running > 0             — vLLM thinks it is serving
#   4. generation_tokens_total is UNCHANGED — it produced nothing since the last poll
# Condition 4 separates a wedge from a server that is slow but still working.
# Conditions 1 and 2 detect the wedge in the first place.
#
# USAGE (run from benchmark/docker/):
#   ./watch_vllm.sh &                 # run in the background and log to stdout
#   WATCH_DRY_RUN=1 ./watch_vllm.sh   # detect and log only, and do not restart
#
#   # watch the gemma-4-12b judge server instead (score stage; port 8002):
#   WATCH_SERVICE=vllm-judge WATCH_METRICS_URL=http://localhost:8002/metrics \
#     WATCH_LOCK_DIR=/tmp/watch_vllm_judge.lock ./watch_vllm.sh &
# WATCH_SERVICE and WATCH_METRICS_URL must name the SAME server. Pointing the
# metrics URL at one server and restarting another would restart a healthy
# process and leave the wedged one running. Give a second instance its own
# WATCH_LOCK_DIR, or the single-instance guard below makes it exit immediately.
set -uo pipefail

POLL_S="${WATCH_POLL_S:-30}"
UTIL_MIN="${WATCH_UTIL_MIN:-90}"
POWER_MAX="${WATCH_POWER_MAX:-90}"
CONSEC="${WATCH_CONSEC:-2}"
DRY_RUN="${WATCH_DRY_RUN:-0}"
METRICS_URL="${WATCH_METRICS_URL:-http://localhost:8000/metrics}"
# The compose service to restart on a confirmed wedge. It was hardcoded to
# vllm-gen until 2026-07-29, when the score stage started running a second
# server (vllm-judge). Both are the same nightly engine on the same SM120 GPU,
# so both can hit the same sampler wedge.
SERVICE="${WATCH_SERVICE:-vllm-gen}"
COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[watchdog $(date -u +%H:%M:%S)] $*"; }

# SINGLE-INSTANCE GUARD — do not remove.
# On 2026-07-21, seven instances ran at the same time. They built up over a day
# of relaunches. An earlier `pkill -f watch_vllm.sh` had silently matched none
# of them. This problem is not only redundant. It BREAKS the detection policy.
# Each instance owns a private `strikes` counter. So N instances that poll at
# different phase offsets let a wedge get confirmed by whichever instance sees
# two consecutive bad samples first. The effective CONSEC becomes 1, not 2.
# Also, several instances can issue overlapping `docker compose restart` calls.
# One call can land while the engine is still loading. A restart during boot
# exhausts the shards' retry budget and kills a run.
# Symptom to recognize: a log shows "WEDGE SUSPECTED (1/2)" with no confirmation
# line, but vllm-gen restarted at that same second. The restart came from
# another instance that logged somewhere else.
# mkdir is atomic on every filesystem here, including MSYS/Windows. `flock` is
# not usable here because this Git-Bash environment does not have it.
LOCK_DIR="${WATCH_LOCK_DIR:-${TMPDIR:-/tmp}/watch_vllm.lock}"
if mkdir "$LOCK_DIR" 2>/dev/null; then
  echo $$ > "$LOCK_DIR/pid"
else
  _owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")"
  if [ -n "$_owner" ] && kill -0 "$_owner" 2>/dev/null; then
    log "another watchdog is already running (pid=$_owner) — exiting."
    log "if that instance is unwanted: kill $_owner && rm -rf $LOCK_DIR"
    exit 0
  fi
  # The lock is stale because the owner process is gone, for example killed or
  # the host rebooted. Take over the lock.
  log "clearing stale lock from dead pid=${_owner:-unknown}"
  echo $$ > "$LOCK_DIR/pid"
fi
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

_metric() {  # $1 = a Prometheus line. Returns its value, or an empty string.
  # Take the LAST whitespace-separated field. Do not use a [0-9.]+ regex.
  # vLLM renders large counters in scientific notation once they pass about
  # 1e6 (for example "vllm:generation_tokens_total{...} 1.088486e+06"). A
  # digits-only regex would silently extract the EXPONENT ("06") instead.
  # This is not a cosmetic bug. The token value would then look constant
  # forever. That makes the "tokens unchanged" condition permanently TRUE and
  # reduces the wedge test to util and power alone. That loses the
  # corroborating signal and risks a false restart of a healthy server. A
  # false positive in the 30-minute soak test caught this bug on 2026-07-21.
  awk '{print $NF}' <<<"$1"
}

_num_eq() {  # Compares numbers. Tolerates scientific notation.
  awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 == b+0)}'
}

log "started: poll=${POLL_S}s util>${UTIL_MIN}% power<${POWER_MAX}W consec=${CONSEC} dry_run=${DRY_RUN}"

strikes=0
prev_tokens=""
while true; do
  gpu="$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "")"
  util="$(echo "$gpu" | awk -F', *' '{print int($1)}')"
  power="$(echo "$gpu" | awk -F', *' '{print int($2)}')"
  metrics="$(curl -s --max-time 10 "$METRICS_URL" 2>/dev/null || echo "")"
  running="$(_metric "$(echo "$metrics" | grep -E '^vllm:num_requests_running' | head -1)")"
  tokens="$(_metric "$(echo "$metrics" | grep -E '^vllm:generation_tokens_total' | head -1)")"

  # Treat any missing reading as inconclusive. Reset the counter instead of
  # risking a false restart.
  if [ -z "${util:-}" ] || [ -z "${power:-}" ] || [ -z "${running:-}" ] || [ -z "${tokens:-}" ]; then
    log "inconclusive sample (util=${util:-?} power=${power:-?} running=${running:-?} tokens=${tokens:-?}) — resetting"
    strikes=0; prev_tokens="$tokens"; sleep "$POLL_S"; continue
  fi

  wedged=0
  if [ "$util" -gt "$UTIL_MIN" ] && [ "$power" -lt "$POWER_MAX" ] \
     && awk "BEGIN{exit !($running > 0)}" \
     && [ -n "$prev_tokens" ] && _num_eq "$tokens" "$prev_tokens"; then
    wedged=1
  fi

  if [ "$wedged" -eq 1 ]; then
    strikes=$((strikes + 1))
    log "WEDGE SUSPECTED (${strikes}/${CONSEC}): util=${util}% power=${power}W running=${running} tokens=${tokens} (unchanged)"
    if [ "$strikes" -ge "$CONSEC" ]; then
      if [ "$DRY_RUN" = "1" ]; then
        log "WEDGE CONFIRMED — dry run, NOT restarting"
      else
        log "WEDGE CONFIRMED — restarting $SERVICE"
        # --profile judge: vllm-judge sits behind that profile, and `compose
        # restart` on a profiled service is a no-op without it. vllm-gen has no
        # profile, so naming one it does not use costs nothing.
        ( cd "$COMPOSE_DIR" && docker compose --profile judge restart "$SERVICE" ) >/dev/null 2>&1 \
          && log "restart issued; active shards should ride it out via RETRY_TIMES backoff" \
          || log "ERROR: restart command failed"
        sleep 120   # give the server time to restart before the next sample
      fi
      strikes=0; prev_tokens=""; sleep "$POLL_S"; continue
    fi
  else
    [ "$strikes" -gt 0 ] && log "recovered before confirmation (util=${util}% power=${power}W)"
    strikes=0
  fi

  prev_tokens="$tokens"
  sleep "$POLL_S"
done
