# clock_sync.sh is a sourced entrypoint fragment for the clock-sync arms.
#
# This file is a companion to benchmark/clock_sync.py, the single WRITER of
# the timestamp file. The shared driver calls that script once per session.
# This fragment owns the environment side. It exports the FAKETIME_* contract
# and seeds the file with REAL time ("+0" means zero offset). Because of this
# seed, any process that boots before the first driver step still boots at the
# real clock (for TLS cert validation and model caches).
#
# Every function is a no-op unless BENCH_CLOCKSYNC=1. Entrypoints call these
# functions only on generate paths. The score and summarize paths stay on the
# real clock by design.
#
# The only per-provider difference is which process gets LD_PRELOAD:
#   mnemosyne   — bench_clocksync_preload runs inside the per-shard subshell,
#                 so the exec'd python (a fully in-process provider) is faked.
#   supermemory — no shell-wide preload. The adapter injects LD_PRELOAD only
#                 into the spawned server child env (_supermemory_server._env).
#   retaindb    — a per-command `env LD_PRELOAD=...` on postmaster and node
#                 only, never shell-wide. The health-wait `date +%s` loops
#                 stay on the real clock.
#
# File format notes (probed 2026-07-24, Debian bookworm libfaketime 0.9.10):
# "@YYYY-MM-DD HH:MM:SS" advances at 1x and re-anchors exactly on each
# rewrite, using the file mtime as the anchor. "+0" means real time.
# FAKETIME_NO_CACHE=1 makes readers re-parse the file on every clock call, for
# live stepping. FAKETIME_DONT_FAKE_MONOTONIC=1 keeps CLOCK_MONOTONIC real, for
# timeouts and sleeps. NO_FAKE_STAT=1 keeps observed file mtimes real, so
# sqlite and other caches do not see 2022 mtimes.

BENCH_LIBFAKETIME="${BENCH_LIBFAKETIME:-/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1}"

bench_clocksync_enabled() { [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; }

# bench_clocksync_prepare <timestamp-file>
# This needs a per-shard file path. Never use a location shared across shards.
# Each shard owns its own timeline.
bench_clocksync_prepare() {
  bench_clocksync_enabled || return 0
  local file="$1"
  if [ -z "$file" ]; then
    echo "[clocksync] FATAL: bench_clocksync_prepare needs a file path" >&2
    exit 2
  fi
  if [ ! -f "$BENCH_LIBFAKETIME" ]; then
    echo "[clocksync] FATAL: libfaketime missing at $BENCH_LIBFAKETIME (rebuild image)" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$file")"
  printf '+0\n' > "$file"
  export BENCH_CLOCKSYNC_FILE="$file"
  export FAKETIME_TIMESTAMP_FILE="$file"
  export FAKETIME_NO_CACHE=1
  export FAKETIME_DONT_FAKE_MONOTONIC=1
  export NO_FAKE_STAT=1
  echo "[clocksync] enabled: file=$file so=$BENCH_LIBFAKETIME (seeded real time)"
}

# This is a shell-wide preload. Every child of the current shell is faked.
# Use it only where that is the intent, such as the Mnemosyne per-shard
# subshell.
bench_clocksync_preload() {
  bench_clocksync_enabled || return 0
  export LD_PRELOAD="$BENCH_LIBFAKETIME"
}

# This is a sanity probe. It verifies that the preload actually changes
# perceived time in this image, then restores the real-time seed. Call it
# after prepare and before boot.
bench_clocksync_probe() {
  bench_clocksync_enabled || return 0
  local file="${BENCH_CLOCKSYNC_FILE:?call bench_clocksync_prepare first}"
  printf '@2019-05-05 12:00:00\n' > "$file"
  local year
  year="$(LD_PRELOAD="$BENCH_LIBFAKETIME" date -u '+%Y')"
  printf '+0\n' > "$file"
  if [ "$year" != "2019" ]; then
    echo "[clocksync] FATAL: probe got year=$year (want 2019) — libfaketime not effective" >&2
    exit 2
  fi
  echo "[clocksync] probe OK"
}
