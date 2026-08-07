#!/usr/bin/env bash
# Container entrypoint for the MemConflict x Mnemosyne benchmark.
#
# This script is the container version of benchmark/run_local.sh and
# run_full_local.sh. It differs in two ways:
#   1. It uses the image's system `python`. It does not need the Windows
#      .venv/Scripts path.
#   2. Endpoints default to host.docker.internal. This lets the harness reach
#      the vLLM servers on the host (vllm-gen :8000, vllm-embed :8001).
#
# Env vars control everything (see docker-compose.yml and README.md):
#   Stage:     STAGE = generate | score | summarize | all   (default: all)
#   Run params: NUM_PERSONAS, NUM_SHARDS, TOP_K, SCORE_WORKERS
#   Features:  EXTRACT=1, MNEMOSYNE_ENHANCED_RECALL=1
#   ML server/models: OPENAI_BASE_URL, OPENAI_MODEL, MNEMOSYNE_EMBEDDING_API_URL,
#                     MNEMOSYNE_EMBEDDING_MODEL, MNEMOSYNE_LLM_BASE_URL, ...
#   Output paths: RESULTS_FILE, SCORES_FILE, CHECKPOINT, SUMMARY_FILE. These
#                 are auto-named by feature tag if unset, so feature runs
#                 never overwrite the baseline.
#
# The script runs any explicit command (`docker run ... image python -c ...`)
# verbatim. This bypasses the stage machinery and helps with one-off
# debugging.
set -euo pipefail

ROOT=/app
cd "$ROOT"

# This sources the shared answer/judge decoding config and the score and
# summarize calls. The fairness contract requires this file to be
# byte-for-byte identical for every provider. The path is relative to THIS
# script, so it resolves no matter what the launch cwd is.
source "$(dirname "${BASH_SOURCE[0]}")/answer_env.sh"

# This sources the clock-sync arm helpers for libfaketime (BENCH_CLOCKSYNC=1).
# The path is relative to THIS script, like answer_env.sh above. Every
# function is a no-op unless BENCH_CLOCKSYNC=1, so a default run is unaffected.
source "$(dirname "${BASH_SOURCE[0]}")/clock_sync.sh"

# This sources the run-contract helpers: serving-envelope capture, the
# manifest and run_contract_hash, and vLLM token accounting. The path is
# relative to THIS script, like the two sources above.
source "$(dirname "${BASH_SOURCE[0]}")/run_contract.sh"

# This sources the named launch presets (PRESET=<name>) and applies them HERE.
# It runs before every `${VAR:-default}` block and every validation gate
# below, so a preset's values feed those defaults instead of conflicting with
# them. A preset that sets BENCH_CLOCKSYNC=1 still reaches the clock-sync TTL
# gate further down. If PRESET is unset, this step is a no-op.
source "$(dirname "${BASH_SOURCE[0]}")/presets.sh"
bench_apply_preset mnemosyne

# --- Answer + judge LLM -> generation server (default: host vllm-gen) ----------
# The answer and judge stages share the endpoint, model, and key. This script
# deliberately does NOT set the DECODING config (temperature, max_tokens,
# thinking, JSON mode) here. bench_answer_env and bench_judge_env
# (answer_env.sh) export decoding config per stage instead, so every provider
# decodes the same way and neither stage's config can leak into the other.
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"           # vLLM ignores this key; the SDK needs a non-empty value
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://host.docker.internal:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.5-4b}"
export MEMCONFLICT_REASONING_EFFORT="${MEMCONFLICT_REASONING_EFFORT:-}"

# --- Mnemosyne embeddings -> embedding server (default: host vllm-embed) -------
export MNEMOSYNE_EMBEDDINGS_VIA_API=1
export MNEMOSYNE_EMBEDDING_API_URL="${MNEMOSYNE_EMBEDDING_API_URL:-http://host.docker.internal:8001/v1}"
export MNEMOSYNE_EMBEDDING_API_KEY="${MNEMOSYNE_EMBEDDING_API_KEY:-local-vllm}"
export MNEMOSYNE_EMBEDDING_MODEL="${MNEMOSYNE_EMBEDDING_MODEL:-gte-modernbert-base}"
export MNEMOSYNE_EMBEDDING_DIM="${MNEMOSYNE_EMBEDDING_DIM:-768}"

# --- Mnemosyne-internal LLM (fact-extraction etc.) -> generation server --------
# The harness calls this LLM only when a feature requests it (for example
# --extract). Setting it for the default baseline is harmless.
export MNEMOSYNE_LLM_ENABLED="${MNEMOSYNE_LLM_ENABLED:-true}"
export MNEMOSYNE_LLM_BASE_URL="${MNEMOSYNE_LLM_BASE_URL:-http://host.docker.internal:8000/v1}"
export MNEMOSYNE_LLM_API_KEY="${MNEMOSYNE_LLM_API_KEY:-local-vllm}"
export MNEMOSYNE_LLM_MODEL="${MNEMOSYNE_LLM_MODEL:-qwen3.5-4b}"
# This records whether the caller set this value explicitly. The CANONICAL
# branch below must be able to raise the default without overwriting a user
# override.
_USER_MNEMOSYNE_LLM_MAX_TOKENS="${MNEMOSYNE_LLM_MAX_TOKENS:-}"
export MNEMOSYNE_LLM_MAX_TOKENS="${MNEMOSYNE_LLM_MAX_TOKENS:-512}"

# --- Fact recall: SURFACE the extracted facts in recall() -----------------------
# --extract (remember(extract=True)) only WRITES facts into Mnemosyne's fact
# tables. beam.recall() merges those facts into results only when
# MNEMOSYNE_FACT_RECALL_ENABLED=1 (it reads that env var directly and defaults
# to off). Without this flag, extraction has zero effect on what the answerer
# retrieves. So this script defaults fact recall to the EXTRACT value: an
# extract run surfaces its facts, and a non-extract run leaves the empty fact
# tables alone. Set the var explicitly to decouple the two.
#
# The default is OFF because that is the Hermes plugin's shipped default. The
# plugin never sets this env var (verified zero hits across
# external/mnemosyne/integrations/hermes/src/), and its README documents the
# default as false. beam.recall() reads the env var directly and defaults to
# off. Leaving it off matches the plugin. It is not a scorer accommodation.
#
# Historical probe (v1, gemma-4-e2b): turning this flag ON merged up to 10
# lossy "conversation stated ..." fact rows into recall (beam.py:6609). This
# filled top-K and dropped SEH@3 to 0.031. The MemConflict judge scores SEH
# *semantically* ("evidence supporting the reference answer... do not require
# exact wording"). It never requires the raw gold turn's exact text. So this
# drop means the fact rows genuinely failed to semantically support the
# answers under judge review. That is worse retrieval quality, not a format
# the scorer rejects. (A separate "AA -> ~0" figure is sometimes attributed to
# this flag, but it actually belongs to the adjacent WM-TTL deletion finding
# below. See docs/BENCHMARK_MATRIX.md. Do not conflate the two.) Extraction is
# still useful because it feeds the veracity conflict detector that drives
# lifecycle retirement, but its facts must NOT be surfaced in recall on this
# path. Set MNEMOSYNE_FACT_RECALL_ENABLED=1 explicitly only if you accept
# that tradeoff.
export MNEMOSYNE_FACT_RECALL_ENABLED="${MNEMOSYNE_FACT_RECALL_ENABLED:-0}"

# --- Fast, durability-free SQLite for the disposable per-persona DBs -----------
# eval_mnemosyne.py wraps sqlite3.connect() at import time, before the lazy
# mnemosyne import. It applies synchronous=OFF, journal_mode=WAL,
# temp_store=MEMORY, and a large cache and mmap to every file-backed
# connection. Mnemosyne opens several connections internally (beam.conn,
# CanonicalStore, and others), and the submodule must not be modified, so
# this central wrapper is the only clean hook. There is no upstream
# MNEMOSYNE_* env var for SQLite pragmas: core/beam.py hardcodes
# journal_mode=WAL, busy_timeout, and foreign_keys, and never touches
# synchronous. The per-persona DBs are throwaway benchmark scratch, so a
# crash just re-runs the shard and durability buys nothing. Meanwhile the
# fsync traffic of 30 parallel shards at SQLite's default synchronous level
# is exactly the disk thrash that choked the previous full run. This is ON
# by default in the container. Set BENCH_SQLITE_FAST=0 to restore stock
# SQLite behavior.
export BENCH_SQLITE_FAST="${BENCH_SQLITE_FAST:-1}"

# --- Lifecycle env guards -------------------------------------------------------
# Lifecycle backdates working_memory.timestamp to the dataset's real dates
# (2022 and later). Mnemosyne's _trim_working_memory() DELETES un-consolidated
# rows older than MNEMOSYNE_WM_TTL_HOURS (default 168h) on the next
# remember(). This would delete almost every backdated row right after
# ingest and gut recall (SEH@3 -> ~0). Raise the TTL well past the simulated
# 2022-to-now span so backdated rows survive. This guard applies to lifecycle
# only; the baseline path uses now() timestamps and is unaffected either way.
# CANONICAL and ORACLE arms build on lifecycle (timestamp restoration and
# surgical retirement), so they force LIFECYCLE=1 here and inherit its env
# guards below.
#   CANONICAL=1 -> --canonical : per-session sleep(force=True) populates
#                  canonical slots through LLM model-refresh and
#                  history-aware canonical retrieval. Tune
#                  MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY_MIN_CONFIDENCE
#                  (upstream default 0.90) if the model's proposals rarely
#                  apply. The team measured the 0.90-clearing behavior on
#                  contract v1 (gemma-4-e2b); re-verify it under contract v2
#                  (qwen3.5-4b).
#   ORACLE=1    -> --oracle    : gold-derived canonical slots (upper bound).
[ "${CANONICAL:-0}" = "1" ] && LIFECYCLE=1
[ "${ORACLE:-0}" = "1" ] && LIFECYCLE=1

# --- Plugin-fidelity + decoupled dataset-time arms ------------------------------
# PLUGIN_CONFIG={off,user,both} -> --plugin_config : this arm writes a
# per-exchange sync_turn, mirroring the Hermes Mnemosyne plugin (fixed
# per-role importances, entity extraction only, temporal recall).
# USE_DATASET_TIME=1 -> --use_dataset_time : this backdates to dataset
# chronology WITHOUT lifecycle retirement. Both arms imply backdated ingest,
# so both need the WM-TTL guard below, the same as LIFECYCLE. The plugin arm
# is mutually exclusive with the extraction and consolidation arms.
# eval_mnemosyne.py's parser.error enforces this too. This script fails fast
# here to avoid launching N doomed shards.
PLUGIN_CONFIG="${PLUGIN_CONFIG:-off}"
if [ "$PLUGIN_CONFIG" != "off" ]; then
  if [ "${EXTRACT:-0}" = "1" ] || [ "${LIFECYCLE:-0}" = "1" ] \
     || [ "${CANONICAL:-0}" = "1" ] || [ "${ORACLE:-0}" = "1" ]; then
    echo "[entrypoint] PLUGIN_CONFIG=$PLUGIN_CONFIG is mutually exclusive with EXTRACT/LIFECYCLE/CANONICAL/ORACLE"; exit 2
  fi
fi
# PLUGIN_PREFETCH_OVERLAY=1 -> --plugin_prefetch_overlay : this replaces the
# plugin arm's read path with the Hermes plugin's FULL prefetch() overlay
# (16-candidate recall, quality and topic filter, canonical merge). Like
# PLUGIN_AUTO_SLEEP, it only means something on the plugin write path, so it
# requires PLUGIN_CONFIG != off. The adapter's parser.error enforces the same
# rule (eval_mnemosyne.py:2341-2348). Checking here fails ONE container
# instead of N argparse-erroring shards.
if [ "${PLUGIN_PREFETCH_OVERLAY:-0}" = "1" ] && [ "$PLUGIN_CONFIG" = "off" ]; then
  echo "[entrypoint] PLUGIN_PREFETCH_OVERLAY=1 requires PLUGIN_CONFIG=user|both"; exit 2
fi
# --- Clock-sync arm gate (BENCH_CLOCKSYNC=1) ------------------------------------
# libfaketime steps each shard's process clock to the dataset's logical
# session date (benchmark/clock_sync.py, driven per session by the shared
# driver). This enforces Mnemosyne's working-memory TTL against perceived
# logical time instead of benchmark wall-clock time. Under the faked clock
# the TTL is LIVE: _trim_working_memory() DELETES un-consolidated
# working-memory rows older than it, and 92.5% of inter-session gaps exceed
# the shipped 168h (median about 29 days). This script allows two arms and
# refuses everything else. It checks these BEFORE the auto-sleep and
# plugin-config checks, so the operator sees the clock-sync-specific reason
# first:
#   * FEATURED (clocksync-ttl): PLUGIN_AUTO_SLEEP=1 with NO explicit
#     MNEMOSYNE_WM_TTL_HOURS. This runs the shipped 168h TTL. Auto-sleep's
#     consolidation exemption is what lets rows survive gaps over 168h,
#     because _trim_working_memory() spares consolidated rows. An explicit
#     TTL override here is ambiguous intent, so this script refuses it.
#     (PLUGIN_AUTO_SLEEP itself still requires PLUGIN_CONFIG != off, enforced
#     just below.)
#   * MINIMAL (clock-normalized minimal rerun, user decision 2026-07-26):
#     PLUGIN_AUTO_SLEEP=0 with an explicit MNEMOSYNE_WM_TTL_HOURS REQUIRED
#     (the minimal preset passes 8760000). Without auto-sleep nothing
#     consolidates, so the shipped 168h TTL would gut recall after the first
#     gap. The long TTL is the same accommodation the pre-clock minimal arm
#     ran; the team kept it deliberately and labeled it in the docs.
#     Requiring it EXPLICIT, instead of defaulting it, keeps the featured
#     arm's shipped-TTL meaning unambiguous.
if [ "${BENCH_CLOCKSYNC:-0}" = "1" ]; then
  if [ "${PLUGIN_AUTO_SLEEP:-0}" = "1" ]; then
    if [ -n "${MNEMOSYNE_WM_TTL_HOURS:-}" ]; then
      echo "[entrypoint] BENCH_CLOCKSYNC=1 + PLUGIN_AUTO_SLEEP=1 refuses an explicit MNEMOSYNE_WM_TTL_HOURS (got '$MNEMOSYNE_WM_TTL_HOURS'): the featured clock-sync arm exists to run Mnemosyne's SHIPPED 168h TTL under the faked logical clock, so overriding the TTL is ambiguous intent"; exit 2
    fi
  else
    if [ -z "${MNEMOSYNE_WM_TTL_HOURS:-}" ]; then
      echo "[entrypoint] BENCH_CLOCKSYNC=1 without PLUGIN_AUTO_SLEEP=1 requires an explicit MNEMOSYNE_WM_TTL_HOURS (the minimal clock-sync preset passes 8760000): under the faked clock the shipped 168h WM TTL is live, 92.5% of inter-session gaps exceed 168h, and without auto-sleep's consolidation exemption recall is gutted after the first gap"; exit 2
    fi
  fi
fi
# PLUGIN_AUTO_SLEEP=1 -> --plugin_auto_sleep : this adds the Hermes plugin's
# real sleep cadence (every 10 exchanges plus per-session-boundary, drained)
# to the plugin arm. It ONLY has meaning on the plugin write path, so it
# requires PLUGIN_CONFIG != off. This script fails fast here, and the
# adapter's parser also enforces it, so N shards do not each argparse-error.
# This flag is inherently exclusive with the sleep-based arms, but
# PLUGIN_CONFIG already forbids those above, so no extra check is needed.
if [ "${PLUGIN_AUTO_SLEEP:-0}" = "1" ] && [ "$PLUGIN_CONFIG" = "off" ]; then
  echo "[entrypoint] PLUGIN_AUTO_SLEEP=1 requires PLUGIN_CONFIG=user|both (the plugin auto-sleep cadence only exists on the plugin write path)"; exit 2
fi
# PLUGIN_SESSION_SLEEP=1 -> --plugin_session_sleep : one sleep(force=True) after
# each session's ingest, before its questions (user ruling 2026-08-02). It
# answers a beam-level behaviour, _trim_working_memory()'s deletion of
# unconsolidated rows past the shipped 168h TTL, so it needs no plugin write
# path and it COMPOSES with PLUGIN_AUTO_SLEEP. It must not combine with the
# sleep-based arms, which run their own per-session sleep. The adapter's parser
# enforces the same rule; failing here stops ONE container instead of N shards.
if [ "${PLUGIN_SESSION_SLEEP:-0}" = "1" ]; then
  if [ "${LIFECYCLE:-0}" = "1" ] || [ "${CANONICAL:-0}" = "1" ] || [ "${ORACLE:-0}" = "1" ]; then
    echo "[entrypoint] PLUGIN_SESSION_SLEEP=1 is mutually exclusive with LIFECYCLE/CANONICAL/ORACLE (those arms already run their own per-session sleep(force=True))"; exit 2
  fi
fi
# Backdated ingest happens for LIFECYCLE, explicit USE_DATASET_TIME, or the
# plugin arm. This script tracks it so the WM-TTL guard covers all three
# cases, not just LIFECYCLE.
BACKDATE=0
if [ "${LIFECYCLE:-0}" = "1" ] || [ "${USE_DATASET_TIME:-0}" = "1" ] || [ "$PLUGIN_CONFIG" != "off" ]; then
  BACKDATE=1
fi

# --- Canonical arm: un-truncate the sleep model-refresh LLM --------------------
# The 512-token MNEMOSYNE_LLM_MAX_TOKENS default above is sized for
# per-message fact extraction. But sleep's model refresh emits ONE
# pretty-printed JSON array for a whole session batch (6-18 proposals at
# about 75-100 tokens each) through local_llm._call_remote_llm, which reads
# this same setting. The dedicated MNEMOSYNE_SLEEP_MODEL_REFRESH_MAX_TOKENS
# only applies on the host-LLM path, which this harness does not use. At 512
# tokens the JSON truncates mid-string, the parse yield drops to about 0, and
# the canonical layer stays empty (measured: 1 proposal in 106 sleeps). A
# value of 3072 fits the longest observed session (about 1300 completion
# tokens) with headroom, well under the vllm-gen context window (contract v2:
# --max-model-len 32768). On evidence gates, measured on contract v1
# (gemma-4-e2b) and needing re-verification under v2 (qwen3.5-4b): the model
# clears the confidence bars (emits 0.9-1.0) but rarely cites 3 or more
# evidence ids. Conflict-supersession, the exact behavior MemConflict scores,
# needs 3 evidence ids by default, so this script relaxes that gate to match
# what the model actually produces.
if [ "${CANONICAL:-0}" = "1" ]; then
  export MNEMOSYNE_LLM_MAX_TOKENS="${_USER_MNEMOSYNE_LLM_MAX_TOKENS:-3072}"
  export MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE="${MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE:-1}"
  export MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE="${MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE:-2}"
fi

# --- Auto-sleep arm: un-truncate the sleep model-refresh LLM ---------------------
# The plugin auto-sleep arm calls the SAME sleep() and sleep_all_sessions()
# model-refresh path the canonical arm does, so it hits the identical
# 512-token trap: sleep's model refresh emits one pretty-printed JSON array
# for a whole session batch through the MNEMOSYNE_LLM_MAX_TOKENS setting, and
# at 512 tokens it truncates mid-string, yielding about 0 parsed proposals
# (see the canonical block above and CLAUDE.md: "MNEMOSYNE_LLM_MAX_TOKENS=512
# silently truncates sleep's model-refresh JSON to zero proposals (needs
# >=2048)"). This script raises the value to 2048, the featured-run LOCKED
# value from user decision 2026-07-22 (the >=2048 floor CLAUDE.md documents,
# not the canonical arm's 3072). It never overwrites an explicit caller
# override. This script DELIBERATELY does NOT relax the evidence and
# confidence gates the canonical arm relaxes. This is a PLUGIN-FIDELITY arm,
# so sleep must run with upstream's default gates exactly as a real Hermes
# deployment would. Only the serving-side truncation defect is corrected
# here.
# PLUGIN_SESSION_SLEEP=1 reaches the SAME model-refresh path, so it needs the
# same 2048 floor even when the auto-sleep cadence is off.
if [ "${PLUGIN_AUTO_SLEEP:-0}" = "1" ] || [ "${PLUGIN_SESSION_SLEEP:-0}" = "1" ]; then
  export MNEMOSYNE_LLM_MAX_TOKENS="${_USER_MNEMOSYNE_LLM_MAX_TOKENS:-2048}"
fi

if [ "$BACKDATE" = "1" ] && [ "${BENCH_CLOCKSYNC:-0}" != "1" ]; then
  # Any backdating arm (LIFECYCLE, USE_DATASET_TIME, or PLUGIN_CONFIG) writes
  # dataset-dated (2022 and later) working_memory rows. _trim_working_memory()
  # compares them against wall-clock now(), so without clock-sync every
  # backdated row is already older than the 168h TTL at ingest and gets
  # deleted. This script raises the TTL past the simulated span so those rows
  # survive.
  #
  # Under BENCH_CLOCKSYNC=1 this script does NOT default the TTL here.
  # libfaketime makes now() track the logical session date, so the faked
  # now() and the backdated rows are finally on the same timeline, and the
  # SHIPPED 168h TTL is meaningful again. The featured clock-sync arm runs
  # with no TTL var at all, because the gate above refused an explicit one.
  # The minimal clock-sync arm's REQUIRED explicit MNEMOSYNE_WM_TTL_HOURS
  # simply passes through untouched.
  # MNEMOSYNE_AUTO_SLEEP_ELIGIBILITY_TTL_HOURS is a separate setting and is
  # left untouched either way.
  export MNEMOSYNE_WM_TTL_HOURS="${MNEMOSYNE_WM_TTL_HOURS:-8760000}"   # ~1000 years
fi
if [ "${LIFECYCLE:-0}" = "1" ]; then
  # sleep() writes episodic summaries. Recall is hybrid over working_memory
  # and episodic, so those summaries would normally take about 30% of top-K.
  # That is real Mnemosyne recall behavior, and the Hermes plugin never sets
  # MNEMOSYNE_EP_LIMIT, so this is NOT a plugin-fidelity setting. It exists
  # solely to ISOLATE the retirement effect for this diagnostic lifecycle
  # arm. EP_LIMIT=0 makes `LIMIT min(0,N)` become `LIMIT 0`, which excludes
  # episodic from recall. This makes lifecycle recall working-memory-only,
  # exactly like the baseline, which never sleeps and so has no episodic.
  # With episodic out of the way, the only remaining difference from baseline
  # is the retirement itself, meaning invalidated stale turns, which is what
  # this arm exists to measure.
  # WARNING: if the lifecycle arm is ever re-run under contract v2, drop this
  # guard, or relabel the arm explicitly as "retirement-isolation diagnostic,
  # non-plugin recall". Do NOT let it silently carry into any new arm. The
  # auto-sleep arm (#7 in docs/BENCHMARK_MATRIX.md) correctly does not
  # inherit it: this guard is gated on LIFECYCLE=1 only, and auto-sleep sets
  # PLUGIN_AUTO_SLEEP=1 with PLUGIN_CONFIG != off, never LIFECYCLE=1.
  export MNEMOSYNE_EP_LIMIT="${MNEMOSYNE_EP_LIMIT:-0}"
fi

# --- Isolated, ephemeral Mnemosyne home (container-local; fresh every run) -----
export HERMES_HOME="${HERMES_HOME:-$ROOT/.hermes}"
export MNEMOSYNE_DATA_DIR="${MNEMOSYNE_DATA_DIR:-$HERMES_HOME/mnemosyne/data}"
mkdir -p "$MNEMOSYNE_DATA_DIR"

log() { echo "[entrypoint] $*"; }

# An explicit command wins: this script runs it and skips the stage machinery entirely.
if [ "$#" -gt 0 ]; then
  log "exec: $*"
  exec "$@"
fi

# --- Run parameters ------------------------------------------------------------
# START_IDX/END_IDX select a dataset persona range [START_IDX, END_IDX), the
# same contract every other provider entrypoint honors (and the one
# preflight_rows.py prefers over NUM_PERSONAS). Unset, the range is
# [0, NUM_PERSONAS) as before. Without this, a single-persona smoke launched
# with NUM_PERSONAS=30 fans out 30 concurrent shards (2026-08-02: one such
# launch exhausted host RAM and was killed by hand).
RANGE_START="${START_IDX:-0}"
RANGE_END="${END_IDX:-${NUM_PERSONAS:-30}}"
TOTAL=$(( RANGE_END - RANGE_START ))   # personas THIS container covers
if [ "$TOTAL" -lt 1 ]; then
  echo "[entrypoint] FATAL: empty persona range [$RANGE_START,$RANGE_END)" >&2
  exit 2
fi
SHARDS="${NUM_SHARDS:-$TOTAL}"     # default: one shard per persona, the maximum granularity
TOPK="${TOP_K:-5}"
STAGE="${STAGE:-all}"

EXTRACT_FLAG=""
[ "${EXTRACT:-0}" = "1" ] && EXTRACT_FLAG="--extract"
# LIFECYCLE=1 -> --lifecycle : per-session veracity consolidation, plus
# sleep(force=True), plus dataset-timestamp restoration, plus retirement
# diagnostics. This flag implies extraction.
LIFECYCLE_FLAG=""
[ "${LIFECYCLE:-0}" = "1" ] && LIFECYCLE_FLAG="--lifecycle"
# --canonical and --oracle imply --lifecycle in eval_mnemosyne.py. This
# script keeps the flags separate so the [DEBUG] banner and diagnostics show
# which arm this is.
CANONICAL_FLAG=""
[ "${CANONICAL:-0}" = "1" ] && CANONICAL_FLAG="--canonical"
ORACLE_FLAG=""
[ "${ORACLE:-0}" = "1" ] && ORACLE_FLAG="--oracle"
if [ -n "$CANONICAL_FLAG" ] && [ -n "$ORACLE_FLAG" ]; then
  log "CANONICAL=1 and ORACLE=1 are mutually exclusive arms"; exit 2
fi
# PLUGIN_CONFIG={user,both} -> --plugin_config : the plugin-fidelity
# sync_turn arm. USE_DATASET_TIME=1 -> --use_dataset_time : backdating
# without retirement.
PLUGIN_FLAG=""
[ "$PLUGIN_CONFIG" != "off" ] && PLUGIN_FLAG="--plugin_config $PLUGIN_CONFIG"
# PLUGIN_AUTO_SLEEP=1 -> --plugin_auto_sleep : the plugin's real sleep cadence.
PLUGIN_AUTO_SLEEP_FLAG=""
[ "${PLUGIN_AUTO_SLEEP:-0}" = "1" ] && PLUGIN_AUTO_SLEEP_FLAG="--plugin_auto_sleep"
# PLUGIN_SESSION_SLEEP=1 -> --plugin_session_sleep : the session-end forced
# sleep. It adds to, and does not replace, the auto-sleep cadence.
PLUGIN_SESSION_SLEEP_FLAG=""
[ "${PLUGIN_SESSION_SLEEP:-0}" = "1" ] && PLUGIN_SESSION_SLEEP_FLAG="--plugin_session_sleep"
# PLUGIN_PREFETCH_OVERLAY=1 -> --plugin_prefetch_overlay : the full plugin read path.
PLUGIN_PREFETCH_OVERLAY_FLAG=""
[ "${PLUGIN_PREFETCH_OVERLAY:-0}" = "1" ] && PLUGIN_PREFETCH_OVERLAY_FLAG="--plugin_prefetch_overlay"
USE_DATASET_TIME_FLAG=""
[ "${USE_DATASET_TIME:-0}" = "1" ] && USE_DATASET_TIME_FLAG="--use_dataset_time"

# Optional smoke-test caps (unset = run the whole persona). MAX_SESSIONS
# trims how many dialogue sessions each persona ingests. MAX_QUESTIONS_PER_SESSION
# trims how many questions each session answers. Set both small for a fast
# end-to-end smoke, for example NUM_PERSONAS=1 MAX_SESSIONS=1
# MAX_QUESTIONS_PER_SESSION=3.
CAP_FLAGS=""
[ -n "${MAX_SESSIONS:-}" ] && CAP_FLAGS="$CAP_FLAGS --max_sessions $MAX_SESSIONS"
[ -n "${MAX_QUESTIONS_PER_SESSION:-}" ] && CAP_FLAGS="$CAP_FLAGS --max_questions_per_session $MAX_QUESTIONS_PER_SESSION"

# This auto-tags outputs so a feature run (extract or enhanced recall) never
# overwrites the default baseline or the committed online results
# (mnemosyne_results.jsonl).
if [ "${CANONICAL:-0}" = "1" ]; then
  TAG="${RUN_TAG:-canonical}"
elif [ "${ORACLE:-0}" = "1" ]; then
  TAG="${RUN_TAG:-oracle}"
elif [ "$PLUGIN_CONFIG" != "off" ] && [ "${PLUGIN_AUTO_SLEEP:-0}" = "1" ]; then
  TAG="${RUN_TAG:-plugin_${PLUGIN_CONFIG}_autosleep}"
elif [ "$PLUGIN_CONFIG" != "off" ] && [ "${PLUGIN_PREFETCH_OVERLAY:-0}" = "1" ]; then
  # This uses a distinct default tag because the overlay changes the READ
  # path. Its results are not comparable to a plain plugin_<config> run and
  # must not overwrite that file.
  TAG="${RUN_TAG:-plugin_${PLUGIN_CONFIG}_prefetch}"
elif [ "$PLUGIN_CONFIG" != "off" ]; then
  TAG="${RUN_TAG:-plugin_${PLUGIN_CONFIG}}"
elif [ "${USE_DATASET_TIME:-0}" = "1" ]; then
  TAG="${RUN_TAG:-datasettime}"
elif [ "${EXTRACT:-0}" = "1" ] || [ "${MNEMOSYNE_ENHANCED_RECALL:-0}" = "1" ] || [ "${LIFECYCLE:-0}" = "1" ]; then
  TAG="${RUN_TAG:-feature}"
else
  TAG="${RUN_TAG:-local}"
fi

RESDIR="$ROOT/mnemosyne/Results"
SCOREDIR="$ROOT/mnemosyne/Scores"

# --- Docker-native scratch (/scratch = benchscratch named volume) --------------
# /app/benchmark is a WINDOWS bind mount. Every write crosses the WSL2 file
# barrier, which is catastrophic for high-churn I/O. Two things used to
# churn:
#   * per-shard cumulative shard_N.jsonl/.json files, REWRITTEN whole
#     (multi-MB) after EVERY persona across up to 30 shards ("30 dbs
#     bottlenecked on disk"),
#   * possibly the per-persona SQLite DBs from tempfile.mkdtemp(). These
#     actually landed in /tmp (overlayfs, VM-local) because TMPDIR was
#     unset, but overlayfs still pays copy-up on every modified page.
# Both now live on the named volume: plain ext4 inside the WSL2 VM, with no
# barrier crossing and no copy-up. The layout is per-tag, so concurrent runs
# with distinct RUN_TAGs never collide. Two runs with the SAME tag still
# collide, as was already true of the old shard dir, so give parallel runs
# distinct tags:
#   $SCRATCH/tmp/$TAG     — TMPDIR: tempfile.mkdtemp() per-persona SQLite DBs
#   $SCRATCH/shards/$TAG  — per-shard cumulative .jsonl/.json rewrites
#   $SCRATCH/hermes/$TAG  — per-shard HERMES_HOME (module-init mnemosyne.db)
# Only the final merged RESULTS_FILE, scores, and the small append-only shard
# logs go to the bind mount, because the host needs to see those live.
# Fallback: outside compose, a plain `docker run` without the volume may find
# /scratch absent or unwritable. In that case this script falls back to /tmp
# (overlayfs, still VM-local and correct, just slower) instead of failing the
# run.
SCRATCH="${SCRATCH:-/scratch}"
if ! mkdir -p "$SCRATCH" 2>/dev/null || [ ! -w "$SCRATCH" ]; then
  SCRATCH="/tmp/benchscratch"
  mkdir -p "$SCRATCH"
  log "WARNING: /scratch unavailable (benchscratch volume not mounted?) — using $SCRATCH (container overlayfs)"
fi

# The volume PERSISTS across runs. This script reclaims THIS tag's scratch
# from any prior run so the volume cannot grow without bound. Growth is
# bounded to one dir set per tag, reaped on the tag's next run. NEVER run
# `rm -rf $SCRATCH` wholesale: a concurrent run with a different tag may be
# live in there right now.
rm -rf "$SCRATCH/tmp/$TAG" "$SCRATCH/hermes/$TAG"
mkdir -p "$SCRATCH/tmp/$TAG"
# tempfile.mkdtemp() in eval_mnemosyne.py creates the per-persona SQLite DBs
# and honors TMPDIR. This single export moves every persona DB onto the
# volume.
export TMPDIR="$SCRATCH/tmp/$TAG"

# Heavy per-shard output rewrites go to the volume. Shard LOGS stay on the
# bind mount (LOGDIR) so the host can tail them live. Logs are small line
# appends, cheap even across the WSL barrier, and host-side monitoring greps
# them.
SHARDDIR="$SCRATCH/shards/$TAG"
LOGDIR="$RESDIR/shards/$TAG"
mkdir -p "$SHARDDIR" "$LOGDIR" "$SCOREDIR"

RESULTS_FILE="${RESULTS_FILE:-$RESDIR/mnemosyne_results_${TAG}.jsonl}"
SCORES_FILE="${SCORES_FILE:-$SCOREDIR/mnemosyne_${TAG}_eval_scores.jsonl}"
CHECKPOINT="${CHECKPOINT:-$SCOREDIR/${TAG}_judged_checkpoint.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$SCOREDIR/summary_${TAG}.json}"

do_generate() {
  # This sets the canonical ANSWER decoding, identical across providers, and
  # writes the best-effort manifest.
  bench_answer_env
  # This captures the serving envelope, the manifest with the run-contract
  # hash, and the token-accounting start snapshot. It runs HERE, outside and
  # before the per-shard subshells that get the libfaketime preload
  # (BENCH_CLOCKSYNC=1), so its provenance timestamps are real wall-clock
  # time, not a faked logical session date. Under STRICT_RUN_CONTRACT=1 or
  # BENCH_CLOCKSYNC=1, an incomplete run contract or a failed
  # serving-envelope capture ABORTS here instead of warning.
  bench_generate_preamble "$ROOT/mnemosyne" "$TAG"
  log "GENERATE personas=$TOTAL shards=$SHARDS top_k=$TOPK extract=${EXTRACT:-0} enhanced_recall=${MNEMOSYNE_ENHANCED_RECALL:-0} lifecycle=${LIFECYCLE:-0} canonical=${CANONICAL:-0} oracle=${ORACLE:-0} use_dataset_time=${USE_DATASET_TIME:-0} plugin_config=${PLUGIN_CONFIG} plugin_auto_sleep=${PLUGIN_AUTO_SLEEP:-0} plugin_session_sleep=${PLUGIN_SESSION_SLEEP:-0} plugin_prefetch_overlay=${PLUGIN_PREFETCH_OVERLAY:-0} backdate=${BACKDATE} fact_recall=${MNEMOSYNE_FACT_RECALL_ENABLED}${CAP_FLAGS:+ caps:$CAP_FLAGS} tag=$TAG sqlite_fast=${BENCH_SQLITE_FAST} scratch=$SCRATCH thinking=${MEMCONFLICT_ENABLE_THINKING} answer_max_tokens=${OPENAI_MAX_TOKENS} -> $RESULTS_FILE"
  # For a fresh run, this drops stale shard fragments so the merge cannot
  # pick up a prior run's output. Shard outputs live on the scratch volume
  # now, and logs live on the bind mount, so this cleans both. It cleans only
  # this tag's files; see the scratch-layout comment above.
  rm -rf "$SHARDDIR"
  mkdir -p "$SHARDDIR"
  rm -f "$LOGDIR"/shard_*.log
  local PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))   # this rounds up (ceiling division)
  local pids=() s START END
  for ((s=0; s<SHARDS; s++)); do
    START=$(( RANGE_START + s * PER )); END=$(( START + PER ))
    (( END > RANGE_END )) && END=$RANGE_END
    (( START >= END )) && continue
    log "  shard $s: personas [$START,$END)"
    # This sets a per-shard HERMES_HOME and MNEMOSYNE_DATA_DIR. Mnemosyne runs
    # init_db() at import time on $MNEMOSYNE_DATA_DIR/mnemosyne.db. With a
    # shared dir, N concurrent shards race on its non-atomic schema migration
    # (ALTER TABLE ... ADD COLUMN) and die with "duplicate column name".
    # Isolating the dir per shard removes the race. Each persona's real
    # memory DB is already a separate tempfile, so this module-init DB is
    # just scratch, and it lives on the scratch volume too. This runs in a
    # subshell so the override stays local.
    (
      export HERMES_HOME="$SCRATCH/hermes/$TAG/shard_${s}"
      export MNEMOSYNE_DATA_DIR="$HERMES_HOME/mnemosyne/data"
      mkdir -p "$MNEMOSYNE_DATA_DIR"
      # Clock-sync (BENCH_CLOCKSYNC=1): this seeds this shard's OWN timestamp
      # file with real time, then LD_PRELOADs libfaketime so the exec'd
      # python, a fully in-process provider, steps its own clock per session
      # through clock_sync.py. Using a per-shard file means each of the N
      # shard processes owns an independent timeline. The probe verifies
      # that libfaketime actually bends perceived time in this shard before
      # boot, and fails the shard if not. All three calls are no-ops unless
      # BENCH_CLOCKSYNC=1.
      bench_clocksync_prepare "$HERMES_HOME/faketime.rc"
      bench_clocksync_probe
      bench_clocksync_preload
      exec python -u "$ROOT/mnemosyne/eval_mnemosyne.py" \
          --start_idx "$START" --end_idx "$END" --top_k "$TOPK" $EXTRACT_FLAG $LIFECYCLE_FLAG $CANONICAL_FLAG $ORACLE_FLAG $USE_DATASET_TIME_FLAG $PLUGIN_FLAG $PLUGIN_AUTO_SLEEP_FLAG $PLUGIN_SESSION_SLEEP_FLAG $PLUGIN_PREFETCH_OVERLAY_FLAG $CAP_FLAGS \
          --output_jsonl_path "$SHARDDIR/shard_${s}.jsonl" \
          --output_json_path "$SHARDDIR/shard_${s}.json"
    ) > "$LOGDIR/shard_${s}.log" 2>&1 &
    pids+=($!)
  done
  log "launched ${#pids[@]} shards; waiting..."
  # This writes periodic progress to container stdout, visible in `docker
  # logs`. Shards each write their own $LOGDIR/shard_N.log, so without this
  # the container log would only show the launch lines. PROGRESS_INTERVAL
  # sets the interval in seconds (default 60; 0 disables it). It counts
  # completed shards, personas done, and any shard errors. Note the split:
  # this counts shard OUTPUTS on the volume ($SHARDDIR), and reads the
  # done/error greps from the LOGS on the bind mount ($LOGDIR). These are the
  # same files the host-side monitor greps, so both views agree.
  local prog_pid=""
  if [ "${PROGRESS_INTERVAL:-60}" != "0" ]; then
    (
      # The counters below legitimately "fail" while nothing matches yet: ls
      # and grep exit nonzero on zero matches, and the outer set -euo
      # pipefail is inherited by subshells. That silently killed the
      # heartbeat on its first tick in the run3_oracle run. The heartbeat
      # must outlive empty globs, so this subshell alone drops -e and
      # pipefail.
      set +e +o pipefail
      while :; do
        sleep "${PROGRESS_INTERVAL:-60}"
        done_sh=$(ls "$SHARDDIR"/shard_*.jsonl 2>/dev/null | wc -l)
        done_p=$(grep -h 'done - sessions' "$LOGDIR"/shard_*.log 2>/dev/null | wc -l)
        errs=$(grep -l 'Traceback' "$LOGDIR"/shard_*.log 2>/dev/null | wc -l)
        log "progress: shards_complete=${done_sh}/${SHARDS} personas_done=${done_p}/${TOTAL} shard_errors=${errs}"
      done
    ) &
    prog_pid=$!
  fi
  local fail=0 p
  for p in "${pids[@]}"; do
    if ! wait "$p"; then fail=1; log "shard pid $p FAILED (see $LOGDIR/*.log)"; fi
  done
  [ -n "$prog_pid" ] && kill "$prog_pid" 2>/dev/null || true
  # This merges shard outputs, in shard order, into the tagged results file.
  # It is the ONE deliberate volume-to-bind-mount copy: a single sequential
  # write of the merged file, instead of 30 shards times N personas of
  # rewrites.
  : > "$RESULTS_FILE"
  for ((s=0; s<SHARDS; s++)); do
    [ -f "$SHARDDIR/shard_${s}.jsonl" ] && cat "$SHARDDIR/shard_${s}.jsonl" >> "$RESULTS_FILE"
  done
  local merged_rows
  merged_rows=$(wc -l < "$RESULTS_FILE")
  log "merged ${merged_rows} persona rows -> $RESULTS_FILE"
  # This closes the token-accounting window, since all shards have exited by
  # now. It never affects the stage's success: a failed snapshot only writes
  # valid:false.
  bench_tokens_finish "$ROOT/mnemosyne" "$TAG"
  # This HARD-FAILS the stage on shard failure or a short merge. Previously
  # `fail` was only LOGGED here and the function returned 0, so the
  # container exited 0 even when every shard had died. The team verified
  # this the hard way: a vllm-gen wedge killed all 30 shards with
  # APIConnectionError, the merged file had ZERO rows, and the run still
  # reported success. That is the same silent-success class as the adapters'
  # exit-0-on-fatal-error, fixed separately: the adapter now exits nonzero,
  # but its status was being swallowed right here by the wait loop. The row
  # assertion is the behavioral half of the check, because shards can also
  # "succeed" while writing nothing, so this script checks both. STAGE=all
  # then stops instead of moving on to score an empty file. The pre-score
  # row gate would also catch it, but a generate stage must not claim
  # success when it produced nothing.
  if [ "$fail" -eq 0 ] && [ "$merged_rows" -eq "$TOTAL" ]; then
    log "all shards OK (${merged_rows}/${TOTAL} personas)"
    return 0
  fi
  if [ "$fail" -ne 0 ]; then
    log "FATAL: one or more shards FAILED -- see $LOGDIR/shard_*.log"
  fi
  if [ "$merged_rows" -ne "$TOTAL" ]; then
    log "FATAL: merged ${merged_rows} persona rows but expected ${TOTAL}"
  fi
  log "GENERATE STAGE FAILED (tag=$TAG) -- refusing to report success"
  return 1
}

do_score() {
  # The shared judge env, score call, and score-stage manifest all live in
  # answer_env.sh's run_score, identical for every provider. The manifest is
  # written there AFTER bench_judge_env so it records the judge decoding.
  run_score "$ROOT/mnemosyne" "$TAG"
}

do_summarize() {
  run_summarize "$ROOT/mnemosyne" "$TAG"
}

case "$STAGE" in
  generate)  do_generate ;;
  score)     do_score ;;
  summarize) do_summarize ;;
  all)       do_generate; do_score; do_summarize ;;
  *) log "unknown STAGE='$STAGE' (expected: generate|score|summarize|all)"; exit 2 ;;
esac
log "STAGE=$STAGE complete."
