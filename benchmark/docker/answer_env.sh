# shellcheck shell=bash
# This file holds the shared answer and judge decoding config. It also holds
# the score and summarize invocation code for every provider entrypoint
# (entrypoint.mnemosyne.sh / .hindsight.sh / .retaindb.sh).
# A caller must source this file. Do not run it on its own.
#
# WHY THIS FILE EXISTS (fairness contract): the benchmark compares memory
# providers under an identical harness. The answer LLM decoding settings
# (temperature, max_tokens, thinking) and the judge LLM decoding settings
# must be byte-for-byte the same for every provider. Otherwise a provider's
# score reflects its decoding config, not its memory quality. In the past
# only entrypoint.mnemosyne.sh pinned these settings. The hindsight and
# retaindb entrypoints exported nothing. Their answer calls ran at the
# vLLM server defaults (uncapped max_tokens, server-default temperature)
# while Mnemosyne ran at temperature 0.2. This was a verified fairness bug.
# Centralizing the exports here restores the contract: one canonical config,
# with no per-provider drift. eval_common.Generate_Answer_With_Retrieved_Memory
# calls the upstream llm_request() with no explicit temperature or max_tokens.
# So these env vars (read by MemConflict's llm_request and by
# benchmark/llm_reasoning.py) control decoding fully for all three providers.
#
# API:
#   bench_answer_env                      export the canonical ANSWER decoding config (before generate)
#   bench_judge_env                       export the canonical JUDGE decoding config (before score)
#   run_score     <provider_dir> <tag>    call benchmark/score_resumable.py
#   run_summarize <provider_dir> <tag>    call benchmark/summarize_scores.py
#
# Each stage re-exports its full decoding config. So the judge config never
# depends on whether the generate stage ran first in the same shell (STAGE=all).
#
# Per-provider knobs that legitimately differ (not decoding, not fairness):
#   SCORE_WORKERS  judge concurrency        BENCH_PYTHON  interpreter (python|python3)

# Repo root (/app in-container): answer_env.sh lives at <root>/benchmark/docker/.
# So the root is two levels up. This works from any launch cwd.
_BENCH_ANSWER_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$(cd "$_BENCH_ANSWER_ENV_DIR/../.." && pwd)}"

# --- survive a vllm-gen restart (shared by EVERY provider, both stages) --------
# The upstream MemConflict llm_request already wraps every answer and judge call
# in tenacity: retry_if_exception_type(Exception),
# wait_random_exponential(WAIT_TIME_LOWER, WAIT_TIME_UPPER),
# stop_after_attempt(RETRY_TIMES). Tenacity reads all three from env at import
# (external/MemConflict/Evaluation/llm_request.py:191-199). So no new retry code
# is needed, and external/ stays untouched. The defaults (5 attempts, 1-8s
# backoff) are too narrow. They exhaust in about 10-30s, well inside the ~120s a
# vllm-gen restart takes to come back healthy. That gap is exactly how a wedge
# restart killed all 30 shards of run #1 with APIConnectionError.
# About 20 attempts capped at 45s gave roughly 6 min of expected cumulative
# backoff. We RAISED the value to 40 (about 13 min) on 2026-07-21, after 6 min
# proved insufficient against BACK-TO-BACK outages: the 15:30:50 wedge, the
# watchdog restart at 15:32:08, and a --force-recreate at 15:36 left vllm-gen
# unusable from about 15:30 to 15:39. From a shard's point of view that is one
# continuous ~8 min gap. 22 of 30 shards exhausted their budget and died with
# APIConnectionError, taking the whole run with them (0 rows banked). The lesson
# is that the budget must cover the WORST realistic cluster of outages, not a
# single clean restart. A plain restart takes about 3 min. A wedge plus watchdog
# recovery takes about 8 min. These outages can overlap with an operator-started
# recreate. 13 min covers that case with margin.
# Do NOT raise the value much further. The retry window also decides how long a
# genuinely fatal misconfiguration (a 400 error, a bad model name) stays
# invisible. The watchdog already bounds a wedge to about 3-8 min. So a budget
# beyond about 15 min buys nothing and only delays real failures.
# The result is that a shard STALLS through a serving restart instead of dying.
# Safe against corruption: llm_request is a stateless request and response.
# eval_common writes Model_Answer only after a successful return. The scorer
# checkpoints per question only on success. So a retry cannot double-count or
# persist a partial answer. A retried call may sample a different answer than
# the aborted call would have. This is consistent with the locked
# unseeded-sampling decision. Accepted cost: a genuinely deterministic error
# (for example a 400) now takes about 13 min to surface instead of about 30s.
# We export the retry policy here, not per-provider, so it stays identical for
# all three providers. This is the same reason the decoding config lives here.
export RETRY_TIMES="${RETRY_TIMES:-40}"
export WAIT_TIME_LOWER="${WAIT_TIME_LOWER:-2}"
export WAIT_TIME_UPPER="${WAIT_TIME_UPPER:-45}"

# --- canonical ANSWER decoding -------------------------------------------------
# THINKING defaults to 1: thinking ON is the canonical answer config. The model
# reasons in a private think block. vllm-gen's CONTRACT v2
# `--reasoning-parser qwen3` strips that block out, so the answer text stays
# clean. The think trace shares the completion token budget, so the cap is
# generous: 16384 with thinking on and 2048 with it off. See the CONTRACT v2
# comment below for why these values grew from v1's 3072/1024. The qwen3
# parser has no bounded reasoning-effort knob on the answer path, unlike the
# judge, so headroom is what prevents truncation.
# Escape hatch (documented, rarely needed, keeps a single override without
# reintroducing per-provider drift):
#   BENCH_ANSWER_MAX_TOKENS  / BENCH_ANSWER_TEMPERATURE
bench_answer_env() {
  local thinking="${THINKING:-1}"
  # Answer generation is free text. An inherited or injected judge-stage
  # MEMCONFLICT_JSON_MODE would force JSON output and suppress thinking
  # (llm_reasoning.py). That would silently change answer decoding, so clear it.
  unset MEMCONFLICT_JSON_MODE
  export MEMCONFLICT_ENABLE_THINKING="$thinking"
  # CONTRACT v2 sampling uses the Qwen3.5 model card's recommended sets
  # (https://huggingface.co/Qwen/Qwen3.5-4B). The v1 canonical temp 0.2 is a
  # rumination trigger on Qwen thinking models: near-greedy decoding loops the
  # think channel. We measured 5 of 7 smoke answers truncated to empty at a
  # 3072 cap, and one still empty at 8192. presence_penalty 1.5 is the card's
  # loop breaker. The caps stay generous because the think trace shares the
  # completion budget; the cloud mimo runs showed the same need.
  if [ "$thinking" = "1" ]; then
    # Card profile "thinking mode, general tasks": temp 1.0, top_p 0.95,
    # top_k 20, min_p 0, presence_penalty 1.5.
    export OPENAI_TEMPERATURE="${BENCH_ANSWER_TEMPERATURE:-1.0}"
    export MEMCONFLICT_TOP_P="${BENCH_ANSWER_TOP_P:-0.95}"
    export MEMCONFLICT_TOP_K="${BENCH_ANSWER_TOP_K:-20}"
    export MEMCONFLICT_MIN_P="${BENCH_ANSWER_MIN_P:-0}"
    export MEMCONFLICT_PRESENCE_PENALTY="${BENCH_ANSWER_PRESENCE_PENALTY:-1.5}"
    export OPENAI_MAX_TOKENS="${BENCH_ANSWER_MAX_TOKENS:-16384}"
  else
    # Card profile "instruct (non-thinking) mode, general tasks": temp 0.7,
    # top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5.
    export OPENAI_TEMPERATURE="${BENCH_ANSWER_TEMPERATURE:-0.7}"
    export MEMCONFLICT_TOP_P="${BENCH_ANSWER_TOP_P:-0.8}"
    export MEMCONFLICT_TOP_K="${BENCH_ANSWER_TOP_K:-20}"
    export MEMCONFLICT_MIN_P="${BENCH_ANSWER_MIN_P:-0}"
    export MEMCONFLICT_PRESENCE_PENALTY="${BENCH_ANSWER_PRESENCE_PENALTY:-1.5}"
    export OPENAI_MAX_TOKENS="${BENCH_ANSWER_MAX_TOKENS:-2048}"
  fi
}

# --- canonical JUDGE decoding --------------------------------------------------
# JSON mode is on (guided-JSON judge output), temperature is 0.2, and the
# output budget is generous. Under CONTRACT v2 (qwen3 parser), the judge
# THINKS at a bounded LOW effort before it emits schema-clamped JSON.
# Grey-area correctness judgments benefit from a short deliberation, and the
# qwen3 reasoning parser enforces the JSON grammar on the post-reasoning
# content. The v1/gemma4 parser could not combine the two, which is why v1
# judges ran thinking-off. JUDGE_THINKING=0 restores the thinking-off judge.
# JUDGE_REASONING_EFFORT bounds the trace (low|medium).
# Every export is set explicitly, so the judge config stays fully
# self-contained and does not depend on the order of the generate stage.
bench_judge_env() {
  export MEMCONFLICT_JSON_MODE=1
  export MEMCONFLICT_ENABLE_THINKING="${JUDGE_THINKING:-1}"
  export MEMCONFLICT_JSON_THINKING="${JUDGE_THINKING:-1}"
  export MEMCONFLICT_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-low}"
  # CONTRACT v2 judge sampling uses the Qwen3.5 card's "thinking mode, precise
  # tasks" set: temp 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
  # The v1 judge ran temp 0.2 thinking-off on gemma. Near-greedy decoding
  # ruminates on Qwen thinking models, and a verdict is a precise task. So the
  # card's precise set became the v2 canonical set.
  export OPENAI_TEMPERATURE="${BENCH_JUDGE_TEMPERATURE:-0.6}"
  export MEMCONFLICT_TOP_P="${BENCH_JUDGE_TOP_P:-0.95}"
  export MEMCONFLICT_TOP_K="${BENCH_JUDGE_TOP_K:-20}"
  export MEMCONFLICT_MIN_P="${BENCH_JUDGE_MIN_P:-0}"
  export MEMCONFLICT_PRESENCE_PENALTY="${BENCH_JUDGE_PRESENCE_PENALTY:-0}"
  # Generous cap: the thinking trace shares the completion budget. Neither
  # reasoning_effort nor a template budget bounds Qwen3.5 thinking on this
  # build (verified 2026-07-20), so headroom, not the cap, should decide.
  export OPENAI_MAX_TOKENS="${SCORE_MAX_TOKENS:-16384}"
}

# --- provider output-path derivation -------------------------------------------
# This is the exact file naming every entrypoint uses today. It derives from
# the provider dir basename (mnemosyne|hindsight|retaindb) plus RUN_TAG. An
# explicit RESULTS_FILE / SCORES_FILE / CHECKPOINT / SUMMARY_FILE env override
# still wins: the generate stage may have written to an overridden
# RESULTS_FILE, and a STAGE=score re-run must read that same path. This sets
# _BENCH_PROVIDER, _BENCH_RESULTS, _BENCH_SCORES, _BENCH_CHECKPOINT, and
# _BENCH_SUMMARY.
_bench_paths() {
  local provider_dir="$1" tag="$2"
  _BENCH_PROVIDER="$(basename "$provider_dir")"
  local resdir="$provider_dir/Results" scoredir="$provider_dir/Scores"
  _BENCH_RESULTS="${RESULTS_FILE:-$resdir/${_BENCH_PROVIDER}_results_${tag}.jsonl}"
  _BENCH_SCORES="${SCORES_FILE:-$scoredir/${_BENCH_PROVIDER}_${tag}_eval_scores.jsonl}"
  _BENCH_CHECKPOINT="${CHECKPOINT:-$scoredir/${tag}_judged_checkpoint.jsonl}"
  _BENCH_SUMMARY="${SUMMARY_FILE:-$scoredir/summary_${tag}.json}"
}

# --- shared SCORE invocation ---------------------------------------------------
# This runs an identical judge env and an identical score_resumable.py call
# for every provider. SCORE_WORKERS is the only per-provider knob (judge
# concurrency). It is also score_resumable.py's own env default (24), so
# passing it explicitly here keeps today's behavior. BENCH_PYTHON lets the
# retaindb image (python3 only) override the interpreter.
#
# CONCURRENCY / TIMEOUT PAIR (raised from 24 to 40 on 2026-07-21, user ruling).
# These two values MUST move together. Per-call judge latency scales with
# concurrency (about 146s at 24 workers, measured over 600 questions). So
# raising workers without raising the timeout just converts extra load into
# TimeoutError, then retry, then MORE load. That cascade is what stopped Arm B.
# It is also why the old value of 24 sat just under the 23-40 concurrent band
# flagged as Class C. We measured that band while the flashinfer wedge was
# still live, and a sampler bug now fixed (e2d5f09) confounded its latency. So
# 40 is back on the table, but only with headroom: 600s is about 2.5x the
# about 243s that 40 workers projects to. If you ever lower the timeout, lower
# workers with it. benchmark/llm_reasoning.py honors MEMCONFLICT_REQUEST_TIMEOUT.
# Upstream hardcodes 300s as a default argument, and env alone cannot reach it.
run_score() {
  local provider_dir="$1" tag="$2"
  _bench_paths "$provider_dir" "$tag"
  bench_judge_env
  export MEMCONFLICT_REQUEST_TIMEOUT="${MEMCONFLICT_REQUEST_TIMEOUT:-600}"
  local py="${BENCH_PYTHON:-python}"
  # Write the score-stage manifest AFTER bench_judge_env. That way the env
  # snapshot records the judge decoding actually used (json_mode=1, thinking
  # on, 16384, temp 0.6 -- CONTRACT v2, see bench_judge_env above), not
  # leftover answer-stage state. Centralizing this call here keeps it
  # identical for every provider.
  "$py" "$BENCH_ROOT/benchmark/write_manifest.py" --provider_dir "$provider_dir" \
      --run_tag "$tag" --stage score || echo "[answer_env] WARN: score manifest write failed"
  # --- PRE-SCORE ROW GATE (fairness/cost gate) -----------------------------
  # A truncated or empty generate stage must not silently consume hours of
  # judge GPU on a partial Results file. The qwen empty-answer spiral is a
  # KNOWN contract-v2 failure mode. This gate FAILS the stage (it propagates
  # through the entrypoint's `set -e`) instead of just warning. SKIP_ROW_GATE=1
  # is the documented escape hatch for a deliberately partial re-score.
  if [ "${SKIP_ROW_GATE:-0}" = "1" ]; then
    echo "[answer_env] SKIP_ROW_GATE=1 -- bypassing the pre-score row gate"
  else
    "$py" "$BENCH_ROOT/benchmark/preflight_rows.py" --results_file "$_BENCH_RESULTS"
  fi
  echo "[answer_env] SCORE ${_BENCH_RESULTS} -> ${_BENCH_SCORES}" \
       "(workers=${SCORE_WORKERS:-40} timeout=${MEMCONFLICT_REQUEST_TIMEOUT}s" \
       "json_mode=${MEMCONFLICT_JSON_MODE} max_tokens=${OPENAI_MAX_TOKENS} temp=${OPENAI_TEMPERATURE})"
  "$py" -u "$BENCH_ROOT/benchmark/score_resumable.py" \
      --input_file "$_BENCH_RESULTS" \
      --output_file "$_BENCH_SCORES" \
      --checkpoint "$_BENCH_CHECKPOINT" \
      --workers "${SCORE_WORKERS:-40}"
}

# --- shared SUMMARIZE invocation -----------------------------------------------
# This aggregation is provider-agnostic. --system is the provider basename.
# For mnemosyne that equals summarize_scores.py's default ("mnemosyne"), so
# this call is identical to the old mnemosyne call that omitted --system.
run_summarize() {
  local provider_dir="$1" tag="$2"
  _bench_paths "$provider_dir" "$tag"
  local py="${BENCH_PYTHON:-python}"
  # --track and --lifecycle_provenance (summarize_scores.py) label a summary
  # with the arm that produced it (for example TRACK=lifecycle,
  # LIFECYCLE_PROVENANCE=canonical). These flags existed before but were never
  # plumbed through here, so every summary's provenance label was dead code.
  # Both are optional env vars, passed only when set. An unset
  # TRACK/LIFECYCLE_PROVENANCE reproduces today's exact call.
  local extra_args=()
  [ -n "${TRACK:-}" ] && extra_args+=(--track "$TRACK")
  [ -n "${LIFECYCLE_PROVENANCE:-}" ] && extra_args+=(--lifecycle_provenance "$LIFECYCLE_PROVENANCE")
  echo "[answer_env] SUMMARIZE ${_BENCH_SCORES} -> ${_BENCH_SUMMARY} (system=${_BENCH_PROVIDER}" \
       "track=${TRACK:-none} lifecycle_provenance=${LIFECYCLE_PROVENANCE:-none})"
  "$py" "$BENCH_ROOT/benchmark/summarize_scores.py" \
      --scores_file "$_BENCH_SCORES" \
      --out_json "$_BENCH_SUMMARY" \
      --system "$_BENCH_PROVIDER" \
      "${extra_args[@]}"
}

# --- shared STAGE dispatch -------------------------------------------------------
# entrypoint.hindsight.sh, .retaindb.sh, .retaindb-server.sh, .mem0.sh, and
# .supermemory.sh all ended with a byte-similar
#   case "$STAGE" in generate) <gen>;; score) run_score ...;; summarize) run_summarize ...;
#   all) <gen>; run_score ...; run_summarize ...;; *) log "unknown STAGE=..."; exit 2;; esac
#   log "done (stage=$STAGE, tag=$TAG)."
# block. GENERATE_FN names a function the CALLER already defined (for example
# do_generate, or a small wrapper like retaindb-server's start_server plus
# do_generate) to invoke for the generate step. run_stage calls run_score and
# run_summarize itself, so per-entrypoint do_score/do_summarize wrappers are
# no longer needed at the call site. This relies on bash's dynamic function
# lookup: `log` here resolves to whichever `log()` the CALLING entrypoint
# defined earlier in the same shell (this file is sourced, never executed
# standalone). So each provider's own log prefix stays exactly as it was when
# the case block lived inline in that file.
#
# entrypoint.mnemosyne.sh is NOT converted to this helper. Its own STAGE
# dispatch emits different text on both paths this block prints on: its
# unknown-STAGE message reads "unknown STAGE='$STAGE' (expected: ...)", quoted
# differently from "unknown STAGE=$STAGE (expected ...)" above, and its final
# line reads "STAGE=$STAGE complete." rather than "done (stage=$STAGE,
# tag=$TAG)." above. Routing it through run_stage would change what gets
# printed. This cleanup must not make that behavior change.
run_stage() {
  local provider_dir="$1" tag="$2" generate_fn="$3"
  case "$STAGE" in
    generate)  "$generate_fn" ;;
    score)     run_score "$provider_dir" "$tag" ;;
    summarize) run_summarize "$provider_dir" "$tag" ;;
    all)       "$generate_fn"; run_score "$provider_dir" "$tag"; run_summarize "$provider_dir" "$tag" ;;
    *)         log "unknown STAGE=$STAGE (expected generate|score|summarize|all)"; exit 2 ;;
  esac
  log "done (stage=$STAGE, tag=$TAG)."
}

# --- unset set-but-empty env vars for a given PREFIX ------------------------------
# Some vendor Config.from_env()-style parsers do, for example,
# int(os.getenv(VAR, str(DEFAULT))). This substitutes DEFAULT only when VAR
# is UNSET. Docker compose's `environment:` block always SETS every key it
# lists. So an unconfigured `${PREFIX_X:-}` entry reaches the container as a
# real, present, EMPTY string, not an absence, and int("") raises an error.
# Call this function before such a parser runs, so the parser sees a true
# absence instead. It is shared by entrypoint.hindsight.sh (HINDSIGHT_API_)
# and entrypoint.retaindb-server.sh (RETAINDB_). It unsets only a var whose
# value is the empty string, never a var that is already unset.
unset_empty_env_with_prefix() {
  local prefix="$1" _v _val
  for _v in $(env | grep -oE "^${prefix}[A-Z_0-9]+" || true); do
    _val="${!_v:-}"
    if [ -z "$_val" ]; then
      unset "$_v"
    fi
  done
}

# --- per-run DB name derivation (RUN_TAG sanitize -> db name) ---------------------
# This function is shared by entrypoint.hindsight.sh and
# entrypoint.retaindb-server.sh. Given a db-name PREFIX (for example
# "hindsight_" or "retaindb_") and the default name to use when RUN_TAG is
# empty, it derives the per-run database name into _BENCH_PG_DB. It lowercases
# RUN_TAG, maps every non-[a-z0-9_] char to "_", then adds the prefix. This is
# the only genuinely identical fragment between the two call sites. The
# surrounding create-db and already-exists guard logic differs in its
# WARN/FATAL wording: Hindsight's text talks about banks and a consolidation
# sweep re-processing them, RetainDB-server's talks about memories. That is a
# real product difference, not accidental duplication, so it stays inline in
# each entrypoint rather than being forced through a shared message.
bench_pg_db_name() {
  local prefix="$1" default_name="$2"
  local _raw_tag="${RUN_TAG:-}"
  if [ -z "$_raw_tag" ]; then
    _BENCH_PG_DB="$default_name"
  else
    local _san
    _san="$(printf '%s' "$_raw_tag" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_]/_/g')"
    _BENCH_PG_DB="${prefix}${_san}"
  fi
}
