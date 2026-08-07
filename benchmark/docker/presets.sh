# shellcheck shell=bash
# Named launch PRESETS (PRESET=<name>). Every provider entrypoint sources this
# file. Do not run it on its own.
#
# WHY: both 2026-07-24 audits list the same P0 finding. A full launch should
# use named minimal and featured presets, not long manual environment lists.
# A smoke should fail if an effective setting differs from the preset. A wave
# launched from a ten-flag command line cannot prove it is the arm it claims
# to be. A wave launched from PRESET=<name> can prove this, because the
# harness records the name in the manifest and feeds it into the run-contract
# hash (benchmark/write_manifest.py -> run_contract.preset).
#
# PRESET UNSET MEANS ZERO BEHAVIOR CHANGE. Every function here returns
# immediately when PRESET is empty. So a run that does not name a preset stays
# byte-identical to a run made before this file existed.
#
# AN UNKNOWN PRESET NAME EXITS 2 and lists the valid names. A typo must never
# silently degrade to "no preset".
#
# --- how a preset assigns a value (read this before adding one) ----------------
# `: "${VAR:=value}"` (only-if-unset) is not enough here. Docker compose's
# `environment:` block sets every key it lists. So an unconfigured
# `PLUGIN_CONFIG: "${PLUGIN_CONFIG:-off}"` reaches the container as a real,
# present "off" value. Only-if-unset would leave that value in place and
# silently run the wrong arm. bench_preset_set therefore takes the COMPOSE
# DEFAULT as an extra argument. It treats "value equals the compose default"
# as "nobody chose this":
#
#   bench_preset_set VAR WANTED [COMPOSE_DEFAULT ...]
#     * VAR unset or empty                        -> set to WANTED
#     * VAR equals one of the COMPOSE_DEFAULTs    -> set to WANTED (this case
#                                                    is indistinguishable from
#                                                    unset inside the container)
#     * VAR set to anything else                  -> KEPT, and logged as an
#                                                    explicit operator override
#
# So an explicit `-e VAR=<something else>` still wins (this keeps the
# documented only-if-unset intent), while a compose default cannot silently
# defeat the preset. The function echoes every decision, so a shard log shows
# exactly which values the preset supplied and which the operator overrode.
#
# ORDERING: entrypoints source this file immediately after answer_env.sh and
# clock_sync.sh, and before their own `${VAR:-default}` blocks and validation
# gates. This ordering matters twice. First, a preset's value feeds those
# defaults instead of fighting them. Second, BENCH_CLOCKSYNC=1 set by a preset
# still reaches every clocksync gate, prepare, and probe path that runs later
# in the same shell (mnemosyne's TTL gate, supermemory's spawn-mode gate,
# retaindb-server's local-Postgres bring-up).

_BENCH_PRESET_LOG_PREFIX="[preset]"

bench_preset_log() { echo "$_BENCH_PRESET_LOG_PREFIX $*"; }

# bench_preset_set VAR WANTED [COMPOSE_DEFAULT ...] sets VAR per the rules above.
bench_preset_set() {
  local var="$1" wanted="$2"; shift 2
  local current="${!var-}"
  if [ -z "$current" ]; then
    export "$var=$wanted"
    bench_preset_log "$var=$wanted (was unset)"
    return 0
  fi
  local d
  for d in "$@"; do
    if [ "$current" = "$d" ]; then
      export "$var=$wanted"
      bench_preset_log "$var=$wanted (was the compose default '$d')"
      return 0
    fi
  done
  if [ "$current" = "$wanted" ]; then
    bench_preset_log "$var=$current (already the preset value)"
    return 0
  fi
  bench_preset_log "$var=$current KEPT — explicit override (preset wanted '$wanted')"
  return 0
}

# The shared part of every clock-normalized preset.
_bench_preset_clocksync_common() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set TOP_K 5 5
  bench_preset_set THINKING 1 1
}

# --- MINIMAL clock-normalized presets, one per provider ------------------------
# "Minimal" means the simplest verified adapter path, with shared top-K=5. It
# is the integration baseline (docs/BENCHMARK_MATRIX.md, "v4 minimal/featured
# run plan"). Each bundle below is that provider's v4-minimal env, read from
# its own manifest_v4min_*_generate.json, plus BENCH_CLOCKSYNC=1.

_preset_mnemosyne_minimal_clocksync() {
  _bench_preset_clocksync_common
  # This runs the plugin write path (Hermes sync_turn), with no consolidation arms.
  bench_preset_set PLUGIN_CONFIG user off
  bench_preset_set PLUGIN_AUTO_SLEEP 0 0
  bench_preset_set PLUGIN_PREFETCH_OVERLAY 0 0
  bench_preset_set EXTRACT 0 0
  bench_preset_set LIFECYCLE 0 0
  bench_preset_set CANONICAL 0 0
  bench_preset_set ORACLE 0 0
  bench_preset_set USE_DATASET_TIME 0 0
  bench_preset_set MNEMOSYNE_FACT_RECALL_ENABLED 0 0
  bench_preset_set MNEMOSYNE_ENHANCED_RECALL 0 0
  # entrypoint.mnemosyne.sh's clock-sync gate requires this value. With
  # PLUGIN_AUTO_SLEEP=0 nothing consolidates. So the shipped 168h working-memory
  # TTL would delete prior-session rows at the first inter-session gap (median
  # about 29 days, 92.5% exceed 168h). The gate refuses to run that combination
  # without an explicit TTL. So the minimal preset passes the same ~1000-year
  # value the pre-clock minimal arm ran. The featured clock-sync arm is the
  # opposite: auto-sleep on, no TTL override, shipped TTL stays live.
  bench_preset_set MNEMOSYNE_WM_TTL_HOURS 8760000
}

_preset_hindsight_minimal_clocksync() {
  _bench_preset_clocksync_common
  # This is Arm A: session-granularity retain, consolidation off, unfiltered
  # recall. RECALL_TYPES is deliberately left empty, so recall returns all fact
  # types, exactly as the v4min arm ran.
  bench_preset_set RETAIN_GRANULARITY session
  bench_preset_set HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION false false
  bench_preset_set PREFER_OBSERVATIONS 0
  bench_preset_set WAIT_CONSOLIDATION 0
  bench_preset_set PLUGIN_NATIVE_RECALL 0
  bench_preset_set STRICT_QUALITY_RUN 0
  # Storage stays the shared hindsight-pg service. Session granularity retains
  # through eval_hindsight._retain_one, which sends no document_id and no
  # update_mode, so this arm never enters the append merge that dropped the
  # caller's retain timestamp (see _preset_hindsight_featured_clocksync). Pinned
  # here so a later change to the compose default cannot silently move the
  # banked arm's storage.
  bench_preset_set HINDSIGHT_PG_MODE shared shared
  # Hindsight's temporal capability stays `native` on this path. It takes retain
  # timestamps and an explicit recall query_timestamp, and eval_hindsight anchors
  # recall "now" at logical noon. The image now ships libfaketime for the
  # featured pg0 arm, but nothing on this path sets LD_PRELOAD, so
  # BENCH_CLOCKSYNC=1 changes no OS clock here. It still declares the temporal
  # contract in the manifest and arms the strict run-contract gate.
}

_preset_mem0_minimal_clocksync() {
  _bench_preset_clocksync_common
  # CONTRACT-V4 ARM, banked and not re-run. The 6-message batch below was
  # forced by contract v4's 512-token bge-small embedder cap; the featured
  # contract v5 embedder (gte-modernbert-base, 8192-token input window) removes
  # that constraint, so v5 arms use the MemConflict authors' 8-message
  # cadence or the plugin's per-exchange cadence, never this workaround.
  # This preset uses `batch` at 3 dialogue turns (6 messages), not `session`
  # (user decision, 2026-07-26). mem0ai 2.x embeds the whole add() input as one
  # related-memory search query before extraction (_add_to_vector_store:47).
  # The shared 512-token bge-small-en-v1.5 embedder rejects anything longer.
  # We measured, on Step4_4.jsonl (5 personas, tokenized on vllm-embed), the
  # percent of windows over 512 tokens:
  #   whole session 100% (261/261, median 4087)  <- a smoke 400'd all 53 add()
  #   8 msgs / 4 turns  8.8%   median 358  11.6 adds/session
  #   6 msgs / 3 turns  1.9%   median 268  15.3 adds/session   <- chosen
  #   4 msgs / 2 turns  0.4%   median 180  22.7 adds/session
  #   2 msgs / 1 turn   0.1%   median  90  44.9 adds/session
  # 3 turns cuts overflow 4.6x versus 8 messages, for about 32% more internal
  # LLM calls. No size choice reaches 0% overflow, since single messages run to
  # 1585 tokens. So eval_mem0.py's embed shim still sends truncate_prompt_tokens
  # and counts the clipped calls (Total_Embed_Truncated_Calls), rather than
  # letting them return 400 and silently drop ingest windows.
  # DEVIATION TO LABEL: this is no longer the MemConflict authors' exact
  # 8-message mem0 cadence (Evaluation/eval_memzero.py). The cause is the
  # shared 512-token embedder every provider uses, not a mem0 tuning choice.
  # The authors drive the hosted platform, whose embedder has no such cap.
  bench_preset_set RETAIN_GRANULARITY batch session exchange
  bench_preset_set MEM0_ADD_BATCH_SIZE 6 8
  bench_preset_set MEM0_VECTOR_MODE server server
}

_preset_supermemory_minimal_clocksync() {
  _bench_preset_clocksync_common
  # SPAWN mode is mandatory under the clock contract. One shared central server
  # has one perceived clock, so it cannot sit at N shards' logical session
  # dates at once (entrypoint.supermemory.sh exits fatally on shared+clocksync).
  bench_preset_set SUPERMEMORY_SERVER_MODE spawn shared
  bench_preset_set SUPERMEMORY_RETAIN_GRANULARITY session session
  bench_preset_set SUPERMEMORY_SEARCH_MODE hybrid hybrid
  bench_preset_set SUPERMEMORY_DOCUMENTS_ARM 0 0
  # A dropped or timed-out document under clock-sync must abort the shard,
  # not bank answers against memories that never finished ingesting
  # (docs/TROUBLESHOOTING.md, Provider: Supermemory).
  bench_preset_set SUPERMEMORY_STRICT_QUALITY 1 0
  # This uses the shared embedder (user ruling 2026-07-22), the same
  # retrieval-embedding surface as every other provider. The model and dim
  # come from the compose defaults, so the serving contract decides them:
  # v4 ran bge-small-en-v1.5 384d, v5 runs gte-modernbert-base 768d.
  bench_preset_set SUPERMEMORY_EMBEDDING_PROVIDER openai openai
  # This runs N spawned servers instead of one shared server. Cap each
  # server's embedding RAM well below the shared-server default, because host
  # RAM has an about 8 GiB vLLM floor.
  bench_preset_set SUPERMEMORY_EMBEDDING_RAM_LIMIT 2gb 8gb
  # SUPERMEMORY_SEARCH_THRESHOLD is intentionally not set here. The vendor
  # default of 0.6 would hand the answerer fewer memories than the shared
  # top-K. That is a harness asymmetry, not a quality signal.
  #
  # Retries are NOT the fix for the node-cron catch-up storm. They stay for
  # ordinary transport blips only. Earlier notes here described the storm as
  # search-latency to stall through with retries; that account is superseded
  # (docs/DECISIONS.md, "Supermemory clock-sync OOM"). The storm is node-cron
  # v4's missed-execution replay inside supermemory-server 0.0.5: six
  # unconditional crons compute missed slots from the WALL clock, which
  # clocksync fakes, and replay them in a synchronous while-loop that never
  # yields, allocating about 0.46 MB per slot with GC starved. A persona's
  # ~3-year fake span is about 163,000 slots across six crons, about 73 GB,
  # which OOM-kills the host, not just stalls search
  # (docs/TROUBLESHOOTING.md, Provider: Supermemory). No vendor knob disables
  # the crons. The real fix is `SUPERMEMORY_RESPAWN_PER_SESSION`, defaulted to
  # 1 under clock-sync in docker-compose.yml (not set here, so the default
  # applies): the adapter reboots the spawned server before every session, so
  # it never lives long enough to reach a cron's second heartbeat and the
  # replay span is structurally zero. Retries below cover what is left over:
  # ordinary `ReadTimeout`s on `/v4/search` and other transport blips, the
  # same shape as the wedge-era RETRY_TIMES=40 mitigation.
  # We pass no compose default here: the var is absent from docker-compose.yml,
  # so 4 is the CODE default (_supermemory_server.py:85). Listing 4 here would
  # let the preset silently overwrite an operator who explicitly asked for 4.
  bench_preset_set SUPERMEMORY_HTTP_RETRIES 30
}

_preset_retaindb_server_minimal_clocksync() {
  _bench_preset_clocksync_common
  bench_preset_set RETAINDB_RETAIN_GRANULARITY session
  bench_preset_set DISABLE_SCHEDULER true
  bench_preset_set RETAINDB_SERVER_PROFILE fast
  bench_preset_set RETAINDB_SERVER_PLUGIN_OVERLAY 1 1
  bench_preset_set RETAINDB_EMBEDDING_MODE remote remote
  bench_preset_set RETAINDB_DISABLE_SEARCH_CACHE true
}

# Honcho minimal: the simplest verified recall path (raw stored conclusions,
# unified observation -- one collection per peer pair instead of directional
# self+cross observation), with the server-side sections that arm never
# reads (summary, peer card) turned off so it spends no LLM calls building
# them. Spawn mode is mandatory under clock-sync: one shared honcho-api/
# -deriver pair has a single perceived clock and cannot serve N shards at
# different logical session dates (entrypoint.honcho.sh's shared+clocksync
# guard exits fatally on shared+clocksync, the same reasoning as Supermemory's).
_preset_honcho_minimal_clocksync() {
  _bench_preset_clocksync_common
  bench_preset_set HONCHO_SERVER_MODE spawn shared
  bench_preset_set HONCHO_RECALL_MODE conclusions hybrid
  bench_preset_set HONCHO_OBSERVATION_MODE unified directional
  bench_preset_set HONCHO_SUMMARY_ENABLED 0 1
  bench_preset_set HONCHO_PEER_CARD_ENABLED 0 1
}

# OpenViking minimal: the ranked-list recall path (/search/search entries only,
# sliced to the shared top-5), with the plugin's own per-exchange ingest
# cadence. Spawn mode is the only mode with a compose service behind it: the
# server keeps its content store and vector index in one local workspace and a
# workspace holds a one-process lock, so there is no shared server to attach to
# and an attached one could not sit at N shards' logical session dates at once
# (entrypoint.openviking.sh exits 2 on shared+clocksync).
_preset_openviking_minimal_clocksync() {
  _bench_preset_clocksync_common
  bench_preset_set RETAIN_GRANULARITY exchange exchange
  # find, not search: the minimal run is the DIAGNOSTIC — an integration
  # proof and deterministic retrieval floor, not a comparison number (user
  # ruling 2026-08-04). The featured prefetch arm, which uses the
  # search/search endpoint, is the arm that is scored and compared.
  bench_preset_set OPENVIKING_RECALL_MODE find prefetch
  bench_preset_set OPENVIKING_SERVER_MODE spawn spawn
}

# --- FEATURED clock-normalized presets, one per provider ----------------------
# "Featured" is the selection comparison: each plugin's real cadence, read
# surface, and shipped consolidation, under the same clock contract. The locked
# per-provider decisions live in docs/DECISIONS.md ("Per-provider
# featured/minimal configuration"). Featured runs hand the answer model the
# PLUGIN-NATIVE recall count, not the shared minimal top-5 — that is why the
# mem0 and supermemory presets set TOP_K=10 (their plugins' budget) while
# mnemosyne and retaindb stay at 5 (their plugins' own top-5).

_preset_mnemosyne_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set THINKING 1 1
  bench_preset_set TOP_K 5 5
  # Plugin write path + the plugin's real prefetch() read surface + the
  # shipped auto-sleep cadence (every 10 exchanges + session end, <=3/session).
  bench_preset_set PLUGIN_CONFIG user off
  bench_preset_set PLUGIN_PREFETCH_OVERLAY 1 0
  bench_preset_set PLUGIN_AUTO_SLEEP 1 0
  # Session-end forced consolidation: "shipped consolidation, manually cadenced"
  # (user ruling 2026-08-02, mirroring HONCHO_DREAM_AFTER_SESSION). The TTL
  # stays shipped; only the cadence is ours. Mechanism: _trim_working_memory()
  # runs inside every remember() and deletes rows with consolidated_at IS NULL
  # older than the shipped 168h TTL, so under the faked clock the next
  # session's first write erases the previous session (median gap 29 days).
  # The plugin's auto-sleep gate needs working>50, which needs exactly the
  # cross-session accumulation the trim prevents. Measured without this on
  # ft27mn: 2 sleep invocations in 277 cadence ticks, 12 episodic rows, and 21
  # of 122 questions with zero recall candidates. sleep(force=True) stamps
  # consolidated_at, which makes the rows trim-exempt.
  bench_preset_set PLUGIN_SESSION_SLEEP 1 0
  bench_preset_set EXTRACT 0 0
  bench_preset_set LIFECYCLE 0 0
  bench_preset_set CANONICAL 0 0
  bench_preset_set ORACLE 0 0
  bench_preset_set USE_DATASET_TIME 0 0
  bench_preset_set MNEMOSYNE_FACT_RECALL_ENABLED 0 0
  bench_preset_set MNEMOSYNE_ENHANCED_RECALL 0 0
  # 2048 is the floor that stops sleep's model-refresh JSON truncating to zero
  # proposals (user override of the earlier 3072 candidate).
  bench_preset_set MNEMOSYNE_LLM_MAX_TOKENS 2048
  # MNEMOSYNE_WM_TTL_HOURS is deliberately NOT set. The featured clock-sync
  # arm runs the SHIPPED 168h TTL under the faked logical clock —
  # entrypoint.mnemosyne.sh exits 2 if an explicit TTL reaches this arm.
}

# Hindsight featured = Arm C (per-exchange append under a stable session doc,
# consolidation on and drained, observation-only recall — the Hermes plugin
# defaults) PLUS the community-recommended missions (user decision 2026-07-30,
# from r/hermesagent comment oor0pn4). Missions are per-bank config upstream;
# set as HINDSIGHT_API_* env they act as the GLOBAL default every persona bank
# inherits (config_resolver.py: global env -> tenant -> bank), so no adapter
# change and every persona gets identical missions.
_preset_hindsight_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set THINKING 1 1
  bench_preset_set TOP_K 5 5
  # Storage: a per-container embedded pg0 cluster under libfaketime, NOT the
  # shared hindsight-pg service. This arm retains with update_mode="append", and
  # hindsight 0.8.4 drops the caller's retain timestamp on that path, so the
  # ftclk1_p0 smoke stamped every fact with the wall clock (2026) while the
  # dataset runs in 2022. Upstream fixed the append merge in 0.8.5 (PR #2684);
  # libfaketime stays because the utcnow() fallback is unchanged and issue #3010
  # still collapses per-item dates. One postmaster shared by co-tenant shards
  # cannot sit at N shards' logical session dates at once, so a faked clock
  # needs a per-container cluster. entrypoint.hindsight.sh refuses pg0 unless
  # BENCH_CLOCKSYNC=1 and the container covers exactly one persona
  # (END_IDX - START_IDX == 1): a persona rollover would rewind the clock over
  # a store that already holds the previous persona's rows.
  bench_preset_set HINDSIGHT_PG_MODE pg0 shared
  bench_preset_set RETAIN_GRANULARITY exchange_append
  bench_preset_set HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION true false
  bench_preset_set WAIT_CONSOLIDATION 1
  bench_preset_set PREFER_OBSERVATIONS 1
  bench_preset_set RECALL_TYPES observation
  # Emit whatever `--budget mid --max_tokens 4096` returns (plugin-native
  # recall count), instead of a fixed-K slice.
  bench_preset_set PLUGIN_NATIVE_RECALL 1
  # Plugin default, kept per user decision (a permissive deployment, not the
  # stricter harness abort-on-fallback arm).
  bench_preset_set STRICT_QUALITY_RUN 0 0
  # Consolidation renders ~3x JSON; these caps fit the 32k window. RECALL_BUDGET
  # low is the pinned 0.8.4 package default, recorded explicitly for the
  # manifest (2026-07-24 audit).
  bench_preset_set HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS 4096
  bench_preset_set HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS 4096
  bench_preset_set HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET low
  # Serving-stability knobs: the compose defaults are already the locked
  # values (temp 0.7, retain cap 8192, retries 3); pinning them here writes
  # them into the preset log and run-contract hash.
  bench_preset_set HINDSIGHT_API_LLM_TEMPERATURE_RETAIN 0.7 0.7
  bench_preset_set HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS 8192 8192
  bench_preset_set HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES 3 3
  # "concise" is the shipped 0.8.4 default; pinned so a vendor default change
  # cannot silently move the arm.
  bench_preset_set HINDSIGHT_API_RETAIN_EXTRACTION_MODE concise
  bench_preset_set HINDSIGHT_API_RETAIN_MISSION 'Retain durable, reusable outcomes likely to matter again across sessions. Prioritize user preferences, decision boundaries, operating constraints, verified conclusions, root-cause findings, stable environment/config facts, recurring workflow rules, notable people/companies/projects, recurring counterparties, important human communications, and actionable outputs from tools, cron, or email such as alerts, major findings, entity creation/backfills, and follow-up-required items. Do not retain conversational filler, duplicate paraphrases, routine status chatter, unverified speculation without evidence labels, intermediate debug noise unless it establishes a durable lesson, or stale experimental residue. When in doubt, keep one compact high-signal summary rather than transcript detail.'
  bench_preset_set HINDSIGHT_API_OBSERVATIONS_MISSION "Track the user's stable preferences, recurring routines, important people and relationships, and how their priorities shift over time."
  # The reflect mission is config-complete per the community recipe, but the
  # benchmark never calls reflect() (recall-path only), so it is inert here.
  bench_preset_set HINDSIGHT_API_REFLECT_MISSION 'Memory bank for Hermes assistant. Prioritize durable context likely to matter in future conversations: user preferences, boundaries, stable environment facts, recurring workflow rules, notable people/companies/projects, important human communications, active opportunities/risks, and verified conclusions. Prefer compact factual memory over transcript-like chatter.'
}

_preset_mem0_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set THINKING 1 1
  # Plugin-native K: the Hermes mem0 plugin searches top_k=10 in both prefetch
  # and the tool. Featured hands the answer model what the plugin injects.
  bench_preset_set TOP_K 10 5
  # The plugin's actual cadence: sync_turn fires one add(infer=True) per
  # completed exchange. Cheap under 2.0.14's single-pass add (~2-3x the
  # batch-6 call count; re-base on v4minc's ~33 s/session).
  bench_preset_set RETAIN_GRANULARITY exchange batch
  bench_preset_set MEM0_VECTOR_MODE server server
  # 2.x vendor threshold default 0.1 on the blended hybrid score would
  # under-fill the top-K (harness asymmetry) — send explicit 0.0.
  bench_preset_set MEM0_SEARCH_THRESHOLD 0.0 0.0
}

_preset_supermemory_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set THINKING 1 1
  # Plugin-native K: the plugin merges up to max_recall_results=10 memories
  # plus the profile block. TOP_K=10 makes the adapter's slice a no-op.
  bench_preset_set TOP_K 10 5
  # The plugin's real endpoints (user: "please match plugin"): full-session
  # ingest via /v4/conversations, recall via /v4/profile.
  bench_preset_set SUPERMEMORY_INGEST_ENDPOINT conversations
  bench_preset_set SUPERMEMORY_RECALL_ENDPOINT profile
  bench_preset_set SUPERMEMORY_RETAIN_GRANULARITY session session
  bench_preset_set SUPERMEMORY_SEARCH_MODE hybrid hybrid
  bench_preset_set SUPERMEMORY_DOCUMENTS_ARM 0 0
  bench_preset_set SUPERMEMORY_STRICT_QUALITY 1 0
  # Spawn is mandatory under clock-sync (one shared server has one perceived
  # clock); per-session respawn defeats the node-cron replay OOM (compose
  # defaults SUPERMEMORY_RESPAWN_PER_SESSION=1 under clocksync).
  bench_preset_set SUPERMEMORY_SERVER_MODE spawn shared spawn
  bench_preset_set SUPERMEMORY_EMBEDDING_PROVIDER openai openai
  bench_preset_set SUPERMEMORY_EMBEDDING_RAM_LIMIT 2gb 8gb
  bench_preset_set SUPERMEMORY_HTTP_RETRIES 30
}

# --- FEATURED clock-normalized preset (RetainDB server) -----------------------
# This preset locks six vars for the featured RetainDB-server arm: per-exchange
# ingest (matches the plugin's sync_turn cadence), the 60s scheduler on so
# runSessionLifecycle() promotes cold SESSION memories and writes a session
# summary, a per-session lifecycle wait so recall never races that promotion,
# the quality extraction profile, and the legacy user-specific promotion mode.
# That promotion mode is the mode that actually creates SESSION-scoped rows for
# the scheduler to promote (see the memory note "RetainDB SESSION scope and
# scheduler under clocksync"). Without it the lifecycle stays inert.
_preset_retaindb_server_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set RETAINDB_RETAIN_GRANULARITY exchange
  bench_preset_set DISABLE_SCHEDULER false
  bench_preset_set RETAINDB_SERVER_WAIT_LIFECYCLE 1
  bench_preset_set RETAINDB_SERVER_PROFILE quality
  bench_preset_set RETAINDB_SERVER_PROMOTION_MODE user_specific_legacy
}

# Honcho featured = the Hermes plugin's real read surface: hybrid recall
# (session summary + user representation/card + dialectic), directional
# observation (self AND cross observation, the plugin's observationMode
# default), summaries and peer cards both on. Spawn mode is mandatory under
# clock-sync for the same reason as the minimal preset above.
_preset_honcho_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set HONCHO_SERVER_MODE spawn shared
  bench_preset_set HONCHO_RECALL_MODE hybrid hybrid
  bench_preset_set HONCHO_OBSERVATION_MODE directional directional
  bench_preset_set HONCHO_SUMMARY_ENABLED 1 1
  bench_preset_set HONCHO_PEER_CARD_ENABLED 1 1
  # Manually schedules a dream after each session (user ruling 2026-07-31):
  # dataset sessions are days apart, so the vendor's idle-based scheduler (60
  # min idle, 8h spacing) would never fire inside a benchmark run otherwise.
  bench_preset_set HONCHO_DREAM_AFTER_SESSION 1 0
  # 32768 under contract v5 (was 8192 under v4's 32768 gen window, where the
  # answer prompt overflowed at session 5 without it). At the v5 131072
  # window this bound effectively never binds. CAVEAT TO PUBLISH with any
  # number: the bound is the plugin's own _truncate_to_budget code at a
  # value the plugin ships unset.
  bench_preset_set HONCHO_CONTEXT_TOKENS 32768 8192
}

# OpenViking featured = the Hermes plugin's real read surface: `prefetch`
# recall (the session-start profile + preferences + entities block, then the
# /search/search entries, selected and formatted by the plugin's own
# _select_recall_candidates and entry format), and the plugin's per-exchange
# ingest cadence. TOP_K stays 5: prefetch sets plugin_native_recall=True, so
# the adapter emits the plugin's own recall_limit=6 entries plus the
# session-start block with no top-K slice, and the shared value only records
# the harness contract. Spawn mode for the same reason as the minimal preset.
_preset_openviking_featured_clocksync() {
  bench_preset_set BENCH_CLOCKSYNC 1 0
  bench_preset_set THINKING 1 1
  bench_preset_set TOP_K 5 5
  bench_preset_set OPENVIKING_RECALL_MODE prefetch prefetch
  bench_preset_set RETAIN_GRANULARITY exchange exchange
  bench_preset_set OPENVIKING_SERVER_MODE spawn spawn
}

# This maps every valid PRESET name to its function. To add a preset, add one
# line here and one function above. The unknown-name error prints this list.
_BENCH_PRESET_NAMES="
mnemosyne_minimal_clocksync
hindsight_minimal_clocksync
mem0_minimal_clocksync
supermemory_minimal_clocksync
retaindb_server_minimal_clocksync
honcho_minimal_clocksync
openviking_minimal_clocksync
mnemosyne_featured_clocksync
hindsight_featured_clocksync
mem0_featured_clocksync
supermemory_featured_clocksync
retaindb_server_featured_clocksync
honcho_featured_clocksync
openviking_featured_clocksync
"

# bench_apply_preset [expected_provider]
# The optional argument is the provider whose entrypoint is calling this
# function. The function refuses a preset named for another provider. Without
# this check, for example PRESET=mem0_minimal_clocksync on the supermemory
# service would silently export nothing useful, and would label the manifest
# with a preset the run did not actually run.
bench_apply_preset() {
  local provider="${1:-}"
  local preset="${PRESET:-}"
  [ -n "$preset" ] || return 0
  local fn="_preset_${preset}"
  if ! declare -F "$fn" >/dev/null 2>&1; then
    echo "$_BENCH_PRESET_LOG_PREFIX FATAL: unknown PRESET='$preset'. Valid names:" >&2
    printf '  %s\n' $_BENCH_PRESET_NAMES >&2
    exit 2
  fi
  if [ -n "$provider" ] && [ "${preset#${provider}_}" = "$preset" ]; then
    echo "$_BENCH_PRESET_LOG_PREFIX FATAL: PRESET='$preset' is not a '$provider' preset" \
         "(a preset must be named <provider>_<arm>). Valid names:" >&2
    printf '  %s\n' $_BENCH_PRESET_NAMES >&2
    exit 2
  fi
  bench_preset_log "applying PRESET=$preset"
  "$fn"
  export PRESET="$preset"
  bench_preset_log "PRESET=$preset applied (recorded in the manifest + run-contract hash)"
}
