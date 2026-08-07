# Benchmark matrix

This document states, for each provider, the configuration its final featured
benchmark run used, and how the featured runs compare. A configuration is one
named set of provider settings. Only the featured configuration of each provider
is described here. The configurations that were tested but not featured are not
listed.

Dataset: `external/MemConflict/Data/Step4_4.jsonl`, 30 personas, 3,750 questions.

The published report at <https://engturtle.github.io/hermes-memconflict/report/>
presents these results as charts and per-provider notes. The repository README at
<https://github.com/EngTurtle/hermes-memconflict> covers the harness that
produced them.

Headline metric: macro answer accuracy, the unweighted mean of answer accuracy
over the three conflict types (dynamic, static, conditional). Answer accuracy is
the fraction of questions whose final answer matches the gold answer. Supporting
evidence hit at K states whether the top-K retrieved memories contain the gold
memory item. Read values from the committed summary JSONs
(`overall.macro_memconflict_protocol`), not from prose.

Every configuration and result in this document belongs to the featured contract,
**contract v5**.

The run tags in this document (`v5ftc`, `v5ftc086`, `v5ftcall`, `v5ftovk`) are
on-disk run identifiers. Each tag names the score and result files of one run
under `<provider>/Scores/` and `<provider>/Results/`.

## Providers

| Provider | Product under test | Featured run |
|---|---|---|
| **Mnemosyne** | `external/mnemosyne` v3.14.0 submodule, via SDK and via its Hermes plugin | `v5ftc` |
| **Hindsight** | `hindsight-all==0.8.6` plus `pg0-embedded==0.15.0` (PostgreSQL 18.1.0, pgvector 0.8.5, pg_trgm 1.6), Docker. The featured configuration runs on a per-container embedded pg0 cluster | `v5ftc086`; second configuration `v5ftcall` |
| **mem0** | `mem0ai[nlp]==2.0.14`, self-hosted `Memory` SDK plus qdrant v1.18.3 | `v5ftc` |
| **Supermemory** | self-hosted server binary plus internal extraction LLM | `v5ftc` |
| **Honcho** | self-hosted `plastic-labs/honcho` v3.0.9 submodule (FastAPI, a separate deriver worker, Postgres/pgvector), SDK `honcho-ai==2.2.0` (the exact Hermes plugin pin) | `v5ftc` |
| **OpenViking** | `openviking==0.4.12`, one pip distribution that ships the `openviking-server` console script (FastAPI/uvicorn) plus self-contained storage (AGFS content store and local vector index in one workspace dir) | `v5ftovk` |
| **RetainDB server** | `@retaindb/server` from the `external/RetainDB` submodule, Postgres/pgvector | `v5ftc` |
| ~~**RetainDB local**~~ | npm `@retaindb/local@0.2.1` | **Ruled out 2026-07-22.** O(n²) search, about 245 s per query at n=2,897, no vendor knob. The adapter is correct and kept for the record |

The two RetainDB entries are different products. The ruling applies only to the
npm local edition.

## Featured configuration per provider

Each block states what the provider was configured to do in its featured run: the
recall and ingestion behavior, the environment variables and flags that were set,
and the vendor version pin. Every run sets `BENCH_CLOCKSYNC=1`, so the generate
stage runs under libfaketime and each session ingests at its logical dataset date.

### Mnemosyne, `mnemosyne/eval_mnemosyne.py`

The featured run writes through the Hermes plugin path and runs the plugin's own
consolidation cadence. It calls no separate fact-extraction pipeline
(`EXTRACT=0`), so it makes no LLM call to ingest.

- **Write path `--plugin_config user`.** One `remember()` per role per exchange,
  importance 0.5 for the user turn and 0.15 for the assistant turn, entity
  extraction only, `temporal_weight=0.2` at halflife 48. This mirrors the
  plugin's `sync_turn()`.
- **Auto-sleep `--plugin_auto_sleep`.** The plugin's real sleep cadence: every 10
  exchanges and once per session boundary, drained before questions. Sleep is
  Mnemosyne's consolidation pass that rewrites working memory. The per-session
  reflection budget is capped at 3 calls by the plugin's own
  `MNEMOSYNE_REFLECT_MAX_CALLS_PER_SESSION`.
- **`MNEMOSYNE_FACT_RECALL_ENABLED=0`**, the plugin default. Fact rows retrieve
  worse: a probe measured supporting evidence hit at 3 of 0.031 with the top-5
  dominated by lossy fact strings.
- **`MNEMOSYNE_LLM_MAX_TOKENS=3072`.** The same knob caps per-message extraction
  and the whole-session model-refresh JSON. At 512 the refresh JSON truncates to
  zero proposals.
- **The shipped 168h working-memory TTL is enforced.** The featured run sets no
  explicit `MNEMOSYNE_WM_TTL_HOURS`.
- Retrieval embedding uses the shared `vllm-embed` server.
- Temporal capability: `controlled_process_clock`.

### Hindsight, `hindsight/eval_hindsight.py`

The featured run is plugin-faithful. An observation is a consolidated fact
Hindsight derives from raw exchanges. Consolidation is the daemon that merges
facts into observations.

- **Recall filtered to observations only (`RECALL_TYPES=observation`).** This is
  the plugin's own default. Recall prefers a consolidated observation
  (`--prefer_observations`).
- **Per-exchange append ingest (`--retain_granularity exchange_append`).** Each
  exchange appends under a stable `document_id` with `update_mode="append"`, the
  cadence the Hermes integration uses.
- **Auto-consolidation on** (`HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=true`), and
  the run blocks until each session's consolidation drains
  (`--wait_consolidation`, `--consolidation_wait_timeout_s 450`).
- **`HINDSIGHT_API_LLM_TEMPERATURE_RETAIN=0.7`.** The shipped value is 0.1. 0.7 is
  the Qwen3.5 card non-thinking value and the temperature the consolidation path
  already inherits. It cut runaway retains without moving the p50 latency.
- **Retain cap `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=4096`**, retries
  `HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES=7`, strict schema
  `HINDSIGHT_API_LLM_STRICT_SCHEMA=1`, consolidation source cap
  `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096`, LLM timeout
  `HINDSIGHT_API_LLM_TIMEOUT=900`.
- **Recall width `--top_k 5`**, `--max_tokens 4096`, `--budget mid`.
- **Storage `HINDSIGHT_PG_MODE=pg0`.** A per-container embedded pg0 cluster inside
  the faked clock domain, one persona per container, because the append merge
  takes dates from the DB clock and no API parameter reaches it.
- Version pin: `hindsight-all==0.8.6` plus `pg0-embedded==0.15.0`.
- Temporal capability: `controlled_process_clock+postgres`.

The second configuration `v5ftcall` is identical except that recall is unfiltered
(`RECALL_TYPES=all`). It is a diagnostic, not the featured run.

### mem0, `mem0/eval_mem0.py`

The featured run ingests at the plugin's own per-turn cadence and runs the
extraction pipeline that is the feature under test.

- **Per-turn ingest (`RETAIN_GRANULARITY=exchange`).** One `add()` per turn, the
  cadence in `plugins/memory/mem0/__init__.py:488-498`.
- **`infer=True`.** The extraction pipeline stays on. Off would store raw turns
  and discard the feature under test.
- **Retrieval `search(query, filters={"user_id": …}, top_k=20, threshold=0.0)`,
  the top 10 reaching the answer model (`TOP_K=10`).** `threshold=0.0` is explicit
  because the vendor default 0.1 cuts on a blended hybrid score below the fairness
  line.
- **The embedder is the shared `vllm-embed`** (v5: `gte-modernbert-base`, 768
  dims), against a self-hosted qdrant v1.18.3, with personas isolated by
  `user_id`.
- Version pin: `mem0ai[nlp]==2.0.14`, the version the Hermes plugin accepts
  (`mem0ai>=2.0.10,<3`).
- Temporal capability: `controlled_process_clock`.

**mem0 2.0.14 is ADD-only.** It uses a single-pass algorithm with hybrid
semantic, BM25, and entity retrieval. The old two-pass update-decision pipeline is
gone: results are hard-coded ADD (confirmed at source 2026-07-28,
`memory/main.py:1165-1168` stamps the event as a string literal, and the two-phase
machinery is dead code). The `Total_Event_UPDATE/DELETE/NONE` counters are 0 by
construction. This is a provider property to report, not a regression to work
around.

### Supermemory, `supermemory/eval_supermemory.py`

The featured run uses the Hermes plugin's default hybrid search. The adapter
spawns a native server binary and drives it over REST.

- **`--search_mode hybrid`** (`/v4/search`, memories first with a doc-chunk
  fallback), the plugin default.
- **Search threshold sent as explicit `0.0` (`SUPERMEMORY_SEARCH_THRESHOLD=0.0`).**
  Omitting it engages the vendor's 0.6 cutoff, which would hand Supermemory fewer
  memories than the shared top-K. The adapter requests the plugin's
  `max_recall_results` of 10 and the top 10 reach the answer model. A profile block
  of about 1,340 characters is also appended to the answer context.
- **Per-session ingest.** The adapter drains the async ingest queue before
  answering (`POST /v3/documents` returns `queued`, a memory is searchable at
  `done`), so recall does not race the queue.
- **Two LLM roles kept apart.** The answer and judge model is harness-locked
  (`OPENAI_*`); the internal extraction model is `SUPERMEMORY_LLM_*`, mapped onto
  the spawned server's own subprocess env.
- **Spawn mode with per-session respawn.** `BENCH_CLOCKSYNC=1` forces spawn mode
  and `SUPERMEMORY_RESPAWN_PER_SESSION=1`, so the server never observes a forward
  clock jump.
- Version pin: `SUPERMEMORY_SERVER_VERSION=0.0.5`. 0.0.6 and 0.0.7-rc.2 ship a
  broken linux-x64 ingest engine.
- Temporal capability: `controlled_process_clock`.

### Honcho, `honcho/eval_honcho.py`

Honcho returns a peer model rather than a ranked memory list. The Hermes plugin
auto-injects a markdown block assembled from named sections, and that block is the
product under test. Vendor terms: the deriver is the extraction worker; the
dialectic is Honcho's internal question-answering call; a dream is a consolidation
pass that rewrites observations into conclusions; an observation is one recorded
statement about a peer.

- **Hybrid recall (`HONCHO_RECALL_MODE=hybrid`), the plugin default.** The
  plugin's own injection order: session summary, user representation plus peer
  card, AI self-representation plus card, then a dialectic answer clipped to 600
  characters. `plugin_native_recall=True`, so no top-K slice and every section
  reaches the answer model. The full render is about 25,000 tokens per question.
- **Directional observation mode (`HONCHO_OBSERVATION_MODE=directional`), summary
  and peer card on** (`HONCHO_SUMMARY_ENABLED=1`, `HONCHO_PEER_CARD_ENABLED=1`).
- **Per-exchange ingest.** One `add_messages` call per exchange, both peers, the
  plugin's own 25000-char chunking with a `"[continued] "` prefix, and no
  `created_at` sent (the plugin never sends one, so `BENCH_CLOCKSYNC` is the
  temporal path).
- **Manual dream after each session (`HONCHO_DREAM_AFTER_SESSION=1`, user ruling
  2026-07-31).** After each session drains, the adapter calls `schedule_dream`
  (POST `/v3/workspaces/{id}/schedule_dream`, `dream_type=omni`) per
  observer-to-observed pair, then drains again. A benchmark run never idles, so the
  shipped 60-minute idle scheduler never fires. Label: "shipped consolidation,
  manually cadenced."
- **Per-session queue drain plus `DERIVER_FLUSH_ENABLED=true`.** The adapter polls
  `queue_status()` until `pending_work_units==0` and `in_progress_work_units==0`.
  The flush bypasses `DERIVER_REPRESENTATION_BATCH_MAX_TOKENS`, so a tail batch
  does not stall the drain.
- **Injection budget `HONCHO_CONTEXT_TOKENS=8192`.** The plugin ships this unset
  (uncapped), and uncapped reached 254k tokens at persona 0 session 5, past the
  window. 8192 is half the remaining prompt budget. Publish this caveat with any
  Honcho number.
- **Deriver output cap `HONCHO_DERIVER_MAX_OUTPUT_TOKENS=2048`** (deriver only),
  deriver presence penalty `HONCHO_DERIVER_PRESENCE_PENALTY=1.5` (the Qwen card
  value, sent per request), global `HONCHO_LLM_MAX_OUTPUT_TOKENS=8192` for the
  dialectic, summary, and dream roles, dialectic input bound
  `HONCHO_DIALECTIC_MAX_INPUT_TOKENS=20000`, request timeout `HONCHO_TIMEOUT=300`.
- **Per-shard spawn (`HONCHO_SERVER_MODE=spawn`).** A shared central pair has one
  perceived clock and cannot serve N shards at different logical session dates.
- Version pins: `plastic-labs/honcho` v3.0.9, SDK `honcho-ai==2.2.0`.
- Temporal capability: `controlled_process_clock`.

The featured run loses about half of the dialectic section to context overflow on
the v5 window, a product property reported under the measurement ruling, not a
configuration to route around.

### OpenViking, `openviking/eval_openviking.py`

OpenViking extracts a memory tree per user, not a flat memory list. Raw session
messages never enter the memory search space. Search results carry no timestamp
field, so every returned item reports `created_at` as `Unknown Time`.

- **Prefetch recall (`OPENVIKING_RECALL_MODE=prefetch`), the plugin's full read
  surface.** A session-start block (the `profile.md` body plus a `preferences/`
  and `entities/` listing under the plugin's token budget) as item 0, then the
  `POST /api/v1/search/search` entries selected by the plugin's own
  `_select_recall_candidates` and rendered in its `- [category] <uri> …` format.
  `plugin_native_recall=True`, so no top-K slice.
- **Per-exchange ingest (`RETAIN_GRANULARITY=exchange`).** One
  `POST /api/v1/sessions/{sid}/messages/batch` per exchange, one commit per
  session, no `created_at` sent.
- **Recall width `OPENVIKING_RECALL_LIMIT=6`**, score floor
  `OPENVIKING_RECALL_SCORE_THRESHOLD=0.15`, injection budgets
  `OPENVIKING_RECALL_MAX_INJECTED_CHARS=4000` and
  `OPENVIKING_PROFILE_TOKEN_BUDGET=6000`, full-body reads
  `OPENVIKING_RECALL_FULL_READ_LIMIT=2` with
  `OPENVIKING_RECALL_PREFER_ABSTRACT=0`. All are plugin defaults.
- **Recall timeouts 60 s and 30 s** (`OPENVIKING_RECALL_TIMEOUT_SECONDS=60`,
  `OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS=30`), against the plugin's 4.0 and
  3.0, because the adapter calls recall inline rather than on a background thread.
  60 is the plugin's own clamp maximum.
- **Internal chat model `qwen3.5-4b` on `vllm-gen`**, `max_tokens` 4096,
  temperature 0.0, `max_concurrent` 8. Embedder `gte-modernbert-base` on
  `vllm-embed`, 768 dims.
- **Per-shard spawn (`OPENVIKING_SERVER_MODE=spawn`).** One server per container,
  its own workspace, no shared backing service.
- Version pin: `openviking==0.4.12`.
- Temporal capability: `controlled_process_clock`.

### RetainDB server, `retaindb_server/eval_retaindb_server.py`

The featured run runs the lifecycle scheduler on and routes mid-confidence facts
to SESSION scope so the scheduler has rows to promote and summarize. Promotion
mode routes facts to memory scopes (USER, PROJECT, SESSION).

- **Plugin recall overlay on (`--plugin_overlay`).** 320-char compaction plus
  dedup, top-5, reproducing the plugin's recall overlay. Recall sends no
  `question_date` (the plugin sends none) and `include_pending: true`.
- **`write_mode:"sync"` on ingest**, so a run never recalls against a
  half-ingested session.
- **Scheduler on**, the featured behavior, built and verified under the faked
  clock. Its 60s `runSessionLifecycle()` job mutates the memory set.
- **`RETAINDB_SERVER_PROMOTION_MODE=user_specific_legacy`.** This is a
  vendor-exposed per-request field the pinned Hermes plugin does not send. Report
  it as a best-effort deployable configuration with plugin-shaped ingest plus a
  benchmark-side `promotion_mode` field, not an unmodified plugin-faithful run.
- **Embedding mode `remote` (`RETAINDB_EMBEDDING_MODE=remote`).** The contract
  embedder at 768 dims, zero-padded to the schema's `vector(1024)`,
  rank-preserving. Search profile `--profile fast` (the server default). Search
  cache off (`RETAINDB_DISABLE_SEARCH_CACHE=true`). Extractor model is the contract
  model (`EXTRACTOR_MODEL`).
- **`BENCH_CLOCKSYNC=1`** runs the node server and an internal per-shard throwaway
  Postgres 18 under libfaketime. `Dockerfile.retaindb-server` bakes `postgresql-18`
  plus `postgresql-18-pgvector`.
- Version pin: `@retaindb/server` from the `external/RetainDB` submodule.
- Temporal capability: `controlled_process_clock+postgres`.

## How the featured run was served

The featured comparison runs on contract v5. The served alias is always
`qwen3.5-4b`, so a manifest's `OPENAI_MODEL` never identifies the checkpoint.
Recover it from the compose file at the run's repo SHA, or from a captured
`serving_envelope_*.json` sidecar written on every generate.

### Contract v5 serving envelope (declared 2026-08-01)

- The image and checkpoint are the pinned nightly digest
  `sha256:9894a751bdd2…c801533e`, `AxionML/Qwen3.5-4B-NVFP4` served `qwen3.5-4b`.
  Engine `v0.23.1rc1.dev1373+g387189c42`, flashinfer 0.6.14.
- **`--max-model-len 131072`.** Sized by the largest measured prompt: Honcho's
  dream accumulates tool results to about 72,708 tokens, plus the 8,192-token
  output reservation, about 81,000 total. The measured KV pool of 434,238 tokens
  equals 3.31 full-window requests, identical at `--gpu-memory-utilization` 0.85
  and 0.74.
- The key generation flags are `--gpu-memory-utilization 0.85
  --max-num-batched-tokens 4096 --kv-cache-dtype fp8 --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --default-chat-template-kwargs {"enable_thinking": false}
  --override-generation-config
  {"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}`.
  Tool-calling is required by Supermemory's memory agent and Honcho's dialectic,
  and is inert otherwise. The serving-side `--override-generation-config` governs
  only providers' paramless internal calls (extraction, consolidation), never the
  answer or judge path.
- **Embedder `Alibaba-NLP/gte-modernbert-base`**, served `gte-modernbert-base`,
  768 dims, `--max-model-len 8192`, `--gpu-memory-utilization 0.07`,
  `--pooler-config '{"use_activation": true}'`. The pooler flag is required
  because the checkpoint ships no Normalize module and vLLM otherwise returns
  unnormalized vectors (measured L2 about 37 to 38). The 8192-token native window
  ends every 512-token truncation shim. RetainDB zero-pads 768 to 1024,
  rank-preserving.
- **GPU split.** `vllm-embed` at 0.07 (a BERT-family encoder that reserves no KV
  cache), `vllm-gen` at 0.85. The score-stage 0.94 profile is unchanged, because
  `vllm-embed` is stopped during scoring.
- **Re-provisioning.** Vector stores are dimension-bound, so v5 uses fresh mem0
  qdrant collections, a fresh `supermemory_data` volume, and Honcho's column
  retype at 768 via the vendor's `configure_embeddings.py`.

### Answer decoding and the judge

- **Answer decoding (all featured runs, `answer_env.sh`), thinking on:**
  `temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5,
  max_tokens 16384`.
- **Every featured result is scored by one judge, `gemma-4-12b`
  (`unsloth/gemma-4-12b-it-NVFP4`), under the penalty rubric (`_gj12pen`), at
  temperature 1.0, top_p 0.95, top_k 64** with `MEMCONFLICT_JSON_MODE=1` and
  `MEMCONFLICT_EVAL_DIR=benchmark/penalty_judge_eval`. The penalty rubric scores a
  correct answer 1.0, a partial 0.5, a refusal 0.0, and a wrong or contradictory
  answer −1.

## Results

Read values from the committed summary JSONs
(`overall.macro_memconflict_protocol`), not from prose.

### THE v5 FEATURED COMPARISON, gemma-4-12b, penalty rubric (`_gj12pen`)

Because −1 is outside the standard MemConflict range, these numbers compare only
to each other. A macro difference under about 0.02 to 0.03 is judge sampling noise
(the repo's own same-machine agreement is 95.9%, plus or minus 0.025 macro). Do
not rank providers inside that band.

`SEH@3 macro` in the table is the macro supporting evidence hit at 3: the
unweighted 3-type mean of whether the top-3 retrieved memories contain the gold
memory item.

| provider | tag | macro AA | dynamic | static | conditional | SEH@3 macro | micro AA | banked |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Honcho | `v5ftc` † | **0.477** | 0.643 | 0.181 | 0.606 | 0.808 | 0.595 | `f9fd247` |
| mem0 | `v5ftc` | **0.392** | 0.250 | 0.090 | 0.836 | 0.672 | 0.304 | `e9eb9cb` |
| Supermemory | `v5ftc` | **0.288** | 0.144 | 0.026 | 0.694 | 0.537 | 0.197 | `d0c1191` |
| Hindsight (second configuration) | `v5ftcall` | **0.281** | 0.455 | 0.114 | 0.275 | 0.452 | 0.401 | `e2fd94d` |
| RetainDB server | `v5ftc` | **0.270** | 0.279 | 0.035 | 0.495 | 0.544 | 0.281 | `2b500f9` |
| Hindsight (featured) | `v5ftc086` | **0.218** | 0.346 | 0.118 | 0.189 | 0.382 | 0.306 | `8d3782c` |
| OpenViking | `v5ftovk` | 0.132 | 0.143 | 0.067 | 0.187 | 0.321 | 0.141 | `3e82082` |
| Mnemosyne | `v5ftc` | 0.116 | 0.344 | **−0.204** | 0.207 | 0.341 | 0.275 | `d21a704` |

**The two Hindsight rows are two configurations of the same provider.**
`v5ftc086` is the featured configuration, with recall filtered to observations
only. `v5ftcall` is a diagnostic second configuration, with recall unfiltered
(`RECALL_TYPES=all`). Only `v5ftc086` is plugin-faithful.

† **Honcho alone was judged at a 49,152-token input window, every other run used
32,768.** Honcho's injected block is about 25k tokens per question, so all 3,750
of its judge prompts exceed the default 16,384-token budget and it was scored on a
judge re-served at `--max-model-len 49152`. This is the only judge-config
difference from the other runs: the same gemma-4-12b model, temperature 1.0,
top_p 0.95, top_k 64, penalty rubric (`docs/DECISIONS.md`, "The Honcho wave alone
is judged at a 49,152-token window").

Diagnostics recorded alongside. Update order recognition (dynamic): Honcho 0.768,
Hindsight `v5ftcall` 0.670, Hindsight `v5ftc086` 0.589, mem0 0.507, RetainDB
0.477, Mnemosyne 0.440, Supermemory 0.378, OpenViking 0.246. Contradiction
recognition (static): Honcho 0.672, mem0 0.536, Supermemory 0.467, RetainDB 0.328,
Hindsight `v5ftcall` 0.306, Mnemosyne 0.242, Hindsight `v5ftc086` 0.186,
OpenViking 0.078.

Findings, all eight runs scored:

- **Supermemory, Hindsight, and RetainDB cluster together.** Supermemory 0.288,
  Hindsight `v5ftcall` 0.281, and RetainDB 0.270 span 0.018 macro, inside the
  noise band, and they get there by different routes. Of the three, Hindsight is
  highest on dynamic (0.455 against RetainDB's 0.279 and Supermemory's 0.144) and
  on static (0.114), and Supermemory is highest on conditional (0.694 against
  RetainDB's 0.495 and Hindsight's 0.275).
- **Static conflicts defeat every provider.** The best static answer accuracy is
  Honcho's 0.181, then Hindsight `v5ftc086`'s 0.118, Hindsight `v5ftcall`'s 0.114,
  and mem0's 0.090. Mnemosyne is negative at −0.204, so it asserts wrong values
  more often than right ones there. The evidence utilization gap on static, which
  is supporting evidence hit at 3 minus answer accuracy for that conflict type,
  runs +0.211 to +0.633 (OpenViking 0.211, Hindsight `v5ftc086` 0.296, Hindsight
  `v5ftcall` 0.372, RetainDB 0.407, mem0 0.501, Mnemosyne 0.535, Supermemory
  0.554, Honcho 0.633): the evidence is retrieved and not used. Honcho's gap is the
  largest, because it retrieves the most static evidence (supporting evidence hit
  at 3 is 0.814) and still answers 0.181. OpenViking's gap is smallest only because
  it retrieves the least (its static supporting evidence hit at 3 is 0.278), so
  there is less retrieved evidence to leave unused.
- **The penalty rubric changes the ranking.** Under a 0.0 floor, Mnemosyne's
  static would read about 0.25 and look competitive. The −1 penalty for a wrong
  answer removes that appearance. Separating "wrong" from "declined to answer" is
  the whole point of the penalty rubric.
- **Supermemory's retrieval numbers measure a different thing.** 904 of 3,750
  questions (24.1%) returned zero ranked memories, with the answer model working
  from the separately injected profile block. Its supporting evidence hit is not
  like-for-like with a provider that always returns a ranked list.

Every banked run was gated on `judge_methods` containing no `rule_based` (a
retry-exhausted judge call silently leaves the penalty rubric and can never score
−1). All eight are clean. RetainDB is the only one with a perfect record, 3,750
`llm_judge` and zero missing answers. mem0 has 3,748 `llm_judge` and 2 missing
answers, Hindsight `v5ftcall` 3,741 and 9, Honcho 3,740 and 10, and Hindsight
`v5ftc086` has the most, 3,739 `llm_judge` and 11 missing answers.

### Reading these numbers

**The judge's evidence surface is capped at top-5, but four featured
configurations give the answer model more.** The LLM judge renders only the first
5 retrieved memories (`MAX_WHITE_BOX_TOP_K = 5`, `eval_scoring.py:39`). Hindsight
plugin-native recall injects a median of 65 observation items (range 15 to 144).
Honcho hybrid recall injects 4 to 6 named markdown sections whose full render is
about 25,000 tokens. mem0 and Supermemory run `TOP_K=10`, so the judge reads 5 of
the 10 items the answer model read, and Supermemory also appends a profile block
of about 1,340 characters that the judge never receives. On these four
configurations, read supporting evidence hit at K, support rank, log-rank at 3,
and the evidence utilization gap as diagnostics of the top-5 slice, not as
measures of the full injected surface. Answer accuracy is unaffected, because the
answer model read the full surface before the judge scored the answer. The
Mnemosyne and RetainDB server featured configurations are unaffected: they ran
`TOP_K=5` and stored at most 5 memories per question, so the judge saw the same
evidence the answer model did. Ruling and the measured surface table: DECISIONS,
"The judge's evidence render stays at top-5".

### Measured generation cost, contract v5 featured wave

`vllm-gen` Prometheus counter deltas for complete 30-persona featured runs, 3,750
questions each. The counter is the only capture point that sees every LLM call a
provider makes, so each total includes that provider's own extraction,
consolidation, and dialectic work, not only the shared answer model.

| provider | tag | prompt tokens | generated tokens | total | per session | per turn | embedding tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Mnemosyne | `v5ftc` | 8,262,116 | 9,825,512 | 18,087,628 | 11,455 | 255 | 5,420,753 |
| OpenViking | `v5ftovk` | 99,610,637 | 19,345,376 | 118,956,013 | 75,336 | 1,674 | 9,834,765 |
| Supermemory | `v5ftc` | 169,555,080 | 18,328,682 | 187,883,762 | 118,989 | 2,644 | 11,737,050 |
| Hindsight | `v5ftc086` | 186,712,354 | 21,967,822 | 208,680,176 | 132,160 | 2,937 | 3,887,976 |
| RetainDB server | `v5ftc` | 257,390,464 | 52,781,967 | 310,172,431 | 196,436 | 4,365 | 3,025,041 |
| mem0 | `v5ftc` | 662,195,542 | 17,148,227 | 679,343,769 | 430,237 | 9,560 | 12,620,121 |
| Honcho | `v5ftc` | 932,135,069 | 42,546,584 | 974,681,653 | 617,278 | 13,716 | 19,551,966 |

**Use per session or per turn, not per question** (user, 2026-08-04). A Hermes
deployment ingests conversation, it does not answer 3,750 quiz items, so
per-question is a benchmark artifact and not a deployment cost. Denominators are
counted from `Step4_4.jsonl` over the same 30 personas: 1,579 sessions and 71,060
dialogue turns (142,129 messages, a mean of 45.0 turns and 90.0 messages per
session). Divide by 1,579 or 71,060, not by 3,750.

Caveats on these figures:

- **Per-turn figures inherit each window's failed-attempt inflation.** A window
  that contains failed persona attempts also holds the tokens those attempts spent
  re-processing the same turns. The overstatement is roughly 10 to 20% for
  Supermemory (nine failed attempts), 15 to 20% for OpenViking (nine attempts,
  most past session 38), and about 1% for mem0 (persona 1, failed at about session
  20). These are bounded estimates: the shared vLLM window cannot separate one
  persona's tokens. The other rows are clean of restarts.
- **Honcho's total is an upper bound, not a steady-state figure.**
  `HONCHO_DREAM_AFTER_SESSION=1` fires the dream consolidation call after every
  session, because a benchmark run never idles. mem0 and Hindsight do not carry
  this caveat: their ingest cadence is exactly what their plugins do, so those two
  totals are directly comparable to each other. Honcho's total is not comparable to
  either.
- mem0's figure is the corrected window covering all 30 personas
  (`token_usage_v5ftc_all30.json`), not the supervisor's window over 29.
- **Supermemory's window includes nine failed persona attempts that produced no
  answers.** It is the cost of obtaining 30 personas in this configuration, not the
  cost of a clean run, and it is not directly comparable to Hindsight, which had
  zero failures.
- **RetainDB's window includes two foreign workloads, about 0.6% combined, left in
  rather than subtracted.** A duplicate persona-0 probe container ran about 86
  minutes (about 1.3M tokens, about 0.42%), and a single-persona OpenViking
  featured smoke ran 12:57 to 14:01 EDT (0.20%). The OpenViking figure is not an
  estimate: replaying the same 12 sessions alone on an idle vLLM consumed 614,736
  generation-model tokens (`openviking/Results/token_usage_v5tokprobe.json`). Treat
  it as a lower bound.
- **The OpenViking window covers nine failed persona attempts across eight
  distinct personas** (persona 10 failed twice), roughly 25 persona-hours that
  produced no answers. Supermemory lost the same number of attempts, so OpenViking
  is the worst in this wave by personas lost, not by attempts.
- **Five of the 30 OpenViking personas ran with a tripled HTTP timeout.** Personas
  7, 10, 22, 26, and 29 were relaunched with `OPENVIKING_HTTP_TIMEOUT=1800` against
  the stock 600 the other 25 carried. Personas 16, 21, and 24 were relaunched at
  the stock 600. All relaunched personas also carried
  `OPENVIKING_RUN_DIR=/tmp/ovk_run`. Persona 24 recovered from the same timeout
  failure without the raise, which is why the deviation is reported rather than
  treated as required. The raise cannot change results for a persona that never
  reached 600 s on one request.
- **Mnemosyne costs an order of magnitude less than anything else here**, 18.1M
  total against Supermemory's 187.9M, 255 tokens per turn against Honcho's 13,716.
  The reason is the configuration, not the product. The featured configuration runs
  `EXTRACT=0`, so it never calls an LLM to ingest. Its 11,601 generation requests
  are 3,750 answers plus roughly 7,850 sleep and summary calls. Compare it to the
  others only if the comparison says so out loud.
- Mnemosyne's sidecar reads `scope=shard` because it ran as one container rather
  than a launcher pool. The value is still the whole-server delta across the full
  run window (20:36:07 to 21:53:47 UTC, 1 h 18 m), so it is a valid run total.
