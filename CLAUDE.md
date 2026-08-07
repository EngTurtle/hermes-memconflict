# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Goal

Compare **self-hostable memory-plugin providers for the Hermes agent** on the
[MemConflict](https://github.com/TaoZhen1110/MemConflict) benchmark. Each provider
runs in a **best-effort configuration** a real Hermes deployment would plausibly
use, not a hand-tuned lab setup.

Providers: **Mnemosyne** (reference), **Hindsight**, **mem0**, **Supermemory**,
**Honcho**, **OpenViking**, **RetainDB server edition**. Status per provider,
plus all measured results and per-arm numbers, is in `docs/BENCHMARK_MATRIX.md`.

**~~RetainDB Local~~ (npm `@retaindb/local`) is RULED OUT (2026-07-22):**
`search()` rebuilds the concept graph per candidate, so search is O(n²), ~245
s/query at n=2,897, no vendor knob, no banked number
(`docs/TROUBLESHOOTING.md`). `retaindb_server/` is a different product and is not
covered.

### The three standing rulings

**1. Fairness lives in the shared harness.** Every provider runs the same
dataset, answer model, judge model, top-K, and provider-agnostic scorer. A change
to that shared harness which moves one provider's numbers invalidates the
comparison. Provider-internal configuration is not that line.

**2. Best-effort configuration, applied evenly** (user, 2026-07-21). Tune every
provider's vendor-exposed knobs, not only the one that visibly misbehaves. Set
`MNEMOSYNE_LLM_MAX_TOKENS=3072`, `HINDSIGHT_API_LLM_TEMPERATURE_RETAIN=0.7`. A
hardcoded value with no knob is a property of the product. Prefer values the
vendor or model card endorses and record the justification. "We tuned until the
number went up" is out of bounds.

**3. Measure what the plugin returns** (user, 2026-07-21). The unit of measurement
is what each provider's Hermes plugin hands the agent at recall time. If a plugin
surfaces extracted facts rather than dialogue turns, facts reach the answer model.
Do not reshape retrieval output to look more like raw evidence.

**The trap behind ruling 3.** The judge scores supporting evidence hit
**semantically**. It ranks the first retrieved memory whose evidence supports
the reference answer, told to "not require exact wording". Extraction and
paraphrase carry no penalty, so storage format is not a confound. A provider's
real retrieval output scoring worse is a finding to report.

## Repository layout

```
benchmark/        shared, provider-agnostic harness (scorer, summarizer, llm glue, docker)
docs/             DECISIONS.md, TROUBLESHOOTING.md, BENCHMARK_MATRIX.md
mnemosyne/  hindsight/  mem0/  supermemory/  honcho/  openviking/  retaindb_server/   adapters
retaindb/         RetainDB local-edition adapter (RULED OUT, kept for the record)
external/         submodules: MemConflict, mnemosyne, RetainDB, honcho, hermes-agent
```

**Dependency invariants:** every provider and harness folder sits exactly one
level under the repo root. Adapters resolve the dataset as
`../external/MemConflict/...` and write to file-relative `Results/` and `Scores/`.
Nesting a provider deeper, or moving `external/`, breaks those paths.

**Per-contract artifact folders.** Past artifacts live under
`<provider>/{Results,Scores}/v<N>/`, plus `unclassified/` for off-contract or
unverifiable smokes. Adapters write NEW runs to the `Results/` and `Scores/` root;
sort a run into `v<N>/` when you bank it. Classify by evidence (manifest
`git_sha` mapped to a compose checkpoint, or a `serving_envelope_*.json` sidecar),
never by run tag. The mnemosyne run tagged `v2_baseline` is contract **v3**.

**Three docs, no new ones.** `BENCHMARK_MATRIX.md`: providers, arms, flags,
results, planned runs. `DECISIONS.md`: why the harness is built this way,
including reversed decisions. `TROUBLESHOOTING.md`: symptom, cause, fix, what did
not work. The env-var surface stays in `benchmark/docker/README.md`.

## Harness contract

`<provider>/eval_<provider>.py` ingests each persona's multi-session dialogue into
an isolated store. Per question it emits one JSONL row with `Model_Answer` and
`Retrieved_Memories` (`memory`/`created_at`/`score`) into `<provider>/Results/`.

Scorer and summarizer are **provider-agnostic** and score any provider's
output:

- `benchmark/score_resumable.py`: resumable LLM-judge scorer, the primary scoring
  stage; a thin wrapper over `external/MemConflict/Evaluation/eval_scoring.py`
  through a `MEMCONFLICT_EVAL_DIR` `sys.path` insert.
- `benchmark/summarize_scores.py`: per-persona scores to headline metrics.
- `benchmark/llm_reasoning.py`: reasoning-effort and JSON-mode wrapper over the
  upstream `llm_request`. **The one cross-folder Python dependency:** every
  adapter does `from llm_reasoning import ...` and prepends `../benchmark` to
  `sys.path`. Keep that insert if you touch an adapter's import block.

### Metrics

Headline: **macro answer accuracy**, the unweighted mean over the three conflict
types (**dynamic, static, conditional**). Also reported: micro answer accuracy,
supporting evidence hit at K, log-rank@3, update order recognition, contradiction
recognition. Every summary carries the `by_conflict_type` split. Supporting
evidence hit is the judge's semantic measure, so it, the evidence utilization gap,
and log-rank all move with a judge-config change.

**Two different things are called EUG.** `EUG_gap@3` is UPSTREAM's evidence
utilization gap, `SEH@3 − AA` per type, unweighted 3-type mean
(`Compute_Evidence_Utilization_Gap`, `diagnose_failures.py:73-96`). Quote this
one when citing MemConflict. `EUG-cond@5` is repo-local: mean answer accuracy over
questions whose evidence reached top-5. Measured values per arm: BENCHMARK_MATRIX.

Dataset: `external/MemConflict/Data/Step4_4.jsonl`, 30 personas, 3,750 questions.
Answer and judge model are the same per comparison, except the deliberate
gemma-4-12b judge arm (`_gj12`, penalty rubric `_gj12pen`) in BENCHMARK_MATRIX.

### Serving contract

The featured runs use contract v5. `vllm-gen` serves `qwen3.5-4b` from checkpoint
**AxionML/Qwen3.5-4B-NVFP4** on the pinned image `vllm/vllm-openai:nightly`
(digest `sha256:9894a751bdd2...c801533e`, engine `v0.23.1rc1.dev1373+g387189c42`,
flashinfer 0.6.14), with `--kv-cache-dtype fp8`, `--max-num-batched-tokens 4096`,
`--max-model-len 131072`, and `--enable-auto-tool-choice --tool-call-parser
qwen3_coder` (required by Supermemory's memory agent and Honcho's dialectic,
inert otherwise). `docs/DECISIONS.md` "Contract v5 serving envelope" records the
rationale.

The embedder is **Alibaba-NLP/gte-modernbert-base** (served `gte-modernbert-base`,
768 dims). The pooler flag `--pooler-config '{"use_activation": true}'` is
REQUIRED (unnormalized L2 ~38 without it). 768 dims keeps a 15-chunk batch under
Supermemory 0.0.5's ~12,288-float embedding cap (15 × 768 = 11,520,
`docs/TROUBLESHOOTING.md`). RetainDB zero-pads 768→1024.

**The served-model alias does not identify the checkpoint.** Recover the
checkpoint from the compose file at the run's repo SHA, not the manifest's
`OPENAI_MODEL`.

## Running

Primary path is Docker, from `benchmark/docker/` (see `docker/README.md`):

```bash
docker compose up -d vllm-gen vllm-embed              # shared inference servers, once
docker compose run -d --rm mnemosyne                  # a full Mnemosyne run
docker compose run -d --rm -e NUM_PERSONAS=1 -e RUN_TAG=smoke mnemosyne   # smoke
docker compose run -d --rm -e STAGE=score -e RUN_TAG=oracle mnemosyne     # re-score only

docker compose run -d --name hs_s0 -e RUN_TAG=full_s0 -e STAGE=generate \
  -e START_IDX=0 -e END_IDX=8 -e NUM_PERSONAS=30 hindsight   # one shard, persona range

benchmark/docker/run_shards.sh <provider> <tag>       # sharded run
```

vLLM servers have no profile, so a bare `up -d` starts only them. Provider
run-services are under the `run` profile. `STAGE` is `generate|score|summarize|all`.
Images are per provider (`Dockerfile.<provider>`, `entrypoint.<provider>.sh`);
renaming either needs `docker compose build <provider>` first. Hindsight shards
share `hindsight-pg`, mem0 shards share `qdrant`, Supermemory shards share the
central `supermemory-server`.

**Presets.** `PRESET=<provider>_{minimal,featured}_clocksync`
(`benchmark/docker/presets.sh`) carries a whole arm's env as one name, records it
in the manifest and run-contract hash, and exits 2 on an unknown or
wrong-provider name. `PRESET` unset changes no behaviour.

A single-persona smoke is `NUM_PERSONAS=1`. Do **not** cap with `MAX_SESSIONS=1`.
Early sessions answer 0 questions. A first clone needs
`git submodule update --init --recursive`.

**Host path, no Docker,** uses the repo-root `.venv` (`.venv/Scripts/python.exe`
on Windows). `mnemosyne/run_local.sh` (local vLLM) and `run.sh` (OpenRouter) set
env then `exec "$@"`; `run_full_local.sh` and `run_full.sh` shard personas across
processes.

```bash
python benchmark/score_resumable.py --input_file <provider>/Results/<file>.jsonl \
  --output_file <provider>/Scores/<file>_eval_scores.jsonl --checkpoint <provider>/Scores/<tag>_judged_checkpoint.jsonl
python benchmark/summarize_scores.py --scores_file <provider>/Scores/<file>_eval_scores.jsonl \
  --out_json <provider>/Scores/summary_<tag>.json --system <provider>
```

`benchmark/score_files.sh` scores result files on the host `.venv` with no provider
entrypoint involved. Pass `--temperature/--top_p/--top_k` explicitly; otherwise
`bench_judge_env` falls back to the qwen contract 0.6/0.95/20 and judges one arm
under different sampling than the rest.

## Mnemosyne arms

Flag or compose env in `eval_mnemosyne.py`. `--lifecycle` (B), `--canonical` (C),
`--oracle` (D) all imply `--extract`.

The `--oracle` arm is **gold-derived, not agent-curated**:
`mnemosyne/oracle_canonical.py` builds canonical slots from gold annotations at
build time, with no gold read at retrieval time. Numbers per arm: BENCHMARK_MATRIX.

- `MNEMOSYNE_FACT_RECALL_ENABLED=0`, the plugin default; a v1 probe showed fact
  rows retrieve worse.
- Set `MNEMOSYNE_WM_TTL_HOURS` high before ingesting backdated dialogue; the
  default 168h deletes it at ingest.
- `MNEMOSYNE_LLM_MAX_TOKENS` ≥2048 (3072 in use); at 512 sleep's model-refresh
  JSON truncates to zero proposals.
- `vllm-embed` is required; without it recall drops the vector path, the run
  still exits 0, and no harness check catches it.

## Hindsight arms

Compose env in `eval_hindsight.py`. Pin `hindsight-all==0.8.6` +
`pg0-embedded==0.15.0`.

- **Arm A**: extraction only, consolidation off.
- **Arm B**: `--prefer_observations --wait_consolidation` plus
  `HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=true`: the out-of-box feature set.
- **Arm C**: arm B plus `--retain_granularity exchange_append` (the per-exchange
  stable-document append the Hermes integration uses) and
  `RECALL_TYPES=observation`.

Arms are faithful in *feature configuration*, not in every sampling default
(ruling 2: retain temperature 0.7, not the shipped 0.1). Label them "out-of-box
feature set, best-effort sampling", never "untouched defaults".

- Set `RECALL_TYPES` on every run; the adapter exits 2 when unset. The plugin
  defaults to `["observation"]`; `all` opts into unfiltered recall. Plain
  `exchange` granularity is cadence-only legacy.
- `HINDSIGHT_API_LLM_TEMPERATURE_RETAIN=0.7` plus a 4096 retain cap. The shipped
  0.1 goes per request, so no server default reaches it, and with the unbounded
  `facts` array in the extraction grammar it repetition-looped ~3% of retains to
  the token cap (54.5% of retain GPU time).
- Extraction runs `enable_thinking:false` with
  `HINDSIGHT_API_LLM_STRICT_SCHEMA=1`; thinking-on cost 384 s per session on Qwen.
  Contract v1 on gemma needed thinking ON for grammar enforcement, so do not
  re-score v1 files under v2+ serving.
- The image needs a real `kill` binary (`procps`); pg0 liveness spawns `kill -0`
  and otherwise the daemon exits at "Database URL is required for migrations".
- vLLM's context-overflow 400 reports the threshold, not the actual size ("at
  least N input tokens" = window − max_tokens + 1). Measure real prompts through
  Hindsight's `llm_requests` table.
- `presence_penalty` cannot be a server-side default in vLLM 0.25.1;
  `get_diff_sampling_param` allowlists only `[repetition_penalty, temperature,
  top_k, top_p, min_p, max_new_tokens]` (`vllm/config/model.py:1506`).
- Consolidation needs `--max-model-len 32768`; its ~3× indented JSON overflowed
  8192 and cost retain 7.6-11% of sessions.
- `entrypoint.hindsight.sh` unsets empty `HINDSIGHT_API_*` vars before exec;
  `HindsightConfig.from_env()` int-parses set-but-empty strings and the daemon
  exits at boot. Keep the guard.

## mem0 arms

`mem0ai==2.0.14`, `from mem0 import Memory`, compose env in `eval_mem0.py`. Fully
self-hosted: internal extraction LLM, embedder, embedded qdrant. Answer/judge role
and mem0's internal LLM both default to the serving model (`OPENAI_*` /
`MEM0_LLM_*`); the embedder defaults to shared `vllm-embed` (the serving
contract decides the model: v4 bge-small-en-v1.5 384d, v5 gte-modernbert-base
768d). `infer=True` stays ON. Off means raw turns and discards the feature
under test.

**2.0.14 is ADD-only and cannot emit UPDATE/DELETE/NONE.** Verified at source
2026-07-28: `memory/main.py:1165-1168` stamps `"event": "ADD"` as a literal and
the two-phase update machinery is dead code; the decision was real in the
**0.1.118** pin only (`main.py:436`). The Hermes plugin declares
`mem0ai>=2.0.10,<3`, so 2.0.14 is what a deployment runs. **A finding to report,
not a regression to work around.** The runtime summary carries
`Total_Event_ADD/UPDATE/DELETE/NONE`, always 100% ADD.

Ingestion arms (`--retain_granularity` / `RETAIN_GRANULARITY`):

- `batch` (**default**): one `add()` per 8-message window
  (`MEM0_ADD_BATCH_SIZE`), the **MemConflict authors' cadence, NOT the plugin's**
  (`external/MemConflict/Evaluation/eval_memzero.py`). 8 is the compose default;
  `mem0_minimal_clocksync` sets **6** (`presets.sh:152`), which is what the banked
  v4minc run used. Cite the manifest.
- `session`: one `add()` per session.
- `exchange`: one `add()` per turn, the **plugin-faithful** cadence
  (`plugins/memory/mem0/__init__.py:488-498`). More internal LLM calls.

**Reference divergence, deliberate:** the authors' runner drives the **hosted**
platform; we run the same cadence self-hosted, same scorer
(`docs/DECISIONS.md:447`).

- `eval_mem0.py` imports `from mem0 import Memory` at the TOP, before any
  `sys.path` insert, and never puts the repo root on `sys.path`; the folder name
  `mem0/` otherwise shadows the SDK. `python mem0/eval_mem0.py` is safe,
  `python -m mem0.eval_mem0` from the repo root is not.
- The adapter `os.environ.setdefault`s `MEM0_TELEMETRY=False` before importing
  mem0; its PostHog events 403 through the egress proxy and flood stderr.
- Host smokes use HuggingFace `all-MiniLM-L6-v2` (`sentence-transformers`,
  `HF_HUB_DISABLE_XET=1`); offline and Docker runs set
  `MEM0_EMBEDDER_PROVIDER=openai` at `vllm-embed`. Host-smoke dims matched
  the v4 contract (384); under v5 the host path is off-contract until its
  shim serves a 768-dim model.
- Sharded runs need `MEM0_VECTOR_MODE=server` (compose default) against shared
  `qdrant`; the embedded store locks its on-disk `path` to one process. Each shard
  gets collection `mem0_<sanitized RUN_TAG>` (`entrypoint.mem0.sh`), personas
  isolate by `user_id`, results merge at JSONL level. Generate passes
  `--reset_collection`; `ALLOW_EXISTING_COLLECTION=1` appends or resumes. qdrant
  ships no shell or curl, so it has no healthcheck and the entrypoint waits for
  readiness itself.

## Supermemory arms

Compose env in `eval_supermemory.py`. A native server binary the adapter spawns
and drives over REST: a local HTTP server *with* an internal extraction LLM.
Headline arm `--search_mode hybrid`, the Hermes plugin default (`/v4/search`,
memories first with a doc-chunk fallback). Alternates: `memories`, the diagnostic
`--documents_arm` (`/v3/search`), and `session|exchange|message` ingest
granularities.

- **Two LLM roles.** The answer/judge model is fairness-locked (`OPENAI_*` via
  `answer_env.sh`); Supermemory's extraction model is `SUPERMEMORY_LLM_*`, which
  `_supermemory_server.py` maps onto the spawned server's own `OPENAI_*`. Keep
  extraction config out of the answer and judge path.
- **Ingestion is async.** `POST /v3/documents` returns `queued`; a memory is
  searchable at `done`. The adapter drains per session (`wait_for_drain`, per-doc
  poll with a `/v3/documents/processing` fallback) before answering, otherwise
  recall races the queue. Ingest wall-time is extraction-model-bound.
- The `/v4/search` threshold is OFF (explicit 0.0): the vendor default 0.6 admits
  fewer memories than the shared top-K, because bge-base cosine sits below 0.6 for
  many relevant memories. `SUPERMEMORY_SEARCH_THRESHOLD=0.6` reproduces it as an
  arm. We request the plugin's `max_recall_results` (10) and slice to top-K.
- The binary ships only via GitHub Releases; `Dockerfile.supermemory` bakes a
  pinned `SUPERMEMORY_SERVER_VERSION` (0.0.5; 0.0.6 and 0.0.7-rc.2 ship a broken
  linux-x64 ingest engine). Keep the pin: the `latest` lookup hits rate-limited
  api.github.com. `_mock_supermemory_server.py` is the egress-blocked fallback
  only, never a headline number.
- The memory agent requires tool-calling (7 function tools, `tool_choice:auto`),
  so contract v4 enables `--enable-auto-tool-choice --tool-call-parser
  qwen3_coder`. Without them the request 400s into disabled telemetry and recall
  degrades to chunk-RAG; the trace is `memory agent failed (Nms)` plus
  `finalized: N chunks, 0 memories`.
- The 0.0.5 workflow dispatcher can die with no log evidence while HTTP stays 200;
  documents then sit at `queued` and every drain times out. Detection is
  behavioural. Probe ingest liveness before each wave; a restart recovers it, but
  queued docs wait for the ~30-minute cron sweep.
- Sharded runs share ONE central server (`SUPERMEMORY_SERVER_MODE=shared`) that
  owns a single embedded data dir, persists on `supermemory_data` with a stable
  bearer key published to `supermemory_shared/api_key`, and gates its healthcheck
  on key-file-present plus API-live. Per-run isolation is by containerTag
  namespace (`RUN_TAG` with `_s<k>` stripped to `<run>_p<persona>`), not a per-run
  DB, so use a fresh RUN_TAG per wave and reclaim `docker_supermemory_data`.
  Throughput caps on `SUPERMEMORY_INGEST_CONCURRENCY` (default 15).
- `spawn` mode (`--no-deps`) is the standalone single-process path; under
  `BENCH_CLOCKSYNC=1` it respawns the server once per session
  (`SUPERMEMORY_RESPAWN_PER_SESSION`).

## Honcho arms

Compose env in `eval_honcho.py`. Self-hosted Honcho (`external/honcho`,
plastic-labs/honcho v3.0.9) is a FastAPI API plus a deriver worker (the extraction
worker) on Postgres/pgvector. It has no ranked memory list; it returns a **peer
model**. The Hermes plugin injects a markdown block of named sections, and that
block is the product under test (ruling 3).

Recall modes (`HONCHO_RECALL_MODE`):

- `hybrid` (HEADLINE, featured, the plugin's `recallMode` default). Plugin
  injection order: session summary, user representation + peer card, AI
  self-representation + card, then a dialectic answer (Honcho's internal
  question-answering call, `ai_peer.chat(query, target=user_peer)`) clipped to
  600 chars. `plugin_native_recall=True`, so no top-K slice.
- `conclusions` (MINIMAL): derived facts, semantically ranked
  (`peer.conclusions_of(target).query(question, top_k=5)`), real `created_at`,
  shared top-K.
- `base` / `dialectic` / `search`: diagnostics. `search` is raw RRF-ranked
  message search, an agent-invoked tool in Hermes, never automatic injection.

Ingestion mirrors `sync_turn`: one `add_messages` per exchange, both peers,
25000-char chunking with a `"[continued] "` prefix, and no `created_at`, because
the plugin sends none (`session.py:45-54`). The plugin-faithful temporal path is
`BENCH_CLOCKSYNC` moving the server clock. `HONCHO_SEND_CREATED_AT=1` is a
vendor-exposed deviation arm, default off. Observation modes: `directional`
(plugin new-install default, featured) / `unified` (plugin legacy default,
minimal: `conclusions` recall never reads the AI self-representation, so
`unified` halves deriver spend on that arm).

Two deviations from the plugin's fire-and-forget behaviour:

1. **Drain.** The adapter polls `queue_status()` after each session's ingest until
   `pending_work_units==0` and `in_progress_work_units==0`.
   `DERIVER_FLUSH_ENABLED=true` bypasses
   `DERIVER_REPRESENTATION_BATCH_MAX_TOKENS`, so a tail batch under threshold does
   not stall the drain.
2. **Featured-arm manual dream** (`HONCHO_DREAM_AFTER_SESSION=1`, user ruling
   2026-07-31). After each drain the adapter calls `schedule_dream` (POST
   `/v3/workspaces/{id}/schedule_dream`, `dream_type=omni`, bypassing the document
   threshold, idle timer, and 8h spacing) per observer→observed pair hybrid recall
   reads, then drains again. A benchmark run never idles, so the scheduler's own
   60-minute idle dream never fires. Label the arm "shipped consolidation,
   manually cadenced" (rationale: `docs/DECISIONS.md`). If dreaming
   lowers macro answer accuracy, report it.

- Run the vendor's `scripts/configure_embeddings.py --yes` after provisioning
  (`_honcho_server.py`'s `apply_embedding_dim_fix()`, which verifies `atttypmod`).
  Migrations hardcode `Vector(1536)` while `src/models.py` sizes from
  `EMBEDDING_VECTOR_DIMENSIONS`, so below 1536 dims the API and deriver refuse to
  boot.
- Set `HONCHO_EMBEDDER_BASE_URL`; `HonchoServer.start()` refuses spawn mode
  without it. An unreachable embedder empties a run at exit 0. Every "save
  representation" fails on the embed step while the deriver keeps draining.
- `HONCHO_LLM_THINKING_EFFORT=low` for a reasoning model
  (`src/llm/backends/openai.py:297`): gpt-oss-20b at default effort spent its
  whole token budget reasoning and logged `Observation Count 0`. Leave unset for
  v4 qwen3.5-4b (non-reasoning).
- Set TRANSPORT next to MODEL; `_normalize_model_transport`
  (`src/config.py:262-275`) splits `openai/gpt-oss-20b` only when transport is
  unset, and OpenRouter does not serve the stripped id.
- `HONCHO_TIMEOUT=300` in Docker (compose default). The plugin's 30 s works
  because it runs the dialectic on a background thread with a one-turn lag; the
  adapter calls it inline, so 30 s empties the dialectic section on slow serving.
- Use `uv sync --frozen` in `external/honcho`; plain `uv sync` rewrites the vendor
  `uv.lock`. The Docker build uses `--frozen`.
- The dialectic requires tool calling (`search_memory`, `search_messages`,
  `search_messages_temporal`); contract v4's flags enable it.

## OpenViking arms

Compose env in `eval_openviking.py`. `openviking==0.4.12` ships one pip
distribution: the `openviking-server` console script (FastAPI/uvicorn) plus
self-contained storage (an AGFS content store and a local vector index in one
workspace dir). The adapter speaks raw `httpx` and never imports the `openviking`
package; the Hermes plugin
(`external/hermes-agent/plugins/memory/openviking/__init__.py`, pinned SHA
`6d17b2a5`) has no SDK client either, and the provider folder `openviking/`
shadows the installed package name, so the repo root never goes on `sys.path`.

OpenViking extracts a memory TREE per user (`viking://user/memories/` with
`profile.md`, `preferences/`, `entities/`, `events/YYYY/MM/DD/`, `cases/`), not a
flat memory list. Raw session messages never enter the memory search space.
Search results carry `uri`, `abstract`, `category`, and `score` and no timestamp
field, so every returned item reports `created_at` as `Unknown Time`.

Recall modes (`--recall_mode` / `OPENVIKING_RECALL_MODE`):

- `prefetch` (HEADLINE, featured, plugin-faithful, the one scored arm): the
  plugin's read surface. A session-start block (the `profile.md` body plus a
  `preferences/` and `entities/` listing) as item 0, then the `POST
  /api/v1/search/search` entries selected by the plugin's own
  `_select_recall_candidates`.
- `find` (MINIMAL, diagnostic): deterministic `POST /api/v1/search/find` with no
  LLM in the retrieval path. The integration proof and retrieval floor, not a
  comparison number (user ruling 2026-08-04): only `prefetch` is scored.
- `search`: the `/api/v1/search/search` entries alone, no session-start block.
  Auxiliary, no planned run.

All three arms pass the plugin's selection width (`recall_limit`, default 6) to
the answer model whole. `plugin_native_recall=True` everywhere, no harness top-K
slice (user ruling 2026-08-04). The scorer is unaffected:
`extract_top_k_retrieved_memories` slices the stored list at its own white-box K
(3 primary, 5 max).

- Ingest cadence `--retain_granularity` / `OPENVIKING_RETAIN_GRANULARITY` /
  `RETAIN_GRANULARITY`: `exchange` (default, the plugin's `sync_turn` cadence, one
  `POST /api/v1/sessions/{sid}/messages/batch` per user-plus-assistant exchange)
  or `session` (one POST per session, chunked at the server's 100-message cap).
- `OPENVIKING_SEND_CREATED_AT=0` (default, plugin-faithful; the plugin sends
  none). `1` is a vendor-exposed deviation, not a planned arm (user ruling
  2026-08-04): the comparison picks a Hermes plugin, so only plugin behaviour is
  measured. Under the default, `BENCH_CLOCKSYNC` is the temporal path.
- `OPENVIKING_RECALL_SCORE_THRESHOLD=0.15`, the plugin default, applied
  client-side after the server search runs at `score_threshold: 0`. Confirmed the
  shipped default across OpenViking's agent integrations (2026-08-04); do not
  lower it.
- `OPENVIKING_RECALL_LIMIT=6`; `OPENVIKING_RECALL_MAX_INJECTED_CHARS=4000` /
  `OPENVIKING_PROFILE_TOKEN_BUDGET=6000`; `OPENVIKING_RECALL_FULL_READ_LIMIT=2` /
  `OPENVIKING_RECALL_PREFER_ABSTRACT=0`, all plugin defaults.
- `OPENVIKING_RECALL_TIMEOUT_SECONDS=60` / `_REQUEST_TIMEOUT_SECONDS=30`, against
  the plugin's 4.0 / 3.0. The plugin runs `prefetch()` on a background thread and
  drops what has not arrived; the adapter calls recall inline, where the same
  budget returns empty recall with no error. 60 is the plugin's own clamp maximum.
  The precedent is `HONCHO_TIMEOUT=300`.
- `OPENVIKING_LLM_*`: qwen3.5-4b on `vllm-gen`, `max_tokens` 4096, temperature 0.0
  (the vendor sample value), `max_concurrent` 8 against the vendor default 64.
  Extraction and search-intent analysis share the fairness-locked answer server,
  so 64 in-flight internal calls per shard would take its scheduling.
- `OPENVIKING_EMBEDDER_*`: gte-modernbert-base on `vllm-embed`, 768 dims. A model
  or dimension change invalidates an existing workspace.
- `ov.conf` omits `rerank` entirely (the config model is `extra: "forbid"`, so
  omission is the off state) and sets `query_planner: null` (falls back to `vlm`,
  one model for both internal roles per the best-effort ruling).

**Drain, a deliberate deviation.** The plugin POSTs
`/api/v1/sessions/{sid}/commit` at session end and moves on. The adapter polls
`GET /api/v1/tasks/{task_id}`, then calls `POST /api/v1/system/wait`, and raises
on a failed or cancelled task, on the drain timeout (`OPENVIKING_DRAIN_TIMEOUT_S`,
default 1800), and on any `error_count > 0`. A broken embedder surfaces in that
`error_count` and nowhere else; otherwise the HTTP path stays 200 and the run
exits 0 with empty recall. One commit per session maps the plugin's
`on_session_end`.

Isolation is dev auth mode plus a per-persona `X-OpenViking-User` header. On
loopback the server's `auth_mode: "dev"` takes identity from headers with no key,
and the `user` value alone scopes every read and write. Each persona's user id is
`<OPENVIKING_USER_PREFIX><persona tag>`, and `begin_persona` wipes it with
`DELETE /api/v1/fs?uri=viking://user/memories&recursive=true` so a re-run is
idempotent.

- `OPENVIKING_SERVER_MODE=spawn` (default and the only sharded topology): one
  server per container, workspace under `openviking/.openviking_runs/`. `shared`
  attaches to an operator-run `OPENVIKING_ENDPOINT`; `entrypoint.openviking.sh`
  exits 2 on `shared` under `BENCH_CLOCKSYNC=1`, because one attached server has
  one perceived clock and cannot sit at N shards' logical session dates at once.
- `BENCH_CLOCKSYNC=1`: only the spawned server child runs under libfaketime,
  injected by `_openviking_server.py` into the child env. Manifest
  `temporal_capability` is `controlled_process_clock`.
- Presets (`benchmark/docker/presets.sh`): `openviking_minimal_clocksync`
  (`find`) and `openviking_featured_clocksync` (`prefetch`). Both force
  `BENCH_CLOCKSYNC=1` and `OPENVIKING_SERVER_MODE=spawn`.

- The server exits 1 at boot with `Unknown config field 'telemetry.enabled'`:
  0.4.12's `TelemetryConfig` is `{"tracer": {...}}` and every config model rejects
  unknown keys. Omit the `telemetry` section from `ov.conf`; `tracer.enabled` and
  `server.usage_reporter.enabled` both default to `False`, so omission is the off
  state.
- gpt-oss-20b at default reasoning effort burns the whole `vlm.max_tokens` budget
  on reasoning, and a long extraction call can outlive OpenRouter's keep-alive
  window, which pads the response with newlines and closes with no JSON. Set
  `OPENVIKING_LLM_EXTRA_BODY='{"reasoning": {"effort": "low"}}'` plus
  `OPENVIKING_LLM_MAX_TOKENS=8192`; `run_openviking.sh` sets both for the
  OpenRouter path. Local qwen3.5-4b is non-reasoning and needs neither. Same
  mechanism as `HONCHO_LLM_THINKING_EFFORT=low`.
- Put the workspace on container-local storage: `OPENVIKING_RUN_DIR=/tmp/ovk_run`
  on every host, native Linux included. Concurrent workspace I/O at shard scale
  trips a queue-worker exception that never acks, so the commit task stays
  `pending` until the drain timeout. Detection is behavioural: any lane silent
  more than 8 min is stuck (normal commits run 30-90 s); kill and relaunch that
  persona rather than waiting out the 1800 s timeout, which restarts it from
  session 0.
- `docker rm -f` leaves the vendor's `.openviking.pid` lock on the per-run
  workspace; a reused workspace reads the lock as held (`DataDirectoryLocked` at
  boot). Delete `openviking/.openviking_runs/<run_tag>/` before relaunching that
  persona.
- In `prefetch` mode `/api/v1/search/search` runs LLM intent analysis and can
  emit zero queries, which empties that question's recall. The plugin does not
  fall back to `find` on an empty result, only on request failure. This is
  plugin-faithful behaviour to report (ruling 3), quantified by the `find`
  diagnostic arm.

## Serving: the silent engine wedge

Solved 2026-07-21, workarounds retired in contract v4 and commented out in the
compose file. Full history in `docs/TROUBLESHOOTING.md`.

- **Signature:** ~100% GPU utilisation at idle power (~64 W of 300 W),
  `Avg generation throughput: 0.0 tokens/s` with `Running>0`, `/health` still 200.
  `nvidia-smi` utilisation-against-power is the only reliable detector.
- **Cause:** `flashinfer-ai/flashinfer#3615`, the multi-CTA radix top-k sampler on
  consumer Blackwell SM120/121 (this host, RTX 5070 Ti).
- **Fix if it recurs:** `VLLM_USE_FLASHINFER_SAMPLER=0` on vllm-gen; confirm by
  the log line `FlashInfer top-p/top-k sampling disabled ...`.
- **Do not regress the vLLM version.** The fix is flashinfer 0.6.14 via vLLM PR
  #47669, merged after `v0.25.1` was cut, so no release tag contains it.
- **Mitigations that stay:** `RETRY_TIMES=40`/2-45 s in `answer_env.sh` (~13 min)
  so shards stall through a restart, and `benchmark/docker/watch_vllm.sh`.
- **Apply serving changes only from a quiet server.** A recreate under live shards
  kills any whose retry budget is spent. A restart costs ~8 min, because
  `llm_request`'s 300 s timeout must expire on each dead socket; do not lower that
  timeout. A 16k-token answer at 27-way concurrency can exceed 200 s.

## Working conventions

- **Never modify anything under `external/`**: pinned submodules, the exact code
  under test.
- Commits are **unsigned**: `git -c commit.gpgsign=false commit ...`.
- Long runs go **detached** (`compose run -d`), then monitor. Never foreground.
- **While a long run is in flight, wake every ~55 minutes**: the prompt cache
  holding session context has a 1-hour TTL. Cache upkeep, not a progress poll. A
  finished container notifies on its own.
- **Never `git add -A`.** `Results/` and `Scores/` hold untracked-but-not-ignored
  artifacts. Stage by path, or repair with
  `git reset -q -- '*/Results/*' '*/Scores/*' 'benchmark/Scores/*'`.
- **Result JSONL is committed as `.7z`, never raw** (2026-07-29).
  `*/Results/**/*.jsonl` is gitignored; commit `<file>.jsonl.7z` beside it and
  extract before scoring:
  `7z e <provider>/Results/v4/<file>.jsonl.7z -o<provider>/Results/v4/`.
  Compression (~27:1) is the fix, not capping the capture: the diagnostic capture
  stores every ranked candidate, and that depth is the measurement.
- **After any merge, grep the whole tree for `<<<<<<<` before committing.** A lone
  opener whose `=======` and `>>>>>>>` were removed passes every syntax check.
- Windows with PowerShell, Git-Bash, and Docker Desktop on WSL2. `git show
  <branch>:<path>` needs `MSYS_NO_PATHCONV=1`, which in the same script mangles
  `git -C /c/Users/...`. Use `cd` plus `git rev-parse` there.
- **Findings go into the three `docs/` files, not new markdown**: a symptom to
  `TROUBLESHOOTING.md`, a config choice plus evidence to `DECISIONS.md`, an arm or
  measured number to `BENCHMARK_MATRIX.md`. No per-run report, review, or reply
  docs at the repo or provider root.
- **A doc that describes a bug is not evidence the bug is live.** Confirm against
  the code before acting on, or re-publishing, any claim about current behaviour.

### Writing style

All prose (docs, READMEs, commit messages, comments, chat replies) follows
**ASD-STE100 Simplified Technical English**
(`.claude/skills/ste-writing-skill/ste-writing-skill.md`): strict mode for
procedures, runbooks, and code comments, STE-flavored for docs and READMEs.

**No em dashes; hold the register.** Prose carries no em dash. Replace each with a
comma, a colon, parentheses, or a new sentence. Keep the impersonal,
no-contractions register: it is the chosen style, not an AI tell. Run
`unslop_text_scan.py` (JCarterJohnson/vibecoded-design-tells, the unslop-ai-text
scanner) before commit and clear the high-severity hits.

**Report a change in this form: Changed [Y] to fix [Z] via [Mechanism].**

**Concise, but in standard prose.** Keep text short, and keep standard prose
grammar and scannable formatting: bullets or short paragraphs, full sentences
with subjects and verbs. Do not use compressed developer shorthand. A sentence
the reader must decode ("The floor is live content, cutting further removes
settings, not prose") is worse than two plain sentences that say the same
thing ("The remaining text is provider settings. Cutting more would delete
settings, not prose.").

**Do not write:**
- Hypothetical disaster framing ("if you skip this, everything silently breaks").
  Give the setting and the failing component: "Set X=Y; without it Z fails at
  <point>."
- A re-explanation of the original problem, or a justification of why the previous
  code was wrong, unless asked for it.
- Conversational filler ("Great question!", "Certainly!", "That changes
  everything").

**Name the mechanism, not the shape of the problem.** Name the run, the flag, the
file, and the number. Lead with the measurement or the failing component. Prefer a
table or a short list of facts over prose that narrates reasoning. If deleting the
numbers would leave the sentence intact, it was not saying anything. A metaphor
names no mechanism, and the vague version is what survives into the durable
record.

**Never invent a term for something that already has a name.** Prefer the
MemConflict paper's vocabulary (`external/MemConflict/README.md:109-116`):

| write this | meaning | scorer key |
|---|---|---|
| answer accuracy | the final answer matches the gold answer | AA |
| supporting evidence hit at K | the top-K retrieved memories contain the gold item | SEH@K |
| support rank | how highly the gold memory item is ranked | SRS |
| update order recognition | the system recognizes update order | `update_awareness_and_order_consistency_score` |
| contradiction recognition | the system recognizes contradictory candidates | `conflict_recognition_score` |
| evidence utilization gap | retrieved gold memories become correct answers | EUG |

After the paper, use the name the code or the vendor uses. An answer that
declines is a **refusal**. Define a vendor term in plain words on first use in any
report or doc (Honcho's dialectic, deriver, dream). Before naming a behaviour,
grep the scorer's `Metrics` keys, this table, and the three `docs/` files.

**Write the words, not the acronym**, on every use: "update order recognition
0.547", not "UOCS 0.547". Keep the acronym only where it names a literal thing to
type or grep: a `Metrics` key, a narrow table column, a quotation. If a behaviour
has no name, say what was counted ("answers containing 'inconsistent' or
'conflicting'") rather than minting a label.

**Banned phrases**, each replacing a fact with a gesture at a fact: "different
altitudes" (name run, metric, value), "load-bearing" (name what breaks and how),
"blast radius" (name the bound), "war story" (name the incident), holistic,
leverage, at a high level, seamless, robust, and `P0` / `must-fix` / `Verdict:`
in prose.

**Code comments say WHY a line exists, never what it does.** Keep the non-obvious
constraint, the upstream bug worked around, the measured value and its source, and
what breaks if the value changes.
