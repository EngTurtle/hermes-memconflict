# Engineering decisions

This file states why the benchmark is built the way it is. Each entry gives the
decision, the mechanism it acts on, and the evidence behind it. [Reversed](#reversed)
lists decisions that were later overturned; they stay on record because the
reasoning that produced them is easy to repeat.

The published report at <https://engturtle.github.io/hermes-memconflict/report/>
states the method these decisions produced. The repository README at
<https://github.com/EngTurtle/hermes-memconflict> covers the project scope and
layout.

---

## Fairness

### The fairness line is the shared harness, not each provider's config

Every provider gets the same dataset, answer model, judge model, judge decoding,
top-K, prompts, and scorer. Inside that boundary, the benchmark tunes each
provider's own exposed knobs to a best-effort setting.

This project selects a provider to deploy, so each provider runs in the
best-effort configuration a real deployment would use. A vendor-exposed setting
that makes a framework work better on the serving model is part of that config,
and declining to use it understates the provider for no practical gain. The
benchmark applies this standard evenly. Where a provider hardcodes a value with no
exposed knob (Mnemosyne's internal temperatures), there is nothing to tune, and
that is a property of the product.

Prefer values the vendor or model card already endorses over hand-picked ones,
and record the justification. "We tuned until the number went up" is out of
bounds.

### What is under test: what the plugin hands the agent

The unit of measurement is what each provider's Hermes plugin returns at recall
time. If a plugin surfaces extracted facts rather than dialogue turns, then facts
are the product under test, and facts are what must reach the answer model.

Keep each provider's retrieval output in the shape its plugin returns. Reshaping
it to look more like raw evidence measures a configuration no Hermes deployment
would run, and publishes a score for a setup that does not exist.

The trap is that the MemConflict judge scores supporting evidence hit
semantically ("evidence supporting the reference answer... do not require exact
wording"), so it does not penalize extracted or paraphrased memories against
verbatim ones. Several early harness decisions started from the opposite, false
premise and suppressed real plugin behavior to satisfy it. When a provider's real
retrieval output scores worse, that is a finding to report, not a configuration
to engineer away.

### Decoding is centralized in one file

Every provider entrypoint sources `benchmark/docker/answer_env.sh`, which exports
the answer and judge decoding for all of them. Before it existed, only
`entrypoint.mnemosyne.sh` set `OPENAI_TEMPERATURE`, so Hindsight answered and
judged at the vLLM server default. That omission silently made the comparison
invalid.

Entrypoints also call the shared `run_score()` / `run_summarize()` from that file
rather than carrying private copies, so judge config cannot diverge per provider.
A check across all six entrypoints confirmed none exports its own
`OPENAI_TEMPERATURE`.

### The top-5 retrieval contract is frozen

Changing K mid-comparison invalidates every completed run. Providers whose native
recall budget is larger (Supermemory's `max_recall_results=10`, RetainDB's server
default 10) request their own budget and then slice to the shared top-K, so the
harness exercises the plugin's own behavior without breaking the shared line.

The adapters over-fetch differently at non-default K (Mnemosyne `max(top_k,5)+8`,
Hindsight trims its whole pre-compaction list, RetainDB requests exactly
`top_k`). This is irrelevant at K=5 and matters only if K ever changes.

### Run one provider at a time

The only overlap allowed is a run's own shards, so serving contention stays
comparable across arms.

### The featured contract is v5

The featured comparison runs on contract v5: the shared serving stack plus
`--max-model-len 131072` on `vllm-gen` and the `gte-modernbert-base` embedder at
768 dims (see "Contract v5 serving envelope" under Serving).

The served alias `qwen3.5-4b` does not identify the checkpoint. To audit which
weights produced a result, read the compose file at the run's repo SHA, not the
manifest's `OPENAI_MODEL`.

### Answer sampling is left unseeded

This matches deployment. The project will measure sampling variance later with a
~300-question replay-repeat, only if close calls need it.

### Score leakage: raw retrieval scores are not shown to the answerer

`eval_common.py` no longer renders `(score=...)` into the answer prompt. A score
of 0.5 means something different for each provider, and the answerer can misread
it as calibrated confidence. The row still stores scores, so supporting evidence
hit at K, log-rank, and EUG@5 are unaffected. This intentionally breaks replay
comparability against pre-change files.

---

## Serving

### Contract v4 serving envelope

Retired in favor of v5 for featured runs. The five v4 changes (nightly image with
flashinfer 0.6.14, checkpoint back to NVFP4, `--kv-cache-dtype fp8`,
`--max-num-batched-tokens 4096`, and tool-calling flags) are described in the
Fairness contract table above; the full stress-run evidence stays in git history.

### Contract v5 serving envelope, the featured contract (2026-08-01)

The featured runs use their own serving contract (user ruling 2026-08-01). Two
changes from v4, both on the shared servers so they reach all six providers
identically.

**`--max-model-len 131072` on `vllm-gen`.** Sized by the largest measured Honcho
prompt: the dream (Honcho's consolidation job) accumulates tool results to
~72,708 tokens, plus the 8,192-token output reservation, ~81,000 total with 38%
headroom. 65,536 revives the dialectic but leaves dream deduction failing. The
checkpoint's native window is 262,144. vLLM allocates KV per actual token, so the
raise costs nothing until the pool runs out; on the 16 GiB card the v5 boot log
reads 434,238 KV tokens = 3.31 full-window requests (identical at util 0.85 and
0.74), enough for single-persona smokes, not a 30-persona wave.

**Embedder swap to `Alibaba-NLP/gte-modernbert-base`**, served name
`gte-modernbert-base`, 768 dimensions, `--max-model-len 8192`,
`--pooler-config '{"use_activation": true}'`. This reverses the 2026-07-26 "keep
bge-small" decision (see Reversed). The 8192-token native window ends every
512-token truncation shim. The pooler flag is REQUIRED because the checkpoint
ships no Normalize module and vLLM otherwise returns unnormalized vectors
(measured L2 ~37-38). The model defines no instruction prefix at all
(`"prompts": {}`). MTEB retrieval is 55.33 against bge-small's ~51.7. vLLM issue
#28564, which ruled this model out on 2026-07-26, does not reproduce on v0.25.1
(live boot, 2026-08-02). RetainDB zero-pads 768 to 1024, rank-preserving (it
padded 384 to 1024 under v4). The served name changed on purpose: adapters pass
the model name in the request body, so a stale default fails loudly with
model-not-found instead of producing a manifest that claims one model's vectors
came from another.

**Amendment, 2026-08-02: the first v5 embedder pick lasted one day.** The initial
pick was `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions (32k input window, MTEB
~61.8). Two problems surfaced the same night, before anything was banked. It is a
decoder (`Qwen3ForCausalLM` under `--convert embed`), so vLLM reserves 3.5 GiB KV
per 32,768-token request where a BERT-family encoder reserves none, forcing
`vllm-embed` to util 0.21, a 16,384-token window, and `vllm-gen` down to 0.74. And
Supermemory 0.0.5 cannot ingest at 1024 dimensions: its embedding step fails any
15-chunk batch over ~12,288 floats (`docs/TROUBLESHOOTING.md`), so every document
over 12 chunks (~9,500 chars) wedges, 99.9% of dataset sessions at session
granularity, where 15x768 = 11,520 floats fits.

User ruling (2026-08-02): switch to a 768-dimension model, zero-pad where a schema
needs wider, re-run the Supermemory smoke. The replacement `gte-modernbert-base`
is an encoder, so it also retires the KV problem: no KV cache, util 0.07,
`vllm-gen` back at 0.85, window at the native 8192. The five persona-27 smokes at
1024 dims validated process, not numbers, and are re-provisioned at 768.

**The shared-embedder claim holds for Supermemory.** Upstream issue #1336 (server
ignores `SUPERMEMORY_EMBEDDING_*`, bundled 768-dim model) was probed against the
pinned 0.0.5 on 2026-08-01 and does NOT apply: one ingest produced eight
`/v1/embeddings` requests at `vllm-embed`, the server reshapes both pgvector
columns to the declared dimension at boot (`reshaped ... -> vector(384)`), and a
negative control (declared 1024 against the 384-wide embedder) failed the upsert.
Both ingest-time and query-time embedding are remote.

**Stores are dimension-bound, so v5 re-provisions every store:** fresh mem0 qdrant
collections, a fresh `supermemory_data` volume, Honcho's column retype via the
vendor's `configure_embeddings.py` at 768, and RetainDB's pad shim padding 768 to
1024. mem0's `MEM0_ADD_BATCH_SIZE` returns to the MemConflict authors' 8 (the 6
existed only to fit the 512-token cap) and `MEM0_EMBED_TRUNCATE_TOKENS` becomes -1
(vLLM's "truncate to the served window", tracking `VLLM_EMBED_MAX_LEN`).

Two v5 open items. Honcho's `EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE=never` is
correct again because gte-modernbert-base has one fixed output width, like
bge-small, so the matryoshka probe planned during the one-day Qwen3-Embedding
interim is moot. Honcho's host-smoke embed shim
(`honcho/_local_embed_server.py`) still serves bge-small at 384 via fastembed, so
host smokes and Docker runs no longer agree on the vector width; host smokes are
off-contract under v5 until that shim is updated or retired.

### Retired serving workarounds, commented out in compose

Three wedge workarounds are commented out in the compose file and must not be
re-added: `VLLM_USE_FLASHINFER_SAMPLER=0`, the compilation-config
`{"cudagraph_mode": "PIECEWISE"}`, and `--no-async-scheduling`. They protected
against the engine wedge while release images shipped flashinfer <= 0.6.13. The
v4 nightly carries 0.6.14 (the fix, PR #47669, in no release tag), and the
saturation stress run verified stability with the sampler active. The full
evidence and elimination matrix stay in
[TROUBLESHOOTING](TROUBLESHOOTING.md#the-silent-engine-wedge-flashinfer-top-k-sampler-race).

### `--max-model-len`: 32768 under v4, 131072 under v5

Raised from 8192 (the pre-v4 value) because Hindsight's consolidation renders its
token budgets as ~3x JSON on top of a ~2.7k-token fixed template. At 8192 it
never fit, and retain lost ~7.6 to 11% of sessions to overflow.

The window stays raised despite the runaway-output caps making it look excessive.
vLLM allocates KV per actual token, not per `max_model_len`. Measured 26 to 30
resident requests against a theoretical 6.11x oversubscription, so lowering the
window frees nothing and breaks consolidation.

### Thinking off for provider-internal calls, on for the shared answer role

The harness sets this server-side via
`--default-chat-template-kwargs '{"enable_thinking": false}'`. Request-level
overrides win, and the answer role explicitly requests thinking-on.

Server-side is the only option because Mnemosyne's pinned submodule exposes no
`extra_body`/`chat_template_kwargs` passthrough at any internal call site, and
editing `external/` is forbidden. Turning internal thinking off at all is
justified because thinking-on structured extraction on Qwen cost ~384s against
~14s per call with no quality gain, and it overflowed internal token budgets.

### `RETRY_TIMES=40`, `MEMCONFLICT_REQUEST_TIMEOUT=600`, `SCORE_WORKERS=40`

The retry budget (~13 min) is sized to the worst realistic cluster of serving
outages (wedge, watchdog restart, and manual recreate stacking), not one clean
restart. The project does not raise it further: the retry window is also how long
a fatal misconfiguration stays invisible.

Judge concurrency and the request timeout move together. Raising workers without
the timeout just converts throughput into timeouts. The measured gain from 24 to
40 workers was only ~13% (5.5 to 4.8 s/question) because KV saturates at 26 to 30
resident requests. The apparent GPU headroom during scoring is the margin that
keeps judge calls under timeout, not free capacity.

Two serving profiles: `VLLM_GEN_GPU_MEM=0.85` with `vllm-embed` up for
generate/ingest, and `0.94` with embed stopped for judging. These live in
`benchmark/docker/.env` (gitignored) rather than inline, so a watchdog
`--force-recreate` reproduces the same config.

### Postgres is tuned as disposable scratch

`synchronous_commit=off`, minimal WAL, `max_connections=200`,
`shared_buffers=768MB`, `effective_cache_size=4GB`. Per-daemon pool capped at min
2 / max 16 so N daemons never exceed `max_connections`.

### Hindsight gets a per-run database

`hindsight_<RUN_TAG>`, sanitized to `[a-z0-9_]`. The project chose this after
verifying live that an Arm-B daemon's startup consolidation sweep enqueued
operations onto a consolidation-off run's live bank when all runs shared one
database. `bank_id` isolation alone does not stop this: the sweep runs
per-database, not per-bank.

### Hindsight reranker moved to GPU (2026-07-22)

During the v3 Arm A full run the CPU TEI reranker (`hindsight-rerank`,
`text-embeddings-inference:cpu-1.8`) queued rerank requests for 5 to 10 s. The
project swapped the image to `120-1.9`, the first TEI tag with sm_120 (consumer
Blackwell) kernels, with `--dtype float16 --max-batch-tokens 8192` and a GPU
reservation, paid for by `vllm-embed` 0.12 to 0.09; vllm-gen stays at 0.85. The
`/rerank` + `/info` wire API is unchanged 1.8 to 1.9, so no Hindsight-side env
changed (`HINDSIGHT_API_RERANKER_PROVIDER=tei`), and fp16 shifts rerank scores
only at the numeric margin. Fallback if the experimental image misbehaves: a
`vllm-openai` instance serving the same cross-encoder (`--runner pooling`) behind
`HINDSIGHT_API_RERANKER_PROVIDER=cohere` +
`..._COHERE_BASE_URL=http://<svc>:<port>/v1/rerank` (vLLM's `/v1/rerank` is
Cohere-wire-compatible; the TEI provider is not). The judge profile
`VLLM_GEN_GPU_MEM=0.94` now requires both `vllm-embed` and `hindsight-rerank`
stopped, because 0.94 plus the reranker's ~0.5 to 0.8 GB does not fit.

### TEI admission permits raised to 4096 (2026-07-22)

This is the complement of the never-cap rule, not a reversal. TEI permits are
acquired per ITEM, so one 128-doc `/rerank` takes ~128 of the default 512. The v4
stack's speed aligned all 10 shards' rerank bursts and momentarily exceeded the
pool. `--max-concurrent-requests 4096` gives roughly 32 concurrent full batches,
about 8x the observed peak. Admission permits are a free semaphore; VRAM is still
bounded by `--max-batch-tokens 8192`. The project raised admission rather than
throttling `HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT`, because a client throttle
slows every recall to defend against a burst that free headroom absorbs outright.

### Clock-sync lifecycle barrier is scheduler-marker-driven, not a fixed grace

An external review (2026-07-27) found the lifecycle barrier released
`done_promotion_summary_skipped` after 15 s of no progress, a value measured on a
near-idle 1-persona server. Under a full wave a VALID summary call can outlast
it, so the barrier would declare a skip, answer without the summary, and
undercount the summary rate. The barrier now uses the scheduler's own per-pass
completion marker (`docs/TROUBLESHOOTING.md`): `summary=skipped` releases
immediately, `summary=<uuid>` waits for the row, and no marker means the pass is
still running so the barrier keeps waiting. Measured 150 s to 5.0 s, so it is a
signal rather than a timer. This fix is FEATURED-only
(`DISABLE_SCHEDULER=true` in every minimal preset), so it never touched the
minimal wave.

Two review findings were assessed and left in place, both unreachable from the
documented invocation. A `*_clocksync` preset with an explicit `BENCH_CLOCKSYNC=0`
would select range mode and reintroduce multi-persona clock rewinds, but nothing
instructs an operator to type that, and the launch command sets no
`BENCH_CLOCKSYNC` so per-persona inference fires. `PERSONA_CONTAINERS` is absent
from the manifest and contract hash, but the `_p<i>` versus `_s<k>` run-tag
suffix identifies the topology unambiguously, and `build_run_contract`
deliberately excludes geometry so every container of a wave agrees on one hash.

### Automatic prefix caching enabled on `vllm-gen` (2026-07-27)

vLLM defaulted prefix caching OFF because Qwen3.5-4B is a hybrid model
(`is_hybrid=True`, gated delta-net linear attention interleaved with full
attention) and vLLM keeps it opt-in for hybrids while the feature matures
(`arg_utils.py:2501`), so OFF was a model-class default, not a project choice.
User decision (2026-07-27): enable `--enable-prefix-caching` on the SHARED
`vllm-gen`, uniform across providers so the fairness line stays unmoved. The gain
is concentrated in the extraction path (a fixed template repeats on every retain,
Hindsight's ~2.7k-token consolidation template, mem0's extraction preamble); the
answer path is generation-bound and shares only ~300 of a ~2,957-token average
prompt.

Sequencing decides whether live shards survive the edit. Every provider service
declares `depends_on: vllm-gen: service_healthy`, so `docker compose run
<provider>` recreates `vllm-gen` on a config change, and editing mid-wave lands
that recreate on a live run (the 2026-07-21 failure that killed 22 of 30 shards).
Order: last run completes, edit compose, recreate from a QUIET server, verify
`enable_prefix_caching=True` in the banner and a nonzero
`vllm:prefix_cache_queries_total`, capture a fresh serving envelope, launch.

---

## Per-provider configuration

### Mnemosyne

- **`MNEMOSYNE_FACT_RECALL_ENABLED=0`** (plugin-default off). Fact rows retrieve
  worse: a v1 probe measured supporting evidence hit at 3 of 0.031 with top-5
  dominated by lossy fact strings.
- **`MNEMOSYNE_LLM_MAX_TOKENS` >= 2048** (3072 in practice). The same knob caps
  per-message extraction and sleep's whole-session model-refresh JSON. 512 suits
  the former and silently truncates the latter to zero proposals. ("sleep" is
  Mnemosyne's consolidation pass that rewrites working memory.)
- **`MNEMOSYNE_WM_TTL_HOURS` raised before ingesting backdated dialogue.** The
  168h default deletes it.
- **`--lifecycle` (retirement) exists to test retirement**, because the baseline
  adapter never triggers consolidation on its own. An SDK trace confirmed
  baseline is append-only plus a zero-LLM regex fact-versioning side table and
  conflict recording only.
- **Retirement is a net negative here by benchmark design, not a Mnemosyne bug.**
  MemConflict scores dynamic questions on retrieving both sides of a change.
  Retirement removes the old side, so supporting evidence hit falls and the model
  can no longer confirm a before/after change. It is also under-inclusive: it
  retires mostly assistant-side turns and leaves first-person stale assertions
  ("I live in Darwin") live.
- **The oracle arm is a deliberate upper bound, not a deployable config.** Here
  "oracle" means canonical slots built from gold annotations at build time,
  standing in for a diligent agent, with no gold read at recall time. Retrieval
  is still question-driven. It separates "can the canonical-store mechanism
  answer conflict questions" (yes) from "will an automated curator populate the
  slots well" (currently no).
- **The auto-sleep arm gets a per-session reflection budget.** This gates both
  the 10-exchange cadence and `on_session_end` at 3 calls/session using the
  plugin's own `MNEMOSYNE_REFLECT_MAX_CALLS_PER_SESSION`. Without it the
  plugin-faithful arm would fire roughly double the real plugin's consolidation
  churn, in the one arm whose entire claim is fidelity.
- **Embedding model `BAAI/bge-small-en-v1.5`**, the model behind Mnemosyne's own
  published LongMemEval number. This is Mnemosyne's best-effort choice, not one
  picked by this project.

### Hindsight

- **`HINDSIGHT_API_LLM_TEMPERATURE_RETAIN` 0.1 to 0.7.** 0.7 is a vendor-endorsed
  value: both the Qwen3.5 card's non-thinking value and what Hindsight's own
  consolidation path already inherits from the server default. That path showed 0
  runaways in 185 calls against retain's 2.76% at temp 0.1, same server, grammar,
  GPU, and run.

  | Metric | temp 0.1 | temp 0.7 + 4096 cap |
  |---|---:|---:|
  | Retain p50 | 3638 ms | 3636 ms |
  | Retain max | 322,750 ms | 32,501 ms |
  | Runaway rate | 2.76% | 1.29% |
  | Mean per retain | 7.42 s | 4.10 s (-45%) |
  | Total LLM time/session | 498 s | 395 s (-21%) |

  The unchanged p50 proves the fix touches only the pathological tail. Trade-off:
  consolidation got worse (mean 4.07 to 6.19 s/call, runaway 0.49% to 1.98%)
  because more facts extracted at 0.7 means more for consolidation to do. Net per
  session is still -21%, and drains stay inside the 450s ceiling. Open: whether
  more facts means better recall or more noise, which only scored results can
  say. Consolidation has no output cap, so its tail is unbounded; watch it in
  arms B/C.
- **`HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS` 16384 to 4096**, so a residual
  runaway truncates at 4096 tokens instead of 16384.
- **`HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES` 3 to 7**, a v1-era finding that still
  holds as the compose default.
- **Arms are a "feature set", never "untouched defaults".** The retain
  temperature override applies to every arm. Label them accordingly.
- **Arm C refuses to run with unfiltered recall.** `exchange_append` with
  `--recall_types` unset triggers `SystemExit(2)`, not a warning: the plugin
  defaults to `["observation"]`, so omitting it would silently measure the wrong
  recall surface while still being labeled Arm C. `RECALL_TYPES=all` is an
  explicit opt-out.
- **Empty Arm C recall emits a scoreable zero-retrieval row.** A zero-retrieval
  question still emits a scoreable row with `Retrieved_Memories: []`. A fallback
  would measure a configuration no real deployment runs.
- **`--strict_quality_run`** converts previously-tolerated silent degradations
  (drain timeout, poll failure, append-mode fallback) into hard shard failures.
- **The featured full run keeps `RECALL_TYPES=observation` (2026-08-02).** The
  featured arm is plugin-faithful by definition, and the plugin default is
  observation-only (user-confirmed assignment 2026-07-30: featured = observation,
  minimal = all, because with consolidation off no observation rows exist). The
  v5 gate smoke `ft27hs2` also ran observation. The persona-0 reading that favours
  `all` (`ftclk2r_p0` macro answer accuracy 0.348 against 0.307) is one persona of
  thirty and stays banked as a probe, not adopted. Launch tag `v5ftc086`.

### mem0

The pin is `mem0ai[nlp]==2.0.14` (2026-07-26). Read this section together with
"mem0 upgraded to 2.0.14" below; the ADD/UPDATE/DELETE/NONE decision that older
bullets discussed no longer exists in the product.

- **8-message batch cadence is the default arm**, matching the MemConflict
  authors' own mem0 runner (`external/MemConflict/Evaluation/eval_memzero.py`).
- **Deliberate divergence from that reference:** the authors drive the hosted
  platform (`MemoryClient`, `api.mem0.ai`, real `timestamp=` on add). This
  project selects locally-hostable providers, so the adapter runs the same
  cadence against the self-hosted `Memory` SDK with `metadata.timestamp`
  (self-hosted `Memory.add` has no timestamp parameter). The scorer is identical
  either way.
- **`infer=True` stays on** (turning it off stores raw turns). Under 2.0.14 the
  rationale is that `infer=False` discards fact extraction and stores raw
  dialogue, which is not what a Hermes deployment running mem0 would surface.
- **The embedder is the shared `vllm-embed`**, matching Mnemosyne and Hindsight's
  retrieval-embedding surface. The serving contract sets the model (v4
  bge-small-en-v1.5 384d, v5 gte-modernbert-base 768d).
- **Sharded runs need a central qdrant**, mirroring Hindsight's move to shared
  Postgres. Isolation is per-run collection, and personas are isolated by
  `user_id`.
- **Ingest cost is the product's design, not a config defect; the project applies
  no tuning** (2026-07-22, re-measured 2026-07-28 under 2.0.14). Under 2.0.14 the
  `infer=True` path makes ONE extraction call per `add()` (the older two-call
  ADD/UPDATE decision is gone), so per-session cost is roughly a quarter of
  0.1.118: the v4minc 30-persona run measured ~33 s/session at 6-way concurrency,
  24,263 `add()` calls yielding 112,320 facts (4.63 facts/call), 6h20m wall-clock
  for the whole wave. Thinking is off on internal calls (no `chat_template_kwargs`
  sent, so the server default applies). JSON parse failures skip and never retry.
  Rejected knobs: prompt trimming (changes the extraction behavior under test for
  negligible decode savings), and `AsyncMemory` (its only remaining objection is
  that it changes the ingestion cadence under test, since batches no longer carry
  a decision-ordering data dependency). This is intrinsic product cost, the same
  kind of finding as the RetainDB-local ruling, but nowhere near disqualifying.

### Supermemory

- **The search threshold is sent as explicit `0.0`**, not omitted. Omitting it
  engages the vendor's 0.6 cutoff, which would hand Supermemory fewer memories
  than the shared top-K every other provider gets, a harness asymmetry rather
  than a quality signal. `SUPERMEMORY_SEARCH_THRESHOLD=0.6` exists as an explicit
  arm to reproduce the vendor default.
- **`hybrid` search mode is the headline arm** because it is the Hermes plugin's
  own default, not because it scores better.
- **The harness drains async ingest before answering**, the same quality-arm
  ruling applied to every provider's async path.
- **The two LLM roles are kept apart.** Answer+judge is harness-locked
  (`OPENAI_*`), and internal extraction is `SUPERMEMORY_LLM_*`, mapped onto the
  spawned server's own subprocess env.
- **Server pinned to 0.0.5, not latest (2026-07-22).** 0.0.6 (current stable) and
  0.0.7-rc.2 both ship a linux-x64 bundle missing `@rivetkit/rivetkit-wasm`, so
  the workflow engine behind async ingest never starts and every document sits at
  `queued` until the 600s drain timeout. 0.0.5 processes a document to `done` in
  about 5s. The evidence is also in the `Dockerfile.supermemory` comment. Bump
  the version only after verifying a document reaches `done` on the candidate tag.

### Honcho

Vocabulary: the **dialectic** is Honcho's internal question-answering call
(`POST /peers/{id}/chat`); a **dream** is the consolidation job that rewrites
observations into conclusions; the **deriver** is the extraction worker; an
**observation** is one extracted memory row, and a **conclusion** is a derived
fact ranked by query.

- **Hybrid recall is the headline arm; conclusions top-5 is the minimal arm.**
  Hybrid is the plugin's own `recallMode` default, and it is section-structured
  with no per-item score or timestamp, so `plugin_native_recall=True` hands the
  answer model every section in plugin order, per ruling 3. Slicing Honcho's peer
  model down to a synthetic top-K would reshape the plugin's real output into a
  configuration no Hermes deployment runs. `conclusions` is the closest Honcho
  analogue of a mem0 or Mnemosyne memory row (ranked, real `created_at`), so it
  is the arm that takes the shared top-K=5.
- **Per-session queue drain, plus `DERIVER_FLUSH_ENABLED=true`.** The plugin's
  `sync_turn` is fire-and-forget. Answering immediately after ingest would measure
  the deriver queue, not the memory, so the adapter polls `queue_status()` until
  `pending_work_units==0` and `in_progress_work_units==0`, the same
  inter-session-idle argument as the Supermemory drain. `DERIVER_FLUSH_ENABLED`
  bypasses the `DERIVER_REPRESENTATION_BATCH_MAX_TOKENS` threshold (default 1024),
  so a tail batch under threshold does not stall the drain.
- **Featured arm triggers a manual dream after each session** (user ruling
  2026-07-31). `HONCHO_DREAM_AFTER_SESSION=1` calls `schedule_dream` (POST
  `/v3/workspaces/{id}/schedule_dream`, `dream_type=omni`, bypassing the document
  threshold, idle timer, and 8h spacing) for each observer-to-observed pair
  hybrid recall reads, then drains again. Honcho's own scheduler dreams after 60
  minutes idle, with 8h between dreams per pair. MemConflict sessions are days
  apart in logical time, so an untouched deployment's idle scheduler would dream
  between almost every session pair, and a benchmark run that never idles would
  leave that feature unmeasured. Label: "shipped consolidation, manually
  cadenced." If dreaming lowers macro answer accuracy, that is a finding to
  report (the v1 Mnemosyne precedent that every automated consolidation feature
  lowered its score), not a configuration to route around.
- **Ingestion sends no `created_at` by default; clock-sync is the temporal
  path.** The plugin's `sync_turn` stamps a local timestamp but never sends it to
  the server (`session.py:45-54`), so the plugin-faithful default is
  `HONCHO_SEND_CREATED_AT=0`, and `BENCH_CLOCKSYNC` moves the server's clock
  instead. `HONCHO_SEND_CREATED_AT=1` is a vendor-exposed deviation arm (the SDK
  supports `created_at` end-to-end), kept off by default because the plugin does
  not use it.
- **The plugin's context budget runs ON at 8192 tokens
  (`HONCHO_CONTEXT_TOKENS`), and the plugin ships it OFF.** Ruling 2:
  `contextTokens` is vendor-exposed and vendor-documented
  (`plugins/memory/honcho/README.md:53,296`), but `_parse_context_tokens`
  (`client.py:145-153`) returns None when unset, so the shipped default is
  uncapped. Uncapped cannot run on a 32768-token window (the shared answer budget
  takes 16384), while a measured featured hybrid block reached 254k tokens at
  persona 0 session 5 and the minimal conclusions arm hit 32,890 prompt tokens on
  a top-5 slice. 8192 is half the remaining prompt budget.
  `truncate_items_to_budget` in `eval_honcho.py` ports the plugin's
  `_truncate_to_budget` (`__init__.py:870-883`): `context_tokens * 4` chars, a
  word-boundary cut kept only past 80% of the budget, `" ..."` appended, applied
  to the final joined block after the top-K slice. Caveat to publish with any
  Honcho number: this bound is the plugin's code at a value the plugin does not
  ship. `HONCHO_CONTEXT_TOKENS=0` reproduces the uncapped path; the summary records
  `Context_Truncated_Questions` and `Context_Items_Dropped`.
- **Deriver output capped at 2048 tokens, deriver-only, plus the card's
  `presence_penalty` 1.5.** Ruling 2, and the trigger is measured: 18 of 79
  documents in smoke `hn_smkft_p0` sat at the 8192-token cap, mean 41,189 chars,
  unique-sentence ratio 0.181. Real observations have a median length of 241
  chars, so 2048 cannot truncate a healthy one and it bounds a residual runaway to
  a quarter of the old damage (the Hindsight retain cap of 4096 is the precedent).
  `HONCHO_LLM_MAX_OUTPUT_TOKENS` stays at 8192 because it feeds EVERY internal
  role; the new `HONCHO_DERIVER_MAX_OUTPUT_TOKENS` and
  `HONCHO_DERIVER_PRESENCE_PENALTY` overlay `DERIVER_MODEL_CONFIG` alone, so the
  dialectic and dream budgets do not move. `presence_penalty` is a real
  `ModelConfig` field (`src/config.py:253`) reaching the request body; 1.5 is the
  Qwen card value `answer_env.sh` already uses for the answer role, and vLLM
  cannot serve it as a default because `get_diff_sampling_param` does not
  allowlist it.
- **`DIALECTIC_MAX_INPUT_TOKENS` lowered from 100,000 to 20,000.** Ruling 2, and
  forced by the window rather than chosen: the vendor default assumes a
  128k-window model, vllm-gen serves 32,768 under v4, and Honcho's own handling of
  an over-long dialectic prompt is to TRUNCATE it (`truncate_messages_to_fit`,
  `src/llm/api.py:333-341`), a path that only runs when the bound sits below the
  window. Left at the default, every dialectic call from persona 0 session 5
  onward returned a 400 on smoke `hn_smkft2_p0`, emptying layer 2 of the hybrid
  headline arm. It does NOT stop the 400s: measured 2026-08-01, it removes 0
  tokens on 209 of 210 truncation events, because the overflow happens at
  tool-loop iteration 1 where there is nothing droppable. It is kept because it
  bounds later iterations at no cost.
- **The dialectic and dream overflows are reported, not configured away**
  (settled 2026-08-01). On the featured contract-v4 single-persona smoke `smkft3`
  the dialectic 500s on 61 of 122 questions, dream deduction fails 97 of 104 and
  induction 71 of 102, all from a vLLM 400 context overflow (mechanisms in
  TROUBLESHOOTING "Provider: Honcho"). On a 32,768-token window no vendor knob
  keeps the `low`-or-higher dialectic or either dream specialist inside the
  window: `DIALECTIC_MAX_INPUT_TOKENS` cannot reach the iteration-1 overflow (the
  conversation is exactly `[system, user]` and the last unit is never dropped, 209
  of 210 events removed 0 tokens), lowering dialectic output raises the input
  threshold to 30,721 against observed ~36,800-token inputs,
  `DIALECTIC_SESSION_HISTORY_MAX_TOKENS` saves at most 4,096 against a 20,739 to
  27,197-token prefetch, `reasoning_level=minimal` is a weaker product mode not the
  plugin's `low` default, the dialectic prefetch size is hardcoded 25
  (`src/dialectic/core.py:170`), `DREAM_*_MAX_INPUT_TOKENS` does not exist,
  `DREAM_MAX_TOOL_ITERATIONS` is never read (specialists hardcode 12 and 10), and
  `DREAM_HISTORY_TOKEN_LIMIT` bounds a method in neither specialist's tool set
  while one maximal `search_memory` result alone exceeds the 24,576-token input
  budget. On v4 serving the featured arm's dialectic and dream failure rates are a
  PRODUCT PROPERTY reported under ruling 3, with about half the dialectic layer
  dead and dream consolidation mostly failing; publish the two rates with any
  featured Honcho number. The shared upstream cause is the deriver's residual
  repetition loops (5.2% of documents pinned at the 2048-token cap in `smkft3`).
  Contract v5's 131072 window is the shared-harness change that lifts this, taken
  because the featured contract is its own serving contract.
- **Workspace per persona.** Honcho scopes conclusions and dreams to an
  observer-to-observed peer pair inside one workspace. A shared workspace would
  let one persona's conclusions leak into another's recall, so isolation is one
  Honcho WORKSPACE per persona (`hermes_<tag>_p<idx>_<sanitized persona id>`), the
  same boundary Hindsight gets from a per-run database and mem0 from a per-run
  collection.

### OpenViking adapter (2026-08-03)

The adapter, the `openviking` compose service, and the two presets were added on
2026-08-03. Every choice below is fixed by this entry.

- **The adapter speaks raw `httpx` and never imports the `openviking` pip
  package.** The Hermes plugin
  (`external/hermes-agent/plugins/memory/openviking/__init__.py`, pinned SHA
  `6d17b2a5`) has no SDK client either, so an SDK path would measure a client the
  deployment does not use. And the provider folder `openviking/` shadows the
  installed package name, so an import from the repo root would resolve to the
  adapter's own folder. The repo root never goes on `sys.path`.
- **Recall timeouts are 60 s and 30 s, against the plugin's 4.0 and 3.0.** The
  plugin runs `prefetch()` on a background thread and joins it for at most 8 s, so
  a slow call costs that turn its memory block and nothing else. The adapter calls
  recall inline in the answer path, where the same budget returns empty recall
  with no error. 60 is the plugin's own clamp maximum (ruling 2). The precedent is
  `HONCHO_TIMEOUT=300` against that plugin's own 30 s.
- **In `prefetch` mode the session-start block is rebuilt for every question**,
  not latched once per session. The plugin caches the block per session id because
  a session is one continuous conversation; every benchmark question is answered
  as an independent turn, so a once-per-session latch would put the block in front
  of the first question only. The block is built once per session and prepended as
  item 0 with `score: None` and `source: "session_start"`, the honcho hybrid arm's
  precedent.
- **The adapter drains, the plugin does not.** The plugin POSTs
  `/api/v1/sessions/{sid}/commit` at session end and moves on. The adapter polls
  `GET /api/v1/tasks/{task_id}`, then calls `POST /api/v1/system/wait`, and raises
  on a failed or cancelled task, on the drain timeout
  (`OPENVIKING_DRAIN_TIMEOUT_S`, 1800 s), and on any `error_count > 0`. The
  `error_count` is the only place a broken embedder appears; otherwise the HTTP
  path stays 200 and the run exits 0 with empty recall.
- **`created_at` is off by default.** The plugin sends no `created_at` on any
  message, so the plugin-faithful temporal path is `BENCH_CLOCKSYNC` moving the
  spawned server's clock. `OPENVIKING_SEND_CREATED_AT=1` is a vendor-exposed
  deviation arm: the extraction prompt anchors relative dates on the first
  message's `created_at`, and event memories are filed under
  `events/{year}/{month}/{day}`.
- **One commit per session maps the plugin's `on_session_end`.** The plugin drains
  its own writers and commits with `keep_recent_count: 0` when a session ends; the
  benchmark's session boundary is the same. Ingest cadence stays the plugin's
  per-exchange `messages/batch` POST (`RETAIN_GRANULARITY=exchange`).
- **Dev auth mode plus a per-persona `X-OpenViking-User` header is the isolation
  boundary.** On loopback the server's `auth_mode: "dev"` takes identity from
  headers with no key, and the `user` value alone scopes every read and write.
  Each persona's user id is `<OPENVIKING_USER_PREFIX><persona tag>`, and
  `begin_persona` wipes it with `DELETE /api/v1/fs?uri=viking://user/memories&
  recursive=true` so a re-run is idempotent.
- **Every shard spawns its own server.** Storage is a local workspace directory
  holding a one-process `.openviking.pid` lock, so there is no shared backing
  service. `entrypoint.openviking.sh` exits 2 on `OPENVIKING_SERVER_MODE=shared`
  under `BENCH_CLOCKSYNC=1`: one attached server has one perceived clock and
  cannot sit at N shards' logical session dates at once.
- **`vlm.max_concurrent` is 8, against the vendor default 64.** Extraction and
  search-intent analysis run on the same `vllm-gen` that serves the shared answer
  role, so 64 in-flight internal calls per shard would take that server's
  scheduling from the fairness-locked answer path.
- **`ov.conf` omits `rerank` entirely and sets `query_planner: null`.** The config
  model is `extra: "forbid"`, so omission is how reranking is turned off (the
  `rerank` section has no enabled flag). `query_planner: null` falls back to
  `vlm`, one model for both internal roles per the best-effort ruling.

### OpenViking arm rulings (user, 2026-08-04)

- **The minimal arm is `find`, and the minimal run is the DIAGNOSTIC.** Only the
  featured `prefetch` arm (the plugin's full read surface, querying
  `search/search`) is scored and enters the final comparison. The `find` minimal
  run is the integration proof and deterministic retrieval floor: `POST
  /api/v1/search/find` with no LLM in the path. `search/search`'s intent analysis
  emitted zero queries on 2 of 4 smoke questions and emptied their recall
  (`docs/TROUBLESHOOTING.md`); that behavior is part of the plugin's read surface
  and is measured in the featured arm.
- **Every arm passes the plugin's selection width to the answer model.** The
  answer engine gets what the plugin gives Hermes: the `recall_limit` (6) selected
  items, whole, `plugin_native_recall=True` in all three modes with no harness
  top-K slice. The scorer is unchanged because upstream
  `extract_top_k_retrieved_memories` slices `Retrieved_Memories[:K]` at its own
  white-box K (`eval_scoring.py:329-344`).
- **The recall score floor stays at the plugin default 0.15.** Checked against
  vendor and community usage 2026-08-04: 0.15 is the shipped default across
  OpenViking's agent integrations, and the one deviating integration raises it to
  0.35. No source lowers it. The Supermemory situation (a server-side 0.6 starving
  the shared top-K) does not apply, because 0.15 is plugin client code.
- **`send_created_at` stays an unused flag.** It stays implemented and defaults 0,
  but the comparison picks a Hermes plugin, so only plugin behavior is measured:
  the plugin sends no timestamps and `BENCH_CLOCKSYNC` is the temporal path.

### RetainDB server edition

- **384-dim bge-small zero-padded to the schema's `vector(1024)`** rather than
  switching embedding models. Zero-padding preserves norm and dot product
  (‖[v,0]‖=‖v‖, ⟨[a,0],[b,0]⟩=⟨a,b⟩), so cosine ranking is bit-for-bit identical.
  This is schema compliance, not a model change. `RETAINDB_EMBEDDING_MODE=local`
  (Xenova/bge-large, 1024-dim) is a genuinely different embedder and is
  off-contract; it exists only to prove wiring without a GPU. Under v5 the pad
  shim pads 768 to 1024.
- **`write_mode:"sync"` on ingest** so a run never recalls against a
  half-ingested session.
- **`DISABLE_SCHEDULER=true`** for the minimal arm, turning off the 60s
  `runSessionLifecycle()` job that mutates the memory set. Scheduler-on is the
  featured arm, built and verified under the faked clock (see "RetainDB
  clock-sync" below and the `promotion_mode` fidelity entry).
- **`RETAINDB_DISABLE_SEARCH_CACHE=true` disables two different things.** The
  semantic cache is a genuine upstream cross-tenant bug, so disabling it is
  unconditionally justified. The exact-key cache's staleness only bites because
  the benchmark compresses logical months into wall-clock seconds inside a 300s
  TTL; a production deployment would rarely re-ask within 300s. That half
  compensates for the benchmark's time compression, not an upstream bug.
- **The project dropped ivfflat indexes post-migration** rather than tuning
  `probes` per-session (no such hook exists). This is correct at per-run-DB scale
  where exact KNN is sub-ms. It is explicitly not the production answer.
- **Vendor patches, never submodule edits.** `external/RetainDB` stays read-only.
  Fixes live in `retaindb_server/server_patches/` and apply at build time.

### RetainDB local edition, ruled out

The project ruled this out 2026-07-22 for O(n²) search. See
[TROUBLESHOOTING](TROUBLESHOOTING.md#provider-retaindb-local-edition-why-it-was-ruled-out).
The adapter is correct and kept for the record. The blocker is the product.

Two decisions from before the ruling are worth keeping:

- **The project investigated and rejected routing its embeddings through the
  shared `vllm-embed`:** `@retaindb/local@0.2.1`'s `embedText()` has no HTTP path
  at all (it dispatches only to in-process Xenova or a hash fallback), so
  offloading would require patching vendor code. This left it the one provider on
  a different embedding model, which the project disclosed.
- **The plugin overlay's "profile" half was left out:** it has no per-item
  score to populate the support-rank field, and it is query-independent (literally
  "last 5 ingested", identical for every question), so reproducing it would need
  10 slots against everyone else's 5. The project discloses this as a
  one-directional bias that could only have helped RetainDB, so its number would
  have been a floor.

---

## Scoring

### A second judge model, gemma-4-12b (2026-07-29, user request)

CLAUDE.md says the answer model and judge model are the same for a given
comparison. This arm deliberately breaks that: the answer model stays
`qwen3.5-4b`, the judge becomes `unsloth/gemma-4-12b-it-NVFP4`. Compose service
`vllm-judge` under profile `judge` on port 8002, so a bare `up -d` cannot start
it beside the Qwen servers. Score artifacts carry the suffix `_gj12`.

Compare a `_gj12` number only against another `_gj12` number. Supporting evidence
hit, EUG, and log-rank are the judge's own semantic support measures, so a judge
swap moves them with answer accuracy. The `_gj12` wave is a fresh five-provider
baseline, not a re-score.

Serving settings that were measured rather than guessed, full evidence in the
`vllm-judge` comments in `docker-compose.yml`:

| setting | outcome |
|---|---|
| `--attention-config '{"use_prefill_decode_attention": true}'` | 3.03x steady-state, verdicts unchanged |
| `SCORE_WORKERS=32` | 1.27x over 16; the judge is KV-bound at ~24 concurrent |
| `--max-num-batched-tokens 8192` | REJECTED, costs 40% of the KV cache |
| `--kv-cache-memory 4484869632` | +4.6% KV (82,513 tokens) |
| `--skip-mm-profiling` | memory no-op, kept only for boot time |

The 3.03x was verified not to change verdicts before adoption. The judge samples
at temperature 1.0, so two runs of the same kernel disagree from sampling alone.
That noise floor was measured first (unified@16w against unified@32w: 95.9%
answer-accuracy agreement, 90.2% rank), then compared against the kernel change
(96.7 to 97.5% answer accuracy, 87.7 to 90.2% rank). The kernel change is inside
the noise floor.

Settled, do not retry: FlashAttention and FlashInfer cannot serve Gemma 4 on this
SM120 card, per-layer mixed routing is upstream-rejected (vllm#48114), and MTP is
impossible because the checkpoint has 1,389 tensors and zero MTP weights.

### Scoring runs from result files, not from provider entrypoints

`benchmark/score_files.sh` takes result JSONL paths plus judge server details and
nothing else. It sources `answer_env.sh` and calls `run_score` / `run_summarize`
directly, on the host `.venv` against the judge's published port.

Judging is already provider-agnostic: `run_score` takes only a provider directory
and a tag, and every tool it calls lives in `benchmark/`. The per-provider Docker
entrypoints exist for the GENERATE stage, where providers need different
infrastructure. Routing a score through them makes the judge inherit that
infrastructure, which is how the hindsight leg of the `_gj12` wave failed (see
TROUBLESHOOTING "Scoring dies on a database the scorer never uses").
`benchmark/docker/score_with_judge.sh`, the compose driver that produced four of
the five `_gj12` results, is superseded by `score_files.sh`, because keeping two
scoring paths is what let the judge sampling diverge between them.

The script sources the judge env rather than re-declaring it, so the fairness
contract stays in ONE place. A second copy of those ten variables would drift and
silently change what the numbers mean.

**`--temperature`, `--top_p`, and `--top_k` are required, with no defaults.**
`bench_judge_env` defaults to temperature 0.6 / top_k 20, the qwen3.5-4b
contract; a gemma-4-12b judge runs 1.0 / 0.95 / 64. Inheriting the default
silently judges one provider under different sampling than the rest of its arm.
That happened on 2026-07-30: hindsight ran 61 questions at 0.6 against the wave's
1.0. The checkpoint was deleted and the guard added, so the script now exits 2
instead of falling back.

### The judge needs forced JSON mode

Without `MEMCONFLICT_JSON_MODE=1`, models burn their whole token budget on
reasoning and return empty content: about 6% of judge calls fell back to
rule-based scoring, each burning 5 retries. With it: 99.3% LLM-judged, 0
fallbacks. A large `OPENAI_MAX_TOKENS` is free here because in JSON mode the model
stops at the closing brace.

### Scoring is checkpoint-resumable

`benchmark/score_resumable.py` writes every verdict to a checkpoint as it lands.
The upstream `eval_scoring.py` path holds all results in memory and writes only on
completion, so a container restart would wipe hours of judging (this happened
twice). It uses the same aggregation logic, checkpointed. Resume, never restart.
To get more judge throughput safely, run a second sequential pass rather than
raising concurrency.

### Answers come from top-5, not top-3

Upstream MemConflict adapters default to top-3. This harness stores 5 for
white-box scoring and answers from 5. Supporting evidence hit at 3 is unaffected
(measured at 3 regardless), but answer accuracy is not strictly comparable to
published top-3 numbers. `--top_k 3` gives the strictly comparable configuration.

### Two different things are called EUG

- **`EUG_gap@3`** is upstream's Evidence Utilization Gap: supporting evidence hit
  at 3 minus answer accuracy per conflict type with an unweighted 3-type mean
  (`external/MemConflict/Evaluation/diagnose_failures.py:73-96`). Quote this when
  citing MemConflict's EUG.
- **`EUG-cond@5`** is a repo-local conditional utilization rate: mean answer
  accuracy over questions whose evidence reached top-5. It is a mean, so a 0.5
  partial-credit answer contributes 0.5, not a "fraction answered correctly".

On the v1 Mnemosyne baseline they are +0.045 and 0.694. Same name, entirely
different quantity.

### Supporting evidence hit is judge-assessed, not literal matching

The judge is asked for the 1-based rank of the first retrieved memory containing
evidence that semantically supports the reference answer. This has two
consequences: storage format is not a confound, and supporting evidence hit, EUG,
and log-rank inherit judge variance. They are not judge-independent retrieval
measures, so a judge-config change moves them too.

### The judge's evidence render stays at top-5 (2026-08-02, user ruling)

The judge renders only the first 5 retrieved memories (`MAX_WHITE_BOX_TOP_K = 5`,
`external/MemConflict/Evaluation/eval_scoring.py:39`). Four featured arms hand the
answer model more than that. The value stays at 5, and the ruling is how to read
the metrics rather than a change to the harness: answer accuracy is the headline,
and supporting evidence hit at K, support rank, log-rank, and the evidence
utilization gap are diagnostics of the top-5 slice on any arm whose answer surface
exceeds it.

Measured answer surface per featured arm:

| arm | what the answer model receives | what the judge renders |
|---|---|---|
| Hindsight | plugin-native recall, median 65 items (range 15 to 144) | first 5 |
| Honcho | hybrid recall, 4 to 6 named markdown sections, about 25,000 tokens median render | first 5 |
| mem0 | `TOP_K=10` | 5 of the 10 |
| Supermemory | `TOP_K=10`, plus a profile block of about 1,340 characters | 5 of the 10; the profile block never reaches the judge |

Honcho's two representation sections carry about 95% of that render; its dialectic
section is the smallest and comes last.

Answer accuracy is unaffected on all four, because the answer model reads the
whole surface before the judge scores the answer. All banked v4 minimal numbers
are unaffected, verified against the stored rows: they ran `TOP_K=5`, stored at
most 5 memories per question, and carry no profile block. The reader-facing
version of this caveat is in `docs/BENCHMARK_MATRIX.md`, "Reading these numbers".

### Penalty judge arm: a wrong answer scores -1 (2026-07-31)

The standard rubric scores a wrong answer and an abstention identically at 0.0, so
answer accuracy cannot separate a provider that states a stale fact from one that
declines to answer. This arm scores a wrong or contradictory answer -1 and keeps
0.0 for a missing or uncertain one. Score artifacts carry the suffix `_gj12pen`.
Measured numbers are in `docs/BENCHMARK_MATRIX.md`, "Penalty judge arm".

A penalty number is never comparable to a standard answer accuracy number. -1 is
outside upstream's metric range. `EUG_gap@3` computed under this rubric is
likewise not comparable to upstream's.

**The generator is the committed artifact; the patched copy is derived.**
`benchmark/make_penalty_judge_evaldir.py` copies
`external/MemConflict/Evaluation/*.py` into the gitignored
`benchmark/penalty_judge_eval/` and applies exact string replacements to the copy.
Each replacement must match exactly once or the script exits 1 rather than write a
half-patched copy. `--verify` diffs the copy against upstream. `external/` is
never modified, per the pinned-submodule rule.

```bash
python benchmark/make_penalty_judge_evaldir.py
MEMCONFLICT_EVAL_DIR=benchmark/penalty_judge_eval \
  benchmark/score_files.sh --temperature 1.0 --top_p 0.95 --top_k 64 \
    --suffix gj12pen <results.jsonl>
```

`benchmark/score_resumable.py` already reads `MEMCONFLICT_EVAL_DIR`, so no wrapper
change was needed.

**Two silent parsers had to be patched in the copy; either one alone voids the
arm.**

- `parse_trinary_score_value` floors every value below 0.25 to 0.0, so a
  judge-returned -1 would arrive as an abstention and the arm would measure
  nothing. A `numeric <= -0.5` branch runs before the existing thresholds, so the
  1.0 / 0.5 / 0.0 boundaries stay byte-identical.
- `conditional_answer_accuracy` is not in `PARTIAL_CREDIT_BLACK_BOX_METRICS`, so
  it routes through `parse_binary_value`, where `numeric != 0` returns 1 and a -1
  becomes full credit, the exact opposite of intent. A third dispatch branch sends
  only that metric through a penalty-aware parser. `parse_binary_value` itself is
  unchanged, so update order recognition and contradiction recognition keep
  upstream's binarization.

**Three limits on reading the other metrics as upstream-comparable** (added
2026-08-05 after review).

- **The prompt is not byte-identical, only the definitions and schemas are.**
  Every metric for a question comes from ONE judge call, and the patch edits three
  sentences inside that same prompt (`eval_scoring.py:456,478,500`), while support
  rank and the diagnostics are requested in the same template at 458/480/501.
  Conditioning drift on support rank is plausible and unmeasured. It can be
  bounded for free by comparing supporting evidence hit at 3 between the existing
  v4minc `_gj12` and `_gj12pen` score files.
- **The evidence utilization gap is inflated.** `summarize_scores.py` labels
  `EUG_gap@3` as upstream's supporting-evidence-hit-at-3 minus answer accuracy,
  but subtracting a penalty answer accuracy is not the upstream quantity:
  mnemosyne static reads 0.331 minus (-0.204) = 0.535. Caveat any gj12pen gap.
- **A judge call that exhausts its retries silently leaves the penalty rubric.**
  `evaluate_question_with_llm` swallows the exception and the row falls back to
  `build_rule_based_result` (`eval_scoring.py:660-662,701`), which the generator
  deliberately does not patch, so that row can never score -1.
  `score_resumable.py:350-355` caches it with a valid fingerprint, so a resume
  never retries it. The only symptom is `rule_based` in `judge_methods`. Gate
  every banked wave on that count being zero; if it is not, delete those qkeys
  from the checkpoint and re-run.

**Two result builders stay at 0.0 on purpose.** `build_rule_based_result` and
`build_missing_answer_result` still score 0.0, never -1. A judge that never ran
cannot assert that an answer is wrong. Read `Judge_Method_Statistics`: any count
outside `llm_judge` is a question the penalty rubric never reached.

**The audit that validates the arm.** A stratified sample of 26 -1 marks, seed 0,
over Hindsight and Mnemosyne and all three conflict types, cross-checked against
the standard arm over all 3,750 questions per provider:

- 24 of 26 sampled -1 marks are genuine wrong answers; zero correct answers were
  marked -1. A census of the highest-risk shape (a bare "Yes." or "No." gold
  answer against a model answer containing an abstention phrase) found 15
  mislabelled abstentions out of 891 total -1 marks, 1.7%, about 2.0% with
  arguable cases.
- The error runs the other way and is larger. Of 16 answers scored 0.0 that were
  read, 4 to 5 assert a fact contradicting the gold and match the construction of
  -1 answers elsewhere. Scaled, roughly 125 answers repo-wide arguably deserved -1
  and did not get it (an estimate from 16 reads), so the -1 counts are a LOWER
  BOUND on wrong assertions.
- Judge nondeterminism at temperature 1.0: standard scores of 0.5 or 1.0 that
  flipped to penalty -1 number 34 for Hindsight (0.91%) and 8 for Mnemosyne
  (0.21%), against baseline off-diagonal movement of 5.8% and 6.8%, so the flip
  rate sits below ordinary judge variance. Conditional-type -1 marks are the
  weakest category (23 of 891, arguable); the dynamic and static -1 marks (869 of
  891) held up.
- **One proxy was tested and discarded.** Counting 0.0 rows whose `Reasoning`
  contains "contradict" gives 167 Hindsight and 127 Mnemosyne, but 8 of 10 sampled
  are abstentions correctly scored 0.0, where the judge used "contradicts the
  reference answer" as generic prose for any mismatch. Do not cite that count as
  an under-marking figure.

**Noise floor for reading small differences.** Supporting evidence hit at 3 and
contradiction recognition move between the two arms even though their rubric lines
are unpatched: Hindsight static supporting evidence hit at 3 reads 0.6639 against
0.6750, and contradiction recognition 0.5167 against 0.5278. That is judge
sampling variance at temperature 1.0, and it bounds how small a difference either
arm can resolve.

### Every run writes a manifest

`benchmark/write_manifest.py` records repo and submodule SHAs, dataset path and
count, full post-defaulting env, arm flags, top-K, serving envelope, and failure
counters. Its `canonical_config` block does a live `os.environ` read per stage,
not a hardcoded mirror; it used to be hardcoded, which made every v2 manifest
misdescribe its own run. Unset vars record as `null` (visible) rather than a wrong
default (invisible).

---

## Harness guards kept during the simplification pass

The 2026-07-22 DRY/KISS/YAGNI pass over the shared harness cut ~350 lines with no
behavior change (one sharded launcher `run_shards.sh`, entrypoint dedup into
`answer_env.sh`, deleted Hindsight embedded/local modes, adapter dead-code
removal). The still-live guards that pass explicitly kept:

- **The judge checkpoint's per-question input fingerprint**, which protects a
  resume from re-judging under a changed prompt.
- **`preflight_rows.py`**, which guards unattended multi-hour judge launches
  against silently truncated generate output (the `MAX_SESSIONS=6` incident
  class). If the row gate ever produces a false positive, downgrade it to
  warn-and-proceed rather than trimming checks.
- **Run manifests.** None of this is on the new-provider integration path, so
  deleting it buys nothing.

RetainDB local's Docker surface (compose service, Dockerfile, entrypoint) stays
untouched: retiring it would be effort spent on a product already ruled out.

---

## Featured and minimal run policy (locked 2026-07-22)

The user locked this 2026-07-22, reviewed against `external/hermes-agent` @
`6d17b2a`, `external/mnemosyne` v3.14.0, and adapters + compose at `496f49a`. It
supersedes any conflicting planned-run entries in `BENCHMARK_MATRIX.md`. The
per-provider featured/minimal knobs live in the Per-provider configuration section
above and in the arm-specific entries below; this section states only the policy.

- **Every provider gets two full-30-persona runs, both scored:** minimal (the
  simplest verified adapter path, shared top-K=5, the integration baseline) and
  featured (plugin-faithful: real Hermes plugin cadence, endpoints, and
  consolidation defaults).
- **Featured retrieval amends the top-5 fairness line, it does not break it.**
  Minimal runs keep the shared top-K=5. Featured runs inject plugin-native K per
  provider (mem0 10, Supermemory `max_recall_results` 10, Mnemosyne prefetch top-5
  with no backfill, RetainDB overlay top-5, Hindsight budget-based recall with no
  fixed K). Supporting evidence hit at 3 and 5 are still computed from ranks
  against whatever each provider returns. The featured fairness line is the
  policy, "the answer model sees exactly what the plugin injects", applied evenly,
  not one shared numeric K.
- **Timestamps use logical dataset time everywhere, not wall clock**, for both
  featured and minimal runs, for every provider.
- **Consolidation defaults ON in featured runs even though v1 evidence says it
  lowers scores** (Mnemosyne lifecycle 0.418 < baseline 0.544). A featured run
  reproduces what an untouched Hermes install ships. A lower score from
  consolidation is a finding about the provider, not a config error to route
  around.
- **Open, and not committed.** Whether a minimal tool-calling agent loop (the LLM may
  choose to call a provider's native tool rather than only receiving automatic
  prefetch injection) is worth building. No run is scheduled.

---

## Arm decisions that still apply under v5

### Clock-sync mechanism via libfaketime

Provider code cannot be timeshifted in-process (Mnemosyne has 138
`datetime.now()` sites, no clean patch seam), so the clock-sync arms take the
OS-level route: selected generate-stage processes run under `libfaketime`
(`LD_PRELOAD` + `FAKETIME_TIMESTAMP_FILE`) so their perceived clock tracks the
dataset's logical session date. This is an additional featured arm per provider
(`BENCH_CLOCKSYNC=1`); the minimal arms stay byte-identical when off, and it
applies to generate only (score, summarize, manifests stay on the real clock).
`benchmark/clock_sync.py` is the single writer of the timestamp file (`set_clock`
per session, before ingest, session date at 12:00 UTC so perceived "now" stays
ahead of provider row timestamps; it raises on an unparseable date rather than
silently keeping the prior clock). `benchmark/docker/clock_sync.sh` owns the env
side. Which process gets `LD_PRELOAD` is the only per-provider difference
(Supermemory: the spawned Bun child; Mnemosyne: the in-process generate python;
RetainDB and Hindsight pg0: node/postgres inside the faked domain).

Per-shard storage is a forcing function. A faked clock is a per-shard timeline and
a shared central store has one clock, so a store any shard must clock-fake cannot
be shared. That is why the Supermemory clocksync arm requires spawn mode (exit 2
on `shared`+clocksync) and the RetainDB and Hindsight-pg0 clocksync arms stand up
throwaway per-shard Postgres inside the faked domain instead of the shared
`hindsight-pg`. Clock-sync realigns only the OS clock the server reads; it adds no
query rewriting, so a provider's native temporal behavior stays as real plugin
behavior to report.

### Supermemory clock-sync: respawn per session on a forward clock jump (2026-07-29)

Four `BENCH_CLOCKSYNC=1` runs died 35 to 54 minutes after server boot from a
node-cron replay storm (six unconditional crons, ~0.46 MB per unyielded replay
iteration, ~163k slots per persona, ~73 GB, host OOM-kill; mechanism in
`docs/TROUBLESHOOTING.md`). The fix (committed `241f71c`): under clock-sync the
adapter respawns the spawned server once per session on the same data dir, so a
server never observes a forward clock jump and the replay span is structurally
zero. Knob: `SUPERMEMORY_RESPAWN_PER_SESSION` (default 1 under clock-sync).
Validated on `clkfix_p9`: 54/54 documents, 113 questions, 0 failures, 54
respawns, 44.3 min, against `smnoclk_p9` at 45.2 min, so the respawn overhead is a
wash. `SUPERMEMORY_HTTP_RETRIES=30` stays in the preset for ordinary transport
blips; it was never what makes a clock-synced run survive the storm.

### mem0 upgraded to 2.0.14 before the rerun (2026-07-26)

The image pinned `mem0ai==0.1.118` while the checked-in Hermes plugin declares
`mem0ai>=2.0.10,<3`, so the benchmark was measuring a version the plugin under
test does not accept. The pin is now `mem0ai[nlp]==2.0.14` (commit `5580f27`),
with `en_core_web_sm` and the fastembed BM25 model baked at image build (both
would otherwise attempt egress-blocked runtime downloads and silently degrade the
hybrid arm), and qdrant server `v1.12.4` to `v1.18.3` to match what the client 2.x
resolves.

**This is a provider generation change and a FINDING, not a config tweak.**
0.1.118 ran a two-pass algorithm (extract facts, then an explicit
ADD/UPDATE/DELETE/NONE decision per fact, exactly what MemConflict probes). 2.0.14
is single-pass and ADD-only: the update-decision pipeline is gone, results are
hard-coded ADD, and hybrid semantic + BM25 + entity retrieval replaces pure vector
search. Confirmed at source 2026-07-28 in the installed package:
`memory/main.py:1165-1168` stamps `"event": "ADD"` as a literal, the `infer=True`
path makes one call on a prompt reading "Your sole operation is ADD", and
`get_update_memory_messages` survives only as dead code. So
`Total_Event_UPDATE/DELETE/NONE` counters are 0 by construction, every mem0
artifact before 2026-07-26 is relabeled legacy-algorithm, and 0.1.118 numbers are
not comparable to 2.0.14. Report the loss of the update decision as a property of
current mem0, not a harness regression.

The upgrade deletes the 0.1.118 frozen-extraction-prompt monkeypatch (2.x resolves
the prompt's Observation/Current Date per `add()` from the process clock, so
`libfaketime` alone covers it) and replaces it with a fail-closed guard:
`BENCH_CLOCKSYNC=1` refuses to run unless mem0ai is 2.x and the legacy
frozen-prompt symbol is absent.

### mem0 search threshold sent as explicit `0.0` (2026-07-26)

2.x `search()` takes `threshold` with a vendor default of `0.1`, applied to a
blended hybrid (semantic + BM25 + entity) score. Left at the default, it would
hand the answerer fewer memories than the shared top-K every other provider gets,
a harness asymmetry rather than a quality signal. The adapter sends `threshold=0.0`
explicitly, exactly as the Supermemory arm sends `0.0` against that vendor's 0.6
default. Retrieval width is `top_k=20` (the 2.x provider default) sliced to the
shared top-5 for the answer context, mirroring RetainDB 10 to 5 and Supermemory
10 to 5. The call shape also changed: `search()` rejects a top-level `user_id` and
dropped `limit=`, so the call is now
`search(query, filters={"user_id": …}, top_k=20, threshold=0.0)`.

### Diagnostic retrieval capture and token accounting are generation-format requirements (2026-07-26)

These can only be recorded while a run generates (commits `025fa40`, `fa91e5b`).
Every question row carries `Provider_Raw_Retrieval` (the unmodified provider
response) and `Normalized_Ranked_Retrieval` (the full ranked list before the
shared top-K slice), plus a `Diag_Retrieval_Depth_Max` field, so top-K curves are
computable offline without a re-generate. This does not touch the fairness
contract: no request was widened, the capture is popped before prompt construction
so the answer context is byte-identical, and the scorer reads named fields only.
`benchmark/token_usage.py` snapshots vLLM's Prometheus counters at run and shard
scope, so a run records input/output tokens consumed by provider-internal
extraction and consolidation as well as by answering and judging. Companion to
both: `run_contract_hash` (see BENCHMARK_MATRIX "Run contract"). Under
`STRICT_RUN_CONTRACT=1` or `BENCH_CLOCKSYNC=1` a run that cannot state its own
contract aborts at generate (exit 3).

### Hindsight KEEPS `query_timestamp` (2026-07-28)

Superseded in part 2026-07-31: the parameter stays on both arms (that half holds),
but the featured arm now runs a per-container embedded pg0 cluster under
libfaketime and declares `temporal_capability=controlled_process_clock+postgres`,
so `native` describes the minimal arm only (see "Hindsight featured arm moves to
embedded pg0" below).

The parity audit found Hindsight sends `query_timestamp` where the plugin sends no
date (`plugins/memory/hindsight/__init__.py:1505-1516`). Unlike RetainDB and
Supermemory it is kept, because it is the vendor's own mechanism for this: a
DB-now audit (2026-07-24, installed wheel) found
`hindsight_api/engine/query_analyzer.py`'s `analyze()` defaults `reference_date =
datetime.now()` when none is passed, and `hindsight_client.py:394` documents
`query_timestamp` as "the query-time anchor for relative temporal expressions and
recency scoring". Removing it would hand the analyzer a real-2026 reference date
for "recently" against 2022-2024 memories. The plugin omits it because in
production wall-clock IS the right anchor; backdated data is what makes the
benchmark differ. Label this a documented, justified divergence, not parity.
Hindsight has no recency window: 969 temporally-worded against 2,781 plain
questions in the banked v4minc run returned 113.05 against 114.45 mean raw
results, zero empties.

### RetainDB recall is plugin-faithful: no `question_date`, explicit `include_pending` (2026-07-28)

The Hermes plugin sends no date on recall; its search payload is
project/query/user_id/session_id/top_k/`include_pending`
(`.../retaindb/__init__.py:230-238`). User ruling: if the plugin does not send the
date, the benchmark must not either, and `BENCH_CLOCKSYNC=1` already puts the query
system at the question date. Verified equivalent first, on a live persona-2 store
faked to 2022-10-12: a temporally-worded query returned the identical row and
excluded the identical older rows whether `question_date` was omitted or set to the
faked date, because the server's fallback "now" is inside the faked domain.
`include_pending: True` is now sent, matching the plugin (omitted and `true` both
returned 4 results where `false` returned 2, pinning the current default).

Fail-closed guard (`eval_retaindb_server.py`): without `question_date` the temporal
anchor IS the server clock, so running without `BENCH_CLOCKSYNC=1` would anchor
recall on real 2026 against a 2022-2025 dataset and empty every "has X changed
recently?" question while exiting 0. The adapter exits 2 instead;
`RETAINDB_SEND_QUESTION_DATE=1` restores the old behavior for a diagnostic. Still
divergent, deliberately not changed: the plugin also sends `session_id` on search,
whose scoping effect is unmeasured. Open item.

### RetainDB clock-sync: `updatedAt` trigger + summary-aware lifecycle barrier (2026-07-26)

Two defects in the clock-sync arm, both confirmed against source before fixing
(commit `025fa40`). Both fixes land in `retaindb_server/`, not `external/`.

1. **`updatedAt` escaped the fake clock.** The original trigger was `BEFORE INSERT`
   and set `createdAt` only. `Memory.updatedAt` is `@updatedAt` with no `@default`
   (`schema.prisma:247-248`), so Prisma supplies it client-side on every INSERT and
   UPDATE from the same Rust query-engine `CLOCK_REALTIME` read that made
   `createdAt` escape `libfaketime`. A promotion or supersession UPDATE
   (`session-lifecycle.ts:147-158`) restamped a dataset-year memory to real-2026,
   corrupting conflict/dedup ordering (`write.ts:475 orderBy:{updatedAt:"desc"}`),
   keyword search (`api/memory.ts:389-391`), and the recency anchor
   `eventDate || documentDate || updatedAt || createdAt` (`search.ts:58-66`). Fix:
   `BEFORE INSERT OR UPDATE`; INSERT forces `createdAt` + `updatedAt` to the faked
   `now()`, UPDATE forces `updatedAt` only, leaving the logical creation date
   intact. Safe because nothing vendor-side needs `updatedAt` at real wall-clock.
2. **The lifecycle barrier released before the summary landed.** The vendor runs
   promotion (fast DB update) and the LLM summary (model round-trip) concurrently
   (`session-lifecycle.ts:252-256`, `Promise.allSettled`), and the adapter returned
   `done_promotion` on the promotion alone. Fix: `_LIFECYCLE_SQL` gained a
   `total_active` count, a session is summary-eligible when
   `total_active >= SESSION_SUMMARY_MIN_MEMORIES` (default 2), eligible sessions
   must show `has_summary`, and a summary-eligible session that times out is a
   run-gate failure. New counter: `Lifecycle_Eligible_Without_Summary`.

### RetainDB `promotion_mode` is a fidelity-labeled arm, not plugin-faithful (2026-07-26)

`RETAINDB_SERVER_PROMOTION_MODE=user_specific_legacy` is what makes the featured
arm's lifecycle non-inert: it routes mid-confidence facts to SESSION scope so the
scheduler has rows to promote and summarize. It is a vendor-exposed per-request
field injected by the benchmark client
(`retaindb_server/_retaindb_server_client.py:141-142`), so using it is legitimate
under the best-effort ruling. But the pinned Hermes plugin does not send it:
`ingest_session` posts exactly five fields (`project`, `session_id`, `user_id`,
`messages`, `write_mode:"sync"`,
`external/hermes-agent/plugins/memory/retaindb/__init__.py:264-268`). Report this
arm verbatim as "best-effort deployable arm, plugin-shaped ingest plus a
benchmark-side `promotion_mode=user_specific_legacy` field the pinned Hermes
plugin does not send; NOT an unmodified-plugin-faithful run." It stays out of any
plugin-faithful headline label, the same convention as Hindsight's "out-of-box
feature set, best-effort sampling".

### Shared embedder was bge-small under v4, gte-modernbert-base under v5 (2026-07-26)

Reversed for the featured contract v5 (2026-08-01); see the v5 serving envelope
and the Reversed table. This decision stands for contract v4 minimal runs, which
are banked and not re-run.

mem0ai 2.x embeds the ENTIRE `add()` input as one related-memory search query
before extraction (`Memory._add_to_vector_store:47`), so the shared embedder's
512-token cap became an ingest-cadence constraint. The table shows the share of
`add()` windows over the cap, on Step4_4.jsonl (5 personas, tokenized on
`vllm-embed`):

| cadence | over 512 | median | p95 | max | adds/session |
|---|---|---|---|---|---|
| whole session | 100% (261/261) | 4087 | n/a | 9180 | 1 |
| 8 msgs / 4 turns | 8.8% (267/3032) | 358 | 564 | 3321 | 11.6 |
| 6 msgs / 3 turns | 1.9% (75/3997) | 268 | 429 | 1804 | 15.3 |
| 4 msgs / 2 turns | 0.4% (21/5918) | 180 | 296 | 1694 | 22.7 |
| 2 msgs / 1 turn | 0.1% (8/11713) | 90 | 156 | 1585 | 44.9 |

Whole-session ingest is unrunnable on 2.x: a 1-persona smoke returned a 400 on all
53 `add()` calls and stored ZERO memories. The v4 minimal arm ran 3 turns per add
(`MEM0_ADD_BATCH_SIZE=6`), 1.7% truncation live. The v4 featured wave ingested per
turn (0.1%). Under v5 the gte-modernbert-base 8192-token window ends the cap
entirely, and `MEM0_ADD_BATCH_SIZE` returns to 8.

The v4 decision NOT to swap the embedder held because a swap redefines the shared
retrieval-embedding surface for every provider (a new contract, non-comparable
with v4), and no model with a window above 512 emits dim 384, so any swap also
reprovisions every store. The strongest candidate then, and the eventual v5 pick,
was `Alibaba-NLP/gte-modernbert-base` (8192 native positions, encoder so no KV
cache, MTEB retrieval 55.33 against bge-small's ~51.7, apache-2.0), chosen because
it needs NO instruction prefix (`prompts: {}`). A wrong prefix degrades retrieval
with no error, and `bge-*-en-v1.5` was tuned to work without instructions, so the
harness already ran in the regime bge v1.5 was built for. The investigation also
verified vLLM's `truncate_prompt_tokens` is a per-request pooling extra-param, not
a server-side default, so no serving flag makes the embedder truncate instead of
returning a 400. `eval_mem0.py`'s embed shim sends it per request and counts the
clipped calls; keep that shim at any cadence because single messages run to 1585
tokens.

### Hindsight featured arm adopts the community mission recipe (2026-07-30)

User decision. The featured Hindsight arm sets the three missions and the
extraction mode from the r/hermesagent community recipe (reddit comment `oor0pn4`
on thread `1try46g`):

- `HINDSIGHT_API_RETAIN_MISSION` retains durable, reusable outcomes (preferences,
  decision boundaries, constraints, verified conclusions, root-cause findings,
  stable environment facts, recurring workflow rules, notable entities, important
  communications, actionable tool/cron/email outputs) and excludes filler,
  duplicate paraphrases, status chatter, unlabeled speculation, debug noise, and
  stale residue.
- `HINDSIGHT_API_OBSERVATIONS_MISSION` tracks stable preferences, recurring
  routines, important people and relationships, and priority shifts over time.
- `HINDSIGHT_API_REFLECT_MISSION` is set for config completeness only; the
  benchmark never calls `reflect()`, so it is inert.
- `HINDSIGHT_API_RETAIN_EXTRACTION_MODE=concise` IS the shipped 0.8.4 default
  (`config.py:948`), pinned so a vendor default change cannot silently move the
  arm.

Mechanism: upstream treats missions as per-bank config (`update_bank_config()`),
resolved global env to tenant to bank (`hindsight_api/config_resolver.py`), so
setting them as `HINDSIGHT_API_*` env on the embedded daemon makes them the global
default every persona bank inherits, and `write_manifest.py` records them into the
run-contract hash. The featured arm is now "Hermes plugin defaults +
community-recommended missions", the only provider whose featured arm carries a
community-sourced prompt config. All five `<provider>_featured_clocksync` presets
are encoded in `benchmark/docker/presets.sh`.

### Hindsight featured arm moves to embedded pg0 under libfaketime (2026-07-31)

User decision. The featured arm retains with `update_mode="append"`, and
hindsight-all 0.8.4 drops the caller's `timestamp=` on that path, so the daemon's
OS clock stamps `mentioned_at`. The `ftclk1_p0` smoke against the shared
`hindsight-pg` at real time stored 2,327 world, 9 experience, and 1,429
observation rows, every one stamped 2026-07-30/31 and not one 2022 date, on a 2022
dataset. Persona 0, the same 122 questions as the banked minimal run: update order
recognition on dynamic questions 0.547 to 0.305, micro answer accuracy 0.475 to
0.344, with nothing failing. Cause and line numbers are in `docs/TROUBLESHOOTING.md`.

`temporal_capability=native` is now arm-scoped. The exemption rested on the retain
timestamp being honoured, which happens on `_retain_one` (session granularity, the
minimal arm) and not on the append merge (the featured arm). So the minimal arm
keeps `native` and the featured arm takes the OS-clock route.

The selector is `HINDSIGHT_PG_MODE=pg0`, set only by
`_preset_hindsight_featured_clocksync`. Storage carries the whole arm: pg0 without
the preload reproduces the defect, and a preload against the shared cluster would
write 2022 rows into a co-tenant database other shards read at real time.
`BENCH_CLOCKSYNC` cannot be the selector because `hindsight_minimal_clocksync`
sets it too and must stay on `hindsight-pg`. Mechanism, one preload point:

- `_inject_daemon_clock_env` (`hindsight/eval_hindsight.py:103`) writes
  `LD_PRELOAD` plus the `FAKETIME_*` contract into `os.environ` immediately before
  `HindsightEmbedded(...)`. The vendor spawns the daemon with
  `subprocess.Popen(cmd, env=os.environ.copy())` and the pg0 Rust CLI orphans the
  postmaster with that environment, so one write fakes the daemon, `initdb`, and
  the postmaster. `LD_PRELOAD` binds at exec, so the adapter and entrypoint shell
  keep the real clock. A missing `.so` raises (fail closed).
- Three conditions gate injection: `BENCH_CLOCKSYNC=1`, `BENCH_CLOCKSYNC_FILE`
  set, and `HINDSIGHT_EMBED_API_DATABASE_URL` unset. The minimal arm fails all
  three. `entrypoint.hindsight.sh` also exits 2 without `BENCH_CLOCKSYNC=1`, unless
  `END_IDX - START_IDX == 1` (a persona rollover would rewind the clock over a live
  store), and on a set `HINDSIGHT_API_DATABASE_URL`. `ALLOW_EXISTING_PG0=1` is the
  only way to relaunch onto an existing cluster.
- `HOME` moves to `/tmp/hs_home_${TAG}`, because pg0 hardcodes its data directory
  to `~/.pg0/instances/<name>/data` and compose mounts the shared
  `hindsight_state` volume at `/home/bench`.
- `HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=0` disables the daemon's idle auto-exit; its
  checker loop compares `time.time()`, which libfaketime fakes, so a faked forward
  jump between sessions reads as idleness and kills the daemon mid-run. Do not
  declare the key empty in compose (the manager int-parses it and the empty-var
  guard matches `HINDSIGHT_API_` only).
- `benchmark/docker/run_shards.sh` detects the arm from `HINDSIGHT_PG_MODE=pg0` or
  `PRESET=hindsight_featured_clocksync`, exports `BENCH_CLOCKSYNC=1`, launches
  `--no-deps` so `depends_on` does not drag `hindsight-pg` up, brings up only
  `vllm-gen` and `vllm-embed`, and defaults the pool to 4 (~2.5 GB per container).
  `hindsight-rerank` must be started by hand.

`BENCH_TEMPORAL_CAPABILITY` is set by entrypoints, never by a preset or operator.
`write_manifest.py:381` validates it against `native`, `controlled_process_clock`,
and `controlled_process_clock+postgres`; an unrecognised value records MISSING so
the strict gate aborts rather than hashing a typo. The provider map stays the
fallback, so the minimal arm and every re-score still read `native`.

`query_timestamp` stays on both arms. `ftclk1_p0` settles the 2026-07-28 open
question: recall applies no upper-bound filter against the anchor. Every question
ran with a 2022 `query_timestamp` while every stored fact carried a 2026
`mentioned_at`, and recall still returned a mean of 113 memories per question,
minimum 70; a filter would have returned zero.

pg0 facts from the built image: the wheel bundles PostgreSQL inside a 33.7 MB Rust
binary (no egress), extracts to `~/.pg0/installation/18.1.0` in 3.8 s, and the
extracted `postgres` is ET_DYN and glibc-linked so `LD_PRELOAD` reaches it (a faked
`SELECT now()` returned 2022-03-14 while the host read 2026-07-31). Base image
Debian 13 trixie, glibc 2.41, libfaketime 0.9.10. Rollback is one line:
`HINDSIGHT_PG_MODE=shared` in the preset disarms the preload, guards, and
capability override. Decisive smoke `ftsmk086_p0`, 2026-07-31: pass.

### Hindsight vendor bump to 0.8.6, and the version confound it creates (2026-07-31)

`Dockerfile.hindsight` now pins `hindsight-all==0.8.6` and `pg0-embedded==0.15.0`.

- **0.8.6 over 0.8.4.** PR #2684 (in 0.8.5) restores `event_date` through the
  append merge; PRs #2930 and #2935 (in 0.8.6) are on the same path, so 0.8.6 is
  taken. Issue #3010 is open and is why libfaketime stays: the merge still
  collapses every item's date to the first item's (`first.get("event_date")`), and
  the `utcnow()` fallback is still there. Verified in the built image at
  `hindsight_api/engine/retain/orchestrator.py:982-983` and `:1009-1010`; the
  `utcnow()` fallback has moved to `:2621`.
- **`pg0-embedded==0.15.0` pinned explicitly**, because hindsight-all asks only for
  `>=0.14.2` and the bundled PostgreSQL would otherwise drift between image builds.
  0.15.0 measured: PostgreSQL 18.1.0, pgvector 0.8.5, pg_trgm 1.6.

So the fix and the fake are both kept: 0.8.6 alone leaves the per-item date
collapse and the fallback, libfaketime alone leaves a vendor bug in the product
under test, and together the daemon writes the logical date and, where the vendor
still discards it, the fallback lands inside the same logical domain.

The bump is a confound, and it is accepted. The banked minimal artifacts (`v4minc`,
30 personas) ran 0.8.4; the featured wave runs 0.8.6. The answer is a rerun, not a
caveat: `v4minc086` regenerates the full 30-persona minimal wave on 0.8.6 with the
same preset, same shared `hindsight-pg`, and `temporal_capability=native`
unchanged. The version-clean pair is `v4minc086` against the featured wave. The
0.8.4 `v4minc` run was the 0.8.4 minimal result, superseded for comparison and
since removed from the tree.

### Honcho's ~25k-token injection is plugin-faithful and at the shipped ceiling (2026-08-02)

Verified at every layer after a user challenge. The real Hermes plugin's hybrid
injection for a peer with any non-trivial history is ~99,300 characters, about
25,000 tokens per turn, replayed verbatim on every later turn of the session, and
nothing reduces it: the plugin's `_truncate_to_budget` ships with its budget unset
(a no-op; `HONCHO_CONTEXT_TOKENS` is that knob), the SDK forwards no size
parameter, the server caps the representation at 100 OBSERVATION ROWS
(`WORKING_REPRESENTATION_MAX_OBSERVATIONS`, default 100) with no token cap, and
Hermes's own context-pressure gate compacts the user's conversation history around
the block while the block survives intact. The size is per-row verbosity (deriver
markdown expands premises and sources, ~530 chars/row), not accumulation: session
3 already carried 62 rows. 96% of the block is the two representations; 52% is the
AI self-representation, fetched with NO search query and therefore unranked
against the question. The adapter matches the plugin field-by-field (calls,
fallbacks, order, 600-char dialectic clip).

If a future arm wants a smaller Honcho injection, lower the server's observation
cap (range 1-1000), the only knob that reduces content, rather than the plugin's
character budget, which cuts the AI self-representation mid-word.

### The Honcho wave alone is judged at a 49,152-token window (2026-08-05)

The second judge (gemma-4-12b, penalty rubric `_gj12pen`) was served at
`--max-model-len 32768` while the harness requests `max_tokens=16384`. vLLM
subtracts the output reservation from the window, so the real input budget was
16,384 tokens. Every one of Honcho's 3,750 judge prompts exceeds that (minimum
17,095, median 24,602, maximum 30,812, measured through the scorer's own
`build_llm_judge_prompt` and the served tokenizer). The wave produced 3,057 HTTP
400s and silently fell back to rule-based scoring before it was stopped.

The decision is to re-serve the judge at 49,152 for the Honcho wave only, at 12
workers, with `max_tokens` and sampling unchanged, and NOT to re-score the four
waves already banked under 32,768.

This does not breach ruling 1. `--max-model-len` is a capacity cap, not a decoding
parameter. The judge declares `max_position_embeddings = 262144` and no
`rope_scaling`, and vLLM applied none at boot, so the first 32,768 positions
compute identically at either setting; sampling, `max_tokens`, the rubric, and the
prompts are untouched. Lowering `max_tokens` to 8,192 was rejected: it still
rejects 1,898 of 3,750 prompts and would truncate real judge calls on the other
waves (the judge thinks, `completion_tokens` includes the reasoning trace, a full
wave recorded 14,438 output tokens against ~180-token verdicts), forcing the
re-score it was meant to avoid. Truncating Honcho's block is barred by ruling 3,
and reporting Honcho "unscorable" would publish a serving shortfall as a product
property. A strict reading wanting all eight waves under one serving command is
answered by the unseeded temperature-1.0 sampling already accepted; if adopted, the
remedy is re-judging the four waves at 49,152, not leaving Honcho unscored. Report
with the number: Honcho's featured injection averages roughly 25,000 judge-prompt
tokens per question, about ten times every other provider, a finding about the
product's read surface under ruling 3.

### Mnemosyne featured arm: shipped consolidation, manually cadenced (2026-08-02, user ruling)

The featured arm adds `--plugin_session_sleep` (`PLUGIN_SESSION_SLEEP=1` in
`mnemosyne_featured_clocksync`): one forced, drained `sleep(force=True)` after each
session's ingest, before that session's questions. It mirrors Honcho's
`HONCHO_DREAM_AFTER_SESSION` ruling and carries the same label: shipped
consolidation, manually cadenced.

Why: the ft27mn smoke measured the featured arm's premise failing. The arm keeps
the shipped 168-hour working-memory TTL because consolidated rows are trim-exempt
(`beam.py:3836-3849` deletes only `consolidated_at IS NULL`), but the plugin's
auto-sleep gate (`working_count > 50`) needs cross-session accumulation that the
trim itself removes: under the faked clock the median inter-session gap is 29 days,
so each session's first write deletes all prior unconsolidated rows. Measured: 2
auto-sleep invocations in 277 ticks, 12 episodic rows, 21 of 122 questions with
zero recall candidates. The banked minimal arm (same persona, same writes, explicit
long TTL) has zero.

The forced sleep is the vendor's own consolidation (`force=True` only skips the age
cutoff, `beam.py:8055-8058`; the count gate exists only in the plugin wrapper, so
beam-level sleep bypasses it). The shipped TTL stays live, and the entrypoint still
refuses an explicit `MNEMOSYNE_WM_TTL_HOURS` on this arm. Runtime summary counts
`Total_Session_Sleep_*`. If consolidated recall scores worse, report it (ruling 3).

### Hindsight featured ranking left at vendor defaults (2026-08-02)

The featured arm delivers recall in the order `hindsight-all` 0.8.6 produces, with
no repo-side re-sort. The vendor's final ranking is

```
weight = normalized_reranker_score * recency_boost * temporal_boost * proof_count_boost
```

(`hindsight_api/engine/search/reranking.py:191-192`). The three boosts come from
hardcoded alphas 0.2, 0.2, and 0.1, giving ranges of 1±0.1, 1±0.1, and 1±0.05, so
a candidate can outrank one whose reranker score is 1.3 to 1.6 times higher.
Delivered order is perfectly monotonic in the fused `score` field (122 of 122
questions in `ft27hs2`, 0 violations) and diverges from the raw reranker score on
119 of 122. The Hermes plugin passes this order through unchanged
(`plugins/memory/hindsight/__init__.py:1516-1519`), so the arm is plugin-faithful
on ordering.

Every ranking knob was surveyed and every one stays at its shipped value: recency
decay function and window, the strategy boosts, the reranker's maximum candidate
count, the per-source caps, and the consolidation dedup threshold. None has a
vendor-endorsed alternative (what ruling 2 asks for), the boost alphas are
hardcoded with no environment variable, and the `reranking` mode parameter cannot
be reached through the HTTP recall route (`hindsight_api/api/http.py:4183-4205`).
Keep the defaults and report the findings: boosted fusion is the product under test
(ruling 3), and supporting evidence sinking below recency-boosted near-duplicates
is a result to publish. Optional diagnostic arm, never a headline number:
`HINDSIGHT_API_RECENCY_DECAY_FUNCTION=none`, a vendor-shipped validated value,
isolates the recency boost's contribution.

---

## Reversed

This table lists decisions that were made and then overturned. It keeps them
because the reasoning is easy to arrive at again.

| Decision | Why it was made | Why it was reversed |
|---|---|---|
| Regress vLLM to `v0.24.0` to escape the wedge | Assumed a 0.25.x regression | The fix (flashinfer 0.6.14, PR #47669) merged two days after v0.25.1 was cut, so no tag contains it. Older tags ship older flashinfer and keep the bug; v0.24.0 also predates the SM120 startup fix |
| The NVFP4 checkpoint causes the wedge | vllm#48718 was open and named NVFP4 | That issue was closed 2026-07-19; the cause was the flashinfer sampler. The v2 to v3 checkpoint swap was unnecessary, and was kept only because zero personas had banked either way |
| Canonical decoding = temp 0.2 / 3072 answer, 4096 judge | The contract two Mnemosyne runs already declared (gemma era) | Superseded by Qwen model-card sampling. Temp 0.2 produces rumination and empty answers on Qwen |
| `SCORE_WORKERS=24`, frozen | Load-induced latency was measured at 23 to 40 concurrent requests | Raised to 40 with `MEMCONFLICT_REQUEST_TIMEOUT` moved to 600 to match. The two must move together |
| Merge the Mnemosyne chronology and plugin-config re-runs | Fewer runs | Rejected: it conflates the chronology delta with the config delta against the committed baseline. Run them separately |
| Sequence RetainDB's full run after the decoding-parity fix | Avoid the confound | Moot: RetainDB Local was ruled out entirely and never ran at full scale |
| Add shards to speed up RetainDB | GPU looked idle | The bound was quadratic CPU compute. 15 shards was 4.6x slower in aggregate than 6 |
| `--consolidation_wait_timeout_s = 300` | Original default | Raised to 450; drains regularly exceed 300s under load |
| Do NOT swap the shared embedder (2026-07-26) | A swap forces a new contract plus store re-provisioning to solve a cap the featured cadence avoids; the candidates researched then either had a vLLM bug (gte-modernbert-base, #28564) or a mandatory prefix that degrades silently | The featured wave IS a new contract by user ruling (2026-08-01), so both costs are already paid. The final v5 embedder is that research's own first choice, gte-modernbert-base, unblocked because #28564 no longer reproduces on v0.25.1. Reversed for v5 only; v4 minimal keeps bge-small |
| v5 embedder = Qwen3-Embedding-0.6B at 1024 dims (2026-08-01) | 32k window, optional prefix, 1024 matched RetainDB's pinned column with no pad | Lasted one day, nothing banked. It is a decoder (needs 3.5 GiB KV per 32k request, forcing a GPU re-split), and Supermemory 0.0.5's embedding step caps a batch at ~12,288 floats, wedging 99.9% of session documents at 1024 dims. User re-ruled to 768 dims: gte-modernbert-base (2026-08-02) |

---

## Open questions

These questions have been raised and are not resolved. Verify current code before
acting on any of these.

1. **Mnemosyne's plugin-faithful arm omits the plugin's auto-sleep.** The real
   plugin auto-runs `sleep()` every 10 turns (`MNEMOSYNE_AUTO_SLEEP_ENABLED`
   defaults True). Since automated consolidation is known to lower Mnemosyne's
   score, omitting it asymmetrically flatters Mnemosyne against Hindsight's B/C
   arms, which do enable their equivalent. The `--plugin_auto_sleep` arm now
   exists. The naming still needs to distinguish "plugin write-path fidelity" from
   full plugin fidelity.
2. **Arm labels overstate what was run.** The project labels Arm A "out-of-box"
   despite storing both roles at uniform importance 0.6 and omitting plugin
   truncation. Proposed relabels: Arm A to `benchmark raw-message baseline`,
   both-role plugin arm to `best-effort plugin write configuration`, oracle to
   `gold-derived canonical capacity ceiling`.
3. **The answer model shares `vllm-gen` with memory maintenance.** Extraction,
   consolidation, answering, and judging all hit one server, so memory-side load
   changes answer latency. Arm B showed that consolidation can saturate it.
   Options: separate servers, or freeze retrieved evidence and replay answers
   after memory work drains.
4. **Qwen judge reliability is unverified.** JSON mode constrains valid JSON, not
   the judge's schema, so "schema-clamped" is a stronger claim than what is
   implemented. With temp 0.6 and no seed, one paired-bootstrap sample does not
   capture sampling variance. Before trusting close calls, the project wants zero
   empty judge responses, zero rule-based fallbacks, and repeat-judge agreement on
   a stratified sample.
5. **Hindsight's logical timestamps are tied within a session:** every appended
   exchange gets the same dataset date, where Mnemosyne advances one minute per
   write. The project added a per-exchange offset at the retain call sites. Confirm
   it covers every provider before comparing chronology-sensitive arms.
6. **Reusing a Hindsight `RUN_TAG` reuses its database.** Retrieval isolation
   survives (bank IDs are unique), but startup consolidation sweeps and resource
   load can be contaminated by the prior run. Use a fresh tag per run.
7. **Missing diagnostics.** Paired bootstrap confidence intervals with McNemar
   counts, per-channel context-token accounting, failure coverage in the summary,
   memory-side LLM call and token counts.
8. **Which model plays the "competent agent"** for a non-gold agent-derived
   canonical arm. This needs a model pick and a VRAM budget. A stronger local
   model preserves the locally-hostable constraint.
