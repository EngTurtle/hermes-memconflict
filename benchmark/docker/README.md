# Containerized benchmark stack

This stack runs the MemConflict memory-provider benchmark as a Docker Compose
stack, including the two shared vLLM inference servers. Docker manages the
full process lifecycle: launch each process detached, read its logs, then
stop it cleanly.

The stack is common across providers. The two vLLM servers start once and
stay up. Each benchmark run is a separate, short-lived container:
`docker compose run -d --rm <provider>` runs the container to completion,
then removes it.

## Services

Three inference servers exist. The two generate-stage servers have no profile,
so a bare `docker compose up -d` starts only them. `vllm-judge` is the
score-stage second judge and sits under the `judge` profile, so the same bare
`up -d` never starts it beside the other two on the one 16 GiB card.

| Inference server | Profile | Port | Role |
|---|---|---|---|
| `vllm-gen` | (default) | `8000` | Answer LLM, the DEFAULT judge, and each provider's internal LLM: `qwen3.5-4b` (checkpoint `AxionML/Qwen3.5-4B-NVFP4`), `--max-model-len 131072`. In-network URL `http://vllm-gen:8000/v1` |
| `vllm-embed` | (default) | `8001` | Recall embeddings: `gte-modernbert-base` (checkpoint `Alibaba-NLP/gte-modernbert-base`), 768 dims, pooling, `--max-model-len 8192`, `--pooler-config '{"use_activation": true}'`. In-network URL `http://vllm-embed:8000/v1` |
| `vllm-judge` | `judge` | `8002` | The OPTIONAL second judge for the score stage only: `gemma-4-12b` (checkpoint `unsloth/gemma-4-12b-it-NVFP4`), `--max-model-len 32768` (`VLLM_JUDGE_MAX_LEN`), `--gpu-memory-utilization 0.9`. It assumes `vllm-gen` AND `vllm-embed` are STOPPED. A number it produces is the `_gj12` arm and compares only against other `vllm-judge` scores, never against a `qwen3.5-4b`-judged score. In-network URL `http://vllm-judge:8000/v1` |

Every provider run-service sits under the `run` profile. `docker compose run
<provider>` launches it explicitly and, through `depends_on: { condition:
service_healthy }`, also starts the servers it needs if they are not already
up. Each `run --rm` container gets a unique name, so several runs (with
distinct `RUN_TAG`) can run at the same time. Every provider writes to its own
`../<provider>/Results` and `../<provider>/Scores` (bind-mounted to the host).

| Provider run-service | Status | Support services it needs first |
|---|---|---|
| `mnemosyne` | **Full, verified.** Runs `mnemosyne/eval_mnemosyne.py`, score, summarize, then exits. Shards internally (`NUM_SHARDS`) | `vllm-gen`, `vllm-embed` |
| `hindsight` | **Full. It runs in Docker.** Image `memconflict-hindsight`. Sharded via `START_IDX`/`END_IDX`. Arms A/B/C | `vllm-gen`, `hindsight-pg`; plus `hindsight-rerank` for the default `HINDSIGHT_RERANK_SOURCE=remote` (NOT auto-started, see below) |
| `mem0` | Run-service. Sharded via `START_IDX`/`END_IDX` | `vllm-gen`, `vllm-embed`, `qdrant` |
| `supermemory` | Run-service. Attaches to one central server | `vllm-gen`, `supermemory-server` |
| `honcho` | Run-service. Shared or spawn server mode | `vllm-gen`, `vllm-embed`, `honcho-pg`, `honcho-db-init`, `honcho-api`, `honcho-deriver` (shared mode); spawn mode needs none of the honcho-* services |
| `openviking` | Run-service. Spawns its own in-container server | `vllm-gen`, `vllm-embed` (no database sidecar) |
| `retaindb` | RetainDB **local** run-service. **Scaffold** (RULED OUT, kept for the record), not yet run in Docker | `vllm-gen` |
| `retaindb-server` | RetainDB **server** edition (`@retaindb/server`, Postgres/pgvector + LLM extraction). **Scaffold** | `vllm-gen`, `vllm-embed` (via `embed_proxy.py`), `hindsight-pg` |

Support services (all under the `run` profile): `hindsight-pg`
(`pgvector/pgvector:pg18`, shared Postgres for Hindsight and RetainDB server),
`hindsight-rerank` (TEI cross-encoder, GPU, NOT in any `depends_on`), `qdrant`
(mem0 vector store), `supermemory-server` (central Supermemory server),
`honcho-pg`/`honcho-db-init`/`honcho-api`/`honcho-deriver` (the Honcho server
tier).

`vllm-gen`'s context window is `--max-model-len 131072` (featured contract
v5). It is sized by the largest measured prompt, Honcho's dream at ~72,708
tokens of accumulated tool results plus the 8,192-token output reservation.
Contract v4 minimal ran 32768, which fits Hindsight's retain and
consolidation budgets. vLLM allocates KV per actual token, not per configured
window, so the window only sets how many concurrent long sequences fit the
same `gpu-memory-utilization` budget: measured KV pool 434,238 tokens, 3.31
full-window requests, identical at 0.85 and at 0.74. That is enough for
single-persona smokes, not for a 30-persona wave. Prompt lengths are
unchanged, so answer and judge behavior is unchanged.

`vllm-embed` serves `Alibaba-NLP/gte-modernbert-base` at 768 dims (amended
2026-08-02, `docs/DECISIONS.md`). Contract v4 minimal ran
`BAAI/bge-small-en-v1.5` at 384 dims with a 512-token input cap.

The serving window is `--max-model-len 8192` (`VLLM_EMBED_MAX_LEN`), the
model's native `max_position_embeddings`, which ends every v4 truncation
shim. `--pooler-config '{"use_activation": true}'` is REQUIRED: the
checkpoint's `modules.json` lists no Normalize module, so vLLM otherwise
returns unnormalized vectors (measured L2 ~37-38) and any consumer scoring by
raw dot product sees a ~38x scale shift. Measured on the live boot
2026-08-02: 15 inputs of 8,192 tokens each return in 1.2 s, every vector 768
dims at L2 1.000000.

The GPU split follows from that: `vllm-embed` at `--gpu-memory-utilization
0.07` (~1.14 GiB; it is a BERT-family encoder and reserves NO KV cache, boot
log `kv_cache_size_tokens=None`), and `vllm-gen`'s generate-profile default
back at 0.85 (`VLLM_GEN_GPU_MEM`). The score-stage 0.94 profile is unchanged,
since `vllm-embed` is stopped during scoring.

Vector stores are dimension-bound, so a v5 run needs fresh mem0 collections, a
fresh `supermemory_data` volume, and Honcho's column retype at 768. Never put
a v5 number in the same table as a v4 number.

## Prerequisites

- Docker Desktop (WSL2) with the NVIDIA Container Toolkit (GPU passthrough).
- HuggingFace cache on the host with the two models already pulled
  (`AxionML/Qwen3.5-4B-NVFP4`, `Alibaba-NLP/gte-modernbert-base`). Mounted read/write so
  nothing re-downloads. Override the path with `HF_CACHE=...` if your cache
  path differs from `C:/Users/ollie/.cache/huggingface`.

## Usage

All commands run from `benchmark/docker/`.

```bash
# 1. Bring the shared inference servers up once (they stay up):
docker compose up -d vllm-gen vllm-embed

# 2. Launch individual benchmark runs (each is a throwaway container):
docker compose run -d --rm mnemosyne                       # full Mnemosyne baseline
docker compose run -d --rm -e ORACLE=1 -e RUN_TAG=oracle mnemosyne
docker compose run -d --rm -e NUM_PERSONAS=1 -e RUN_TAG=smoke mnemosyne

# Watch a detached run (name is auto-generated; find it with `docker ps`):
docker logs -f <container>

# Re-run a single stage against existing artifacts:
docker compose run -d --rm -e STAGE=score     -e RUN_TAG=oracle mnemosyne
docker compose run -d --rm -e STAGE=summarize -e RUN_TAG=oracle mnemosyne

# Hindsight: full run-service (build their image on first run, see below):
docker compose run -d --rm hindsight
docker compose run -d --rm -e START_IDX=0 -e END_IDX=8 -e RUN_TAG=shard0 hindsight   # sharded

# RetainDB: still a scaffold, expect tuning:
docker compose run -d --rm retaindb

# 3. Stop the vLLM servers when done:
docker compose down
```

Each provider run-service exits when its run finishes (`restart: "no"`). The
vLLM servers stay up (`restart: unless-stopped`), and other runs reuse them.

## Running each provider

All commands run from `benchmark/docker/`. Bring the two generate-stage
servers up once first (`docker compose up -d vllm-gen vllm-embed`). Each
command below shows a single-persona smoke (`NUM_PERSONAS=1`); drop
`NUM_PERSONAS`/`MAX_SESSIONS` for a full 30-persona run, and give each wave a
fresh `RUN_TAG`. Do NOT cap a smoke with `MAX_SESSIONS=1`: persona 0 answers
its first questions in session 5, so a one-session cap answers nothing. The
`STAGE=all` default runs generate, then score, then summarize. The generic
knobs (`STAGE`, `NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`,
`MAX_QUESTIONS_PER_SESSION`, `SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` /
`OPENAI_MODEL`) apply to every provider. The support services in each
provider's `depends_on` start automatically; the two exceptions
(`hindsight-rerank` and any `--no-deps` spawn mode) are called out below.

The scored comparison arm for each provider is a named preset
(`benchmark/docker/presets.sh`, passed as `-e PRESET=<name>`). The preset
carries the whole arm's env under one name and records it in the manifest.

```bash
# --- Mnemosyne (shards internally via NUM_SHARDS) --------------------------
docker compose run -d --rm -e NUM_PERSONAS=1 -e RUN_TAG=smoke mnemosyne
docker compose run -d --rm -e ORACLE=1 -e RUN_TAG=oracle mnemosyne   # arm D upper bound

# --- Hindsight (needs hindsight-pg; hindsight-rerank for remote rerank) ----
docker compose --profile run up -d hindsight-rerank    # REQUIRED first; not in depends_on
docker compose run -d --rm -e NUM_PERSONAS=1 -e RUN_TAG=smoke hindsight       # arm A
docker compose run -d --name hs_s0 -e RUN_TAG=full_s0 -e STAGE=generate \
  -e START_IDX=0 -e END_IDX=8 -e NUM_PERSONAS=30 hindsight                    # one shard

# --- mem0 (needs qdrant; sharded runs need MEM0_VECTOR_MODE=server) --------
docker compose run -d --rm -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke mem0

# --- Supermemory (attaches to the central supermemory-server) -------------
docker compose run -d --rm -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke supermemory

# --- Honcho (shared mode: needs honcho-pg/-db-init/-api/-deriver) ----------
docker compose run -d --rm -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke honcho
docker compose run -d --rm -e HONCHO_SERVER_MODE=spawn --no-deps \
  -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke honcho                 # spawn mode

# --- OpenViking (spawns its own in-container server; no DB sidecar) --------
docker compose run -d --rm -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke openviking

# --- RetainDB server (needs hindsight-pg; embeds through embed_proxy.py) ---
docker compose run -d --rm -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke retaindb-server
```

Sharded full-run waves go through `benchmark/docker/run_shards.sh <provider>
<tag>`, which accepts `hindsight | mem0 | supermemory | retaindb_server |
honcho | openviking` (note the underscore in `retaindb_server`, which maps to
the `retaindb-server` service). Mnemosyne is not a `run_shards.sh` target: it
shards internally through `NUM_SHARDS`. See the per-provider control-knob
sections below for each provider's full env surface.

## Hindsight (full run-service)

`hindsight` runs fully in Docker (image `memconflict-hindsight`, built from
`Dockerfile.hindsight`). It installs `hindsight-all` 0.8.6 with
`pg0-embedded` 0.15.0 pinned (bundled PostgreSQL 18.1.0, pgvector 0.8.5,
pg_trgm 1.6, plus local embedding and reranker models) as a non-root user
(`bench`), because `initdb` refuses to run as root. hindsight-all asks only
for `pg0-embedded>=0.14.2`, so the explicit pin is what keeps the bundled
PostgreSQL the same between image builds. The image also bakes
`libfaketime` 0.9.10, for the featured clock-sync arm below. It also
installs `procps`. pg0's
liveness check shells out to the standalone `kill` binary through
`kill -0 <pid>`. Without `procps`, the daemon reports every Postgres
instance as stopped and dies with `"Database URL is required for
migrations"`. The image installs CPU-only `torch` before `hindsight-all`, so
the transitive torch dependency does not resolve to the CUDA build (about
7GB of `nvidia-*` wheels). This container has no GPU, so embeddings and the
reranker run on CPU.

Its answer/judge LLM and its internal LLM (fact extraction plus recall query
understanding) both default to the shared `vllm-gen`. To use the
OpenRouter-verified path instead, override `OPENAI_BASE_URL` and
`HINDSIGHT_LLM_BASE_URL` to `https://openrouter.ai/api/v1` and pass `-e
OPENROUTER_API_KEY=...`. See "Hindsight run-service control knobs" below for
sharding and the arm B/C env vars, and see `../../docs/BENCHMARK_MATRIX.md`
for the arm definitions.

State: the named volume `hindsight_state` mounts at `/home/bench` (the
container's `$HOME`). It used to hold the downloaded HF embedding/reranker
weights and, per shard, the embedded-Postgres data directory (`~/.pg0`). We
removed both local/embedded modes on 2026-07-22 (remote/shared only), so
this volume is now mostly unused. The volume persists across runs. It is
safe to run `docker volume rm hindsight_state` between clean-slate runs,
because everything in it is re-creatable.

**`HINDSIGHT_PG_MODE=pg0` does not use this volume.** pg0 hardcodes its data
directory to `~/.pg0/instances/<name>/data`, and `hindsight_state` is
SHARED, so N per-persona containers would land on one cluster path. The
entrypoint exports `HOME=/tmp/hs_home_${TAG}` for that arm, onto the
container's own filesystem, which is what `entrypoint.retaindb-server.sh`
does for its cluster.

### Shared Postgres (`hindsight-pg`)

By default (`HINDSIGHT_PG_MODE=shared`), every Hindsight shard-daemon
connects to one tuned `pgvector/pgvector:pg18` service (`hindsight-pg`),
instead of each daemon booting its own embedded PostgreSQL 18 (`pg0`)
cluster. Reason: the old per-shard embedded clusters used heavy RAM (each
daemon used ~1.9GB RSS, of which ~0.3-0.4GB was its private postmaster).
They also left an ~8GB `docker_hindsight_state` volume holding N separate
initdb clusters plus WAL, on a space-limited disk.

- **Per-run database isolation (2026-07-20).** Each run gets its own
  database inside the shared server, `hindsight_<RUN_TAG>` (the entrypoint
  sanitizes `RUN_TAG` to `[a-z0-9_]`, lowercase, and maps an empty tag to
  `hindsight_default`). The entrypoint creates this database on demand,
  before the daemon boots. **Why the shared single database was a hazard:**
  a consolidation-enabled (Arm B) daemon's startup sweep enqueues
  consolidation ops for every bank in its database. With all runs in one
  database, the sweep reached across runs. We verified this live on
  2026-07-20: an Arm-B daemon enqueued consolidation onto a
  consolidation-OFF run's live bank (cross-arm contamination, a fairness
  hazard). Daemons also claimed and processed ops for banks they did not
  own, and the sweep also caught orphan banks from dead runs, which wasted
  GPU time. `bank_id` isolation alone did not fix this, because the sweep
  runs per database. Per-run databases fix both problems: a run only ever
  sees its own banks, and each fresh run starts against a fresh catalog.
  - Within one sharded run, each shard's `RUN_TAG` (`armB_s0`, `armB_s1`,
    and so on) sanitizes to a distinct database per shard. This is fine:
    banks never span shards. Migrations run per database under the advisory
    lock, so each shard database migrates on its own, with no cross-shard
    coordination needed.
  - **`HINDSIGHT_PG_DB` override.** Compose sets this to empty by default,
    so the derived per-run name is the norm. Setting it to a non-empty
    value forces one specific database and brings back cross-run sharing.
    Two runs pointed at the same name share one catalog and lose this
    isolation, so set this only on purpose. This var belongs to the run
    container. The `hindsight-pg` server's own `POSTGRES_DB` (its
    maintenance database) is separate and stays `hindsight`.
  - Within a single database, personas stay separated by a
    globally-unique `bank_id`. Multiple shard-daemons migrating one
    database is a supported Hindsight topology. Schema migrations
    serialize on a per-schema pg advisory lock (`hindsight_api/migrations.py`),
    with `CREATE EXTENSION` inside it.
- **Extensions.** The extensions are `vector` (pgvector, the default
  `HINDSIGHT_API_VECTOR_EXTENSION`) and `pg_trgm`. Text search uses native
  `tsvector`. `hindsight-pg-init.sql` seeds these extensions into
  `template1`, so every on-demand per-run database inherits them at
  creation, and also seeds them into the `POSTGRES_DB` maintenance
  database. The entrypoint's creation script (`hindsight_create_db.py`,
  asyncpg, because the image has no `psql`) also runs
  `CREATE EXTENSION IF NOT EXISTS` in each new database, as a backup over
  the daemons' own idempotent creates.
- **Cleanup between waves.** Per-run databases accumulate in the shared
  cluster. To reclaim space, either drop the whole cluster between waves
  (`docker rm -f hindsight-pg && docker volume rm docker_hindsight_pg`,
  which gives a fresh catalog at the next start), or run
  `DROP DATABASE hindsight_<tag>` for each finished run. The run image has
  no `psql`, so use the `pgvector/pgvector:pg18` image, or connect from the
  host.
- **Tuning** (see the inline comments on the `hindsight-pg` service). This
  is a disposable benchmark scratch database on a RAM-constrained WSL2 VM
  (23.6GB total, of which the vLLM containers hold ~8GB). We trade
  durability for speed here, the same way `BENCH_SQLITE_FAST` does for
  Mnemosyne:

  | setting | value | why |
  |---|---|---|
  | `max_connections` | 200 | Worst case: 10 shards times a per-daemon pool of 16 equals 160, plus headroom for reserved connections, autovacuum, and admin |
  | `shared_buffers` | 768MB | Small working set. The OS page cache handles the rest |
  | `effective_cache_size` | 4GB | Planner hint for available OS cache |
  | `maintenance_work_mem` | 256MB | Speeds up pgvector and GIN index builds, and autovacuum |
  | `work_mem` | 16MB | Recall sort budget, per sort node times connections. Kept modest |
  | `synchronous_commit` | off | Trades durability for speed. A crash loses only in-flight transactions, with no corruption (unlike `fsync=off`) |
  | `wal_level` / `max_wal_senders` | minimal / 0 | Single-node scratch database with no replication, so less WAL |
  | `max_wal_size` / `checkpoint_timeout` | 4GB / 15min | Avoids checkpoint storms under heavy ingest |
  | `max_parallel_workers_per_gather` | 0 | Many small queries. Parallel workers only add backend and RAM overhead |

  The entrypoint caps the per-daemon pool to `min 2 / max 16`
  (`HINDSIGHT_API_DB_POOL_MIN_SIZE` / `_MAX_SIZE`) in shared mode, so N
  concurrent daemons never exceed `max_connections`. Peak RAM budget for the
  service: shared_buffers 768MB, plus maintenance_work_mem 256MB
  (transient), plus backends (~10MB base each, plus work_mem on demand).
  Steady-state usage is ~1.2-1.5GB.

- **Savings** (5-shard consolidation run). Old setup: 5 embedded pg0
  clusters, about 1.5-2GB of postmaster RSS across the daemons, plus 5
  initdb data directories and WAL on disk. New setup: one ~1.3-1.5GB shared
  server plus one `hindsight_pg` volume. Total RAM is roughly net-neutral to
  modestly lower, because the model weights (~1.3-1.5GB per daemon)
  dominate and stay unchanged. But the per-daemon postmaster (~0.3-0.4GB
  × 5, about 1.5-2GB) is freed, and disk use drops from N clusters to one.
  The RAM win is larger for the 10-shard Arm-A default (~3-4GB of
  postmaster freed for one shared server). `hindsight_state` shrinks to
  just the HF cache, so the old ~8GB `docker_hindsight_state` volume
  becomes reclaimable with `docker volume rm docker_hindsight_state`. Only
  remove it once no hindsight container references it. Do not remove a
  volume that is in use.
- **Supported modes: `shared` and `pg0`.** The legacy `local` and
  `embedded` values were removed on 2026-07-22 and the entrypoint still
  fails hard on them, and on any other unknown value. `pg0` was added on
  2026-07-31 for the featured clock-sync arm only, a per-container
  embedded cluster inside the faked clock domain, described under
  "Clock sync" below. `shared` remains the default and the only mode the
  minimal arm uses.
- **Further RAM savings.** The shared `vllm-embed` can now serve the
  embedding model remotely, instead of each daemon loading it. See the next
  subsection.

### Remote embeddings + reranker (`HINDSIGHT_EMBED_SOURCE` / `HINDSIGHT_RERANK_SOURCE`)

Without this, a Hindsight daemon would load two local CPU models: the
`BAAI/bge-small-en-v1.5` embedder and the `cross-encoder/ms-marco-MiniLM-L-6-v2`
reranker, plus the `torch` and `sentence-transformers` runtime they share.
Both models now always run on shared servers: `HINDSIGHT_EMBED_SOURCE` and
`HINDSIGHT_RERANK_SOURCE` accept only `remote`. We removed the legacy
per-daemon local-model mode for both on 2026-07-22, because every real run
already used `remote`. The entrypoint fails hard on any other value, and
git history has the old branches.

`HINDSIGHT_EMBED_SOURCE=remote` points the embedder at the already-running
shared `vllm-embed`, which serves the contract embedder every other provider
uses (`gte-modernbert-base`, dim 768). The entrypoint exports:

```
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=http://vllm-embed:8000/v1
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=gte-modernbert-base
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=local-vllm   # dummy; vLLM ignores it
```

`HINDSIGHT_RERANK_SOURCE=remote` points the reranker at a shared
`hindsight-rerank` server, a HuggingFace text-embeddings-inference (TEI)
container that serves the same ms-marco cross-encoder over its `/rerank`
endpoint. `vllm-embed` cannot serve a cross-encoder, so this is a separate
small service. It has used GPU since 2026-07-22 (see the compose comment).
The entrypoint exports:

```
HINDSIGHT_API_RERANKER_PROVIDER=tei
HINDSIGHT_API_RERANKER_TEI_URL=http://hindsight-rerank:80
```

Both features use hard provider dispatch in `hindsight_api`
(`create_embeddings_from_env` maps to `OpenAIEmbeddings`, and
`create_cross_encoder_from_env` maps to `RemoteTEICrossEncoder`, in
`hindsight_api/engine/{embeddings.py,cross_encoder.py}`). The local model
classes are never constructed. Notes:

- **RAM.** `torch` and `sentence-transformers` (~430 MB resident) load only
  when the embedder or the reranker runs locally (`config.py:2225` gates
  the `sentence_transformers` import on `provider == "local"`, and every
  `torch` import in the package is lazy and provider-gated). Because both
  now always run `remote`, every daemon gets this RAM saving. Importing the
  full `memory_engine` leaves `torch`/`sentence_transformers` unloaded (the
  default `DateparserQueryAnalyzer` is pure Python), which drops ~0.6 GB
  static usage (and more under load) per daemon, compared with the removed
  local-model mode (a live local-mode daemon ran ~1.05 GB RSS with
  `libtorch_cpu.so` mapped).
- **Numerics.** The reranker change swaps the serving location, not the
  model. The embedder change is a serving-envelope change: `vllm-embed`
  serves the shared contract embedder, so Hindsight uses the same embedder as
  every other provider instead of the vendor's local bge-small. The embedding
  dimension auto-detects to 768 at startup. Run manifests record the
  `HINDSIGHT_API_*` env snapshot, which lets an audit confirm which mode
  ran. Do not set `HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS`: leave it
  unset so the request omits the field and auto-detect runs. Under contract
  v4's bge-small, vLLM answered a set `dimensions` with `400 "does not
  support matryoshka"`.
- **Starting the TEI server.** `hindsight-rerank` sits under the `run`
  profile. It is deliberately absent from `hindsight`'s `depends_on`, so it
  never starts on its own (compose cannot make a `depends_on` conditional
  on an env value). Before a remote-rerank run, start it explicitly and
  wait for it to become healthy: `docker compose --profile run up -d
  hindsight-rerank` (the `run` profile flag is required, since a bare `up -d`
  silently skips it). If you skip this step, the daemon dies after a 179 s
  wait with `RuntimeError: Failed to connect to TEI server at
  http://hindsight-rerank:80: [Errno -2] Name or service not known`, then
  `Failed to start daemon for profile '<profile>'` from `Setup_Hindsight` →
  `client._ensure_started()`. Nothing else in the log names the reranker as
  the cause. Image `ghcr.io/huggingface/text-embeddings-inference:120-1.9`
  (the first TEI tag with sm_120 kernels, still experimental), run with
  `--model-id cross-encoder/ms-marco-MiniLM-L-6-v2 --port 80 --dtype
  float16`. TEI has no `--gpu-memory-utilization` flag, so it takes what the
  two vLLM services leave: `vllm-gen` at 0.85 plus `vllm-embed` at 0.07 leaves
  ~1.2 GiB on the 16 GiB card for TEI and driver overhead. Confirm with
  `nvidia-smi` at boot.
  The cpu-1.8 image it replaced queued rerank requests for 5-10 s
  under full-run load. The model cache lives in a throwaway
  `hindsight_rerank_cache` volume.

`hindsight-pg` also sits under the `run` profile, so a bare
`docker compose up -d` still starts only the vLLM servers. `docker compose
run hindsight` brings it up through `depends_on: { condition:
service_healthy }`. No host port is published, so the database is
reachable only on the compose network. Override `HINDSIGHT_PG_*` to point
at an external Postgres instead.

## RetainDB (scaffold)

`retaindb` is still wired into the compose file, but its image has never
run in Docker. We smoke-tested the adapter on the host, against OpenRouter
`gpt-oss-120b`, instead. `Dockerfile.retaindb` installs the documented
dependencies (Node 22 plus `@retaindb/local` through `npm ci`), with `TODO`
markers where in-container behavior still needs verification. Its
answer/judge LLM defaults to the shared `vllm-gen`. For the verified
OpenRouter path, override `OPENAI_BASE_URL` to
`https://openrouter.ai/api/v1` and pass `-e OPENROUTER_API_KEY=...`. Expect
one or two rounds of image tuning before it runs end-to-end.

## RetainDB server edition (scaffold)

`retaindb-server` benchmarks `@retaindb/server` (Postgres/pgvector plus LLM
extraction), a different product from the `retaindb` local edition. The
image builds the server from the `external/RetainDB` submodule. At
container start, the entrypoint derives a per-run database on the shared
`hindsight-pg`, runs `prisma migrate deploy`, launches `node dist/index.js`,
waits for it to become healthy, then dispatches `STAGE` (the Python adapter
only attaches to the server). Sharding follows Hindsight's model: one
container, one server, and one per-run database per shard, split by
`START_IDX`/`END_IDX` with a distinct `RUN_TAG`.

Embeddings use the contract embedder, the shared `vllm-embed`
(`gte-modernbert-base`, 768-dim, the same embedder every other provider
uses), through `retaindb_server/embed_proxy.py`. This proxy translates
RetainDB's own `{inputs}→{embeddings}` protocol to vLLM's OpenAI
`/v1/embeddings`, and right-pads to 1024 dimensions to fit RetainDB's
`vector(1024)` schema. The pad adds 256 zeros under v5 (640 under contract
v4's 384-dim bge-small), which preserves the vector norm and leaves ranking
unchanged. This setup adds no extra GPU service and no
GPU-budget change. The extraction LLM talks to `vllm-gen` directly.
`RETAINDB_EMBEDDING_MODE=local` is an off-contract CPU debug knob only. It
uses a different embedder, so its numbers are not comparable.

```bash
docker compose up -d vllm-gen vllm-embed                 # contract embedder pulled up via depends_on anyway
docker compose run -d --rm -e NUM_PERSONAS=1 -e MAX_SESSIONS=6 -e RUN_TAG=smoke retaindb-server
# sharded full-run generate (each shard: own container/server/db):
docker compose run -d --name rds_s0 -e RUN_TAG=full_s0 -e STAGE=generate \
  -e START_IDX=0 -e END_IDX=8 -e NUM_PERSONAS=30 retaindb-server
```

Full env inventory and the host (OpenRouter) smoke are in
`../../docs/DECISIONS.md`.

## Mnemosyne run-service control knobs (all via env)

Pass these with `docker compose run -d --rm -e VAR=value mnemosyne`.
(`hindsight` and the `retaindb` scaffold also honor the generic subset:
`STAGE`, `NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`,
`MAX_QUESTIONS_PER_SESSION`, `SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` /
`OPENAI_MODEL`, plus their own provider vars. Hindsight's vars are in the
next section.)

| Var | Default | Meaning |
|---|---|---|
| `STAGE` | `all` | `generate` \| `score` \| `summarize` \| `all` |
| `NUM_PERSONAS` | `30` | Personas to run |
| `NUM_SHARDS` | = personas | Concurrent generation processes (one per persona = max granularity) |
| `MAX_SESSIONS` | all | Caps the dialogue sessions ingested per persona. It truncates to the first N sessions (note: early sessions are question-light, so small values answer few or zero questions) |
| `MAX_QUESTIONS_PER_SESSION` | all | Cap questions answered per session (smoke testing) |
| `TOP_K` | `5` | Retrieved memories the answer LLM sees |
| `SCORE_WORKERS` | `24` | Concurrent judge requests during scoring |
| `VLLM_JUDGE_MAX_LEN` | `32768` | Context window for the second judge (`vllm-judge`, gemma-4-12b). vLLM subtracts `max_tokens` from the window, so the default leaves 16,384 tokens of input. Honcho's featured wave needs `49152`, see DECISIONS "The Honcho wave alone is judged at a 49,152-token window". Raising it shrinks concurrency against a fixed KV budget, so lower the judge worker count to match. Takes effect only on `up -d --force-recreate` |
| `PROGRESS_INTERVAL` | `60` | Seconds between shard-progress lines on the container stdout (`docker logs`). `0` turns this off |
| `EXTRACT` | `0` | `1` turns on `--extract` (Mnemosyne LLM fact-extraction on ingest). Facts feed the veracity conflict detector and recall's always-on fact-index boost. The fact TEXT surfaces only when `MNEMOSYNE_FACT_RECALL_ENABLED=1` |
| `MNEMOSYNE_FACT_RECALL_ENABLED` | `0` | Merges extracted-fact rows into recall results. This is off by plugin default (the Hermes plugin never sets this env var, and its README documents the default as false). A v1 probe with this ON showed the fact rows scored worse under the judge (SEH@3 0.031). We keep this off for plugin fidelity, not to satisfy the scorer's format |
| `MNEMOSYNE_ENHANCED_RECALL` | `0` | `1` → enhanced recall pipeline |
| `LIFECYCLE` | `0` | `1` turns on `--lifecycle`: dataset-timestamp restoration, per-session surgical veracity retirement, and `Retirement_Diagnostics` (arm B in `../../docs/BENCHMARK_MATRIX.md`). This implies extraction. It also auto-raises `MNEMOSYNE_WM_TTL_HOURS` and sets `MNEMOSYNE_EP_LIMIT=0` |
| `CANONICAL` | `0` | `1` turns on `--canonical` (arm C): lifecycle, plus per-session `sleep(force=True)` (model-refresh populates canonical slots), plus history-aware canonical retrieval. This auto-raises `MNEMOSYNE_LLM_MAX_TOKENS` to 3072 (512 truncates the model-refresh JSON) and relaxes the model-refresh evidence gates |
| `ORACLE` | `0` | `1` turns on `--oracle` (arm D, the upper bound): lifecycle, plus canonical slots derived from dataset gold (`mnemosyne/oracle_canonical.py`), plus history-aware canonical retrieval. This is mutually exclusive with `CANONICAL` |
| `BENCH_SQLITE_FAST` | `1` | Sets `synchronous=OFF`, WAL, and a big cache/mmap on every sqlite connection (disposable benchmark databases. This fixes 30-shard fsync thrash). `0` restores stock behavior, for an A/B comparison |
| `THINKING` | `1` | Controls canonical answer decoding (identical for every provider, routed through `answer_env.sh`). `1` turns on Gemma-4's native thinking for answer generation: the model reasons in a private `<\|channel>thought` block, which the server's `--reasoning-parser gemma4` strips out, so the answer stays clean. This setting auto-suppresses on the JSON-mode judge. The answer token budget is 3072. `0` turns thinking off and sets the budget to 1024 |
| `OPENAI_BASE_URL` / `OPENAI_MODEL` | `vllm-gen` / `qwen3.5-4b` | Answer+judge server / model |
| `MNEMOSYNE_EMBEDDING_API_URL` / `MNEMOSYNE_EMBEDDING_MODEL` / `_DIM` | `vllm-embed` / `gte-modernbert-base` / `768` | Embedding server / model / dim. Contract v4 ran `bge-small-en-v1.5` / `384` |
| `MNEMOSYNE_LLM_BASE_URL` / `MNEMOSYNE_LLM_MODEL` | `vllm-gen` / `qwen3.5-4b` | Mnemosyne-internal LLM (only used with `--extract`/conflict detection) |
| `RUN_TAG` / `RESULTS_FILE` | auto | Override output naming/paths |

### Output tagging

The system auto-tags outputs, so runs never overwrite each other or the
committed baseline (`mnemosyne_results.jsonl`):

- Baseline (no features): a new run writes
  `mnemosyne/Results/mnemosyne_results_local.jsonl`,
  `mnemosyne/Scores/mnemosyne_local_eval_scores.jsonl`, and
  `mnemosyne/Scores/summary_local.json`. (We have since filed the committed
  v1 baseline artifacts under `mnemosyne/{Results,Scores}/v1/`, see
  BENCHMARK_MATRIX "Contract envelopes". New runs still land at the
  `Results/`/`Scores/` root.)
- `CANONICAL=1` tags the run `_canonical`. `ORACLE=1` tags it `_oracle`.
  Other features (`EXTRACT` / `MNEMOSYNE_ENHANCED_RECALL` / `LIFECYCLE`)
  tag it `_feature`.

Set `RUN_TAG=foo` to force a custom tag, or `RESULTS_FILE=...` for a fully
explicit path. Results/Scores land on the host via the `mnemosyne/`
bind-mount.

## Hindsight run-service control knobs (all via env)

Pass these with `docker compose run -d --rm -e VAR=value hindsight`. This
service also honors the generic subset shared with Mnemosyne above
(`STAGE`, `NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`,
`MAX_QUESTIONS_PER_SESSION`, `SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` /
`OPENAI_MODEL`).

| Var | Default | Meaning |
|---|---|---|
| `START_IDX` / `END_IDX` | `0` / `NUM_PERSONAS` | Sets the persona index range `[START_IDX, END_IDX)` this container evaluates. This is the sharding mechanism for a full run: launch several containers with disjoint ranges and distinct `RUN_TAG`s |
| `HINDSIGHT_LLM_BASE_URL` / `HINDSIGHT_LLM_MODEL` | `vllm-gen` / `qwen3.5-4b` | Hindsight-internal LLM (retain fact-extraction + recall query understanding) |
| `HINDSIGHT_API_LLM_STRICT_SCHEMA` | `1` | Controls grammar-enforced structured output. The compose default targets the OpenRouter/gpt-oss path. Local-vLLM runs pass `0`, because vLLM's server accepts but does not enforce `response_format` grammars (a gemma4 reasoning-parser interaction). Strict mode was always only a gpt-oss/OpenRouter workaround. See `../../docs/TROUBLESHOOTING.md` |
| `HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION` | `false` | Switches between arm A and arms B/C. `false` selects arm A (extraction-only). `true`, with the flags below, selects arm B or C |
| `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS` | `8192` | Output-token cap for retain's extraction LLM call. A runaway bound, not a capacity bound: legitimate completions max at 1,863 tokens. Hindsight's own default 64000 exceeds the contract v4 window (32768) and the server rejects it. Must stay above `retain_chunk_size` (3000) |
| `HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES` | `7` | Retry attempts for the retain LLM call. This was an ad hoc `-e` override on early shard runs, and is now a documented compose default. Gemma, served through `vllm-gen`, needs more retries than the package default (`3`) to survive occasional malformed-JSON extraction attempts |
| `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET` / `_SOURCE_FACTS_MAX_TOKENS` / `_MAX_COMPLETION_TOKENS` | unset → package default | Arm B/C consolidation bounds (the same context-window reasoning as the retain cap above). These are unset by default, because arm A does not consolidate. Pass explicit values on arm B/C runs (recommended pair: `SOURCE_FACTS_MAX_TOKENS=4096`, `MAX_COMPLETION_TOKENS=4096`). The entrypoint unsets any of these left as an empty string before running, because `HindsightConfig.from_env()` applies its own default only when the var is fully absent, not present but empty |
| `PREFER_OBSERVATIONS` | unset (off) | `1` turns on `--prefer_observations` (arm B/C): consolidated observations replace their raw source facts in `recall()` results, instead of both appearing |
| `WAIT_CONSOLIDATION` | unset (off) | `1` turns on `--wait_consolidation` (arm B/C): block after each session's retain until that session's consolidation ops drain. This wait is snapshot-scoped, so a pre-existing stuck op cannot block it. This way, questions see fully-consolidated memory |
| `RETAIN_GRANULARITY` | unset → `session` | `exchange` (arm C) makes one `retain()` call per user-and-assistant exchange pair, matching the official Hindsight-Hermes plugin's `post_llm_call`. `message` makes one call per individual message. This is much slower, and mirrors the Mnemosyne adapter's per-message mode |
| `HINDSIGHT_PG_MODE` | `shared` | `shared` points the daemon at the `hindsight-pg` service. `pg0` runs a per-container embedded PostgreSQL cluster instead, for the featured clock-sync arm: the `exchange_append` merge takes `mentioned_at` from the DB clock, so that arm needs a cluster it can clock-fake. `pg0` is gated: it requires `BENCH_CLOCKSYNC=1`, requires `END_IDX - START_IDX == 1`, and exits 2 on a set `HINDSIGHT_API_DATABASE_URL`. Any other value exits 2 |
| `ALLOW_EXISTING_PG0` | `0` | The pg0 equivalent of `ALLOW_EXISTING_DB`. Generate refuses to start when `$HOME/.pg0/instances` already exists, because a relaunch under the same `RUN_TAG` would ingest into the previous run's banks. `1` reuses them, and the prior run's rows stay in the store |
| `HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT` | `0` in the `pg0` branch, otherwise unset | `0` disables the daemon's idle auto-exit (`daemon.py:59` returns before the checker loop at `<= 0`). The loop compares `time.time()`, which libfaketime fakes, so a faked forward jump of weeks between sessions would otherwise read as idleness and kill the daemon mid-run. The vendor docstring at `cli.py:20` claims "default: 300" and is stale. The code default is 0. **Never declare this key in compose.** `daemon_embed_manager.py:495` int-parses it, and the entrypoint's empty-var guard matches `HINDSIGHT_API_` only, so a compose-set empty value crashes the boot |

Results and scores land on the host through the `hindsight/` bind-mount,
tagged the same way as Mnemosyne
(`hindsight/Results/hindsight_results_<tag>.jsonl`, and so on, where
`<tag>` is `RUN_TAG` or `hindsight` by default).

## mem0 run-service control knobs (all via env)

Pass these with `docker compose run -d --rm -e VAR=value mem0`. This service
also honors the generic subset shared with Mnemosyne above (`STAGE`,
`NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`, `MAX_QUESTIONS_PER_SESSION`,
`SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` / `OPENAI_MODEL`). mem0 is fully
self-hosted: an internal extraction LLM, an embedder, and a qdrant vector
store. The `qdrant` service must be up first (it is in `depends_on`, so
`docker compose run mem0` starts it). The `mem0/` folder name shadows the SDK,
so run the adapter as `python mem0/eval_mem0.py`, never `python -m
mem0.eval_mem0`.

| Var | Default | Meaning |
|---|---|---|
| `START_IDX` / `END_IDX` | `0` / `NUM_PERSONAS` | Persona index range `[START_IDX, END_IDX)` for this shard. Empty covers `[0, NUM_PERSONAS)`. `run_shards.sh mem0 <tag>` sets these per shard |
| `MEM0_LLM_PROVIDER` / `MEM0_LLM_MODEL` / `MEM0_LLM_BASE_URL` | `openai` / `qwen3.5-4b` / `vllm-gen` | mem0-internal extraction LLM. Under the `mem0ai==2.0.14` pin this is fact extraction ONLY (the product is ADD-only; the 0.1.x UPDATE/DELETE decision path is dead code) |
| `MEM0_LLM_TEMPERATURE` / `MEM0_LLM_MAX_TOKENS` | `0.7` / `2048` | Internal extraction LLM sampling and output budget |
| `MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` / `MEM0_EMBEDDER_BASE_URL` / `MEM0_EMBEDDER_DIMS` | `openai` / `gte-modernbert-base` / `vllm-embed` / `768` | Embedder, the shared `vllm-embed` (contract v4 ran `bge-small-en-v1.5` / `384`). `MEM0_EMBEDDER_PROVIDER=huggingface` selects the in-image `sentence-transformers` fallback |
| `MEM0_EMBED_TRUNCATE_TOKENS` | `-1` | Per-input cap the adapter's embed shim sends as vLLM's `truncate_prompt_tokens`. `-1` truncates to the served model's own `max_model_len`, so it tracks `VLLM_EMBED_MAX_LEN` (8192) with no second knob |
| `RETAIN_GRANULARITY` | `batch` | `batch` (one `add()` per `MEM0_ADD_BATCH_SIZE`-message window, the MemConflict authors' cadence) \| `session` (one `add()` per session) \| `exchange` (one `add()` per turn, the plugin-faithful cadence, more internal LLM calls) |
| `MEM0_ADD_BATCH_SIZE` | `8` | Window size for `batch` granularity. `mem0_minimal_clocksync` sets 6 |
| `MEM0_VECTOR_MODE` | `server` | `server` connects every shard to the shared `qdrant` service, each shard in its own collection. `embedded` uses a per-process on-disk qdrant (host-smoke behavior, not shareable across shards, its on-disk `path` locks to one process) |
| `MEM0_QDRANT_HOST` / `MEM0_QDRANT_PORT` | `qdrant` / `6333` | Shared qdrant address (`server` mode) |
| `MEM0_QDRANT_ON_DISK` | `false` | `false` keeps vectors in RAM (faster, disposable scratch); `true` stores them on disk in the server |
| `MEM0_COLLECTION` | derived from `RUN_TAG` | Explicit collection override. Empty derives `mem0_<sanitized RUN_TAG>` |
| `MEM0_TELEMETRY` | `False` | Off by default: mem0's PostHog events 403 through the egress proxy and flood stderr |
| `OPENROUTER_API_KEY` | empty | For the OpenRouter path, set `OPENAI_BASE_URL` / `MEM0_LLM_BASE_URL` to `https://openrouter.ai/api/v1` and pass this key |

Results and scores land on the host through the `mem0/` bind-mount, tagged the
same way as Mnemosyne (`mem0/Results/mem0_results_<tag>.jsonl`, and so on,
where `<tag>` is `RUN_TAG` or `mem0` by default).

## Supermemory run-service control knobs (all via env)

Pass these with `docker compose run -d --rm -e VAR=value supermemory`. This
service also honors the generic subset shared with Mnemosyne above (`STAGE`,
`NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`, `MAX_QUESTIONS_PER_SESSION`,
`SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` / `OPENAI_MODEL`). Supermemory is
a native server binary the adapter drives over REST. In the default `shared`
mode every shard attaches to one central `supermemory-server` (in `depends_on`,
so `docker compose run supermemory` starts it), isolated by containerTag
namespace. The server's own throughput knob, `SUPERMEMORY_INGEST_CONCURRENCY`
(default `15`), lives on the `supermemory-server` service, not the shard, and a
change takes effect only on a server RECREATE.

| Var | Default | Meaning |
|---|---|---|
| `SUPERMEMORY_SERVER_MODE` | `shared` | `shared` attaches to the central `supermemory-server`. `spawn` runs a standalone disposable server inside the shard; pair it with `docker compose run --no-deps` |
| `SUPERMEMORY_SERVER_HOST` / `SUPERMEMORY_SERVER_PORT` | `supermemory-server` / `8787` | Central server address (`shared` mode) |
| `SUPERMEMORY_CONTAINER_NAMESPACE` | derived from `RUN_TAG` | Per-run containerTag namespace. Empty derives it from `RUN_TAG` with a trailing `_s<k>` stripped, so all shards of a run share one prefix. Use a fresh `RUN_TAG` per wave |
| `SUPERMEMORY_ATTACH_URL` / `SUPERMEMORY_KEY_FILE` | empty | Two-server topology only: attach a shard to server A (`/shared/api_key`) or server B (`/shared_b/api_key`). `run_shards.sh` sets these when `SUPERMEMORY_TWO_SERVERS=1` |
| `SUPERMEMORY_LLM_API_KEY` / `SUPERMEMORY_LLM_BASE_URL` / `SUPERMEMORY_LLM_MODEL` | empty | Supermemory's INTERNAL extraction LLM (server-side). Empty defaults it to the shared answer endpoint/model. Override to drive extraction with a different model without touching the fairness-locked answer/judge config |
| `SUPERMEMORY_EMBEDDING_PROVIDER` / `_MODEL` / `_DIMENSIONS` / `_BASE_URL` / `_API_KEY` | `openai` / `gte-modernbert-base` / `768` / `vllm-embed` / `local-vllm` | Embedder. `PROVIDER=local` (blank the rest) restores the in-process Xenova bge-base fallback. Store data is dimension-bound: never mix providers in one `supermemory_data` volume |
| `SUPERMEMORY_SEARCH_MODE` | `hybrid` | The Hermes plugin default (`/v4/search`, memories first with a doc-chunk fallback). Alternate: `memories` |
| `SUPERMEMORY_SEARCH_THRESHOLD` | empty (0.0, OFF) | The `/v4/search` threshold. Empty means the explicit 0.0 the plugin uses (the vendor default 0.6 admits fewer memories than the shared top-K). Set `0.6` to reproduce the vendor default as an arm |
| `SUPERMEMORY_INGEST_ENDPOINT` / `SUPERMEMORY_RECALL_ENDPOINT` | empty (minimal arm) | Empty selects the minimal arm (`/v3/documents` + `/v4/search`). The featured run sets `conversations` + `profile` to drive the exact plugin endpoints (`/v4/conversations` ingest, `/v4/profile` auto-recall) |
| `SUPERMEMORY_DOCUMENTS_ARM` | `0` | `1` is the diagnostic `/v3/search` document-chunk arm |
| `SUPERMEMORY_RERANK` / `SUPERMEMORY_REWRITE_QUERY` | `0` / `0` | Server-side rerank and query-rewrite toggles |
| `SUPERMEMORY_RETAIN_GRANULARITY` | `session` | `session` (full-session document, the plugin's session-end ingest) \| `exchange` (per-turn) \| `message` |
| `SUPERMEMORY_DRAIN_TIMEOUT` | `600` | Max seconds to wait for a session's async ingest to reach `done` before answering, so recall never races the queue |
| `SUPERMEMORY_STRICT_QUALITY` | `0` | `1` aborts the shard nonzero on any drain timeout or extraction failure instead of answering against missing memories. Off for smokes |
| `SUPERMEMORY_RESPAWN_PER_SESSION` | `1` | Clock-sync arms only (ignored without `BENCH_CLOCKSYNC=1`): restart the spawned server once per session to stop node-cron v4 replaying every missed 30-minute slot and OOMing the host. `0` reproduces the OOM |
| `SUPERMEMORY_EMBEDDING_RAM_LIMIT` | `8gb` | Spawn-mode ingest-memory backpressure watermark. The vendor default `1gb` idled the GPU for ~8-minute stretches. Shared mode gets this from the `supermemory-server` service instead |
| `OPENROUTER_API_KEY` | empty | For the OpenRouter path, set `OPENAI_BASE_URL` to `https://openrouter.ai/api/v1` and pass this key |

The binary ships pinned at `SUPERMEMORY_SERVER_VERSION` 0.0.5 in
`Dockerfile.supermemory` (0.0.6 and 0.0.7-rc.2 ship a broken linux-x64 ingest
engine). Results and scores land on the host through the `supermemory/`
bind-mount, tagged the same way as Mnemosyne
(`supermemory/Results/supermemory_results_<tag>.jsonl`, and so on, where
`<tag>` is `RUN_TAG` or `supermemory` by default).

## Honcho run-service control knobs (all via env)

Pass these with `docker compose run -d --rm -e VAR=value honcho`. This
service also honors the generic subset shared with Mnemosyne above
(`STAGE`, `NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`,
`MAX_QUESTIONS_PER_SESSION`, `SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` /
`OPENAI_MODEL`).

Honcho is a server product: a FastAPI `honcho-api` process plus a separate
`honcho-deriver` worker, both backed by the shared `honcho-pg` Postgres
(pgvector). `honcho-db-init` runs once, before either: it provisions the
schema (`scripts/provision_db.py`) and then retypes every `vector` column from
the vendor's hardcoded 1536 dims to the shared embedder's 768 (the migrations
never read `EMBEDDING_VECTOR_DIMENSIONS`; only a fresh `create_all()` would).
Stores are dimension-bound, so a contract v4 store (384) cannot be reused
under v5. Reprovision.
`HONCHO_SERVER_MODE` picks how the `honcho` run-service reaches this server:

| Var | Default | Meaning |
|---|---|---|
| `HONCHO_SERVER_MODE` | `shared` | `shared` attaches to the compose `honcho-api`/`honcho-deriver` services over REST (the `hindsight-pg` / `supermemory-server` central-service analog). `spawn` launches a disposable API+deriver (plus, under `BENCH_CLOCKSYNC=1`, its own in-container Postgres) inside this container instead. Pair it with `docker compose run --no-deps`, since compose has no conditional `depends_on` and this service always lists `honcho-api` as a dependency for the `shared` default |
| `HONCHO_BASE_URL` | `http://honcho-api:8000` (shared mode) / unset (spawn mode) | Honcho server URL. The entrypoint sets this only in `shared` mode |
| `HONCHO_API_KEY` | `local` | Bearer key. `AUTH_USE_AUTH=false` on the server, so any non-empty value is accepted |
| `HONCHO_TIMEOUT` | `300` | SDK request timeout (seconds). **Not the plugin's own 30s.** The adapter calls the dialectic inline over this same timeout (`eval_honcho.py --timeout`), and 30s can silently empty the dialectic layer of `hybrid` recall on slow serving: the call just times out and that section drops from the memory-context block with no error surfaced anywhere in the run. 300 matches the harness's own answer-path request timeout |
| `HONCHO_SERVER_DIR` | `/app/external/honcho` (in-container) | Spawn mode only: the uv-synced vendored Honcho checkout `honcho/_honcho_server.py` execs `uvicorn`/the deriver against |
| `HONCHO_SERVER_PORT` | `0` (ephemeral) | Spawn mode only: fixed port instead of an OS-assigned one |
| `HONCHO_SERVER_PYTHON` | unset → `<HONCHO_SERVER_DIR>/.venv/bin/python` if present | Spawn mode only: interpreter override for the spawned API/deriver children. The default resolves the uv-synced venv this image bakes at build time, and reports the missing venv instead of silently falling back to the harness's own interpreter (which lacks the server's dependency set) |
| `HONCHO_RUN_DIR` | unset → `.honcho_runs/<RUN_TAG>` next to the adapter | Spawn mode only: where the spawned API/deriver child logs (`api.log`, `deriver.log`) and any in-container Postgres data dir land. Gitignored (`honcho/.honcho_runs/`) |
| `HONCHO_PG_DSN` | `postgresql+psycopg://postgres:postgres@honcho-pg:5432/postgres` | Postgres connection string. Shared mode: read by `honcho-pg`/`honcho-db-init`/`honcho-api`/`honcho-deriver`. Spawn mode: read by the adapter's own in-container cluster once `HONCHO_PG_CREATE_DB=1` creates it (adapter default `postgresql+psycopg://postgres:postgres@localhost:5432/postgres`) |
| `HONCHO_PG_CREATE_DB` | `0` | Spawn mode only: `1` creates a per-run database on the in-container Postgres before migrating it |
| `HONCHO_PG_DB` | unset → `honcho_<sanitized RUN_TAG>` | Spawn mode only: explicit database name override for `HONCHO_PG_CREATE_DB=1` |
| `HONCHO_PG_DROP_DB` | unset → vendor default `true` | Spawn mode only: drop-and-recreate the per-run database on `HONCHO_PG_CREATE_DB=1` instead of reusing an existing one |
| `HONCHO_DB_SCHEMA` | `public` | Postgres schema. Read by `honcho-db-init`, `honcho-api`, `honcho-deriver`, and the adapter's own spawn-mode child processes |
| `HONCHO_WORKSPACE_PREFIX` | derived from `RUN_TAG` | Every persona's Honcho workspace is `<prefix>p<idx>_<sanitized persona id>`. Empty → the entrypoint derives `hermes_<sanitized RUN_TAG>_`, the same way `entrypoint.mem0.sh` derives `MEM0_COLLECTION` |
| `HONCHO_USER_PEER_ID` / `HONCHO_AI_PEER_ID` | `user` / `hermes` | Peer ids inside each persona's workspace |
| `HONCHO_OBSERVATION_MODE` | `directional` | `directional` (plugin default: self AND cross observation, both `observe_me`/`observe_others` true) \| `unified` (one shared collection per peer pair, no self/cross split, pairs with the `conclusions` recall arm, which never reads the directional split) |
| `HONCHO_RECALL_MODE` | `hybrid` | `hybrid` (headline; plugin `recallMode` default: session summary + user representation/card + dialectic, assembled into one memory-context block) \| `base` (summary + representation/card only, no dialectic call) \| `dialectic` (`peer.chat()` only) \| `search` (diagnostic `h.search()`, RRF-ranked raw messages, never a headline number) \| `conclusions` (minimal arm: raw stored conclusions, no summary/peer-card section) |
| `HONCHO_SUMMARY_ENABLED` / `HONCHO_PEER_CARD_ENABLED` | `1` / `1` | Server-side `SUMMARY_ENABLED` / `PEER_CARD_ENABLED`. The minimal preset turns both off, because `HONCHO_RECALL_MODE=conclusions` never reads either section, and leaving them on would still spend internal-LLM calls building them |
| `HONCHO_DIALECTIC_REASONING_LEVEL` | `low` | Base reasoning tier for `peer.chat()` (`minimal`\|`low`\|`medium`\|`high`\|`max`) |
| `HONCHO_DIALECTIC_DYNAMIC` | `1` | Plugin-faithful dynamic scaling: `+1` level at query length ≥120 chars, `+2` at ≥400, capped at `HONCHO_REASONING_LEVEL_CAP` |
| `HONCHO_REASONING_LEVEL_CAP` | `high` | Ceiling for the dynamic scaling above |
| `HONCHO_DIALECTIC_MAX_CHARS` | `600` | Dialectic result clip length (plugin-style, word-boundary + " …") |
| `HONCHO_DIALECTIC_MAX_INPUT_CHARS` | `10000` | Dialectic query truncation length. This clips the adapter's QUERY text only; it does not touch Honcho's internal prompt assembly. That is the next row |
| `HONCHO_DIALECTIC_MAX_INPUT_TOKENS` | `20000` | `DIALECTIC_MAX_INPUT_TOKENS`, the vendor bound on the whole dialectic prompt. Its default 100000 (`src/config.py:936`) assumes a 128k-window model, which contract v4's 32768 window is not. Honcho truncates to this bound rather than erroring (`truncate_messages_to_fit`, `src/llm/api.py:333-341`), but only when it is set below the window. Left at the default, every dialectic call 400s once the representation passes about 24,577 tokens, which empties layer 2 of the hybrid arm. Set in both the run-service (spawn) and the `honcho-api` anchor env |
| `HONCHO_MESSAGE_MAX_CHARS` | `25000` | Per-message ingest chunk size (plugin's `messageMaxChars`, with a `"[continued] "` prefix on overflow chunks) |
| `HONCHO_SEARCH_LIMIT` | `10` | `h.search()` result cap, `HONCHO_RECALL_MODE=search` only |
| `HONCHO_CONTEXT_TOKENS` | `8192` | Token budget for the assembled injection block, a port of the plugin's `_truncate_to_budget` (`__init__.py:870-883`): cut to `tokens × 4` chars, keep the word-boundary cut only past 80% of the budget, append `" …"`. Applied to the final joined block of BOTH recall paths, after the top-K slice. The plugin SHIPS `contextTokens` unset, which `_parse_context_tokens` (`client.py:145-153`) reads as uncapped, unrunnable here, because an uncapped hybrid block measured 254k tokens at persona 0 session 5, past even the contract v5 131072 window. 8192 is half the budget left after the shared answer role's 16384 under contract v4's 32768 window. `0` restores the shipped uncapped behavior. The per-persona summary records `Context_Truncated_Questions` and `Context_Items_Dropped` |
| `HONCHO_SEND_CREATED_AT` | `0` | `1` sends the session's dataset timestamp as each message's `created_at` (a vendor-exposed capability the plugin itself does not use; the plugin-faithful temporal path is `BENCH_CLOCKSYNC` instead) |
| `HONCHO_DRAIN_TIMEOUT_S` / `HONCHO_DRAIN_POLL_S` | `1800` / `2.0` | After each session's ingest, the adapter polls `h.queue_status()` until the deriver queue drains (a deviation from the plugin's fire-and-forget write, see the design notes), instead of racing recall against ingestion. Raise the timeout on a slow shared server under multi-shard load |
| `HONCHO_DERIVER_FLUSH` | `1` | `DERIVER_FLUSH_ENABLED`: process incoming messages immediately instead of batching to a token/age threshold (`DERIVER_REPRESENTATION_BATCH_MAX_TOKENS`, vendor default 1024), so the drain above converges in bounded time per session |
| `HONCHO_DERIVER_WORKERS` | `4` | `DERIVER_WORKERS`: deriver queue-consumer concurrency |
| `HONCHO_DERIVER_POLL_S` / `HONCHO_DERIVER_POLL_MAX_S` | `1.0` / `2.0` | `DERIVER_POLLING_SLEEP_INTERVAL_SECONDS` / `_SLEEP_MAX_INTERVAL_SECONDS`: idle-poll backoff floor/ceiling for the deriver's queue consumer loop |
| `HONCHO_DERIVER_STARTUP_JITTER_S` | `0.0` | `DERIVER_POLLING_STARTUP_JITTER_SECONDS`. The vendor default jitters the first poll by up to 30s so co-started peer instances do not poll in lockstep. This topology runs exactly one deriver, so that jitter is pure added latency on every drain. Set it to `0.0` here and in the `honcho-api`/`honcho-deriver` anchor env |
| `HONCHO_LLM_MODEL` / `HONCHO_LLM_BASE_URL` / `HONCHO_LLM_API_KEY` | `qwen3.5-4b` / `vllm-gen` / falls back to `OPENAI_API_KEY` | Honcho-internal LLM (deriver extraction, all five dialectic reasoning levels, session summary, dream deduction/induction). One model for every internal role, per the best-effort ruling |
| `HONCHO_LLM_MAX_OUTPUT_TOKENS` | `8192` | Output budget for every internal role. Spawn mode only (`honcho/_honcho_server.py`'s `child_env`); the shared `honcho-api`/`honcho-deriver` anchor does not read this var directly today |
| `HONCHO_DERIVER_MAX_OUTPUT_TOKENS` | `2048` | `DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS`, an overlay on the global row above so the deriver alone is capped. At 8192 the deriver repetition-looped: 18 of 79 documents in smoke `hn_smkft_p0` sat at the cap, mean 41,189 chars, unique-sentence ratio 0.181, and those rows then overflowed the recall window (32768 under contract v4). Real observations have a median length of 241 chars. Set in both the run-service and the `honcho-deriver` anchor env |
| `HONCHO_DERIVER_PRESENCE_PENALTY` | `1.5` | `DERIVER_MODEL_CONFIG__PRESENCE_PENALTY`, deriver only. The Qwen card value the answer role already uses (`answer_env.sh`). vLLM's `get_diff_sampling_param` does not allowlist `presence_penalty`, so a provider-internal call gets it only as a per-request kwarg. Empty sends no penalty |
| `HONCHO_LLM_THINKING_EFFORT` | unset | Maps to `reasoning_effort` on the internal LLM. Left empty for qwen3.5-4b (not a reasoning model); needed for a reasoning model, where the default effort can spend the entire output budget reasoning and return empty content (observed: gpt-oss-20b's deriver call burned 8192 tokens on reasoning alone and emitted zero observations, the same failure mode `MEMCONFLICT_REASONING_EFFORT=low` avoids on the harness answer role). Spawn mode only |
| `HONCHO_EMBEDDER_MODEL` / `HONCHO_EMBEDDER_BASE_URL` / `HONCHO_EMBEDDER_API_KEY` / `HONCHO_EMBEDDER_DIMS` | `gte-modernbert-base` / `vllm-embed` / `local-vllm` / `768` | Embedder. Matches every other provider's retrieval-embedding surface. Contract v4 ran `bge-small-en-v1.5` / `384` |
| `HONCHO_EMBEDDER_DIMENSIONS_MODE` | `never` | `EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE`. The pydantic-settings default `auto` sends an OpenAI `dimensions=` request parameter whenever `HONCHO_EMBEDDER_DIMS` is set. vLLM's pooling endpoint 400s on that parameter for a model with no matryoshka support, which both contract v4's bge-small and contract v5's gte-modernbert-base are: each has one fixed output width, so `never` is simply correct. Set in both the run-service and the `honcho-api`/`honcho-deriver` anchor env |
| `HONCHO_EMBED_SHIM` | `0` | Host-smoke group (see below). Docker always sets `HONCHO_EMBEDDER_BASE_URL`, so this is inert here either way; set explicitly so the manifest records the intent |
| `HONCHO_EMBED_PROXY` | `1` | Spawn-mode generate only. Starts `honcho/embed_proxy.py` and repoints `HONCHO_EMBEDDER_BASE_URL` at it. The proxy adds `truncate_prompt_tokens` to every upstream embedding request. `vllm-embed` answers 400 to any input above its served window, and Honcho's `simple_batch_embed` (`external/honcho/src/embedding_client.py:251`) neither chunks nor length-checks, so one long observation drops the whole "save representation" call for both observers (`src/crud/representation.py:111`). Measured under contract v4's 512-token bge-small window, smoke `hn_smkmin_p0b`: 14 dropped saves against 11 completed deriver batches in persona 0, sessions 0-2. Contract v5 serves an 8192-token window against featured-cadence inputs that peak at ~1,585 tokens, so the proxy is now a backstop, not the routine path. `EMBEDDING_MAX_INPUT_TOKENS` does not fix it. That path never reads it, and Honcho counts tokens with tiktoken `cl100k_base`, not the embedder's tokenizer. Set `0` to measure the unproxied failure rate again |
| `HONCHO_EMBED_PROXY_PORT` / `_UPSTREAM` / `_TRUNCATE` / `_TIMEOUT_S` | `3198` / `HONCHO_EMBEDDER_BASE_URL` / `-1` / `120` | Proxy knobs (`honcho/embed_proxy.py`). `-1` tells vLLM to truncate to the served model's own `max_model_len`, so the cut follows the embedder rather than a number hardcoded in the proxy. The proxy binds 127.0.0.1 only: every client is a child process of the same container |
| `HONCHO_DREAM_ENABLED` | unset (vendor default `true`) | Passthrough only, **not** force-set by this run-service or by `honcho-api`/`honcho-deriver`, because `DREAM_ENABLED` is a bool field and an unconfigured `${VAR:-}` would reach the container as a present, empty string, which pydantic-settings bool parsing rejects (the same trap `unset_empty_env_with_prefix` works around for `HINDSIGHT_API_*`). Dreams idle-trigger after 60 minutes (`DREAM_IDLE_TIMEOUT_MINUTES`), so they do not fire mid-run at this scale. Override with an explicit `-e DREAM_ENABLED=false` on `honcho`/`honcho-api`/`honcho-deriver` if a long-idle smoke needs them suppressed |
| `HONCHO_DREAM_AFTER_SESSION` | `0` | `1` = after each session's ingest+drain, manually schedule an omni dream per observed pair via `POST /v3/workspaces/{id}/schedule_dream` and drain again; simulates the inter-session idle that fires Honcho's automatic dream scheduler in a real deployment (dataset sessions are days apart). The featured preset sets `1`; the minimal preset leaves the default `0` |
| `HONCHO_EMBED_SHIM_HOST` / `_PORT` / `_MODEL` | `127.0.0.1` / `8099` / vendor default | Host-smoke group: only read when `HONCHO_EMBED_SHIM` is truthy AND `HONCHO_EMBEDDER_BASE_URL` is empty (`honcho/_local_embed_server.py`, started by `honcho/run_honcho.sh` for a host smoke with no `vllm-embed`). Not part of the Docker topology. Every Docker run sets `HONCHO_EMBEDDER_BASE_URL`, so these three are never read there. The shim still serves bge-small at 384 dims via fastembed, so a host smoke does not match contract v5 and its numbers are off-contract |

Two named presets bundle the arms above (`benchmark/docker/presets.sh`,
`docker compose run -d --rm -e PRESET=honcho_minimal_clocksync honcho`):

- `honcho_minimal_clocksync`: spawn mode, `conclusions` recall, `unified`
  observation, summaries and peer cards off, no per-session dream trigger.
- `honcho_featured_clocksync`: spawn mode, `hybrid` recall, `directional`
  observation (the plugin's real read surface), summaries and peer cards on,
  per-session dream trigger on (`HONCHO_DREAM_AFTER_SESSION=1`).

Results and scores land on the host through the `honcho/` bind-mount, tagged
the same way as Mnemosyne (`honcho/Results/honcho_results_<tag>.jsonl`, and
so on, where `<tag>` is `RUN_TAG` or `honcho` by default).

## OpenViking run-service control knobs (all via env)

Pass these with `docker compose run -d --rm -e VAR=value openviking`. This
service also honors the generic subset shared with Mnemosyne above
(`STAGE`, `NUM_PERSONAS`, `TOP_K`, `MAX_SESSIONS`,
`MAX_QUESTIONS_PER_SESSION`, `SCORE_WORKERS`, `RUN_TAG`, `OPENAI_BASE_URL` /
`OPENAI_MODEL`).

OpenViking ships as ONE pip distribution: `openviking==0.4.12` installs the
SDK and the `openviking-server` console script, and that server keeps its AGFS
content store and its vector index in one local workspace directory. So this
provider has no database sidecar and no central compose service.
`openviking/_openviking_server.py` writes an `ov.conf` JSON file per run and
starts the server as a child process. Personas are isolated by the
`X-OpenViking-User` header (the vendor's own LoCoMo harness pattern), which
also scopes the wipe the adapter runs at the start of each persona.

The adapter speaks raw `httpx`, never the SDK: the Hermes plugin has no SDK
client either, and the provider folder `/app/openviking` shadows the pip
package name.

| Var | Default | Meaning |
|---|---|---|
| `OPENVIKING_SERVER_MODE` | `spawn` | `spawn` starts a disposable server inside this container, with its workspace under `openviking/.openviking_runs/` (gitignored). `shared` attaches to an operator-run server at `OPENVIKING_ENDPOINT` and only health-checks it. There is no compose service for `shared`: a workspace holds a one-process `.openviking.pid` lock. The entrypoint exits 2 on `shared` under `BENCH_CLOCKSYNC=1` (one attached server has one perceived clock and cannot serve N shards at different logical session dates) and exits 2 on `shared` with no `OPENVIKING_ENDPOINT` |
| `OPENVIKING_ENDPOINT` | unset (spawn) | Server URL for `shared` mode. The `spawn` branch unsets it, so an inherited value cannot defeat the mode selection |
| `OPENVIKING_SERVER_BIN` | unset → the console script next to `sys.executable`, then `shutil.which` | Explicit path to `openviking-server` |
| `OPENVIKING_SERVER_PORT` | `0` | `0` asks the OS for an ephemeral port, so co-tenant shards never collide. Set a fixed port for a hand-driven server |
| `OPENVIKING_RUN_DIR` | unset → `.openviking_runs/<sanitized RUN_TAG>` next to the adapter | Where the generated `ov.conf` and the spawned server's `server.log` land. Gitignored (`openviking/.openviking_runs/`) |
| `OPENVIKING_WORKSPACE` | unset → `<OPENVIKING_RUN_DIR>/data` | Server storage directory (content store + vector index). One workspace holds one server process |
| `OPENVIKING_API_KEY` | empty | Empty selects the server's `auth_mode: "dev"` on loopback: identity comes from the `X-OpenViking-Account` / `X-OpenViking-User` / `X-OpenViking-Actor-Peer` headers with no key. A set key switches the client to `X-API-Key` + `Authorization: Bearer` |
| `OPENVIKING_ACCOUNT` | `default` | `X-OpenViking-Account` header (tenant) |
| `OPENVIKING_USER_PREFIX` | derived from `RUN_TAG` | Every persona's user id is `<prefix><persona tag>` (the last 8 chars of the persona id, non-alphanumerics mapped to `_`). Empty → the entrypoint derives `<sanitized RUN_TAG>_`, so two waves on one attached server never share a user id |
| `OPENVIKING_AGENT` | `hermes` | `X-OpenViking-Actor-Peer` header, and the `peer_id` on every assistant message |
| `OPENVIKING_RECALL_MODE` | `prefetch` | `prefetch` (headline, featured, the scored comparison arm, the plugin's own read surface: a session-start block built from `profile.md` plus a `preferences/` and `entities/` listing, then the `/api/v1/search/search` entries selected by the plugin's `_select_recall_candidates`) \| `find` (minimal, diagnostic: deterministic `/api/v1/search/find`, level-2 leaves, full content bodies, `recall_limit` items kept, no LLM in the retrieval path; integration proof, not a comparison number) \| `search` (the search entries alone; auxiliary, no planned run). Every arm passes the plugin's `recall_limit` selection (6) to the answer model whole; the scorer slices the stored list at its own K (user ruling 2026-08-04) |
| `RETAIN_GRANULARITY` / `OPENVIKING_RETAIN_GRANULARITY` | `exchange` | `exchange` is the plugin's `sync_turn` cadence: one `POST /api/v1/sessions/{sid}/messages/batch` per user+assistant exchange. `session` sends one POST per session, chunked at the server's 100-message batch cap. The entrypoint resolves both spellings to one value, so the two manifest keys cannot disagree |
| `OPENVIKING_RECALL_LIMIT` | `6` | Plugin default. The server request asks for `max(limit × 4, 20)` candidates, and the client-side selection cuts to this after ranking |
| `OPENVIKING_RECALL_SCORE_THRESHOLD` | `0.15` | Plugin default, applied client-side. The server search itself runs at `score_threshold: 0`, exactly as the plugin sends it |
| `OPENVIKING_RECALL_MAX_INJECTED_CHARS` | `4000` | Plugin default: running character budget over the joined recall entries. An oversized entry is skipped, and a later smaller one can still fit |
| `OPENVIKING_PROFILE_TOKEN_BUDGET` | `6000` | Plugin default for the session-start block. The plugin counts in quarter-token units (`6` per CJK char, `1` otherwise), caps the profile at half the budget, and splits the rest between the two listings |
| `OPENVIKING_RECALL_FULL_READ_LIMIT` | `2` | Plugin default: how many selected items may spend a `GET /api/v1/content/read` call for the full body. Everything else uses the `abstract` |
| `OPENVIKING_RECALL_PREFER_ABSTRACT` | `0` | Plugin default. `1` uses the abstract for every item and spends no full reads |
| `OPENVIKING_RECALL_RESOURCES` | `0` | Plugin default. `1` adds `"resource"` to the search `context_type` and harvests `result["resources"]` after `result["memories"]` |
| `OPENVIKING_RECALL_TIMEOUT_SECONDS` / `_REQUEST_TIMEOUT_SECONDS` | `60` / `30` | **Not the plugin's 4.0/3.0.** The plugin joins prefetch on a background thread and drops whatever has not arrived by then; the adapter calls recall INLINE, so the plugin budget empties recall under benchmark serving latency with nothing logged. 60 is the plugin's own clamp maximum (the same reasoning as `HONCHO_TIMEOUT=300`) |
| `OPENVIKING_SEND_CREATED_AT` | `0` | `1` sends `created_at` per message (session date midnight UTC + the message index in seconds). The plugin sends none, so `0` is plugin-faithful and `BENCH_CLOCKSYNC` is the temporal path. The extraction prompt anchors relative dates on the first message's `created_at`, and event memories are filed under `events/{year}/{month}/{day}` |
| `OPENVIKING_HTTP_TIMEOUT` | `600` | httpx per-request timeout for the ingest and drain calls. It bounds a commit POST, which extraction makes long. The two recall budgets above are separate |
| `OPENVIKING_DRAIN_TIMEOUT_S` / `_POLL_S` | `1800` / `1.0` | After each session's ingest the adapter POSTs `/api/v1/sessions/{sid}/commit`, polls `GET /api/v1/tasks/{task_id}` at 1 s, then calls `POST /api/v1/system/wait` (timeout `min(600, remaining budget)`) until the embedding and semantic queues empty. It raises on a failed or cancelled task, on this timeout, and on any `error_count > 0`. A broken embedder surfaces in that count and nowhere else. This is a deviation from the plugin's fire-and-forget commit, the same one honcho and supermemory make |
| `OPENVIKING_LLM_MODEL` / `_BASE_URL` / `_API_KEY` | `qwen3.5-4b` / `vllm-gen` / falls back to `OPENAI_API_KEY` | The `vlm` section of `ov.conf`: OpenViking's internal chat model, used for memory extraction at commit and for the search-intent analysis of `/api/v1/search/search`. One model for both internal roles, per the best-effort ruling. Extraction uses tool calling and degrades silently to fewer memories on a tool-less endpoint, so contract v4's `--enable-auto-tool-choice --tool-call-parser qwen3_coder` matter here |
| `OPENVIKING_LLM_MAX_TOKENS` / `_TEMPERATURE` / `_TIMEOUT` | `4096` / `0.0` / `600` | `vlm.max_tokens`, `vlm.temperature` (the vendor sample-config value), `vlm.timeout` |
| `OPENVIKING_LLM_MAX_CONCURRENT` | `8` | `vlm.max_concurrent`. The vendor default is 64: one shard with 64 in-flight extraction calls starves the same `vllm-gen` that serves the shared answer role |
| `OPENVIKING_LLM_EXTRA_BODY` | *(empty)* | JSON merged into `vlm.extra_request_body`, a vendor knob. Reasoning models need `{"reasoning": {"effort": "low"}}`. At default effort gpt-oss-20b returned empty extraction responses or outlived OpenRouter's keep-alive window (`docs/TROUBLESHOOTING.md`, "Provider: OpenViking"). Local qwen3.5-4b leaves it empty |
| `OPENVIKING_EMBEDDER_MODEL` / `_BASE_URL` / `_API_KEY` / `_DIMS` | `gte-modernbert-base` / `vllm-embed` / `local-vllm` / `768` | The `embedding.dense` section of `ov.conf`, `encoding_format: "float"`. The same retrieval-embedding surface as every other provider. `_openviking_server.py` raises at `start()` when the base URL is empty in spawn mode. Changing the model or the dimension invalidates an existing workspace, so a contract v4 workspace (384 dims) cannot be reused under v5 |

Two named presets bundle the arms above (`benchmark/docker/presets.sh`,
`docker compose run -d --rm -e PRESET=openviking_minimal_clocksync openviking`):

- `openviking_minimal_clocksync`: spawn mode, `find` recall (deterministic
  `/search/find`, the diagnostic retrieval floor), per-exchange ingest.
- `openviking_featured_clocksync`: spawn mode, `prefetch` recall (the plugin's
  real read surface, no top-K slice), per-exchange ingest.

Sharded waves run `benchmark/docker/run_shards.sh openviking <tag>`: every
shard spawns its own server, the pre-step brings up `vllm-gen` and
`vllm-embed`, and the clock-sync arm adds `--no-deps`.

Results and scores land on the host through the `openviking/` bind-mount,
tagged the same way as Mnemosyne
(`openviking/Results/openviking_results_<tag>.jsonl`, and so on, where `<tag>`
is `RUN_TAG` or `openviking` by default).

## Running the judge / scoring

`STAGE` selects which part of the pipeline a provider run-service executes:

| `STAGE` | What runs |
|---|---|
| `generate` | Ingest each persona's dialogue, answer every question, write `Results/<provider>_results_<tag>.jsonl` |
| `score` | LLM-judge each answer, write `Scores/<provider>_<tag>_eval_scores.jsonl` plus a `<tag>_judged_checkpoint.jsonl` |
| `summarize` | Reduce the per-question scores to `Scores/summary_<tag>.json` |
| `all` (default) | `generate`, then `score`, then `summarize` |

Scoring is resumable and provider-agnostic. `run_score` (in `answer_env.sh`)
runs the same `bench_judge_env` and `score_resumable.py` call for every
provider, a thin wrapper over `external/MemConflict/Evaluation/eval_scoring.py`;
`SCORE_WORKERS` is the only per-provider knob (judge concurrency, default 24 on
Mnemosyne, 8 elsewhere). Re-score an existing Results file in place with the
same tag:

```bash
docker compose run -d --rm -e STAGE=score     -e RUN_TAG=<tag> <provider>
docker compose run -d --rm -e STAGE=summarize -e RUN_TAG=<tag> <provider>
```

**The default judge is the answer model.** Per the harness contract the judge
model equals the answer model, so `STAGE=score` sends the judge requests to
`OPENAI_BASE_URL`, which defaults to `vllm-gen` (`qwen3.5-4b`). No separate
judge server is needed for a normal run, and `vllm-gen` stays up throughout.

**The `vllm-judge` second-judge arm (`_gj12`).** To re-judge a banked wave with
`gemma-4-12b` instead, the judge server must run and the answer servers must be
stopped (it takes `--gpu-memory-utilization 0.9` of the one card). Bring it up
under the `judge` profile:

```bash
docker compose down                                   # free the card
docker compose --profile judge up -d vllm-judge       # gemma-4-12b on :8002
```

Then score off the entrypoints entirely with `benchmark/score_files.sh` (host
`.venv`), which reads a Results path and judge details and enters no provider
entrypoint. It requires the three sampling flags explicitly, so one arm never
silently inherits the qwen contract sampling:

```bash
benchmark/score_files.sh --temperature 1.0 --top_p 0.95 --top_k 64 \
  hindsight/Results/v4/hindsight_results_v4minc.jsonl
```

`score_files.sh` defaults to `--base_url http://localhost:8002/v1`, `--model
gemma-4-12b`, `--suffix gj12`, `--workers 32`. Every artifact carries the
`_gj12` suffix so a gemma-judged file never overwrites a qwen-judged one. The
superseded `benchmark/docker/score_with_judge.sh` drives the same second judge
through the Docker entrypoints; it is kept only for the record.

**The penalty-rubric judge arm (`_gj12pen`).** The standard rubric collapses a
wrong answer and an absent answer into one 0.0 score.
`benchmark/make_penalty_judge_evaldir.py` writes a patched copy of the upstream
Evaluation package to `benchmark/penalty_judge_eval/`, whose per-conflict-type
rubric scores a wrong or contradictory answer -1 and keeps 0.0 for a missing or
uncertain one. Build the copy once, then point `score_files.sh` at it through
`MEMCONFLICT_EVAL_DIR` with the `gj12pen` suffix:

```bash
python benchmark/make_penalty_judge_evaldir.py
MEMCONFLICT_EVAL_DIR=benchmark/penalty_judge_eval \
  benchmark/score_files.sh --temperature 1.0 --top_p 0.95 --top_k 64 \
    --suffix gj12pen hindsight/Results/v4/hindsight_results_v4minc.jsonl
```

The two arms are NOT comparable: -1 is outside upstream's metric range. The
featured v5 comparison is judged under this gemma-4-12b penalty rubric
(`docs/BENCHMARK_MATRIX.md`, "THE v5 FEATURED COMPARISON").

## Ad-hoc / debugging

Any explicit command bypasses the stage machinery and runs verbatim (the
env vars are still wired in):

```bash
docker compose run --rm mnemosyne python -c "import mnemosyne, os; print(mnemosyne.__version__, os.environ['OPENAI_BASE_URL'])"
docker compose run --rm mnemosyne python -u mnemosyne/eval_mnemosyne.py --start_idx 0 --end_idx 1
```

### `INPUT_JSONL`: run a provider against a probe dataset

`-e INPUT_JSONL=/app/benchmark/probes/<file>.jsonl` swaps the dataset for
one run (the mem0, hindsight, retaindb-server, and supermemory entrypoints
support this). It is inert when unset, so it cannot affect a normal run.
Use it to ask a provider a targeted question through its real adapter and
preset, instead of a parallel harness.

`benchmark/probes/relative_dates_persona.jsonl` is the worked example. It
has one session dated 2022-11-22 that says "started my new job
**yesterday**", "**last week** I moved", and "flying to Tokyo **tomorrow**".
Run it under a `*_minimal_clocksync` preset while the real clock reads a
different year, then inspect the STORED FACT TEXT. Row timestamps only
prove the process clock is faked. They do not prove that extraction wrote a
correctly anchored date into the fact. This distinction is what caught
Supermemory resolving every relative phrase against a hallucinated 2023
"today" (`docs/BENCHMARK_MATRIX.md`, relative-date table). Probe output is
evidence, never a banked result. Tag it `probe_*`.

## Pointing at different servers/models

The harness only speaks HTTP. To run against other endpoints, such as a
remote vLLM or a different model, override the `OPENAI_*` / `MNEMOSYNE_*`
vars. For example, keep the compose `mnemosyne` service, but set
`OPENAI_BASE_URL` to an external URL. The service then uses that URL
instead of the bundled `vllm-gen`.

## Rebuilds

Each provider service builds from its own Dockerfile (`Dockerfile.mnemosyne`,
`Dockerfile.hindsight`, `Dockerfile.retaindb`). A rebuild is needed only
when that provider's Python dependencies change (`requirements-lock.txt` or
a pinned submodule): run `docker compose build mnemosyne` (or `hindsight` /
`retaindb`). Harness edits, both the `.py` files and the
`entrypoint.<provider>.sh` scripts, are picked up live through the
`benchmark/` and `<provider>/` bind-mounts (the Mnemosyne entrypoint runs
from `/app/benchmark/docker/entrypoint.mnemosyne.sh`). No rebuild is needed
for these edits.

## Clock sync (`BENCH_CLOCKSYNC`)

This is an additional featured arm per provider. Selected generate-stage
processes run under `libfaketime` (`LD_PRELOAD`), so their perceived OS
clock tracks the dataset's logical session date instead of benchmark
wall-time. The shared driver (`benchmark/eval_common.py`
`Generate_Single_Persona_Eval`) calls `clock_sync.set_clock(session_date)`
once per session, before ingest. This writes the session date at 12:00 UTC
to the shard's timestamp file, in libfaketime's `@YYYY-MM-DD HH:MM:SS`
form. That session's ingest and Q&A then run under this faked clock. This
arm is off by default, and produces byte-identical output to the minimal
arm when off: `benchmark/clock_sync.py` is a no-op unless both vars are
set, and the entrypoint fragment `benchmark/docker/clock_sync.sh` runs only
on generate paths. Score, summarize, and the run manifests always stay on
the real clock, by design.

| Var | Default | Meaning |
|---|---|---|
| `BENCH_CLOCKSYNC` | `0` (unset) | Master enable. `1` turns the arm on |
| `BENCH_CLOCKSYNC_FILE` | set by the entrypoint | Per-shard timestamp file path. Never shared across shards, because each shard owns its own timeline. Exported as `FAKETIME_TIMESTAMP_FILE` |
| `BENCH_LIBFAKETIME` | `/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1` | Path to the `libfaketime.so.1` baked into the image |

`bench_clocksync_prepare` seeds the file with real time (`+0`) and exports
the FAKETIME contract: `FAKETIME_NO_CACHE=1` (readers re-parse on each
clock call, so live stepping works), `FAKETIME_DONT_FAKE_MONOTONIC=1`
(timeouts and sleeps stay real), and `NO_FAKE_STAT=1` (observed file
mtimes stay real). It never sets `LD_PRELOAD` itself. The warm-boot rule follows from the file
being seeded to real time: any process that boots before the first
per-session step (TLS cert validation, model caches) boots warm, at the
real clock. Perceived time jumps to the logical date only after the
driver's first `set_clock` call.

The preloaded process is the only difference between providers:

| Provider (arm) | Preloaded process | How | Gates |
|---|---|---|---|
| Supermemory (`clocksync`) | Only the spawned Bun server child | The adapter's `_supermemory_server._env()` injects `LD_PRELOAD` into the child env. The shell itself never sets it | Requires `SUPERMEMORY_SERVER_MODE=spawn`. The entrypoint refuses `shared` plus clocksync with exit code 2, because one central server cannot run N shard timelines at once. A boot-time Bun honor probe ingests a document under a sentinel 2019 clock, and aborts if the server does not stamp it with the year 2019 (first boot only). The adapter also restarts the spawned server once per session on the same data dir, after the clock steps and before the session's first document, so no server process ever observes a forward clock jump: node-cron v4 would otherwise replay every missed 30-minute slot of that jump at ~0.46 MB per slot and OOM the host (`docs/TROUBLESHOOTING.md`, Provider: Supermemory). `SUPERMEMORY_RESPAWN_PER_SESSION=0` disables it; the flag is ignored without clocksync |
| Mnemosyne (`clocksync-ttl` featured / minimal rerun) | the in-process generate python | `bench_clocksync_preload` inside the per-shard subshell | Two sanctioned combinations. Any other combination exits with code 2. Featured: `PLUGIN_AUTO_SLEEP=1` with no explicit `MNEMOSYNE_WM_TTL_HOURS`. This runs the shipped 168h WM TTL, because auto-sleep's consolidation exemption carries rows across gaps over 168h. Minimal: `PLUGIN_AUTO_SLEEP=0` with an explicit `MNEMOSYNE_WM_TTL_HOURS` required (the preset passes 8760000). Without auto-sleep, nothing consolidates, so the shipped TTL would gut recall. This is the same accommodation the pre-clock minimal run used |
| RetainDB server (`clocksync`) | the node server AND an internal per-shard throwaway Postgres 18 | A per-command `env LD_PRELOAD=…` on the postmaster and node processes only, never shell-wide. The health-wait `date +%s` loops stay real | `Dockerfile.retaindb-server` bakes in `postgresql-18` and `postgresql-18-pgvector`, because a shared central `hindsight-pg` cannot be clock-faked per shard. We verified scheduler-promotion under the faked clock on 2026-07-24 (bbd19e1), which needs `RETAINDB_SERVER_PROMOTION_MODE=user_specific_legacy` (see `docs/BENCHMARK_MATRIX.md`). The clocksync pg trigger also forces `updatedAt` on INSERT/UPDATE (`server_patches/clocksync_created_at.sql`) |
| mem0 (`clocksync`) | the in-process generate python (per-shard container) | `bench_clocksync_prepare`+`probe`+`preload` subshell in `entrypoint.mem0.sh` `do_generate` | mem0ai 2.x computes extraction-prompt dates per `add()` call from the process clock, so libfaketime alone covers it (we deleted the 0.1.118 frozen-prompt monkeypatch). `eval_mem0.py` refuses `BENCH_CLOCKSYNC=1` on a non-2.x mem0ai. Stored `created_at` is also set explicitly through `metadata` |
| Hindsight MINIMAL (`hindsight_minimal_clocksync`, arm A) | none, no OS-level preload | API-level: the dataset `timestamp=` on retain, and a noon `query_timestamp=` on recall (`temporal_capability=native`) | `BENCH_CLOCKSYNC=1` changes no process clock here. It declares the temporal contract in the manifest and arms the strict run-contract gate. Session granularity goes through `_retain_one`, which sends no `document_id` and no `update_mode`, so the vendor honours the retain timestamp and no OS clock is needed. This arm stays on the shared `hindsight-pg` |
| Hindsight FEATURED (`hindsight_featured_clocksync`, arm C) | the spawned `hindsight-api` daemon, its `initdb`, and its pg0 postmaster | `hindsight/eval_hindsight.py` `_inject_daemon_clock_env` writes `LD_PRELOAD` plus the `FAKETIME_*` contract into `os.environ` immediately before `HindsightEmbedded(...)`. The vendor spawns the daemon with `env=os.environ.copy()`, the pg0 SDK passes no `env=`, and the pg0 CLI orphans the postmaster with the daemon's environment, so one write covers all three. The entrypoint shell and the adapter stay on the REAL clock, so their poll deadlines cannot be stretched | Requires `HINDSIGHT_PG_MODE=pg0` and one persona per container; the entrypoint exits 2 otherwise. `exchange_append` retains under `update_mode="append"`, and the vendor's append merge takes `mentioned_at` from the DB clock, so this arm needs a cluster it owns. The shared `hindsight-pg` has one clock for all co-tenant shards. A missing `.so` raises rather than warns: a silently un-faked daemon reproduces the `ftclk1_p0` defect and still passes the strict gate. Manifest reads `controlled_process_clock+postgres`, set by the entrypoint. `bench_hs_pg0_report` prints the extensions and the `mentioned_at` range per fact type into every shard log |

Launch each run with a fresh `RUN_TAG`. Results merge at the JSONL level as
usual:

```bash
# Mnemosyne: single container, shards internally. TTL arm requires auto-sleep.
docker compose run -d --rm -e BENCH_CLOCKSYNC=1 -e PLUGIN_AUTO_SLEEP=1 \
  -e PLUGIN_CONFIG=user -e RUN_TAG=cs_full mnemosyne

# Supermemory: run_shards forces spawn + --no-deps + EMBEDDING_RAM_LIMIT=2gb,
# refuses SUPERMEMORY_TWO_SERVERS=1, and brings up only vllm-gen.
BENCH_CLOCKSYNC=1 benchmark/docker/run_shards.sh supermemory cs_full

# RetainDB server: run_shards runs --no-deps and brings up only vllm-gen/vllm-embed;
# each shard's own throwaway Postgres 18 shares the faked clock with its node server.
BENCH_CLOCKSYNC=1 benchmark/docker/run_shards.sh retaindb_server cs_full

# Hindsight FEATURED: the reranker is NOT in depends_on, and --no-deps below
# keeps hindsight-pg down, so start the reranker first or every shard fails
# its first recall.
docker compose up -d hindsight-rerank
# run_shards detects the arm from the preset (or from HINDSIGHT_PG_MODE=pg0),
# exports BENCH_CLOCKSYNC=1 so PERSONA_CONTAINERS defaults to 1, adds --no-deps,
# brings up only vllm-gen + vllm-embed, and defaults the pool to 4
# (~2.5 GB/container: ~1.9 GB daemon + 0.3-0.4 GB postmaster + adapter).
PRESET=hindsight_featured_clocksync benchmark/docker/run_shards.sh hindsight v5ftc086

# Hindsight MINIMAL: unchanged. Shared hindsight-pg, no OS clock change.
PRESET=hindsight_minimal_clocksync benchmark/docker/run_shards.sh hindsight v5minc086
```

## Fairness: shared decoding config + run manifests

All provider entrypoints source `benchmark/docker/answer_env.sh`, the
single source of truth for the answer/judge decoding config (the fairness
contract). It exports the canonical decoding for each stage, so no provider
can drift:

- `bench_answer_env` (pre-generate) sets: `OPENAI_TEMPERATURE=0.2`,
  `MEMCONFLICT_ENABLE_THINKING=${THINKING:-1}` (thinking on is canonical),
  and `OPENAI_MAX_TOKENS=3072` (1024 when `THINKING=0`). Override with
  `BENCH_ANSWER_MAX_TOKENS` / `BENCH_ANSWER_TEMPERATURE`.
- `bench_judge_env` (pre-score) sets: `MEMCONFLICT_JSON_MODE=1`,
  `OPENAI_MAX_TOKENS=${SCORE_MAX_TOKENS:-4096}`, and
  `OPENAI_TEMPERATURE=0.2`. Each stage re-exports its full config, so in a
  `STAGE=all` run the judge never inherits the answer stage's budget.
- `run_score` / `run_summarize` build the same per-provider file paths and
  invoke `score_resumable.py` / `summarize_scores.py` identically.
  `SCORE_WORKERS` is the only per-provider knob (`BENCH_PYTHON=python3` on
  the retaindb image).

Before this change, only the Mnemosyne entrypoint pinned decoding.
Hindsight and retaindb ran at the vLLM server defaults, a fairness bug that
this fix corrects.

Each generate-start and score-start also writes a best-effort manifest to
`<provider>/Scores/manifest_<RUN_TAG>_<stage>.json`
(`benchmark/write_manifest.py`, one file per stage). The score-stage
manifest is written inside `run_score` after `bench_judge_env`, so it
records the judge decoding actually used. Each manifest records: the UTC
timestamp, the stage, repo/submodule SHAs (or `unavailable_in_container`
plus an optional `GIT_SHA` passthrough), the dataset path and line count,
the canonical answer/judge config block (diffable across providers), and a
redacted env snapshot. A manifest failure never aborts a run.

### Thinking split (contract v2): shared answer role ON, every internal LLM OFF

Under the Qwen3.5 serving model (contract v2), the fairness rule for
reasoning is two-sided. We enforce it in two different places:

- **Shared answer and judge roles.** `answer_env.sh` and
  `benchmark/llm_reasoning.py` (above) pin these per request. The answer
  role runs thinking on (canonical `THINKING=1`). The judge runs thinking
  off, under JSON mode.
- **Every provider's internal, memory-side LLM** runs thinking off. This
  covers the structured extraction, consolidation, model-refresh, and
  conflict-detection calls a real deployment makes on ingest. Reason:
  thinking-on structured extraction on Qwen measured ~384s per call versus
  ~14s with thinking off, with no quality gain, and it overflows the
  internal token budgets. No real operator would run it this way.

`vllm-gen` enforces the internal half server-side, with
`--default-chat-template-kwargs '{"enable_thinking": false}'`. Qwen3.5's
chat template defaults `enable_thinking` to on when unset, so any call that
sends no `chat_template_kwargs` would otherwise reason at length. The
server default flips that to off. vLLM merges this default under any
request-level `chat_template_kwargs`, so a request-level setting always
wins:

| caller | sends `chat_template_kwargs`? | effective thinking |
|---|---|---|
| shared **answer** (`THINKING=1`) | yes, `{"enable_thinking": true}` (llm_reasoning.py) | on. The request overrides the server default |
| shared **judge** (JSON mode) | no (suppressed under JSON mode) | off. It inherits the server default, and the guided-JSON grammar engages from token 0 |
| **Mnemosyne** internal (`--extract`/`--lifecycle`/`--canonical`) | no. The pinned submodule posts fixed payloads with no injection hook | off. It inherits the server default. This is the fix |
| **Hindsight** retain/consolidation | yes, `HINDSIGHT_API_LLM_EXTRA_BODY` `{"enable_thinking": false}` | off. This is explicit. The server default only backs it up |
| **RetainDB** | n/a, no internal LLM | n/a |

Why the fix is server-side for Mnemosyne specifically: its internal calls
hit `/chat/completions` with a fixed request dictionary, and the pinned
submodule exposes no `extra_body` / `chat_template_kwargs` passthrough
through any env var or config field. So a per-request fix is impossible
without editing `external/`, which is forbidden. The server default is the
only lever that does not touch the submodule, and it covers all of
Mnemosyne's internal call sites at once. Note: this only governs calls that
reach `vllm-gen`. A Mnemosyne run pointed at a different
`MNEMOSYNE_LLM_BASE_URL`, or the unused cloud `ExtractionClient`/OpenRouter
path, would not inherit it.
