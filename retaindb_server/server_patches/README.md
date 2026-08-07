# `server_patches/`: RetainDB server fresh-deploy fixes

`@retaindb/server` does **not** run correctly from pristine submodule source:
bringing it up on this host required the fixes listed in the table below.
Some make the server boot or build. Others were found once it was scoring
near-zero (a response-mapping bug, cache defects, and a recall-killing index
default). One more was found later, at scale. They are **upstream defects /
deploy gaps**, reported here as findings and applied as an explicit patch
layer so `Dockerfile.retaindb-server` and `serve_local.sh` build a working
server without ever writing under `external/RetainDB` (the pinned submodule
stays read-only).

The layer is applied in this order (both the Docker build and `serve_local.sh`):

```
copy submodule out  ->  git apply server_patches/*.patch
                    ->  cp schema.introspected.prisma packages/server/prisma/schema.prisma
                    ->  pnpm install  ->  prisma generate  ->  build
runtime:  prisma migrate deploy  ->  seed.sql + post_migrate.sql  ->  node dist/index.js
```

`prisma migrate deploy` uses the **original** migrations dir (unchanged: the
migrations themselves are fine); only the *generator input* schema is swapped.

## Fix map

| # | Fix | Artifact |
|---|-----|----------|
| 1 | Prisma `$use` middleware API removed in `@prisma/client@6.19.3`; ported to `$extends` | `0001-db-index-prisma-use-to-extends.patch` |
| 2 | `sharp` native postinstall build blocks `pnpm install` | `0002-package-json-sharp-neverbuilt.patch` |
| 3 | Shipped `schema.prisma` describes 27 of 67 tables (stale subset) | schema replacement (`schema.introspected.prisma`) |
| 4 | Fresh `migrate deploy` leaves `organizations`/`users` empty, so the first ingest 500s | `seed.sql` |
| 5 | Boot fails without a non-empty `ENCRYPTION_KEY` | `ENCRYPTION_KEY` env var |
| 6 | Search response builder reads temporal fields at the wrong nesting level | `0003-search-response-temporal-nested.patch` |
| 7 | Semantic and exact-key search caches corrupt benchmark retrieval | `0004-disable-search-cache-env.patch` |
| 8 | Empty-table ivfflat indexes give untrained centroids and miss true nearest neighbors | `post_migrate.sql` |
| 9 | Malformed `searchMemories` rows 500 the search endpoint | `0005-search-filter-malformed-rows.patch` |

Fix 2 also needed a sandbox-specific `sharp` runtime stub in this environment.
That stub is not a committed artifact (manual, local-only, host-smoke-only);
see fix 2 below for the workaround.

## The fixes

Fixes 1-5 make the server boot or build. The temporal-nested and
search-cache patches (`0003`, `0004`), plus the ivfflat index drop
(`post_migrate.sql`), were found once it was scoring near-zero (micro AA
0.004 on a 1-persona smoke) and traced to server-side defects that corrupt
measurement. Fix 9 (`0005-search-filter-malformed-rows.patch`) was found
later, at scale, after the ivfflat drop.

### 1. `0001-db-index-prisma-use-to-extends.patch`: Prisma `$use` removed

The pinned lockfile resolves `@prisma/client@6.19.3`, which **removed the `$use`
middleware API**. `packages/server/src/db/index.ts` registers a `$use` middleware
(BigInt→string conversion at the DB boundary), so the server throws at boot
(`prisma.$use is not a function`). The patch ports that one middleware to the
`$extends` query API (`query.$allModels.$allOperations`), preserving behaviour:
like `$use`, `$allOperations` does not intercept raw `$queryRaw`/`$executeRaw`
calls, and the production singleton caches the *base* client (extensions are
re-derived). Pure client-API migration, no logic change.

### 2. `0002-package-json-sharp-neverbuilt.patch`: `sharp` marked never-built (plus the sandbox stub)

`0002` adds `"pnpm": { "neverBuiltDependencies": ["sharp"] }` to the root
`package.json`. `sharp` (native libvips, pulled transitively) is on the
**image / document-ingest path only, never the memory path**, and its native
postinstall build is the flaky part. Marking it never-built is harmless for
the benchmark and avoids a postinstall failure blocking `pnpm install`.

Separately, the bring-up needed a **sandbox-specific `sharp` runtime stub**
(this host's egress proxy blocks `sharp`'s libvips binary download, so even the
JS shim must be neutralised at runtime). That stub is **NOT** committed here: it is
an artifact of *this* sandbox's network policy, not an upstream gap, and injecting
a stub into `node_modules` on a normal host would mask a real problem. It is
therefore **host-smoke-only**. If `pnpm install` on a restricted host still trips
over `sharp` at runtime, stub `node_modules/sharp/lib/sharp.js` to export no-ops.
The Docker build has open egress, so it does not need the stub; `neverBuiltDependencies`
is enough there.

### 3. `schema.introspected.prisma`: shipped schema is a stale subset

The shipped `packages/server/prisma/schema.prisma` describes **27 of the 67
tables** the migrations actually create, so the generated Prisma client rejects
columns the code writes: first failure `Unknown argument conversationId` on
`message.createMany`. `schema.introspected.prisma` is a full `prisma db pull`
introspection of the migrated database (67 models), captured from the working
build. The build **replaces** `schema.prisma` with it before `prisma generate` so
the client matches the real (migrated) schema. It is a build artifact, not a
`.patch`, because it is a wholesale replacement, not a hunk edit (diffing 702→1742
lines would be unreadable and fragile).

### 4. `seed.sql`: fresh-deploy bootstrap gap (no default org/user)

The server auto-creates a project on first request; that row FKs to
`organizations(id)`, whose `ownerId` FKs to `users(id)`. A pristine
`migrate deploy` leaves both tables empty, so the first ingest 500s. `seed.sql`
idempotently (`ON CONFLICT DO NOTHING`) inserts the default org + owner user:
the exact two rows the live bring-up created (captured from the `retaindb_smoke`
DB). Applied by the entrypoint / `serve_local.sh` **after** `migrate deploy`,
**before** the server starts. Safe to re-run (score/summarize reruns reuse the db).

### 5. `ENCRYPTION_KEY`: required to boot (handled in the entrypoints, not here)

Boot fails without a non-empty `ENCRYPTION_KEY` (≥32 chars; it guards agent-task
**connector credential** encryption, not the memory path). Not a file patch:
`entrypoint.retaindb-server.sh` and `serve_local.sh` default it to a documented,
non-secret dev value (`ENCRYPTION_KEY` env-overridable for a real deployment).

### 6. `0003-search-response-temporal-nested.patch`: search drops temporal (upstream bug)

`packages/server/src/api/memory.ts` (search response builder, ~L862) reads
`r.memory.documentDate` / `eventDate` / `validFrom` / `validUntil` at the **top
level** of the engine's memory object, but the engine
(`engine/memory/search.ts`, all three mapping blocks) returns those **nested under
`memory.temporal`**. So `temporal.document_date` was `null` in **every** search
response even though the DB stored correct dates (verified live: 176/176 memories
had `documentDate`; the response had all nulls). Since the adapter maps
`temporal.document_date → created_at`, this poisoned the scorer's temporal signal.
The patch reads `r.memory.temporal?.documentDate ?? r.memory.documentDate` (same
for the other three) so it works with both shapes. `api/memory.ts` only; minimal.
This is an **upstream bug reported as a finding**. No behaviour is changed beyond
returning the dates the DB already holds.

### 7. `0004-disable-search-cache-env.patch`: env knob to bypass the search caches

`engine/memory/search.ts` `searchMemories` has two caches that corrupt benchmark
retrieval:

- **Semantic cache** (`getFromSemanticCache(queryEmbedding)`) is keyed **only** on
  query-embedding similarity (threshold 0.85, `engine/cache.ts`) with **no
  project/user/question_date scoping**: similar-sounding queries return another
  scope's cached results wholesale (**cross-tenant leakage**; the live eval log
  showed hits at 0.915-0.986 across personas).
- **Exact-key cache** (`cacheKey`) **omits `question_date`**, so MemConflict's
  re-asked dynamic questions (same text, later logical date, all within the 300s
  TTL because the benchmark compresses months into seconds) get **stale results**
  and bypass temporal reranking.

The patch adds `const SEARCH_CACHE_DISABLED = /^true$/i.test(process.env.RETAINDB_DISABLE_SEARCH_CACHE || "false")`
and guards **both** cache reads and **all** cache writes (the simple get/set, the
semantic get/set, the empty-results set, the early-exit set, and the final set) with
it. **Cache key shapes and thresholds are unchanged**: the knob only bypasses
get/set. Default `false` = vendor behaviour unchanged; the benchmark entrypoints
export `RETAINDB_DISABLE_SEARCH_CACHE=true`.

This is a **harness-artifact mitigation, not purely an upstream bug**: the
global-embedding-keyed semantic cache *is* a cross-tenant bug upstream, but the
TTL/exact-key staleness only bites because the benchmark compresses logical months
into wall-clock seconds; a production deployment would rarely re-ask the same
question within the 300s TTL. The knob preserves vendor default behaviour when unset.

### 8. `post_migrate.sql`: drop empty-table ivfflat indexes (recall killer)

**This was the dominant retrieval-recall corruption in the first two smokes.** The
vendor migrations create ivfflat ANN indexes (`lists=100`) on the vector columns
(`memories_embedding_idx`, `chunks_embedding_idx`, `entities_embedding_idx`) while
the tables are **empty**. ivfflat trains its centroids from the rows present at
build time, so an index built on an empty table has degenerate/untrained centroids;
with pgvector's default `ivfflat.probes = 1` the search then scans a single,
effectively-random cluster and **misses the true nearest neighbours**.

Verified live: the true top-2 evidence memories (cosine **0.657 / 0.629**) never
appeared under the index scan (top hit only **0.149-0.595**, and **unstable across
runs**), whereas an **exact** scan (or `ivfflat.probes = 100`) returns them at
**ranks 1-2**. The server's Prisma `$queryRaw` retrieval path exposes no hook to
`SET ivfflat.probes` per session, so the fix is deployer-side: **drop the indexes**.
At benchmark scale (a few hundred to a few thousand rows per per-run DB) an exact
KNN scan is sub-millisecond, and correctness beats ANN. `post_migrate.sql` is
`DROP INDEX IF EXISTS ...` (idempotent), applied after `migrate deploy` (by
`apply_seed.py` in the container, `psql` on the host, the same mechanism as the
seed).

Upstream deploy gap reported as a finding, not a code change. **A production
deployment at scale would instead rebuild ivfflat AFTER bulk load** (so centroids
train on real data) and tune `ivfflat.probes`, or use an HNSW index (no
centroid-training dependency); dropping is correct only because the benchmark's
per-run DBs are tiny and disposable.

### 9. `0005-search-filter-malformed-rows.patch`: drop malformed search rows instead of 500ing

A rare `searchMemories` path can emit a result row without a `.memory` object;
the `/v1/memory/search` response builder then throws (`Cannot read properties of
undefined (reading 'id')`) and the whole request 500s. Observed roughly once per
~500 searches, and only after the ivfflat drop (exact KNN widened the candidate
flow into previously-starved code paths). The patch filters such rows out with a
`console.warn` before formatting, preserving the response contract for the valid
rows. Upstream bug, reported as a finding. The benchmark client additionally
retries 5xx responses (adapter-side) so one transient failure cannot kill a
multi-hour generate stage.

## Regenerating / verifying the patches

The `.patch` files are `diff -u` against the pristine submodule with `a/`/`b/`
prefixes, so they apply with either `git apply -p1` or `patch -p1` from the copied
tree root. To re-verify they apply cleanly against pristine source:

```bash
T=$(mktemp -d); cp -a external/RetainDB/. "$T/"; rm -rf "$T/.git"; cd "$T"
for p in <repo>/retaindb_server/server_patches/000*.patch; do git apply --check -p1 "$p"; done
```

`<repo>` above is the path to this repository's root on the machine running the
command (the checkout containing `retaindb_server/`).
