-- Post-migrate index management for @retaindb/server (deployer-side, NOT a code change).
--
-- WHY: the vendor migrations create ivfflat ANN indexes (lists=100) on the vector
-- columns while the tables are EMPTY:
--     memories_embedding_idx, chunks_embedding_idx, entities_embedding_idx
-- An ivfflat index trains its centroids from the rows present AT BUILD TIME, so an
-- index built on an empty table has degenerate/untrained centroids. Combined with
-- pgvector's default ivfflat.probes = 1, ANN recall then COLLAPSES: the search
-- scans a single, effectively-random cluster and misses the true nearest neighbours.
--
-- VERIFIED LIVE (first two smokes): the true top-2 evidence memories (cosine
-- 0.657 / 0.629) never appeared under the index scan (top hit only 0.149-0.595 and
-- UNSTABLE across runs), while an exact scan — or ivfflat.probes = 100 — returns
-- them at ranks 1-2. This was the DOMINANT retrieval-recall corruption behind the
-- near-zero smoke scores. The server's Prisma $queryRaw retrieval path exposes no
-- hook to `SET ivfflat.probes` per session, so the fix is deployment-side.
--
-- FIX: drop the ivfflat indexes. At benchmark scale (a few hundred to a few
-- thousand rows per per-run database) an EXACT KNN scan is sub-millisecond, and
-- correctness beats ANN here. Idempotent (IF EXISTS) so it is safe on every boot
-- and on a reused db.
--
-- PRODUCTION NOTE: a real deployment at scale would NOT drop these — it would build
-- the ivfflat index AFTER bulk load (so centroids train on real data) and tune
-- ivfflat.probes, or use an HNSW index (no centroid-training dependency). Dropping
-- is correct ONLY because the benchmark's per-run DBs are tiny and disposable.

DROP INDEX IF EXISTS memories_embedding_idx;
DROP INDEX IF EXISTS chunks_embedding_idx;
DROP INDEX IF EXISTS entities_embedding_idx;
