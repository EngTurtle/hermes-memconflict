-- Shared Hindsight Postgres: one-time cluster init (runs only when the
-- hindsight_pg data volume is EMPTY, against the POSTGRES_DB database).
--
-- Pre-create the extensions Hindsight's default config needs so the FIRST
-- daemon's migration finds them already present:
--   * vector  (pgvector) — HINDSIGHT_API_VECTOR_EXTENSION default "pgvector";
--                          the initial_schema migration runs CREATE EXTENSION
--                          IF NOT EXISTS vector.
--   * pg_trgm             — migration c1a2b3d4e5f6 (GIN trigram index on
--                          entities.canonical_name) runs CREATE EXTENSION
--                          IF NOT EXISTS pg_trgm.
-- Both statements inside Hindsight's migrations are already idempotent and
-- serialized by a pg advisory lock, so this is belt-and-suspenders: it removes
-- any first-boot herd race when 5-10 shard daemons connect and migrate at once.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Per-run database isolation (added 2026-07-20): each run now gets its OWN
-- database (hindsight_<run_tag>), created on demand by the entrypoint, instead
-- of every run sharing this POSTGRES_DB. New databases are created from the
-- template1 template, so seed the SAME extensions into template1 here — every
-- future per-run database then INHERITS vector + pg_trgm at creation time with
-- no extra step. (The entrypoint's creation script also runs CREATE EXTENSION
-- IF NOT EXISTS in each new db as belt-and-suspenders, e.g. for a db that
-- pre-dates this seeding or an explicit HINDSIGHT_PG_DB override.)
\connect template1
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
