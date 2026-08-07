-- Fresh-deploy seed for @retaindb/server (upstream ships none).
--
-- WHY: the server auto-creates a project on the first request whose row FKs to
-- organizations(id); organizations.ownerId in turn FKs to users(id). A pristine
-- `prisma migrate deploy` leaves both tables EMPTY, so the very first ingest 500s
-- on the missing default org/user. This seeds exactly the two rows the bring-up
-- agent's live server created (captured from the retaindb_smoke DB), so project
-- auto-creation resolves its foreign keys.
--
-- Idempotent (ON CONFLICT DO NOTHING) so it is safe to re-run on every boot and on
-- a reused db (score/summarize reruns). Column set = the NOT-NULL-without-default
-- columns only; everything else takes its schema default. Insert users FIRST
-- (organizations.ownerId -> users.id).

INSERT INTO users (id, email, name, "updatedAt")
VALUES ('default-user', 'default@retaindb.local', 'default', CURRENT_TIMESTAMP)
ON CONFLICT (id) DO NOTHING;

INSERT INTO organizations (id, name, slug, "ownerId", "updatedAt")
VALUES ('default', 'Default', 'default', 'default-user', CURRENT_TIMESTAMP)
ON CONFLICT (id) DO NOTHING;
