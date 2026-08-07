#!/usr/bin/env python3
"""Apply the post-migrate SQL files to the RetainDB server database over asyncpg.

Runs, in order, against ``DATABASE_URL``:
  1. seed.sql          — default org and owner user that the server's project
                         auto-creation references by foreign key (upstream
                         ships no seed).
  2. post_migrate.sql  — drops the empty-table ivfflat indexes, so retrieval
                         uses an exact KNN scan (fix #7: an ivfflat index
                         built on empty tables, with probes=1, collapses ANN
                         recall). See server_patches/README.md.
  3. clocksync_created_at.sql — applied only when BENCH_CLOCKSYNC=1. A
                         BEFORE INSERT trigger forces memories.createdAt onto
                         the faked postmaster clock, because Prisma's
                         client-side @default(now()) otherwise escapes
                         libfaketime through the query engine's vDSO. Off by
                         default, so shared-hindsight-pg runs stay
                         byte-identical.

entrypoint.retaindb-server.sh uses this script, because the node:20 image
has no ``psql`` but ships asyncpg, a harness dependency. asyncpg's
``Connection.execute`` with no arguments uses the simple-query protocol,
which runs multiple semicolon-separated statements in one call, so each file
applies atomically. serve_local.sh (host) instead uses ``psql -f`` on the
same files. This script is the container-side equivalent. Every file is
idempotent (``ON CONFLICT DO NOTHING`` or ``IF EXISTS``), so re-running it on
every boot, or against a reused database, is safe.
"""
import asyncio
import os
import sys

try:
    import asyncpg
except ImportError as exc:  # pragma: no cover - container always has it
    print(f"[retaindb-server] FATAL: asyncpg unavailable for post-migrate SQL: {exc}",
          file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
SQL_FILES = ["seed.sql", "post_migrate.sql"]
# Clock-sync arm: force createdAt onto the faked DB clock (see the file
# header). Gated, so default runs never touch the shared hindsight-pg schema.
if os.environ.get("BENCH_CLOCKSYNC") == "1":
    SQL_FILES.append("clocksync_created_at.sql")
DATABASE_URL = os.environ.get("DATABASE_URL")


async def main() -> None:
    if not DATABASE_URL:
        print("[retaindb-server] FATAL: DATABASE_URL not set for post-migrate SQL", file=sys.stderr)
        sys.exit(1)
    conn = await asyncpg.connect(dsn=DATABASE_URL)
    try:
        for name in SQL_FILES:
            path = os.path.join(HERE, name)
            with open(path, "r", encoding="utf-8") as f:
                sql = f.read()
            await conn.execute(sql)
            print(f"[retaindb-server] applied {name} (idempotent)", flush=True)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
