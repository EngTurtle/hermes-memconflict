#!/usr/bin/env python3
"""Create a per-run Hindsight database on the shared Postgres. This script is
idempotent.

entrypoint.hindsight.sh calls this script in HINDSIGHT_PG_MODE=shared, before
it exports the daemon URL, to give each run its own database. Per-run
isolation was added on 2026-07-20. See the entrypoint's shared-mode block for
the fairness reason.

The hindsight image ships no `psql`, but it ships `asyncpg`, a Hindsight
dependency. So this script creates the database over a Postgres protocol
connection to the always-present maintenance `postgres` database.

Behavior:
  * fresh database    -> CREATE DATABASE succeeds. This logs "created".
  * existing database -> this catches DuplicateDatabaseError and logs
                          "already exists". The script stays idempotent.
  * unreachable server -> this retries up to 10 times, 3 seconds apart, then
                          prints a clear message and exits 1. Continuing
                          instead would let the daemon boot a confusing pg0
                          fallback, or die with "Database URL is required".

This script also creates the vector and pg_trgm extensions in the new
database, as a belt-and-suspenders step on top of template1 inheritance
(hindsight-pg-init.sql seeds template1). The pgvector image ships both
extensions, so CREATE EXTENSION always succeeds here.

The entrypoint sets the connection parameters through env variables:
  HS_PG_HOST HS_PG_PORT HS_PG_USER HS_PG_PASSWORD HS_PG_DB
"""
import asyncio
import os
import sys

# asyncpg is a Hindsight runtime dependency. This try/except guards the
# import, so a host-side syntax check with py_compile never needs asyncpg
# installed. Only the container runtime needs it.
try:
    import asyncpg
except ImportError as exc:  # pragma: no cover - container always has it
    print(f"[hindsight] FATAL: asyncpg unavailable for per-run db creation: {exc}",
          file=sys.stderr)
    sys.exit(1)

HOST = os.environ["HS_PG_HOST"]
PORT = int(os.environ["HS_PG_PORT"])
USER = os.environ["HS_PG_USER"]
PASSWORD = os.environ["HS_PG_PASSWORD"]
DBNAME = os.environ["HS_PG_DB"]

CONNECT_RETRIES = 10
CONNECT_DELAY_S = 3
CREATE_RETRIES = 10
CREATE_DELAY_S = 2


def quote_ident(name: str) -> str:
    """Double-quote an identifier as a defensive step, even though the
    caller already sanitizes it."""
    return '"' + name.replace('"', '""') + '"'


async def connect(database: str):
    """Connect with a bounded retry loop. Returns a connection, or exits with
    a non-zero status."""
    last_err = None
    for _ in range(CONNECT_RETRIES):
        try:
            return await asyncpg.connect(
                host=HOST, port=PORT, user=USER, password=PASSWORD, database=database)
        except Exception as err:  # noqa: BLE001 - report and retry any connect error, whatever its type
            last_err = err
            await asyncio.sleep(CONNECT_DELAY_S)
    print(f"[hindsight] FATAL: cannot reach postgres at {HOST}:{PORT} "
          f"(db={database!r}) after {CONNECT_RETRIES} tries: {last_err}",
          file=sys.stderr)
    sys.exit(1)


async def main() -> None:
    admin = await connect("postgres")
    try:
        last_err = None
        for _ in range(CREATE_RETRIES):
            try:
                await admin.execute(f"CREATE DATABASE {quote_ident(DBNAME)}")
                print(f"[hindsight] created database {DBNAME}", flush=True)
                break
            except asyncpg.exceptions.DuplicateDatabaseError:
                print(f"[hindsight] database {DBNAME} already exists", flush=True)
                break
            except Exception as err:  # noqa: BLE001
                # This error is often transient. For example, "source
                # database template1 is being accessed by other users" can
                # happen. Sibling shard daemons can create their own
                # databases at the same time. This retries a few times
                # before it gives up.
                last_err = err
                await asyncio.sleep(CREATE_DELAY_S)
        else:
            print(f"[hindsight] FATAL: could not create database {DBNAME}: {last_err}",
                  file=sys.stderr)
            sys.exit(1)
    finally:
        await admin.close()

    # As a belt-and-suspenders step, this makes sure the extensions exist in
    # the database, even if template1 inheritance did not apply. For example,
    # HINDSIGHT_PG_DB might point to a pre-existing database created before
    # template1 was seeded.
    conn = await connect(DBNAME)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
