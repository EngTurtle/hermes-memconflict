"""Dump RetainDB-server `memories` rows for the relative-date probe.

Run this script inside the retaindb-server probe container, after you source
the entrypoint. Sourcing first makes DATABASE_URL point at the run's local
clocksync Postgres, while the postmaster is still up. The entrypoint's EXIT
trap has not fired yet at that point.

    docker compose run --name probe_reldate_rdb --entrypoint bash retaindb-server \
      -c 'source /app/benchmark/docker/entrypoint.retaindb-server.sh \
          && python3 /app/benchmark/probes/dump_retaindb_probe.py'

This script prints every row's content, type, temporal columns, and metadata
as JSON lines. It skips embedding vectors. The script is read-only. Its
output is probe evidence only and is never banked as a result.
"""
import asyncio
import json
import os

import asyncpg

SKIP_COLS = {"embedding"}


async def main() -> None:
    url = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(url)
    try:
        cols = [r["column_name"] for r in await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'memories' ORDER BY ordinal_position")]
        keep = [c for c in cols if c not in SKIP_COLS]
        col_sql = ", ".join(f'"{c}"' for c in keep)
        rows = await conn.fetch(
            f'SELECT {col_sql} FROM memories ORDER BY "createdAt"')
        print(f"[dump] memories rows={len(rows)} cols={keep}", flush=True)
        for row in rows:
            out = {}
            for c in keep:
                v = row[c]
                out[c] = v.isoformat() if hasattr(v, "isoformat") else v
            print("MEMROW\t" + json.dumps(out, ensure_ascii=False, default=str),
                  flush=True)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
