"""Logical-clock stepping for libfaketime-preloaded provider processes.

The clock-sync arms (BENCH_CLOCKSYNC=1) run selected provider processes under
libfaketime (LD_PRELOAD plus FAKETIME_TIMESTAMP_FILE). This makes their
perceived OS clock track the dataset's logical session date instead of
benchmark wall time. This module is the only writer of the timestamp file.
The shared driver calls set_clock(...) once per session, right before ingest,
and the session's ingest and question answering then run under that clock.
Which processes are preloaded is a per-provider decision made in the
entrypoints and adapters (see benchmark/docker/clock_sync.sh). This module
never sets LD_PRELOAD.

File format (probed 2026-07-24 on Debian bookworm libfaketime 0.9.10, see
docs/DECISIONS.md): the absolute "@%Y-%m-%d %H:%M:%S" form advances at 1x and
re-anchors exactly to the written instant on each rewrite (libfaketime anchors
"start at" to the file's mtime). This holds whether the rewrite comes from the
faked process itself (Mnemosyne) or from outside it (Supermemory or RetainDB).
"+0" means zero offset, that is, real time. It seeds the file so anything that
boots before the first step warm-boots at the real clock (TLS and model
caches).

Every function is a no-op unless BENCH_CLOCKSYNC=1 and BENCH_CLOCKSYNC_FILE
are both set, so default runs stay byte-identical with this module imported.
When both are set, set_clock fails closed on an unparseable session timestamp:
it raises and kills the shard rather than leaving the previous session's clock
in place. See set_clock.
"""

import os
from datetime import datetime, timezone
from typing import Optional

# Debian path for the apt libfaketime package. Override with BENCH_LIBFAKETIME.
DEFAULT_LIBFAKETIME = "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"

# The one definition of the recall-time "now" hour, in UTC. The faked OS clock
# (set_clock below) and the explicit recall-time parameters some providers
# pass (eval_common.Parse_Query_Now_Timestamp imports this constant) both
# anchor to it, so parameter-based and OS-clock providers can never drift
# apart. Noon keeps perceived "now" ahead of provider-internal ingest row
# timestamps, which anchor at midnight plus per-turn or per-exchange minute
# bumps.
RECALL_NOW_HOUR_UTC = 12


def clock_sync_enabled() -> bool:
    return (
        os.environ.get("BENCH_CLOCKSYNC") == "1"
        and bool(os.environ.get("BENCH_CLOCKSYNC_FILE"))
    )


def _format_faketime(target: datetime) -> str:
    return target.strftime("@%Y-%m-%d %H:%M:%S")


def _write_timestamp_file(payload: str) -> str:
    # Replaces the file atomically. Preloaded readers re-parse it on every
    # clock call (FAKETIME_NO_CACHE=1), and must never see a torn write.
    path = os.environ["BENCH_CLOCKSYNC_FILE"]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(payload + "\n")
    os.replace(tmp, path)
    return payload


def set_clock(dt: Optional[datetime]) -> Optional[str]:
    """Step the perceived clock to dt's date at 12:00 UTC.

    Returns the written payload, or None when clock-sync is disabled.

    Fails closed when clock-sync is enabled. A session whose timestamp did not
    parse would otherwise run under the previous session's clock, silently
    contaminating that session and every one after it in the shard with a
    wrong logical "now". A contaminated result is worse than no result, so
    this aborts instead."""
    if not clock_sync_enabled():
        return None
    if dt is None:
        raise RuntimeError(
            "[clocksync] clock-sync enabled but session timestamp unparseable - "
            "aborting shard rather than reusing the previous session's clock"
        )
    # Noon convention: session dates carry no time of day. Anchoring
    # recall-time "now" at 12:00 keeps perceived now ahead of
    # provider-internal row timestamps, such as Mnemosyne's midnight-anchored
    # plus 1h1m monotonic bumps.
    target = datetime(dt.year, dt.month, dt.day, RECALL_NOW_HOUR_UTC, 0, 0, tzinfo=timezone.utc)
    return _write_timestamp_file(_format_faketime(target))


def seed_real_time() -> Optional[str]:
    """Reset the perceived clock to real time. "+0" means zero offset."""
    if not clock_sync_enabled():
        return None
    return _write_timestamp_file("+0")
