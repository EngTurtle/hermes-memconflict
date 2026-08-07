"""MemConflict evaluation adapter for the RetainDB server edition.

The shared ``benchmark/eval_common.py`` driver runs the provider-agnostic
pipeline: dataset iteration, dialogue flattening, the answer LLM call,
results-row emission, and compaction. This file adds only the
RetainDB-server binding. It attaches to a running server, ingests each
session, and recalls per question.

DIFFERENCES FROM THE retaindb/ LOCAL EDITION
---------------------------------------------
``retaindb/`` benchmarks ``@retaindb/local``, a heuristic Node bundle with no
database and no LLM. This folder benchmarks ``@retaindb/server``
(``@retaindb/server@1.0.0``), a different product:

  * INGESTION USES LLM EXTRACTION, NOT HEURISTICS. ``POST
    /v1/memory/ingest/session`` runs an LLM extraction pass
    (``EXTRACTOR_MODEL``) over the session and stores the extracted memories
    in Postgres with pgvector. Ingesting a session therefore spends LLM
    calls, unlike the local edition, where only the answer and judge calls
    used an LLM.
    Warning: the vendor's ``.env.example`` advertises ``EXTRACTION_MODEL``.
    The code never reads that variable
    (``engine/memory/extractor.ts`` reads ``EXTRACTOR_MODEL``, with a
    fallback chain of INFERENCE_MODEL, then EXTRACTOR_MODEL, then
    OPENAI_MODEL, then gpt-5.4-mini). Setting ``EXTRACTION_MODEL`` has no
    effect. The entrypoint sets ``EXTRACTOR_MODEL``.

  * RETRIEVAL RANKS BY VECTOR SEARCH FIRST, THEN BY TIME. ``POST
    /v1/memory/search`` ranks over ``vector(1024)`` pgvector embeddings,
    with a lexical fallback and merge, plus a ``question_date``-driven
    temporal rank. The relevance field in the response is ``similarity``,
    not ``score``. Memories carry a real ``temporal`` block. Embeddings come
    from the contract embedder, the shared ``vllm-embed`` /
    ``bge-small-en-v1.5`` model at 384 dimensions, zero-padded to 1024 by
    ``embed_proxy.py`` (see docs/DECISIONS.md). The ``EMBEDDING_MODE=local``
    bge-large path is off contract.

  * TIME IS TRUE TEMPORAL TIME. Each message's ISO ``timestamp`` flows into
    the stored memory's ``temporal.document_date`` and ``event_date``
    fields. The local edition instead uses wall-clock ``created_at``, copied
    into metadata. We therefore pass the dataset session ``Date`` as every
    message's ``timestamp``, and we pass the current session's date as the
    recall ``question_date``. This makes the server's temporal ranking treat
    the benchmark chronology as its "now", instead of the wall clock. See
    ``recall`` and ``_QUESTION_DATE_KEY`` below.

  * THE ADAPTER ONLY ATTACHES TO THE SERVER. It never starts a node
    process. A shell script starts the server and its ``prisma migrate
    deploy`` step (the Docker entrypoint, or ``serve_local.sh``).
    ``begin_run`` waits for the server to become healthy and fails with a
    clear message if it does not.

Each persona gets its own ``project`` slug, RetainDB's tenancy boundary,
created automatically. Ingest and search also send a stable ``user_id``, so
USER-scoped memories attach and retrieve together.
"""

import argparse
import asyncio
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# asyncpg is present in the container image (used by server_patches/apply_seed.py).
# It is the observable for the featured session-lifecycle wait. The adapter
# polls the shared Postgres directly for the summary and promotion rows the
# scheduler writes, because there is no REST lifecycle-status endpoint and the
# search API does not expose a memory's metadata. The guard lets a host smoke
# without asyncpg or DATABASE_URL report the wait as disabled, instead of
# crashing.
try:  # pragma: no cover - always present in the Docker image
    import asyncpg
except Exception:  # noqa: BLE001
    asyncpg = None

# --- Wire in the upstream MemConflict Evaluation helpers -------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMCONFLICT_EVAL_DIR = os.environ.get(
    "MEMCONFLICT_EVAL_DIR",
    os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Evaluation"),
)
MEMCONFLICT_EVAL_DIR = os.path.abspath(MEMCONFLICT_EVAL_DIR)
if MEMCONFLICT_EVAL_DIR not in sys.path:
    sys.path.insert(0, MEMCONFLICT_EVAL_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# The shared harness modules (eval_common, llm_reasoning, the scorers) live in
# ../benchmark. Add that path so imports work regardless of the launch cwd.
_SHARED_HARNESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark"))
if _SHARED_HARNESS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_HARNESS_DIR)

from dotenv import load_dotenv

import eval_common  # noqa: E402
from eval_common import (  # noqa: E402  (re-exports keep old imports working)
    Pair_Exchange_Turns,
    Parse_Query_Now_Timestamp,
    Parse_Session_Timestamp,
    ProviderBinding,
    add_common_eval_args,
    record_provider_retrieval,
)

# Attach-only REST client (health-wait, no process spawning).
from _retaindb_server_client import RetainDBServerClient  # noqa: E402

load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# ctx key that holds the current session's ISO date. ingest_session sets it and
# recall() reads it back.
_QUESTION_DATE_KEY = "question_date_iso"

# The Hermes plugin's search payload is project/query/user_id/session_id/top_k/
# include_pending. It sends no date
# (external/hermes-agent/plugins/memory/retaindb/__init__.py:230-238). This
# adapter used to send `question_date`, a parameter no Hermes deployment ever
# sends. The default is now OFF. Instead, the benchmark must put the server's
# clock at the question date (BENCH_CLOCKSYNC does this), rather than pass a
# date the plugin does not send (user ruling 2026-07-28).
#
# Verified equivalent on a live store the same day: with the container clock
# faked to 2022-10-12, a temporal query returned the same row, and excluded the
# same older ones, whether `question_date` was omitted or set to the faked
# date. The server's fallback "now" falls inside the faked time domain.
#
# Without clock-sync, the fallback is real wall-clock time (2026), which would
# empty every temporally-worded recall against a 2022-2025 dataset. Omitting
# the date is correct only under clock-sync. Running without clock-sync fails
# closed below. Set RETAINDB_SEND_QUESTION_DATE=1 for a deliberate diagnostic
# contrast.
_SEND_QUESTION_DATE = os.environ.get("RETAINDB_SEND_QUESTION_DATE", "0") == "1"


# --------------------------------------------------------------------------
# Hermes-plugin retrieval overlay (what the plugin actually hands the agent)
# --------------------------------------------------------------------------
# Ported unchanged in spirit from ``retaindb/eval_retaindb.py``. The thing under
# test is the plugin's behavior as the Hermes memory provider (CLAUDE.md Goal):
# the memory text the plugin puts in front of the agent at recall time, not the
# raw row the backend stores. The Hermes retaindb plugin's ``_build_overlay``
# collapses whitespace in each recalled memory, truncates it to 320 characters,
# then removes duplicates on a normalized key while keeping order. This recall
# shaping does not depend on the backend, so it applies the same way to the
# server edition.
#
# One text, two consumers: the compacted string goes into both the answer
# context and the stored ``Retrieved_Memories[].memory``. Never split them.
# Feeding the judge untruncated text the agent never saw would inflate SEH@K
# and misbook a compaction failure as a reasoning failure in EUG-cond@5. See
# docs/DECISIONS.md for the full rationale, carried over verbatim.
_PLUGIN_COMPACT_CHAR_LIMIT = 320
# The plugin's paramless query slices the first 5 rows (``results[:5]``). The
# server's default top_k is 10 and it slices after ranking. Requesting 10 and
# slicing to the harness top_k ourselves reproduces the plugin's ranking (the
# first top_k rows match) and keeps the shared top-K as the fairness line.
_PLUGIN_QUERY_SERVER_TOPK = 10


def _plugin_compact(text: Any) -> str:
    """Reproduce the plugin's ``_compact`` literally."""
    return re.sub(r"\s+", " ", str(text or "")).strip()[:_PLUGIN_COMPACT_CHAR_LIMIT]


def _plugin_norm(text: Any) -> str:
    """Build the plugin's ``_norm`` dedup key from an already-compacted string.

    This normalizes the compacted string, so two rows that differ only past
    character 320 dedup to one item.
    """
    return re.sub(r"[^a-z0-9 ]", "", _plugin_compact(text).lower())


def _env_flag(name: str, default: bool) -> bool:
    """Read an env-var flag for arm selection. Unset returns ``default``.

    This stays local because the shared driver has no such helper, and
    adapters must not add to it. ``0/false/no/off/""`` are false. Anything
    else is true.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _iso(timestamp: Optional[datetime]) -> Optional[str]:
    # The server validates message `timestamp` and search `question_date` with
    # zod `z.string().datetime()`. By default this rejects a numeric timezone
    # offset such as `+00:00` and accepts only a `Z` suffix. Plain
    # `datetime.isoformat()` on a UTC-aware datetime emits `+00:00`, which
    # causes an HTTP 400 on every ingest and search. Normalize to UTC and
    # render with a `Z` suffix instead.
    if not timestamp:
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------
def Add_Session_Dialogue_To_RetainDB_Server(
    client: RetainDBServerClient,
    project: str,
    user_id: str,
    session_label: str,
    server_session_id: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
    granularity: str = "session",
    promotion_mode: Optional[str] = None,
) -> Tuple[float, int, int, int, int, int]:
    """Ingest one session's dialogue with one or more sync ``ingest/session`` calls.

    ``granularity="session"`` (minimal, the default) sends the whole session
    in a single ingest/session call. This mirrors the retaindb/ local
    adapter's session-granularity date rule: one session-level timestamp (the
    dataset session ``Date``, in UTC ISO form) is stamped on every message, so
    both RetainDB editions ingest dialogue with the same timing.

    ``granularity="exchange"`` (featured, the Hermes plugin cadence) sends one
    ingest/session call per completed exchange. Each call carries exactly the
    two verbatim messages of that user/assistant turn, as
    ``{role, content, timestamp}``, with ``write_mode="sync"``. This
    faithfully reproduces the plugin's ``sync_turn``
    (``hermes-agent/plugins/memory/retaindb/__init__.py:628-640``, which
    queues ``[{user},{assistant}]`` per turn, and the write-behind queue
    drains with one ``ingest_session`` call each, ``:381-408``). Exchanges
    with an empty user turn are dropped, because ``sync_turn`` returns early
    on falsy ``user_content``. All exchanges of the session share the
    session's ``Date`` timestamp, the finest time resolution the dataset
    provides, and the same ``server_session_id``, so the server's
    session-lifecycle groups them as one session. The benchmark calls each
    ingest synchronously, the same drained-quality rule used for every other
    provider. The plugin's SQLite write-behind queue is an operational detail
    this adapter does not reproduce.

    The server schema requires every message's ``timestamp``. It flows into
    ``temporal.document_date`` and ``event_date``, so it is genuinely
    temporal.

    Returns (add_duration_ms, created, relations, errors, ingest_calls,
    dropped_exchanges).
    """
    if not dialogue_messages:
        return 0.0, 0, 0, 0, 0, 0
    ts_iso = _iso(timestamp)

    if granularity == "exchange":
        groups = Pair_Exchange_Turns(dialogue_messages)
    else:
        groups = [dialogue_messages]

    start = time.time()
    created = relations = errors = ingest_calls = dropped = 0
    for group in groups:
        if granularity == "exchange":
            # Plugin faithfulness: `sync_turn` returns early on falsy
            # user_content, so a turn with no non-empty user content is
            # never synced.
            user_msg = next((m for m in group if m.get("role") == "user"), None)
            if user_msg is None or not str(user_msg.get("content", "")).strip():
                dropped += 1
                continue
        messages = [
            {"role": m["role"], "content": m["content"], "timestamp": ts_iso}
            for m in group
        ]
        print(f"[DEBUG] project={project} session {session_label} "
              f"ingest_call msgs={len(messages)} granularity={granularity} write_mode=sync")
        try:
            resp = client.ingest_session(
                project=project,
                session_id=server_session_id,
                messages=messages,
                user_id=user_id,
                write_mode="sync",
                promotion_mode=promotion_mode,
            )
            created += int(resp.get("memories_created", 0) or 0)
            relations += int(resp.get("relations_created", 0) or 0)
            _errs = resp.get("errors")
            errors += len(_errs) if isinstance(_errs, list) else int(_errs or 0)
            ingest_calls += 1
        except Exception as e:  # pragma: no cover
            print(f"[DEBUG] ingest failed project={project} session={session_label} "
                  f"group_len={len(messages)}: {e}")
            errors += len(messages)
            ingest_calls += 1

    return (time.time() - start) * 1000.0, created, relations, errors, ingest_calls, dropped


# --------------------------------------------------------------------------
# Featured consolidation: session-lifecycle wait
# --------------------------------------------------------------------------
# The featured arm runs the server's 60-second scheduler
# (DISABLE_SCHEDULER=false) with a lowered SESSION_INACTIVITY_THRESHOLD_MS, so
# runSessionLifecycle() promotes each cold session's eligible SESSION memories
# to USER scope and writes a USER-scoped session summary
# (external/RetainDB .../engine/memory/session-lifecycle.ts). The driver
# ingests session i, answers it, then ingests session i+1. So session i's
# promotions and summaries must land before session i+1 is answered. This
# poll is that barrier.
#
# Observable: findStaleSessions excludes a session once a summary row exists
# for it (session-lifecycle.ts:81-88). That same row is our "processed"
# signal. We poll the shared Postgres for the exact condition the scheduler
# uses:
#   * has_summary  — a USER-scoped row with metadata.session_summary='true'
#                    and source_session_id=<this session>. The server writes
#                    this when the session had at least
#                    SESSION_SUMMARY_MIN_MEMORIES (default 2) memories.
#   * has_promotion — any row for the session with
#                    metadata.promoted_from_session='true' (a SESSION-to-USER
#                    scope flip). This is the other lifecycle side effect,
#                    caught for sessions with a summary-ineligible count that
#                    still had promotable memories.
#   * total_active  — all active memories for the session, in any scope.
#                    This is the exact population generateSessionSummary
#                    counts before it bails out below
#                    SESSION_SUMMARY_MIN_MEMORIES (session-lifecycle.ts:
#                    170-183, `where {sessionId, isActive:true}`), so it
#                    decides whether a summary will ever be written for this
#                    session.
# Short-circuit: if the session has zero SESSION-scoped active memories on
# the first poll, findStaleSessions can never select it, because it requires
# scope='SESSION'. There is then nothing to wait for, so the function
# returns immediately. Ingest uses write_mode=sync, so all rows are
# committed by the time this runs, and the first read is authoritative.
#
# Promotion alone is not a sufficient release signal for a summary-eligible
# session. runSessionLifecycle fires promoteSessionMemories and
# generateSessionSummary at the same time (Promise.allSettled,
# session-lifecycle.ts:252-256). Promotion only touches the database, while
# the summary needs an LLM round trip, so promoted rows become visible well
# before the summary row. Releasing on has_promotion alone would let the
# driver answer session i+1 while session i's summary is still in flight.
# See the release table in _wait_for_session_lifecycle_async.
_LIFECYCLE_SQL = """
SELECT
  (SELECT count(*) FROM memories
     WHERE "sessionId" = $1 AND scope = 'SESSION' AND "isActive" = true) AS session_scoped,
  (SELECT count(*) FROM memories
     WHERE "sessionId" = $1 AND "isActive" = true) AS total_active,
  EXISTS(SELECT 1 FROM memories
     WHERE "sessionId" = $1 AND "isActive" = true
       AND (metadata->>'session_summary') = 'true'
       AND (metadata->>'source_session_id') = $1) AS has_summary,
  EXISTS(SELECT 1 FROM memories
     WHERE "sessionId" = $1 AND "isActive" = true
       AND (metadata->>'promoted_from_session') = 'true') AS has_promotion
"""

# The server's own summary-eligibility threshold (session-lifecycle.ts:23,
# `parseInt(process.env.SESSION_SUMMARY_MIN_MEMORIES ?? "2", 10)`). Read from
# the same env var, so the barrier and the server agree. The compose service
# passes one value to both.
_SUMMARY_MIN_MEMORIES = int(os.environ.get("SESSION_SUMMARY_MIN_MEMORIES") or "2")

# Fallback-only grace period, held before releasing a session whose summary
# can never arrive (see the "promotion consumed the SESSION rows" case in the
# release table below). This applies only when the server log is
# unavailable. The server promotes and summarizes at the same time
# (`Promise.allSettled`, session-lifecycle.ts:253-256), so at the instant
# session_scoped hits 0, a summary may still be in flight. The observed
# promotion-to-settle time is about 1 second on a near-idle server. That
# measurement does not hold under a full concurrent wave, where a valid
# summary LLM call can outlive any fixed grace period. This is why the
# primary release signal is now the scheduler's own per-pass completion
# marker in the server log (below), not this timer.
_SUMMARY_SKIP_GRACE_S = float(
    os.environ.get("RETAINDB_LIFECYCLE_SUMMARY_GRACE_S") or "15"
)

# --- Definitive release signal: the scheduler's per-pass completion marker -----
# processSession logs exactly one line per session per lifecycle pass,
# emitted only after both promoteSessionMemories and generateSessionSummary
# have settled (session-lifecycle.ts:252-268):
#   [session-lifecycle] <sessionId>: promoted=N skipped=M summary=<uuid|skipped>
# `summary=skipped` on that line is therefore definitive for the pass: the
# summary attempt already returned null, so no summary is coming from it, and
# no grace period is needed. `summary=<uuid>` means a summary row was
# stored, since the store happens before the log line, and the database poll
# confirms it. The entrypoint redirects the node server's stdout and stderr
# to a file at $RETAINDB_SERVER_LOG (entrypoint.retaindb-server.sh), so the
# adapter can read the marker. When the file is unavailable (an older image,
# a host smoke, or a non-Docker path), the barrier falls back to the
# fixed-grace heuristic above and flags this in the returned dict
# (`release_signal`).
_LIFECYCLE_LOG_BACKSCAN_BYTES = int(
    os.environ.get("RETAINDB_SERVER_LOG_BACKSCAN_BYTES") or str(256 * 1024)
)


class _LifecycleLogScanner:
    """Incrementally scan $RETAINDB_SERVER_LOG for one session's pass markers.

    Each lifecycle wait uses one instance. The first poll seeks to end of
    file minus a small backscan. The marker cannot predate this session's
    ingest, which finished milliseconds before the wait starts, so a bounded
    backscan always covers the lost-race window. Later polls read only the
    newly appended bytes. Reads are binary and stop at the last newline, so a
    marker line mid-write is never half-parsed. ``available`` flips to False
    permanently on any OSError or an unset path. The caller then uses the
    grace fallback.
    """

    def __init__(self, server_session_id: str) -> None:
        self.path = os.environ.get("RETAINDB_SERVER_LOG") or ""
        self.available = bool(self.path)
        self._offset: Optional[int] = None
        # `: promoted=` right after the id stops sid "...__1" from matching a
        # line for "...__10". The summary value is the last token on the line.
        self._re = re.compile(
            (r"\[session-lifecycle\] " + re.escape(server_session_id)
             + r": promoted=(\d+) skipped=(\d+) summary=(\S+)").encode("utf-8")
        )
        self.marker_summary: Optional[str] = None  # "skipped" or a memory uuid
        # promoted= from the most recent completed pass. This is the
        # promotion terminal signal, already carried on the marker line. A
        # completed pass that promoted 0 rows while SESSION rows remain
        # means those rows cannot be promoted (blocked by PROMOTABLE_TYPES,
        # or below the confidence floor), so no future pass will move them
        # either. promoted>0 means the pass was still making progress, so a
        # later tick may promote more.
        self.marker_promoted: Optional[int] = None
        self.marker_count = 0  # completed passes seen for this session

    def poll(self) -> None:
        if not self.available:
            return
        try:
            with open(self.path, "rb") as f:
                if self._offset is None:
                    f.seek(0, os.SEEK_END)
                    self._offset = max(0, f.tell() - _LIFECYCLE_LOG_BACKSCAN_BYTES)
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            self.available = False
            return
        # Consume only complete lines. A trailing partial line is re-read on
        # the next poll.
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self._offset += cut + 1
        for m in self._re.finditer(chunk[: cut + 1]):
            self.marker_summary = m.group(3).decode("utf-8", "replace")
            try:
                self.marker_promoted = int(m.group(1))
            except (TypeError, ValueError):  # pragma: no cover - regex guarantees digits
                self.marker_promoted = None
            self.marker_count += 1


async def _wait_for_session_lifecycle_async(
    dsn: str, server_session_id: str, timeout_s: float, poll_interval_s: float
) -> Dict[str, Any]:
    """Poll until this session's lifecycle observable actually lands.

    Release table (``summary_eligible`` means ``total_active >=
    SESSION_SUMMARY_MIN_MEMORIES``, the server's own generateSessionSummary
    bail-out point; ``marker`` means this session's most recent
    completed-pass line in $RETAINDB_SERVER_LOG, see
    ``_LifecycleLogScanner``):

      session_scoped==0 (first poll)          -> skipped_no_session_scope   [db]
      has_summary                             -> done_summary               [db /
                                                 log_marker if marker=<uuid> seen]
      has_promotion, NOT summary_eligible     -> done_promotion             [db]
      marker says summary=skipped AND
        (has_promotion OR session_scoped
         stable across one extra poll)        -> done_promotion_summary_skipped
                                                 IMMEDIATELY               [log_marker]
      marker says summary=<uuid>              -> keep polling for the DB row
                                                 (the DB is authoritative)
      log readable, NO marker yet             -> KEEP WAITING (the pass has not
                                                 finished; a summary may still be
                                                 in flight; this is the case the
                                                 old fixed grace wrongly released)
      log UNAVAILABLE, has_promotion,
        session_scoped unchanged for grace    -> done_promotion_summary_skipped
                                                 [db_grace fallback, old behavior]
      bound elapsed                           -> timeout                    [timeout]

    Why the marker is definitive: ``processSession`` logs its one
    ``[session-lifecycle] <sid>: promoted=N skipped=M summary=<uuid|skipped>``
    line only after both promotion and summary have settled
    (Promise.allSettled, session-lifecycle.ts:252-268). ``summary=skipped``
    therefore means ``generateSessionSummary`` already returned null for that
    pass. It does this without logging, on short LLM output
    (``if (!summary || summary.length < 20) return null``,
    session-lifecycle.ts:217-218). Promotion, running at the same time,
    normally flips every SESSION row to USER, so ``findStaleSessions``
    (which selects on ``scope='SESSION'``, :79) can never re-select the
    session, so no later tick retries the summary. The previous release rule
    here was a 15-second no-progress grace, measured on a near-idle
    1-persona server. Under a full concurrent wave, a valid summary LLM call
    can take longer than that after promotion completes, so the grace could
    release early, answer without the summary, and undercount the summary
    rate. The marker removes the guesswork: no marker yet means the pass is
    still running, so keep waiting.

    The straggler case still releases through the marker. A session whose
    promoter refuses a row forever (``promoted=0 skipped=1`` re-emitted
    every tick, observed on session 15 of smk_rdbf_clk2) has no promotion
    row. Its ``summary=skipped`` marker is confirmed by one extra poll of
    unchanged session_scoped, which guards the small window where a later
    pass is mid-flight, and it releases in about one poll interval, not 15
    seconds.

    Measured 2026-07-27 (`smk_rdbf_clk`): every session logged
    ``promoted=N skipped=0 summary=skipped`` about 1 second after the tick.
    The summary is not always skipped, though. `smk_rdbf_clk2` produced
    ``done_summary`` on 3 of its first 16 sessions, about 19 percent. Treat a
    skipped summary as a rate, not a certainty. The skip release exists for
    the sessions that lose that race.
    """
    start = time.time()
    conn = await asyncpg.connect(dsn=dsn)
    first = True
    log_scan = _LifecycleLogScanner(server_session_id)
    log_fallback_announced = False
    # Fallback-only progress watchdog. The grace period measures time without
    # progress, meaning session_scoped is unchanged, not the overall wait
    # time, so a draining session never trips it. Used only when the server
    # log is unavailable.
    last_session_scoped: Optional[int] = None
    no_progress_since: Optional[float] = None
    try:
        while True:
            row = await conn.fetchrow(_LIFECYCLE_SQL, server_session_id)
            log_scan.poll()
            session_scoped = int(row["session_scoped"])
            total_active = int(row["total_active"])
            has_summary = bool(row["has_summary"])
            has_promotion = bool(row["has_promotion"])
            summary_eligible = total_active >= _SUMMARY_MIN_MEMORIES
            waited = time.time() - start
            observed = {
                "waited_s": waited,
                "session_scoped": session_scoped,
                "total_active": total_active,
                "summary_eligible": summary_eligible,
                "log_marker_available": log_scan.available,
                "log_marker_summary": log_scan.marker_summary,
                "log_marker_promoted": log_scan.marker_promoted,
                "log_marker_passes": log_scan.marker_count,
            }
            if has_summary:
                # The DB row is authoritative for done_summary. Credit the
                # marker as the signal when the log already announced the
                # same uuid.
                signal = (
                    "log_marker"
                    if log_scan.marker_summary not in (None, "skipped")
                    else "db"
                )
                return {"status": "done_summary", "release_signal": signal, **observed}
            # Promotion alone completes only a session that will never get a
            # summary. Otherwise the summary LLM call may still be in
            # flight, and releasing here would race it.
            if has_promotion and not summary_eligible:
                return {"status": "done_promotion", "release_signal": "db", **observed}
            if log_scan.available and log_scan.marker_summary == "skipped":
                # A completed pass declared no summary, so the summary
                # question is settled for that pass. Promotion is a separate
                # question, and the marker says nothing about it. Rows still
                # at SESSION scope stay invisible to recall, which queries by
                # user_id against USER scope, and they keep the session
                # selectable by findStaleSessions, so a later pass can still
                # promote them. Releasing while session_scoped is still
                # moving would answer this session's questions without
                # memories that were about to become visible. That is what
                # the no-progress grace this replaced was accidentally
                # protecting.
                # Promotion convergence has its own terminal signal, carried
                # on the same marker line: `promoted=`. A completed pass
                # that promoted 0 rows while SESSION rows remain means those
                # rows cannot be promoted (blocked by PROMOTABLE_TYPES or the
                # confidence floor), so no later pass moves them either.
                # That is the straggler case, whose signature is a repeated
                # `promoted=0 skipped=1`. `promoted>0` means the pass was
                # making progress, so the next tick may promote more, and we
                # keep waiting.
                # Do not substitute "session_scoped unchanged across a poll"
                # for this: the scheduler ticks about every 60 seconds and
                # the barrier polls every few seconds, so an unchanged count
                # usually just falls between ticks. Releasing on it drops
                # rows that were about to be promoted (measured: released at
                # 5.0 seconds with 8 rows still at SESSION scope).
                if session_scoped == 0 or log_scan.marker_promoted == 0:
                    return {
                        "status": "done_promotion_summary_skipped",
                        "release_signal": "log_marker",
                        "residual_session_scoped": session_scoped,
                        **observed,
                    }
            elif log_scan.marker_summary not in (None, "skipped"):
                # The marker names a summary uuid. The row is committed
                # before the line is logged, so has_summary should fire on
                # the next poll or polls. Keep waiting; the outer timeout
                # still bounds a metadata mismatch.
                pass
            if not log_scan.available:
                # Fallback for a missing or unreadable log (older image,
                # host smoke, permissions): use the pre-marker fixed-grace
                # heuristic, so nothing regresses, flagged with
                # release_signal=db_grace.
                if not log_fallback_announced:
                    log_fallback_announced = True
                    print(
                        "[DEBUG] lifecycle_wait: RETAINDB_SERVER_LOG unavailable "
                        f"({os.environ.get('RETAINDB_SERVER_LOG') or 'unset'}); "
                        "falling back to the fixed no-progress grace "
                        f"({_SUMMARY_SKIP_GRACE_S:.0f}s)"
                    )
                now = time.time()
                if session_scoped != last_session_scoped:
                    last_session_scoped = session_scoped
                    no_progress_since = now
                elif has_promotion and no_progress_since is not None \
                        and now - no_progress_since >= _SUMMARY_SKIP_GRACE_S:
                    return {
                        "status": "done_promotion_summary_skipped",
                        "release_signal": "db_grace",
                        "no_progress_s": now - no_progress_since,
                        "residual_session_scoped": session_scoped,
                        **observed,
                    }
            if first and session_scoped == 0:
                return {
                    "status": "skipped_no_session_scope",
                    "release_signal": "db",
                    **observed,
                }
            first = False
            if waited >= timeout_s:
                return {
                    "status": "timeout",
                    "release_signal": "timeout",
                    **observed,
                    "has_promotion": has_promotion,
                }
            await asyncio.sleep(poll_interval_s)
    finally:
        await conn.close()


def Wait_For_Session_Lifecycle(
    server_session_id: str, timeout_s: float, poll_interval_s: float
) -> Dict[str, Any]:
    """Block until the server's scheduler has processed ``server_session_id``.

    Returns a status dict, recorded per session in the runtime summary.
    ``status`` is one of: ``done_summary``, ``done_promotion``, or
    ``done_promotion_summary_skipped`` (lifecycle ran and left an
    observable); ``skipped_no_session_scope`` (nothing for lifecycle to do);
    ``timeout`` (the bound elapsed, recorded, and the run proceeds); or
    ``disabled`` (no DATABASE_URL or asyncpg, the host smoke path).
    ``release_signal`` records which observable released the barrier:
    ``log_marker`` (the scheduler's own completed-pass line in
    $RETAINDB_SERVER_LOG, the definitive signal), ``db`` (a summary or
    promotion row), ``db_grace`` (log unavailable, fixed-grace fallback), or
    ``timeout``. This lets runs be audited for how often the fallback fired.
    ``done_promotion`` is returned only for a session that is not
    summary-eligible. A summary-eligible session must show its summary row,
    a ``summary=skipped`` pass marker, or time out.
    """
    dsn = os.environ.get("DATABASE_URL")
    if not dsn or asyncpg is None:
        return {
            "status": "disabled",
            "release_signal": "disabled",
            "waited_s": 0.0,
            "reason": "DATABASE_URL unset" if not dsn else "asyncpg unavailable",
        }
    try:
        return asyncio.run(
            _wait_for_session_lifecycle_async(dsn, server_session_id, timeout_s, poll_interval_s)
        )
    except Exception as e:  # pragma: no cover - never let the barrier kill a run
        return {"status": "error", "release_signal": "error", "waited_s": 0.0, "reason": str(e)}


# --------------------------------------------------------------------------
# Recall
# --------------------------------------------------------------------------
def _result_created_at(result: Dict[str, Any]) -> str:
    """Pick the best temporal anchor for the scorer's created_at slot.

    The server returns a real ``temporal`` block. Prefer document_date, then
    event_date, then valid_from, then a wall-clock fallback.
    """
    memory = result.get("memory") or {}
    temporal = memory.get("temporal") if isinstance(memory, dict) else None
    temporal = temporal or {}
    for candidate in (
        temporal.get("document_date"),
        temporal.get("event_date"),
        temporal.get("valid_from"),
        memory.get("created_at") if isinstance(memory, dict) else None,
    ):
        if candidate:
            return str(candidate)
    return "Unknown Time"


def Search_RetainDB_Server_For_Question(
    client: RetainDBServerClient,
    project: str,
    user_id: str,
    question_text: str,
    top_k: int,
    question_date: Optional[str],
    profile: Optional[str],
    plugin_overlay: bool = True,
    stats: Optional[Dict[str, Any]] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Recall memories for one question and map them to the scorer's memory shape.

    Maps the server's response fields to the harness contract:
      * ``memory``      = compacted or raw ``results[].memory.content``
      * ``score``       = ``results[].similarity`` (the server's relevance field)
      * ``created_at``  = ``temporal.document_date``, else event_date, else
                          valid_from, else a wall-clock fallback (see
                          ``_result_created_at``)
    Diagnostic fields (id, type, similarity, confidence, fallback mode) pass
    through, the same way other adapters keep their sub-scores.

    ``plugin_overlay=True`` (default, used for a headline run) reproduces the
    query half of the Hermes plugin's ``_build_overlay``: fetch at the
    server's default top-K, slice to the harness top_k, apply ``_compact``
    per item (whitespace collapse plus a 320-character cut), then dedup on
    ``_norm`` while keeping order. The compacted text becomes ``memory`` on
    the stored row. The shared driver also renders this text into the answer
    prompt, so it is one text with two consumers. Do not split them.

    ``plugin_overlay=False`` stores and answers from raw ``content`` (no
    dedup, a plain top_k request), so a diagnostic run can isolate how much
    score compaction costs. This is not plugin-faithful. Never use it for a
    headline number.
    """
    start = time.time()
    fetch_k = max(top_k, _PLUGIN_QUERY_SERVER_TOPK) if plugin_overlay else top_k
    response = client.search(
        project=project,
        query=question_text,
        top_k=fetch_k,
        user_id=user_id,
        # Plugin-faithful: the Hermes plugin sends no date, so neither do we.
        # The temporal anchor comes from the server's clock (faked to the
        # session date by BENCH_CLOCKSYNC), not from a parameter. See
        # _SEND_QUESTION_DATE.
        question_date=question_date if _SEND_QUESTION_DATE else None,
        profile=profile,
    )
    duration_ms = (time.time() - start) * 1000.0

    results = response.get("results") or []
    fallback = response.get("fallback")
    # Diagnostic capture, taken from the response before the overlay's top_k
    # slice, per-item compaction, and dedup below. The point is to record
    # what the server ranked, not what survived. Raw is the single /search
    # response this question was answered from. Ranked text is the
    # uncompacted ``memory.content`` in rank order. ``fetch_k`` is the
    # plugin's own server top-K (10), which the overlay already requests and
    # this capture does not raise, so a depth curve on a RetainDB file stops
    # at 10.
    record_provider_retrieval(diag, raw=response, ranked=[
        {
            "memory": str((r.get("memory") or {}).get("content", "")
                          if isinstance(r.get("memory"), dict) else ""),
            "created_at": _result_created_at(r),
            "score": r.get("similarity"),
        }
        for r in results if isinstance(r, dict)
    ])
    if plugin_overlay:
        # The plugin uses ``results[:5]``. This matches the harness top_k,
        # identical at the default value of 5.
        results = results[:top_k]

    retrieved: List[Dict[str, Any]] = []
    seen_norm: List[str] = []
    for result in results:
        memory = result.get("memory") or {}
        raw_text = str(memory.get("content", "") if isinstance(memory, dict) else "")
        item: Dict[str, Any] = {
            "memory": raw_text,
            "created_at": _result_created_at(result),
            "score": result.get("similarity"),
            "id": memory.get("id") if isinstance(memory, dict) else None,
            "type": memory.get("type") if isinstance(memory, dict) else None,
            "confidence": memory.get("confidence") if isinstance(memory, dict) else None,
            "similarity": result.get("similarity"),
            "retrieval_fallback": fallback,
        }
        if plugin_overlay:
            compacted = _plugin_compact(raw_text)
            norm = _plugin_norm(compacted)
            if not compacted:
                # The plugin's `if c and n not in seen` rule: an empty
                # compaction never claims a slot.
                if stats is not None:
                    stats["overlay_dropped_empty"] = stats.get("overlay_dropped_empty", 0) + 1
                continue
            if norm in seen_norm:
                if stats is not None:
                    stats["overlay_dropped_duplicate"] = stats.get("overlay_dropped_duplicate", 0) + 1
                continue
            seen_norm.append(norm)
            item["memory"] = compacted
            # Diagnostics only: record the length of the stored row, never
            # the untruncated text. Keeping the untruncated text would tempt
            # a future split that feeds it to the judge. The count is all a
            # truncation-impact analysis needs.
            item["plugin_overlay_compacted"] = len(raw_text) > len(compacted)
            item["source_chars"] = len(raw_text)
            if stats is not None:
                stats["overlay_items"] = stats.get("overlay_items", 0) + 1
                if len(raw_text) > len(compacted):
                    stats["overlay_truncated"] = stats.get("overlay_truncated", 0) + 1
                    stats["overlay_chars_dropped"] = (
                        stats.get("overlay_chars_dropped", 0) + len(raw_text) - len(compacted)
                    )
        retrieved.append(item)
    return retrieved, duration_ms


# --------------------------------------------------------------------------
# Provider binding (the only RetainDB-server-specific surface the driver sees)
# --------------------------------------------------------------------------
class RetainDBServerBinding(ProviderBinding):
    memory_system = "retaindb_server"
    store_id_key = "RetainDB_Server_Project"
    runtime_summary_key = "RetainDB_Server_Runtime_Summary"
    stage_name = "retaindb_server_answer_generation"
    stage_note = "RetainDB server retrieval and question answering"

    def __init__(self, client: Optional[RetainDBServerClient], profile: Optional[str],
                 plugin_overlay: bool = True, granularity: str = "session",
                 wait_lifecycle: bool = False, lifecycle_timeout_s: float = 150.0,
                 lifecycle_poll_s: float = 5.0, promotion_mode: Optional[str] = None):
        self.client = client
        # Server search profile: fast, balanced, or quality. None means
        # omit, which falls back to the server default (fast). Recorded per
        # persona in the runtime summary.
        self.profile = profile
        # Reproduce the Hermes plugin's recall-time overlay compaction (the
        # default).
        self.plugin_overlay = plugin_overlay
        # Ingest cadence: "session" (minimal, one call per session) or
        # "exchange" (featured, the plugin's per-turn sync_turn cadence).
        self.granularity = granularity
        # Featured consolidation: after each session's ingest, wait for the
        # server's scheduler to run runSessionLifecycle() over it (promote
        # and summarize) before moving to the next session. This requires
        # the entrypoint to start the server with DISABLE_SCHEDULER=false
        # and a lowered SESSION_INACTIVITY_THRESHOLD_MS. The barrier polls
        # Postgres directly.
        self.wait_lifecycle = wait_lifecycle
        self.lifecycle_timeout_s = lifecycle_timeout_s
        self.lifecycle_poll_s = lifecycle_poll_s
        # Server scope-inference pipeline, a vendor-exposed per-request
        # field. None means omit, which falls back to the server default
        # (session_state_v1). "user_specific_legacy" routes mid-confidence
        # user facts to SESSION scope, so the session-lifecycle scheduler
        # has rows to promote and summarize. Env RETAINDB_SERVER_PROMOTION_MODE.
        self.promotion_mode = promotion_mode

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        # A unique project is RetainDB's tenancy boundary, giving full
        # per-persona isolation. A stable user_id keeps USER-scoped memories
        # attaching to, and retrieving from, the same persona.
        return {
            "store_id": f"mc_{persona_id[-8:]}_{uuid.uuid4().hex[:8]}",
            "user_id": f"user_{persona_id[-8:]}",
            "persona_tag": persona_id[-8:],
            "total_created": 0,
            "total_relations": 0,
            "total_errors": 0,
            "total_ingest_calls": 0,
            "total_dropped_exchanges": 0,
            "lifecycle_waits": [],
            "total_lifecycle_wait_s": 0.0,
            _QUESTION_DATE_KEY: None,
        }

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        timestamp = Parse_Session_Timestamp(session_item)
        session_label = str(session_item.get("Session_ID", session_index))
        # Make the server session id unique per persona, so session-lifecycle
        # grouping, and the lifecycle-wait poll, match only this persona's
        # session. Session_ID labels ("1", "2", ...) repeat across personas,
        # and a bare label would collide within one shared per-run database
        # (an earlier persona's already-summarized session would satisfy the
        # wait for a later persona instantly). store_id carries a
        # per-persona uuid, so prefixing with it makes the id globally
        # unique.
        server_session_id = f"{ctx['store_id']}__{session_label}"
        # Stash this session's temporal-search question_date for the
        # questions answered against it. The shared driver ingests session i,
        # then immediately answers session i's questions, so the "now" a
        # question is posed at is this session's date. This is the noon
        # recall-"now" anchor (eval_common.Parse_Query_Now_Timestamp), the
        # same instant the faked OS clock uses, not the midnight
        # ``timestamp`` above, which stays the per-message ingest timestamp.
        # Only the recall question_date is noon. Ingest is untouched. An
        # unparseable date becomes None, so recall omits question_date and
        # the server falls back to wall-clock "now".
        ctx[_QUESTION_DATE_KEY] = _iso(Parse_Query_Now_Timestamp(session_item))

        add_ms, created, relations, errors, ingest_calls, dropped = (
            Add_Session_Dialogue_To_RetainDB_Server(
                self.client, ctx["store_id"], ctx["user_id"], session_label,
                server_session_id, dialogue, timestamp, granularity=self.granularity,
                promotion_mode=self.promotion_mode,
            )
        )
        ctx["total_created"] += created
        ctx["total_relations"] += relations
        ctx["total_errors"] += errors
        ctx["total_ingest_calls"] += ingest_calls
        ctx["total_dropped_exchanges"] += dropped
        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
              f"created={created} relations={relations} errors={errors} "
              f"ingest_calls={ingest_calls} dropped={dropped} ingest_ms={add_ms:.0f}")

        meta = {
            "Dialogue_Added_To_Memory": created > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Retain_Granularity": self.granularity,
            "Ingest_Calls": ingest_calls,
            "Dropped_Empty_Exchanges": dropped,
            "Memories_Created": created,
            "Relations_Created": relations,
            "Ingest_Errors": errors,
            "Session_Timestamp_Passed": _iso(timestamp),
            "Add_Duration_ms": add_ms,
        }

        # Featured consolidation barrier: wait for the scheduler to promote
        # and summarize this session before the next one is ingested and
        # answered.
        if self.wait_lifecycle:
            wait = Wait_For_Session_Lifecycle(
                server_session_id, self.lifecycle_timeout_s, self.lifecycle_poll_s
            )
            ctx["lifecycle_waits"].append(wait)
            ctx["total_lifecycle_wait_s"] += float(wait.get("waited_s", 0.0) or 0.0)
            meta["Lifecycle_Wait_Status"] = wait.get("status")
            meta["Lifecycle_Release_Signal"] = wait.get("release_signal")
            meta["Lifecycle_Wait_ms"] = float(wait.get("waited_s", 0.0) or 0.0) * 1000.0
            meta["Lifecycle_Total_Active"] = wait.get("total_active")
            meta["Lifecycle_Summary_Eligible"] = wait.get("summary_eligible")
            print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
                  f"lifecycle_wait status={wait.get('status')} "
                  f"release_signal={wait.get('release_signal')} "
                  f"session_scoped={wait.get('session_scoped')} "
                  f"total_active={wait.get('total_active')} "
                  f"summary_eligible={wait.get('summary_eligible')} "
                  f"waited_s={wait.get('waited_s', 0.0):.1f}")

        return meta

    def recall(self, ctx, question_text, top_k):
        return Search_RetainDB_Server_For_Question(
            self.client, ctx["store_id"], ctx["user_id"], question_text, top_k,
            question_date=ctx.get(_QUESTION_DATE_KEY),
            profile=self.profile,
            plugin_overlay=self.plugin_overlay,
            stats=ctx.setdefault("overlay_stats", {}),
            diag=ctx,
        )

    def persona_count_extras(self, ctx):
        extras = {
            "Total_Memories_Created": ctx["total_created"],
            "Total_Relations_Created": ctx["total_relations"],
            "Total_Ingest_Errors": ctx["total_errors"],
            "Total_Ingest_Calls": ctx["total_ingest_calls"],
            "Retain_Granularity": self.granularity,
            "Promotion_Mode": self.promotion_mode or "session_state_v1 (server default)",
            "Dropped_Empty_Exchanges": ctx["total_dropped_exchanges"],
            "Search_Profile": self.profile or "fast (server default)",
            # Record which retrieval product measured this persona, directly
            # in the results file, so a Results/*.json can never be mistaken
            # for the other arm. The manifest also captures RETAINDB_SERVER_*.
            "Plugin_Overlay_Compaction": self.plugin_overlay,
            "Plugin_Overlay_Profile_Half": (
                "omitted (unscored + query-independent; see docs/DECISIONS.md)"
                if self.plugin_overlay else "n/a"
            ),
        }
        if self.plugin_overlay:
            stats = ctx.get("overlay_stats") or {}
            extras.update({
                "Overlay_Items_Emitted": stats.get("overlay_items", 0),
                "Overlay_Items_Truncated": stats.get("overlay_truncated", 0),
                "Overlay_Chars_Dropped": stats.get("overlay_chars_dropped", 0),
                "Overlay_Items_Dropped_Duplicate": stats.get("overlay_dropped_duplicate", 0),
                "Overlay_Items_Dropped_Empty": stats.get("overlay_dropped_empty", 0),
            })
        return extras

    def persona_tail_extras(self, ctx):
        # Session-lifecycle barrier stats, featured arm only. A per-session
        # status histogram, plus the total and mean wait, make the added
        # wall time auditable and flag sessions the barrier could not
        # confirm (timeout or no-session-scope).
        if not self.wait_lifecycle:
            return {}
        waits = ctx.get("lifecycle_waits") or []
        status_counts: Dict[str, int] = {}
        signal_counts: Dict[str, int] = {}
        for w in waits:
            s = str(w.get("status"))
            status_counts[s] = status_counts.get(s, 0) + 1
            sig = str(w.get("release_signal"))
            signal_counts[sig] = signal_counts.get(sig, 0) + 1
        n = len(waits)
        total_s = ctx.get("total_lifecycle_wait_s", 0.0)
        # Summary-eligible sessions are the ones the barrier holds for a
        # summary row. A timeout among them is the result-gate failure to
        # look for.
        eligible = [w for w in waits if w.get("summary_eligible")]
        return {
            "Lifecycle_Wait_Enabled": True,
            "Lifecycle_Wait_Timeout_s": self.lifecycle_timeout_s,
            "Lifecycle_Wait_Poll_s": self.lifecycle_poll_s,
            "Lifecycle_Summary_Min_Memories": _SUMMARY_MIN_MEMORIES,
            "Lifecycle_Sessions_Waited": n,
            "Lifecycle_Sessions_Summary_Eligible": len(eligible),
            "Lifecycle_Eligible_Without_Summary": sum(
                1 for w in eligible if w.get("status") != "done_summary"
            ),
            "Lifecycle_Status_Counts": status_counts,
            # db_grace here means the server-log marker was unavailable and
            # the fixed-grace fallback released the session. Audit those runs.
            "Lifecycle_Release_Signal_Counts": signal_counts,
            "Lifecycle_Total_Wait_s": total_s,
            "Lifecycle_Mean_Wait_s": (total_s / n) if n else 0.0,
        }


def Generate_User_RetainDB_Server_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    profile: Optional[str],
    base_url: str,
    api_key: Optional[str],
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
    plugin_overlay: bool = True,
    granularity: str = "session",
    wait_lifecycle: bool = False,
    lifecycle_timeout_s: float = 150.0,
    lifecycle_poll_s: float = 5.0,
    promotion_mode: Optional[str] = None,
) -> bool:
    print(f"[DEBUG] base_url={base_url}  profile={profile or 'fast (server default)'}  "
          f"plugin_overlay={plugin_overlay}  granularity={granularity}  "
          f"promotion_mode={promotion_mode or 'session_state_v1 (server default)'}  "
          f"wait_lifecycle={wait_lifecycle}"
          + (f" (timeout={lifecycle_timeout_s}s poll={lifecycle_poll_s}s)" if wait_lifecycle else "")
          + ("  (compact 320c + dedup, profile half omitted)" if plugin_overlay
             else "  (RAW rows -- diagnostic arm, NOT plugin-faithful)"))

    binding = RetainDBServerBinding(
        client=None, profile=profile, plugin_overlay=plugin_overlay,
        granularity=granularity, wait_lifecycle=wait_lifecycle,
        lifecycle_timeout_s=lifecycle_timeout_s, lifecycle_poll_s=lifecycle_poll_s,
        promotion_mode=promotion_mode,
    )

    def setup():
        # Attach-only: a shell script starts the server. Wait for it to
        # become healthy, then bind.
        client = RetainDBServerClient(base_url, api_key=api_key)
        client.wait_healthy()
        print(f"[retaindb-server] attached to server at {base_url}", flush=True)
        binding.client = client

    def teardown():
        # A shell script owns the server's lifecycle, so there is nothing
        # to tear down here.
        pass

    return eval_common.run_eval(
        binding=binding,
        input_jsonl_path=input_jsonl_path,
        output_jsonl_path=output_jsonl_path,
        output_json_path=output_json_path,
        top_k=top_k,
        start_idx=start_idx,
        end_idx=end_idx,
        max_sessions=max_sessions,
        max_questions_per_session=max_questions_per_session,
        overwrite_existing_answers=overwrite_existing_answers,
        setup=setup,
        teardown=teardown,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RetainDB server evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "retaindb_server_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "retaindb_server_results.json"),
        top_k_help="How many recalled memories the answer LLM sees (up to 5 are always stored "
                   "for white-box scoring). The upstream MemConflict adapters answer "
                   "from top-3, so use --top_k 3 for strict answer-accuracy comparability.",
    )
    parser.add_argument("--base_url", type=str,
                        default=os.environ.get("RETAINDB_SERVER_BASE_URL", "http://127.0.0.1:3000"),
                        help="Base URL of an already-running RetainDB server (attach-only; the server "
                             "and its migrations start shell-side, either the Docker entrypoint or "
                             "retaindb_server/serve_local.sh on a host).")
    parser.add_argument("--api_key", type=str, default=os.environ.get("RETAINDB_SERVER_API_KEY"),
                        help="Bearer token if the server started with RETAINDB_API_KEY set "
                             "(default unset means open access).")
    parser.add_argument("--profile", type=str,
                        default=os.environ.get("RETAINDB_SERVER_PROFILE"),
                        choices=["fast", "balanced", "quality"],
                        help="Server search profile. Unset means omitted, so the server default (fast) applies.")
    # Env-driven default, like the local adapter: entrypoint.retaindb-server.sh
    # selects the arm through RETAINDB_SERVER_PLUGIN_OVERLAY, which reaches this
    # default. write_manifest.py records every RETAINDB_SERVER_* variable.
    parser.add_argument("--plugin_overlay", dest="plugin_overlay", action="store_true",
                        default=_env_flag("RETAINDB_SERVER_PLUGIN_OVERLAY", True),
                        help="Reproduce the Hermes plugin's recall overlay: per-item whitespace "
                             "collapse, 320-character compaction, and normalized dedup, applied to both "
                             "the answer context and the stored Retrieved_Memories (default ON; "
                             "set env RETAINDB_SERVER_PLUGIN_OVERLAY=0 to disable).")
    parser.add_argument("--no_plugin_overlay", dest="plugin_overlay", action="store_false",
                        help="Diagnostic arm: store and answer from raw search rows. Not plugin-faithful. "
                             "Never use for a headline number.")
    parser.add_argument("--retain_granularity", type=str,
                        default=(os.environ.get("RETAINDB_RETAIN_GRANULARITY") or "session").strip().lower(),
                        choices=["session", "exchange"],
                        help="Ingest cadence. 'session' (minimal, default): one ingest/session call "
                             "for the whole session. 'exchange' (featured): one ingest/session call "
                             "per user/assistant turn (2 verbatim messages), the Hermes plugin's "
                             "sync_turn cadence. Env RETAINDB_RETAIN_GRANULARITY.")
    parser.add_argument("--wait_lifecycle", dest="wait_lifecycle", action="store_true",
                        default=_env_flag("RETAINDB_SERVER_WAIT_LIFECYCLE", False),
                        help="Featured consolidation: after each session's ingest, block until the "
                             "server's scheduler has run session-lifecycle over it (promote SESSION to USER "
                             "and write a session summary), by polling Postgres. Requires the server started "
                             "with DISABLE_SCHEDULER=false and a lowered SESSION_INACTIVITY_THRESHOLD_MS. "
                             "Env RETAINDB_SERVER_WAIT_LIFECYCLE=1.")
    parser.add_argument("--lifecycle_timeout_s", type=float,
                        default=float(os.environ.get("RETAINDB_LIFECYCLE_WAIT_TIMEOUT_S", "150")),
                        help="Max seconds to wait per session for lifecycle (default 150, about twice the "
                             "hardcoded 60-second scheduler tick, plus margin). On timeout the run proceeds and "
                             "records the status. Env RETAINDB_LIFECYCLE_WAIT_TIMEOUT_S.")
    parser.add_argument("--lifecycle_poll_s", type=float,
                        default=float(os.environ.get("RETAINDB_LIFECYCLE_POLL_INTERVAL_S", "5")),
                        help="Lifecycle poll interval in seconds (default 5). Env "
                             "RETAINDB_LIFECYCLE_POLL_INTERVAL_S.")
    parser.add_argument("--promotion_mode", type=str,
                        default=(os.environ.get("RETAINDB_SERVER_PROMOTION_MODE") or None),
                        choices=["session_state_v1", "user_specific_legacy"],
                        help="Vendor-exposed scope-inference pipeline, sent per ingest request. "
                             "Omitted (default) means the server default (session_state_v1) applies, which routes "
                             "ordinary user facts to USER or PROJECT scope and never creates SESSION-scoped "
                             "rows for the scheduler. 'user_specific_legacy' routes mid-confidence facts "
                             "to SESSION scope, so wait_lifecycle actually exercises promotion and summary. "
                             "Env RETAINDB_SERVER_PROMOTION_MODE.")
    args = parser.parse_args()

    # Fail closed on the one combination that silently produces garbage. This
    # adapter no longer sends `question_date` (the Hermes plugin does not), so
    # the recall temporal anchor is the server's clock. Under BENCH_CLOCKSYNC
    # that clock holds the session date and all is well. Without it, the
    # server falls back to real wall-clock time in 2026 against a 2022-2025
    # dataset, and RetainDB's roughly 7-day recency window would empty every
    # temporally-worded question. That would be a run that exits 0 and
    # produces a meaningless number. Refuse rather than bank that.
    if not _SEND_QUESTION_DATE and os.environ.get("BENCH_CLOCKSYNC") != "1":
        print(
            "eval_retaindb_server: refusing to run without a temporal anchor.\n"
            "  The adapter no longer sends `question_date` (plugin-faithful), so recall\n"
            "  depends on the server's clock being at the session date, that is, on\n"
            "  BENCH_CLOCKSYNC=1. It is unset, so the server would anchor on real\n"
            "  wall-clock time, and RetainDB's roughly 7-day recency window would empty every\n"
            "  'has X changed recently?' question.\n"
            "  Fix: set BENCH_CLOCKSYNC=1 (use a *_clocksync preset), or set\n"
            "  RETAINDB_SEND_QUESTION_DATE=1 for a deliberate non-plugin-faithful\n"
            "  diagnostic contrast.",
            file=sys.stderr, flush=True,
        )
        sys.exit(2)

    # eval_common.run_eval() returns False, not an exception, on a fatal
    # error. Propagate that as a nonzero exit, otherwise a run that died
    # mid-way would look successful, and the entrypoint's `set -e` would never
    # fire (see the local adapter's matching note).
    ok = Generate_User_RetainDB_Server_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        profile=args.profile,
        base_url=args.base_url,
        api_key=args.api_key,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
        plugin_overlay=args.plugin_overlay,
        granularity=args.retain_granularity,
        wait_lifecycle=args.wait_lifecycle,
        lifecycle_timeout_s=args.lifecycle_timeout_s,
        lifecycle_poll_s=args.lifecycle_poll_s,
        promotion_mode=args.promotion_mode,
    )
    raise SystemExit(0 if ok else 1)
