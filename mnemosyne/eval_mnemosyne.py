"""MemConflict evaluation adapter for the Mnemosyne memory system.

The shared ``benchmark/eval_common.py`` driver holds the provider-agnostic
pipeline: dataset iteration, dialogue flattening, the answer prompt and LLM
call, results-row emission, and compaction. Every provider uses this same
driver. This file adds only the Mnemosyne-specific binding: per-persona
SQLite store setup, ``remember()`` ingestion, ``recall()`` retrieval, and the
lifecycle, canonical, and oracle arm logic.

Pipeline, per persona (one row of ``Data/Step4_4.jsonl``):
  * The code creates a fresh, isolated SQLite database for the persona.
  * The code ingests every session in ``Full_Session_Chain`` in chronological
    order. Each dialogue message becomes one ``remember()`` call with
    role-prefixed content.

    NOTE ON TIME: the code ingests messages in dialogue order, but the
    baseline does not pass the dataset's simulated session dates into
    Mnemosyne. Each ``remember()`` call gets the current wall-clock time, and
    the default ``recall()`` uses ``temporal_weight=0.0``. Ranking is driven
    by dense similarity, FTS, and uniform importance, not by simulated
    chronology. ``--lifecycle`` (and arms built on it) backdates rows to the
    dataset chronology.
  * After the code ingests a session, it answers each of that session's
    questions. It calls ``recall()`` for memories, then asks the answer LLM
    to answer using only the top ``top_k`` of them.

The upstream ``eval_scoring.py`` scores the produced JSONL unchanged. Each
question carries ``Model_Answer`` and ``Retrieved_Memories`` (``memory``,
``created_at``, ``score``), the exact format the generic scorer expects.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# The shared harness modules (eval_common, llm_reasoning, the scorers) live
# in ../benchmark. This insert makes them importable no matter the launch
# directory.
_SHARED_HARNESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark"))
if _SHARED_HARNESS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_HARNESS_DIR)

from dotenv import load_dotenv

import eval_common  # noqa: E402
from eval_common import (  # noqa: E402  (re-exports keep old imports working)
    ANSWER_SYSTEM_PROMPT,          # re-exported for measure_context.py (via alias below)
    MAX_STORED_RETRIEVED_MEMORIES,
    Build_Retrieved_Memory_Context,  # re-exported for measure_context.py
    Build_Session_Dialogue_List,     # re-exported for measure_context.py
    Parse_Query_Now_Timestamp,
    ProviderBinding,
    add_common_eval_args,
    opt_int,
    record_provider_retrieval,
)

# The code imports Mnemosyne lazily inside Setup. This makes sure any
# embedding-related environment variables the caller set are already in place.

load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# Back-compat alias. The answer prompt is the shared one, see eval_common.
MNEMOSYNE_ANSWER_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Fast, durability-free SQLite for the disposable per-persona DBs
# --------------------------------------------------------------------------
# Each persona gets a throwaway SQLite DB (tempfile.mkdtemp in
# MnemosyneBinding.begin_persona). With up to 30 parallel shards, fsync
# traffic from SQLite's default durability settings dominated ingest
# wall-time on the Docker Desktop / WSL2 host. These DBs are pure benchmark
# scratch. A crash only means the shard re-runs, so durability guarantees buy
# nothing here and we trade them all for speed.
#
# Mnemosyne opens its own sqlite connections internally (beam.conn is the
# main one, CanonicalStore shares it, and other components may open their own
# connections to the same file), and the submodule must not be modified.
# Upstream exposes no MNEMOSYNE_* env var for sqlite pragmas (core/beam.py
# hardcodes journal_mode=WAL, busy_timeout, and foreign_keys, and never sets
# synchronous). So the one hook that catches every connection is wrapping
# sqlite3.connect() here at module level. Mnemosyne is imported lazily inside
# Setup_Mnemosyne(), so installing the wrapper at import time is guaranteed to
# run first. Patching the attribute on the shared sqlite3 module object also
# covers mnemosyne's own `import sqlite3`.
#
# Pragmas applied to each new file-backed connection:
#   synchronous=OFF   — skips every fsync. This is the main speed gain (WAL's
#                       default is NORMAL, rollback-journal's is FULL, both
#                       fsync on commit).
#   journal_mode=WAL  — deliberately not MEMORY. Mnemosyne's connection
#                       factory runs `PRAGMA journal_mode=WAL` itself right
#                       after connect (core/beam.py), so MEMORY would be
#                       overridden immediately on the main connection anyway.
#                       A MEMORY journal is also unsafe with the multiple
#                       connections mnemosyne opens to the same file. WAL with
#                       synchronous=OFF already removes the fsyncs, which is
#                       where the time went.
#   temp_store=MEMORY — keeps sort and index temp b-trees in RAM, not in temp
#                       files.
#   cache_size=-65536 — 64 MiB page cache (negative value = KiB) so hot pages
#                       stay in-process instead of round-tripping the
#                       filesystem.
#   mmap_size=256MiB  — reads pages straight from the OS page cache through
#                       mmap.
#
# Gated by BENCH_SQLITE_FAST=1: the container entrypoint defaults it on.
# Unset or 0 (for example bare host runs) keeps stock SQLite behavior, so
# this stays backward-compatible. The code is defensive by design: it
# swallows pragma failures, leaves ':memory:' and URI mode=memory databases
# untouched, and falls back to returning the connection unmodified on any
# unexpected error.
def _install_fast_sqlite_pragmas() -> None:
    import sqlite3

    # A second call (re-import, exec) must not wrap the connection twice.
    if getattr(sqlite3.connect, "_bench_fast_sqlite", False):
        return
    _original_connect = sqlite3.connect

    _FAST_PRAGMAS = (
        "PRAGMA synchronous=OFF",
        "PRAGMA journal_mode=WAL",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA cache_size=-65536",
        "PRAGMA mmap_size=268435456",
    )

    def _fast_connect(*args, **kwargs):
        # Passes every argument through untouched. Mnemosyne uses factory=,
        # check_same_thread=, and others, and connect must keep working
        # exactly as before.
        conn = _original_connect(*args, **kwargs)
        try:
            # `database` may be the first positional arg or a kwarg, and may
            # be a str, bytes, or PathLike. Normalize it for the memory-DB
            # checks below.
            database = kwargs.get("database", args[0] if args else "")
            db_name = os.fsdecode(database) if database else ""
            # Tunes only file-backed DBs. In-memory DBs gain nothing, and
            # journal_mode changes on shared-cache memory DBs can misbehave.
            if db_name and db_name != ":memory:" and "mode=memory" not in db_name:
                for pragma in _FAST_PRAGMAS:
                    try:
                        conn.execute(pragma)
                    except sqlite3.Error:
                        pass  # A tuning pragma must never break a connect.
        except Exception:
            pass  # Worst case: behave exactly like the stock connect.
        return conn

    _fast_connect._bench_fast_sqlite = True
    sqlite3.connect = _fast_connect


if os.environ.get("BENCH_SQLITE_FAST", "0").strip() == "1":
    _install_fast_sqlite_pragmas()


# --------------------------------------------------------------------------
# Mnemosyne backend
# --------------------------------------------------------------------------
def Setup_Mnemosyne(db_path: str, session_id: str):
    """Construct an isolated Mnemosyne instance backed by ``db_path``."""
    from pathlib import Path
    from mnemosyne.core.memory import Mnemosyne

    return Mnemosyne(session_id=session_id, db_path=Path(db_path))


def Add_Session_Dialogue_To_Mnemosyne(
    memory,
    dialogue_messages: List[Dict[str, Any]],
    importance: float = 0.6,
    extract: bool = False,
    backdate: bool = False,
    session_base: Optional[datetime] = None,
) -> Tuple[float, int, int, Optional[datetime]]:
    """Ingest one session's messages as individual memories.

    ``remember()`` returns None when Mnemosyne's core write filter
    (``mnemosyne.core.filters.should_remember``) rejects a message. The code
    counts only messages that were actually stored, so the ingest count is a
    stored-memory count, not an attempted-write count.

    The code applies ``importance=0.6`` uniformly. Mnemosyne's own default is
    0.5. Because every memory gets the same value here, the importance term
    is constant across memories and has no effect on relative ranking.

    ``extract=True`` turns on Mnemosyne's LLM fact-extraction on ingest
    (``remember(extract=True)`` calls ``_extract_and_store_facts``), which
    calls the Mnemosyne-internal LLM set by ``MNEMOSYNE_LLM_BASE_URL`` and
    ``MNEMOSYNE_LLM_MODEL``. This is best-effort and never fails
    ``remember()``, but it adds one extraction LLM call per stored message,
    so it is much slower.

    ``backdate=True`` (with ``session_base`` set) backdates each stored
    ``working_memory`` row to ``session_base + timedelta(minutes=turn_index)``
    (turn order preserved, messages one minute apart). The caller chooses
    ``session_base`` so consecutive sessions are always more than one hour
    apart. This makes the simulated chronology drive Mnemosyne's ``sleep()``
    cross-session conflict detection, which needs a gap over one hour,
    instead of every row sharing wall-clock ``datetime.now()``. The
    timestamp string matches the core write format exactly
    (``datetime.isoformat()``, see beam.py:3295). The non-backdated path
    issues no UPDATEs and stays unchanged.

    Backdating is decoupled from retirement. Both ``--lifecycle`` (which runs
    ``Run_Session_Lifecycle``) and ``--use_dataset_time`` (which does not)
    drive it. The binding invokes the retirement pass separately.

    Returns (add_duration_ms, stored_count, filtered_count, last_ts). Here
    ``last_ts`` is the datetime of the last message's slot in this session
    (turn order), or None in the non-backdated path.
    """
    if not dialogue_messages:
        return 0.0, 0, 0, session_base
    start = time.time()
    stored = 0
    filtered = 0
    do_backdate = backdate and session_base is not None
    last_ts: Optional[datetime] = None
    total_msgs = len(dialogue_messages)
    for turn_index, message in enumerate(dialogue_messages):
        content = f"{message['role']}: {message['content']}"
        if extract:
            _extract_t0 = time.time()
        mem_id = memory.remember(content, source="conversation", importance=importance,
                                 extract=extract)
        if extract:
            _extract_ms = (time.time() - _extract_t0) * 1000.0
            print(f"[DEBUG] extract_remember msg={turn_index + 1}/{total_msgs} ms={_extract_ms:.0f}")
        if do_backdate:
            # Advances the running clock per message in turn order, so the
            # cross-session gap is measured from the previous message.
            ts = session_base + timedelta(minutes=turn_index)
            last_ts = ts
        if mem_id:
            stored += 1
            if do_backdate:
                # Matches the core write format. beam.py:3295 stores
                # ``datetime.now().isoformat()`` into working_memory.timestamp.
                try:
                    memory.beam.conn.execute(
                        "UPDATE working_memory SET timestamp = ? WHERE id = ?",
                        (ts.isoformat(), mem_id),
                    )
                except Exception as exc:  # pragma: no cover - non-fatal
                    print(f"[DEBUG] lifecycle timestamp UPDATE failed for {mem_id}: {exc}")
        else:
            filtered += 1
        if (turn_index + 1) % 20 == 0:
            print(f"[DEBUG] persona-ingest msg={turn_index + 1}/{total_msgs} stored={stored}")
    if do_backdate:
        # Batch the commit once per session for speed.
        try:
            memory.beam.conn.commit()
        except Exception as exc:  # pragma: no cover - non-fatal
            print(f"[DEBUG] lifecycle timestamp commit failed: {exc}")
    return (time.time() - start) * 1000.0, stored, filtered, last_ts


# --------------------------------------------------------------------------
# Plugin-fidelity ingestion (opt-in via --plugin_config {user,both})
# --------------------------------------------------------------------------
# This is a faithful replica of the Hermes Mnemosyne plugin's per-turn write
# path (MnemosyneMemoryProvider.sync_turn, integrations/hermes/src/
# mnemosyne_hermes/__init__.py:1278). The plugin writes one remember() call
# per role per completed exchange. It uses fixed per-role importances and
# entity extraction, not LLM fact extraction. It gates on raw content length
# and truncates content to a per-role character limit. These constants
# replicate the plugin's default limits. The plugin reads the same limits
# from env vars via _sync_turn_user_limit() / _sync_turn_assistant_limit()
# (__init__.py:427 / :444), so this code honors the same env overrides for
# parity. A value of 0 disables truncation, matching the plugin.
def _plugin_user_limit() -> int:
    """Plugin per-turn USER truncation limit (default 500; __init__.py:427)."""
    raw = os.environ.get("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "500").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 500


def _plugin_assistant_limit() -> int:
    """Plugin per-turn ASSISTANT truncation limit (default 800; __init__.py:444)."""
    raw = os.environ.get("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "800").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 800


def Add_Session_Dialogue_Plugin_Sync(
    memory,
    dialogue_messages: List[Dict[str, Any]],
    plugin_config: str,
    backdate: bool = False,
    session_base: Optional[datetime] = None,
    on_sync_turn: Optional[Callable[[], None]] = None,
) -> Tuple[float, int, int, Optional[datetime], Dict[str, int]]:
    """Ingest one session as plugin-faithful per-exchange ``sync_turn`` writes.

    This mirrors ``MnemosyneMemoryProvider.sync_turn`` (integrations/hermes/
    src/mnemosyne_hermes/__init__.py:1278-1317). The code groups the
    session's flattened dialogue into user/assistant exchanges with
    ``eval_common.Pair_Exchange_Turns``. For each exchange the plugin is
    called with ``(user_content, assistant_content)`` and writes:

      * the USER role (when ``plugin_config`` is "user" or "both"), gated on
        ``len(user_content) > 5``, as ``"[USER] " + user_content[:user_limit]``
        with ``importance=0.5``, ``source="conversation"``, ``extract_entities=True``;
      * the ASSISTANT role (only when ``plugin_config == "both"``), gated on
        ``len(assistant_content) > 10``, as ``"[ASSISTANT] " + assistant_content
        [:assistant_limit]`` with ``importance=0.15``, the same source, and
        entity extraction on.

    There is no LLM fact extraction (``extract=False``). The plugin's
    automatic writes do entity extraction only (``extract_entities=True``),
    never ``extract``.

    The real plugin never issues a ``sync_turn`` call for a lone assistant
    message (an exchange with no user half), so the code drops it and counts
    it in the returned ``dropped`` stat. A lone trailing user message is
    written, because it is the user role of an exchange whose assistant half
    is empty.

    ``on_sync_turn`` (opt-in, the --plugin_auto_sleep arm) runs exactly once
    per exchange for which the plugin would have issued a ``sync_turn`` call,
    that is every user-led exchange, after its writes. This mirrors the
    plugin's ``self._turn_count += 1`` at the tail of ``sync_turn``
    (__init__.py:1318). Lone-assistant (dropped) exchanges do not tick it,
    because the real plugin's Hermes host never calls ``sync_turn`` for them.
    The callback owns the cadence math (``% 10``) and the actual sleep, so
    this function stays cadence-agnostic.

    ``backdate`` and ``session_base`` apply the same dataset-chronology
    timestamp restoration as the lifecycle path, advancing one minute per
    actual write.

    Returns (add_duration_ms, stored_count, filtered_count, last_ts, stats).
    Here ``stats`` carries per-role counters: user_written, assistant_written,
    dropped, user_truncated, assistant_truncated, filtered.
    """
    stats = {
        "user_written": 0, "assistant_written": 0, "dropped": 0,
        "user_truncated": 0, "assistant_truncated": 0, "filtered": 0,
    }
    if not dialogue_messages:
        return 0.0, 0, 0, session_base, stats

    write_user = plugin_config in ("user", "both")
    write_assistant = plugin_config == "both"
    user_limit = _plugin_user_limit()
    assistant_limit = _plugin_assistant_limit()

    start = time.time()
    stored = 0
    do_backdate = backdate and session_base is not None
    last_ts: Optional[datetime] = None
    turn_index = 0

    def _write(content: str, importance: float) -> Optional[str]:
        nonlocal stored, last_ts, turn_index
        mem_id = memory.remember(
            content, source="conversation", importance=importance,
            extract_entities=True,
        )
        ts = None
        if do_backdate:
            ts = session_base + timedelta(minutes=turn_index)
            last_ts = ts
        if mem_id:
            stored += 1
            if do_backdate:
                try:
                    memory.beam.conn.execute(
                        "UPDATE working_memory SET timestamp = ? WHERE id = ?",
                        (ts.isoformat(), mem_id),
                    )
                except Exception as exc:  # pragma: no cover - non-fatal
                    print(f"[DEBUG] plugin timestamp UPDATE failed for {mem_id}: {exc}")
        else:
            stats["filtered"] += 1
        turn_index += 1
        return mem_id

    exchanges = eval_common.Pair_Exchange_Turns(dialogue_messages)
    total_exchanges = len(exchanges)
    for exch_idx, group in enumerate(exchanges, start=1):
        first = group[0]
        if first.get("role") != "user":
            # Lone assistant exchange. The plugin never invokes sync_turn for it.
            stats["dropped"] += len(group)
            continue
        user_content = str(first.get("content", ""))
        assistant_content = ""
        if len(group) > 1 and group[1].get("role") == "assistant":
            assistant_content = str(group[1].get("content", ""))

        # USER role. The sync_turn gate needs truthy raw content longer than 5 chars.
        if write_user and user_content and len(user_content) > 5:
            uc = user_content[:user_limit] if user_limit > 0 else user_content
            if user_limit > 0 and len(user_content) > user_limit:
                stats["user_truncated"] += 1
            if _write(f"[USER] {uc}", 0.5):
                stats["user_written"] += 1
        # ASSISTANT role. The sync_turn gate needs truthy raw content longer than 10 chars.
        if write_assistant and assistant_content and len(assistant_content) > 10:
            ac = assistant_content[:assistant_limit] if assistant_limit > 0 else assistant_content
            if assistant_limit > 0 and len(assistant_content) > assistant_limit:
                stats["assistant_truncated"] += 1
            if _write(f"[ASSISTANT] {ac}", 0.15):
                stats["assistant_written"] += 1

        # The plugin's sync_turn ran for this user-led exchange. This ticks the
        # auto-sleep cadence after its writes, at the same point the plugin does
        # (__init__.py:1318). Lone-assistant exchanges already hit `continue`
        # above, so they never reach here, matching the plugin: no sync_turn
        # means no tick.
        if on_sync_turn is not None:
            on_sync_turn()

        if exch_idx % 20 == 0:
            print(f"[DEBUG] persona-ingest msg={exch_idx}/{total_exchanges} stored={stored}")

    if do_backdate:
        try:
            memory.beam.conn.commit()
        except Exception as exc:  # pragma: no cover - non-fatal
            print(f"[DEBUG] plugin timestamp commit failed: {exc}")
    return (time.time() - start) * 1000.0, stored, stats["filtered"], last_ts, stats


# --------------------------------------------------------------------------
# Plugin auto-sleep cadence (opt-in via --plugin_auto_sleep, composes with
# --plugin_config)
# --------------------------------------------------------------------------
# GROUND TRUTH: the real Hermes Mnemosyne plugin
# (integrations/hermes/src/mnemosyne_hermes/__init__.py).
#
#   * The plugin fires consolidation on a turn cadence inside sync_turn():
#       self._turn_count += 1
#       if self._auto_sleep_enabled and self._turn_count % 10 == 0:
#           self._maybe_auto_sleep()
#     (__init__.py:1318-1320). Here "turn" means one sync_turn() call, which
#     is one completed user/assistant exchange. So the cadence is every 10
#     exchanges, not every 10 raw messages. Our plugin ingest issues writes
#     per exchange through Add_Session_Dialogue_Plugin_Sync, so this code
#     counts exchanges to match.
#
#   * _maybe_auto_sleep() (__init__.py:1383) is not an unconditional sleep.
#     It does four things in order:
#       1. Reads get_working_stats()["total"] and proceeds only when working
#          is above self._auto_sleep_threshold (default 50, __init__.py:569).
#       2. Computes cutoff = now - WM_TTL_HOURS // 2 and calls
#          _count_unconsolidated_before(cutoff). It returns early if 0 rows
#          are eligible.
#       3. Reserves a per-session reflection budget (default max 3 sleeps
#          per session, __init__.py:576), and skips once exhausted.
#       4. Calls sleep_all_sessions() if present, else sleep(), with no
#          force=True (__init__.py:1404), on a daemon thread joined for
#          only 5 seconds (MNEMOSYNE_AUTO_SLEEP_TIMEOUT).
#
#   * on_session_end() (__init__.py:2540) also runs beam.sleep() once per
#     session, also unforced and budget-gated.
#
# This code reproduces that behavior for the benchmark, with two deliberate
# deviations, each justified by the fairness contract:
#
#   (a) CADENCE, faithful: fires every EXCHANGE_CADENCE (10) exchanges,
#       tracked across the whole persona. The plugin's _turn_count is
#       per-provider-instance, which equals per-persona-conversation here,
#       and is never reset per session. The code also fires once more at
#       each session boundary to mirror on_session_end().
#
#   (b) sleep_all_sessions vs sleep, faithful: this code prefers
#       sleep_all_sessions, matching the plugin's own preference. For our
#       single-session-id-per-persona store the two calls are equivalent
#       (one session).
#
#   (c) force=True, a deliberate deviation in mechanism with an equivalent
#       effect. The plugin calls sleep unforced and relies on an age cutoff
#       (now - TTL // 2) to select rows old enough to consolidate. That
#       mechanism cannot work under this benchmark's compressed clock, and
#       it fails silently in both directions. Both failures were measured,
#       and neither raised an error:
#         1. The pre-check gate used beam's inflated TTL (about 1000 years,
#            set by the entrypoint so backdated rows are not expired). The
#            cutoff landed in the year 1526, nothing was older, eligible
#            was 0, and 290 ticks fired 0 sleeps.
#         2. Fixing that gate (see _plugin_auto_sleep_eligibility_ttl_hours)
#            let it pass, but sleep_all_sessions recomputes the same cutoff
#            from the same inflated constant internally (beam.py:8369). That
#            produced 159 fired sleeps, every one status="no_op", with 0
#            summaries and 0 proposals.
#       These two uses of the constant cannot be decoupled without editing
#       the pinned submodule. force=True sets the cutoff to datetime.max,
#       which under backdating selects exactly the rows the plugin's real
#       84-hour cutoff would select: every row is stamped 2022-23, so all
#       rows are far older than any realistic cutoff. It does not
#       over-consolidate anything. The canonical arm already does the same
#       thing for the same reason (Run_Session_Sleep calls sleep(force=True)),
#       so the two sleep-based arms stay consistent with each other.
#
#   (d) DRAINED, not 5-second async, a deliberate quality-arm deviation. The
#       plugin joins the sleep daemon thread for only 5 seconds and
#       continues if it overruns (__init__.py:1405-1409). This benchmark
#       measures the consolidated state, so a half-finished consolidation
#       would be an artifact of wall-clock timing, not of the algorithm.
#       This is the same "drained, not async" ruling applied to Hindsight
#       arm-B/C auto-consolidation (WAIT_CONSOLIDATION). So this code calls
#       sleep synchronously, inline, and lets it run to completion before
#       any question is answered. This is the only faithful way to compare
#       quality.
#
#   (e) Reflection budget (max 3 per session), reproduced per dataset
#       session. _reserve_reflection_budget (__init__.py:887) gates both the
#       cadence path (:1398) and on_session_end (:2549) against
#       _reflect_max_calls_per_session (default 3, env
#       MNEMOSYNE_REFLECT_MAX_CALLS_PER_SESSION, a negative value disables
#       it, :576). The plugin initializes the counter per provider instance
#       (:577) and never resets it elsewhere, and Hermes constructs the
#       provider per session (on_session_end is its lifecycle terminus). So
#       the faithful mapping is one shared budget of 3 per dataset session,
#       consumed by both the cadence path and session_end, and reset at each
#       session boundary. Without this the arm would fire about 5 to 6
#       sleeps per session (about 45 exchanges gives 4 cadence ticks plus 1
#       session end), nearly double the real plugin's consolidation churn,
#       in the arm whose entire claim is plugin fidelity. The code logs and
#       counts budget-exhausted ticks (auto_sleep_budget_skipped) so the
#       summary shows how often the cap applied, matching the plugin's
#       reflect_budget_exhausted skips.
#
# The code honors the threshold (working > 50) and the eligibility gates,
# because they are part of the algorithm and decide whether a cadence tick
# does real work.
PLUGIN_AUTO_SLEEP_EXCHANGE_CADENCE = 10  # plugin: _turn_count % 10 == 0 (__init__.py:1319)


def _plugin_auto_sleep_threshold() -> int:
    """Working-memory count above which auto-sleep does real work.

    This mirrors the plugin's ``self._auto_sleep_threshold`` default of 50
    (__init__.py:569, overridable there through config ``sleep_threshold``).
    The env var name is not the same one Mnemosyne uses for max-tokens and
    similar settings. This dedicated MNEMOSYNE_AUTO_SLEEP_THRESHOLD lets a
    smoke test lower it, but it defaults to the plugin's 50 so a full run
    matches production behavior.
    """
    raw = os.environ.get("MNEMOSYNE_AUTO_SLEEP_THRESHOLD", "50").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 50


def _plugin_auto_sleep_eligibility_ttl_hours() -> int:
    """TTL in hours for auto-sleep's eligibility cutoff.

    This is deliberately not the benchmark's inflated working-memory TTL.

    The plugin computes ``cutoff = now - WM_TTL_HOURS // 2`` and consolidates
    rows with ``timestamp < cutoff`` (beam._count_unconsolidated_before,
    beam.py:4127). In a real deployment WM_TTL is the shipped 168 hours, so
    the cutoff sits about 84 hours in the past, and rows become eligible once
    they are a few days old. That is the behavior this code reproduces.

    This benchmark raises MNEMOSYNE_WM_TTL_HOURS to about 1000 years
    (entrypoint.mnemosyne.sh) for a different reason: without it, backdated
    2022 dialogue would be deleted as expired on the next remember() call.
    Feeding that inflated TTL into the cutoff arithmetic puts the cutoff in
    the year 1526. No backdated row is older than that, eligible stays 0, and
    auto-sleep silently never fires. Measured: 290 cadence and session-end
    ticks fired 0 sleeps. That made the arm a no-op that would have produced
    a duplicate of the non-sleep arm. Two settings that are each correct on
    their own defeat each other through one shared knob.

    So the eligibility cutoff uses the plugin's real default TTL (168 hours)
    unless overridden. Against backdated 2022-23 rows, every unconsolidated
    row is then older than the cutoff and eligible, matching what an
    always-on deployment would have consolidated over the simulated
    multi-month span.
    """
    raw = os.environ.get("MNEMOSYNE_AUTO_SLEEP_ELIGIBILITY_TTL_HOURS", "168").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 168


def _plugin_reflect_max_calls() -> Optional[int]:
    """Per-session reflection (sleep) budget.

    This mirrors the plugin's ``_reflect_max_calls_per_session``
    (__init__.py:576): default 3, the same env var name, and, like the
    plugin's ``_parse_env_optional_int``, a negative value disables the cap
    entirely and returns None.
    """
    raw = os.environ.get("MNEMOSYNE_REFLECT_MAX_CALLS_PER_SESSION", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return None if value < 0 else value


def Run_Plugin_Auto_Sleep(
    memory, persona_tag: str, trigger: str, exchange_index: int,
    sleep_totals: Dict[str, int], reflect_state: Optional[Dict[str, int]] = None,
) -> None:
    """Fire one plugin-faithful auto-sleep pass, drained to completion.

    This replicates ``MnemosyneMemoryProvider._maybe_auto_sleep``
    (__init__.py:1383): gate on working-memory count above threshold, then
    on there being any unconsolidated rows before the age cutoff, then call
    ``sleep_all_sessions()`` (falling back to ``sleep()``) without ``force``,
    exactly as the plugin does. Unlike the plugin, this code runs it inline
    with no 5-second daemon join. This is a quality arm, so consolidation
    must fully drain before questions are answered, the same ruling as
    Hindsight WAIT_CONSOLIDATION.

    Every invocation is logged to stdout with the persona, trigger
    (``cadence`` or ``session_end``), exchange index, working count,
    eligibility, wall-clock duration, and the proposal, applied, and summary
    counts the API returns. A 30-persona run can be monitored purely by
    tailing these lines. Any failure logs and returns, because a broken
    sleep must never crash the persona.

    Updates ``sleep_totals`` in place: ``auto_sleep_invocations`` (cadence
    ticks that fired real work), ``auto_sleep_skipped`` (ticks gated out
    below threshold or with no eligible rows), plus cumulative summaries,
    proposals, and applied counts so persona_count_extras can prove the arm
    did work.
    """
    beam = getattr(memory, "beam", None)
    if beam is None:
        return
    # Gate 1: working-memory count over threshold (plugin __init__.py:1387).
    threshold = _plugin_auto_sleep_threshold()
    working = 0
    try:
        working = int(beam.get_working_stats().get("total", 0) or 0)
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[DEBUG] auto_sleep persona {persona_tag}: get_working_stats failed: {exc}")
        return
    if working <= threshold:
        sleep_totals["auto_sleep_skipped"] = sleep_totals.get("auto_sleep_skipped", 0) + 1
        print(f"[DEBUG] auto_sleep persona {persona_tag} trigger={trigger} "
              f"exch={exchange_index} SKIP working={working}<=threshold={threshold}")
        return
    # Gate 2: any unconsolidated rows before the age cutoff (plugin
    # :1393-1396). The plugin uses now - WM_TTL_HOURS // 2. With the
    # backdated-ingest TTL guard (about 1000 years), that cutoff sits
    # centuries in the past, so every 2022-or-later backdated row is older
    # than it and eligible, matching what a real long-running deployment
    # would have consolidated over the simulated span. This code reads the
    # same WM_TTL the beam module compiled at import, not a re-parse, so the
    # cutoff matches the plugin's arithmetic exactly.
    eligible = None
    try:
        # This uses the eligibility TTL, not beam.WORKING_MEMORY_TTL_HOURS.
        # See _plugin_auto_sleep_eligibility_ttl_hours() for why feeding the
        # benchmark's inflated expiry-prevention TTL into this arithmetic
        # makes eligible always 0 and turns the arm into a silent no-op.
        cutoff = (datetime.now()
                  - timedelta(hours=_plugin_auto_sleep_eligibility_ttl_hours() // 2)).isoformat()
        eligible = int(beam._count_unconsolidated_before(cutoff))
    except Exception as exc:  # pragma: no cover - non-fatal
        # If the private helper is unavailable, this falls through and lets
        # sleep() decide. Its own SELECT applies the identical cutoff filter.
        print(f"[DEBUG] auto_sleep persona {persona_tag}: eligibility probe failed "
              f"({exc}); proceeding to sleep() which re-applies the cutoff")
    if eligible == 0:
        sleep_totals["auto_sleep_skipped"] = sleep_totals.get("auto_sleep_skipped", 0) + 1
        print(f"[DEBUG] auto_sleep persona {persona_tag} trigger={trigger} "
              f"exch={exchange_index} SKIP working={working} eligible=0")
        return
    # Gate 3: per-session reflection budget (plugin
    # _reserve_reflection_budget, __init__.py:887, called from both the
    # cadence path :1398 and on_session_end :2549). This code reserves
    # budget only after gates 1 and 2 pass, exactly like the plugin, so a
    # below-threshold tick never consumes budget. ingest_session resets
    # reflect_state per dataset session, see deviation note (e) above for
    # why that maps to the plugin's per-provider-instance counter.
    if reflect_state is not None:
        max_calls = _plugin_reflect_max_calls()
        if max_calls is not None and reflect_state.get("used", 0) >= max_calls:
            sleep_totals["auto_sleep_budget_skipped"] = (
                sleep_totals.get("auto_sleep_budget_skipped", 0) + 1)
            print(f"[DEBUG] auto_sleep persona {persona_tag} trigger={trigger} "
                  f"exch={exchange_index} SKIP reflect_budget_exhausted "
                  f"used={reflect_state.get('used', 0)}/{max_calls}")
            return
        reflect_state["used"] = reflect_state.get("used", 0) + 1
    # Fires sleep, preferring sleep_all_sessions (plugin :1404) with no
    # force, since the plugin never forces. Runs inline so it drains fully
    # before questions are answered.
    sleep_fn: Callable[..., Any] = getattr(beam, "sleep_all_sessions", None) or memory.sleep
    t0 = time.time()
    try:
        # force=True is required here and is equivalent to the plugin's
        # unforced behavior under this benchmark's backdated clock, see
        # deviation note (c) above for the full measurement (159 fired
        # sleeps, all no-ops, before force=True was added).
        result = sleep_fn(force=True) or {}
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[DEBUG] auto_sleep persona {persona_tag} trigger={trigger} "
              f"exch={exchange_index} sleep FAILED: {exc}")
        return
    dur_ms = (time.time() - t0) * 1000.0
    # sleep_all_sessions() and sleep() return different shapes, so this reads defensively.
    summaries = int(result.get("summaries_created", 0) or 0)
    refresh = result.get("model_refresh") or {}
    proposals = int(refresh.get("proposals", 0) or 0)
    applied = int(refresh.get("applied", 0) or 0)
    status = result.get("status", "?")
    sleep_totals["auto_sleep_invocations"] = sleep_totals.get("auto_sleep_invocations", 0) + 1
    sleep_totals["summaries_created"] = sleep_totals.get("summaries_created", 0) + summaries
    sleep_totals["mr_proposals"] = sleep_totals.get("mr_proposals", 0) + proposals
    sleep_totals["model_refresh_applied"] = sleep_totals.get("model_refresh_applied", 0) + applied
    # Keeps sleep's own model-refresh bookkeeping rows out of the next sleep
    # batch and out of recall. Search_Mnemosyne_For_Question already filters
    # them from the answer context by source. Stamping consolidated_at stops
    # the next cadence tick from re-summarizing consolidation plumbing, the
    # same neutralizer the canonical arm applies in Run_Session_Sleep.
    try:
        beam.conn.execute(
            "UPDATE working_memory SET consolidated_at = ? "
            "WHERE source = 'sleep_model_refresh_proposal' AND consolidated_at IS NULL",
            (datetime.now().isoformat(),),
        )
        beam.conn.commit()
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[DEBUG] auto_sleep persona {persona_tag}: proposal-row stamp failed: {exc}")
    print(f"[DEBUG] auto_sleep persona {persona_tag} trigger={trigger} "
          f"exch={exchange_index} FIRED working={working} eligible={eligible} "
          f"status={status} dur_ms={dur_ms:.0f} summaries={summaries} "
          f"mr_proposals={proposals} mr_applied={applied} "
          f"invocations={sleep_totals['auto_sleep_invocations']}")


def Run_Plugin_Session_Sleep(
    memory, persona_tag: str, session_tag: Any, sleep_totals: Dict[str, int],
) -> None:
    """Force one consolidation pass at the end of a dataset session.

    WHY THIS EXISTS (user ruling 2026-08-02, mirroring Honcho's
    HONCHO_DREAM_AFTER_SESSION, "shipped consolidation, manually cadenced").
    Under BENCH_CLOCKSYNC the shipped 168h working-memory TTL is live, and
    ``_trim_working_memory`` (beam.py:3825), which runs inside every
    ``remember()``, DELETES each row that still has ``consolidated_at IS
    NULL`` and a timestamp before the cutoff (beam.py:3836-3849). The median
    gap between dataset sessions is 29 days, so the next session's first
    write removes the previous session's unconsolidated rows. The plugin's
    own auto-sleep cannot prevent that: its first gate needs
    ``working > 50`` (__init__.py:1387), and that count needs exactly the
    cross-session accumulation the trim removes. Measured on the ft27mn
    smoke: 2 sleep invocations in 277 cadence ticks, 12 episodic rows, and
    21 of 122 questions with zero recall candidates.

    THE MECHANISM. ``sleep(force=True)`` moves the age cutoff to
    ``datetime.max`` (beam.py:8056-8058) and stamps ``consolidated_at`` on
    every row it claims (beam.py:8098-8106). The trim's DELETE is gated on
    ``consolidated_at IS NULL``, so those rows become exempt (beam.py:3839,
    and the docstring at :3828: "consolidated rows are exempt from trim").
    Recall still returns them: the working-memory read path applies no
    ``consolidated_at`` filter (beam.py:5659-5679). So the TTL stays shipped,
    the vendor's own ``sleep()`` does the consolidation, and only the cadence
    is ours.

    This uses the same call selection as ``Run_Plugin_Auto_Sleep``
    (``sleep_all_sessions`` when the beam has it, else ``memory.sleep``), so
    with both flags on the two paths fire one mechanism, not two. It stamps
    the model-refresh bookkeeping rows for the same reason that function
    does. It runs inline and drained, after this session's ingest and before
    its questions (the quality-arm ruling, the same as Hindsight
    WAIT_CONSOLIDATION). Any failure logs and returns, because a broken sleep
    must not kill the persona.

    Updates ``sleep_totals`` in place: ``session_sleep_invocations``,
    ``session_sleep_no_op`` (passes that found nothing to consolidate), the
    working rows consolidated and episodic summaries produced, the
    model-refresh counts, and the last working-memory census. The census is
    overwritten each session, so the final value is the end-of-persona state
    and shows how much history survived the TTL.
    """
    beam = getattr(memory, "beam", None)
    if beam is None:
        return
    sleep_fn: Callable[..., Any] = getattr(beam, "sleep_all_sessions", None) or memory.sleep
    t0 = time.time()
    try:
        result = sleep_fn(force=True) or {}
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[DEBUG] session_sleep persona {persona_tag} session {session_tag}: "
              f"sleep(force=True) failed: {exc}")
        return
    dur_ms = (time.time() - t0) * 1000.0
    # sleep() and sleep_all_sessions() return different shapes, so this reads
    # defensively, the same as the auto-sleep accounting above.
    status = result.get("status", "?")
    items = int(result.get("items_consolidated", 0) or 0)
    summaries = int(result.get("summaries_created", 0) or 0)
    refresh = result.get("model_refresh") or {}
    proposals = int(refresh.get("proposals", 0) or 0)
    applied = int(refresh.get("applied", 0) or 0)
    sleep_totals["session_sleep_invocations"] = (
        sleep_totals.get("session_sleep_invocations", 0) + 1)
    if status == "no_op":
        sleep_totals["session_sleep_no_op"] = sleep_totals.get("session_sleep_no_op", 0) + 1
    sleep_totals["session_sleep_items_consolidated"] = (
        sleep_totals.get("session_sleep_items_consolidated", 0) + items)
    sleep_totals["session_sleep_summaries_created"] = (
        sleep_totals.get("session_sleep_summaries_created", 0) + summaries)
    sleep_totals["session_sleep_mr_proposals"] = (
        sleep_totals.get("session_sleep_mr_proposals", 0) + proposals)
    sleep_totals["session_sleep_mr_applied"] = (
        sleep_totals.get("session_sleep_mr_applied", 0) + applied)
    # Keeps sleep's own model-refresh bookkeeping rows out of the next sleep
    # batch and out of recall, the same neutralizer Run_Session_Sleep and
    # Run_Plugin_Auto_Sleep apply.
    try:
        beam.conn.execute(
            "UPDATE working_memory SET consolidated_at = ? "
            "WHERE source = 'sleep_model_refresh_proposal' AND consolidated_at IS NULL",
            (datetime.now().isoformat(),),
        )
        beam.conn.commit()
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[DEBUG] session_sleep persona {persona_tag}: proposal-row stamp failed: {exc}")
    # Working-memory census right after the pass. unconsolidated is the count
    # the next session's trim can still delete, so a run where this stays
    # small is the proof the arm works.
    wm_total = wm_consolidated = wm_unconsolidated = -1
    try:
        stats = beam.get_working_stats()
        wm_total = int(stats.get("total", 0) or 0)
        wm_consolidated = int(stats.get("consolidated", 0) or 0)
        wm_unconsolidated = int(stats.get("unconsolidated", 0) or 0)
        sleep_totals["session_sleep_wm_total"] = wm_total
        sleep_totals["session_sleep_wm_consolidated"] = wm_consolidated
        sleep_totals["session_sleep_wm_unconsolidated"] = wm_unconsolidated
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[DEBUG] session_sleep persona {persona_tag}: working census failed: {exc}")
    print(f"[DEBUG] session_sleep persona {persona_tag} session {session_tag} FIRED "
          f"status={status} dur_ms={dur_ms:.0f} items={items} summaries={summaries} "
          f"mr_proposals={proposals} mr_applied={applied} wm_total={wm_total} "
          f"wm_consolidated={wm_consolidated} wm_unconsolidated={wm_unconsolidated} "
          f"invocations={sleep_totals['session_sleep_invocations']}")


def Search_Mnemosyne_For_Question(
    memory, question_text: str, top_k: int,
    temporal_weight: float = 0.0, temporal_halflife: Optional[float] = None,
    query_time: Optional[Any] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Recall the top-K memories for one question.

    The code passes Mnemosyne's own documented default hybrid weights
    (0.5/0.3/0.2) explicitly rather than relying on the ``None`` defaults.
    This has no effect on the standard recall path, because
    ``_normalize_weights`` maps ``None`` to exactly these values. It is
    required for ``MNEMOSYNE_ENHANCED_RECALL=1``: enhanced recall's
    ``recall_enhanced`` runs ``kwargs.pop("vec_weight", 0.5)``, which
    returns ``None`` instead of 0.5 because ``memory.recall`` always
    forwards the key. ``adjust_weights`` then runs ``None * intent.vec_bias``
    and crashes for any query intent other than "general". Passing real
    floats avoids that upstream bug.

    ``temporal_weight`` and ``temporal_halflife`` add the Hermes plugin's
    temporal recency weighting. The plugin's own prefetch passes
    ``temporal_weight=0.2, temporal_halflife=48``
    (integrations/hermes/.../__init__.py:1170). Both core
    ``Mnemosyne.recall`` (memory.py:422) and ``beam.recall`` (beam.py:5378)
    accept these kwargs alongside the explicit vec/fts/importance weights,
    so there is no conflict. The code omits them from the call unless
    ``temporal_weight`` is non-zero, so every non-plugin arm's recall call
    stays byte-identical to before.

    ``query_time`` is the recency reference ("now") the temporal boost
    decays from (``beam._parse_query_time`` calls ``_temporal_boost``,
    beam.py:1423/1470). When the persona's rows are backdated to the
    dataset clock, the caller passes the logical session date here, so the
    boost is measured against the simulated chronology instead of
    wall-clock. ``None`` (the default) reproduces the framework fallback
    ``datetime.now(timezone.utc)``. The code omits it from the call unless
    it is non-``None``, so a non-backdated wall-clock arm's recall stays
    byte-identical.
    """
    start = time.time()
    # Over-fetches a small buffer, so filtering sleep bookkeeping rows below
    # cannot leave the answer context short of ``top_k`` real memories.
    recall_kwargs: Dict[str, Any] = dict(
        top_k=top_k + 8,
        vec_weight=float(os.environ.get("MNEMOSYNE_VEC_WEIGHT", "0.5")),
        fts_weight=float(os.environ.get("MNEMOSYNE_FTS_WEIGHT", "0.3")),
        importance_weight=float(os.environ.get("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2")),
    )
    if temporal_weight:
        recall_kwargs["temporal_weight"] = temporal_weight
        recall_kwargs["temporal_halflife"] = temporal_halflife
    if query_time is not None:
        recall_kwargs["query_time"] = query_time
    results = memory.recall(question_text, **recall_kwargs)
    duration_ms = (time.time() - start) * 1000.0

    # Diagnostic capture. "Raw" is the memory.recall() candidate rows this
    # question was answered from, from one recall call. "Ranked" is all of
    # them in Mnemosyne's own rank order, with no top_k cut and no
    # bookkeeping-row exclusion. The difference against Retrieved_Memories
    # is exactly what the harness drops below. The captured depth is the
    # top_k+8 over-fetch this path already requests. The code does not
    # widen the request for the capture.
    record_provider_retrieval(diag, raw=results, ranked=[
        {
            "memory": str(item.get("content", "")),
            "created_at": item.get("timestamp", "Unknown Time"),
            "score": item.get("score"),
        }
        for item in (results or []) if isinstance(item, dict)
    ])

    retrieved: List[Dict[str, Any]] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        # sleep()'s model-refresh writes "[MODEL_REFRESH_PROPOSAL] ..."
        # bookkeeping rows into working_memory
        # (source="sleep_model_refresh_proposal", beam.py:8285). They are
        # consolidation plumbing, not conversational memory, so the code
        # never shows them to the answer LLM or the SEH scorer. This is a
        # no-op for non-canonical arms, because that source value never
        # occurs there.
        if item.get("source") == "sleep_model_refresh_proposal":
            continue
        score = item.get("score")
        retrieved.append({
            "memory": str(item.get("content", "")),
            "created_at": item.get("timestamp", "Unknown Time"),
            "score": round(float(score), 6) if isinstance(score, (int, float)) else score,
            "dense_score": item.get("dense_score"),
            "fts_score": item.get("fts_score"),
            # Retrieval-provenance diagnostics. Without these, a fact-index
            # boost or insert would be indistinguishable from plain hybrid
            # retrieval in the benchmark output. The scorer ignores extra keys.
            "source": item.get("source"),
            "tier": item.get("tier"),
            "fact_match": item.get("fact_match"),
            "entity_match": item.get("entity_match"),
        })
        if len(retrieved) >= top_k:
            break
    return retrieved, duration_ms


# --------------------------------------------------------------------------
# Plugin prefetch() retrieval overlay (opt-in via --plugin_prefetch_overlay)
# --------------------------------------------------------------------------
# WHY THIS ARM EXISTS
#
# --plugin_config {user,both} faithfully reproduces the Hermes plugin's write
# path: per-exchange sync_turn, fixed per-role importances, entity extraction
# only. But its read path is a hybrid: raw ``memory.recall()`` plus only the
# plugin's temporal weighting (temporal_weight=0.2, halflife=48,
# __init__.py:1170), and none of the rest of ``prefetch()``. The unit under
# test is what the plugin hands the agent (CLAUDE.md Goal), so that gap is a
# real fidelity hole. This overlay closes the gap as a separate arm, so the
# difference is a measurable delta between overlay and no-overlay, with
# everything else byte-identical, rather than an unverifiable claim.
#
# GROUND TRUTH: MnemosyneMemoryProvider.prefetch(), integrations/hermes/src/
# mnemosyne_hermes/__init__.py:1158-1241. In source order:
#
#   1. RECALL (:1168-1182). ``self._beam.recall(query=..., top_k=max(
#      _PREFETCH_TOP_K * 2, 16) -> 16, temporal_weight=0.2,
#      temporal_halflife=48)``. The plugin adds ``author_id`` only when
#      ``beam.author_id or $MNEMOSYNE_AUTHOR_ID`` is non-empty (:1180-1181),
#      because passing a real one would trip beam.recall's (1=1) clause and
#      skip session and channel filtering. Note the over-fetch: 16
#      candidates, not the 13 (top_k+8) the non-overlay path fetches. The
#      plugin passes no vec, fts, or importance weights, so they resolve to
#      Mnemosyne's 0.5/0.3/0.2 through _normalize_weights.
#   2. CANONICAL ROWS (:1184-1193). ``_canonical_prefetch_rows(store,
#      self._canonical_owner(), query)`` with the default limit of 3
#      (:299). The store is ``beam.canonical`` (constructed on demand).
#      ``_canonical_owner()`` (:2046) is the Hermes profile id, defaulting
#      to "default", and initialize() stamps it onto
#      ``beam.canonical_owner_id`` (:1111), which is also beam's own
#      default (beam.py:2943). So the adapter, reading
#      ``beam.canonical_owner_id``, sees the same owner. Selection is
#      purely lexical over the fact body, deliberately excluding
#      category/name labels (:318-322): query tokens intersected with body
#      tokens must contain at least one "distinctive" token, meaning not in
#      the generic {user, owner, assistant, agent, system, profile,
#      identity, default} set, and then either 2 or more distinctive
#      tokens, or 0.30 or more coverage (:326-334). The score is synthetic:
#      ``min(1.0, 0.72 + coverage*0.24 + min(overlap,3)*0.03)`` (:335). The
#      row carries importance 0.95, keyword_score max(0.35, coverage),
#      fact_match=True, trust_tier CANONICAL, source "canonical:<category>",
#      and the plugin sorts the candidates by (score, keyword_score) and
#      cuts them to 3.
#   3. EARLY EXIT (:1195-1196). No recall results and no canonical rows
#      returns "".
#   4. QUALITY / TOPIC FILTER over the recall rows only (:1201-1215), in
#      this exact order per row:
#        a. ``_is_low_quality_prefetch(content)`` (:265) drops content that
#           is empty, or a single "word" that is 8 chars or fewer or is a
#           stopword.
#        b. ``_prefetch_source_quality(r) <= 0`` (:364) drops the row.
#           Quality starts at 1.0 and every other multiplier is strictly
#           positive (1.12 distilled, 0.72 conversation, 0.68 [USER], 0.80
#           [IDENTITY], 0.90 memoria_source), so the only way to reach 0.0
#           is ``content.upper().startswith("[ASSISTANT]")`` (:221, :368).
#           This is the reported "[ASSISTANT] rows are excluded" rule, and
#           it is a hard exclusion, not a demotion.
#        c. Topical signal (:1207-1212): ``signal = max(keyword_score,
#           fts_score, dense_score)``, floored at 0.20 when fact_match or
#           entity_match is set (:353-361). The row must clear 0.18 if it
#           is "raw" (source == "conversation" or content starts with
#           [USER] or [IDENTITY], :384), else 0.08. Every row the plugin
#           write path stores is source="conversation", so in this arm the
#           0.18 gate applies to all recall rows.
#        d. ``score < 0.20 and importance < 0.65`` drops the row (:1213-1214).
#      Canonical rows bypass all of (a) through (d). The plugin appends
#      them after the loop (:1217-1218).
#   5. ORDER + DEDUP (:1219-1220). A single stable descending sort of the
#      filtered recall rows plus canonical rows by ``_prefetch_adjusted_score``,
#      which equals ``(score*0.65 + signal*0.35 + importance*0.05) *
#      source_quality`` (:390). Then ``_semantic_dedup_prefetch(threshold=0.72)``
#      (:397) and ``[:_PREFETCH_TOP_K]`` (5, :213).
#      DEDUP DETAIL, the reported "0.72 threshold": similarity is lexical,
#      not embedding, over ``_prefetch_tokens(content)`` (:285). This is the
#      content with any [USER]/[ASSISTANT]/[IDENTITY] prefix stripped,
#      lowercased, regex-tokenized, sentence punctuation trimmed, and
#      tokens of 2 chars or fewer and dedup-stopwords removed. A later row
#      is dropped when either ``jaccard >= 0.72`` or ``containment >= 0.86``
#      against any already-kept row (:409-413). So the containment rule,
#      not the 0.72 threshold, is what collapses a short row into a longer
#      superset. The row that wins a collision is the one that appears
#      earlier in the sorted list, that is the one with the higher adjusted
#      score. A row whose token set is empty is dropped outright (:401-403).
#   6. ASSEMBLY (:1221-1238). If nothing survives, the result is "" and the
#      agent gets no memory block at all, because the plugin does not
#      backfill. Otherwise, up to 5 lines appear under a "## Mnemosyne
#      Context" heading. Per item, the content is
#      ``_format_prefetch_content(content, _prefetch_content_char_limit())``,
#      then whitespace-collapsed with ``" ".join(content.split())``. The
#      char limit defaults to 0, meaning no truncation (:177-193, env
#      MNEMOSYNE_PREFETCH_CONTENT_CHARS). This is confirmed: the docstring
#      says the old hardcoded 200-char cap "often removed the actual fact".
#      Each line is decorated
#      ``[ts[:16]] (importance X.XX[, source S])[ [TRUST]] content``.
#
# prefetch() vs the mnemosyne_recall TOOL: prefetch() is the MemoryProvider
# hook Hermes calls automatically every turn and injects as "## Mnemosyne
# Context". The tool (``_handle_recall``, :1626) is agent-initiated, and the
# system prompt tells the agent to read the injected block first and reach
# for retrieval tools only when it is "missing, stale, or insufficient"
# (:1137-1140). The tool is a genuinely different pipeline: caller-supplied
# top_k (default 5) and ``temporal_weight`` defaulting to 0.0 (:1629), no
# quality or topic filter at all, canonical merged with limit=max(2,
# min(top_k,5)), sorted by raw ``score`` rather than the adjusted score
# (:1674), then the same ``_semantic_dedup_prefetch`` and a [:top_k] cut. So
# the tool is near-raw top-5 plus canonical merge plus dedup at temporal
# 0.0. MemConflict has no agent loop that could decide to call a tool, so
# the automatic surface (prefetch) is the correct one to reproduce.
#
# HOW THIS IS IMPLEMENTED: by importing and calling the plugin's own
# module-level helpers out of the pinned submodule, never by
# re-implementing them here. So every threshold above (0.18/0.08,
# 0.20/0.65, 0.72/0.86, the 0.65/0.35/0.05 blend, the quality multipliers)
# comes from the code under test and cannot drift from it. mnemosyne_hermes
# imports cleanly standalone. Every hermes_cli, batch_tool, and persona
# import at module scope is wrapped in try/except with a fallback
# (__init__.py:34-45, :67-131), so no Hermes install is required.
#
# DELIBERATE DEVIATIONS, all stated rather than approximated silently:
#
#   (i)   The code passes vec, fts, and importance weights explicitly
#         (0.5/0.3/0.2), which prefetch() does not do. This has no effect
#         on the standard recall path, because _normalize_weights maps
#         None to exactly these values, and it is required for
#         MNEMOSYNE_ENHANCED_RECALL=1. See Search_Mnemosyne_For_Question's
#         docstring for the upstream crash this avoids. Keeping it
#         identical to the non-overlay arm also makes the A/B delta
#         attributable to the overlay alone.
#   (ii)  The code drops rows with ``source == "sleep_model_refresh_proposal"``
#         from the recall candidates before the plugin filter runs.
#         prefetch() does not do this, because those rows are not in
#         _PREFETCH_DISTILLED_SOURCES and carry no excluded prefix, so the
#         real plugin would surface them. This code drops them because the
#         non-overlay plugin arm already drops them in
#         Search_Mnemosyne_For_Question, and removing the filter here would
#         put a second difference into the A/B. They only ever exist under
#         --plugin_auto_sleep, and the code counts the drop
#         (Total_Prefetch_Model_Refresh_Dropped), so it is never invisible.
#   (iii) The code does not reproduce the plugin's per-line decoration
#         ``(importance X, source S) [TRUST]`` or its ``ts[:16]`` display
#         truncation. The prompt format is the shared harness contract:
#         eval_common.Build_Retrieved_Memory_Context renders "N.
#         [created_at] memory" for every provider, and re-shaping it per
#         provider is exactly the kind of harness change CLAUDE.md
#         forbids. What is under test, which memories and what text per
#         memory, is reproduced exactly. ``created_at`` keeps the full
#         timestamp, the same instant with more precision, so it stays
#         comparable with every other arm.
#   (iv)  ``self._should_filter``, skip-contexts, and gateway session
#         scoping are Hermes-runtime concerns with no analogue in a
#         benchmark persona, and the adapter cannot reach them.
#
# HARNESS CONTRACT: this returns at most 5 items and does not backfill when
# the overlay yields fewer, because the plugin does not backfill either.
# :1220-1222 simply emits what survived, or nothing at all. The code counts
# every shortfall.


# Where the plugin package lives inside the repo, and inside the container:
# the Dockerfile copies the whole external/mnemosyne submodule to
# /app/external/mnemosyne, and CURRENT_DIR is /app/mnemosyne, so this
# resolves either way. It is not pip-installed (pip install -e
# external/mnemosyne installs the ``mnemosyne`` core package only), hence
# the explicit sys.path insert.
_PLUGIN_HERMES_SRC_DIR = os.environ.get(
    "MNEMOSYNE_HERMES_SRC_DIR",
    os.path.join(CURRENT_DIR, "..", "external", "mnemosyne",
                 "integrations", "hermes", "src"),
)

# Every plugin symbol the overlay calls. The code checks these up front, so
# a submodule bump that renames one fails the run at startup instead of
# silently degrading the overlay into "recall with extra steps".
_PLUGIN_PREFETCH_SYMBOLS = (
    "_PREFETCH_TOP_K",
    "_is_low_quality_prefetch",
    "_prefetch_source_quality",
    "_prefetch_topic_signal",
    "_prefetch_is_raw",
    "_prefetch_adjusted_score",
    "_semantic_dedup_prefetch",
    "_canonical_prefetch_rows",
    "_prefetch_content_char_limit",
    "_format_prefetch_content",
)

_plugin_prefetch_module = None


def Load_Plugin_Prefetch_Module():
    """Import the pinned Hermes plugin module and hand back its prefetch helpers.

    This is read-only use of the submodule, because CLAUDE.md forbids
    modifying external/. It raises on any failure. The caller
    (MnemosyneBinding.__init__) does this eagerly, so a broken import kills
    the run before persona 0 rather than turning the arm into a silent
    no-op.
    """
    global _plugin_prefetch_module
    if _plugin_prefetch_module is not None:
        return _plugin_prefetch_module
    import importlib

    src_dir = os.path.abspath(_PLUGIN_HERMES_SRC_DIR)
    if not os.path.isdir(src_dir):
        raise RuntimeError(
            f"--plugin_prefetch_overlay: Hermes plugin source not found at {src_dir} "
            "(set MNEMOSYNE_HERMES_SRC_DIR, or rebuild the image so "
            "external/mnemosyne is present)"
        )
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    module = importlib.import_module("mnemosyne_hermes")
    missing = [n for n in _PLUGIN_PREFETCH_SYMBOLS if not hasattr(module, n)]
    if missing:
        raise RuntimeError(
            "--plugin_prefetch_overlay: mnemosyne_hermes at "
            f"{getattr(module, '__file__', '?')} is missing prefetch helpers "
            f"{missing} — the pinned plugin changed; re-derive the overlay "
            "against the new source instead of running a half-faithful arm"
        )
    _plugin_prefetch_module = module
    return module


def New_Prefetch_Stats() -> Dict[str, Any]:
    """Per-persona overlay counters.

    These counters are deliberately fine-grained. An overlay that silently
    no-ops, a failure mode that has hit several arms in this project, looks
    the same as a working one unless every stage reports how many rows it
    moved. Read an all-zero block here, in particular questions == 0,
    candidates == 0, or all filter_* zero on the `both` config, as a broken
    run, not a clean one.
    """
    return {
        "questions": 0,             # questions the overlay actually handled
        "candidates": 0,            # rows returned by beam.recall (cap 16/question)
        "mr_dropped": 0,            # deviation (ii): sleep bookkeeping rows
        "filter_low_quality": 0,    # _is_low_quality_prefetch
        "filter_assistant": 0,      # source_quality <= 0 == the [ASSISTANT] exclusion
        "filter_signal": 0,         # topical signal below 0.18 (raw) / 0.08
        "filter_score_importance": 0,   # score < 0.20 and importance < 0.65
        "kept_after_filter": 0,
        "canonical_considered": 0,  # questions where a canonical store was readable
        "canonical_merged": 0,      # canonical rows appended (<=3/question)
        "dedup_dropped": 0,         # rows collapsed by _semantic_dedup_prefetch
        "over_top_k_dropped": 0,    # survivors beyond the plugin's [:5]
        "truncated": 0,             # items shortened by the content char limit
        "final_items": 0,           # total items handed to the answer model
        "short_questions": 0,       # questions that ended with <5 items
        "shortfall_items": 0,       # sum(5 - final_count) over those questions
        "empty_questions": 0,       # questions where the overlay returned NOTHING
        "final_count_hist": {str(i): 0 for i in range(6)},
        "canonical_rows_in_final": 0,
        "content_char_limit": None,  # resolved MNEMOSYNE_PREFETCH_CONTENT_CHARS
        "author_id_scoped": False,   # whether recall was author_id-scoped (:1180)
    }


def Search_Mnemosyne_Plugin_Prefetch(
    memory, question_text: str,
    temporal_weight: float = 0.2, temporal_halflife: Optional[float] = 48.0,
    stats: Optional[Dict[str, Any]] = None,
    query_time: Optional[Any] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Retrieve one question's memories through the plugin's full prefetch()
    overlay, using the plugin's own helper functions.

    This mirrors __init__.py:1158-1238 step for step. See the long block
    above for the cited ground truth and the four declared deviations. It
    returns the same (retrieved, duration_ms) pair every other recall path
    returns, with at most ``_PREFETCH_TOP_K`` (5) items and no backfill.

    ONE TEXT, TWO CONSUMERS: the ``memory`` string built at the bottom of
    this function is the only rendering of a retrieved item. eval_common
    feeds the same list to Build_Retrieved_Memory_Context (the answer
    model) and stores it verbatim as ``Retrieved_Memories`` (the SEH and
    log-rank judge). The judge must never see a richer text than the
    answerer saw. That would inflate SEH against a context the agent never
    had, and would misattribute the resulting misses to reasoning
    (EUG-cond@5) rather than to retrieval.
    """
    mh = Load_Plugin_Prefetch_Module()
    if stats is None:
        stats = New_Prefetch_Stats()
    start = time.time()

    # 1. Recall (__init__.py:1168-1182). top_k is the plugin's own
    # over-fetch width (16), not the harness top_k, because the number of
    # candidates the filter sees is part of the behavior under test. The
    # final cut back to 5 happens at step 5, exactly as in :1220.
    prefetch_top_k = int(getattr(mh, "_PREFETCH_TOP_K", 5))
    recall_kwargs: Dict[str, Any] = dict(
        top_k=max(prefetch_top_k * 2, 16),
        temporal_weight=temporal_weight,
        temporal_halflife=temporal_halflife,
        # Deviation (i), see the block above.
        vec_weight=float(os.environ.get("MNEMOSYNE_VEC_WEIGHT", "0.5")),
        fts_weight=float(os.environ.get("MNEMOSYNE_FTS_WEIGHT", "0.3")),
        importance_weight=float(os.environ.get("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2")),
    )
    # Anchors the temporal boost to the logical session clock when the
    # caller supplied it (backdated arms). None reproduces the plugin's
    # wall-clock now.
    if query_time is not None:
        recall_kwargs["query_time"] = query_time
    beam = getattr(memory, "beam", None)
    author_id = (getattr(beam, "author_id", None)
                 or os.environ.get("MNEMOSYNE_AUTHOR_ID"))
    if author_id:
        # :1180-1181. The plugin scopes by author_id only when it is
        # explicitly non-empty. Nothing sets it in this benchmark, so the
        # flag stays False and recall keeps its normal scoping.
        recall_kwargs["author_id"] = author_id
        stats["author_id_scoped"] = True
    results = memory.recall(question_text, **recall_kwargs) or []
    # Held for the diagnostic capture before the mr-proposal filter rebinds
    # ``results``. The overlay makes two provider calls per question, this
    # recall and the canonical lookup below, so the raw capture records
    # both halves.
    raw_recall_rows = list(results)
    stats["questions"] += 1
    stats["candidates"] += len(results)

    # Deviation (ii): drops sleep's "[MODEL_REFRESH_PROPOSAL]" bookkeeping rows.
    if results:
        kept_rows = [r for r in results
                     if isinstance(r, dict)
                     and r.get("source") != "sleep_model_refresh_proposal"]
        stats["mr_dropped"] += len(results) - len(kept_rows)
        results = kept_rows

    # 2. Canonical rows (__init__.py:1184-1193).
    canonical_rows: List[Dict[str, Any]] = []
    try:
        store = getattr(beam, "canonical", None)
        if store is None:
            from mnemosyne.core.canonical import CanonicalStore
            store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
            beam.canonical = store
        # Owner: the plugin stamps _canonical_owner() onto
        # beam.canonical_owner_id (:1111), and beam itself defaults it to
        # "default" (beam.py:2943), so reading it here gives the same owner
        # the plugin would pass.
        owner_id = str(getattr(beam, "canonical_owner_id", "") or "").strip() or "default"
        canonical_rows = mh._canonical_prefetch_rows(store, owner_id, question_text) or []
        stats["canonical_considered"] += 1
    except Exception as exc:  # matches the plugin's bare except (:1192-1193)
        print(f"[DEBUG] prefetch overlay: canonical lookup failed: {exc}")
        canonical_rows = []
    stats["canonical_merged"] += len(canonical_rows)

    # Diagnostic capture, raw half. This question's provider answer is both
    # calls the overlay makes, so the code records both: the recall
    # candidate rows (the plugin's own 16-wide over-fetch, unchanged) and
    # the canonical prefetch rows (the plugin's own limit=3, unchanged).
    # Ranked starts empty, so the early exit below stores an honest
    # zero-depth row. The code re-records it once the overlay's own ranking
    # is known.
    raw_prefetch = {"recall": raw_recall_rows, "canonical_prefetch_rows": canonical_rows}
    record_provider_retrieval(diag, raw=raw_prefetch, ranked=[])

    # 3. Early exit (__init__.py:1195-1196).
    if not results and not canonical_rows:
        stats["empty_questions"] += 1
        stats["short_questions"] += 1
        stats["shortfall_items"] += prefetch_top_k
        stats["final_count_hist"]["0"] += 1
        return [], (time.time() - start) * 1000.0

    # 4. Quality / topic filter (__init__.py:1201-1215). Order matters and
    # is reproduced exactly. The code counts each rejection reason
    # separately, so the arm can be shown to have done something.
    filtered: List[Dict[str, Any]] = []
    for r in results:
        if mh._is_low_quality_prefetch(r.get("content", "")):
            stats["filter_low_quality"] += 1
            continue
        if mh._prefetch_source_quality(r) <= 0:
            # The only path to quality 0.0 is the [ASSISTANT] prefix
            # (:221, :368), see the ground-truth block, item 4b.
            stats["filter_assistant"] += 1
            continue
        signal = mh._prefetch_topic_signal(r)
        score = float(r.get("score") or 0.0)
        importance = float(r.get("importance") or 0.0)
        required_signal = 0.18 if mh._prefetch_is_raw(r) else 0.08
        if signal < required_signal:
            stats["filter_signal"] += 1
            continue
        if score < 0.20 and importance < 0.65:
            stats["filter_score_importance"] += 1
            continue
        filtered.append(r)
    stats["kept_after_filter"] += len(filtered)

    # Canonical rows bypass the filter entirely. The plugin appends them
    # after it (:1217-1218), so they only compete on the adjusted score in
    # step 5.
    if canonical_rows:
        filtered.extend(canonical_rows)

    # 5. Order, semantic dedup, and top-5 (__init__.py:1219-1220). A single
    # stable descending sort by the plugin's adjusted score. The dedup then
    # walks that order, so the higher-scoring row always wins a collision.
    filtered.sort(key=mh._prefetch_adjusted_score, reverse=True)
    pre_dedup = len(filtered)
    deduped = mh._semantic_dedup_prefetch(filtered)
    stats["dedup_dropped"] += pre_dedup - len(deduped)
    # Diagnostic capture, ranked half: the overlay's full ranked candidate
    # list, ordered by the plugin's adjusted score, before the plugin's own
    # top-5 cut on the next line. Text is the pre-format ``content``. The
    # answer-context rendering, char-limit plus whitespace collapse,
    # happens in step 6 and stays exclusive to Retrieved_Memories.
    record_provider_retrieval(diag, raw=raw_prefetch, ranked=[
        {
            "memory": str(r.get("content", "")),
            "created_at": r.get("timestamp", "Unknown Time"),
            "score": r.get("score"),
        }
        for r in deduped if isinstance(r, dict)
    ])
    final_rows = deduped[:prefetch_top_k]
    stats["over_top_k_dropped"] += max(0, len(deduped) - len(final_rows))

    # 6. Assembly (__init__.py:1221-1238).
    if not final_rows:
        stats["empty_questions"] += 1
        stats["short_questions"] += 1
        stats["shortfall_items"] += prefetch_top_k
        stats["final_count_hist"]["0"] += 1
        return [], (time.time() - start) * 1000.0

    content_limit = mh._prefetch_content_char_limit()
    stats["content_char_limit"] = content_limit
    retrieved: List[Dict[str, Any]] = []
    for r in final_rows:
        raw_content = r.get("content", "")
        content = mh._format_prefetch_content(raw_content, content_limit)
        if content != raw_content:
            stats["truncated"] += 1
        # Whitespace collapse is part of what the agent sees (:1230), so it
        # is part of the single text both consumers get.
        content = " ".join(content.split())
        score = r.get("score")
        is_canonical = str(r.get("source") or "").startswith("canonical:")
        if is_canonical:
            stats["canonical_rows_in_final"] += 1
        retrieved.append({
            "memory": content,
            # Full timestamp, not the plugin's display-only ts[:16],
            # deviation (iii). Same instant, identical to every other
            # arm's field.
            "created_at": r.get("timestamp", "Unknown Time"),
            "score": round(float(score), 6) if isinstance(score, (int, float)) else score,
            "dense_score": r.get("dense_score"),
            "fts_score": r.get("fts_score"),
            "source": r.get("source"),
            "tier": r.get("tier"),
            "fact_match": r.get("fact_match"),
            "entity_match": r.get("entity_match"),
            # Overlay-specific provenance, so a stored row can be re-derived:
            # which ranking actually ordered this list, and whether the row
            # came from the canonical store rather than from recall.
            "prefetch_adjusted_score": round(float(mh._prefetch_adjusted_score(r)), 6),
            "prefetch_topic_signal": round(float(mh._prefetch_topic_signal(r)), 6),
            "prefetch_source_quality": round(float(mh._prefetch_source_quality(r)), 6),
            "prefetch_canonical": is_canonical,
        })

    n = len(retrieved)
    stats["final_items"] += n
    stats["final_count_hist"][str(min(n, 5))] += 1
    if n < prefetch_top_k:
        # No backfill. The plugin emits only what survived (:1220-1238).
        # The code records the shortfall as a property of the overlay,
        # rather than repairing it.
        stats["short_questions"] += 1
        stats["shortfall_items"] += prefetch_top_k - n
    return retrieved, (time.time() - start) * 1000.0


# --------------------------------------------------------------------------
# Canonical-history retrieval (opt-in via --canonical / --oracle)
# --------------------------------------------------------------------------
# MemConflict's historical questions ("did X change?", "where did they live
# before?") lose their evidence when retirement invalidates the stale turn.
# Mnemosyne's own answer to that is the canonical layer: each (category,
# name) slot keeps every superseded version. These helpers route historical
# questions to that history with no gold leakage. Slot selection uses only
# the question text (CanonicalStore.search is substring-based, canonical.py:353).

_HISTORICAL_QUESTION_MARKERS = (
    "used to", "before", "previous", "previously", "originally", "at first",
    "initially", "ever", "change", "changed", "changing", "switch", "moved",
    "no longer", "not anymore", "anymore", "any more", "still", "history",
    "over time", "earlier", "back then", "in the past", "first time",
    "all the", "each time", "every time", "again", "used them", "old",
)

_CANONICAL_STOPWORDS = frozenset(
    "the a an is are was were be been being what where when who whom how why "
    "which did does do done has have had user assistant they their them this "
    "that these those and or but for with from about into onto over under "
    "you your yours his her hers its our ours will would can could should "
    "may might must not now then than there here also just very really "
    "please tell say said asked mention mentioned currently current".split()
)


def Detect_Historical_Question(question_text: str) -> bool:
    q = " " + " ".join(str(question_text).lower().split()) + " "
    return any(marker in q for marker in _HISTORICAL_QUESTION_MARKERS)


def _question_content_tokens(question_text: str) -> List[str]:
    tokens = []
    for raw in str(question_text).lower().split():
        word = "".join(ch for ch in raw if ch.isalnum())
        if len(word) >= 3 and word not in _CANONICAL_STOPWORDS:
            tokens.append(word)
    return tokens


def _format_slot_line(row: Dict[str, Any]) -> str:
    body = str(row.get("body", "")).strip().replace("\n", " ")
    return f"{row.get('category')}/{row.get('name')}: {body[:300]}"


def Build_Canonical_Context(
    memory, question_text: str, max_slots: int = 8, max_versions: int = 6
) -> Tuple[str, Dict[str, Any]]:
    """Question-driven canonical slot lookup with history expansion.

    Selection uses only the question text, with no gold. A slot is relevant
    when a content word of the question appears in its category, name, or
    body. Historical questions also get each relevant slot's full version
    chain, oldest first, so the answer LLM sees the evolution in order. When
    a historical question matches no slot by substring, the code falls back
    to all current slots' histories. The per-owner canonical set is a small
    identity card, and substring matching legitimately misses cases like
    "lived" against a "residence" slot.

    Returns ("", diag) when there is nothing to show. The answer context is
    then byte-identical to the non-canonical path.
    """
    diag = {"slots_considered": 0, "slots_shown": 0, "historical": False,
            "history_versions_shown": 0}
    store = getattr(memory.beam, "canonical", None)
    if store is None:
        return "", diag
    owner_id = str(getattr(memory.beam, "canonical_owner_id", "") or "").strip() or "default"
    try:
        slots = store.list(owner_id) or []
    except Exception:
        return "", diag
    # Keeps only current rows, one per (category, name).
    current: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in slots:
        if not isinstance(row, dict) or row.get("valid_until"):
            continue
        current[(str(row.get("category")), str(row.get("name")))] = row
    diag["slots_considered"] = len(current)
    if not current:
        return "", diag

    historical = Detect_Historical_Question(question_text)
    diag["historical"] = historical
    tokens = _question_content_tokens(question_text)
    relevant = []
    for key, row in sorted(current.items()):
        haystack = f"{key[0]} {key[1]} {row.get('body', '')}".lower()
        if any(tok in haystack for tok in tokens):
            relevant.append(row)
    if not relevant and historical:
        relevant = [current[k] for k in sorted(current.keys())]
    relevant = relevant[:max_slots]
    if not relevant:
        return "", diag
    diag["slots_shown"] = len(relevant)

    lines = ["Canonical profile facts (consolidated current state):"]
    for row in relevant:
        lines.append(f"- {_format_slot_line(row)}")
    if historical:
        for row in relevant:
            try:
                versions = store.history(
                    owner_id, str(row.get("category")), str(row.get("name"))) or []
            except Exception:
                continue
            if len(versions) < 2:
                continue  # No superseded history. The current line is already shown.
            versions = list(reversed(versions))[-max_versions:]  # Oldest first.
            lines.append(
                f"History of {row.get('category')}/{row.get('name')} (oldest first):")
            for v in versions:
                until = v.get("valid_until")
                span = f"{str(v.get('valid_from', ''))[:10]} → " + (
                    str(until)[:10] if until else "current")
                body = str(v.get("body", "")).strip().replace("\n", " ")
                lines.append(f"  v{v.get('version')} [{span}]: {body[:300]}")
                diag["history_versions_shown"] += 1
    return "\n".join(lines), diag


def Run_Session_Sleep(memory, persona_tag: str, session_tag: Any,
                      sleep_totals: Dict[str, int]) -> None:
    """Run the real production consolidation (``sleep(force=True)``) once per
    session, to populate the canonical layer through the model-refresh path.

    Run 2's surgical lifecycle deliberately skipped this. Here the code
    neutralizes its two pollution vectors instead of avoiding them:
      * episodic summaries: kept out of recall by ``MNEMOSYNE_EP_LIMIT=0``;
      * "[MODEL_REFRESH_PROPOSAL]" working_memory rows: filtered from the
        answer context by source in ``Search_Mnemosyne_For_Question``, and
        stamped with ``consolidated_at`` below so the next session's sleep
        does not recursively re-summarize consolidation plumbing.
    Any failure logs and returns.
    """
    try:
        result = memory.sleep(force=True) or {}
    except Exception as exc:
        print(f"[DEBUG] canonical persona {persona_tag} session {session_tag}: "
              f"sleep(force=True) failed: {exc}")
        return
    try:
        sleep_totals["sleep_conflicts_resolved"] = (
            sleep_totals.get("sleep_conflicts_resolved", 0)
            + int(result.get("conflicts_resolved", 0) or 0))
        sleep_totals["summaries_created"] += int(result.get("summaries_created", 0) or 0)
        refresh = result.get("model_refresh") or {}
        sleep_totals["mr_proposals"] = (
            sleep_totals.get("mr_proposals", 0) + int(refresh.get("proposals", 0) or 0))
        sleep_totals["model_refresh_applied"] += int(refresh.get("applied", 0) or 0)
    except Exception as exc:
        print(f"[DEBUG] canonical persona {persona_tag} session {session_tag}: "
              f"sleep totals accounting failed: {exc}")
    # Keeps sleep's own bookkeeping rows out of the next sleep batch.
    try:
        memory.beam.conn.execute(
            "UPDATE working_memory SET consolidated_at = ? "
            "WHERE source = 'sleep_model_refresh_proposal' AND consolidated_at IS NULL",
            (datetime.now().isoformat(),),
        )
        memory.beam.conn.commit()
    except Exception as exc:
        print(f"[DEBUG] canonical persona {persona_tag} session {session_tag}: "
              f"proposal-row consolidated_at stamp failed: {exc}")


# --------------------------------------------------------------------------
# Production lifecycle / stale-fact retirement (opt-in via --lifecycle)
# --------------------------------------------------------------------------
def Parse_Session_Base_Datetime(
    session_date: Any, last_ts: Optional[datetime]
) -> Optional[datetime]:
    """Compute a per-session base datetime for timestamp restoration.

    This anchors to the dataset's real ``Date`` (``YYYY-MM-DD`` at midnight)
    whenever it advances normally, the common case needing no adjustment. It
    enforces a minimal gap of one hour and one minute measured from
    ``last_ts`` (the previous message's timestamp) when the real date would
    collide with or fall behind the prior session. The tightest
    cross-session pair is the previous session's last message to the next
    session's first message. Keeping that pair more than one hour apart
    satisfies ``sleep()``'s conflict-detection gate, which needs a gap over
    one hour, without any multi-day drift.

    Returns None if the Date cannot be parsed and there is no prior clock
    to extend from. The caller then skips backdating for that session.
    """
    session_base: Optional[datetime] = None
    try:
        session_base = datetime.strptime(str(session_date).strip(), "%Y-%m-%d")
    except Exception:
        session_base = None
    if session_base is None:
        # Falls back to a synthetic monotonic base, so ordering and the gap still hold.
        if last_ts is None:
            return None
        return last_ts + timedelta(hours=1, minutes=1)
    if last_ts is not None:
        session_base = max(session_base, last_ts + timedelta(hours=1, minutes=1))
    return session_base


def Run_Session_Lifecycle(memory, persona_tag: str, session_tag: Any,
                          sleep_totals: Dict[str, int]) -> None:
    """Surgically retire stale conflicting facts. Non-fatal, no sleep().

    This replaces the old ``sleep(force=True)`` lifecycle, which polluted
    recall with lossy episodic summaries and model-refresh-proposal
    working_memory rows that break MemConflict's raw-turn SEH scoring.
    Instead, for each unresolved conflict (same subject and predicate,
    different object) the code picks a winner by source-row recency: the
    fact whose backdated source rows have the later max
    ``working_memory.timestamp`` is the current one. It resolves the
    conflict in the veracity consolidator, then invalidates the loser's
    source working_memory rows with ``memory.invalidate()``, so
    ``recall()`` (which returns raw working_memory turns) stops surfacing
    the stale turn and returns only the current one.

    Every step is guarded, so a failure never crashes the persona. There is
    no sleep() call, so there is no episodic or summary pollution.
    ``summaries_created`` and ``model_refresh_applied`` stay at 0.
    """
    try:
        vc = memory.beam.veracity_consolidator
    except Exception as exc:
        print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
              f"veracity_consolidator unavailable: {exc}")
        return
    if vc is None:
        return
    try:
        conn = memory.beam.conn
    except Exception as exc:
        print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
              f"beam.conn unavailable: {exc}")
        return

    # 1) Fetches unresolved conflicts. Prefers the SDK, falls back to raw SQL.
    conflicts: List[Dict[str, Any]] = []
    try:
        raw = vc.get_conflicts()
        for c in (raw or []):
            conflicts.append({
                "id": c["id"],
                "fact_a_id": c["fact_a_id"],
                "fact_b_id": c["fact_b_id"],
            })
    except Exception as exc:
        print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
              f"get_conflicts() failed ({exc}); falling back to raw SQL")
        conflicts = []
        try:
            rows = conn.execute(
                "SELECT id, fact_a_id, fact_b_id FROM conflicts WHERE resolution IS NULL"
            ).fetchall()
            for row in rows:
                conflicts.append({
                    "id": row[0], "fact_a_id": row[1], "fact_b_id": row[2],
                })
        except Exception as exc2:
            print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
                  f"conflicts fallback SQL failed: {exc2}")
            return

    if "rows_invalidated" not in sleep_totals:
        sleep_totals["rows_invalidated"] = 0

    def _load_fact(fact_id: Any) -> Optional[Dict[str, Any]]:
        try:
            row = conn.execute(
                "SELECT id, sources_json, confidence, updated_at "
                "FROM consolidated_facts WHERE id = ?",
                (fact_id,),
            ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        try:
            sources = json.loads(row[1]) if row[1] else []
            if not isinstance(sources, list):
                sources = []
        except Exception:
            sources = []
        # latest_ts is the max timestamp across this fact's source working_memory rows.
        latest_ts = None
        if sources:
            try:
                placeholders = ",".join("?" for _ in sources)
                ts_row = conn.execute(
                    f"SELECT MAX(timestamp) FROM working_memory WHERE id IN ({placeholders})",
                    tuple(sources),
                ).fetchone()
                if ts_row is not None:
                    latest_ts = ts_row[0]
            except Exception:
                latest_ts = None
        try:
            confidence = float(row[2]) if row[2] is not None else 0.0
        except Exception:
            confidence = 0.0
        return {
            "id": row[0], "sources": sources,
            "latest_ts": latest_ts, "confidence": confidence,
        }

    for conflict in conflicts:
        try:
            fact_a = _load_fact(conflict.get("fact_a_id"))
            fact_b = _load_fact(conflict.get("fact_b_id"))
            if fact_a is None or fact_b is None:
                continue

            # Winner selection: the fact whose source rows have the later
            # max timestamp is the current one, because backdated ISO
            # timestamps sort lexically. On a tie or missing timestamps,
            # the higher confidence wins.
            a_ts, b_ts = fact_a["latest_ts"], fact_b["latest_ts"]
            if a_ts is not None and b_ts is not None and a_ts != b_ts:
                winner, loser = (fact_a, fact_b) if a_ts > b_ts else (fact_b, fact_a)
            elif a_ts is not None and b_ts is None:
                winner, loser = fact_a, fact_b
            elif b_ts is not None and a_ts is None:
                winner, loser = fact_b, fact_a
            else:
                winner, loser = (
                    (fact_a, fact_b)
                    if fact_a["confidence"] >= fact_b["confidence"]
                    else (fact_b, fact_a)
                )

            # Resolves the conflict in the veracity consolidator. The
            # winning_fact_id is the second argument, and this sets
            # superseded_by on the loser and marks it resolved.
            try:
                vc.resolve_conflict(conflict.get("id"), winner["id"])
                sleep_totals["conflicts_resolved"] += 1
            except Exception as exc:
                print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
                      f"resolve_conflict failed for {conflict.get('id')}: {exc}")

            # Invalidates the loser's source working_memory rows, so recall
            # stops returning the stale turn. The replacement is a winner
            # source row, or None. A row can be a source of both facts,
            # because consolidate_fact runs per remember() call, so one
            # turn can feed either side of a conflict. Invalidating those
            # rows would erase the winner's own evidence, and in the
            # replacement_id == src_id case it produced a self-referencing
            # superseded_by row, caught by DB inspection. The code skips
            # shared sources.
            winner_sources = set(winner["sources"])
            replacement_id = winner["sources"][0] if winner["sources"] else None
            for src_id in loser["sources"]:
                if src_id in winner_sources:
                    continue
                try:
                    memory.invalidate(src_id, replacement_id=replacement_id)
                    sleep_totals["rows_invalidated"] += 1
                except Exception as exc:
                    print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
                          f"invalidate({src_id}) failed: {exc}")
        except Exception as exc:
            print(f"[DEBUG] lifecycle persona {persona_tag} session {session_tag}: "
                  f"conflict {conflict.get('id')} handling failed: {exc}")


def Collect_Retirement_Diagnostics(memory, sleep_totals: Dict[str, int]) -> Dict[str, Any]:
    """Query the persona DB for fact-retirement lifecycle diagnostics.

    Every query sits in its own try/except, because tables and columns
    vary across Mnemosyne builds: ``canonical_facts`` may not exist, and
    migration adds the ``superseded_by`` and ``valid_until`` working_memory
    columns only once ``invalidate()`` runs. A missing table or column must
    never crash the run.
    """
    conn = memory.beam.conn
    diagnostics: Dict[str, Any] = {}

    # --- Veracity consolidator stats + raw conflict counts ---
    veracity: Dict[str, Any] = {}
    try:
        consolidator = memory.beam.veracity_consolidator
        if consolidator is not None:
            stats = consolidator.get_stats()
            veracity["active_facts"] = stats.get("active_facts")
            veracity["superseded_facts"] = stats.get("superseded_facts")
            veracity["unresolved_conflicts"] = stats.get("unresolved_conflicts")
    except Exception as exc:
        veracity["stats_error"] = str(exc)
    try:
        veracity["conflicts_total"] = conn.execute(
            "SELECT COUNT(*) FROM conflicts").fetchone()[0]
    except Exception as exc:
        veracity["conflicts_total_error"] = str(exc)
    try:
        veracity["conflicts_resolved"] = conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE resolution IS NOT NULL").fetchone()[0]
    except Exception as exc:
        veracity["conflicts_resolved_error"] = str(exc)
    diagnostics["veracity"] = veracity

    # --- consolidated_facts supersession ---
    consolidated: Dict[str, Any] = {}
    try:
        consolidated["superseded"] = conn.execute(
            "SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL").fetchone()[0]
    except Exception as exc:
        consolidated["superseded_error"] = str(exc)
    try:
        consolidated["total"] = conn.execute(
            "SELECT COUNT(*) FROM consolidated_facts").fetchone()[0]
    except Exception as exc:
        consolidated["total_error"] = str(exc)
    diagnostics["consolidated_facts"] = consolidated

    # --- canonical_facts current/retired (table may not exist) ---
    canonical: Dict[str, Any] = {}
    try:
        canonical["current"] = conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE valid_until IS NULL").fetchone()[0]
    except Exception as exc:
        canonical["current_error"] = str(exc)
    try:
        canonical["retired"] = conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE valid_until IS NOT NULL").fetchone()[0]
    except Exception as exc:
        canonical["retired_error"] = str(exc)
    # Slot inventory. This checks whether the model names slots consistently
    # across sessions ("residence" every time, not residence, location, and
    # home_city as separate names). A versions count above 1 means real
    # supersession history exists for the history-aware retrieval.
    try:
        rows = conn.execute(
            "SELECT category, name, COUNT(*) AS versions "
            "FROM canonical_facts GROUP BY category, name "
            "ORDER BY versions DESC, category, name LIMIT 50").fetchall()
        canonical["slots"] = [
            {"category": r[0], "name": r[1], "versions": r[2]} for r in rows]
    except Exception as exc:
        canonical["slots_error"] = str(exc)
    diagnostics["canonical_facts"] = canonical

    # Extraction visibility. Without this, an --extract arm whose internal
    # LLM silently failed would look identical to baseline.
    extraction: Dict[str, Any] = {}
    try:
        extraction["facts_total"] = conn.execute(
            "SELECT COUNT(*) FROM facts").fetchone()[0]
    except Exception as exc:
        extraction["facts_total_error"] = str(exc)
    try:
        extraction["annotations_total"] = conn.execute(
            "SELECT COUNT(*) FROM annotations").fetchone()[0]
    except Exception as exc:
        extraction["annotations_total_error"] = str(exc)
    diagnostics["extraction"] = extraction

    # --- working_memory invalidation (superseded_by/valid_until are migration-added) ---
    working: Dict[str, Any] = {}
    try:
        working["invalidated"] = conn.execute(
            "SELECT COUNT(*) FROM working_memory "
            "WHERE superseded_by IS NOT NULL OR valid_until IS NOT NULL").fetchone()[0]
    except Exception as exc:
        working["invalidated_error"] = str(exc)
    try:
        working["total"] = conn.execute(
            "SELECT COUNT(*) FROM working_memory").fetchone()[0]
    except Exception as exc:
        working["total_error"] = str(exc)
    diagnostics["working_memory"] = working

    diagnostics["sleep_totals"] = dict(sleep_totals)
    return diagnostics


# --------------------------------------------------------------------------
# Provider binding (the only Mnemosyne-specific surface the driver sees)
# --------------------------------------------------------------------------
class MnemosyneBinding(ProviderBinding):
    memory_system = "mnemosyne"
    store_id_key = "Mnemosyne_Session_ID"
    runtime_summary_key = "Mnemosyne_Runtime_Summary"
    stage_name = "mnemosyne_answer_generation"
    stage_note = "Mnemosyne retrieval and question answering"

    def __init__(self, importance: float, extract: bool, lifecycle: bool,
                 canonical: bool, oracle: bool,
                 use_dataset_time: bool = False, plugin_config: str = "off",
                 plugin_auto_sleep: bool = False,
                 plugin_session_sleep: bool = False,
                 plugin_prefetch_overlay: bool = False):
        self.importance = importance
        self.extract = extract
        self.lifecycle = lifecycle
        self.canonical = canonical
        self.oracle = oracle
        self.plugin_config = plugin_config  # "off" | "user" | "both"
        # Plugin auto-sleep arm: fires the plugin's real sleep cadence
        # during the plugin ingest path (every 10 exchanges plus at each
        # session boundary), drained to completion before questions. This
        # is only meaningful with a plugin arm active. The __main__ parser
        # enforces plugin_config != "off".
        self.plugin_auto_sleep = plugin_auto_sleep and plugin_config != "off"
        # Session-end forced consolidation: one sleep(force=True) after each
        # dataset session's ingest, before its questions. This composes with
        # the auto-sleep cadence above; see Run_Plugin_Session_Sleep for the
        # measured reason the plugin's own gate cannot fire at benchmark
        # cadence. It works on either ingest path, because the trim it
        # answers to lives in beam, not in the plugin. The __main__ parser
        # rejects the sleep-based arms, which run their own sleep.
        self.plugin_session_sleep = plugin_session_sleep
        # Plugin prefetch() retrieval overlay: replaces the hybrid
        # recall-plus-temporal read path with the plugin's real prefetch()
        # pipeline (quality/topic filter, [ASSISTANT] exclusion, canonical
        # merge, adjusted-score sort, semantic dedup, top-5, no backfill).
        # This is only meaningful on the plugin write path. The __main__
        # parser enforces plugin_config != "off".
        self.plugin_prefetch_overlay = plugin_prefetch_overlay and plugin_config != "off"
        if self.plugin_prefetch_overlay:
            # Fails fast at construction, before persona 0. If the pinned
            # plugin module cannot be imported, or lost a helper, this
            # raises here rather than letting the arm quietly fall through
            # to plain recall and report a "successful" run that measured
            # the other arm.
            mh = Load_Plugin_Prefetch_Module()
            if int(getattr(mh, "_PREFETCH_TOP_K", 5)) != MAX_STORED_RETRIEVED_MEMORIES:
                print(f"[DEBUG] WARNING: plugin _PREFETCH_TOP_K="
                      f"{getattr(mh, '_PREFETCH_TOP_K', None)} != harness "
                      f"MAX_STORED_RETRIEVED_MEMORIES={MAX_STORED_RETRIEVED_MEMORIES}; "
                      "the overlay follows the PLUGIN's value (it is the behaviour "
                      "under test) — stored/answered row counts will differ from "
                      "every other arm")
            print(f"[DEBUG] prefetch overlay: using plugin helpers from "
                  f"{getattr(mh, '__file__', '?')} (_PREFETCH_TOP_K="
                  f"{getattr(mh, '_PREFETCH_TOP_K', None)}, content_char_limit="
                  f"{mh._prefetch_content_char_limit()})")
        self.use_dataset_time = use_dataset_time
        # Dataset-chronology backdating is active for the lifecycle arms
        # (which build on it), for an explicit --use_dataset_time run, and
        # for the plugin-fidelity arm (which implies it). Retirement is a
        # separate step gated on self.lifecycle only, so --use_dataset_time
        # and --plugin_config backdate without any Run_Session_Lifecycle call.
        self.backdate = bool(lifecycle or use_dataset_time or plugin_config != "off")
        # Plugin temporal recency weighting. The plugin prefetch uses 0.2
        # and 48 (integrations/hermes/.../__init__.py:1170), and this is
        # env-overridable. Only the plugin recall path uses this. A value
        # of 0.0 everywhere else keeps recall unchanged.
        # This uses `or "default"`, not a get() default, because compose
        # passes these vars set but empty ("${VAR:-}"), and float('')
        # crashes. This is the same set-but-empty trap documented for
        # HINDSIGHT_API_* in CLAUDE.md.
        self.temporal_weight = float(os.environ.get("MNEMOSYNE_TEMPORAL_WEIGHT") or "0.2")
        self.temporal_halflife = float(os.environ.get("MNEMOSYNE_TEMPORAL_HALFLIFE") or "48")
        if canonical or oracle:
            # Activates the shared driver's extra-context path: canonical
            # profile facts, and history for historical questions, appended
            # after the raw retrieved memories.
            self.extra_answer_context = self._canonical_context_hook

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        persona_tag = persona_id[-8:]
        session_id = f"mnemo_{persona_tag}_{uuid.uuid4().hex[:8]}"
        tmp_dir = tempfile.mkdtemp(prefix="mnemo_eval_")
        db_path = os.path.join(tmp_dir, "persona.db")

        oracle_slots: Optional[Dict[int, List[Dict[str, str]]]] = None
        if self.oracle:
            try:
                from oracle_canonical import Build_Oracle_Slots_For_Persona
                oracle_slots = Build_Oracle_Slots_For_Persona(persona_item)
                n_slots = sum(len(v) for v in (oracle_slots or {}).values())
                print(f"[DEBUG] oracle: derived {n_slots} slot writes across "
                      f"{len(oracle_slots or {})} sessions")
            except Exception as exc:
                print(f"[DEBUG] oracle slot derivation failed: {exc} — "
                      f"persona runs without oracle slots")

        return {
            "store_id": session_id,
            "persona_tag": persona_tag,
            "tmp_dir": tmp_dir,
            "db_path": db_path,
            "memory": Setup_Mnemosyne(db_path, session_id),
            "oracle_slots": oracle_slots,
            # Canonical retrieval is inert for an oracle persona whose slot
            # derivation failed. This mirrors the pre-refactor behavior of
            # not activating the canonical path at all in that case.
            "canonical_active": self.canonical or bool(oracle_slots),
            "last_ts": None,
            # Recency reference ("now") for this session's questions, set
            # per session in ingest_session. This anchors recall's temporal
            # boost to the dataset clock the rows are backdated onto,
            # mirroring RetainDB's per-question ``question_date``. None
            # means the framework's wall-clock now.
            "question_query_time": None,
            "total_added": 0,
            "total_filtered": 0,
            # Persona-level sync_turn counter driving the auto-sleep
            # cadence. The plugin's _turn_count is per-provider-instance,
            # which equals per-persona conversation here, and is never
            # reset per session. So this spans all of the persona's
            # sessions, and every 10th user-led exchange trips sleep.
            "plugin_turn_count": 0,
            # Per-persona plugin-sync counters. This is None unless the
            # plugin arm is on, so a non-plugin run's runtime summary stays
            # byte-identical.
            "plugin_stats": (
                {"user_written": 0, "assistant_written": 0, "dropped": 0,
                 "user_truncated": 0, "assistant_truncated": 0, "filtered": 0}
                if self.plugin_config != "off" else None
            ),
            # Per-persona prefetch-overlay counters. This is None unless
            # the overlay arm is on, so every other arm's runtime summary
            # stays byte-identical.
            "prefetch_stats": (
                New_Prefetch_Stats() if self.plugin_prefetch_overlay else None
            ),
            "sleep_totals": {
                "conflicts_resolved": 0, "summaries_created": 0,
                "model_refresh_applied": 0, "rows_invalidated": 0,
                # Auto-sleep arm counters, 0 unless --plugin_auto_sleep is
                # on. invocations counts cadence and session-end ticks that
                # fired real work. skipped counts ticks gated out below
                # threshold or with no eligible rows. budget_skipped counts
                # ticks gated out by the per-session reflection budget
                # (plugin default 3 per session, see deviation note (e)).
                "auto_sleep_invocations": 0, "auto_sleep_skipped": 0,
                "auto_sleep_budget_skipped": 0,
                "mr_proposals": 0,
                # Session-end forced-sleep counters, 0 unless
                # --plugin_session_sleep is on. no_op counts passes that
                # found nothing left to consolidate. The wm_* keys hold the
                # last census, so after the final session they report how
                # many working rows survived the shipped TTL and how many
                # are still exposed to the next trim.
                "session_sleep_invocations": 0, "session_sleep_no_op": 0,
                "session_sleep_items_consolidated": 0,
                "session_sleep_summaries_created": 0,
                "session_sleep_mr_proposals": 0, "session_sleep_mr_applied": 0,
                "session_sleep_wm_total": 0, "session_sleep_wm_consolidated": 0,
                "session_sleep_wm_unconsolidated": 0,
            },
            "retirement_diagnostics": None,
        }

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        memory = ctx["memory"]
        session_base = None
        if self.backdate:
            session_base = Parse_Session_Base_Datetime(session_item.get("Date"), ctx["last_ts"])
        # Stashes this session's logical "now" for the questions answered
        # against it. The shared driver ingests session i then immediately
        # answers session i's questions
        # (eval_common.Generate_Single_Persona_Eval), so the reference time
        # a question is posed at is this session's date. This is the noon
        # recall-"now" anchor (eval_common.Parse_Query_Now_Timestamp), the
        # same instant clock_sync fakes the OS clock to, so every
        # provider's recall "now" agrees. It is deliberately not
        # session_base: session_base is the ingest timestamp base (Date at
        # midnight, bumped for monotonicity against the prior session, used
        # below and left unchanged), and noon sits after all of this
        # session's midnight-plus-per-turn-minute ingest rows, so
        # recall-"now" is never before the memories it ranks. The code sets
        # this only when backdating is active. Otherwise rows are
        # wall-clock and query_time must stay None, so the temporal
        # reference matches the stored timestamps.
        #
        # PLUGIN-FAITHFUL (2026-07-28): the Hermes provider never populates
        # `query_time`. Its automatic prefetch path omits it entirely
        # (external/mnemosyne/hermes_memory_provider/__init__.py:2181-2204),
        # and while the `mnemosyne_recall` tool schema exposes it (:543-547)
        # the integration only forwards whatever the answering model typed.
        # So under clock-sync this code omits it too, and lets
        # beam._parse_query_time(None) read the faked OS clock
        # (beam.py:1431-1432). clock_sync sets that clock to the same noon
        # instant Parse_Query_Now_Timestamp computes, because both import
        # RECALL_NOW_HOUR_UTC, so the two mechanisms agree to within the
        # real seconds elapsed since set_clock. This is the same ruling as
        # the RetainDB adapter: the benchmark puts the query system at the
        # question date rather than passing a parameter no deployment
        # sends. Non-clocksync backdated arms keep the parameter, because
        # there the faked clock does not exist and it is the only way to
        # give the temporal boost a sane reference.
        _clocksync_on = os.environ.get("BENCH_CLOCKSYNC") == "1"
        query_now = Parse_Query_Now_Timestamp(session_item) if self.backdate else None
        ctx["question_query_time"] = (
            query_now.isoformat()
            if (not _clocksync_on and self.backdate
                and session_base is not None and query_now is not None) else None
        )

        plugin_meta: Dict[str, Any] = {}
        if self.plugin_config != "off":
            # Cadence callback: the plugin increments _turn_count per
            # sync_turn and calls _maybe_auto_sleep() every 10th
            # (__init__.py:1318-1320). This code replicates that here, mid-
            # session, at the exact exchange the plugin would, using the
            # persona-level running counter in ctx.
            on_sync_turn = None
            # Fresh per-session reflection budget. The plugin initializes
            # this counter per provider instance, which equals per Hermes
            # session, see deviation note (e). The cadence closure and the
            # session_end call below share this budget.
            reflect_state = {"used": 0}
            if self.plugin_auto_sleep:
                def on_sync_turn() -> None:
                    ctx["plugin_turn_count"] += 1
                    if ctx["plugin_turn_count"] % PLUGIN_AUTO_SLEEP_EXCHANGE_CADENCE == 0:
                        Run_Plugin_Auto_Sleep(
                            memory, ctx["persona_tag"], "cadence",
                            ctx["plugin_turn_count"], ctx["sleep_totals"],
                            reflect_state=reflect_state,
                        )
            add_ms, added, filtered, session_last_ts, pstats = Add_Session_Dialogue_Plugin_Sync(
                memory, dialogue, self.plugin_config,
                backdate=self.backdate, session_base=session_base,
                on_sync_turn=on_sync_turn,
            )
            # Session-boundary consolidation: the plugin's on_session_end()
            # runs beam.sleep() once when a session closes (__init__.py:2540).
            # This code fires it here, drained, after this session's ingest
            # and before its questions are answered, using the same
            # eligibility gates as the cadence path.
            if self.plugin_auto_sleep:
                Run_Plugin_Auto_Sleep(
                    memory, ctx["persona_tag"], "session_end",
                    ctx["plugin_turn_count"], ctx["sleep_totals"],
                    reflect_state=reflect_state,
                )
            ps = ctx["plugin_stats"]
            for k, v in pstats.items():
                ps[k] += v
            plugin_meta = {
                "Plugin_Config": self.plugin_config,
                "Plugin_User_Written": pstats["user_written"],
                "Plugin_Assistant_Written": pstats["assistant_written"],
                "Plugin_Sync_Dropped_Messages": pstats["dropped"],
                "Plugin_User_Truncated": pstats["user_truncated"],
                "Plugin_Assistant_Truncated": pstats["assistant_truncated"],
            }
        else:
            add_ms, added, filtered, session_last_ts = Add_Session_Dialogue_To_Mnemosyne(
                memory, dialogue, self.importance, self.extract,
                backdate=self.backdate, session_base=session_base,
            )
        if self.backdate and session_last_ts is not None:
            ctx["last_ts"] = session_last_ts
        ctx["total_added"] += added
        ctx["total_filtered"] += filtered

        # Session-end forced consolidation (--plugin_session_sleep). This runs
        # after the whole session's writes and after the auto-sleep cadence
        # above, so it consolidates whatever those gates left behind. The
        # rows must be stamped consolidated BEFORE the next session's first
        # remember() calls _trim_working_memory, which deletes unconsolidated
        # rows older than the shipped 168h TTL (median session gap: 29 days).
        if self.plugin_session_sleep:
            Run_Plugin_Session_Sleep(
                memory, ctx["persona_tag"],
                session_item.get("Session_ID", session_index),
                ctx["sleep_totals"],
            )

        # Runs the real per-session consolidation lifecycle after ingesting
        # this session's messages and before answering its questions.
        if self.lifecycle:
            Run_Session_Lifecycle(
                memory, ctx["persona_tag"], session_item.get("Session_ID", session_index),
                ctx["sleep_totals"],
            )
        # Arm C: full production sleep. This populates canonical slots
        # through the LLM model-refresh path, run after surgical retirement
        # so slot proposals are inferred from the already-cleaned working set.
        if self.canonical:
            Run_Session_Sleep(
                memory, ctx["persona_tag"], session_item.get("Session_ID", session_index),
                ctx["sleep_totals"],
            )
        # Oracle arm: deterministically upserts gold-derived slots for this
        # session. This is an upper bound that separates model-refresh
        # quality from history-retrieval quality. CanonicalStore.remember()
        # versions supersession automatically when the body changes.
        oracle_slots = ctx["oracle_slots"]
        if oracle_slots and session_index in oracle_slots:
            self._write_oracle_slots(ctx, session_item, session_index)

        print(f"[DEBUG] persona {ctx['persona_tag']} "
              f"session {session_item.get('Session_ID', session_index)} added={added} "
              f"ingest_ms={add_ms:.0f}")
        meta = {
            "Dialogue_Added_To_Memory": added > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Added_Memory_Count": added,           # actually stored (remember() returned an id)
            "Filtered_Message_Count": filtered,    # rejected by Mnemosyne's write filter
            "Add_Duration_ms": add_ms,
        }
        meta.update(plugin_meta)  # empty dict for non-plugin arms (baseline unchanged)
        return meta

    def _write_oracle_slots(self, ctx, session_item, session_index):
        memory = ctx["memory"]
        store = getattr(memory.beam, "canonical", None)
        owner = str(getattr(memory.beam, "canonical_owner_id", "") or "").strip() or "default"
        # Backdates slot validity to the dataset session date, so the
        # history spans shown to the answer LLM ("[2022-01-30 -> ...]")
        # reflect simulated chronology, not the wall clock.
        date_iso = None
        try:
            date_iso = datetime.strptime(
                str(session_item.get("Date", "")).strip(), "%Y-%m-%d").isoformat()
        except Exception:
            pass
        for slot in ctx["oracle_slots"][session_index]:
            try:
                row = store.remember(owner, slot["category"], slot["name"],
                                     slot["body"], source="oracle_gold")
                ctx["sleep_totals"]["oracle_slots_written"] = (
                    ctx["sleep_totals"].get("oracle_slots_written", 0) + 1)
                if date_iso and row.get("status") in ("created", "updated") and row.get("id"):
                    conn = memory.beam.conn
                    conn.execute(
                        "UPDATE canonical_facts SET valid_from = ? WHERE id = ?",
                        (date_iso, row["id"]))
                    # The just-superseded predecessor got valid_until=now().
                    # This rewrites it to this session's date. The strict
                    # '>' leaves already-backdated historical rows untouched.
                    conn.execute(
                        "UPDATE canonical_facts SET valid_until = ? "
                        "WHERE owner_id = ? AND category = ? AND name = ? "
                        "AND valid_until > ?",
                        (date_iso, owner, slot["category"], slot["name"], date_iso))
                    conn.commit()
            except Exception as exc:
                print(f"[DEBUG] oracle persona {ctx['persona_tag']} session {session_index}: "
                      f"slot write failed ({slot.get('category')}/{slot.get('name')}): {exc}")

    def recall(self, ctx, question_text, top_k):
        search_top_k = max(top_k, MAX_STORED_RETRIEVED_MEMORIES)
        if self.plugin_prefetch_overlay:
            # Full plugin prefetch() overlay. It owns its own over-fetch
            # width (the plugin's 16) and its own final cut (the plugin's
            # _PREFETCH_TOP_K = 5), so this code deliberately does not
            # forward search_top_k. The candidate width and the output
            # width are both part of the behavior under test.
            return Search_Mnemosyne_Plugin_Prefetch(
                ctx["memory"], question_text,
                temporal_weight=self.temporal_weight,
                temporal_halflife=self.temporal_halflife,
                stats=ctx["prefetch_stats"],
                query_time=ctx.get("question_query_time"),
                diag=ctx,
            )
        if self.plugin_config != "off":
            # Plugin-faithful recall: hybrid weights plus the plugin's
            # temporal recency weighting (temporal_weight and
            # temporal_halflife). This uses the same top-5 over-fetch and
            # slice contract as every other arm.
            return Search_Mnemosyne_For_Question(
                ctx["memory"], question_text, search_top_k,
                temporal_weight=self.temporal_weight,
                temporal_halflife=self.temporal_halflife,
                query_time=ctx.get("question_query_time"),
                diag=ctx,
            )
        return Search_Mnemosyne_For_Question(
            ctx["memory"], question_text, search_top_k,
            query_time=ctx.get("question_query_time"),
            diag=ctx,
        )

    def _canonical_context_hook(self, ctx, question_text):
        if not ctx.get("canonical_active"):
            return "", {}
        canon_start = time.time()
        canon_context, canon_diag = Build_Canonical_Context(ctx["memory"], question_text)
        canon_ms = (time.time() - canon_start) * 1000.0
        fields = {
            "Canonical_Context_Diagnostics": canon_diag,
            "Canonical_Search_Duration_ms": canon_ms,
        }
        if canon_context:
            fields["Canonical_Context"] = canon_context
        return canon_context, fields

    def persona_count_extras(self, ctx):
        extras = {
            "Total_Added_Memories": ctx["total_added"],        # actually stored
            "Total_Filtered_Messages": ctx["total_filtered"],  # rejected by write filter
        }
        ps = ctx.get("plugin_stats")
        if self.plugin_config != "off" and ps is not None:
            extras.update({
                "Plugin_Config": self.plugin_config,
                "Total_Plugin_User_Written": ps["user_written"],
                "Total_Plugin_Assistant_Written": ps["assistant_written"],
                "Total_Plugin_Dropped_Messages": ps["dropped"],
                "Total_Plugin_User_Truncated": ps["user_truncated"],
                "Total_Plugin_Assistant_Truncated": ps["assistant_truncated"],
            })
        # Auto-sleep arm: surfaces the invocation count, its skips, and what
        # the sleeps produced, so the manifest and summary can prove the
        # arm actually invoked consolidation on the plugin cadence, rather
        # than silently doing nothing.
        if self.plugin_auto_sleep:
            st = ctx["sleep_totals"]
            extras.update({
                "Plugin_Auto_Sleep": True,
                "Total_Plugin_Sync_Turns": ctx.get("plugin_turn_count", 0),
                "Total_Auto_Sleep_Invocations": st.get("auto_sleep_invocations", 0),
                "Total_Auto_Sleep_Skipped": st.get("auto_sleep_skipped", 0),
                "Total_Auto_Sleep_Budget_Skipped": st.get("auto_sleep_budget_skipped", 0),
                "Total_Auto_Sleep_Summaries_Created": st.get("summaries_created", 0),
                "Total_Auto_Sleep_MR_Proposals": st.get("mr_proposals", 0),
                "Total_Auto_Sleep_MR_Applied": st.get("model_refresh_applied", 0),
            })
        # Session-end forced-sleep arm: one invocation per ingested session,
        # so Total_Session_Sleep_Invocations must equal the persona's session
        # count. The consolidated-row and summary counts prove the pass did
        # real work, and Session_Sleep_Working_Unconsolidated is what the
        # next session's TTL trim could still delete.
        if self.plugin_session_sleep:
            st = ctx["sleep_totals"]
            extras.update({
                "Plugin_Session_Sleep": True,
                "Total_Session_Sleep_Invocations": st.get("session_sleep_invocations", 0),
                "Total_Session_Sleep_No_Op": st.get("session_sleep_no_op", 0),
                "Total_Session_Sleep_Items_Consolidated":
                    st.get("session_sleep_items_consolidated", 0),
                "Total_Session_Sleep_Summaries_Created":
                    st.get("session_sleep_summaries_created", 0),
                "Total_Session_Sleep_MR_Proposals": st.get("session_sleep_mr_proposals", 0),
                "Total_Session_Sleep_MR_Applied": st.get("session_sleep_mr_applied", 0),
                "Session_Sleep_Working_Rows": st.get("session_sleep_wm_total", 0),
                "Session_Sleep_Working_Consolidated":
                    st.get("session_sleep_wm_consolidated", 0),
                "Session_Sleep_Working_Unconsolidated":
                    st.get("session_sleep_wm_unconsolidated", 0),
            })
        # Prefetch-overlay arm: every stage of the overlay reports how many
        # rows it moved. This block is the proof the arm ran. If the
        # overlay silently degraded to plain recall it would be absent
        # entirely, and if it ran but did nothing the counters would be
        # visibly all-zero. In particular, Prefetch_Assistant_Excluded == 0
        # under --plugin_config both, or Prefetch_Recall_Candidates == 0,
        # are broken-run signatures, not clean ones. Read this together
        # with Prefetch_Final_Count_Histogram, which shows exactly how
        # often the overlay handed the answer model fewer than 5 memories,
        # because the plugin does not backfill and neither does this code.
        ps_pf = ctx.get("prefetch_stats")
        if self.plugin_prefetch_overlay and ps_pf is not None:
            q = max(1, ps_pf["questions"])
            extras.update({
                "Plugin_Prefetch_Overlay": True,
                "Prefetch_Questions": ps_pf["questions"],
                "Prefetch_Recall_Candidates": ps_pf["candidates"],
                "Prefetch_Model_Refresh_Dropped": ps_pf["mr_dropped"],
                "Prefetch_Filtered_Low_Quality": ps_pf["filter_low_quality"],
                "Prefetch_Assistant_Excluded": ps_pf["filter_assistant"],
                "Prefetch_Filtered_Low_Signal": ps_pf["filter_signal"],
                "Prefetch_Filtered_Score_Importance": ps_pf["filter_score_importance"],
                "Prefetch_Kept_After_Filter": ps_pf["kept_after_filter"],
                "Prefetch_Canonical_Questions_Considered": ps_pf["canonical_considered"],
                "Prefetch_Canonical_Merged": ps_pf["canonical_merged"],
                "Prefetch_Canonical_In_Final": ps_pf["canonical_rows_in_final"],
                "Prefetch_Dedup_Dropped": ps_pf["dedup_dropped"],
                "Prefetch_Over_Top_K_Dropped": ps_pf["over_top_k_dropped"],
                "Prefetch_Content_Truncated": ps_pf["truncated"],
                "Prefetch_Content_Char_Limit": ps_pf["content_char_limit"],
                "Prefetch_Author_Id_Scoped": ps_pf["author_id_scoped"],
                "Prefetch_Final_Items": ps_pf["final_items"],
                "Prefetch_Mean_Items_Per_Question": round(ps_pf["final_items"] / q, 4),
                "Prefetch_Short_Questions": ps_pf["short_questions"],
                "Prefetch_Shortfall_Items": ps_pf["shortfall_items"],
                "Prefetch_Empty_Questions": ps_pf["empty_questions"],
                "Prefetch_Final_Count_Histogram": dict(ps_pf["final_count_hist"]),
            })
        return extras

    def persona_result_extras(self, ctx):
        # This key is present only in lifecycle, canonical, or oracle mode.
        # It is omitted entirely otherwise, so a non-lifecycle baseline run
        # stays byte-identical.
        return {"Retirement_Diagnostics": ctx.get("retirement_diagnostics")}

    def end_persona(self, ctx):
        memory = ctx.get("memory")
        # Overlay heartbeat in the shard log, so a run can be judged live,
        # by rows moved, without waiting for the runtime summary.
        ps_pf = ctx.get("prefetch_stats")
        if self.plugin_prefetch_overlay and ps_pf is not None:
            print(
                f"[DEBUG] prefetch overlay persona {ctx['persona_tag']}: "
                f"questions={ps_pf['questions']} candidates={ps_pf['candidates']} "
                f"drop_lowq={ps_pf['filter_low_quality']} "
                f"drop_assistant={ps_pf['filter_assistant']} "
                f"drop_signal={ps_pf['filter_signal']} "
                f"drop_score_imp={ps_pf['filter_score_importance']} "
                f"canonical_merged={ps_pf['canonical_merged']} "
                f"dedup_dropped={ps_pf['dedup_dropped']} "
                f"over_top_k={ps_pf['over_top_k_dropped']} "
                f"final_items={ps_pf['final_items']} "
                f"short_q={ps_pf['short_questions']} "
                f"empty_q={ps_pf['empty_questions']} "
                f"hist={ps_pf['final_count_hist']}"
            )
        # Collects retirement diagnostics while the DB connection is still open.
        if memory is not None and (self.lifecycle or self.canonical or ctx.get("oracle_slots")):
            sleep_totals = ctx["sleep_totals"]
            try:
                ctx["retirement_diagnostics"] = Collect_Retirement_Diagnostics(memory, sleep_totals)
            except Exception as exc:
                ctx["retirement_diagnostics"] = {
                    "error": str(exc), "sleep_totals": dict(sleep_totals)}
            v = (ctx["retirement_diagnostics"] or {}).get("veracity", {})
            cf = (ctx["retirement_diagnostics"] or {}).get("canonical_facts", {})
            wm = (ctx["retirement_diagnostics"] or {}).get("working_memory", {})
            ex = (ctx["retirement_diagnostics"] or {}).get("extraction", {})
            print(
                f"[DEBUG] lifecycle persona {ctx['persona_tag']}: "
                f"conflicts={v.get('conflicts_total')} resolved={v.get('conflicts_resolved')} "
                f"superseded_facts={v.get('superseded_facts')} "
                f"rows_invalidated={sleep_totals.get('rows_invalidated')} "
                f"canonical_current={cf.get('current')} canonical_retired={cf.get('retired')} "
                f"mr_proposals={sleep_totals.get('mr_proposals', 0)} "
                f"mr_applied={sleep_totals.get('model_refresh_applied', 0)} "
                f"facts_extracted={ex.get('facts_total')} "
                f"wm_invalidated={wm.get('invalidated')}"
            )
        # Optional: persists the per-persona DB for offline inspection
        # before removing the temp dir. This copies the WAL and SHM files
        # too, so a later sqlite3 open replays any uncommitted pages. Set
        # MNEMOSYNE_KEEP_DB=<dir> to enable it.
        _keep_db = os.environ.get("MNEMOSYNE_KEEP_DB", "").strip()
        if _keep_db:
            try:
                os.makedirs(_keep_db, exist_ok=True)
                for _suffix in ("", "-wal", "-shm"):
                    _src = ctx["db_path"] + _suffix
                    if os.path.exists(_src):
                        shutil.copy2(_src, os.path.join(
                            _keep_db, f"persona_{ctx['persona_tag']}.db{_suffix}"))
            except Exception as _e:
                print(f"[DEBUG] MNEMOSYNE_KEEP_DB copy failed: {_e}")
        shutil.rmtree(ctx["tmp_dir"], ignore_errors=True)


def Generate_User_Mnemosyne_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
    importance: float,
    extract: bool = False,
    lifecycle: bool = False,
    canonical: bool = False,
    oracle: bool = False,
    use_dataset_time: bool = False,
    plugin_config: str = "off",
    plugin_auto_sleep: bool = False,
    plugin_session_sleep: bool = False,
    plugin_prefetch_overlay: bool = False,
) -> bool:
    if plugin_config != "off":
        print(f"[DEBUG] Plugin-fidelity mode ON (plugin_config={plugin_config}; "
              "per-exchange sync_turn ingestion mirroring the Hermes plugin — fixed "
              "per-role importances, entity extraction only, temporal recall; implies "
              "--use_dataset_time; no LLM fact extraction / lifecycle / canonical)")
    if plugin_auto_sleep:
        print(f"[DEBUG] Plugin AUTO-SLEEP arm ON (cadence={PLUGIN_AUTO_SLEEP_EXCHANGE_CADENCE} "
              "exchanges + per-session-boundary; unforced sleep_all_sessions, threshold "
              f"working>{_plugin_auto_sleep_threshold()}, drained-to-completion before "
              "questions; mirrors MnemosyneMemoryProvider._maybe_auto_sleep / "
              "on_session_end)")
    if plugin_session_sleep:
        print("[DEBUG] Plugin SESSION-SLEEP arm ON (one sleep(force=True) after each "
              "session's ingest, before its questions, drained; shipped consolidation, "
              "manually cadenced — the consolidated rows are exempt from "
              "_trim_working_memory, so the shipped 168h WM TTL keeps deleting only "
              "unconsolidated rows)")
    if plugin_prefetch_overlay:
        print("[DEBUG] Plugin PREFETCH OVERLAY arm ON (full MnemosyneMemoryProvider."
              "prefetch() read path: 16-candidate recall, quality/topic filter, "
              "[ASSISTANT] hard exclusion, canonical merge (limit 3), "
              "adjusted-score sort, lexical semantic dedup (jaccard>=0.72 / "
              "containment>=0.86), top-5, NO backfill — executed via the pinned "
              "plugin's own helper functions)")
    if use_dataset_time and not plugin_auto_sleep:
        print("[DEBUG] use_dataset_time ON (dataset-chronology backdating WITHOUT "
              "lifecycle retirement)")
    if lifecycle:
        print("[DEBUG] Lifecycle mode ON (timestamp restoration + per-session "
              "consolidation + retirement diagnostics; extract forced on)")
    if canonical:
        print("[DEBUG] Canonical mode ON (per-session sleep(force=True) model-refresh "
              "populates canonical slots; history-aware canonical retrieval per question)")
    if oracle:
        print("[DEBUG] Oracle-canonical mode ON (gold-derived canonical slots — "
              "upper-bound arm; history-aware canonical retrieval per question)")
    binding = MnemosyneBinding(
        importance=importance,
        extract=extract,
        lifecycle=lifecycle,
        canonical=canonical,
        oracle=oracle,
        use_dataset_time=use_dataset_time,
        plugin_config=plugin_config,
        plugin_auto_sleep=plugin_auto_sleep,
        plugin_session_sleep=plugin_session_sleep,
        plugin_prefetch_overlay=plugin_prefetch_overlay,
    )
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mnemosyne evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "mnemosyne_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "mnemosyne_results.json"),
        top_k_help="How many retrieved memories the answer LLM sees (up to 5 are "
                   "always stored for white-box scoring). The committed run used 5; "
                   "NOTE the upstream MemConflict adapters (eval_memzero.py etc.) "
                   "answer from top-3 by default, so use --top_k 3 for strict "
                   "answer-accuracy comparability with published numbers.",
        max_sessions_help="Cap sessions ingested per persona (default: all).",
    )
    parser.add_argument("--importance", type=float, default=None,
                        help="Uniform importance for every remembered message (default 0.6). "
                             "Mnemosyne's own default is 0.5; a uniform value has no effect on "
                             "relative ranking. Mutually exclusive with --plugin_config (which "
                             "uses the plugin's fixed per-role importances 0.5 / 0.15).")
    parser.add_argument("--extract", action="store_true",
                        help="Enable Mnemosyne LLM fact-extraction on ingest "
                             "(remember(extract=True)). Requires MNEMOSYNE_LLM_BASE_URL / "
                             "MNEMOSYNE_LLM_MODEL to point at an LLM (one extraction call "
                             "per stored message — much slower).")
    parser.add_argument("--lifecycle", action="store_true",
                        help="Enable production lifecycle / stale-fact retirement mode: "
                             "restore dataset timestamps on ingest, run the real per-session "
                             "consolidation (veracity_consolidator.run_consolidation_pass() + "
                             "sleep(force=True)) so conflicting/stale facts get retired, and "
                             "attach Retirement_Diagnostics per persona. Implies --extract "
                             "(lifecycle needs extracted facts). Container entrypoint maps "
                             "LIFECYCLE=1 -> --lifecycle.")
    parser.add_argument("--canonical", action="store_true",
                        help="Arm C — full production lifecycle: --lifecycle PLUS per-session "
                             "sleep(force=True) (model-refresh populates canonical identity "
                             "slots) PLUS history-aware canonical retrieval appended to the "
                             "answer context. AA is the primary metric for this arm (raw-turn "
                             "SEH is structurally biased against retirement). Container "
                             "entrypoint maps CANONICAL=1 -> --canonical.")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle upper-bound arm: --lifecycle PLUS canonical slots derived "
                             "deterministically from dataset gold (build-time gold use only; "
                             "retrieval stays question-driven) PLUS history-aware canonical "
                             "retrieval. Separates model-refresh quality from history-retrieval "
                             "quality. Mutually exclusive with --canonical. Container "
                             "entrypoint maps ORACLE=1 -> --oracle.")
    parser.add_argument("--use_dataset_time", action="store_true",
                        help="Backdate ingested working_memory rows to the dataset's simulated "
                             "session chronology (2022+) WITHOUT lifecycle retirement — the "
                             "timestamp-restoration half of --lifecycle, decoupled. --lifecycle / "
                             "--canonical / --oracle / --plugin_config all imply it. Requires the "
                             "WM TTL raised (MNEMOSYNE_WM_TTL_HOURS) or Mnemosyne trims the "
                             "backdated rows. Container entrypoint maps USE_DATASET_TIME=1.")
    parser.add_argument("--plugin_config", choices=["off", "both", "user"], default="off",
                        help="Plugin-fidelity ingestion mirroring the Hermes Mnemosyne plugin's "
                             "sync_turn(): one remember() per role per completed exchange, fixed "
                             "per-role importances (user 0.5 / assistant 0.15), entity extraction "
                             "only (NO LLM fact extraction), per-role length gates + truncation, "
                             "temporal recall (temporal_weight=0.2, halflife=48). 'user' writes "
                             "only the user role (plugin default sync_roles={'user'}); 'both' also "
                             "writes assistant. Implies --use_dataset_time. Mutually exclusive with "
                             "--extract / --lifecycle / --canonical / --oracle / --importance. "
                             "Container entrypoint maps PLUGIN_CONFIG={off,user,both}.")
    parser.add_argument("--plugin_auto_sleep", action="store_true",
                        help="Add the Hermes plugin's real sleep cadence to the plugin arm: fire "
                             "the plugin's auto-sleep every "
                             f"{PLUGIN_AUTO_SLEEP_EXCHANGE_CADENCE} user-led exchanges "
                             "(MnemosyneMemoryProvider.sync_turn's _turn_count %% 10) AND once at "
                             "each session boundary (on_session_end), gated on the same "
                             "working>threshold + eligible-rows checks, using the plugin's own "
                             "unforced sleep_all_sessions(). Runs DRAINED-to-completion before "
                             "questions (quality-arm ruling, same as Hindsight WAIT_CONSOLIDATION). "
                             "REQUIRES --plugin_config user|both (the plugin's auto-sleep only "
                             "exists on the plugin write path). Container entrypoint maps "
                             "PLUGIN_AUTO_SLEEP=1.")
    parser.add_argument("--plugin_session_sleep", action="store_true",
                        help="Force one sleep(force=True) after each session's ingest and "
                             "before its questions, drained to completion — the featured "
                             "clock-sync arm's manually cadenced consolidation (user ruling "
                             "2026-08-02, mirroring Honcho's HONCHO_DREAM_AFTER_SESSION). "
                             "Under BENCH_CLOCKSYNC the shipped 168h working-memory TTL "
                             "deletes unconsolidated rows at the next session's first write "
                             "(median session gap 29 days), while the plugin's own auto-sleep "
                             "gate needs working>50, which needs the accumulation the trim "
                             "removes. Consolidated rows are exempt from the trim, so this "
                             "keeps the TTL shipped and the history alive. COMPOSES with "
                             "--plugin_auto_sleep (the plugin cadence stays on). Mutually "
                             "exclusive with --lifecycle / --canonical / --oracle, which run "
                             "their own per-session sleep. Container entrypoint maps "
                             "PLUGIN_SESSION_SLEEP=1.")
    parser.add_argument("--plugin_prefetch_overlay", action="store_true",
                        help="Replace the plugin arm's hybrid read path (raw recall + "
                             "temporal weighting only) with the Hermes plugin's FULL "
                             "prefetch() retrieval overlay — MnemosyneMemoryProvider."
                             "prefetch(), integrations/hermes/.../__init__.py:1158-1238: "
                             "16-candidate recall, the quality/topic filter (low-quality "
                             "fragments, HARD [ASSISTANT] exclusion, topical-signal floor "
                             "0.18 raw / 0.08 distilled, score<0.20 and importance<0.65), "
                             "owner-scoped canonical merge (limit 3, lexical), "
                             "adjusted-score ordering, lexical semantic dedup "
                             "(jaccard>=0.72 or containment>=0.86) and the plugin's top-5 "
                             "cut with NO backfill. Executed by CALLING the pinned "
                             "plugin's own helper functions, so no threshold is "
                             "re-implemented here. REQUIRES --plugin_config user|both "
                             "(prefetch() is the plugin's read surface; running it over a "
                             "non-plugin write path would measure neither arm). Composes "
                             "with --plugin_auto_sleep. Container entrypoint maps "
                             "PLUGIN_PREFETCH_OVERLAY=1.")
    args = parser.parse_args()

    if args.canonical and args.oracle:
        parser.error("--canonical and --oracle are mutually exclusive arms")

    # --plugin_auto_sleep only has meaning on the plugin write path, because
    # the auto-sleep cadence lives inside MnemosyneMemoryProvider.sync_turn.
    # It also must not combine with --canonical, which is itself a
    # sleep-based arm (per-session sleep(force=True)). Combining them would
    # double-invoke sleep with conflicting force semantics. --canonical is
    # already rejected below through the plugin_config exclusivity list,
    # but this guards --plugin_auto_sleep explicitly for a clear error.
    if args.plugin_auto_sleep:
        if args.plugin_config == "off":
            parser.error("--plugin_auto_sleep requires --plugin_config user|both "
                         "(the plugin auto-sleep cadence only exists on the plugin write path)")
        if args.canonical or args.oracle or args.lifecycle:
            parser.error("--plugin_auto_sleep is mutually exclusive with the sleep-based "
                         "consolidation arms (--lifecycle / --canonical / --oracle); "
                         "combining them would double-invoke sleep")

    # --plugin_session_sleep answers a beam-level behaviour (the working-memory
    # TTL trim), so it is coherent on any ingest path, including the plain
    # plugin arm and a bare --use_dataset_time run. It must NOT combine with
    # the sleep-based arms: --lifecycle, --canonical, and --oracle already run
    # their own per-session consolidation, so combining them double-invokes
    # sleep on the same rows for one session.
    if args.plugin_session_sleep and (args.canonical or args.oracle or args.lifecycle):
        parser.error("--plugin_session_sleep is mutually exclusive with the sleep-based "
                     "consolidation arms (--lifecycle / --canonical / --oracle), which "
                     "already run their own per-session sleep(force=True)")

    # --plugin_prefetch_overlay reproduces the plugin's read surface, so it
    # is only coherent on top of the plugin's write surface. Bolting
    # prefetch()'s [ASSISTANT] or raw-source filtering onto, for example,
    # the lifecycle or oracle arms would measure a configuration that
    # exists nowhere. This fails fast, in the same style as the
    # --plugin_auto_sleep guard above, and the entrypoint mirrors this
    # check so a bad combination does not launch many doomed shards.
    if args.plugin_prefetch_overlay:
        if args.plugin_config == "off":
            parser.error("--plugin_prefetch_overlay requires --plugin_config user|both "
                         "(prefetch() is the plugin's read surface for the plugin's own "
                         "write path)")
        if args.canonical or args.oracle or args.lifecycle or args.extract:
            parser.error("--plugin_prefetch_overlay is mutually exclusive with the "
                         "extraction/consolidation arms (--extract / --lifecycle / "
                         "--canonical / --oracle)")

    plugin_config = args.plugin_config
    if plugin_config != "off":
        # The plugin arm reproduces the plugin's automatic-write behavior
        # exactly, so it cannot combine with the LLM-extraction or
        # consolidation arms, or a custom uniform importance, because it
        # uses the plugin's fixed per-role importances.
        conflicts = []
        if args.extract:
            conflicts.append("--extract")
        if args.lifecycle:
            conflicts.append("--lifecycle")
        if args.canonical:
            conflicts.append("--canonical")
        if args.oracle:
            conflicts.append("--oracle")
        if args.importance is not None:
            conflicts.append("--importance")
        if conflicts:
            parser.error(
                "--plugin_config is mutually exclusive with "
                + ", ".join(conflicts)
                + " (it faithfully replays the plugin's automatic per-role writes)"
            )

    # The --importance default is a sentinel (None), so --plugin_config can
    # detect an explicit override. This resolves to the historical 0.6
    # default here.
    importance = 0.6 if args.importance is None else args.importance

    # Canonical and oracle arms build on the lifecycle machinery (timestamp
    # restoration plus surgical retirement). Lifecycle needs extracted facts.
    lifecycle = args.lifecycle or args.canonical or args.oracle
    extract = args.extract or lifecycle
    # --lifecycle (and the arms built on it) and --plugin_config all imply
    # dataset-time backdating. --use_dataset_time requests it standalone,
    # with no retirement.
    use_dataset_time = args.use_dataset_time or lifecycle or (plugin_config != "off")

    ok = Generate_User_Mnemosyne_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
        importance=importance,
        extract=extract,
        lifecycle=lifecycle,
        canonical=args.canonical,
        oracle=args.oracle,
        use_dataset_time=use_dataset_time,
        plugin_config=plugin_config,
        plugin_auto_sleep=args.plugin_auto_sleep,
        plugin_session_sleep=args.plugin_session_sleep,
        plugin_prefetch_overlay=args.plugin_prefetch_overlay,
    )
    # Propagates failure as a nonzero exit. eval_common.run_eval() catches
    # fatal exceptions and returns False. Without this, __main__ falls off
    # the end and exits 0, so the entrypoint's `set -e` never fires, and
    # STAGE=all would go on to score a partial or empty results file. A run
    # that died at persona 2 of 30 would then look successful. This fails
    # loudly instead.
    raise SystemExit(0 if ok else 1)
