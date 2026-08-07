"""MemConflict evaluation adapter for the RetainDB memory system.

The shared driver in ``benchmark/eval_common.py`` handles the pipeline steps
common to every provider: dataset iteration, dialogue flattening, the answer
prompt, the answer LLM call, and result emission. This file adds only the
RetainDB-specific parts: server setup, ingestion, and retrieval.

DIFFERENCES FROM MNEMOSYNE AND HINDSIGHT
-----------------------------------------------------------
RetainDB Local is not a Python SDK. It has no internal LLM. It is a small
Node HTTP server (``dist/cli.js``). The adapter drives it over REST (see
``_retaindb_server.py``). Two facts follow from this.

  * INGESTION IS HEURISTIC, NOT LLM-EXTRACTED. ``POST /v1/memory/ingest/session``
    turns each dialogue message into one stored memory. It applies a quality
    gate (it drops low-signal lines like "ok, thanks"), a content-hash dedup
    step, and a rule-based memory-type inference. Unlike Hindsight, ingestion
    does not use an LLM call. The memory system runs locally and offline. The
    only LLM in this pipeline is the shared answer and judge model.

  * RETRIEVAL USES A FIXED HYBRID FUSION. ``POST /v1/memory/search`` ranks
    results with BM25, vector cosine similarity over the configured embedding
    provider, and concept-graph overlap. It fuses these with RRF (k=60), then
    applies a proximity rerank plus recency, decay, and reinforcement boosts.
    The only tunable ML setting is the embedding provider: ``hash`` (default,
    zero-dependency, deterministic, lexical-like) or ``local-transformers``
    (Xenova/all-MiniLM-L6-v2, real semantic embeddings, runs in-process). Set
    it with ``--embedding_provider`` (see docs/BENCHMARK_MATRIX.md). RetainDB
    Local does not honor the server-package retrieval knobs (profile, hybrid
    weights, rerank toggle, threshold). The only per-query lever is ``top_k``.

  * THE PLUGIN HANDS HERMES COMPACTED TEXT, NOT RAW TEXT. The product under
    test is not the raw search row. It is the plugin's ``_build_overlay``
    block: each memory whitespace-collapsed and truncated to 320 characters,
    then deduped on a normalized key. This adapter reproduces that transform
    (see ``_plugin_compact`` and ``Search_RetainDB_For_Question``). It is on
    by default. Set ``RETAINDB_PLUGIN_OVERLAY=0`` to disable it for diagnosis.

TIME. The adapter passes the dataset's simulated session ``Date`` as each
message's ``timestamp``. It stores this in memory metadata and reports it as
the scorer's ``created_at``. RetainDB Local stamps each memory's internal
``created_at`` with wall-clock ingest time, not the dataset date. So its
recency scoring reflects ingest order, not session date. The adapter ingests
sessions in chronological order, so ingest order still matches the benchmark
chronology, and later-session facts still score as more recent. True
date-aware recency would need RetainDB to accept a per-memory timestamp. It
does not support that today.

Isolation is per-persona. Each persona gets a unique ``project`` slug inside
one disposable server. This is RetainDB's native tenancy boundary, used
instead of a separate database file.
"""

import argparse
import os
import re
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

# Shared, provider-agnostic harness modules (eval_common, llm_reasoning, the
# scorers) live in ../benchmark; make them importable regardless of launch cwd.
_SHARED_HARNESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark"))
if _SHARED_HARNESS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_HARNESS_DIR)

from dotenv import load_dotenv

import eval_common  # noqa: E402
from eval_common import (  # noqa: E402  (re-exports keep old imports working)
    Pair_Exchange_Turns,
    Parse_Session_Timestamp,
    ProviderBinding,
    add_common_eval_args,
)

# RetainDB server manager + REST client (owns a disposable Node server).
from _retaindb_server import RetainDBServer, RetainDBClient  # noqa: E402

load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))


# --------------------------------------------------------------------------
# Hermes-plugin retrieval overlay (what RetainDB actually hands the agent)
# --------------------------------------------------------------------------
# WHY THIS EXISTS. This project tests the plugin's behavior as Hermes' memory
# provider (CLAUDE.md Goal). That means the memory text the plugin puts in
# front of the agent at recall time, not the raw row RetainDB stores. The
# Hermes plugin never shows the agent a raw search row. Every recalled memory
# passes through ``_build_overlay``, which whitespace-collapses it and cuts
# it to 320 characters before it builds the block. Source, re-read verbatim
# 2026-07-21 from NousResearch/hermes-agent @ main,
# ``plugins/memory/retaindb/__init__.py``:
#
#     def _build_overlay(profile, query_result, local_entries=None) -> str:
#         def _compact(s): return re.sub(r"\s+", " ", str(s or "")).strip()[:320]
#         def _norm(s):    return re.sub(r"[^a-z0-9 ]", "", _compact(s).lower())
#         seen = [_norm(e) for e in (local_entries or []) if _norm(e)]
#         profile_items = []                       # profile["memories"][:5]
#         query_items   = []                       # query_result["results"][:5]
#         ... for each: c = _compact(content); n = _norm(c)
#             if c and n not in seen: seen.append(n); <half>_items.append(c)
#         lines = ["[RetainDB Context]", "Profile:"] + [f"- {i}" ...]
#                 + ["Relevant memories:"] + [f"- {i}" ...]
#
# The overlay is not tool-only. ``_prefetch_context`` builds it on every turn
# (query_context + get_profile -> _build_overlay), and ``prefetch()`` hands
# the block to the agent. So compaction is part of the plugin's normal recall.
#
# COMPACTION RUNS PER ITEM, NOT PER BLOCK. ``_compact`` runs on each memory on
# its own, then the items get assembled. This matches our per-item
# ``Retrieved_Memories`` contract: we compact each item and keep it as its
# own row, so the harness's item shape does not change.
#
# ONE TEXT, TWO CONSUMERS. DO NOT SPLIT THIS.
# The compacted string goes to both the answer context and the stored
# ``Retrieved_Memories[].memory``. The answer model must not see 320
# characters while the judge sees the full row. We considered and rejected
# that split, for two reasons:
#   1. It would inflate SEH@K beyond anything Hermes ever saw. The judge
#      would find supporting evidence in text the agent never received, so
#      the retrieval metric would describe a setup that does not exist.
#   2. It would misattribute failures. A question the agent gets wrong
#      because the evidence was truncated is an evidence failure. If the
#      judge still sees the untruncated row, it scores as gold-reached-top-5
#      plus answered-wrong, so it lands in EUG-cond@5 as a reasoning failure.
#      EUG-cond@5 exists to separate those two cases. Feeding it mismatched
#      texts silently corrupts that diagnostic.
# An earlier version of this adapter skipped compaction. It claimed 320
# characters would truncate evidence the scorer must see. That claim was
# false: MemConflict's SEH is judged semantically (eval_scoring.py asks for
# the rank of the first retrieved memory with evidence that supports the
# reference answer, "do not require exact wording"). Nothing matches
# retrieved text against gold turns literally. Truncation that costs
# RetainDB points is a finding about the plugin, not a harness defect to fix.
#
# NOT REPRODUCED: the overlay's profile half. See the long note on
# ``_PLUGIN_PROFILE_HALF_OMITTED`` below and docs/DECISIONS.md.
_PLUGIN_COMPACT_CHAR_LIMIT = 320
# ``_build_overlay`` slices each half with a hardcoded ``[:5]``. We slice the
# query half to the harness's top_k instead (5 by default, so this matches):
# the shared top-K is the fairness line for all providers, and a run at
# --top_k 3 must still answer from 3 memories.
_PLUGIN_QUERY_CONTEXT_SERVER_TOPK = 10
# The plugin's overlay half comes from ``query_context`` -> ``POST
# /v1/context/query``, which sends no ``top_k``. RetainDB Local defaults that
# to ``topK = min(max(input.top_k || 10, 1), 100)`` = 10, then slices
# (``this.rerank(...).sort(...).slice(0, topK)``). Rerank and sort run over
# the full candidate set, so the first 5 rows are the same whether we ask for
# 5 or 10. We request 10 and slice to top_k ourselves. This reproduces the
# plugin's ranking exactly, and also its reinforcement side effect: search()
# bumps ``access_count``, ``last_accessed_at``, and ``strength`` on exactly
# the rows it returns. Asking for 5 instead of 10 would reinforce a
# different set of memories and change later questions' ranking.
#
# NOTE on /v1/context/query vs /v1/memory/search in RetainDB Local (verified
# by reading @retaindb/local@0.2.1 dist/cli.js): both handlers run the same
# ``const results = await runtime.search(body)``. /v1/context/query returns
# the same ``results`` array plus a pre-joined ``context`` string and a
# ``meta`` block. It applies no extra ranking and no ``max_tokens`` trim of
# ``results``. So calling /v1/memory/search here is not an approximation of
# the plugin's query half. It is the same server-side computation, and it is
# the endpoint that carries the per-item ``score`` the harness contract needs.


def _plugin_compact(text: Any) -> str:
    """Reproduce the plugin's ``_compact`` literally: regex, strip, 320-char cut."""
    return re.sub(r"\s+", " ", str(text or "")).strip()[:_PLUGIN_COMPACT_CHAR_LIMIT]


def _plugin_norm(text: Any) -> str:
    """Reproduce the plugin's ``_norm`` dedup key: compact, lowercase, keep only
    ``[a-z0-9 ]``. This normalizes the already-compacted string, so two rows
    that differ only past character 320 dedup to one item."""
    return re.sub(r"[^a-z0-9 ]", "", _plugin_compact(text).lower())


# THE PROFILE HALF IS OMITTED. This is the one place the old structural
# objection still applies. We disclose it here rather than drop it silently
# (docs/DECISIONS.md). ``_prefetch_context`` builds the overlay from two
# halves: ``get_profile`` -> ``GET /v1/memory/profile/{user_id}`` (first 5),
# then the query results (first 5), deduped against the profile items.
# RetainDB Local does implement that route, so this omission is deliberate,
# not a server limit. Three reasons:
#   1. It carries no per-item score. The handler is
#      ``json(c, { memories, count })`` over ``runtime.profile(...)``, which is
#      ``filter(project, user_id, active).sort(created_at DESC).slice(0, limit)``,
#      a plain recency dump. The harness contract needs a per-item ``score``
#      on every stored row (SEH@K and log-rank@K read it). None exists here,
#      and inventing one would be worse than omitting the half.
#   2. It is query-independent, and it degenerates under this adapter.
#      RetainDB Local stamps ``created_at`` at wall-clock ingest time (see
#      the TIME note in the module docstring), so "profile" here is the last
#      5 memories ingested for the persona, the tail of the final session,
#      the same for every question. It is not retrieval.
#   3. Including it would break the shared top-K, the fairness line. The
#      plugin's overlay carries up to 10 items, but every other provider
#      answers from 5. Handing RetainDB 10 context items is a harness change
#      that helps only one provider, which CLAUDE.md forbids. Squeezing the
#      profile half into the 5-item budget would be worse: 5
#      query-independent recency rows would displace the real search hits.
# Net effect on the measurement: the omitted half can only have helped
# RetainDB (5 extra items, always ranked ahead of the query hits,
# occasionally containing the answer by luck). So the reported number is, if
# anything, a floor. The only other effect is on dedup: a profile item can
# suppress a query item, which would shorten our list, not lengthen it.
_PLUGIN_PROFILE_HALF_OMITTED = True


def _env_flag(name: str, default: bool) -> bool:
    """Read an env var as a boolean flag for arm selection. Return ``default`` if unset.

    This helper stays local. The shared driver has no such helper, and
    adapters must not add one to it. ``0``, ``false``, ``no``, ``off``, and
    empty string are false. Anything else is true.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


# --------------------------------------------------------------------------
# RetainDB backend
# --------------------------------------------------------------------------
def Setup_RetainDB(
    profile: str,
    embedding_provider: str,
    embedding_model: Optional[str],
    base_url: Optional[str],
) -> Tuple[RetainDBClient, Optional[RetainDBServer]]:
    """Return a RetainDB REST client.

    If ``base_url`` is set, attach to an already-running server and own no
    server lifecycle. Otherwise spawn a disposable local server and own it.
    """
    if base_url:
        client = RetainDBClient(base_url)
        client.health()  # fail fast if the server is unreachable
        print(f"[retaindb] attached to external server at {base_url}", flush=True)
        return client, None

    server = RetainDBServer(
        profile=profile,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    client = server.start()
    return client, server


def _iso(timestamp: Optional[datetime]) -> Optional[str]:
    return timestamp.isoformat() if timestamp else None


def Add_Session_Dialogue_To_RetainDB(
    client: RetainDBClient,
    project: str,
    user_id: str,
    session_label: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
    granularity: str,
) -> Tuple[float, int, int]:
    """Ingest one session's dialogue into RetainDB.

    ``granularity='session'`` (default): one ``ingest/session`` call carries
    every message. RetainDB stores one memory per message, with its quality
    gate and dedup. ``granularity='message'``: one ``/v1/memory`` add call
    per message. This gives finer control over memory_type and importance,
    but is otherwise equivalent, since neither path uses an LLM.

    Returns (add_duration_ms, memories_created, memories_skipped_or_failed).
    """
    if not dialogue_messages:
        return 0.0, 0, 0
    ts_iso = _iso(timestamp)
    start = time.time()
    created = 0
    skipped = 0

    if granularity == "message":
        print(f"[DEBUG] project={project} session {session_label} "
              f"ingest_call msgs={len(dialogue_messages)}")
        for message in dialogue_messages:
            prefix = "User asked" if message["role"] == "user" else "Agent responded"
            content = f"{prefix}: {message['content']}"
            mem_type = "event" if message["role"] == "user" else "project_state"
            try:
                resp = client.add_memory(
                    project=project,
                    content=content,
                    memory_type=mem_type,
                    session_id=session_label,
                    user_id=user_id,
                    metadata={"timestamp": ts_iso, "role": message["role"]},
                )
                # A returned memory_type of 'skipped' means RetainDB judged the line low-signal.
                if resp.get("memory_type") == "skipped" or resp.get("active") is False:
                    skipped += 1
                else:
                    created += 1
            except Exception as e:  # pragma: no cover
                print(f"[DEBUG] add_memory failed project={project}: {e}")
                skipped += 1
    else:  # 'session'
        messages = [
            {"role": m["role"], "content": m["content"], "timestamp": ts_iso}
            for m in dialogue_messages
        ]
        print(f"[DEBUG] project={project} session {session_label} "
              f"ingest_call msgs={len(messages)}")
        try:
            resp = client.ingest_session(
                project=project,
                session_id=session_label,
                messages=messages,
                user_id=user_id,
            )
            created = int(resp.get("memories_created", 0) or 0)
            skipped = int(resp.get("skipped", 0) or 0)
        except Exception as e:  # pragma: no cover
            print(f"[DEBUG] ingest_session failed project={project}: {e}")
            skipped = len(dialogue_messages)

    return (time.time() - start) * 1000.0, created, skipped


def Add_Session_Exchanges_To_RetainDB(
    client: RetainDBClient,
    project: str,
    user_id: str,
    session_label: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
) -> Tuple[float, int, int, int, int]:
    """Plugin-faithful ingest: one ``ingest/session`` call per user/assistant exchange.

    This mirrors the Hermes ``retaindb`` memory plugin's ``sync_turn``
    (NousResearch/hermes-agent, ``plugins/memory/retaindb/__init__.py`` @
    main). The plugin fires once per completed exchange and queues one
    ingest call carrying exactly the two ``[{user}, {assistant}]`` message
    dicts, with verbatim content (no role-prefix rewrite, no truncation),
    both sharing one timestamp. Its only client-side gate drops an exchange
    whose ``user_content`` is empty or falsy.

    We reproduce this per exchange, with drained synchronous ingest (the
    quality-arm ruling used for every provider) in place of the plugin's
    async SQLite write-behind queue:

      * ``[user, assistant]`` group: one ingest_session call with both
        messages verbatim.
      * lone ``[user]`` group (no assistant reply): ingest it alone. The
        plugin writes the user role of an exchange, and a missing assistant
        reply does not drop it.
      * lone ``[assistant]`` group (no preceding user): drop it, and count
        the drop. This mirrors the plugin's empty-user gate: ``sync_turn``
        returns early when there is no ``user_content``.

    Returns (add_duration_ms, memories_created, memories_skipped, exchanges_ingested,
    exchanges_dropped_no_user).
    """
    if not dialogue_messages:
        return 0.0, 0, 0, 0, 0
    ts_iso = _iso(timestamp)
    start = time.time()
    created = 0
    skipped = 0
    exchanges_ingested = 0
    exchanges_dropped_no_user = 0

    exchanges = Pair_Exchange_Turns(dialogue_messages)
    total_exchanges = len(exchanges)
    # Below 40 exchanges, print every one. Above that, print only every 10th,
    # so a long session does not flood stdout.
    cap_mode = total_exchanges > 40

    for exch_idx, group in enumerate(exchanges, start=1):
        if not group:
            continue
        # A size-1 group led by a non-user role is a lone assistant turn.
        # The empty-user gate drops it (the plugin's only client-side gate).
        if group[0].get("role") != "user":
            exchanges_dropped_no_user += 1
            continue
        messages = [
            {"role": m["role"], "content": m["content"], "timestamp": ts_iso}
            for m in group
        ]
        if (not cap_mode) or (exch_idx % 10 == 0):
            print(f"[DEBUG] project={project} session {session_label} "
                  f"ingest_call exchange={exch_idx}/{total_exchanges} msgs={len(messages)}")
        try:
            resp = client.ingest_session(
                project=project,
                session_id=session_label,
                messages=messages,
                user_id=user_id,
                write_mode="sync",
            )
            created += int(resp.get("memories_created", 0) or 0)
            skipped += int(resp.get("skipped", 0) or 0)
            exchanges_ingested += 1
        except Exception as e:  # pragma: no cover
            print(f"[DEBUG] exchange ingest failed project={project} session={session_label}: {e}")
            skipped += len(messages)

    return (
        (time.time() - start) * 1000.0,
        created,
        skipped,
        exchanges_ingested,
        exchanges_dropped_no_user,
    )


def _result_created_at(result: Dict[str, Any]) -> str:
    """Return the best temporal anchor for a recalled memory, for the scorer's created_at slot.

    Prefer the dataset session date stored in metadata.timestamp. Fall back
    to RetainDB's wall-clock created_at.
    """
    metadata = result.get("metadata") or {}
    memory = result.get("memory") or {}
    inner_meta = memory.get("metadata") if isinstance(memory, dict) else None
    for candidate in (
        metadata.get("timestamp"),
        (inner_meta or {}).get("timestamp"),
        memory.get("created_at") if isinstance(memory, dict) else None,
        result.get("created_at"),
    ):
        if candidate:
            return str(candidate)
    return "Unknown Time"


def Search_RetainDB_For_Question(
    client: RetainDBClient,
    project: str,
    user_id: str,
    question_text: str,
    top_k: int,
    plugin_overlay: bool = True,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Recall memories for one question and map them to the scorer's memory shape.

    ``plugin_overlay=True`` (default, and what a headline run uses) reproduces
    the Hermes plugin's ``_build_overlay`` query half. It fetches at the
    server default top-K, slices to the harness top_k, runs per-item
    ``_compact`` (whitespace collapse plus 320-char cut), then dedups on
    ``_norm`` in order. The compacted text becomes ``memory`` on the stored
    row. The shared driver also renders this into the answer prompt through
    ``Build_Retrieved_Memory_Context``: one text, two consumers (see the long
    note above; do not split them).

    ``plugin_overlay=False`` restores the pre-2026-07-21 behavior (raw
    ``content``, no dedup, plain ``top_k`` request), so a diagnostic run can
    isolate how much of RetainDB's score is lost to compaction. This is not
    plugin-faithful. Do not use it for a headline number.

    Per-item ``score``, ``created_at``, and every diagnostic key stay the
    same in both modes. Only the memory text differs.

    DEDUP CAN YIELD FEWER THAN top_k ITEMS, ON PURPOSE. The plugin dedups
    after its ``[:5]`` slice and does not backfill from the next search
    hits, so a duplicate simply costs it an overlay slot. We reproduce that.
    Backfilling would hand the answer model, and the judge, a 6th-ranked
    memory the agent would never have seen. That is exactly the kind of
    quiet upgrade this adapter must not do. Exact duplicates are rare here
    (RetainDB dedups by content hash on ingest, so a ``_norm`` collision
    needs two rows that differ only in punctuation, case, or only past
    character 320). When it happens, the row count drops to 4, which the
    scorer handles natively (SEH@5 just has one fewer slot to hit).
    """
    start = time.time()
    # See _PLUGIN_QUERY_CONTEXT_SERVER_TOPK. Request what the plugin's
    # paramless query_context makes the server request, so the
    # reinforcement side effect touches the same rows. The ranking of the
    # first top_k stays unaffected.
    fetch_k = max(top_k, _PLUGIN_QUERY_CONTEXT_SERVER_TOPK) if plugin_overlay else top_k
    response = client.search(project=project, query=question_text, top_k=fetch_k, user_id=user_id)
    duration_ms = (time.time() - start) * 1000.0

    results = response.get("results") or []
    if plugin_overlay:
        # ``list((query_result or {}).get("results") or [])[:5]`` -- the
        # plugin's hardcoded 5 becomes the harness top_k (the same value at
        # the default of 5).
        results = results[:top_k]

    retrieved: List[Dict[str, Any]] = []
    seen_norm: List[str] = []
    for result in results:
        scores = result.get("scores") or {}
        raw_text = str(result.get("content", ""))
        item: Dict[str, Any] = {
            "memory": raw_text,
            "created_at": _result_created_at(result),
            "score": result.get("score"),
            "id": result.get("id"),
            "type": result.get("type"),
            "bm25_score": scores.get("bm25"),
            "vector_score": scores.get("vector"),
            "graph_score": scores.get("graph"),
            "retrieval_source": result.get("retrieval_source"),
        }
        if plugin_overlay:
            compacted = _plugin_compact(raw_text)
            norm = _plugin_norm(compacted)
            if not compacted:
                # ``if c and n not in seen`` -- the plugin drops an empty
                # compaction before it can claim an overlay slot.
                if stats is not None:
                    stats["overlay_dropped_empty"] = stats.get("overlay_dropped_empty", 0) + 1
                continue
            if norm in seen_norm:
                if stats is not None:
                    stats["overlay_dropped_duplicate"] = stats.get("overlay_dropped_duplicate", 0) + 1
                continue
            seen_norm.append(norm)
            item["memory"] = compacted
            # Diagnostics only. Record the LENGTH of the row RetainDB stored,
            # never the untruncated text. Keeping the full string here would
            # tempt a future change to feed it to the judge, and the count is
            # all a truncation-impact analysis needs.
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
# Provider binding (the only RetainDB-specific surface the driver sees)
# --------------------------------------------------------------------------
class RetainDBBinding(ProviderBinding):
    memory_system = "retaindb"
    store_id_key = "RetainDB_Project"
    runtime_summary_key = "RetainDB_Runtime_Summary"
    stage_name = "retaindb_answer_generation"
    stage_note = "RetainDB retrieval and question answering"

    def __init__(self, client: RetainDBClient, granularity: str, plugin_overlay: bool = True):
        self.client = client
        self.granularity = granularity
        # Reproduce the Hermes plugin's recall-time overlay compaction (default).
        self.plugin_overlay = plugin_overlay

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        # A unique project is RetainDB's tenancy boundary, giving full per-persona isolation.
        return {
            "store_id": f"mc_{persona_id[-8:]}_{uuid.uuid4().hex[:8]}",
            "user_id": f"user_{persona_id[-8:]}",
            "persona_tag": persona_id[-8:],
            "total_created": 0,
            "total_skipped": 0,
            "total_exchanges_ingested": 0,
            "total_exchanges_dropped_no_user": 0,
        }

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        timestamp = Parse_Session_Timestamp(session_item)
        session_label = str(session_item.get("Session_ID", session_index))

        if self.granularity == "exchange":
            # Plugin-faithful arm: one ingest_session per user/assistant exchange.
            (add_ms, created, skipped,
             exchanges_ingested, exchanges_dropped) = Add_Session_Exchanges_To_RetainDB(
                self.client, ctx["store_id"], ctx["user_id"], session_label,
                dialogue, timestamp,
            )
            ctx["total_created"] += created
            ctx["total_skipped"] += skipped
            ctx["total_exchanges_ingested"] += exchanges_ingested
            ctx["total_exchanges_dropped_no_user"] += exchanges_dropped
            print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
                  f"exchanges_ingested={exchanges_ingested} "
                  f"exchanges_dropped_no_user={exchanges_dropped} "
                  f"created={created} skipped={skipped} ingest_ms={add_ms:.0f}")
            return {
                "Dialogue_Added_To_Memory": created > 0,
                "Dialogue_Message_Count": len(dialogue),
                "Exchanges_Ingested": exchanges_ingested,
                "Exchanges_Dropped_No_User": exchanges_dropped,
                "Memories_Created": created,
                "Memories_Skipped": skipped,
                "Retain_Granularity": self.granularity,
                "Session_Timestamp_Passed": _iso(timestamp),
                "Add_Duration_ms": add_ms,
            }

        add_ms, created, skipped = Add_Session_Dialogue_To_RetainDB(
            self.client, ctx["store_id"], ctx["user_id"], session_label,
            dialogue, timestamp, self.granularity,
        )
        ctx["total_created"] += created
        ctx["total_skipped"] += skipped
        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
              f"created={created} skipped={skipped} ingest_ms={add_ms:.0f}")
        return {
            "Dialogue_Added_To_Memory": created > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Memories_Created": created,
            "Memories_Skipped": skipped,
            "Retain_Granularity": self.granularity,
            "Session_Timestamp_Passed": _iso(timestamp),
            "Add_Duration_ms": add_ms,
        }

    def recall(self, ctx, question_text, top_k):
        return Search_RetainDB_For_Question(
            self.client, ctx["store_id"], ctx["user_id"], question_text, top_k,
            plugin_overlay=self.plugin_overlay,
            stats=ctx.setdefault("overlay_stats", {}),
        )

    def persona_count_extras(self, ctx):
        extras = {
            "Total_Memories_Created": ctx["total_created"],
            "Total_Memories_Skipped": ctx["total_skipped"],
            # Record which retrieval product measured this persona, in the
            # results file itself, so a Results/*.json can never be mistaken
            # for the other arm. The manifest also captures RETAINDB_*.
            "Plugin_Overlay_Compaction": self.plugin_overlay,
            "Plugin_Overlay_Profile_Half": (
                "omitted (unscored + query-independent; see docs/DECISIONS.md)"
                if self.plugin_overlay else "n/a"
            ),
        }
        if self.plugin_overlay:
            # These stats let the report say how much of RetainDB's result
            # comes from the 320-char cut rather than from ranking.
            stats = ctx.get("overlay_stats") or {}
            extras.update({
                "Overlay_Items_Emitted": stats.get("overlay_items", 0),
                "Overlay_Items_Truncated": stats.get("overlay_truncated", 0),
                "Overlay_Chars_Dropped": stats.get("overlay_chars_dropped", 0),
                "Overlay_Items_Dropped_Duplicate": stats.get("overlay_dropped_duplicate", 0),
                "Overlay_Items_Dropped_Empty": stats.get("overlay_dropped_empty", 0),
            })
        if self.granularity == "exchange":
            extras["Total_Exchanges_Ingested"] = ctx["total_exchanges_ingested"]
            extras["Total_Exchanges_Dropped_No_User"] = ctx["total_exchanges_dropped_no_user"]
        return extras


def Generate_User_RetainDB_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    granularity: str,
    profile: str,
    embedding_provider: str,
    embedding_model: Optional[str],
    base_url: Optional[str],
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
    plugin_overlay: bool = True,
) -> bool:
    print(f"[DEBUG] granularity={granularity}  embeddings={embedding_provider}  "
          f"plugin_overlay={plugin_overlay}"
          + ("  (compact 320c + dedup, profile half omitted)" if plugin_overlay
             else "  (RAW rows -- diagnostic arm, NOT plugin-faithful)"))
    server_holder: Dict[str, Any] = {}
    binding_holder: Dict[str, Any] = {}

    def setup():
        client, server = Setup_RetainDB(profile, embedding_provider, embedding_model, base_url)
        server_holder["server"] = server
        binding_holder["binding"].client = client

    def teardown():
        server = server_holder.get("server")
        if server is not None:
            server.close()

    binding = RetainDBBinding(client=None, granularity=granularity, plugin_overlay=plugin_overlay)
    binding_holder["binding"] = binding
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
    parser = argparse.ArgumentParser(description="Run RetainDB evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "retaindb_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "retaindb_results.json"),
        top_k_help="How many recalled memories the answer LLM sees (up to 5 are always stored "
                   "for white-box scoring). NOTE the upstream MemConflict adapters answer "
                   "from top-3, so use --top_k 3 for strict answer-accuracy comparability.",
    )
    parser.add_argument("--retain_granularity", type=str, default="session",
                        choices=["session", "message", "exchange"],
                        help="'session' (default): one ingest/session call per session. "
                             "'message': one /v1/memory add per message. "
                             "'exchange': plugin-faithful (hermes-agent retaindb plugin @ main) "
                             "-- one ingest/session call per user/assistant exchange, both roles "
                             "verbatim sharing one timestamp; lone assistant turns dropped.")
    # This flag is env-driven, not entrypoint-driven. This adapter must not
    # edit benchmark/docker/entrypoint.retaindb.sh, which is shared-harness
    # territory. So the arm is selected through
    # `docker compose run -e RETAINDB_PLUGIN_OVERLAY=0 ...`, which reaches
    # argparse's default. write_manifest.py snapshots every RETAINDB_* var,
    # so the manifest records which arm produced a given Results file.
    parser.add_argument("--plugin_overlay", dest="plugin_overlay", action="store_true",
                        default=_env_flag("RETAINDB_PLUGIN_OVERLAY", True),
                        help="Reproduce the Hermes plugin's recall overlay: per-item whitespace "
                             "collapse + 320-char compaction + normalised dedup, applied to BOTH "
                             "the answer context and the stored Retrieved_Memories (default ON; "
                             "env RETAINDB_PLUGIN_OVERLAY=0 to disable).")
    parser.add_argument("--no_plugin_overlay", dest="plugin_overlay", action="store_false",
                        help="Diagnostic arm: store/answer from RAW search rows (pre-2026-07-21 "
                             "behaviour). NOT plugin-faithful -- never use for a headline number.")
    parser.add_argument("--embedding_provider", type=str,
                        default=os.environ.get("RETAINDB_EMBEDDING_PROVIDER", "hash"),
                        choices=["hash", "local-transformers"],
                        help="RetainDB vector backend. 'hash' (default): zero-dependency deterministic "
                             "embeddings (lexical-ish, fastest). 'local-transformers': Xenova/all-MiniLM-L6-v2 "
                             "semantic embeddings in-process (needs @xenova/transformers; higher recall quality).")
    parser.add_argument("--embedding_model", type=str,
                        default=os.environ.get("RETAINDB_EMBEDDING_MODEL"),
                        help="Override the local-transformers model (default Xenova/all-MiniLM-L6-v2).")
    parser.add_argument("--base_url", type=str, default=os.environ.get("RETAINDB_BASE_URL"),
                        help="Attach to an already-running RetainDB server instead of spawning one.")
    parser.add_argument("--profile", type=str,
                        default=os.environ.get("RETAINDB_PROFILE", f"memconflict_{uuid.uuid4().hex[:8]}"),
                        help="Disposable server profile (data isolation dir). Unique per run by default.")
    args = parser.parse_args()

    # eval_common.run_eval() returns False, not an exception, on a fatal
    # error. It catches internally so per-persona incremental output
    # survives a mid-run crash. This return value was once discarded, so the
    # process always exited 0. A run that died at persona 2 of 30 then
    # looked successful, the entrypoint's `set -e` never fired, and
    # STAGE=all went straight into scoring the partial file. Propagate
    # failure as a nonzero exit, so the entrypoint, and preflight_rows.py as
    # a second check, actually stop the run.
    ok = Generate_User_RetainDB_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        granularity=args.retain_granularity,
        profile=args.profile,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        base_url=args.base_url,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
        plugin_overlay=args.plugin_overlay,
    )
    raise SystemExit(0 if ok else 1)
