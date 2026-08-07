"""MemConflict evaluation adapter for the self-hosted Supermemory memory system.

The shared ``benchmark/eval_common.py`` driver holds the provider-agnostic
pipeline: dataset iteration, dialogue flattening, the answer prompt and answer
LLM call, results-row emission, and compaction. This pipeline is identical by
construction for every provider. This file supplies only the
Supermemory-specific binding: server setup, ingestion (with the async
processing drain), and retrieval.

WHAT IS DIFFERENT ABOUT SUPERMEMORY (vs. Mnemosyne / Hindsight / RetainDB)
-------------------------------------------------------------------------
Self-hosted Supermemory is a hybrid of the other two server-style providers:

  * IT HAS AN INTERNAL LLM (like Hindsight, unlike RetainDB). Ingesting a
    document spends LLM calls: the server summarizes the content, chunks it
    contextually, and **extracts memories** ("facts, updates, contradiction
    resolution, auto-forget"). That extraction model is a separate knob that
    Supermemory exposes. Set it on the SERVER PROCESS through its
    ``OPENAI_*`` vars (fed from ``SUPERMEMORY_LLM_*``). Keep it strictly
    separate from the shared answer+judge model the harness calls. See
    ``_supermemory_server.py`` and docs/DECISIONS.md.

  * INGESTION IS ASYNCHRONOUS. ``POST /v3/documents`` returns instantly with
    ``status:"queued"``. Extraction, embedding, and indexing run in a
    background queue (queued -> extracting -> chunking -> embedding ->
    indexing -> done). A memory becomes searchable only at ``done``. This
    adapter therefore DRAINS the queue after each session's ingest, and
    before it answers that session's questions (this mirrors Hindsight's
    WAIT_CONSOLIDATION). Without the drain, recall races the queue and
    returns nothing.

  * THE PLUGIN HANDS HERMES EXTRACTED MEMORIES. The Hermes ``supermemory``
    memory provider (NousResearch/hermes-agent, and Supermemory's own
    integrations/hermes doc) recalls through ``/v4/search`` with
    ``search_mode`` default ``hybrid`` (memories first, document chunks as
    fallback) and ``max_recall_results`` default 10, then feeds those memory
    strings into the agent's context. Per the project ruling (CLAUDE.md),
    facts are the product under test: this adapter takes the server's
    extracted ``memory`` text verbatim and hands it straight to the answer
    model, with no reshaping toward raw dialogue. If extraction or recall
    costs points, report that as a FINDING about Supermemory. Do not
    engineer it away.

TIME: the adapter passes the dataset's simulated session ``Date`` in each
document's ``metadata.session_date`` (the API allows only primitive metadata
values) and prefers it as the scorer's ``created_at``. Supermemory stamps
each memory's own ``updatedAt`` at extraction wall-clock time. Without our
metadata, recency would instead reflect INGEST ORDER (a caveat shared with
RetainDB).

Isolation is per persona, through a unique ``containerTag`` (Supermemory's
tenancy and user-graph boundary -- "the graph is formed on top of container
tags"), inside one disposable server.
"""

import argparse
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# --- Load the upstream MemConflict Evaluation helpers -----------------------
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

# The shared, provider-agnostic harness modules (eval_common, llm_reasoning,
# and the scorers) live in ../benchmark. Add that path so imports work
# regardless of the launch cwd.
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
    record_provider_retrieval,
)

# Supermemory server manager and REST client. It owns a disposable server process.
from _supermemory_server import (  # noqa: E402
    ServerDiedError,
    SupermemoryClient,
    SupermemoryServer,
)

# Clock-sync core: the single writer of the libfaketime timestamp file. It
# lives in ../benchmark, already on sys.path through _SHARED_HARNESS_DIR above.
# Every call is a no-op unless BENCH_CLOCKSYNC=1 and BENCH_CLOCKSYNC_FILE are
# set.
import clock_sync  # noqa: E402

load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# The Hermes supermemory plugin's default recall budget (`max_recall_results`).
# The plugin merges up to 10 memories into context. This harness fixes the
# shared top-K as the fairness line (every provider answers from 5 memories).
# So this adapter requests the plugin's budget of 10 and slices to top_k,
# exactly as eval_retaindb requests the server default of 10 and then slices.
# See docs/DECISIONS.md.
_PLUGIN_MAX_RECALL_RESULTS = 10

# The Hermes plugin wraps its auto-injected recall in a <supermemory-context>
# block with this exact intro (plugins/memory/supermemory/__init__.py:261-264).
# The FEATURED /v4/profile arm renders the static and dynamic profile facts
# with the SAME headers, intro, and wrapper that the plugin's
# _format_prefetch_context uses (:224-266). The answer model therefore sees
# byte-identical profile framing. The block carries ONLY the static and
# dynamic sections. The profile's search-result items are emitted instead as
# per-item Retrieved_Memories rows, so SEH scores them, and never twice.
_PREFETCH_INTRO = (
    "The following is background context from long-term memory. Use it silently "
    "when relevant. Do not force memories into the conversation."
)


def _render_profile_block(static_facts: List[Any], dynamic_facts: List[Any],
                          max_results: int) -> str:
    """Reproduce the plugin's _format_prefetch_context rendering of the static
    and dynamic profile sections. The search section is omitted here; it
    becomes rows instead.

    This mirrors _deduplicate_recall (static wins over dynamic on an
    exact-string match), then the '## User Profile (Persistent)' and
    '## Recent Context' section layout. Each section is capped at
    max_results (the plugin's max_recall_results). Returns '' when there are
    no profile facts, exactly like the plugin.
    """
    seen = set()
    out_static: List[str] = []
    out_dynamic: List[str] = []
    for fact in static_facts or []:
        if fact and fact not in seen:
            seen.add(fact)
            out_static.append(fact)
    for fact in dynamic_facts or []:
        if fact and fact not in seen:
            seen.add(fact)
            out_dynamic.append(fact)
    out_static = out_static[:max_results]
    out_dynamic = out_dynamic[:max_results]
    if not out_static and not out_dynamic:
        return ""
    sections = []
    if out_static:
        sections.append("## User Profile (Persistent)\n"
                        + "\n".join(f"- {item}" for item in out_static))
    if out_dynamic:
        sections.append("## Recent Context\n"
                        + "\n".join(f"- {item}" for item in out_dynamic))
    body = "\n\n".join(sections)
    return f"<supermemory-context>\n{_PREFETCH_INTRO}\n\n{body}\n</supermemory-context>"


def _env_flag(name: str, default: bool) -> bool:
    """Read an env var as a boolean. Returns ``default`` when unset. Treats
    ``0``, ``false``, ``no``, ``off``, and empty string as false."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _sanitize_tag(text: str) -> str:
    """Lowercase the text and map every non-[a-z0-9_] character to '_', for a
    safe containerTag (this mirrors the Hindsight per-run DB-name
    sanitization). It also folds hyphens, because this code cannot probe the
    real server's tag-charset validation, and persona UUIDs carry hyphens.
    The vendor's own examples use only [a-z0-9_]."""
    return re.sub(r"[^a-z0-9_]", "_", str(text).strip().lower())


def _opt_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw != "" else None
    except ValueError:
        return None


def _year_of(value: Any) -> Optional[int]:
    """Extract the year from a server-stamped timestamp (ISO string, or epoch
    seconds or milliseconds), best effort. Returns None if it cannot parse
    the value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e11:  # epoch milliseconds
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).year
        except (ValueError, OverflowError, OSError):
            return None
    s = str(value).strip()
    # An ISO-8601 string starts with a four-digit year.
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


# The Bun honor probe costs one extraction-LLM call and one throwaway
# document, so it runs once per PROCESS, on the first server boot only. The
# clock-sync arm respawns the server once per session (see
# SupermemoryBinding._respawn_server), and re-probing every boot would add an
# extraction call per session and prove nothing new: the same binary with the
# same LD_PRELOAD env cannot start ignoring libfaketime mid-run.
_FAKETIME_PROBE_DONE = False


def _bun_faketime_probe(client: SupermemoryClient) -> None:
    """Verify that the spawned Bun server honors libfaketime before a
    clock-sync run.

    The child process preloads libfaketime. Stepping the shared timestamp
    file to a sentinel date (2019-05-05) must make the server stamp ingested
    documents with that logical year, not the benchmark's wall-clock time.
    This probe ingests one throwaway document into a DEDICATED probe
    containerTag (namespace isolation is enough to prevent cross-run
    interference), drains it, reads back its server-stamped
    createdAt/updatedAt, and requires year 2019. If this check fails, the
    whole clock-sync premise is void, so the probe aborts loudly.

    This is a no-op unless clock-sync is enabled (BENCH_CLOCKSYNC=1 and the
    file are set), and it runs only on the first server boot of the process."""
    global _FAKETIME_PROBE_DONE
    if not clock_sync.clock_sync_enabled() or _FAKETIME_PROBE_DONE:
        return
    _FAKETIME_PROBE_DONE = True
    probe_tag = f"clocksync_probe_{uuid.uuid4().hex[:8]}"
    print(f"[clocksync] Bun honor probe: ingest under sentinel 2019-05-05 "
          f"tag={probe_tag}", flush=True)
    clock_sync.set_clock(datetime(2019, 5, 5, tzinfo=timezone.utc))
    try:
        add = client.add_document(
            content="User: This is a clock-sync honor probe document.\n"
                    "Assistant: Acknowledged, probe recorded.",
            container_tag=probe_tag, metadata={"probe": "clocksync"},
        )
        doc_id = add.get("id")
        client.wait_for_drain([str(doc_id)] if doc_id else None, timeout=180)
        stamped: Dict[str, Any] = {}
        if doc_id:
            try:
                stamped = client.get_document(str(doc_id)) or {}
            except Exception as exc:  # pragma: no cover
                print(f"[clocksync] probe get_document failed: {exc}", flush=True)
        created_year = _year_of(stamped.get("createdAt"))
        updated_year = _year_of(stamped.get("updatedAt"))
        print(f"[clocksync] probe stamped createdAt={stamped.get('createdAt')} "
              f"(year={created_year}) updatedAt={stamped.get('updatedAt')} "
              f"(year={updated_year})", flush=True)
        if created_year != 2019 and updated_year != 2019:
            raise SystemExit(
                "[clocksync] FATAL: Bun does not honor libfaketime — probe document "
                f"stamped createdAt={stamped.get('createdAt')} "
                f"updatedAt={stamped.get('updatedAt')} (expected year 2019 under the "
                "sentinel clock). The clock-sync arm cannot bend the server's clock, "
                "so its results would be invalid. Aborting.")
        print("[clocksync] Bun honor probe OK (server stamped year 2019)", flush=True)
    finally:
        # Restore the real-time seed. The driver re-steps per session from here.
        clock_sync.seed_real_time()


# --------------------------------------------------------------------------
# Backend setup
# --------------------------------------------------------------------------
def Setup_Supermemory(
    data_dir: str,
    base_url: Optional[str],
    api_key: Optional[str],
    embedding_provider: str,
    embedding_model: Optional[str],
    embedding_dimensions: Optional[str],
    llm_model: Optional[str],
) -> Tuple[SupermemoryClient, Optional[SupermemoryServer]]:
    """Return a Supermemory REST client.

    If the caller provides ``base_url``, this function attaches to an
    already-running server (this also needs ``api_key``). Otherwise it spawns
    a disposable local server and owns it.
    """
    if base_url:
        key = api_key or os.environ.get("SUPERMEMORY_API_KEY")
        if not key:
            raise ValueError(
                "SUPERMEMORY_BASE_URL set but no SUPERMEMORY_API_KEY -- attach mode "
                "needs the server's bearer key."
            )
        client = SupermemoryClient(base_url, key)
        code = client.ping()  # fail fast if the server is unreachable or rejects the key
        if code in (401, 403):
            raise ValueError(
                f"attach to {base_url} rejected the bearer key (HTTP {code}) -- the "
                f"SUPERMEMORY_API_KEY is wrong or stale (e.g. the central server's data "
                f"dir was reset without clearing the published key). Wipe both the data "
                f"and shared volumes together, or supply the correct key.")
        print(f"[supermemory] attached to external server at {base_url} (ping {code})", flush=True)
        return client, None

    server = SupermemoryServer(
        data_dir=data_dir,
        api_key=api_key,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    client = server.start()
    # Clock-sync arm (spawn mode only): the server child preloads libfaketime
    # through _env(). Verify that the Bun binary actually honors it before any
    # real ingest, so a silently-ignored preload fails fast instead of banking
    # a wall-clock run mislabeled as clock-synced. This is a no-op unless
    # BENCH_CLOCKSYNC=1.
    _bun_faketime_probe(client)
    return client, server


def _iso(timestamp: Optional[datetime]) -> Optional[str]:
    return timestamp.isoformat() if timestamp else None


def _format_dialogue_as_document(dialogue_messages: List[Dict[str, Any]]) -> str:
    """Render a session's turns as a plain conversation transcript for ingest.

    Supermemory extracts memories from natural conversational content. This
    function hands it the dialogue verbatim, as ``User: ...\\nAssistant:
    ...``, with no role rewriting and no injected facts. This is the document
    body for the ``session`` granularity.
    """
    lines = []
    for m in dialogue_messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _result_created_at(result: Dict[str, Any]) -> str:
    """Pick the best temporal anchor for a recalled memory, for the scorer's
    created_at slot.

    PLUGIN-FAITHFUL ORDER (2026-07-28): this function prefers the SERVER's
    own ``updatedAt`` or ``createdAt``. This is what the plugin renders
    relative time from (``_format_prefetch_context``,
    external/hermes-agent/plugins/memory/supermemory/__init__.py:244-249).
    This adapter used to prefer a ``session_date`` that it injected into
    metadata, so the ``[created_at]`` line the answer model saw was this
    adapter's value, not the provider's. On smk_sm_clk2, every row reported
    this injected midnight session date, while the same row's server-stamped
    ``updatedAt`` read ...T12:00:00Z. ``updatedAt`` also correctly tracks a
    memory revised in a later session, which a frozen session_date cannot do.
    The metadata keys stay LAST as a fallback for non-clock-synced runs,
    where the server's stamp is real wall-clock time.
    """
    metadata = result.get("metadata") or {}
    for candidate in (
        result.get("updatedAt"),
        result.get("createdAt"),
        metadata.get("session_date"),
        metadata.get("timestamp"),
    ):
        if candidate:
            return str(candidate)
    return "Unknown Time"


def _unit_messages(unit: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Map a group of dialogue turns to the /v4/conversations message shape
    (a list of {role, content} dicts). This is exactly what the plugin's
    ingest_conversation sends (__init__.py:743-767)."""
    return [{"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in unit]


def Add_Session_As_Document(
    client: SupermemoryClient,
    container_tag: str,
    session_label: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
    granularity: str,
    ingest_endpoint: str = "documents",
) -> Tuple[List[str], int]:
    """Ingest one session's dialogue into Supermemory. This call is async;
    the caller must drain it.

    There are two ingest endpoints:

    ``ingest_endpoint='documents'`` (minimal arm, default): the function
    posts each unit as a plain-text transcript through ``/v3/documents``.
    ``ingest_endpoint='conversations'`` (FEATURED arm): the function posts
    each unit as a ``{role, content}`` message list through
    ``/v4/conversations``. This is the exact endpoint the real Hermes plugin
    uses at session end (__init__.py:397-417, :743-771). The server returns
    one document id per conversation, drainable the same way as a
    /v3/documents add.

    Granularity selects the unit size, for both endpoints:
    ``session`` (plugin cadence): one unit is the whole session transcript.
    ``exchange``: one unit is one user/assistant pair (lone assistant turns
    are dropped).
    ``message``: one unit is one message.

    Returns (submitted_document_ids, dropped_count).
    """
    if not dialogue_messages:
        return [], 0
    ts_iso = _iso(timestamp)
    base_meta: Dict[str, Any] = {"session_id": str(session_label)}
    # PLUGIN-FAITHFUL (2026-07-28): the Hermes plugin sends NO per-document
    # date on either ingest path. Neither `documents.add`
    # (external/hermes-agent/plugins/memory/supermemory/__init__.py:319-334)
    # nor `/v4/conversations` (:397-417) sends one. Under clock-sync, the
    # server already stamps createdAt/updatedAt at the logical session date
    # (verified on the probe_reldate_sm rows: updatedAt
    # 2022-11-22T12:00:00Z). Injecting the date here would be redundant, and
    # would post a document no real deployment sends. The RetainDB adapter
    # follows the same rule. WITHOUT clock-sync, the server would stamp real
    # 2026 dates, so the date is still needed in that case.
    if ts_iso and not clock_sync.clock_sync_enabled():
        base_meta["session_date"] = ts_iso

    doc_ids: List[str] = []
    dropped = 0
    use_conversations = ingest_endpoint == "conversations"

    def _submit(unit: List[Dict[str, Any]], idx_label: str) -> None:
        """Submit one unit through the selected endpoint. Record its document id."""
        nonlocal dropped
        try:
            if use_conversations:
                messages = _unit_messages(unit)
                if not any(m["content"].strip() for m in messages):
                    dropped += 1
                    return
                # conversationId mirrors the plugin's session_id. The suffix
                # keeps it unique for finer granularities, so drains never alias.
                conv_id = str(session_label) if idx_label == "full" \
                    else f"{session_label}_{idx_label}"
                resp = client.ingest_conversation(
                    conversation_id=conv_id, messages=messages,
                    container_tag=container_tag,
                    metadata={**base_meta, "type": "full_session"},
                )
            else:
                content = _format_dialogue_as_document(unit)
                if not content.strip():
                    dropped += 1
                    return
                resp = client.add_document(
                    content=content, container_tag=container_tag,
                    metadata=dict(base_meta),
                    custom_id=f"{container_tag}_{session_label}_{idx_label}",
                )
            did = resp.get("id")
            if did:
                doc_ids.append(str(did))
            else:
                dropped += 1
        except ServerDiedError:
            # A dead server drops EVERY remaining unit of this session, so
            # counting one drop and continuing would submit the rest into the
            # void. The caller decides whether the session can be re-submitted.
            raise
        except Exception as e:  # pragma: no cover
            print(f"[DEBUG] {ingest_endpoint} ingest failed tag={container_tag} "
                  f"session={session_label} {idx_label}: {e}")
            dropped += 1

    try:
        if granularity == "message":
            for i, m in enumerate(dialogue_messages):
                _submit([m], f"m{i}")
        elif granularity == "exchange":
            for i, group in enumerate(Pair_Exchange_Turns(dialogue_messages)):
                if not group or group[0].get("role") != "user":
                    dropped += 1  # empty-user gate: drop a lone assistant turn
                    continue
                _submit(group, f"x{i}")
        else:  # 'session'
            _submit(dialogue_messages, "full")
    except ServerDiedError as exc:
        # Hand the caller what the store already accepted. An empty list is
        # the only state in which re-submitting this session is duplicate-safe.
        exc.accepted_doc_ids = list(doc_ids)
        exc.dropped = dropped
        raise

    return doc_ids, dropped


def Search_Supermemory_For_Question(
    client: SupermemoryClient,
    container_tag: str,
    question_text: str,
    top_k: int,
    search_mode: str,
    threshold: Optional[float],
    rerank: bool,
    rewrite_query: bool,
    documents_arm: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Recall memories for one question, mapped to the scorer's memory shape.

    Headline path (``documents_arm=False``): calls ``/v4/search`` in
    ``search_mode`` (default ``hybrid``). Each result carries either
    ``memory`` (an extracted memory) or ``chunk`` (a document fallback in
    hybrid mode) as text, plus ``similarity`` as the score and ``updatedAt``
    as time. This function requests the plugin's recall budget and slices to
    the shared top_k. The mapped ``memory`` text reaches both the answer
    model (through eval_common) and the stored ``Retrieved_Memories``: one
    text, two consumers.

    Diagnostic ``documents`` arm (``documents_arm=True``): calls
    ``/v3/search``, which returns documents with matching chunks. This
    function flattens the result to one row per top chunk.

    THRESHOLD: this function sends ``threshold=None`` to the server as an
    EXPLICIT ``0.0``, never omitted. The vendor default for an omitted
    ``/v4/search`` ``threshold`` field is 0.6 (per the searching-memories
    docs). Omitting the field would silently engage that vendor cutoff.
    Supermemory would then answer from fewer memories than the shared top-K,
    which is exactly the harness asymmetry this default avoids. A caller who
    wants the vendor default sets ``SUPERMEMORY_SEARCH_THRESHOLD=0.6`` (the
    documented arm).
    """
    start = time.time()
    effective_threshold = 0.0 if threshold is None else threshold
    fetch_k = max(top_k, _PLUGIN_MAX_RECALL_RESULTS)

    retrieved: List[Dict[str, Any]] = []
    if documents_arm:
        response = client.search_documents(
            query=question_text, container_tag=container_tag, limit=fetch_k,
            chunk_threshold=effective_threshold, rerank=rerank, rewrite_query=rewrite_query,
        )
        for doc in response.get("results", []):
            doc_score = doc.get("score")
            doc_meta = doc.get("metadata") or {}
            # Prefer the SERVER's own stamp, like the memories path since
            # 2026-07-28. Under clock-sync it already IS the logical session
            # date, and it is what the plugin renders relative time from.
            # This adapter's injected metadata stays last, for non-clock-synced
            # runs only.
            doc_created = str(doc.get("createdAt") or doc.get("updatedAt")
                              or doc_meta.get("session_date") or "Unknown Time")
            for chunk in (doc.get("chunks") or []):
                retrieved.append({
                    "memory": str(chunk.get("content", "")),
                    "created_at": doc_created,
                    "score": chunk.get("score", doc_score),
                    "id": doc.get("documentId"),
                    "type": "document_chunk",
                    "document_score": doc_score,
                    "is_relevant": chunk.get("isRelevant"),
                })
    else:
        response = client.search_memories(
            query=question_text, container_tag=container_tag, limit=fetch_k,
            threshold=effective_threshold, rerank=rerank, rewrite_query=rewrite_query,
            search_mode=search_mode,
        )
        for r in response.get("results", []):
            # Memory results carry `memory`. Hybrid document-fallback results
            # carry `chunk`. Take whichever field exists as the memory text.
            text = r.get("memory")
            result_type = "memory"
            if text is None:
                text = r.get("chunk")
                result_type = "chunk"
            retrieved.append({
                "memory": str(text or ""),
                "created_at": _result_created_at(r),
                "score": r.get("similarity"),
                "id": r.get("id"),
                "type": result_type,
                "version": r.get("version"),
            })

    duration_ms = (time.time() - start) * 1000.0
    # Diagnostic capture BEFORE the shared-top_k slice below. Raw is the one
    # /v4/search (or /v3/search) response this question was answered from.
    # Ranked is all ``fetch_k`` mapped rows. fetch_k is the plugin's own
    # max_recall_results (10), which this arm already requests and does not
    # raise for the capture. So a Supermemory file supports a depth curve up
    # to 10, and no further.
    record_provider_retrieval(diag, raw=response, ranked=retrieved)
    # Slice to the shared top_k (the fairness line). The answer model and
    # scorer see K.
    return retrieved[:top_k], duration_ms


def Profile_Recall_For_Question(
    client: SupermemoryClient,
    container_tag: str,
    question_text: str,
    top_k: int,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float, str, int, int]:
    """FEATURED recall path: calls /v4/profile, the endpoint the plugin
    auto-injects through prefetch -> get_profile (__init__.py:714-729,
    :356-379).

    This function splits the profile response into the two things the
    harness needs:

      * ``searchResults.results`` becomes per-item Retrieved_Memories rows
        (memory text, created_at, similarity score), exactly like the
        /v4/search arm, so SEH@K and log-rank score them. The function
        requests the plugin's ``max_recall_results`` (10) budget, then
        slices to the shared top_k. This is the same "request 10, slice to
        shared top-K" fairness line that CLAUDE.md fixes for every provider.
      * ``profile.static`` and ``profile.dynamic`` become the profile TEXT
        block. ``_render_profile_block`` renders it exactly as the plugin's
        _format_prefetch_context would, and the function returns it for
        injection into the answer context and for the auditable
        ``Profile_Block`` sidecar. These profile facts are the plugin's
        always-on injected context. The function does not emit them as
        scored rows, because the plugin also keeps them separate from the
        search results.

    profile_frequency gating: the plugin includes the static and dynamic
    facts only on turn <=1, or every profile_frequency=50 turns (:719). In
    this harness, every question is answered as an independent, fresh
    "turn 1" conversation (there is no multi-turn agent loop that carries
    _turn_count forward). The faithful equivalent is therefore turn_count<=1,
    so this function includes the profile block on EVERY question. See the
    report.

    Returns (retrieved_rows[:top_k], duration_ms, profile_block, static_count,
    dynamic_count).
    """
    start = time.time()
    # The plugin passes query[:200] to get_profile (:718). This call matches it.
    profile = client.get_profile(query=(question_text or "")[:200],
                                 container_tag=container_tag)
    static = profile.get("static") or []
    dynamic = profile.get("dynamic") or []
    search_results = profile.get("search_results") or []

    retrieved: List[Dict[str, Any]] = []
    for r in search_results[:_PLUGIN_MAX_RECALL_RESULTS]:
        if not isinstance(r, dict):
            continue
        text = r.get("memory")
        result_type = "memory"
        if text is None:
            text = r.get("chunk")
            result_type = "chunk"
        retrieved.append({
            "memory": str(text or ""),
            "created_at": _result_created_at(r),
            "score": r.get("similarity"),
            "id": r.get("id"),
            "type": result_type,
        })

    profile_block = _render_profile_block(static, dynamic, _PLUGIN_MAX_RECALL_RESULTS)
    duration_ms = (time.time() - start) * 1000.0
    # Diagnostic capture BEFORE the shared-top_k slice. Raw is the WHOLE
    # /v4/profile response from one call. It is the provider's complete
    # answer to this question: searchResults plus the static and dynamic
    # profile facts, so the profile half is auditable too. Ranked is the
    # searchResults rows at the plugin's own max_recall_results (10) budget,
    # which this arm already requests and does not widen. The profile facts
    # are deliberately NOT ranked rows, because the plugin keeps them
    # separate and never scores them. They stay only in the raw capture and
    # in Profile_Block.
    record_provider_retrieval(diag, raw=profile, ranked=retrieved)
    # Slice rows to the shared top_k (the fairness line). The profile block
    # is the plugin's separate always-on injection, and is not part of the
    # top-K rows.
    return retrieved[:top_k], duration_ms, profile_block, len(static), len(dynamic)


# --------------------------------------------------------------------------
# Provider binding (the only Supermemory-specific surface the driver sees)
# --------------------------------------------------------------------------
class SupermemoryBinding(ProviderBinding):
    memory_system = "supermemory"
    store_id_key = "Supermemory_Container_Tag"
    runtime_summary_key = "Supermemory_Runtime_Summary"
    stage_name = "supermemory_answer_generation"
    stage_note = "Supermemory retrieval and question answering"

    def __init__(
        self,
        client: Optional[SupermemoryClient],
        granularity: str,
        search_mode: str,
        threshold: Optional[float],
        rerank: bool,
        rewrite_query: bool,
        documents_arm: bool,
        drain_timeout: float,
        container_namespace: Optional[str] = None,
        strict_quality: bool = False,
        ingest_endpoint: str = "documents",
        recall_endpoint: str = "search",
    ):
        self.client = client
        # The spawned server, set by the driver's setup() hook. It stays None
        # in attach mode (SUPERMEMORY_BASE_URL, the shared central server),
        # where this process does not own the process and cannot respawn it.
        self.server: Optional[SupermemoryServer] = None
        # Respawn the spawned server once per session. This applies ONLY to
        # clock-sync arms, and _respawn_server carries the reason. The env var
        # turns it off; it never turns it on without clock-sync. An EMPTY
        # value counts as unset here, unlike _env_flag: an empty compose var
        # must not silently disable the OOM mitigation.
        _respawn = os.environ.get("SUPERMEMORY_RESPAWN_PER_SESSION", "").strip().lower()
        self.respawn_per_session = (
            clock_sync.clock_sync_enabled()
            and _respawn not in ("0", "false", "no", "off")
        )
        self.granularity = granularity
        self.search_mode = search_mode
        self.threshold = threshold
        self.rerank = rerank
        self.rewrite_query = rewrite_query
        self.documents_arm = documents_arm
        self.drain_timeout = drain_timeout
        self.strict_quality = strict_quality
        # FEATURED plugin-faithful path: ingest through /v4/conversations,
        # recall through /v4/profile (the endpoints the real Hermes plugin
        # uses). 'documents' plus 'search' is the minimal arm (unchanged
        # /v3/documents plus /v4/search).
        self.ingest_endpoint = ingest_endpoint
        self.recall_endpoint = recall_endpoint
        # When recall goes through /v4/profile, the static and dynamic
        # profile facts are the plugin's separate always-injected context
        # block. This wires the shared driver's extra_answer_context hook,
        # so that block is appended to the answer context after the raw
        # retrieved-memory rows (SEH then scores only the rows), and its
        # provenance fields (Profile_Block / counts) land on the question row.
        if recall_endpoint == "profile":
            self.extra_answer_context = self._profile_extra_context
        # Per-run containerTag namespace, for a SHARED central server that
        # many shards attach to (the analog of Hindsight's per-run database).
        # When set, tags are deterministic and run-scoped (`<ns>_p<persona>`),
        # so different runs never collide on the one shared store, and a
        # run's data stays reclaimable by tag prefix. When unset (standalone
        # or spawn mode), a uuid keeps tags globally unique. The namespace is
        # sanitized to Supermemory-safe tag characters.
        self.container_namespace = _sanitize_tag(container_namespace) if container_namespace else None

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        # containerTag is Supermemory's tenancy and user-graph boundary. It is
        # namespaced and deterministic under a shared server, and uuid-unique
        # when standalone.
        if self.container_namespace:
            # The full, sanitized persona id guarantees a unique tag per
            # persona, even across shards writing to the one shared store.
            store_id = f"{self.container_namespace}_p{_sanitize_tag(persona_id)}"
        else:
            store_id = f"mc_{persona_id[-8:]}_{uuid.uuid4().hex[:8]}"
        return {
            "store_id": store_id,
            "persona_tag": persona_id[-8:],
            "total_documents_submitted": 0,
            "total_documents_dropped": 0,
            "total_drain_failed": 0,
            "total_drain_timeouts": 0,
            "total_drain_dropped": 0,
            "total_drain_wait_s": 0.0,
            "total_server_respawns": 0,
            "total_session_recoveries": 0,
        }

    def _respawn_server(self, ctx, session_index, timestamp) -> None:
        """Restart the spawned server on the SAME data dir, once per session.

        WHY: node-cron v4 inside the 0.0.5 server replays "missed executions"
        at its SECOND heartbeat, which lands at least 30 REAL minutes after
        boot. The replay is a synchronous while-loop that walks one 30-minute
        slot per unit of fake time traveled since the previous heartbeat, and
        allocates ~0.46 MB per slot with GC starved. Six crons are registered
        and the binary exposes no disable knob. Under BENCH_CLOCKSYNC=1 the
        clock steps forward to each session's date, so one persona's ~3-year
        span is ~163k slots, ~73 GB RSS, and the host OOMs. This killed all
        four clock-sync runs (docs/TROUBLESHOOTING.md, Provider: Supermemory).

        The shared driver steps the clock BEFORE it calls ingest_session
        (benchmark/eval_common.py:761), so a server booted here starts already
        at the session's logical date. It therefore never observes a forward
        jump during its life, the replay loop never gets a span to walk, and a
        server almost never survives long enough for heartbeat #2 to fire.

        This runs for EVERY session, including the first. The first boot
        happens in setup(), under real wall time (the Bun honor probe restores
        it), so without an unconditional respawn session 1 would be served by a
        process that saw a multi-year clock jump. One code path is also easier
        to verify than a session-1 special case, and a boot costs seconds.
        """
        if not self.respawn_per_session or self.server is None:
            return
        self._boot_server(ctx, "session_boundary", session_index, timestamp)

    def _boot_server(self, ctx, reason, session_index, timestamp) -> None:
        """Restart the spawned server on the SAME data dir and rebind the
        client, with the boot retry below. Two callers: the per-session
        clock-sync respawn above, and the crash recovery in ingest_session.
        Both need an identical boot; only the reason differs."""
        t0 = time.time()
        # remove_data=False: the store must survive the restart. Only the
        # process is disposable.
        self.server.close(remove_data=False)
        if self.client is not None:
            self.client.close()  # its pooled sockets point at the dead port
        # start() returns a client bound to THIS boot's port and bearer key.
        # Rebinding self.client is what refreshes recall too: ingest_session
        # and recall() both read self.client, and nothing else holds one.
        # Boot retry: across ~1,600 respawns in a full 30-persona run, the
        # 0.0.5 binary intermittently dies at startup while opening the
        # encrypted local storage ("access to a null reference ...
        # getWasmTableEntry", v4minc2 persona 4, boot 11 of 54). The store is
        # intact and the next boot succeeds, so a failed boot retries instead
        # of failing the persona.
        last_exc: Optional[BaseException] = None
        for attempt in range(1, 4):
            try:
                self.client = self.server.start()
                break
            except RuntimeError as exc:
                last_exc = exc
                self.server.close(remove_data=False)
                print(f"[supermemory] server boot attempt {attempt}/3 failed "
                      f"({exc}); retry in {3 * attempt}s", flush=True)
                time.sleep(3 * attempt)
        else:
            raise last_exc
        ctx["total_server_respawns"] += 1
        fake_date = timestamp.date().isoformat() if timestamp else "unparsed"
        print(f"[supermemory] server respawn reason={reason} fake_date={fake_date} "
              f"session_index={session_index} boot_elapsed_s={time.time() - t0:.1f} "
              f"boot={self.server.boot_count} port={self.server.port}", flush=True)

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        timestamp = Parse_Session_Timestamp(session_item)
        session_label = str(session_item.get("Session_ID", session_index))
        # Before any document of this session is posted, and after the driver
        # has stepped the clock to this session's date.
        self._respawn_server(ctx, session_index, timestamp)
        t0 = time.time()

        # SESSION RECOVERY, submission phase only. The 0.0.5 binary segfaults
        # in its PGlite WASM load path at ~0.59% per boot, so a session's POST
        # can hit a corpse. Re-submitting is duplicate-safe ONLY while the
        # store has accepted no document of this session: a document already
        # accepted is already extracting, and posting it again would duplicate
        # its memories and change what the run measures. So this retries only
        # when accepted_doc_ids is empty, and only once. A second death in the
        # same session raises and the persona aborts.
        recovered = False
        while True:
            try:
                doc_ids, dropped = Add_Session_As_Document(
                    self.client, ctx["store_id"], session_label, dialogue, timestamp,
                    self.granularity, ingest_endpoint=self.ingest_endpoint,
                )
                break
            except ServerDiedError as exc:
                accepted = list(getattr(exc, "accepted_doc_ids", None) or [])
                if accepted or recovered or self.server is None:
                    raise
                recovered = True
                ctx["total_session_recoveries"] += 1
                print(f"[supermemory] server died during submission, persona "
                      f"{ctx['persona_tag']} session {session_label}: {exc}; "
                      f"nothing accepted yet, respawning and re-submitting once",
                      flush=True)
                self._boot_server(ctx, "server_died_submit", session_index, timestamp)

        # DRAIN: block until the async pipeline has processed THESE documents
        # (per-document polling), so the session's questions recall the
        # freshly-extracted memories. Skip the drain entirely when nothing
        # was submitted. Otherwise an empty doc_ids would fall back to the
        # ACCOUNT-GLOBAL processing count, which on a shared central server
        # includes OTHER shards' pending documents, and could block up to
        # drain_timeout for a session that ingested nothing.
        if doc_ids:
            try:
                drain = self.client.wait_for_drain(
                    doc_ids=doc_ids, timeout=self.drain_timeout,
                )
            except ServerDiedError as exc:
                # The store ALREADY accepted these documents, so this session
                # must not be re-submitted: the accepted copies may be half
                # extracted, and a second copy would duplicate memories. Record
                # the death as an unfinished drain and let the strict-quality
                # guard below abort the persona -- now in seconds rather than
                # after the transport-retry budget (~15 min in v4minc2).
                print(f"[supermemory] server died during drain, persona "
                      f"{ctx['persona_tag']} session {session_label}: {exc}; "
                      f"{len(doc_ids)} document(s) already accepted, NOT "
                      f"re-submitting", flush=True)
                drain = {"drained": False, "failed": [], "timed_out": True,
                         "server_died": True, "remaining": len(doc_ids),
                         "elapsed_s": round(time.time() - t0, 2), "polls": 0}
        else:
            drain = {"drained": True, "failed": [], "timed_out": False,
                     "elapsed_s": 0.0, "polls": 0}
        add_ms = (time.time() - t0) * 1000.0

        # STRICT-QUALITY guard (opt-in, for headline runs): a drain timeout, a
        # failed extraction, or a DROPPED document means this session's
        # memories are missing or incomplete, so answering now would silently
        # mis-measure the run. `dropped` is checked here too: a document that
        # never got a document id (empty content, a failed POST) never enters
        # `doc_ids`, so it never reaches wait_for_drain, and an all-dropped
        # session short-circuits to `drained=True` at the `else` branch above
        # with nothing actually verified (docs/TROUBLESHOOTING.md, Provider: Supermemory).
        # This mirrors Hindsight's STRICT_QUALITY_RUN: abort the shard with a
        # nonzero exit (run_eval catches it and exits 1) rather than bank a
        # degraded run. Off by default, so smokes can tolerate a transient
        # timeout.
        if self.strict_quality and (drain.get("timed_out") or drain.get("failed") or dropped):
            raise RuntimeError(
                f"STRICT_QUALITY: persona {ctx['persona_tag']} session {session_label} "
                f"ingest degraded (timed_out={drain.get('timed_out')} "
                f"failed={len(drain.get('failed', []))} dropped={dropped}); aborting "
                f"rather than answering against missing memories. Set "
                f"SUPERMEMORY_STRICT_QUALITY=0 to downgrade to a warning.")

        # Not strict: the session keeps its incomplete memories, exactly as a
        # tolerated drain timeout does. The process still has to be replaced,
        # or every later request in this persona would raise ServerDiedError.
        if drain.get("server_died") and self.server is not None:
            self._boot_server(ctx, "server_died_drain", session_index, timestamp)

        ctx["total_documents_submitted"] += len(doc_ids)
        ctx["total_documents_dropped"] += dropped
        if drain.get("failed"):
            ctx["total_drain_failed"] += len(drain["failed"])
        if drain.get("timed_out"):
            ctx["total_drain_timeouts"] += 1
        if dropped:
            ctx["total_drain_dropped"] += dropped
        ctx["total_drain_wait_s"] += float(drain.get("elapsed_s", 0.0) or 0.0)

        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
              f"docs_submitted={len(doc_ids)} dropped={dropped} "
              f"drained={drain.get('drained')} drain_s={drain.get('elapsed_s')} "
              f"drain_failed={len(drain.get('failed', []))} "
              f"timed_out={drain.get('timed_out')} ingest_ms={add_ms:.0f}")
        return {
            "Dialogue_Added_To_Memory": len(doc_ids) > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Documents_Submitted": len(doc_ids),
            "Documents_Dropped": dropped,
            "Retain_Granularity": self.granularity,
            "Ingest_Drained": bool(drain.get("drained")),
            "Ingest_Drain_Wait_s": drain.get("elapsed_s"),
            "Ingest_Drain_Failed": len(drain.get("failed", [])),
            "Ingest_Drain_Timed_Out": bool(drain.get("timed_out")),
            "Session_Timestamp_Passed": _iso(timestamp),
            "Add_Duration_ms": add_ms,
        }

    def recall(self, ctx, question_text, top_k):
        if self.recall_endpoint == "profile":
            retrieved, dur_ms, profile_block, n_static, n_dynamic = \
                Profile_Recall_For_Question(
                    self.client, ctx["store_id"], question_text, top_k, diag=ctx,
                )
            # Stash the profile block for _profile_extra_context, which the
            # shared driver calls next for this question. recall and
            # extra_answer_context are two separate driver calls, so ctx
            # carries the handoff.
            ctx["_profile_block"] = profile_block
            ctx["_profile_static_count"] = n_static
            ctx["_profile_dynamic_count"] = n_dynamic
            return retrieved, dur_ms
        return Search_Supermemory_For_Question(
            self.client, ctx["store_id"], question_text, top_k,
            search_mode=self.search_mode, threshold=self.threshold,
            rerank=self.rerank, rewrite_query=self.rewrite_query,
            documents_arm=self.documents_arm, diag=ctx,
        )

    def _profile_extra_context(self, ctx, question_text):
        """Return (profile_block, provenance_fields) for the shared driver.
        The driver appends the block to the answer context. It records the
        fields on the question row (Profile_Block survives compaction
        through the driver's passthrough list), so the featured run stays
        auditable."""
        del question_text  # recall() already fetched the profile. ctx carries it.
        block = ctx.get("_profile_block", "") or ""
        fields = {
            "Profile_Block": block,
            "Profile_Static_Count": ctx.get("_profile_static_count", 0),
            "Profile_Dynamic_Count": ctx.get("_profile_dynamic_count", 0),
        }
        return block, fields

    def persona_count_extras(self, ctx):
        return {
            "Total_Documents_Submitted": ctx["total_documents_submitted"],
            "Total_Documents_Dropped": ctx["total_documents_dropped"],
            "Total_Drain_Failed": ctx["total_drain_failed"],
            "Total_Drain_Timeouts": ctx["total_drain_timeouts"],
            "Total_Drain_Dropped": ctx["total_drain_dropped"],
            "Total_Drain_Wait_s": round(ctx["total_drain_wait_s"], 2),
            # Clock-sync arms only. 0 means the server ran for the whole
            # persona, which is the wall-clock configuration.
            "Total_Server_Respawns": ctx["total_server_respawns"],
            # Sessions re-submitted after the server died with NOTHING of that
            # session accepted. Each one also counts a respawn above. A
            # nonzero value means the vendor's PGlite WASM crash hit this
            # persona and the session was re-ingested exactly once.
            "Total_Session_Recoveries": ctx["total_session_recoveries"],
            # Records which ingest and retrieval combination measured this
            # persona, directly in the results file, so a Results/*.json can
            # never be mistaken for another arm. (The manifest also captures
            # SUPERMEMORY_*.)
            "Ingest_Endpoint": ("/v4/conversations" if self.ingest_endpoint == "conversations"
                                else "/v3/documents"),
            "Recall_Endpoint": ("/v4/profile" if self.recall_endpoint == "profile"
                                else ("documents(/v3)" if self.documents_arm else "/v4/search")),
            "Search_Mode": ("documents(/v3)" if self.documents_arm else self.search_mode),
            "Retain_Granularity": self.granularity,
            "Search_Threshold": self.threshold,
            "Search_Rerank": self.rerank,
            "Search_Rewrite_Query": self.rewrite_query,
        }


def Generate_User_Supermemory_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    granularity: str,
    search_mode: str,
    threshold: Optional[float],
    rerank: bool,
    rewrite_query: bool,
    documents_arm: bool,
    drain_timeout: float,
    base_url: Optional[str],
    api_key: Optional[str],
    data_dir: str,
    embedding_provider: str,
    embedding_model: Optional[str],
    embedding_dimensions: Optional[str],
    llm_model: Optional[str],
    container_namespace: Optional[str],
    strict_quality: bool,
    ingest_endpoint: str,
    recall_endpoint: str,
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
) -> bool:
    print(f"[DEBUG] granularity={granularity}  "
          f"ingest_endpoint={'/v4/conversations' if ingest_endpoint == 'conversations' else '/v3/documents'}  "
          f"recall_endpoint={'/v4/profile' if recall_endpoint == 'profile' else ('documents(/v3)' if documents_arm else '/v4/search')}  "
          f"search_mode={'documents(/v3)' if documents_arm else search_mode}  "
          f"embeddings={embedding_provider}  extraction_model={llm_model or '(server default)'}  "
          f"threshold={threshold} rerank={rerank} rewrite_query={rewrite_query} "
          f"drain_timeout={drain_timeout}s")
    server_holder: Dict[str, Any] = {}
    binding_holder: Dict[str, Any] = {}

    def setup():
        client, server = Setup_Supermemory(
            data_dir, base_url, api_key, embedding_provider, embedding_model,
            embedding_dimensions, llm_model,
        )
        server_holder["server"] = server
        bound = binding_holder["binding"]
        bound.client = client
        bound.server = server
        if bound.respawn_per_session and server is None:
            # Attach mode: another process owns the server, so this shard
            # cannot restart it. The node-cron replay that _respawn_server
            # avoids still applies to that shared server.
            print("[clocksync] WARNING: per-session respawn requested but this "
                  "process attached to an external server (SUPERMEMORY_BASE_URL); "
                  "the node-cron missed-execution replay is NOT mitigated.",
                  flush=True)

    def teardown():
        server = server_holder.get("server")
        if server is not None:
            server.close()

    binding = SupermemoryBinding(
        client=None, granularity=granularity, search_mode=search_mode,
        threshold=threshold, rerank=rerank, rewrite_query=rewrite_query,
        documents_arm=documents_arm, drain_timeout=drain_timeout,
        container_namespace=container_namespace, strict_quality=strict_quality,
        ingest_endpoint=ingest_endpoint, recall_endpoint=recall_endpoint,
    )
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
    parser = argparse.ArgumentParser(description="Run Supermemory evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "supermemory_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "supermemory_results.json"),
        top_k_help="How many recalled memories the answer LLM sees (up to 5 stored for "
                   "white-box scoring). Upstream MemConflict answers from top-3.",
        max_sessions_help="Cap sessions ingested per persona (default: all). Small value for smoke.",
    )
    parser.add_argument("--retain_granularity", type=str,
                        default=os.environ.get("SUPERMEMORY_RETAIN_GRANULARITY", "session"),
                        choices=["session", "exchange", "message"],
                        help="'session' (default): one /v3/documents add per session (full-session "
                             "ingest, the plugin's session-end path). 'exchange': one add per "
                             "user/assistant exchange (per-turn capture). 'message': one add per message.")
    parser.add_argument("--search_mode", type=str,
                        default=os.environ.get("SUPERMEMORY_SEARCH_MODE", "hybrid"),
                        choices=["hybrid", "memories"],
                        help="/v4/search mode. 'hybrid' (default, the Hermes plugin default): memories "
                             "first, document chunks as fallback. 'memories': memory entries only.")
    parser.add_argument("--documents_arm", dest="documents_arm", action="store_true",
                        default=_env_flag("SUPERMEMORY_DOCUMENTS_ARM", False),
                        help="Diagnostic arm: recall via /v3/search (documents+chunks) instead of "
                             "/v4/search (memories). NOT the plugin path; never a headline number.")
    parser.add_argument("--ingest_endpoint", type=str,
                        default=os.environ.get("SUPERMEMORY_INGEST_ENDPOINT", "documents"),
                        choices=["documents", "conversations"],
                        help="FEATURED plugin-faithful ingest: 'conversations' posts each session as a "
                             "{role,content} message list to /v4/conversations (the endpoint the real "
                             "Hermes plugin uses at session end, __init__.py:397-417). 'documents' "
                             "(default, minimal arm) posts a plain transcript to /v3/documents.")
    parser.add_argument("--recall_endpoint", type=str,
                        default=os.environ.get("SUPERMEMORY_RECALL_ENDPOINT", "search"),
                        choices=["search", "profile"],
                        help="FEATURED plugin-faithful recall: 'profile' calls /v4/profile (the plugin's "
                             "auto-injected prefetch path, __init__.py:714-729) -- its searchResults "
                             "become the scored Retrieved_Memories rows and its static/dynamic profile "
                             "facts are injected into the answer context + recorded in Profile_Block. "
                             "'search' (default, minimal arm) calls /v4/search. Mutually informative with "
                             "--documents_arm (which forces /v3/search and overrides profile/search).")
    def _opt_float_arg(v):
        return float(v) if v not in (None, "", "none") else None
    parser.add_argument("--search_threshold", type=_opt_float_arg,
                        default=_opt_float_env("SUPERMEMORY_SEARCH_THRESHOLD"),
                        help="Minimum similarity for a returned memory. DEFAULT (unset) sends an "
                             "EXPLICIT 0.0 (not omitted) so the server returns the top-K by "
                             "similarity, keeping Supermemory's K comparable to every other "
                             "provider's shared top-K. (Omitting the field would instead engage the "
                             "vendor default 0.6 and drop sub-0.6 memories, so we send 0.0 rather "
                             "than rely on omission.) Set e.g. 0.6 to reproduce the vendor default.")
    parser.add_argument("--rerank", dest="rerank", action="store_true",
                        default=_env_flag("SUPERMEMORY_RERANK", False),
                        help="Ask /v4|/v3 search to rerank results (slower, more accurate).")
    parser.add_argument("--rewrite_query", dest="rewrite_query", action="store_true",
                        default=_env_flag("SUPERMEMORY_REWRITE_QUERY", False),
                        help="Ask search to rewrite the query before embedding (+latency).")
    parser.add_argument("--drain_timeout", type=float,
                        default=float(os.environ.get("SUPERMEMORY_DRAIN_TIMEOUT", "600")),
                        help="Max seconds to wait for a session's async ingest to finish before "
                             "answering its questions (default 600).")
    parser.add_argument("--base_url", type=str, default=os.environ.get("SUPERMEMORY_BASE_URL"),
                        help="Attach to an already-running Supermemory server instead of spawning one "
                             "(needs --api_key / SUPERMEMORY_API_KEY).")
    parser.add_argument("--api_key", type=str, default=os.environ.get("SUPERMEMORY_API_KEY"),
                        help="Bearer key for attach mode, or to preset the spawned server's key.")
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get("SUPERMEMORY_DATA_DIR",
                                               os.path.join(CURRENT_DIR, ".supermemory_runs",
                                                            f"run_{uuid.uuid4().hex[:8]}")),
                        help="Disposable server data dir (isolation). Unique per run by default.")
    parser.add_argument("--embedding_provider", type=str,
                        default=os.environ.get("SUPERMEMORY_EMBEDDING_PROVIDER", "local"),
                        help="Supermemory embedding backend: 'local' (default, Xenova/bge-base-en-v1.5, "
                             "no key), 'openai', 'gemini', or an OpenAI-compatible remote.")
    parser.add_argument("--embedding_model", type=str,
                        default=os.environ.get("SUPERMEMORY_EMBEDDING_MODEL"),
                        help="Override the embedding model id for the chosen provider.")
    parser.add_argument("--embedding_dimensions", type=str,
                        default=os.environ.get("SUPERMEMORY_EMBEDDING_DIMENSIONS"),
                        help="Vector size; must match the embedding model.")
    parser.add_argument("--llm_model", type=str,
                        default=os.environ.get("SUPERMEMORY_LLM_MODEL"),
                        help="Supermemory's INTERNAL extraction model (server-side OPENAI_MODEL). "
                             "Separate from the harness answer/judge model. Falls back to the "
                             "server default when unset.")
    parser.add_argument("--container_namespace", type=str,
                        default=os.environ.get("SUPERMEMORY_CONTAINER_NAMESPACE"),
                        help="Per-run containerTag namespace for a SHARED central server (analog of "
                             "Hindsight's per-run database). Set -> deterministic run-scoped tags "
                             "`<ns>_p<persona>`; unset -> uuid-unique standalone tags. Sharded runs "
                             "pass their RUN_TAG so each run isolates on the one shared store.")
    parser.add_argument("--strict_quality", action="store_true",
                        default=_env_flag("SUPERMEMORY_STRICT_QUALITY", False),
                        help="Headline-run switch (mirrors Hindsight's STRICT_QUALITY_RUN): abort "
                             "nonzero on a session's ingest drain TIMEOUT or extraction FAILURE "
                             "rather than silently answering against missing memories. Off by "
                             "default (env SUPERMEMORY_STRICT_QUALITY=1 to enable).")
    args = parser.parse_args()

    # run_eval() returns False on a fatal error. It catches errors internally,
    # so per-persona incremental output survives a mid-run crash. This
    # propagates as a nonzero exit, so the entrypoint's `set -e` stops the
    # run instead of scoring a partial file.
    ok = Generate_User_Supermemory_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        granularity=args.retain_granularity,
        search_mode=args.search_mode,
        threshold=args.search_threshold,
        rerank=args.rerank,
        rewrite_query=args.rewrite_query,
        documents_arm=args.documents_arm,
        drain_timeout=args.drain_timeout,
        base_url=args.base_url,
        api_key=args.api_key,
        data_dir=os.path.abspath(args.data_dir),
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        llm_model=args.llm_model,
        container_namespace=args.container_namespace,
        strict_quality=args.strict_quality,
        ingest_endpoint=args.ingest_endpoint,
        recall_endpoint=args.recall_endpoint,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )
    raise SystemExit(0 if ok else 1)
