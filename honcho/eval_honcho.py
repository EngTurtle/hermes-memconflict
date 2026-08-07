"""MemConflict evaluation adapter for the Honcho memory system.

The shared ``benchmark/eval_common.py`` driver runs the provider-agnostic
pipeline: dataset iteration, dialogue flattening, the answer prompt and answer
LLM call, results-row emission, and compaction. This file supplies only the
Honcho-specific binding: workspace setup, ingestion, drain, and recall.

WHAT IS DIFFERENT ABOUT Honcho (vs. mem0 / Hindsight / Supermemory)
-------------------------------------------------------------------
Honcho does not return a ranked list of memories. It returns a **peer model**.
The Hermes ``honcho`` plugin injects a markdown block assembled from up to
five named sections, and that block IS the recall product under test (project
ruling 3, CLAUDE.md):

  1. the session summary (``session.context(summary=True)``),
  2. the user's working REPRESENTATION, seen from the AI peer's perspective
     (``ai_peer.context(target=user_peer, search_query=<latest user message>)``),
  3. the user PEER CARD from the same call,
  4. the AI peer's own representation and identity card,
  5. a DIALECTIC answer: an LLM call on Honcho's own backend, over the target
     peer's full representation (``ai_peer.chat(query, target=user_peer,
     reasoning_level=...)``), clipped to 600 characters.

Sections 1-4 are layer 1 ("base"); section 5 is layer 2 ("dialectic"). None of
them carries a similarity score or a per-item timestamp, so every returned
item reports ``score: None`` and ``created_at: "Unknown Time"``. That is not a
gap in the adapter. Honcho has no per-memory score to report. For these arms
the binding sets ``plugin_native_recall=True``, so the harness hands the
answer model every section, in plugin injection order, instead of a top-K
slice.

Two further arms return a RANKED LIST instead, so they take the shared top-K
like every other provider, and ``plugin_native_recall`` is False for them:

  * ``conclusions`` — the derived facts themselves, semantically ranked
    (``peer.conclusions_of(target).query(question, top_k=K)``). This is the
    same observer-to-observed scope the plugin's ``honcho_conclude`` tools use
    (``session.py:1219-1233``): in directional mode the AI peer is the
    observer of the user peer. It is the closest Honcho analogue of a mem0 or
    Mnemosyne memory row, and it carries a real ``created_at``.
  * ``search`` — raw message search, a diagnostic only.

``plugin_native_recall`` is therefore an INSTANCE attribute here, resolved
from the recall mode, not a class constant.

Ingestion is PER EXCHANGE, both roles, one ``add_messages`` call per exchange
with the plugin's own 25000-character chunking and its ``"[continued] "``
prefix (``plugins/memory/honcho/__init__.py:1226-1269,1328-1362``;
``session.py:421-458``). The plugin sends no ``created_at`` and no metadata
(``session.py:45-54``), so neither does this adapter by default.

TWO LLM ROLES, KEPT APART. Honcho's internal models (deriver, dialectic,
summary, dream, peer card) are configured on the SPAWNED SERVER processes
through ``HONCHO_LLM_*`` (see ``_honcho_server.py``). The shared answer and
judge model is the fairness-locked harness model, reached through
``eval_common`` and ``OPENAI_*`` in this process. This file never calls the
answer model, and it never lets the two configurations meet.

THREE DELIBERATE DEVIATIONS FROM THE PLUGIN, all in the harness direction:

  1. DRAIN. The plugin is fire-and-forget: ``sync_turn`` writes messages on a
     daemon thread and never waits for the deriver. A benchmark that answered
     immediately after ingest would measure the QUEUE, not the memory. So this
     adapter polls ``queue_status()`` after each session's ingest until no work
     unit is pending or in progress. A drain timeout raises, because a stuck
     deriver invalidates the persona rather than lowering its score.
  2. FRESH RECALL PER QUESTION. The plugin caches the base layer and consumes
     the dialectic result one turn late (``__init__.py:669-846``). This adapter
     fetches both fresh for every question, so a question is scored against the
     memory state that exists when it is asked.
  3. NO AGENT TOOLS. ``honcho_search``, ``honcho_profile``, and
     ``honcho_reasoning`` are agent-invoked tools in Hermes, not part of
     automatic injection. The ``search`` recall arm here is a DIAGNOSTIC, not
     the headline configuration.
  4. OPTIONAL DREAM TRIGGER (``HONCHO_DREAM_AFTER_SESSION=1``, off by
     default). Honcho consolidates on idle: 60 quiet minutes, 8 hours between
     dreams for one peer pair. Benchmark sessions run back to back in real
     time, so that scheduler would never fire, and the consolidation feature
     would go unmeasured even though a real deployment's sessions are days
     apart. See ``Schedule_Dream_And_Drain``.
  5. CONTEXT BUDGET ON, AT 8192 TOKENS (``HONCHO_CONTEXT_TOKENS``). The bound
     itself is the plugin's (``_truncate_to_budget``,
     ``__init__.py:870-883``), but the plugin ships ``contextTokens`` UNSET,
     which ``_parse_context_tokens`` (``client.py:145-153``) reads as
     uncapped. Uncapped does not run on a 32768-token serving window: a
     hybrid block measured 254k tokens at persona 0 session 5. See
     ``truncate_items_to_budget``.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Import the honcho SDK here, while sys.path[0] is still this script's own
# directory (``<repo>/honcho``). That directory holds this file but no
# ``honcho/`` subpackage, so ``import honcho`` binds the installed SDK. The
# collision is the same trap the mem0 adapter documents: this PROVIDER FOLDER
# and the SDK PACKAGE share a name. Run this file as
# ``python honcho/eval_honcho.py``. Running it as ``python -m
# honcho.eval_honcho`` from the repo root would put the repo root on
# sys.path, and ``import honcho`` would then resolve to this folder. Nothing
# below ever inserts the repo root.
from honcho import Honcho  # noqa: E402
from honcho.session import SessionPeerConfig  # noqa: E402

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

# The shared, provider-agnostic harness modules (eval_common, llm_reasoning,
# the scorers) live in ../benchmark. This makes them importable regardless of
# launch cwd.
_SHARED_HARNESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark"))
if _SHARED_HARNESS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_HARNESS_DIR)

from dotenv import load_dotenv  # noqa: E402

import eval_common  # noqa: E402
from eval_common import (  # noqa: E402
    Pair_Exchange_Turns,
    Parse_Session_Timestamp,
    ProviderBinding,
    add_common_eval_args,
    opt_int,
    record_provider_retrieval,
)

from _honcho_server import HonchoServer  # noqa: E402
from _local_embed_server import LocalEmbedServer  # noqa: E402

load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# --- Plugin constants (mirrored, not invented) -----------------------------
#: Honcho's message size cap. The plugin splits above it and prefixes every
#: continuation chunk (``__init__.py:1226-1269``).
CONTINUED_PREFIX = "[continued] "
#: The five reasoning levels, in order (``session.py:609``).
LEVEL_ORDER = ("minimal", "low", "medium", "high", "max")
#: Query-length thresholds for the plugin's reasoning heuristic
#: (``__init__.py:1042-1060``): +1 level at 120 chars, +2 at 400.
HEURISTIC_LENGTH_MEDIUM = 120
HEURISTIC_LENGTH_HIGH = 400
#: ``add_messages`` accepts at most 100 messages per call.
MAX_MESSAGES_PER_BATCH = 100
#: Honcho returns no per-item timestamp for a representation, card, summary,
#: or dialectic answer, so every recall item carries the harness placeholder.
UNKNOWN_TIME = "Unknown Time"

#: Recall modes whose payload IS the plugin's injection block: an ordered set
#: of named sections, not a ranked list. The harness must not re-slice these
#: to top-K, so the binding sets ``plugin_native_recall`` from this set. The
#: 'conclusions' and 'search' arms return ranked lists instead and take the
#: shared top-K like every other provider.
PLUGIN_NATIVE_MODES = ("hybrid", "base", "dialectic")

#: Observation presets, copied from ``plugins/memory/honcho/client.py:316-325``.
#: 'directional' is the plugin default: all four observe flags on.
OBSERVATION_PRESETS: Dict[str, Dict[str, bool]] = {
    "directional": {
        "user_observe_me": True, "user_observe_others": True,
        "ai_observe_me": True, "ai_observe_others": True,
    },
    "unified": {
        "user_observe_me": True, "user_observe_others": False,
        "ai_observe_me": False, "ai_observe_others": True,
    },
}


def sanitize_id(value: str) -> str:
    """Match Honcho's id pattern ``^[a-zA-Z0-9_-]+``."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))


def chunk_message(content: str, limit: int) -> List[str]:
    """Port of the plugin's ``_chunk_message`` (``__init__.py:1226-1269``).

    Splits at a paragraph break, then a sentence break, then a word break,
    and hard-cuts only when none of those lands past 30% of the window.
    Every continuation chunk carries the ``"[continued] "`` prefix so
    Honcho's representation engine can rejoin the message. Kept
    character-for-character identical to the plugin, because the chunk
    boundaries decide what the deriver sees.
    """
    if len(content) <= limit:
        return [content]

    prefix_len = len(CONTINUED_PREFIX)
    chunks: List[str] = []
    remaining = content
    first = True
    while remaining:
        effective = limit if first else limit - prefix_len
        if len(remaining) <= effective:
            chunks.append(remaining if first else CONTINUED_PREFIX + remaining)
            break

        segment = remaining[:effective]
        cut = segment.rfind("\n\n")
        if cut < effective * 0.3:
            cut = segment.rfind(". ")
            if cut >= 0:
                cut += 2  # include the period and the space
        if cut < effective * 0.3:
            cut = segment.rfind(" ")
        if cut < effective * 0.3:
            cut = effective  # hard cut

        chunk = remaining[:cut].rstrip()
        remaining = remaining[cut:].lstrip()
        if not first:
            chunk = CONTINUED_PREFIX + chunk
        chunks.append(chunk)
        first = False

    return chunks


def apply_reasoning_heuristic(base: str, query: str, cap: str, enabled: bool) -> str:
    """Port of the plugin's ``_apply_reasoning_heuristic`` (``__init__.py:1042-1060``)."""
    if not enabled or not query or base not in LEVEL_ORDER:
        return base
    n = len(query)
    if n < HEURISTIC_LENGTH_MEDIUM:
        bump = 0
    elif n < HEURISTIC_LENGTH_HIGH:
        bump = 1
    else:
        bump = 2
    cap_level = cap if cap in LEVEL_ORDER else "high"
    return LEVEL_ORDER[min(LEVEL_ORDER.index(base) + bump, LEVEL_ORDER.index(cap_level))]


def clip_dialectic(text: str, max_chars: int) -> str:
    """Port of the plugin's injection cap (``session.py:677-684``)."""
    if not text or not max_chars or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def truncate_items_to_budget(
    items: List[Dict[str, Any]], context_tokens: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Port of the plugin's ``_truncate_to_budget`` (``__init__.py:870-883``).

    The plugin joins its recall sections with ``"\\n\\n"`` and cuts the WHOLE
    block to ``context_tokens * 4`` characters at a word boundary, keeping the
    cut only when the last space sits past 80% of the budget, then appends
    ``" …"``. It runs on the final joined block of every automatic-injection
    path, so this applies it to the item list after the top-K slice, which is
    the same text in the same order.

    ``context_tokens`` 0 means no bound, which is the plugin's shipped state:
    ``_parse_context_tokens`` (``client.py:145-153``) returns None when
    ``contextTokens`` is unset. Returns (kept items, dropped count, 1 if an
    item was cut else 0).
    """
    if not context_tokens or context_tokens <= 0 or not items:
        return items, 0, 0
    budget_chars = context_tokens * 4
    total = sum(len(str(i.get("memory") or "")) for i in items) + 2 * (len(items) - 1)
    if total <= budget_chars:
        return items, 0, 0

    kept: List[Dict[str, Any]] = []
    used = 0
    cut = 0
    for index, item in enumerate(items):
        text = str(item.get("memory") or "")
        # The separator the plugin's own join contributes before this item.
        sep = 2 if index else 0
        if used + sep + len(text) <= budget_chars:
            kept.append(item)
            used += sep + len(text)
            continue
        room = budget_chars - used - sep
        if room > 0:
            truncated = text[:room]
            last_space = truncated.rfind(" ")
            if last_space > room * 0.8:
                truncated = truncated[:last_space]
            trimmed = dict(item)
            trimmed["memory"] = truncated + " …"
            kept.append(trimmed)
            cut = 1
        break
    return kept, len(items) - len(kept), cut


def _normalize_card(card: Any) -> str:
    """Render a peer card the way the plugin does: newline-joined lines."""
    if card is None:
        return ""
    if isinstance(card, str):
        return card.strip()
    if isinstance(card, (list, tuple)):
        return "\n".join(str(x) for x in card if x).strip()
    return str(card).strip()


# --------------------------------------------------------------------------
# Honcho recall
# --------------------------------------------------------------------------
class HonchoRecall:
    """Assemble one question's recall payload, mirroring the plugin.

    Everything this class returns is what the plugin would inject into the
    Hermes system prompt for that turn. Sections that come back empty are
    omitted, exactly as ``_format_first_turn_context`` omits them.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        observation_mode: str = "directional",
        reasoning_level: str = "low",
        dialectic_dynamic: bool = True,
        reasoning_level_cap: str = "high",
        dialectic_max_chars: int = 600,
        dialectic_max_input_chars: int = 10000,
        search_limit: int = 10,
    ):
        self.mode = mode
        self.observation = OBSERVATION_PRESETS.get(
            observation_mode, OBSERVATION_PRESETS["directional"])
        self.observation_mode = observation_mode
        self.reasoning_level = reasoning_level
        self.dialectic_dynamic = dialectic_dynamic
        self.reasoning_level_cap = reasoning_level_cap
        self.dialectic_max_chars = dialectic_max_chars
        self.dialectic_max_input_chars = dialectic_max_input_chars
        self.search_limit = search_limit

    # -- observer resolution -------------------------------------------------
    def observer_for(self, user_peer_id: str, ai_peer_id: str, target: str) -> str:
        """Port of ``_resolve_observer_target`` (``session.py:1087-1101``).

        When the AI peer observes others, EVERY context and dialectic call
        about the user runs from the AI peer's perspective. That is the
        plugin's directional default, and it decides which peer's
        representation of the user the benchmark reads.
        """
        if target == ai_peer_id:
            return ai_peer_id
        if self.observation["ai_observe_others"]:
            return ai_peer_id
        return target

    def dream_pairs(self, ctx: Dict[str, Any]) -> List[Tuple[str, str]]:
        """The observer-to-observed pairs a dream must consolidate.

        These are exactly the pairs hybrid recall reads: the user's
        representation and card, seen from whichever peer observes the user,
        plus the AI peer's own self-view. The plugin always fetches the AI
        self-view as ``ai -> ai`` regardless of observation mode
        (``session.py:768``), so that pair is unconditional.
        """
        user_id: str = ctx["user_peer_id"]
        ai_id: str = ctx["ai_peer_id"]
        pairs = [(self.observer_for(user_id, ai_id, user_id), user_id), (ai_id, ai_id)]
        seen: List[Tuple[str, str]] = []
        for pair in pairs:
            if pair not in seen:
                seen.append(pair)
        return seen

    # -- layer 1: base context ----------------------------------------------
    def fetch_peer_context(
        self, client: Honcho, peer_id: str, target: str,
        search_query: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Port of ``_fetch_peer_context`` (``session.py:974-1016``).

        The two fallbacks are not decoration. ``peer.context()`` can return an
        empty representation or an empty card for a peer that has both, so
        the plugin re-asks through ``peer.representation()`` and
        ``peer.get_card()``. Skipping them would under-report the base layer
        and make Honcho look emptier than the plugin sees it.
        """
        peer = client.peer(peer_id)
        representation, card = "", ""
        try:
            kwargs: Dict[str, Any] = {"target": target}
            if search_query is not None:
                kwargs["search_query"] = search_query
            peer_ctx = peer.context(**kwargs)
            representation = str(getattr(peer_ctx, "representation", "") or "").strip()
            card = _normalize_card(getattr(peer_ctx, "peer_card", None))
        except Exception as e:
            print(f"[honcho] peer.context failed ({peer_id} -> {target}): {e}", flush=True)

        if not representation:
            try:
                representation = str(peer.representation(target=target) or "").strip()
            except Exception as e:
                print(f"[honcho] peer.representation failed ({peer_id} -> {target}): {e}",
                      flush=True)
        if not card:
            try:
                card = _normalize_card(peer.get_card(target=target))
            except Exception as e:
                print(f"[honcho] peer.get_card failed ({peer_id} -> {target}): {e}",
                      flush=True)
        return representation, card

    def base_sections(self, ctx: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Port of ``get_prefetch_context`` + ``_format_first_turn_context``."""
        client: Honcho = ctx["client"]
        user_id: str = ctx["user_peer_id"]
        ai_id: str = ctx["ai_peer_id"]
        sections: List[Dict[str, Any]] = []
        raw: Dict[str, Any] = {}

        # 1. Session summary for the question's OWN session.
        summary_text = ""
        session_id = ctx.get("current_session_id")
        if session_id:
            try:
                session_ctx = client.session(session_id).context(summary=True)
                summary = getattr(session_ctx, "summary", None)
                summary_text = str(getattr(summary, "content", "") or "").strip()
            except Exception as e:
                print(f"[honcho] session summary failed ({session_id}): {e}", flush=True)
        raw["summary"] = summary_text
        if summary_text:
            sections.append(_section("session_summary", "## Session Summary", summary_text))

        # 2 + 3. User representation and peer card, from the observer's
        # perspective, with the question as the semantic search query
        # (``session.py:759-762``).
        observer_id = self.observer_for(user_id, ai_id, user_id)
        representation, card = self.fetch_peer_context(
            client, observer_id, target=user_id, search_query=question)
        raw["representation"] = representation
        raw["card"] = card
        if representation:
            sections.append(_section("user_representation", "## User Representation", representation))
        if card:
            sections.append(_section("user_peer_card", "## User Peer Card", card))

        # 4. The AI peer's own representation and card. The plugin fetches
        # this with NO search query (``session.py:768``).
        ai_representation, ai_card = self.fetch_peer_context(client, ai_id, target=ai_id)
        raw["ai_representation"] = ai_representation
        raw["ai_card"] = ai_card
        if ai_representation:
            sections.append(_section("ai_representation", "## AI Self-Representation", ai_representation))
        if ai_card:
            sections.append(_section("ai_peer_card", "## AI Identity Card", ai_card))

        ctx.setdefault("_raw_recall", {}).update(raw)
        return sections

    # -- layer 2: dialectic --------------------------------------------------
    def dialectic_section(self, ctx: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Port of ``dialectic_query`` (``session.py:615-687``)."""
        client: Honcho = ctx["client"]
        user_id: str = ctx["user_peer_id"]
        ai_id: str = ctx["ai_peer_id"]

        query = question
        if len(query) > self.dialectic_max_input_chars:
            query = query[:self.dialectic_max_input_chars].rsplit(" ", 1)[0]
        level = apply_reasoning_heuristic(
            self.reasoning_level, query, self.reasoning_level_cap, self.dialectic_dynamic)

        result = ""
        try:
            if self.observation["ai_observe_others"]:
                result = client.peer(ai_id).chat(
                    query, target=user_id, reasoning_level=level) or ""
            else:
                result = client.peer(user_id).chat(query, reasoning_level=level) or ""
        except Exception as e:
            # The plugin swallows a dialectic failure and injects nothing.
            # Mirror that, and count it, so a run reports how often the layer
            # was silent instead of hiding it in a stack trace.
            print(f"[honcho] dialectic failed (level={level}): {e}", flush=True)
            ctx["dialectic_errors"] = ctx.get("dialectic_errors", 0) + 1

        ctx.setdefault("_raw_recall", {})["dialectic_unclipped"] = result
        ctx["_last_reasoning_level"] = level
        if not result.strip():
            ctx["dialectic_empty"] = ctx.get("dialectic_empty", 0) + 1
            return []
        clipped = clip_dialectic(result.strip(), self.dialectic_max_chars)
        return [{"memory": clipped, "created_at": UNKNOWN_TIME, "score": None,
                 "source": "dialectic"}]

    # -- ranked arm: derived conclusions ------------------------------------
    def conclusions_section(
        self, ctx: Dict[str, Any], question: str, top_k: int,
    ) -> List[Dict[str, Any]]:
        """Semantically rank the derived facts Honcho holds about the user.

        The observer-to-observed scope mirrors the plugin's
        ``_conclusions_scope`` (``session.py:1219-1233``): the AI peer owns
        conclusions about the user whenever it observes others, which is the
        directional default; otherwise the user peer owns its own. This is a
        RANKED list with real timestamps, so it takes the shared top-K.
        """
        client: Honcho = ctx["client"]
        user_id: str = ctx["user_peer_id"]
        ai_id: str = ctx["ai_peer_id"]
        observer_id = self.observer_for(user_id, ai_id, user_id)
        try:
            scope = client.peer(observer_id).conclusions_of(user_id)
            conclusions = scope.query(question, top_k=max(1, top_k)) or []
        except Exception as e:
            print(f"[honcho] conclusions query failed "
                  f"(observer={observer_id}, observed={user_id}): {e}", flush=True)
            return []

        items: List[Dict[str, Any]] = []
        for conclusion in conclusions:
            content = str(getattr(conclusion, "content", "") or "").strip()
            if not content:
                continue
            created = getattr(conclusion, "created_at", None)
            items.append({
                "memory": content,
                "created_at": created.isoformat() if isinstance(created, datetime) else (
                    str(created) if created else UNKNOWN_TIME),
                # Honcho's conclusion query returns rank order and no score.
                "score": None,
                "source": "conclusion",
            })
        # The SDK's Conclusion carries a `level` field (explicit / deductive /
        # inductive / contradiction), but v3.0.9's API response schema does
        # NOT serialize it (src/schemas/api.py:435-449), so the SDK fills its
        # default "explicit" for every row. Recording it here would publish a
        # constant that looks like data. The split is visible only in the
        # server's `documents` table.
        ctx.setdefault("_raw_recall", {})["conclusion_hits"] = len(items)
        return items

    # -- diagnostic arm: raw message search ---------------------------------
    def search_section(self, ctx: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Port of ``search_context`` (``session.py:1129-1217``), minus the
        plugin's character budget.

        This is the ONE arm that returns raw dialogue turns rather than the
        peer model. It is a diagnostic: in Hermes this path is an
        agent-invoked tool, never part of automatic injection.
        """
        client: Honcho = ctx["client"]
        user_id: str = ctx["user_peer_id"]
        ai_id: str = ctx["ai_peer_id"]
        query = (question or "").strip()[:4000]
        if not query:
            return []
        messages: List[Any] = []
        try:
            messages = client.search(query, filters={"peer_perspective": user_id},
                                     limit=self.search_limit) or []
        except Exception as e:
            print(f"[honcho] workspace search failed, falling back to peer search: {e}",
                  flush=True)
            try:
                messages = client.peer(user_id).search(query, limit=self.search_limit) or []
            except Exception as e2:
                print(f"[honcho] peer search fallback failed: {e2}", flush=True)
                return []

        items: List[Dict[str, Any]] = []
        for message in messages:
            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                continue
            author = getattr(message, "peer_id", "") or "unknown"
            who = "assistant" if author == ai_id else author
            session_label = getattr(message, "session_id", "") or ""
            created = getattr(message, "created_at", None)
            items.append({
                "memory": f"[{who}{f' · {session_label}' if session_label else ''}] {content}",
                "created_at": created.isoformat() if isinstance(created, datetime) else (
                    str(created) if created else UNKNOWN_TIME),
                # Honcho's search is RRF-ranked and returns no score field.
                "score": None,
                "source": "search",
            })
        ctx.setdefault("_raw_recall", {})["search_hits"] = len(items)
        return items

    # -- entry point ---------------------------------------------------------
    def recall(self, ctx: Dict[str, Any], question: str,
               top_k: int = 5) -> List[Dict[str, Any]]:
        if self.mode == "conclusions":
            return self.conclusions_section(ctx, question, top_k)
        if self.mode == "search":
            return self.search_section(ctx, question)
        if self.mode == "base":
            return self.base_sections(ctx, question)
        if self.mode == "dialectic":
            return self.dialectic_section(ctx, question)
        # hybrid (default): layer 1 then layer 2, in the plugin's own order.
        return self.base_sections(ctx, question) + self.dialectic_section(ctx, question)


def _section(source: str, header: str, body: str) -> Dict[str, Any]:
    """One recall item.

    ``memory`` keeps the plugin's markdown header, because the header is part
    of what the plugin hands the agent: it names which layer of the peer
    model the text came from. ``source`` repeats the tag as a machine-readable
    field for the per-section counters.
    """
    return {"memory": f"{header}\n{body}", "created_at": UNKNOWN_TIME,
            "score": None, "source": source}


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def Add_Session_Dialogue_To_Honcho(
    ctx: Dict[str, Any],
    session_id: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
    message_max_chars: int,
    send_created_at: bool,
) -> Tuple[float, int, int]:
    """Ingest one session's dialogue at the plugin's own cadence.

    ONE ``add_messages`` call per exchange, user chunks first, then assistant
    chunks (``__init__.py:1328-1362`` feeding ``session.py:421-458``). The
    plugin never sends ``created_at`` or metadata, so neither does this
    function unless ``HONCHO_SEND_CREATED_AT=1`` asks for the vendor-exposed
    timestamp. In that arm each message advances a few seconds inside the
    session's own date, so the stored order matches the dialogue order.

    Returns (add_duration_ms, messages_added, exchanges).
    """
    client: Honcho = ctx["client"]
    session = client.session(session_id)
    user_peer = client.peer(ctx["user_peer_id"])
    ai_peer = client.peer(ctx["ai_peer_id"])

    # Peer observation is configured ONCE per session, at creation, exactly
    # as the plugin does (``session.py:196-209``). Without it, Honcho derives
    # nothing about the peers and every representation stays empty.
    observation = ctx["observation"]
    try:
        session.add_peers([
            (user_peer, SessionPeerConfig(observe_me=observation["user_observe_me"],
                                          observe_others=observation["user_observe_others"])),
            (ai_peer, SessionPeerConfig(observe_me=observation["ai_observe_me"],
                                        observe_others=observation["ai_observe_others"])),
        ])
    except Exception as e:
        print(f"[honcho] add_peers failed for session {session_id}: {e}", flush=True)

    start = time.time()
    added = 0
    exchanges = Pair_Exchange_Turns(dialogue_messages)
    clock = timestamp
    for exch_idx, group in enumerate(exchanges, start=1):
        payload = []
        for message in group:
            peer = user_peer if message.get("role") == "user" else ai_peer
            for chunk in chunk_message(str(message.get("content", "")), message_max_chars):
                if not chunk:
                    continue
                if send_created_at and clock is not None:
                    payload.append(peer.message(chunk, created_at=clock))
                    clock = clock + timedelta(seconds=5)
                else:
                    payload.append(peer.message(chunk))
        if not payload:
            continue
        # One exchange rarely reaches the 100-message batch cap. Guard anyway:
        # a single 2.5 MB turn would chunk past it and the call would 422.
        for batch_start in range(0, len(payload), MAX_MESSAGES_PER_BATCH):
            batch = payload[batch_start:batch_start + MAX_MESSAGES_PER_BATCH]
            try:
                session.add_messages(batch)
                added += len(batch)
            except Exception as e:
                print(f"[honcho] add_messages failed session={session_id} "
                      f"exchange={exch_idx}: {e}", flush=True)
                ctx["add_failures"] = ctx.get("add_failures", 0) + 1
    return (time.time() - start) * 1000.0, added, len(exchanges)


def Drain_Honcho_Queue(
    ctx: Dict[str, Any], timeout_s: float, poll_s: float,
) -> Dict[str, Any]:
    """Block until the deriver has processed every queued work unit.

    DEVIATION FROM THE PLUGIN, and a required one. ``sync_turn`` never waits:
    Hermes answers the next turn from whatever representation exists. A
    benchmark that answered right after ingest would measure queue latency
    instead of memory quality. ``queue_status()`` is workspace-scoped by the
    client's own ``workspace_id``. Only the pending and in-progress counters
    matter; ``completed_work_units`` resets between polls and cannot be used
    as a finish signal.

    A timeout RAISES. A persona whose deriver never drained has an unknown
    memory state, and scoring it would publish a number for a configuration
    that never existed.
    """
    client: Honcho = ctx["client"]
    start = time.time()
    polls = 0
    peak_pending = 0
    while True:
        polls += 1
        status = client.queue_status()
        pending = int(getattr(status, "pending_work_units", 0) or 0)
        in_progress = int(getattr(status, "in_progress_work_units", 0) or 0)
        total = int(getattr(status, "total_work_units", 0) or 0)
        peak_pending = max(peak_pending, pending + in_progress)
        if pending == 0 and in_progress == 0:
            return {
                "Drain_Duration_ms": (time.time() - start) * 1000.0,
                "Drain_Polls": polls,
                "Queue_Work_Units": total,
                "Queue_Peak_Outstanding": peak_pending,
                "Drain_Timed_Out": False,
            }
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"honcho deriver did not drain within {timeout_s:.0f}s "
                f"(workspace={ctx.get('store_id')}, pending={pending}, "
                f"in_progress={in_progress}, total={total}). A stuck deriver "
                f"invalidates this persona.")
        time.sleep(poll_s)


def Schedule_Dream_And_Drain(
    ctx: Dict[str, Any],
    pairs: List[Tuple[str, str]],
    timeout_s: float,
    poll_s: float,
) -> Dict[str, Any]:
    """Consolidate this workspace's observations, then wait for the result.

    A dream is Honcho's consolidation pass: it turns explicit observations
    into deductive and inductive conclusions and refreshes the peer card.
    Honcho's own scheduler fires it on IDLE — 60 minutes of quiet, with at
    least 8 hours between dreams for one peer pair. MemConflict sessions are
    DAYS apart in logical time, so a real deployment would dream between
    almost every pair of sessions. A benchmark that ran them back to back
    would never idle, so the consolidation feature under test would never
    fire. ``POST /v3/workspaces/{id}/schedule_dream`` bypasses the document
    threshold, the idle timer, and the spacing rule, which makes the
    inter-session idle explicit instead of accidental. Same argument as the
    drain deviation above.

    The dream's work units go through the same queue as derivation, so the
    second drain is what proves the dream finished.
    """
    client: Honcho = ctx["client"]
    start = time.time()
    requested = 0
    errors = 0
    for observer, observed in pairs:
        try:
            # No session scope: a dream consolidates ACROSS sessions, which
            # is the whole point of scheduling it between them.
            client.schedule_dream(observer=observer, observed=observed)
            requested += 1
        except Exception as e:
            errors += 1
            print(f"[honcho] schedule_dream failed ({observer} -> {observed}): {e}",
                  flush=True)
    drain = Drain_Honcho_Queue(ctx, timeout_s, poll_s)
    return {
        "Dream_Duration_ms": (time.time() - start) * 1000.0,
        "Dream_Requests": requested,
        "Dream_Errors": errors,
        "Dream_Work_Units": drain["Queue_Work_Units"],
        "Dream_Drain_Polls": drain["Drain_Polls"],
    }


# --------------------------------------------------------------------------
# Provider binding (the only Honcho-specific surface the driver sees)
# --------------------------------------------------------------------------
class HonchoBinding(ProviderBinding):
    memory_system = "honcho"
    store_id_key = "Honcho_Workspace_ID"
    runtime_summary_key = "Honcho_Runtime_Summary"
    stage_name = "honcho_answer_generation"
    stage_note = "Honcho retrieval and question answering"
    # PER-MODE, set on the instance in __init__. The hybrid, base, and
    # dialectic arms return the plugin's own injection block, so the harness
    # must not re-slice them to top-K (CLAUDE.md ruling 3). The conclusions
    # and search arms return ranked lists and take the shared top-K.
    plugin_native_recall = True

    def __init__(
        self,
        base_url: str,
        recall: HonchoRecall,
        api_key: str = "local",
        timeout: float = 30.0,
        workspace_prefix: str = "hermes_run_",
        user_peer_id: str = "user",
        ai_peer_id: str = "hermes",
        observation_mode: str = "directional",
        message_max_chars: int = 25000,
        send_created_at: bool = False,
        drain_timeout_s: float = 1800.0,
        drain_poll_s: float = 2.0,
        dream_after_session: bool = False,
        context_tokens: int = 8192,
    ):
        self.base_url = base_url
        self.recall_helper = recall
        self.api_key = api_key
        self.timeout = timeout
        self.workspace_prefix = workspace_prefix
        self.user_peer_id = sanitize_id(user_peer_id)
        self.ai_peer_id = sanitize_id(ai_peer_id)
        self.observation_mode = observation_mode
        self.message_max_chars = message_max_chars
        self.send_created_at = send_created_at
        self.drain_timeout_s = drain_timeout_s
        self.drain_poll_s = drain_poll_s
        self.dream_after_session = dream_after_session
        self.context_tokens = context_tokens
        self.plugin_native_recall = recall.mode in PLUGIN_NATIVE_MODES
        self._persona_index = -1

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        # One WORKSPACE per persona. A workspace is Honcho's own tenancy
        # boundary: peers, sessions, conclusions, and peer cards all live
        # inside it, so nothing can leak between personas.
        self._persona_index += 1
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        workspace = (f"{self.workspace_prefix}p{self._persona_index}_"
                     f"{sanitize_id(persona_id)[-16:]}")
        client = Honcho(base_url=self.base_url, workspace_id=workspace,
                        api_key=self.api_key, timeout=self.timeout)
        print(f"[honcho] persona {persona_id[-8:]} -> workspace '{workspace}'", flush=True)
        return {
            "store_id": workspace,
            "persona_tag": persona_id[-8:],
            "client": client,
            "user_peer_id": self.user_peer_id,
            "ai_peer_id": self.ai_peer_id,
            "observation": OBSERVATION_PRESETS.get(
                self.observation_mode, OBSERVATION_PRESETS["directional"]),
            "current_session_id": None,
            "total_messages": 0,
            "total_exchanges": 0,
            "total_drain_ms": 0.0,
            "total_work_units": 0,
            "add_failures": 0,
            "dialectic_empty": 0,
            "dialectic_errors": 0,
            "section_counts": {},
            "context_truncated_questions": 0,
            "context_items_dropped": 0,
            "total_dream_ms": 0.0,
            "total_dream_requests": 0,
            "total_dream_errors": 0,
            "total_dream_work_units": 0,
        }

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        timestamp = Parse_Session_Timestamp(session_item)
        session_label = str(session_item.get("Session_ID", session_index))
        session_id = sanitize_id(session_label)
        ctx["current_session_id"] = session_id

        add_ms, added, exchanges = Add_Session_Dialogue_To_Honcho(
            ctx, session_id, dialogue, timestamp,
            self.message_max_chars, self.send_created_at,
        )
        ctx["total_messages"] += added
        ctx["total_exchanges"] += exchanges

        drain = Drain_Honcho_Queue(ctx, self.drain_timeout_s, self.drain_poll_s)
        ctx["total_drain_ms"] += drain["Drain_Duration_ms"]
        ctx["total_work_units"] += drain["Queue_Work_Units"]

        dream: Dict[str, Any] = {}
        if self.dream_after_session:
            dream = Schedule_Dream_And_Drain(
                ctx, self.recall_helper.dream_pairs(ctx),
                self.drain_timeout_s, self.drain_poll_s,
            )
            ctx["total_dream_ms"] += dream["Dream_Duration_ms"]
            ctx["total_dream_requests"] += dream["Dream_Requests"]
            ctx["total_dream_errors"] += dream["Dream_Errors"]
            ctx["total_dream_work_units"] += dream["Dream_Work_Units"]

        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
              f"messages={added} exchanges={exchanges} ingest_ms={add_ms:.0f} "
              f"drain_ms={drain['Drain_Duration_ms']:.0f} "
              f"work_units={drain['Queue_Work_Units']}"
              + (f" dream_ms={dream['Dream_Duration_ms']:.0f} "
                 f"dream_units={dream['Dream_Work_Units']}" if dream else ""),
              flush=True)
        return {
            "Dialogue_Added_To_Memory": added > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Messages_Added": added,
            "Exchange_Count": exchanges,
            "Retain_Granularity": "exchange",
            "Session_Timestamp_Passed": timestamp.isoformat() if timestamp else None,
            "Add_Duration_ms": add_ms,
            **drain,
            **dream,
        }

    def recall(self, ctx, question_text, top_k):
        start = time.time()
        ctx["_raw_recall"] = {}
        items = self.recall_helper.recall(ctx, question_text, top_k)
        duration_ms = (time.time() - start) * 1000.0
        raw = dict(ctx.pop("_raw_recall", {}))
        raw["reasoning_level"] = ctx.get("_last_reasoning_level")
        raw["recall_mode"] = self.recall_helper.mode
        # The raw capture keeps the UNCLIPPED dialectic answer, the full
        # representation and card, and the pre-budget item list. The returned
        # items carry the plugin's own 600-character dialectic clip and its
        # context-token budget, which is what the answer model sees.
        record_provider_retrieval(ctx, raw=raw, ranked=items)
        if not self.plugin_native_recall:
            # Ranked arms take the shared top-K first. The full ranked list
            # stays in the diagnostic capture above, so a deeper offline curve
            # is still computable without changing what was measured.
            items = items[:top_k]
        # Plugin-native arms are NOT sliced: eval_common hands every section
        # to the answer model in this order, the way the plugin injects the
        # block. The context-token budget is the plugin's own bound on that
        # block and it applies to both paths.
        items, dropped, cut = truncate_items_to_budget(items, self.context_tokens)
        if dropped or cut:
            ctx["context_truncated_questions"] += 1
            ctx["context_items_dropped"] += dropped
        counts = ctx["section_counts"]
        for item in items:
            source = str(item.get("source") or "unknown")
            counts[source] = counts.get(source, 0) + 1
        return items, duration_ms

    def end_persona(self, ctx):
        # The per-persona workspace IS the isolation boundary, and it stays
        # in the run database as evidence. The whole database is disposable
        # (HONCHO_PG_CREATE_DB), so nothing leaks into the next run.
        return None

    def persona_count_extras(self, ctx):
        return {
            "Total_Messages_Added": ctx["total_messages"],
            "Total_Exchanges": ctx["total_exchanges"],
            "Total_Add_Failures": ctx["add_failures"],
            "Total_Drain_Time_ms": ctx["total_drain_ms"],
            "Total_Queue_Work_Units": ctx["total_work_units"],
            # Zero on every count means the dream arm is off, which is the
            # default. A nonzero Total_Dream_Errors means consolidation was
            # requested and refused, so the arm is mislabelled.
            "Total_Dream_Time_ms": ctx["total_dream_ms"],
            "Total_Dream_Requests": ctx["total_dream_requests"],
            "Total_Dream_Errors": ctx["total_dream_errors"],
            "Total_Dream_Work_Units": ctx["total_dream_work_units"],
        }

    def persona_tail_extras(self, ctx):
        counts = ctx["section_counts"]
        return {
            "Recall_Mode": self.recall_helper.mode,
            "Observation_Mode": self.observation_mode,
            "Dream_After_Session": self.dream_after_session,
            # Per-section counts show which layers of the peer model actually
            # produced text. An all-zero user_representation count means the
            # deriver produced nothing, which is a finding, not a crash.
            "Recall_Section_Counts": dict(counts),
            # How often the injection block hit the context-token budget, and
            # how many whole sections that bound removed. A high count on the
            # hybrid arm means the deriver is producing text the plugin would
            # never inject in full.
            "Context_Tokens": self.context_tokens,
            "Context_Truncated_Questions": ctx["context_truncated_questions"],
            "Context_Items_Dropped": ctx["context_items_dropped"],
            "Dialectic_Empty_Count": ctx["dialectic_empty"],
            "Dialectic_Error_Count": ctx["dialectic_errors"],
        }


def Generate_User_Honcho_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    binding: "HonchoBinding",
    server: Optional[HonchoServer],
    embed_server: Optional[LocalEmbedServer],
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
) -> bool:
    server_holder: Dict[str, Any] = {"server": server, "embed": embed_server}

    def setup():
        embed = server_holder.get("embed")
        if embed is not None:
            # Start the shim BEFORE the server: HonchoServer refuses to boot
            # without an embedder base URL, and the children read it from the
            # env at spawn time.
            os.environ["HONCHO_EMBEDDER_BASE_URL"] = embed.start()
        srv = server_holder.get("server")
        if srv is not None:
            srv.embedder_base_url = os.environ.get("HONCHO_EMBEDDER_BASE_URL")
            binding.base_url = srv.start()

    def teardown():
        srv = server_holder.get("server")
        if srv is not None:
            srv.close()
        embed = server_holder.get("embed")
        if embed is not None:
            embed.close()

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
    parser = argparse.ArgumentParser(description="Run Honcho evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "honcho_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "honcho_results.json"),
        top_k_help="How many recalled memories the answer LLM sees. It applies to the RANKED "
                   "arms only ('conclusions', 'search'), and 'conclusions' also asks Honcho "
                   "for exactly this many. The 'hybrid', 'base', and 'dialectic' arms are "
                   "section-structured, so they set plugin_native_recall and the answer model "
                   "sees EVERY assembled section. Either way this value stamps Actual_Top_K "
                   "and bounds the scorer's white-box depth.",
        default_start_idx=int(_env("START_IDX", "0")),
        default_end_idx=opt_int(_env("END_IDX")),
    )
    # --- recall arm ---------------------------------------------------------
    parser.add_argument("--recall_mode", type=str,
                        default=_env("HONCHO_RECALL_MODE", "hybrid"),
                        choices=["hybrid", "base", "dialectic", "conclusions", "search"],
                        help="'hybrid' (default, the plugin's own recallMode default): base "
                             "context sections plus the dialectic answer. 'base': layer 1 only "
                             "(summary, representations, peer cards). 'dialectic': layer 2 only. "
                             "Those three inject the plugin's whole block, with no top-K slice. "
                             "'conclusions': the derived facts, semantically ranked and sliced "
                             "to the shared top-K — the minimal arm. 'search': DIAGNOSTIC "
                             "raw-message search — in Hermes this is an agent-invoked tool, "
                             "never part of automatic injection.")
    parser.add_argument("--observation_mode", type=str,
                        default=_env("HONCHO_OBSERVATION_MODE", "directional"),
                        choices=["directional", "unified"],
                        help="Plugin preset for the four SessionPeerConfig observe flags "
                             "(client.py:316-325). 'directional' is the plugin default.")
    parser.add_argument("--dialectic_reasoning_level", type=str,
                        default=_env("HONCHO_DIALECTIC_REASONING_LEVEL", "low"),
                        choices=list(LEVEL_ORDER))
    parser.add_argument("--dialectic_dynamic", type=int,
                        default=int(_truthy(_env("HONCHO_DIALECTIC_DYNAMIC"), True)),
                        help="1 (default) applies the plugin's query-length heuristic: +1 "
                             "reasoning level at 120 chars, +2 at 400, capped at "
                             "--reasoning_level_cap.")
    parser.add_argument("--reasoning_level_cap", type=str,
                        default=_env("HONCHO_REASONING_LEVEL_CAP", "high"),
                        choices=list(LEVEL_ORDER))
    parser.add_argument("--dialectic_max_chars", type=int,
                        default=int(_env("HONCHO_DIALECTIC_MAX_CHARS", "600")))
    parser.add_argument("--dialectic_max_input_chars", type=int,
                        default=int(_env("HONCHO_DIALECTIC_MAX_INPUT_CHARS", "10000")))
    parser.add_argument("--message_max_chars", type=int,
                        default=int(_env("HONCHO_MESSAGE_MAX_CHARS", "25000")))
    parser.add_argument("--search_limit", type=int,
                        default=int(_env("HONCHO_SEARCH_LIMIT", "10")),
                        help="Result cap for the diagnostic 'search' arm only.")
    parser.add_argument("--context_tokens", type=int,
                        default=int(_env("HONCHO_CONTEXT_TOKENS", "8192")),
                        help="Token budget for the assembled injection block, the plugin's "
                             "own `contextTokens` knob (README:53,296). The port cuts to "
                             "context_tokens*4 characters at a word boundary "
                             "(__init__.py:870-883). 0 means uncapped, which is what the "
                             "plugin SHIPS: _parse_context_tokens returns None when the key "
                             "is unset. Uncapped is unrunnable here — a 32768-token serving "
                             "window minus the 16384-token answer budget leaves 16384, and "
                             "an uncapped hybrid block measured 254k tokens at session 5. "
                             "8192 is half the remaining budget, so the system prompt and "
                             "the question always fit.")
    # --- ingest -------------------------------------------------------------
    parser.add_argument("--send_created_at", type=int,
                        default=int(_truthy(_env("HONCHO_SEND_CREATED_AT"), False)),
                        help="1 passes the dataset session Date as each message's created_at. "
                             "The plugin never sends one (session.py:45-54), so the default 0 "
                             "is the plugin-faithful path; BENCH_CLOCKSYNC moves the server's "
                             "own clock instead.")
    parser.add_argument("--dream_after_session", type=int,
                        default=int(_truthy(_env("HONCHO_DREAM_AFTER_SESSION"), False)),
                        help="1 schedules a dream (Honcho's consolidation pass) after each "
                             "session's drain, then drains again. Honcho's own scheduler fires "
                             "dreams on 60 minutes of idle with 8-hour spacing; dataset "
                             "sessions are days apart in logical time, so a real deployment "
                             "would dream between most of them. Needs DREAM_ENABLED (the "
                             "vendor default).")
    parser.add_argument("--drain_timeout_s", type=float,
                        default=float(_env("HONCHO_DRAIN_TIMEOUT_S", "1800")))
    parser.add_argument("--drain_poll_s", type=float,
                        default=float(_env("HONCHO_DRAIN_POLL_S", "2.0")))
    # --- connection ---------------------------------------------------------
    parser.add_argument("--base_url", type=str, default=_env("HONCHO_BASE_URL"),
                        help="Attach to an already-running Honcho API instead of spawning one.")
    parser.add_argument("--api_key", type=str, default=_env("HONCHO_API_KEY", "local"))
    parser.add_argument("--timeout", type=float, default=float(_env("HONCHO_TIMEOUT", "30")),
                        help="SDK HTTP timeout. 30 is the plugin's own default "
                             "(client.py:245). The plugin runs the dialectic on a background "
                             "thread, so 30 costs it nothing; this adapter calls it inline, so "
                             "a slow internal model needs a larger value or the dialectic "
                             "layer silently returns empty.")
    parser.add_argument("--workspace_prefix", type=str,
                        default=_env("HONCHO_WORKSPACE_PREFIX",
                                     f"hermes_{_env('RUN_TAG', 'run')}_"))
    parser.add_argument("--user_peer_id", type=str, default=_env("HONCHO_USER_PEER_ID", "user"))
    parser.add_argument("--ai_peer_id", type=str, default=_env("HONCHO_AI_PEER_ID", "hermes"))
    args = parser.parse_args()

    recall_helper = HonchoRecall(
        mode=args.recall_mode,
        observation_mode=args.observation_mode,
        reasoning_level=args.dialectic_reasoning_level,
        dialectic_dynamic=bool(args.dialectic_dynamic),
        reasoning_level_cap=args.reasoning_level_cap,
        dialectic_max_chars=args.dialectic_max_chars,
        dialectic_max_input_chars=args.dialectic_max_input_chars,
        search_limit=args.search_limit,
    )
    # ATTACH when a base URL is given (a shared central server, the sharded
    # topology). SPAWN otherwise: this process owns the API, the deriver, and
    # the run database.
    server = None if args.base_url else HonchoServer()
    # SPAWN mode with no embedder endpoint configured: serve bge-small-en-v1.5
    # in this process, at the same 384 dimensions vllm-embed uses, so a host
    # run needs no separate service. Set HONCHO_EMBED_SHIM=0 to refuse
    # instead, which is what a Docker run wants: there the embedder must be
    # the shared vllm-embed, not a per-shard copy.
    embed_server = None
    if (server is not None and not _env("HONCHO_EMBEDDER_BASE_URL")
            and _truthy(_env("HONCHO_EMBED_SHIM"), True)):
        embed_server = LocalEmbedServer(port=opt_int(_env("HONCHO_EMBED_SHIM_PORT")) or 0)
    binding = HonchoBinding(
        base_url=args.base_url or "",
        recall=recall_helper,
        api_key=args.api_key,
        timeout=args.timeout,
        workspace_prefix=sanitize_id(args.workspace_prefix),
        user_peer_id=args.user_peer_id,
        ai_peer_id=args.ai_peer_id,
        observation_mode=args.observation_mode,
        message_max_chars=args.message_max_chars,
        send_created_at=bool(args.send_created_at),
        drain_timeout_s=args.drain_timeout_s,
        drain_poll_s=args.drain_poll_s,
        dream_after_session=bool(args.dream_after_session),
        context_tokens=args.context_tokens,
    )
    print(f"[DEBUG] recall_mode={args.recall_mode} observation={args.observation_mode} "
          f"level={args.dialectic_reasoning_level} dynamic={bool(args.dialectic_dynamic)} "
          f"cap={args.reasoning_level_cap} timeout={args.timeout}s "
          f"plugin_native_recall={binding.plugin_native_recall} "
          f"context_tokens={args.context_tokens} "
          f"dream_after_session={bool(args.dream_after_session)} "
          f"mode={'attach' if args.base_url else 'spawn'}", flush=True)

    # run_eval() returns False, not an exception, on a fatal error, so
    # per-persona incremental output survives a mid-run crash. Propagate that
    # as a nonzero exit, so the entrypoint's `set -e` stops the run instead of
    # scoring a partial file.
    ok = Generate_User_Honcho_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        binding=binding,
        server=server,
        embed_server=embed_server,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )
    sys.exit(0 if ok else 1)
