"""Shared MemConflict benchmark driver. This is the single provider-agnostic pipeline.

Each provider adapter (``../<provider>/eval_<provider>.py``) used to copy its
own dataset iteration, dialogue flattening, answer prompt, answer LLM call,
results-row emission, and compaction. This copying caused answer-decoding
drift between providers. The pipeline now lives HERE, once. A provider
adapter supplies only a :class:`ProviderBinding`, for store setup, teardown,
ingestion, and recall, then calls :func:`run_eval`.

FAIRNESS CONTRACT (why this module exists): the answer system prompt, the
answer user-prompt template, the answer LLM call (env-driven decoding through
``llm_reasoning``, with no decoding arguments in code), the question loop, the
top-K slicing, and the results-row schema are shared. Construction makes them
identical for every provider. No provider-side code may reimplement them.

The upstream ``eval_scoring.py`` scores the produced JSONL unchanged. Each
question ends up with ``Model_Answer`` and ``Retrieved_Memories``
(``memory`` / ``created_at`` / ``score``), which is exactly what the generic
scorer consumes.
"""

import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

try:
    import jsonlines
except ImportError:  # pragma: no cover
    jsonlines = None

# Steps the logical clock for the clock-sync arms. This is a no-op unless
# BENCH_CLOCKSYNC=1.
import clock_sync

# Reasoning-aware wrapper around the upstream llm_request. Environment
# variables (OPENAI_* / MEMCONFLICT_*) drive all answer decoding: model,
# temperature, max_tokens, and thinking. The call below passes no decoding
# arguments, so no provider can diverge on the answer path by accident.
from llm_reasoning import llm_request, calculate_cumulative_cost  # noqa: E402


# --------------------------------------------------------------------------
# The shared answer-generation contract (identical for every provider)
# --------------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

MAX_STORED_RETRIEVED_MEMORIES = 5


# --------------------------------------------------------------------------
# JSONL helpers (stdlib fallback if jsonlines is unavailable)
# --------------------------------------------------------------------------
def load_jsonl_items(input_file: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if jsonlines is not None:
        with jsonlines.open(input_file) as reader:
            for item in reader:
                items.append(item)
        return items
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl_items(output_file: str, items: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Dataset shaping
# --------------------------------------------------------------------------
def extract_dialogue_turn_order(key_name: str) -> int:
    try:
        return int(str(key_name).split("_")[-1])
    except Exception:
        return 10 ** 9


def Build_Session_Dialogue_List(session_dialogue: Any) -> List[Dict[str, Any]]:
    """Flatten a Session_Dialogue dict into a chronological role/content list."""
    if not isinstance(session_dialogue, dict):
        return []
    flattened: List[Dict[str, Any]] = []
    for turn_key in sorted(session_dialogue.keys(), key=extract_dialogue_turn_order):
        turn_value = session_dialogue.get(turn_key, [])
        if not isinstance(turn_value, list):
            continue
        for message_item in turn_value:
            if not isinstance(message_item, dict):
                continue
            role = message_item.get("role")
            content = message_item.get("content")
            if role in ("user", "assistant") and content not in (None, ""):
                flattened.append({"role": role, "content": str(content)})
    return flattened


def Parse_Session_Timestamp(session_item: Dict[str, Any]) -> Optional[datetime]:
    """Convert the dataset session ``Date`` field to an aware UTC datetime.

    Every provider uses this as the logical ingest timestamp. This keeps the
    simulated chronology identical across all three providers. The function
    returns None if the date is not parsable.
    """
    session_date = str(session_item.get("Date", "")).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(session_date, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def Parse_Query_Now_Timestamp(session_item: Dict[str, Any]) -> Optional[datetime]:
    """The single definition of recall-time "now".

    This is the session's logical ``Date``, anchored at noon
    (``clock_sync.RECALL_NOW_HOUR_UTC`` = 12:00 UTC).

    A provider passes this value as its explicit recall-time parameter
    (Hindsight ``query_timestamp``, Mnemosyne ``query_time``, RetainDB
    ``question_date``). The time is deliberately noon, not the midnight that
    :func:`Parse_Session_Timestamp` returns for ingest row timestamps.
    :func:`clock_sync.set_clock` fakes the OS clock to the same noon. Both
    functions import the one ``RECALL_NOW_HOUR_UTC`` constant, so they cannot
    drift apart. A provider that reads the faked OS clock (Supermemory) and a
    provider that receives an explicit parameter therefore agree on "now" to
    the same instant. This removes cross-provider recency skew. Ingest stays
    at midnight, plus per-turn or per-exchange minute bumps, which is always
    before noon. So recall-"now" is never earlier than the memories it ranks.
    The function returns None, meaning the parameter is omitted and the
    provider falls back to its wall clock, in exactly the same cases as
    :func:`Parse_Session_Timestamp`. So anchor-off and non-backdated runs stay
    byte-identical.
    """
    base = Parse_Session_Timestamp(session_item)
    if base is None:
        return None
    return base.replace(hour=clock_sync.RECALL_NOW_HOUR_UTC, minute=0,
                        second=0, microsecond=0)


def Pair_Exchange_Turns(dialogue_messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group a flattened session's messages into user and assistant exchange pairs.

    This mirrors the Hermes memory plugins. Each plugin makes one ingest call
    per completed exchange: one user turn plus the assistant turn that
    answered it. A lone or unpaired message, such as a dangling user turn
    with no assistant reply yet, or two same-role turns in a row, forms its
    own group. The function never drops such a message.
    """
    pairs: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(dialogue_messages)
    while i < n:
        current = dialogue_messages[i]
        nxt = dialogue_messages[i + 1] if i + 1 < n else None
        if current.get("role") == "user" and nxt is not None and nxt.get("role") == "assistant":
            pairs.append([current, nxt])
            i += 2
        else:
            pairs.append([current])
            i += 1
    return pairs


# --------------------------------------------------------------------------
# Answer generation (the one place the answer LLM is called)
# --------------------------------------------------------------------------
def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]]) -> str:
    # NOTE (fairness): this code intentionally does not render the raw provider
    # similarity `score` here. Each provider score comes from a different,
    # uncalibrated scale (Mnemosyne's hybrid blend, Hindsight's RRF and
    # rerank fusion, RetainDB's BM25, vector, and graph RRF). Feeding this
    # score to the answer LLM would give it a cross-provider "confidence"
    # signal that is not actually comparable. That signal would itself be a
    # fairness defect: the model could learn to trust or distrust a
    # provider's score distribution instead of the memory content. Every
    # stored ``Retrieved_Memories`` item still records the score (see the
    # question-row assembly in Answer_Questions_For_One_Session), because
    # SEH@K, log-rank@K, and EUG@5 need it for analysis. Only the rendered
    # prompt text drops it.
    lines = ["Retrieved memories:"]
    if not retrieved_memories:
        lines.append("No relevant memories found.")
    else:
        for idx, item in enumerate(retrieved_memories, start=1):
            memory_text = item.get("memory", "")
            created_at = item.get("created_at", "Unknown Time")
            lines.append(f"{idx}. [{created_at}] {memory_text}")
    return "\n".join(lines)


def Build_Answer_User_Prompt(context_text: str, question_text: str) -> str:
    """The answer user-prompt template.

    ``replay_answers.py`` also uses this template. Keep it byte-stable. Any
    change breaks replay comparability.

    ``context_text`` normally comes from Build_Retrieved_Memory_Context, which
    dropped the rendered `(score=...)` suffix as a fairness fix (see its
    docstring). Because of this, the prompt this function assembles is no
    longer byte-identical to the one that answered pre-change Results rows.
    A result produced after that change is not replay-comparable against a
    pre-change Results file. No v2 headline runs existed when this changed,
    so nothing committed is invalidated.
    """
    return (
        "Retrieved Memory Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Answer:"
    )


def Generate_Answer_With_Retrieved_Memory(
    system_prompt: str, context_text: str, question_text: str
) -> Tuple[str, Dict[str, Any], float]:
    user_prompt = Build_Answer_User_Prompt(context_text, question_text)
    start = time.time()
    answer_text, cost_info = llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        return_parsed_json=False,
        extract_json=False,
    )
    duration_ms = (time.time() - start) * 1000.0
    if isinstance(answer_text, tuple):
        answer_text = answer_text[0]
    zero_cost = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_cost_usd": 0.0, "model": None, "pricing_available": False,
    }
    return str(answer_text).strip(), cost_info or zero_cost, duration_ms


def _accumulate_cost(acc: Dict[str, Any], cost_info: Dict[str, Any]) -> None:
    acc["input_tokens"] += cost_info.get("input_tokens", 0) or 0
    acc["output_tokens"] += cost_info.get("output_tokens", 0) or 0
    acc["total_tokens"] += cost_info.get("total_tokens", 0) or 0
    acc["total_cost_usd"] += cost_info.get("total_cost_usd", 0.0) or 0.0
    if acc.get("model") is None:
        acc["model"] = cost_info.get("model")
    if cost_info.get("pricing_available") is True:
        acc["pricing_available"] = True


# --------------------------------------------------------------------------
# Diagnostic retrieval capture (shared; unconditional in this generation format)
# --------------------------------------------------------------------------
# WHY: a stored row used to keep only the compact answer context
# (``Retrieved_Memories``, top-5). This made an offline top-10/20 retrieval
# curve, or any retrieval-only diagnostic, impossible to compute from the
# artifacts without a full re-run. Every row now also carries:
#
#   Provider_Raw_Retrieval      the provider's own recall response for this
#                               question, JSON-coerced (see _json_safe).
#   Normalized_Ranked_Retrieval the full ranked list the provider actually
#                               returned, in provider rank order, normalized to
#                               the Retrieved_Memories item schema
#                               (memory/created_at/score). This is the list
#                               before the harness or plugin top-K slice,
#                               dedup, and compaction.
#
# HARD CONSTRAINT: this capture never widens a provider request. Adapters
# capture only what they already fetch (RetainDB and Supermemory already ask
# for 10 and slice to 5, so 10 items are captured; mem0 already asks for
# exactly top_k, so the ranked list equals Retrieved_Memories, which is
# correct and expected). Raising a provider limit to get a deeper curve would
# change the measured configuration.
#
# The answer path is untouched. The answer context is still built from
# ``retrieved`` (see Answer_Questions_For_One_Session). These two fields are
# write-only provenance. The scorer reads only named fields, so extra keys
# are inert (benchmark/score_resumable.py -> eval_scoring.extract_model_answer
# and extract_top_k_retrieved_memories both read by name).

#: These are ctx side-channel keys. An adapter writes them through
#: record_provider_retrieval(). The question loop pops each key once per
#: question. The leading underscore marks that these are never stored field
#: names.
DIAG_RAW_CTX_KEY = "_diag_provider_raw"
DIAG_RANKED_CTX_KEY = "_diag_ranked_retrieval"
DIAG_DEPTH_MAX_CTX_KEY = "_diag_retrieval_depth_max"

#: An optional bound on the stored diagnostic ranked list. The default, 0,
#: means unbounded, which is the intended setting. The Hindsight
#: plugin-native arm legitimately returns about 157 observations for one
#: question, and that depth is the measurement. Set BENCH_DIAG_MAX_RANKED=<n>
#: only if row size must be capped, and record this in the run's manifest or
#: docs. A capped file cannot support a depth curve past n.
DIAG_MAX_RANKED = int(os.environ.get("BENCH_DIAG_MAX_RANKED", "0") or 0)

_JSON_SAFE_MAX_DEPTH = 12


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Coerce a provider response into a JSON-serializable structure.

    Provider SDKs return pydantic models, dataclasses, or datetimes.
    json.dumps cannot write any of these. This function prefers real
    structure, in order: model_dump, to_dict, dict, __dict__. It falls back
    to ``repr`` for a leaf value nothing else can express. A lossy leaf is
    better than an unwritable row, because a row that fails to serialize
    would lose the whole persona's results file.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth >= _JSON_SAFE_MAX_DEPTH:
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth + 1) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _json_safe(method(), _depth + 1)
            except Exception:
                pass
    inner = getattr(value, "__dict__", None)
    if isinstance(inner, dict) and inner:
        return {str(k): _json_safe(v, _depth + 1)
                for k, v in inner.items() if not str(k).startswith("_")}
    slots = getattr(type(value), "__slots__", None)
    if slots:
        # __slots__ classes have no instance __dict__. Read the declared fields instead.
        out = {}
        for name in ([slots] if isinstance(slots, str) else list(slots)):
            if str(name).startswith("_"):
                continue
            try:
                out[str(name)] = _json_safe(getattr(value, name), _depth + 1)
            except Exception:
                pass
        if out:
            return out
    return repr(value)


def Normalize_Ranked_Item(item: Dict[str, Any]) -> Dict[str, Any]:
    """One ranked item in the stored ``Retrieved_Memories`` schema.

    This deliberately keeps only the three scored fields. The diagnostic
    list is a retrieval-depth record, not a second copy of every provider
    sub-score. Those sub-scores stay on Retrieved_Memories, for the items
    that reached the answer context.
    """
    return {
        "memory": str(item.get("memory", "")),
        "created_at": item.get("created_at", "Unknown Time"),
        "score": item.get("score"),
    }


def record_provider_retrieval(
    diag: Optional[Dict[str, Any]],
    raw: Any = None,
    ranked: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Adapter-to-driver handoff for one question's diagnostic capture.

    ``diag`` is the per-persona ctx, or None. A value of None makes this call
    a no-op, so a search helper can still run from a smoke script. ``ranked``
    is the provider's full ranked list, already mapped to harness items. Omit
    it when the adapter's ``recall()`` already returns that full list,
    because the driver then falls back to the returned list. The question
    loop consumes and clears this capture once per question, so a stale
    capture can never attach to the next question.
    """
    if diag is None:
        return
    diag[DIAG_RAW_CTX_KEY] = raw
    if ranked is not None:
        diag[DIAG_RANKED_CTX_KEY] = list(ranked)


def Take_Diagnostic_Retrieval(
    ctx: Dict[str, Any], retrieved: List[Dict[str, Any]]
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Pop this question's capture and return (raw, normalized_ranked).

    This falls back to ``retrieved`` when the adapter recorded no explicit
    ranked list, because its recall already returns the provider's full
    ranked list, for example Hindsight. It also tracks the per-persona
    maximum depth for the runtime summary.
    """
    raw = ctx.pop(DIAG_RAW_CTX_KEY, None)
    ranked = ctx.pop(DIAG_RANKED_CTX_KEY, None)
    if ranked is None:
        ranked = retrieved
    items = [Normalize_Ranked_Item(i) for i in (ranked or []) if isinstance(i, dict)]
    if DIAG_MAX_RANKED > 0:
        items = items[:DIAG_MAX_RANKED]
    if len(items) > int(ctx.get(DIAG_DEPTH_MAX_CTX_KEY, 0) or 0):
        ctx[DIAG_DEPTH_MAX_CTX_KEY] = len(items)
    return _json_safe(raw), items


def _zero_cost(note: str) -> Dict[str, Any]:
    return {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_cost_usd": 0.0, "model": None, "pricing_available": False,
        "note": note,
    }


# --------------------------------------------------------------------------
# Provider binding — the ONLY provider-specific surface
# --------------------------------------------------------------------------
class ProviderBinding:
    """What a provider adapter implements. Everything else is shared.

    A binding owns the store handles and the provider API calls. It must not
    build prompts, call the answer LLM, or shape result rows. This module
    does all of that. ``ctx`` is a per-persona dict. The binding creates this
    dict in ``begin_persona`` and threads it through the hooks. It holds
    store handles, ids, and running counters.
    """

    #: e.g. "mnemosyne" — stamped into every row's Memory_System.
    memory_system: str = "provider"
    #: Result-item key for the per-persona store id, e.g. "Mnemosyne_Session_ID".
    store_id_key: str = "Store_ID"
    #: Result-item key for the runtime summary, e.g. "Mnemosyne_Runtime_Summary".
    runtime_summary_key: str = "Runtime_Summary"
    #: Observable-cost stage name, e.g. "mnemosyne_answer_generation".
    stage_name: str = "answer_generation"
    #: _zero_cost note, e.g. "Mnemosyne retrieval and question answering".
    stage_note: str = "retrieval and question answering"
    #: When True, the answer LLM sees every retrieved item, in order. The
    #: stored Retrieved_Memories then keeps all of them too, instead of the
    #: ``retrieved[:top_k]`` answer slice and the
    #: ``max(top_k, MAX_STORED_RETRIEVED_MEMORIES)`` storage cap. The
    #: default, False, keeps every provider and arm byte-identical. The
    #: Hindsight plugin-native arm, the featured Arm C, sets this flag. The
    #: real Hermes Hindsight plugin injects the full token-budgeted recall
    #: result to the agent, not a top-K slice. Measuring "what the plugin
    #: hands Hermes" therefore requires the harness not to re-slice it.
    #: SEH@K is unaffected, because the scorer independently caps at its own
    #: WHITE_BOX_TOP_K_VALUES. This flag only widens the answer context and
    #: the stored provenance.
    plugin_native_recall: bool = False

    # -- lifecycle ---------------------------------------------------------
    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        """Create the isolated per-persona store and return the ctx dict.

        The ctx dict must contain ``store_id``: the bank id, session id,
        project id, or whatever value identifies the isolated store in the
        results.
        """
        raise NotImplementedError

    def ingest_session(
        self,
        ctx: Dict[str, Any],
        session_item: Dict[str, Any],
        dialogue: List[Dict[str, Any]],
        session_index: int,
    ) -> Dict[str, Any]:
        """Ingest one session's dialogue.

        This also runs any provider lifecycle steps between ingest and
        answering, such as consolidation waits, sleep, or retirement. It
        returns the provider fields for Session_Memory_Metadata. The
        returned dict must include ``Add_Duration_ms`` and
        ``Dialogue_Added_To_Memory``.
        """
        raise NotImplementedError

    def recall(
        self, ctx: Dict[str, Any], question_text: str, top_k: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Retrieve scored memories for one question.

        This returns (retrieved, duration_ms). Each retrieved item carries
        at least ``memory``, ``created_at``, and ``score``. Extra
        diagnostic keys pass through to the stored rows, and the scorer
        ignores them.
        """
        raise NotImplementedError

    def end_persona(self, ctx: Dict[str, Any]) -> None:
        """Tear down and clean up the per-persona store. A finally block calls this."""

    # -- optional hooks ------------------------------------------------------
    #: When set, this callback runs per question, after recall. It returns
    #: (extra_context_text, extra_question_fields). If the text is
    #: non-empty, it is appended after the raw retrieved-memory context,
    #: never interleaved, because SEH scores only the raw block. The fields
    #: are stored on the question row. The Mnemosyne canonical and oracle
    #: arms use this hook.
    extra_answer_context: Optional[
        Callable[[Dict[str, Any], str], Tuple[str, Dict[str, Any]]]
    ] = None

    def persona_count_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Provider ingest counters for the runtime summary, for example
        Total_Added_Memories or Total_Retain_Calls_OK. These are inserted
        after the timing totals."""
        return {}

    def persona_tail_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Trailing runtime-summary fields, for example consolidation-wait stats."""
        return {}

    def persona_result_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Extra top-level result-item fields, for example Retirement_Diagnostics.
        This method omits any key whose value is None."""
        return {}


# --------------------------------------------------------------------------
# Question loop (shared)
# --------------------------------------------------------------------------
def Answer_Questions_For_One_Session(
    binding: ProviderBinding,
    ctx: Dict[str, Any],
    session_item: Dict[str, Any],
    top_k: int,
    max_questions: Optional[int],
    overwrite_existing_answers: bool,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    stage_cost = _zero_cost(binding.stage_note)
    answered = 0
    retrieval_ms = 0.0
    response_ms = 0.0
    extra_context_questions = 0
    use_extra_context = binding.extra_answer_context is not None

    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    metadata["Memory_System"] = binding.memory_system
    metadata["Top_K"] = top_k

    for q_idx, question_item in enumerate(session_questions):
        if max_questions is not None and answered >= max_questions:
            break
        existing = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing not in (None, ""):
            continue
        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue

        retrieved, search_ms = binding.recall(ctx, question_text, top_k)
        # This pops the diagnostic capture before anything else touches ctx,
        # so it can only ever belong to this question. It is write-only
        # provenance. Nothing below reads it, so the answer context is
        # unaffected.
        raw_retrieval, ranked_retrieval = Take_Diagnostic_Retrieval(ctx, retrieved)
        # Plugin-native recall is opt-in and off by default. When on, the
        # answer LLM sees every retrieved item, in order, matching a plugin
        # that injects its full token-budgeted recall result rather than a
        # top-K slice. The default keeps the historical top_k slice, so
        # every other provider and arm stays byte-identical.
        context_items = retrieved if binding.plugin_native_recall else retrieved[:top_k]
        answer_context = Build_Retrieved_Memory_Context(context_items)
        if use_extra_context:
            extra_text, extra_fields = binding.extra_answer_context(ctx, question_text)
            question_item.update(extra_fields)
            if extra_text:
                # This text is appended after the raw memories, so the
                # retrieved-turn context that SEH scores stays unchanged.
                answer_context = f"{answer_context}\n\n{extra_text}"
                extra_context_questions += 1
        answer_text, cost_info, resp_ms = Generate_Answer_With_Retrieved_Memory(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            context_text=answer_context,
            question_text=question_text,
        )

        question_item["Retrieved_Memory_Context"] = answer_context
        question_item["Retrieved_Memories"] = retrieved
        # These fields store the full provider recall response and the full
        # ranked list, at the depth the provider was already asked for. This
        # request is never widened. The list length implies the depth. The
        # per-persona maximum lands in the runtime summary as
        # Diag_Retrieval_Depth_Max.
        question_item["Provider_Raw_Retrieval"] = raw_retrieval
        question_item["Normalized_Ranked_Retrieval"] = ranked_retrieval
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_ms
        question_item["Response_Duration_ms"] = resp_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = binding.memory_system
        session_questions[q_idx] = question_item

        retrieval_ms += search_ms
        response_ms += resp_ms
        answered += 1
        _accumulate_cost(stage_cost, cost_info)

    metadata["Session_Retrieval_Time_ms"] = retrieval_ms
    metadata["Session_Response_Time_ms"] = response_ms
    metadata["Session_Answered_Question_Count"] = answered
    if use_extra_context:
        metadata["Session_Canonical_Context_Question_Count"] = extra_context_questions
    session_item["Session_Questions"] = session_questions
    session_item["Session_Memory_Metadata"] = metadata
    return session_item, answered, stage_cost


def place_metadata_before_event_types(session_item: Dict[str, Any]) -> Dict[str, Any]:
    target = "Session_Memory_Metadata"
    value = copy.deepcopy(session_item.get(target))
    reordered: Dict[str, Any] = {}
    inserted = False
    for key, val in session_item.items():
        if key == target:
            continue
        if key == "Event_Types" and not inserted:
            reordered[target] = value
            inserted = True
        reordered[key] = val
    if not inserted:
        reordered[target] = value
    return reordered


# --------------------------------------------------------------------------
# Compaction (keeps output files small, because the scorer needs only answers and retrievals)
# --------------------------------------------------------------------------
#: Compaction preserves these provider-provenance question keys, when
#: present. Arms that never set them omit the keys, so other providers are
#: unaffected.
_PASSTHROUGH_QUESTION_KEYS = (
    "Canonical_Context", "Canonical_Context_Diagnostics",
    "Canonical_Search_Duration_ms",
    # The Supermemory /v4/profile featured arm uses these three keys: the
    # static/dynamic profile block the plugin injects into the answer
    # context, plus its section counts. This is provenance only and is
    # never scored. Every other provider and arm omits it, so this is
    # purely additive.
    "Profile_Block", "Profile_Static_Count", "Profile_Dynamic_Count",
)


def Build_Compact_Question(question_item: Dict[str, Any], keep_top_k: int,
                           keep_all_retrieved: bool = False) -> Dict[str, Any]:
    compact = {
        "question_id": question_item.get("question_id"),
        "question": question_item.get("question"),
        "answer": question_item.get("answer"),
        "conflict_type": question_item.get("conflict_type"),
        "ability_target": question_item.get("ability_target"),
        "difficulty": question_item.get("difficulty"),
        "Model_Answer": question_item.get("Model_Answer"),
        "Memory_Search_Duration_ms": question_item.get("Memory_Search_Duration_ms"),
        "Response_Duration_ms": question_item.get("Response_Duration_ms"),
        "Actual_Top_K": question_item.get("Actual_Top_K"),
        "Memory_System": question_item.get("Memory_System"),
    }
    retrieved = question_item.get("Retrieved_Memories", [])
    if not isinstance(retrieved, list):
        compact["Retrieved_Memories"] = []
    elif keep_all_retrieved:
        # The plugin-native arm keeps the full injected set, so the stored
        # row records exactly what the plugin handed the agent. SEH@K still
        # caps independently.
        compact["Retrieved_Memories"] = retrieved
    else:
        keep = max(keep_top_k, MAX_STORED_RETRIEVED_MEMORIES)
        compact["Retrieved_Memories"] = retrieved[:keep]
    # The diagnostic capture survives compaction untruncated. Storing the
    # depth is the whole point, so this capture is never cut back to
    # keep_top_k like the answer context above. Only BENCH_DIAG_MAX_RANKED
    # bounds it, and it is unbounded by default.
    compact["Provider_Raw_Retrieval"] = question_item.get("Provider_Raw_Retrieval")
    ranked = question_item.get("Normalized_Ranked_Retrieval")
    compact["Normalized_Ranked_Retrieval"] = ranked if isinstance(ranked, list) else []
    for key in _PASSTHROUGH_QUESTION_KEYS:
        if key in question_item:
            compact[key] = question_item[key]
    return compact


def Build_Compact_Session(session_item: Dict[str, Any], keep_top_k: int,
                          keep_all_retrieved: bool = False) -> Dict[str, Any]:
    compact = {
        "Session_ID": session_item.get("Session_ID"),
        "Date": session_item.get("Date"),
        "Question_Trigger_Types": copy.deepcopy(session_item.get("Question_Trigger_Types", [])),
        "Session_Question_Count": session_item.get("Session_Question_Count", 0),
        "Session_Memory_Metadata": copy.deepcopy(session_item.get("Session_Memory_Metadata", {})),
        "Session_Questions": [],
    }
    questions = session_item.get("Session_Questions", [])
    if isinstance(questions, list):
        compact["Session_Questions"] = [
            Build_Compact_Question(q, keep_top_k, keep_all_retrieved=keep_all_retrieved)
            for q in questions if isinstance(q, dict)
        ]
    return compact


# --------------------------------------------------------------------------
# Per-persona driver (shared)
# --------------------------------------------------------------------------
def Generate_Single_Persona_Eval(
    binding: ProviderBinding,
    persona_item: Dict[str, Any],
    top_k: int,
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
):
    previous_cost = persona_item.get("token_cost", None)
    full_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
    if max_sessions is not None:
        full_chain = full_chain[:max_sessions]

    persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
    persona_tag = persona_id[-8:]

    total_answered = 0
    answered_sessions = 0
    add_ms = 0.0
    retrieval_ms = 0.0
    response_ms = 0.0
    stage_cost = _zero_cost(binding.stage_note)

    persona_start = time.monotonic()
    ctx = binding.begin_persona(persona_item)
    try:
        for idx, session_item in enumerate(full_chain):
            # In clock-sync arms, this steps the perceived OS clock to this
            # session's logical date before ingest. The session's ingest and
            # Q&A then run under that clock. This step must precede
            # session_start, because in arms where this process is
            # preloaded, such as Mnemosyne, timers must not span the jump.
            clock_sync.set_clock(Parse_Session_Timestamp(session_item))
            session_start = time.monotonic()
            dialogue = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))

            print(
                f"[DEBUG] persona {persona_tag} session {session_item.get('Session_ID', idx)} "
                f"ingest_start ({idx + 1}/{len(full_chain)})"
            )
            provider_meta = binding.ingest_session(ctx, session_item, dialogue, idx)
            add_ms += provider_meta.get("Add_Duration_ms", 0.0) or 0.0

            session_questions = session_item.get("Session_Questions", [])
            meta: Dict[str, Any] = {
                "Memory_System": binding.memory_system,
                "Top_K": top_k,
            }
            meta.update(provider_meta)
            session_item["Session_Memory_Metadata"] = meta

            if not isinstance(session_questions, list) or len(session_questions) == 0:
                meta["Session_Retrieval_Time_ms"] = 0.0
                meta["Session_Response_Time_ms"] = 0.0
                meta["Session_Answered_Question_Count"] = 0
                meta["Session_Total_Runtime_ms"] = (time.monotonic() - session_start) * 1000.0
                full_chain[idx] = place_metadata_before_event_types(session_item)
                print(
                    f"[DEBUG] persona {persona_tag} session {session_item.get('Session_ID', idx)} "
                    f"answered=0 ingest_ms={provider_meta.get('Add_Duration_ms', 0.0) or 0.0:.0f} "
                    f"retrieve_ms=0 answer_ms=0 "
                    f"session_ms={meta['Session_Total_Runtime_ms']:.0f} "
                    f"persona_elapsed_s={(time.monotonic() - persona_start):.0f}"
                )
                continue

            answered_sessions += 1
            updated, answered, call_cost = Answer_Questions_For_One_Session(
                binding=binding,
                ctx=ctx,
                session_item=session_item,
                top_k=top_k,
                max_questions=max_questions_per_session,
                overwrite_existing_answers=overwrite_existing_answers,
            )
            total_answered += answered
            _accumulate_cost(stage_cost, call_cost)

            meta = updated.get("Session_Memory_Metadata", {})
            retrieval_ms += meta.get("Session_Retrieval_Time_ms", 0.0) or 0.0
            response_ms += meta.get("Session_Response_Time_ms", 0.0) or 0.0
            meta["Session_Total_Runtime_ms"] = (time.monotonic() - session_start) * 1000.0
            updated["Session_Memory_Metadata"] = meta
            full_chain[idx] = place_metadata_before_event_types(updated)
            print(
                f"[DEBUG] persona {persona_tag} session {session_item.get('Session_ID', idx)} "
                f"answered={answered} ingest_ms={provider_meta.get('Add_Duration_ms', 0.0) or 0.0:.0f} "
                f"retrieve_ms={meta.get('Session_Retrieval_Time_ms', 0.0) or 0.0:.0f} "
                f"answer_ms={meta.get('Session_Response_Time_ms', 0.0) or 0.0:.0f} "
                f"session_ms={meta.get('Session_Total_Runtime_ms', 0.0):.0f} "
                f"persona_elapsed_s={(time.monotonic() - persona_start):.0f}"
            )
    finally:
        binding.end_persona(ctx)

    persona_total_ms = (time.monotonic() - persona_start) * 1000.0
    print(
        f"[DEBUG] persona {persona_tag} summary sessions={len(full_chain)} "
        f"answered_sessions={answered_sessions} questions_answered={total_answered} "
        f"total_ingest_ms={add_ms:.0f} total_retrieve_ms={retrieval_ms:.0f} "
        f"total_answer_ms={response_ms:.0f} persona_wall_s={(persona_total_ms / 1000.0):.0f}"
    )
    runtime_summary: Dict[str, Any] = {
        "Persona_Add_Time_ms": add_ms,
        "Persona_Retrieval_Time_ms": retrieval_ms,
        "Persona_Response_Time_ms": response_ms,
        "Persona_Total_Runtime_ms": persona_total_ms,
    }
    runtime_summary.update(binding.persona_count_extras(ctx))
    runtime_summary.update({
        "Average_Add_Time_Per_Session_ms": (add_ms / len(full_chain)) if full_chain else 0.0,
        "Average_Retrieval_Time_Per_Session_ms": (retrieval_ms / answered_sessions) if answered_sessions else 0.0,
        "Average_Response_Time_Per_Session_ms": (response_ms / answered_sessions) if answered_sessions else 0.0,
    })
    runtime_summary.update(binding.persona_tail_extras(ctx))
    # This is the deepest Normalized_Ranked_Retrieval seen on this persona.
    # It shows how far offline retrieval curves (top-10/20) can be computed
    # from this file. Every provider reports it the same way. A value of 0
    # means no question was answered.
    runtime_summary["Diag_Retrieval_Depth_Max"] = int(ctx.get(DIAG_DEPTH_MAX_CTX_KEY, 0) or 0)

    observable = {
        "Stage_Name": binding.stage_name,
        "Input_Tokens": stage_cost.get("input_tokens", 0),
        "Output_Tokens": stage_cost.get("output_tokens", 0),
        "Total_Tokens": stage_cost.get("total_tokens", 0),
        "Total_Cost_USD": stage_cost.get("total_cost_usd", 0.0),
        "Model": stage_cost.get("model"),
        "Pricing_Available": bool(stage_cost.get("pricing_available")),
    }
    final_cost = calculate_cumulative_cost(previous_cost, stage_cost)
    result = {
        "ID": persona_item.get("ID"),
        "Memory_System": binding.memory_system,
        binding.store_id_key: ctx.get("store_id"),
        "Eval_Top_K": top_k,
        "Answered_Session_Count": answered_sessions,
        "Answered_Question_Count": total_answered,
        binding.runtime_summary_key: runtime_summary,
        "Observable_Token_Cost_Summary": observable,
        "token_cost": final_cost,
        "Full_Session_Chain": [
            Build_Compact_Session(s, top_k, keep_all_retrieved=binding.plugin_native_recall)
            for s in full_chain
        ],
    }
    for key, value in binding.persona_result_extras(ctx).items():
        if value is not None:
            result[key] = value
    return result, total_answered, answered_sessions


# --------------------------------------------------------------------------
# Top-level driver (shared)
# --------------------------------------------------------------------------
def run_eval(
    binding: ProviderBinding,
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
    setup: Optional[Callable[[], None]] = None,
    teardown: Optional[Callable[[], None]] = None,
) -> bool:
    """Iterate personas, delegate provider work to ``binding``, and write results.

    ``setup`` and ``teardown`` bracket the whole run, for shared daemon or
    server startup and shutdown. The per-persona store lifecycle lives in
    the binding.
    """
    print(f"Processing file: {input_jsonl_path}")
    print(f"Output file: {output_jsonl_path}")
    print(f"[DEBUG] Top-K retrieval: {top_k}")
    try:
        all_personas = load_jsonl_items(input_jsonl_path)
        selected = all_personas[start_idx:end_idx] if end_idx is not None else all_personas[start_idx:]
        print(f"[DEBUG] Read {len(all_personas)} personas; evaluating {len(selected)}")

        if setup is not None:
            setup()

        results: List[Dict[str, Any]] = []
        for i, persona_item in enumerate(selected):
            abs_idx = start_idx + i
            print(f"[DEBUG] Persona {abs_idx + 1}/{len(all_personas)} (ID={persona_item.get('ID')})")
            result_item, total_answered, answered_sessions = Generate_Single_Persona_Eval(
                binding=binding,
                persona_item=persona_item,
                top_k=top_k,
                max_sessions=max_sessions,
                max_questions_per_session=max_questions_per_session,
                overwrite_existing_answers=overwrite_existing_answers,
            )
            results.append(result_item)
            # This writes incrementally, so a long run stays crash-safe.
            write_jsonl_items(output_jsonl_path, results)
            out_dir = os.path.dirname(output_json_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results[0] if len(results) == 1 else results, f, ensure_ascii=False, indent=2)
            print(f"[DEBUG] Persona {abs_idx + 1} done - sessions={answered_sessions} questions={total_answered}")

        print(f"[DEBUG] Successfully processed {binding.memory_system} evaluation")
        return True
    except Exception as e:  # pragma: no cover
        print(f"Error processing {binding.memory_system} evaluation: {e}:{traceback.format_exc()}")
        return False
    finally:
        if teardown is not None:
            teardown()


# --------------------------------------------------------------------------
# Small CLI helpers shared by the adapters
# --------------------------------------------------------------------------
def opt_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("", "none", "all", "-1"):
        return None
    return int(value)


def add_common_eval_args(
    parser,
    *,
    default_input_jsonl_path: str,
    default_output_jsonl_path: str,
    default_output_json_path: str,
    top_k_help: str,
    default_start_idx: int = 0,
    default_end_idx: Optional[int] = None,
    max_sessions_help: str = "Cap sessions ingested per persona (default: all). "
                            "Use a small value for smoke tests.",
) -> None:
    """Register the nine dataset and slicing CLI arguments every provider adapter shares.

    Some values vary per provider, so callers pass them in: the three path
    defaults live in each provider's own ``Results/`` folder, ``--top_k``'s
    help text differs per provider in wording and length ("memories" versus
    "facts"), ``--max_sessions``'s help has two phrasings, and mem0 drives
    ``--start_idx`` and ``--end_idx`` from the environment. Everything else
    is fixed here, so no adapter can drift: argument names, types, the
    ``opt_int`` idiom, the identical ``--max_questions_per_session`` help
    text, and ``--overwrite_existing_answers``. Adapters add their own
    provider-specific flags directly after calling this function.
    """
    parser.add_argument("--input_jsonl_path", type=str, default=default_input_jsonl_path)
    parser.add_argument("--output_jsonl_path", type=str, default=default_output_jsonl_path)
    parser.add_argument("--output_json_path", type=str, default=default_output_json_path)
    parser.add_argument("--top_k", type=int, default=5, help=top_k_help)
    parser.add_argument("--start_idx", type=int, default=default_start_idx)
    parser.add_argument("--end_idx", type=lambda v: opt_int(v), default=default_end_idx)
    parser.add_argument("--max_sessions", type=lambda v: opt_int(v), default=None,
                        help=max_sessions_help)
    parser.add_argument("--max_questions_per_session", type=lambda v: opt_int(v), default=None,
                        help="Cap questions answered per session (default: all).")
    parser.add_argument("--overwrite_existing_answers", action="store_true")
