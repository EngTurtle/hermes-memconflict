"""MemConflict evaluation adapter for the OpenViking memory system.

The shared ``benchmark/eval_common.py`` driver runs the provider-agnostic
pipeline: dataset iteration, dialogue flattening, the answer prompt and the
answer LLM call, results-row emission, and compaction. This file supplies only
the OpenViking-specific binding: per-persona identity, ingestion, drain, and
recall.

THIS ADAPTER NEVER IMPORTS THE ``openviking`` PIP PACKAGE. The provider folder
that holds this file is also named ``openviking``, so ``import openviking``
would bind to the folder. Every call goes over HTTP with ``httpx``, which is
also what the Hermes plugin does: the plugin has no SDK dependency either
(``plugins/memory/openviking/__init__.py``, ``_VikingClient``). Nothing below
puts the repo root on ``sys.path``.

WHAT IS DIFFERENT ABOUT OpenViking (vs. mem0 / Hindsight / Honcho)
------------------------------------------------------------------
OpenViking stores extracted memories as a FILESYSTEM: ``viking://`` URIs under
``user/memories/`` with ``profile.md``, ``preferences/``, ``entities/``,
``events/YYYY/MM/DD/``, ``cases/``, and ``patterns``. Raw dialogue turns are
not in the memory search space. A search hit is a node in that tree, and it
carries an ``abstract`` (the ~256-character summary), a ``level`` (0, 1, or 2),
a ``category``, and a score — but NO timestamp. Every recall item therefore
reports ``created_at: "Unknown Time"``.

The plugin's recall (``prefetch``) has two parts, and this adapter ports both:

  * PART A, the session-start block: the profile file plus a listing of the
    ``preferences/`` and ``entities/`` directories, assembled under a
    quarter-token budget (``_build_session_start_memory_block``,
    ``__init__.py:3210``).
  * PART B, the per-query search: ``POST /api/v1/search/search`` with a
    candidate limit of ``max(recall_limit*4, 20)``, then CLIENT-SIDE
    selection, re-ranking, dedupe, and a character budget
    (``_select_recall_candidates`` 3338, ``_build_prefetch_entries`` 3416).

Both parts are what the plugin hands the agent, so the ``prefetch`` arm sets
``plugin_native_recall`` and the harness does not re-slice it (CLAUDE.md
ruling 3).

DELIBERATE DEVIATIONS FROM THE PLUGIN, all recorded in docs/DECISIONS.md:

  1. DRAIN. The plugin commits and moves on (``on_session_end``,
     ``__init__.py:3860``). This adapter commits, polls the extraction task to
     ``completed``, then blocks on ``POST /api/v1/system/wait``. A persona
     whose extraction had not finished would be scored against an unknown
     memory state.
  2. SESSION-START BLOCK PER QUESTION. The plugin latches the block once per
     session id (``_profile_prefetched_sessions``, 3264) because a chat
     session is one continuous context. Every benchmark question is answered
     as an independent turn, so the block is rebuilt into each question's
     context. It is still BUILT once per session and cached, so the arm costs
     one profile read and two listings per session, as the plugin does.
  3. RECALL TIMEOUTS 60/30, not the plugin's 4.0/3.0. The plugin's budget is
     tuned for an interactive CLI, where an empty recall is better than a
     stalled turn. Under benchmark serving latency 4 seconds empties recall
     silently. 60 is the plugin's own clamp maximum (``_recall_config``,
     2953-2964).
  4. NO ``created_at`` BY DEFAULT. The plugin sends none, so the plugin-
     faithful temporal path is BENCH_CLOCKSYNC moving the server clock.
     ``OPENVIKING_SEND_CREATED_AT=1`` is the vendor-exposed deviation arm.

TWO LLM ROLES, KEPT APART. OpenViking's extraction and query-planning model
is configured on the SPAWNED SERVER through ``OPENVIKING_LLM_*`` (see
``_openviking_server.py``). The shared answer and judge model is the
fairness-locked harness model, reached through ``eval_common`` and
``OPENAI_*`` in this process. This file never calls the answer model.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

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

from _openviking_server import OpenVikingServer  # noqa: E402

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


# --------------------------------------------------------------------------
# Plugin constants (mirrored, not invented)
# --------------------------------------------------------------------------
#: The server resolves these URIs against the request's X-OpenViking-User
#: header, so they carry no uid segment. The plugin uses exactly these
#: (``__init__.py:93-95``); a uid-qualified form would read a different tree.
PROFILE_URI = "viking://user/memories/profile.md"
PREFERENCES_URI = "viking://user/memories/preferences"
ENTITIES_URI = "viking://user/memories/entities"
MEMORIES_URI = "viking://user/memories"
#: Listing parameters for the session-start block (``__init__.py:96-101``).
SESSION_START_LIST_PARAMS: Dict[str, Any] = {
    "output": "agent",
    "recursive": True,
    "abs_limit": 512,
    "node_limit": 512,
}
#: Below this length the plugin runs no search at all (``__init__.py:89``).
RECALL_QUERY_MIN_CHARS = 5
#: Floor under the remaining recall budget (``__init__.py:90``).
RECALL_MIN_TIMEOUT_SECONDS = 0.05
#: ``messages/batch`` accepts at most 100 messages per POST.
SESSION_MESSAGE_BATCH_LIMIT = 100
#: A search hit is a filesystem node with no timestamp field, so every item
#: carries the harness placeholder.
UNKNOWN_TIME = "Unknown Time"
#: Result cap for the diagnostic ``find`` arm. Deterministic retrieval with
#: no LLM intent analysis, so the depth is fixed rather than derived from the
#: plugin's candidate formula.
FIND_LIMIT = 10

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_user_id(value: str) -> str:
    return _SAFE_ID.sub("_", str(value))


# --------------------------------------------------------------------------
# HTTP client (port of the plugin's _VikingClient, __init__.py:252-415)
# --------------------------------------------------------------------------
class OpenVikingHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def status_code_of(error: Exception) -> Optional[int]:
    if isinstance(error, OpenVikingHTTPError):
        return error.status_code
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def unwrap_result(resp: Any) -> Any:
    """Return the payload body whether or not the server wrapped it.

    OpenViking answers some routes as ``{"result": ...}`` and others as the
    bare body. This is the plugin's own ``_unwrap_result``
    (``__init__.py:4073``), and it is the ONE place that handles the
    difference: every extractor below calls it first.
    """
    if isinstance(resp, dict) and "result" in resp:
        return resp.get("result")
    return resp


class VikingClient:
    """Persona-scoped HTTP client.

    Identity is a header set, not a URL path: ``X-OpenViking-User`` is what
    isolates one persona's memory tree from the next, which is the same
    mechanism the vendor's own LoCoMo harness uses. ``X-OpenViking-Actor-Peer``
    names the agent peer. With an API key set, the key pair replaces the
    tenant headers, exactly as the plugin decides it
    (``_headers``, ``__init__.py:270``).
    """

    def __init__(self, endpoint: str, api_key: str = "", account: str = "default",
                 user: str = "default", agent: str = "hermes",
                 timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or ""
        self.account = account
        self.user = user
        self.agent = agent
        self.default_timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def headers(self, include_tenant: Optional[bool] = None) -> Dict[str, str]:
        if include_tenant is None:
            include_tenant = not bool(self.api_key)
        h = {"Content-Type": "application/json"}
        if self.agent:
            h["X-OpenViking-Actor-Peer"] = self.agent
        if include_tenant:
            if self.account:
                h["X-OpenViking-Account"] = self.account
            if self.user:
                h["X-OpenViking-User"] = self.user
        if self.api_key:
            h["X-API-Key"] = self.api_key
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def _parse(self, resp: httpx.Response) -> Dict[str, Any]:
        """Port of ``_VikingClient._parse_response`` (``__init__.py:317``).

        A 2xx body can still carry ``status: "error"``, so the status line
        alone does not decide success.
        """
        try:
            data = resp.json()
        except Exception:
            data = None

        if resp.status_code >= 400:
            message = (resp.text or "")[:300]
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    code = error.get("code", "HTTP_ERROR")
                    raise OpenVikingHTTPError(
                        f"{code}: {error.get('message', message)}", resp.status_code)
                if data.get("status") == "error":
                    raise OpenVikingHTTPError(str(data), resp.status_code)
            raise OpenVikingHTTPError(message or f"HTTP {resp.status_code}",
                                      resp.status_code)

        if isinstance(data, dict) and data.get("status") == "error":
            error = data.get("error")
            if isinstance(error, dict):
                raise OpenVikingHTTPError(
                    f"{error.get('code', 'OPENVIKING_ERROR')}: {error.get('message', '')}")
            raise OpenVikingHTTPError(str(data))

        if data is None:
            return {}
        return data

    @staticmethod
    def _needs_trusted_identity_retry(error: Exception) -> bool:
        message = str(error)
        return ("Trusted mode requests must include X-OpenViking-Account" in message
                or "Trusted mode requests must include X-OpenViking-User" in message)

    def _send(self, send) -> Dict[str, Any]:
        """Port of ``_send_with_trusted_identity_retry`` (``__init__.py:304``).

        An API-key request omits the tenant headers. A server in trusted mode
        answers such a request with an explicit complaint, and the plugin then
        retries WITH the headers. The retry cannot fire in the dev-auth arms,
        where the key is empty and the headers always go out.
        """
        try:
            return self._parse(send(self.headers()))
        except Exception as error:
            if not self.api_key or not self._needs_trusted_identity_retry(error):
                raise
            return self._parse(send(self.headers(include_tenant=True)))

    def get(self, path: str, params: Optional[Dict[str, Any]] = None,
            timeout: Optional[float] = None) -> Dict[str, Any]:
        wait = timeout if timeout is not None else self.default_timeout
        return self._send(lambda headers: self._client.get(
            f"{self.endpoint}{path}", params=params, headers=headers, timeout=wait))

    def post(self, path: str, payload: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        wait = timeout if timeout is not None else self.default_timeout
        return self._send(lambda headers: self._client.post(
            f"{self.endpoint}{path}", json=payload or {}, headers=headers, timeout=wait))

    def delete(self, path: str, params: Optional[Dict[str, Any]] = None,
               timeout: Optional[float] = None) -> Dict[str, Any]:
        wait = timeout if timeout is not None else self.default_timeout
        return self._send(lambda headers: self._client.request(
            "DELETE", f"{self.endpoint}{path}", params=params, headers=headers,
            timeout=wait))

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Recall budget and payload extractors (ported)
# --------------------------------------------------------------------------
def remaining_recall_timeout(deadline: float, per_request_timeout: float) -> float:
    """Port of ``_remaining_recall_timeout`` (``__init__.py:2301``)."""
    remaining = deadline - time.monotonic()
    if remaining <= RECALL_MIN_TIMEOUT_SECONDS:
        raise TimeoutError("OpenViking recall budget exhausted")
    return min(per_request_timeout, remaining)


def extract_text_content(resp: Any) -> str:
    result = unwrap_result(resp)
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("content") or result.get("text") or "").strip()
    return ""


def extract_read_content(resp: Any) -> str:
    result = unwrap_result(resp)
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("content", "text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def extract_memory_listing(resp: Any) -> List[Dict[str, str]]:
    """Port of ``_extract_memory_listing`` (``__init__.py:2993``).

    Directory nodes are dropped and abstracts are whitespace-collapsed to 200
    characters, so one long abstract cannot consume the listing's budget.
    """
    result = unwrap_result(resp)
    if not isinstance(result, list):
        return []
    entries: List[Dict[str, str]] = []
    for raw in result:
        if not isinstance(raw, dict) or raw.get("isDir"):
            continue
        name = str(raw.get("rel_path") or raw.get("name") or "").strip()
        if not name.endswith(".md"):
            continue
        abstract = " ".join(str(raw.get("abstract") or "").split())[:200]
        entries.append({"name": name, "abstract": abstract})
    entries.sort(key=lambda entry: entry["name"])
    return entries


# --------------------------------------------------------------------------
# Session-start block: quarter-token budgeting (ported verbatim in behaviour)
# --------------------------------------------------------------------------
def token_units(content: str) -> int:
    """Quarter-token units, OpenViking's shared estimator (``__init__.py:3011``).

    A CJK character counts 6 units and everything else 1, so four units make
    one token for Latin text.
    """
    return sum(6 if ord(ch) >= 0x3000 else 1 for ch in content)


def take_token_prefix(content: str, max_units: int) -> str:
    if max_units <= 0:
        return ""
    used = 0
    for index, ch in enumerate(content):
        used += 6 if ord(ch) >= 0x3000 else 1
        if used > max_units:
            return content[:index]
    return content


def take_token_suffix(content: str, max_units: int) -> str:
    if max_units <= 0:
        return ""
    used = 0
    start = len(content)
    for idx in range(len(content) - 1, -1, -1):
        used += 6 if ord(content[idx]) >= 0x3000 else 1
        if used > max_units:
            return content[start:]
        start = idx
    return content


def truncate_profile_content(content: str, max_units: int) -> str:
    """Port of ``_truncate_profile_content`` (``__init__.py:3046``).

    A long profile keeps its first 8 lines and its tail, with the middle
    elided, because the head holds the identity lines and the tail holds the
    most recent facts.
    """
    content = content.strip()
    if token_units(content) <= max_units:
        return content

    def _head_only() -> str:
        marker = "\n... [profile truncated]"
        marker_units = token_units(marker)
        if marker_units >= max_units:
            return take_token_prefix(content, max_units)
        head = take_token_prefix(content, max_units - marker_units).rstrip()
        return f"{head}{marker}" if head else take_token_prefix(content, max_units)

    lines = content.split("\n")
    head_line_count = 8
    if len(lines) <= head_line_count + 4:
        return _head_only()

    marker = "\n... [profile middle elided] ...\n"
    remaining = max_units - token_units(marker)
    if remaining <= 0:
        return _head_only()

    head = take_token_prefix("\n".join(lines[:head_line_count]), remaining // 2).rstrip()
    tail = take_token_suffix("\n".join(lines[head_line_count:]),
                             remaining - token_units(head)).lstrip()
    return f"{head}{marker}{tail}" if tail else _head_only()


def assemble_session_start_memory_block(
    profile: str, preference_lines: List[str], entity_lines: List[str],
) -> str:
    """Port of ``_assemble_session_start_memory_block`` (``__init__.py:3153``)."""
    lines: List[str] = []
    if profile:
        lines.extend([f'<user-profile uri="{PROFILE_URI}">', profile, "</user-profile>"])
    if preference_lines or entity_lines:
        lines.append("<available-memories>")
        lines.extend(preference_lines)
        lines.extend(entity_lines)
        lines.append("</available-memories>")
    return "\n".join(lines)


def format_memory_listing(
    uri: str, entries: List[Dict[str, str]], max_units: int,
) -> Tuple[List[str], int]:
    """Port of ``_format_memory_listing`` (``__init__.py:3173``).

    Returns (lines, units used). An over-budget listing degrades to a one-line
    stub that names the entry count, so the agent still learns the directory
    exists.
    """
    if not entries or max_units <= 0:
        return [], 0

    header = f"  {uri}/"
    header_units = token_units(header)
    if header_units > max_units:
        stub = f"  {uri}/  ({len(entries)} entries; use `viking_search`)"
        stub_units = token_units(stub)
        return ([stub], stub_units) if stub_units <= max_units else ([], 0)

    lines = [header]
    used = header_units
    newline_units = token_units("\n")
    for index, entry in enumerate(entries):
        abstract = entry.get("abstract", "")
        description = f" — {abstract}" if abstract else ""
        line = f"    - {entry['name']}{description}"
        line_units = newline_units + token_units(line)
        if used + line_units > max_units:
            remaining = len(entries) - index
            tail = f"    ... +{remaining} more, use `viking_search`"
            tail_units = newline_units + token_units(tail)
            if used + tail_units <= max_units:
                lines.append(tail)
                used += tail_units
            break
        lines.append(line)
        used += line_units
    return lines, used


def build_session_start_memory_block(
    *, profile: str, preferences: List[Dict[str, str]],
    entities: List[Dict[str, str]], token_budget: int,
) -> str:
    """Port of ``_build_session_start_memory_block`` (``__init__.py:3210``).

    The budget is in quarter-token units: ``token_budget * 4`` total, minus
    the scaffold tags, with the profile capped at half and the remainder split
    between the two listings.
    """
    profile = profile.strip()
    if not profile and not preferences and not entities:
        return ""

    placeholder = "\0"
    scaffold = assemble_session_start_memory_block(
        placeholder if profile else "",
        [placeholder] if preferences else [],
        [placeholder] if entities else [],
    )
    placeholder_count = int(bool(profile)) + int(bool(preferences)) + int(bool(entities))
    overhead_units = token_units(scaffold) - placeholder_count
    available_units = max(0, (token_budget * 4) - overhead_units)

    profile_text = ""
    if profile and available_units > 0:
        profile_units = min(available_units, token_budget * 2)
        profile_text = truncate_profile_content(profile, profile_units)
        available_units -= token_units(profile_text)

    preference_budget = available_units // 2 if (preferences and entities) else available_units
    preference_lines, preference_units = format_memory_listing(
        PREFERENCES_URI, preferences, preference_budget)
    available_units -= preference_units
    entity_lines, _ = format_memory_listing(ENTITIES_URI, entities, available_units)

    return assemble_session_start_memory_block(profile_text, preference_lines, entity_lines)


# --------------------------------------------------------------------------
# Candidate selection and re-rank (ported)
# --------------------------------------------------------------------------
def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def raw_score(item: Dict[str, Any]) -> Optional[float]:
    """The server's own score, unclamped. Selection uses the clamped value;
    the stored row keeps the raw one, because the scorer's log-rank reads it."""
    try:
        return float(item.get("score"))
    except (TypeError, ValueError):
        return None


def recall_category(item: Dict[str, Any]) -> str:
    return str(item.get("category") or "").strip() or "memory"


def recall_abstract(item: Dict[str, Any]) -> str:
    """Port of ``_recall_abstract`` (``__init__.py:3301``)."""
    for key in ("abstract", "overview", "text", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(item.get("uri") or "").strip()


def dedupe_key(item: Dict[str, Any]) -> str:
    """Port of ``_dedupe_key`` (``__init__.py:3310``).

    Events and cases dedupe by URI, not by abstract text: two events can share
    a summary and still be different occurrences, which is exactly the
    distinction a conflict question asks about.
    """
    uri = str(item.get("uri") or "").strip()
    category = str(item.get("category") or "").strip().lower() or "unknown"
    abstract = " ".join(recall_abstract(item).lower().split())
    uri_lower = uri.lower()
    if abstract and "/events/" not in uri_lower and "/cases/" not in uri_lower:
        return f"abstract:{category}:{abstract}"
    return f"uri:{uri}"


def query_tokens(query: str) -> List[str]:
    tokens = []
    for raw in query.lower().replace("_", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 2:
            tokens.append(token)
    return tokens[:8]


def recall_rank(item: Dict[str, Any], tokens: List[str]) -> float:
    """Port of ``_recall_rank`` (``__init__.py:3330``).

    The server score is re-ranked client-side: +0.12 for a level-2 leaf (the
    full memory body rather than a directory summary) and up to +0.2 for query
    tokens found in the URI or abstract.
    """
    text = f"{item.get('uri', '')} {recall_abstract(item)}".lower()
    overlap = sum(1 for token in tokens if token in text)
    return clamp_score(item.get("score")) + (0.12 if item.get("level") == 2 else 0.0) \
        + min(0.2, overlap * 0.05)


def select_recall_candidates(
    items: List[Dict[str, Any]], query: str, *, limit: int, score_threshold: float,
) -> List[Dict[str, Any]]:
    """Port of ``_select_recall_candidates`` (``__init__.py:3338``)."""
    seen_uri = set()
    seen_key = set()
    filtered: List[Dict[str, Any]] = []
    for item in items:
        uri = str(item.get("uri") or "").strip()
        if not uri or uri in seen_uri:
            continue
        if clamp_score(item.get("score")) < score_threshold:
            continue
        key = dedupe_key(item)
        if key in seen_key:
            continue
        seen_uri.add(uri)
        seen_key.add(key)
        filtered.append(item)

    tokens = query_tokens(query)
    filtered.sort(key=lambda item: recall_rank(item, tokens), reverse=True)
    return filtered[:limit]


def format_recall_entry(item: Dict[str, Any], content: str) -> str:
    """Port of the entry shape in ``_build_prefetch_entries`` (``__init__.py:3442``).

    No score and no timestamp: this is the text the plugin injects.
    """
    return "\n".join([
        f"- [{recall_category(item)}]",
        f"  <uri>{item.get('uri', '')}</uri>",
        *[f"  {line}" for line in content.splitlines()],
    ])


# --------------------------------------------------------------------------
# Recall configuration
# --------------------------------------------------------------------------
class RecallConfig:
    """The plugin's recall knobs, with the plugin's own clamps."""

    def __init__(self, limit: int, score_threshold: float, max_injected_chars: int,
                 profile_token_budget: int, timeout_seconds: float,
                 request_timeout_seconds: float, full_read_limit: int,
                 prefer_abstract: bool, resources: bool):
        self.limit = max(1, min(100, limit))
        self.score_threshold = max(0.0, min(1.0, score_threshold))
        self.max_injected_chars = max(100, min(50000, max_injected_chars))
        self.profile_token_budget = max(500, min(50000, profile_token_budget))
        self.timeout_seconds = max(0.25, min(60.0, timeout_seconds))
        self.request_timeout_seconds = max(0.25, min(60.0, request_timeout_seconds))
        self.full_read_limit = max(0, min(100, full_read_limit))
        self.prefer_abstract = prefer_abstract
        self.resources = resources


# --------------------------------------------------------------------------
# Provider binding (the only OpenViking-specific surface the driver sees)
# --------------------------------------------------------------------------
class OpenVikingBinding(ProviderBinding):
    memory_system = "openviking"
    store_id_key = "OpenViking_User_ID"
    runtime_summary_key = "OpenViking_Runtime_Summary"
    stage_name = "openviking_answer_generation"
    stage_note = "OpenViking retrieval and question answering"
    # Set True on the instance in __init__: EVERY mode passes the plugin's
    # own selection width (recall_limit, default 6) to the answer model
    # whole, with no harness top-K slice (user ruling 2026-08-04). The
    # scorer is unaffected — it slices the stored list at its own K.
    plugin_native_recall = False

    def __init__(
        self,
        base_url: str,
        recall_mode: str = "prefetch",
        retain_granularity: str = "exchange",
        cfg: Optional[RecallConfig] = None,
        api_key: str = "",
        account: str = "default",
        user_prefix: str = "",
        agent: str = "hermes",
        send_created_at: bool = False,
        drain_timeout_s: float = 1800.0,
        drain_poll_s: float = 1.0,
        http_timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.recall_mode = recall_mode
        self.retain_granularity = retain_granularity
        self.cfg = cfg or RecallConfig(6, 0.15, 4000, 6000, 60.0, 30.0, 2, False, False)
        self.api_key = api_key
        self.account = account
        self.user_prefix = user_prefix
        self.agent = agent
        self.send_created_at = send_created_at
        self.drain_timeout_s = drain_timeout_s
        self.drain_poll_s = drain_poll_s
        self.http_timeout = http_timeout
        self.server_version: Optional[str] = None
        # Set by the run's setup() once the server is up. Recall failures
        # probe it: the plugin swallows a failed recall, but a dead server
        # is not a per-query miss and must abort instead of emptying every
        # remaining question's context at exit 0.
        self.server: Optional[OpenVikingServer] = None
        # persona_tag is persona_id[-8:], so two personas sharing a suffix
        # would share one user tree and the second begin_persona would wipe
        # the first. Track tag ownership and refuse the collision.
        self._tag_owners: Dict[str, str] = {}
        # All arms: the answer model gets the plugin's own selection width
        # (recall_limit, default 6), not a harness top-K slice (user ruling
        # 2026-08-04). Safe for scoring because the upstream scorer slices
        # the stored list itself: extract_top_k_retrieved_memories takes
        # Retrieved_Memories[:K] at its own K (eval_scoring.py:329-344).
        self.plugin_native_recall = True

    # -- lifecycle -----------------------------------------------------------
    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        persona_tag = persona_id[-8:]
        owner = self._tag_owners.setdefault(persona_tag, persona_id)
        if owner != persona_id:
            raise RuntimeError(
                f"persona tag '{persona_tag}' maps to both '{owner}' and "
                f"'{persona_id}'; the second wipe would delete the first "
                f"persona's memory tree.")
        user_id = sanitize_user_id(f"{self.user_prefix}{persona_tag}")
        client = VikingClient(self.base_url, api_key=self.api_key, account=self.account,
                              user=user_id, agent=self.agent, timeout=self.http_timeout)
        # Wipe this user's memory tree so a re-run under the same tag cannot
        # answer from the previous run's extraction. 404 means the tree does
        # not exist yet, which is the normal first-run state.
        try:
            client.delete("/api/v1/fs", params={"uri": MEMORIES_URI, "recursive": True})
            print(f"[openviking] persona {persona_tag} -> user '{user_id}' (memories wiped)",
                  flush=True)
        except Exception as e:
            if status_code_of(e) != 404:
                print(f"[openviking] memory wipe failed for user {user_id}: {e}", flush=True)
            else:
                print(f"[openviking] persona {persona_tag} -> user '{user_id}' (no prior tree)",
                      flush=True)
        return {
            "store_id": user_id,
            "persona_tag": persona_tag,
            "client": client,
            "current_session_id": None,
            "current_session_index": -1,
            "session_block_cache": {},
            "total_messages": 0,
            "total_batch_posts": 0,
            "add_failures": 0,
            "commit_failures": 0,
            "search_fallbacks": 0,
            "full_reads": 0,
            "empty_recalls": 0,
            "recall_errors": 0,
            "total_drain_ms": 0.0,
            "memories_extracted": {},
        }

    def end_persona(self, ctx: Dict[str, Any]) -> None:
        client = ctx.get("client")
        if client is not None:
            client.close()

    # -- ingestion -----------------------------------------------------------
    def _build_messages(self, dialogue: List[Dict[str, Any]],
                        session_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """One OpenViking message per dialogue turn.

        The assistant message carries ``peer_id``, which is how OpenViking
        attributes the turn to the agent peer rather than the user
        (``_turn_batch_payload``, ``__init__.py:2439``).
        """
        base = Parse_Session_Timestamp(session_item) if self.send_created_at else None
        out: List[Dict[str, Any]] = []
        for index, message in enumerate(dialogue):
            role = "user" if message.get("role") == "user" else "assistant"
            payload: Dict[str, Any] = {
                "role": role,
                "parts": [{"type": "text", "text": str(message.get("content", ""))}],
            }
            if role == "assistant" and self.agent:
                payload["peer_id"] = self.agent
            if base is not None:
                # One second per message inside the session's own date, so the
                # stored order matches the dialogue order.
                payload["created_at"] = (base + timedelta(seconds=index)).isoformat()
            out.append(payload)
        return out

    def _post_batches(self, ctx: Dict[str, Any], sid: str,
                      dialogue: List[Dict[str, Any]],
                      session_item: Dict[str, Any]) -> Tuple[int, int]:
        """Send this session's messages. Returns (messages sent, POST count).

        No session-create call: the first ``messages/batch`` POST creates the
        session, which is what the plugin relies on.
        """
        client: VikingClient = ctx["client"]
        messages = self._build_messages(dialogue, session_item)
        if not messages:
            return 0, 0

        groups: List[List[Dict[str, Any]]] = []
        if self.retain_granularity == "session":
            for start in range(0, len(messages), SESSION_MESSAGE_BATCH_LIMIT):
                groups.append(messages[start:start + SESSION_MESSAGE_BATCH_LIMIT])
        else:
            # Plugin cadence: one POST per completed exchange. Pair_Exchange_Turns
            # groups the same flattened dialogue every provider ingests.
            offset = 0
            for pair in Pair_Exchange_Turns(dialogue):
                groups.append(messages[offset:offset + len(pair)])
                offset += len(pair)

        sent = 0
        posts = 0
        for group in groups:
            if not group:
                continue
            try:
                client.post(f"/api/v1/sessions/{sid}/messages/batch", {"messages": group})
                sent += len(group)
                posts += 1
            except Exception as e:
                ctx["add_failures"] += 1
                print(f"[openviking] messages/batch failed session={sid}: {e}", flush=True)
        if sent == 0:
            # A session whose every batch POST failed must not commit: the
            # extraction would run on nothing, the drain would pass, and the
            # persona would be scored against a store missing this session.
            raise RuntimeError(
                f"openviking ingest sent 0 of {len(messages)} messages for "
                f"session {sid}: every messages/batch POST failed.")
        return sent, posts

    def _commit_and_drain(self, ctx: Dict[str, Any], sid: str) -> Dict[str, Any]:
        """Commit the session, wait for extraction, then wait for the queues.

        DEVIATION FROM THE PLUGIN, and a required one. ``on_session_end``
        posts the commit and returns (``__init__.py:3888``). Extraction runs
        asynchronously, so a benchmark that answered right after the commit
        would measure the extraction queue instead of the memory. Any failure
        RAISES: a persona whose extraction did not finish has an unknown
        memory state, and scoring it would publish a number for a
        configuration that never existed.
        """
        client: VikingClient = ctx["client"]
        start = time.time()
        deadline = start + self.drain_timeout_s

        try:
            response = client.post(f"/api/v1/sessions/{sid}/commit",
                                   {"keep_recent_count": 0})
        except Exception:
            ctx["commit_failures"] += 1
            raise
        commit = unwrap_result(response)
        commit = commit if isinstance(commit, dict) else {}
        status = str(commit.get("status") or "")
        task_id = commit.get("task_id")

        task_status = status
        extracted: Optional[Dict[str, Any]] = None
        if task_id:
            task_status, extracted = self._await_task(client, str(task_id), deadline)
        commit_ms = (time.time() - start) * 1000.0

        queue_start = time.time()
        queues = self._wait_processed(client, deadline)
        queue_ms = (time.time() - queue_start) * 1000.0

        return {
            "Commit_Task_Status": task_status,
            "Commit_Wait_ms": commit_ms,
            "Queue_Wait_ms": queue_ms,
            "Memories_Extracted": extracted,
            "Queue_Report": queues,
        }

    def _await_task(self, client: VikingClient, task_id: str,
                    deadline: float) -> Tuple[str, Optional[Dict[str, Any]]]:
        while True:
            payload = unwrap_result(client.get(f"/api/v1/tasks/{task_id}"))
            payload = payload if isinstance(payload, dict) else {}
            status = str(payload.get("status") or "").lower()
            if status == "completed":
                result = payload.get("result")
                extracted = None
                if isinstance(result, dict):
                    counts = result.get("memories_extracted")
                    extracted = counts if isinstance(counts, dict) else None
                return status, extracted
            if status in ("failed", "cancelled"):
                raise RuntimeError(
                    f"openviking extraction task {task_id} ended {status}: "
                    f"{json.dumps(payload, ensure_ascii=False)[:1000]}")
            if time.time() > deadline:
                raise TimeoutError(
                    f"openviking extraction task {task_id} still '{status}' after "
                    f"{self.drain_timeout_s:.0f}s. A stuck extraction invalidates "
                    f"this persona.")
            time.sleep(self.drain_poll_s)

    def _wait_processed(self, client: VikingClient, deadline: float) -> Dict[str, Any]:
        """Block until the embedding and semantic queues are empty.

        A 2xx from ``/system/wait`` means the server saw every queue complete:
        the handler runs ``wait_complete``, which polls ``is_all_complete``
        and answers only then (``resource_service.py:1781``,
        ``queue_manager.py:404``, verified in the 0.4.12 wheel). Its own
        timeout comes back as non-2xx DEADLINE_EXCEEDED (HTTP 504), so the
        loop re-issues bounded waits until ``drain_timeout_s`` is spent —
        the drain budget belongs to the adapter, not to one request's cap.
        ``error_count`` is the ONLY place a broken embedder surfaces: the
        commit still reports ``accepted`` and the HTTP API stays 200.
        """
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"openviking queues still busy after {self.drain_timeout_s:.0f}s "
                    f"drain budget. An undrained persona has an unknown memory "
                    f"state and must not be scored.")
            timeout_s = int(max(1.0, min(600.0, remaining)))
            try:
                response = client.post("/api/v1/system/wait", {"timeout": timeout_s},
                                       timeout=timeout_s + 30.0)
            except Exception as e:
                if "DEADLINE_EXCEEDED" in str(e):
                    continue
                raise
            report = unwrap_result(response)
            report = report if isinstance(report, dict) else {}
            failures = {
                name: queue for name, queue in report.items()
                if isinstance(queue, dict) and int(queue.get("error_count") or 0) > 0
            }
            if failures:
                raise RuntimeError(
                    f"openviking queues reported errors: "
                    f"{json.dumps(failures, ensure_ascii=False)[:2000]}")
            return report

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        sid = f"{ctx['store_id']}_s{session_index}"
        ctx["current_session_id"] = sid
        ctx["current_session_index"] = session_index
        timestamp = Parse_Session_Timestamp(session_item)

        start = time.time()
        sent, posts = self._post_batches(ctx, sid, dialogue, session_item)
        add_ms = (time.time() - start) * 1000.0
        ctx["total_messages"] += sent
        ctx["total_batch_posts"] += posts

        drain = self._commit_and_drain(ctx, sid)
        ctx["total_drain_ms"] += drain["Commit_Wait_ms"] + drain["Queue_Wait_ms"]
        counts = drain.get("Memories_Extracted")
        if isinstance(counts, dict):
            for key, value in counts.items():
                try:
                    ctx["memories_extracted"][key] = ctx["memories_extracted"].get(key, 0) + int(value)
                except (TypeError, ValueError):
                    continue

        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_item.get('Session_ID', session_index)} "
              f"messages={sent} posts={posts} ingest_ms={add_ms:.0f} "
              f"commit_ms={drain['Commit_Wait_ms']:.0f} queue_ms={drain['Queue_Wait_ms']:.0f} "
              f"extracted={drain.get('Memories_Extracted')}", flush=True)

        return {
            "Add_Duration_ms": add_ms,
            "Dialogue_Added_To_Memory": sent > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Messages_Added": sent,
            "Batch_Post_Count": posts,
            "Retain_Granularity": self.retain_granularity,
            "Session_Timestamp_Passed": timestamp.isoformat() if timestamp else None,
            "Commit_Task_Status": drain["Commit_Task_Status"],
            "Commit_Wait_ms": drain["Commit_Wait_ms"],
            "Queue_Wait_ms": drain["Queue_Wait_ms"],
            "Memories_Extracted": drain["Memories_Extracted"],
        }

    # -- recall: part A, the session-start block ------------------------------
    def _session_start_item(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build (once per session, then cached) the plugin's session-start block.

        The plugin latches this per session id and injects it on the first
        turn only. Every benchmark question is an independent turn, so the
        cached block is attached to each question of the session. The server
        traffic stays the plugin's: one profile read and two listings per
        session.
        """
        index = ctx.get("current_session_index", -1)
        cache: Dict[int, str] = ctx["session_block_cache"]
        if index not in cache:
            cache[index] = self._read_session_start_block(ctx)
        block = cache[index]
        if not block:
            return []
        return [{"memory": block, "created_at": UNKNOWN_TIME, "score": None,
                 "uri": PROFILE_URI, "source": "session_start"}]

    def _read_session_start_block(self, ctx: Dict[str, Any]) -> str:
        client: VikingClient = ctx["client"]
        cfg = self.cfg
        deadline = time.monotonic() + cfg.timeout_seconds
        try:
            timeout = remaining_recall_timeout(deadline, cfg.request_timeout_seconds)
            profile = extract_text_content(
                client.get("/api/v1/content/read", params={"uri": PROFILE_URI},
                           timeout=timeout))
        except Exception as e:
            if status_code_of(e) not in (404, 410):
                # A read error is not an empty profile. The plugin returns no
                # block at all in that case, so neither does this.
                print(f"[openviking] profile read failed: {e}", flush=True)
                return ""
            profile = ""

        preferences = self._list_memories(ctx, PREFERENCES_URI, deadline)
        entities = self._list_memories(ctx, ENTITIES_URI, deadline)
        return build_session_start_memory_block(
            profile=profile, preferences=preferences, entities=entities,
            token_budget=cfg.profile_token_budget)

    def _list_memories(self, ctx: Dict[str, Any], uri: str,
                       deadline: float) -> List[Dict[str, str]]:
        client: VikingClient = ctx["client"]
        try:
            timeout = remaining_recall_timeout(deadline, self.cfg.request_timeout_seconds)
            return extract_memory_listing(client.get(
                "/api/v1/fs/ls", params={"uri": uri, **SESSION_START_LIST_PARAMS},
                timeout=timeout))
        except Exception:
            return []

    # -- recall: part B, the per-query search ---------------------------------
    def _search_candidates(self, ctx: Dict[str, Any], query: str,
                           deadline: float) -> Tuple[List[Dict[str, Any]], Any]:
        """Port of ``_post_prefetch_search`` (``__init__.py:2308``).

        ``search/search`` runs LLM intent analysis and accepts a session id.
        ANY failure falls back ONCE to the deterministic ``search/find``,
        without the session id, which is the plugin's own recovery path.
        """
        client: VikingClient = ctx["client"]
        cfg = self.cfg
        payload = {
            "query": query,
            "limit": max(cfg.limit * 4, 20),
            "score_threshold": 0,
            "context_type": ["memory", "resource"] if cfg.resources else "memory",
        }
        session_id = ctx.get("current_session_id")
        response = None
        if session_id:
            try:
                timeout = remaining_recall_timeout(deadline, cfg.request_timeout_seconds)
                response = client.post("/api/v1/search/search",
                                       {**payload, "session_id": session_id},
                                       timeout=timeout)
            except TimeoutError:
                raise
            except Exception as e:
                ctx["search_fallbacks"] += 1
                print(f"[openviking] search/search failed, falling back to search/find: {e}",
                      flush=True)
        if response is None:
            timeout = remaining_recall_timeout(deadline, cfg.request_timeout_seconds)
            response = client.post("/api/v1/search/find", payload, timeout=timeout)

        result = unwrap_result(response)
        candidates: List[Dict[str, Any]] = []
        if isinstance(result, dict):
            for key in ("memories", "resources"):
                for item in result.get(key) or []:
                    if isinstance(item, dict):
                        candidates.append(item)
        return candidates, response

    def _resolve_content(self, ctx: Dict[str, Any], item: Dict[str, Any], *,
                         deadline: float, read_state: Dict[str, int]) -> str:
        """Port of ``_resolve_recall_content`` (``__init__.py:3378``).

        A level-2 leaf, or an item with no summary at all, is worth one of the
        two full reads. Everything else uses the abstract the search already
        returned.
        """
        client: VikingClient = ctx["client"]
        cfg = self.cfg
        abstract = recall_abstract(item)
        has_summary = any(
            isinstance(item.get(key), str) and item.get(key).strip()
            for key in ("abstract", "overview", "text", "content"))
        if cfg.prefer_abstract and has_summary:
            return abstract
        uri = str(item.get("uri") or "")
        if uri and (item.get("level") == 2 or not has_summary):
            if read_state["full_reads"] >= cfg.full_read_limit:
                return abstract
            try:
                timeout = remaining_recall_timeout(deadline, cfg.request_timeout_seconds)
                read_state["full_reads"] += 1
                ctx["full_reads"] += 1
                content = extract_read_content(client.get(
                    "/api/v1/content/read", params={"uri": uri}, timeout=timeout))
                if content:
                    return content
            except Exception as e:
                print(f"[openviking] full read failed for {uri}: {e}", flush=True)
        return abstract

    def _build_entries(self, ctx: Dict[str, Any], items: List[Dict[str, Any]],
                       deadline: float, source: str) -> List[Dict[str, Any]]:
        """Port of ``_build_prefetch_entries`` (``__init__.py:3416``).

        The character budget uses ``continue``, not ``break``: an oversized
        entry is skipped and a smaller later one can still fit.
        """
        out: List[Dict[str, Any]] = []
        total_chars = 0
        read_state = {"full_reads": 0}
        for item in items:
            content = self._resolve_content(ctx, item, deadline=deadline,
                                            read_state=read_state)
            if not content:
                continue
            entry = format_recall_entry(item, content)
            separator_chars = 1 if out else 0
            projected = total_chars + separator_chars + len(entry)
            if projected > self.cfg.max_injected_chars:
                continue
            out.append({"memory": entry, "created_at": UNKNOWN_TIME,
                        "score": raw_score(item), "uri": str(item.get("uri") or ""),
                        "source": source})
            total_chars = projected
        return out

    def _assert_server_alive(self, error: Exception) -> None:
        server = self.server
        if server is not None and not server.alive():
            raise RuntimeError(
                "openviking server no longer answers its health probe; the "
                f"failed recall is not a per-query miss: {error}") from error

    def _recall_search(self, ctx: Dict[str, Any],
                       question: str) -> Tuple[List[Dict[str, Any]], Any]:
        query = (question or "").strip()
        if len(query) < RECALL_QUERY_MIN_CHARS:
            return [], None
        deadline = time.monotonic() + self.cfg.timeout_seconds
        try:
            candidates, response = self._search_candidates(ctx, query, deadline)
            selected = select_recall_candidates(
                candidates, query, limit=self.cfg.limit,
                score_threshold=self.cfg.score_threshold)
            return self._build_entries(ctx, selected, deadline, "search"), response
        except Exception as e:
            # The plugin swallows a failed recall and injects nothing. Mirror
            # that for a live server, and count it — but a dead server must
            # abort, not empty every remaining question at exit 0.
            self._assert_server_alive(e)
            print(f"[openviking] recall failed: {e}", flush=True)
            ctx["recall_errors"] += 1
            return [], {"error": str(e)}

    def _recall_find(self, ctx: Dict[str, Any],
                     question: str) -> Tuple[List[Dict[str, Any]], Any]:
        """MINIMAL (diagnostic) arm: deterministic retrieval, no LLM intent analysis.

        ``level: [2]`` restricts the hits to leaf memories, so the directory
        ``.abstract.md`` and ``.overview.md`` nodes never occupy a slot. The
        server is asked for FIND_LIMIT (10, the endpoint default) and the
        arm keeps ``recall_limit`` of them; the full hit list stays in the
        diagnostic capture. Content is read only for kept items.
        """
        client: VikingClient = ctx["client"]
        query = (question or "").strip()
        if not query:
            return [], None
        deadline = time.monotonic() + self.cfg.timeout_seconds
        try:
            timeout = remaining_recall_timeout(deadline, self.cfg.request_timeout_seconds)
            response = client.post("/api/v1/search/find", {
                "query": query,
                "target_uri": MEMORIES_URI,
                "limit": FIND_LIMIT,
                "context_type": "memory",
                "level": [2],
            }, timeout=timeout)
        except Exception as e:
            self._assert_server_alive(e)
            print(f"[openviking] search/find failed: {e}", flush=True)
            ctx["recall_errors"] += 1
            return [], {"error": str(e)}

        result = unwrap_result(response)
        items: List[Dict[str, Any]] = []
        hits = result.get("memories") if isinstance(result, dict) else None
        for item in hits or []:
            if len(items) >= self.cfg.limit:
                break
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "")
            content = ""
            if uri:
                try:
                    timeout = remaining_recall_timeout(deadline,
                                                       self.cfg.request_timeout_seconds)
                    ctx["full_reads"] += 1
                    content = extract_read_content(client.get(
                        "/api/v1/content/read", params={"uri": uri}, timeout=timeout))
                except Exception:
                    content = ""
            content = content or recall_abstract(item)
            if not content:
                continue
            items.append({"memory": content, "created_at": UNKNOWN_TIME,
                          "score": raw_score(item), "uri": uri, "source": "find"})
        return items, response

    def recall(self, ctx, question_text, top_k):
        start = time.time()
        if self.recall_mode == "find":
            items, response = self._recall_find(ctx, question_text)
        elif self.recall_mode == "search":
            items, response = self._recall_search(ctx, question_text)
        else:
            search_items, response = self._recall_search(ctx, question_text)
            items = self._session_start_item(ctx) + search_items
        duration_ms = (time.time() - start) * 1000.0

        raw = {"recall_mode": self.recall_mode, "response": response,
               "item_count": len(items)}
        # The capture keeps the full selected list, before any slice, so an
        # offline retrieval curve stays computable without a re-run.
        record_provider_retrieval(ctx, raw=raw, ranked=items)
        if not items:
            ctx["empty_recalls"] += 1
        # No top_k slice in any arm: every mode already caps at
        # cfg.limit, and plugin_native_recall passes the list through
        # whole (user ruling 2026-08-04). top_k only stamps the scorer's
        # white-box depth.
        return items, duration_ms

    # -- runtime summary -----------------------------------------------------
    def persona_count_extras(self, ctx):
        return {
            "Total_Messages_Added": ctx["total_messages"],
            "Total_Batch_Posts": ctx["total_batch_posts"],
            "Total_Add_Failures": ctx["add_failures"],
            "Commit_Failure_Count": ctx["commit_failures"],
            "Total_Drain_Time_ms": ctx["total_drain_ms"],
        }

    def persona_tail_extras(self, ctx):
        return {
            "Recall_Mode": self.recall_mode,
            "Retain_Granularity": self.retain_granularity,
            # A nonzero fallback count means search/search (the LLM intent
            # path) failed and the run measured search/find instead.
            "Search_Fallback_Count": ctx["search_fallbacks"],
            "Full_Read_Count": ctx["full_reads"],
            # Questions whose recall returned nothing, and how many of those
            # came from a raised request rather than an empty result set.
            "Empty_Recall_Count": ctx["empty_recalls"],
            "Recall_Error_Count": ctx["recall_errors"],
            # Per-type extraction counts, summed over the persona's sessions.
            # All zero means the extraction model produced nothing.
            "Memories_Extracted_Total": dict(ctx["memories_extracted"]),
        }

    def persona_result_extras(self, ctx):
        return {
            "OpenViking_Recall_Mode": self.recall_mode,
            "OpenViking_Retain_Granularity": self.retain_granularity,
            "OpenViking_Send_Created_At": self.send_created_at,
            "OpenViking_Server_Version": self.server_version,
        }


def Generate_User_OpenViking_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    binding: "OpenVikingBinding",
    server: OpenVikingServer,
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
) -> bool:
    def setup():
        binding.base_url = server.start()
        binding.server_version = server.version
        binding.server = server

    def teardown():
        server.close()

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
    parser = argparse.ArgumentParser(
        description="Run OpenViking evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(
            CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(
            CURRENT_DIR, "Results", "openviking_results.jsonl"),
        default_output_json_path=os.path.join(
            CURRENT_DIR, "Results", "openviking_results.json"),
        top_k_help="Scorer white-box depth only (stamps Actual_Top_K). The answer "
                   "model always sees the plugin's own selection width (recall_limit, "
                   "default 6) in every arm — user ruling 2026-08-04; the scorer "
                   "slices the stored list itself.",
        default_start_idx=int(_env("START_IDX", "0")),
        default_end_idx=opt_int(_env("END_IDX")),
    )
    # --- recall arm ---------------------------------------------------------
    parser.add_argument("--recall_mode", type=str,
                        default=_env("OPENVIKING_RECALL_MODE", "prefetch"),
                        choices=["prefetch", "search", "find"],
                        help="'prefetch' (default, FEATURED, plugin-faithful): the "
                             "session-start block plus the per-query search entries, "
                             "injected whole — the arm that is scored and enters the "
                             "final comparison. 'find' (MINIMAL, diagnostic): "
                             "deterministic retrieval (POST /api/v1/search/find, level 2 "
                             "leaves, full content bodies, recall_limit items kept), no "
                             "LLM intent analysis — the integration proof and retrieval "
                             "floor, not a comparison number. 'search': the search "
                             "entries alone, an unplanned auxiliary reduction.")
    parser.add_argument("--retain_granularity", type=str,
                        default=_env("OPENVIKING_RETAIN_GRANULARITY",
                                     _env("RETAIN_GRANULARITY", "exchange")),
                        choices=["exchange", "session"],
                        help="'exchange' (default) is the plugin cadence: one "
                             "messages/batch POST per completed exchange (sync_turn, "
                             "__init__.py:3706). 'session' sends the whole session in "
                             "100-message chunks.")
    parser.add_argument("--send_created_at", type=int,
                        default=int(_truthy(_env("OPENVIKING_SEND_CREATED_AT"), False)),
                        help="1 stamps each message with the session Date plus its index "
                             "in seconds. The plugin never sends created_at, so the "
                             "default 0 is the plugin-faithful path; BENCH_CLOCKSYNC "
                             "moves the server's own clock instead.")
    # --- recall knobs (plugin names, plugin defaults except the timeouts) ----
    parser.add_argument("--recall_limit", type=int,
                        default=int(_env("OPENVIKING_RECALL_LIMIT", "6")),
                        help="How many candidates survive client-side selection. The "
                             "search asks for max(4x, 20).")
    parser.add_argument("--recall_score_threshold", type=float,
                        default=float(_env("OPENVIKING_RECALL_SCORE_THRESHOLD", "0.15")))
    parser.add_argument("--recall_max_injected_chars", type=int,
                        default=int(_env("OPENVIKING_RECALL_MAX_INJECTED_CHARS", "4000")))
    parser.add_argument("--profile_token_budget", type=int,
                        default=int(_env("OPENVIKING_PROFILE_TOKEN_BUDGET", "6000")),
                        help="Token budget for the session-start block. Spent in "
                             "quarter-token units: profile capped at half, remainder "
                             "split between the preferences and entities listings.")
    parser.add_argument("--recall_full_read_limit", type=int,
                        default=int(_env("OPENVIKING_RECALL_FULL_READ_LIMIT", "2")),
                        help="Per question, how many selected items may spend a "
                             "content/read call for the full body instead of the abstract.")
    parser.add_argument("--recall_prefer_abstract", type=int,
                        default=int(_truthy(_env("OPENVIKING_RECALL_PREFER_ABSTRACT"), False)))
    parser.add_argument("--recall_resources", type=int,
                        default=int(_truthy(_env("OPENVIKING_RECALL_RESOURCES"), False)),
                        help="1 adds context_type 'resource' to the search, so ingested "
                             "documents compete with extracted memories.")
    parser.add_argument("--recall_timeout_s", type=float,
                        default=float(_env("OPENVIKING_RECALL_TIMEOUT_SECONDS", "60")),
                        help="Whole-recall budget. The plugin ships 4.0, tuned for an "
                             "interactive CLI where an empty recall beats a stalled turn; "
                             "under benchmark serving latency that empties recall "
                             "silently. 60 is the plugin's own clamp maximum.")
    parser.add_argument("--recall_request_timeout_s", type=float,
                        default=float(_env("OPENVIKING_RECALL_REQUEST_TIMEOUT_SECONDS", "30")),
                        help="Per-request slice of the recall budget. Plugin default 3.0; "
                             "same reason as --recall_timeout_s.")
    # --- drain --------------------------------------------------------------
    parser.add_argument("--drain_timeout_s", type=float,
                        default=float(_env("OPENVIKING_DRAIN_TIMEOUT_S", "1800")))
    parser.add_argument("--drain_poll_s", type=float,
                        default=float(_env("OPENVIKING_DRAIN_POLL_S", "1.0")))
    # --- connection / identity ----------------------------------------------
    parser.add_argument("--base_url", type=str, default=_env("OPENVIKING_ENDPOINT"),
                        help="Attach to an already-running OpenViking server instead of "
                             "spawning one.")
    parser.add_argument("--api_key", type=str, default=_env("OPENVIKING_API_KEY", ""),
                        help="Empty (default) means dev auth mode: identity comes from "
                             "the X-OpenViking-Account and X-OpenViking-User headers.")
    parser.add_argument("--account", type=str, default=_env("OPENVIKING_ACCOUNT", "default"))
    parser.add_argument("--user_prefix", type=str, default=_env("OPENVIKING_USER_PREFIX", ""),
                        help="Prefix for the per-persona X-OpenViking-User value. A "
                             "shared or attached server needs the run tag here, so two "
                             "runs cannot share a persona's memory tree.")
    parser.add_argument("--agent", type=str, default=_env("OPENVIKING_AGENT", "hermes"),
                        help="Actor peer header and the assistant messages' peer_id.")
    parser.add_argument("--http_timeout", type=float,
                        default=float(_env("OPENVIKING_HTTP_TIMEOUT", "600")),
                        help="Default HTTP timeout for ingest and commit. Recall uses the "
                             "recall budget instead. Commit is extraction-model-bound.")
    args = parser.parse_args()

    cfg = RecallConfig(
        limit=args.recall_limit,
        score_threshold=args.recall_score_threshold,
        max_injected_chars=args.recall_max_injected_chars,
        profile_token_budget=args.profile_token_budget,
        timeout_seconds=args.recall_timeout_s,
        request_timeout_seconds=args.recall_request_timeout_s,
        full_read_limit=args.recall_full_read_limit,
        prefer_abstract=bool(args.recall_prefer_abstract),
        resources=bool(args.recall_resources),
    )
    binding = OpenVikingBinding(
        base_url=args.base_url or "",
        recall_mode=args.recall_mode,
        retain_granularity=args.retain_granularity,
        cfg=cfg,
        api_key=args.api_key,
        account=args.account,
        user_prefix=args.user_prefix,
        agent=args.agent,
        send_created_at=bool(args.send_created_at),
        drain_timeout_s=args.drain_timeout_s,
        drain_poll_s=args.drain_poll_s,
        http_timeout=args.http_timeout,
    )
    # ATTACH when a base URL is given (a shared central server). SPAWN
    # otherwise: this process owns the server, its config, and its workspace.
    server = OpenVikingServer(base_url=args.base_url)
    print(f"[DEBUG] recall_mode={args.recall_mode} "
          f"retain_granularity={args.retain_granularity} "
          f"send_created_at={bool(args.send_created_at)} "
          f"plugin_native_recall={binding.plugin_native_recall} "
          f"recall_limit={cfg.limit} threshold={cfg.score_threshold} "
          f"timeout={cfg.timeout_seconds}/{cfg.request_timeout_seconds}s "
          f"mode={'attach' if args.base_url else 'spawn'}", flush=True)

    # run_eval() returns False, not an exception, on a fatal error, so
    # per-persona incremental output survives a mid-run crash. Propagate that
    # as a nonzero exit, so the entrypoint's `set -e` stops the run instead of
    # scoring a partial file.
    ok = Generate_User_OpenViking_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        binding=binding,
        server=server,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )
    sys.exit(0 if ok else 1)
