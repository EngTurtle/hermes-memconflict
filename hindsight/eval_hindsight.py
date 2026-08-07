"""MemConflict evaluation adapter for the Hindsight memory system.

The shared driver ``benchmark/eval_common.py`` holds the provider-agnostic
pipeline. That pipeline does dataset iteration, dialogue flattening, the answer
prompt and answer LLM call, results-row emission, and compaction. Every
provider uses the same driver, so those stages are identical by construction.
This file supplies only the Hindsight binding: daemon setup, retain ingestion
with consolidation-drain waits, and recall.

WHAT IS DIFFERENT ABOUT HINDSIGHT (vs. Mnemosyne / Mem0)
--------------------------------------------------------
Hindsight is NOT a raw message store. ``retain()`` runs an LLM fact-extraction
pass over the supplied content. It stores the extracted facts plus optional
entity and graph links, and a background worker then consolidates them.
``recall()`` queries those extracted facts, not the verbatim messages.
``recall()`` is token-budgeted (``max_tokens`` / ``budget``) rather than top-K.
Two direct consequences for this adapter follow.

  * Ingestion drives the Hindsight LLM. By default this adapter sends one
    ``retain()`` per session with the whole role-prefixed session dialogue.
    Extraction is then one LLM call per session, not one per message.
    ``--retain_granularity message`` restores the per-message behavior of the
    Mnemosyne adapter. That mode is much slower, because it makes one
    extraction call per message.
  * The adapter passes the dataset's simulated session date as
    ``retain(timestamp=)``. Hindsight's temporal machinery (recency decay,
    occurred_at) then sees the benchmark chronology. This is important for the
    Dynamic (temporal) conflicts in MemConflict.

A unique ``bank_id`` in a single embedded daemon isolates each persona. A bank
is Hindsight's native tenancy boundary, so a separate DB file is not necessary.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# --- Make the upstream MemConflict Evaluation helpers importable -----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMCONFLICT_EVAL_DIR = os.environ.get(
    "MEMCONFLICT_EVAL_DIR",
    os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Evaluation"),
)
MEMCONFLICT_EVAL_DIR = os.path.abspath(MEMCONFLICT_EVAL_DIR)
if MEMCONFLICT_EVAL_DIR not in sys.path:
    sys.path.insert(0, MEMCONFLICT_EVAL_DIR)

# The shared provider-agnostic harness modules live in ../benchmark. They are
# eval_common, llm_reasoning, and the scorers. This insert makes them
# importable from any launch directory. Keep it.
_SHARED_HARNESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark"))
if _SHARED_HARNESS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_HARNESS_DIR)

from dotenv import load_dotenv

import eval_common  # noqa: E402
from eval_common import (  # noqa: E402  (re-exports keep old imports working)
    Build_Session_Dialogue_List,  # re-exported for _smoke_retain_one.py
    Pair_Exchange_Turns,
    Parse_Query_Now_Timestamp,
    Parse_Session_Timestamp,
    ProviderBinding,
    add_common_eval_args,
    opt_int,
    record_provider_retrieval,
)

# Setup imports Hindsight lazily. The caller sets the provider, base_url,
# embeddings, and reranker environment variables first. A top-level import
# would read them too early.


class StrictQualityRunError(RuntimeError):
    """Signals a broken quality-arm invariant under --strict_quality_run only.

    Two conditions raise it. A consolidation drain timed out or could not be
    polled. An exchange_append arm left the plugin-faithful stable-document
    'append' path. Both silently degrade what the arm measures. Default
    exploratory runs never raise this and keep the tolerant log-and-continue
    behavior.

    The raise propagates from ingest_session to run_eval, which
    prints the traceback and returns False, and then to __main__, which exits
    nonzero. The shard aborts loudly instead of scoring a mis-measured file.
    The message names the exact condition and the persona-bank and session that
    tripped it, so a 5-9h re-run is diagnosable from the container log alone."""


load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))


# --------------------------------------------------------------------------
# Hindsight backend
# --------------------------------------------------------------------------
def _inject_daemon_clock_env() -> Optional[str]:
    """Preload libfaketime into every process HindsightEmbedded spawns.

    Returns the .so path when the injection engaged, None when it did not.

    WHICH PROCESSES THIS REACHES. HindsightEmbedded starts the hindsight-api
    daemon with ``subprocess.Popen(cmd, env=os.environ.copy())``, so the daemon
    inherits whatever this function writes into ``os.environ``. The daemon then
    starts the embedded pg0 cluster through the pg0 SDK, which passes no
    ``env=`` of its own, and the pg0 Rust CLI orphans the postmaster with the
    daemon's environment. One write here therefore fakes the daemon, initdb,
    and the postmaster. LD_PRELOAD binds at exec, so THIS adapter process keeps
    the real clock: the time.time() deadlines it owns
    (Wait_For_Consolidation_Drain and the HINDSIGHT_CLIENT_TIMEOUT set below)
    stay wall-clock and cannot be stretched by a faked forward jump.

    WHY THE FEATURED ARM NEEDS IT. exchange_append retains under a stable
    document_id with update_mode="append". The append merge drops the caller's
    ``timestamp=``, and content build falls back to utcnow(), so the daemon's
    OS clock is what stamps ``mentioned_at``. The ftclk1_p0 smoke ran that path
    un-faked and stamped 2026 dates on a 2022 dataset: update order
    recognition 0.547 -> 0.305, micro answer accuracy 0.475 -> 0.344. A missing
    .so raises instead of warning, because a silently un-faked daemon
    reproduces that defect and passes the strict run-contract gate.

    THREE INDEPENDENT REASONS TO DO NOTHING. Clock-sync off, no timestamp file,
    or an external shared Postgres. The last one matters most: the minimal arm
    sets HINDSIGHT_EMBED_API_DATABASE_URL and writes into a co-tenant cluster
    that other shards read at real time, so faking it there would corrupt their
    rows. The FAKETIME_* values match the contract that clock_sync.py and
    benchmark/docker/clock_sync.sh define: NO_CACHE so a rewritten file steps a
    live process, DONT_FAKE_MONOTONIC so timeouts and sleeps stay real, and
    NO_FAKE_STAT so observed file mtimes stay real. The variables stay set for
    the life of the process, so a daemon respawn is faked too.
    """
    if os.environ.get("BENCH_CLOCKSYNC") != "1":
        return None
    if not os.environ.get("BENCH_CLOCKSYNC_FILE"):
        return None
    if os.environ.get("HINDSIGHT_EMBED_API_DATABASE_URL"):
        return None

    libfaketime = os.environ.get(
        "BENCH_LIBFAKETIME",
        "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1",
    )
    if not os.path.exists(libfaketime):
        raise RuntimeError(
            f"BENCH_CLOCKSYNC=1 with embedded pg0, but libfaketime is missing at "
            f"{libfaketime}. The daemon would stamp appended facts with wall-clock "
            f"dates and the run would score as valid. Rebuild the image or set "
            f"BENCH_LIBFAKETIME."
        )

    os.environ["LD_PRELOAD"] = libfaketime
    os.environ["FAKETIME_TIMESTAMP_FILE"] = os.environ["BENCH_CLOCKSYNC_FILE"]
    os.environ["FAKETIME_NO_CACHE"] = "1"
    os.environ["FAKETIME_DONT_FAKE_MONOTONIC"] = "1"
    os.environ["NO_FAKE_STAT"] = "1"
    return libfaketime


def Setup_Hindsight(profile: str):
    """Construct an embedded Hindsight daemon.

    This function configures only the daemon LLM. That LLM does fact extraction
    on retain and query understanding on recall. The ``HINDSIGHT_API_*``
    environment variables control everything else, and the daemon reads them at
    startup. They cover the embeddings (default local BAAI/bge-small-en-v1.5),
    the reranker (default local cross-encoder), the retrieval strategy weights,
    and consolidation. See docs/BENCHMARK_MATRIX.md.
    """
    from hindsight import HindsightEmbedded

    provider = os.environ.get("HINDSIGHT_LLM_PROVIDER", "openai")
    model = os.environ.get("HINDSIGHT_LLM_MODEL", "openai/gpt-oss-120b")
    base_url = os.environ.get("HINDSIGHT_LLM_BASE_URL") or None
    api_key = (
        os.environ.get("HINDSIGHT_LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    log_level = os.environ.get("HINDSIGHT_LOG_LEVEL", "warning")

    # Storage backend. By default HindsightEmbedded starts a per-profile
    # embedded pg0 cluster. A postgresql:// value in
    # HINDSIGHT_EMBED_API_DATABASE_URL points the daemon at an EXTERNAL shared
    # Postgres instead. entrypoint.hindsight.sh sets it when
    # HINDSIGHT_PG_MODE=shared. The embed manager maps the value onto the
    # daemon's HINDSIGHT_API_DATABASE_URL, parse_pg0_url() then reports
    # is_pg0=False, and no embedded Postgres starts. Unset gives None and the
    # embedded pg0, which is byte-identical to the legacy behavior.
    database_url = os.environ.get("HINDSIGHT_EMBED_API_DATABASE_URL") or None

    # Must run BEFORE the constructor. HindsightEmbedded spawns the daemon from
    # os.environ, and LD_PRELOAD binds at exec, so a later write reaches
    # nothing.
    _faketime_so = _inject_daemon_clock_env()
    if _faketime_so:
        print(f"[DEBUG] clock-sync: daemon and embedded pg0 preload {_faketime_so} "
              f"(file={os.environ['FAKETIME_TIMESTAMP_FILE']})")

    client = HindsightEmbedded(
        profile=profile,
        llm_provider=provider,
        llm_api_key=api_key,
        llm_model=model,
        llm_base_url=base_url,
        database_url=database_url,
        log_level=log_level,
    )
    # Touch the daemon so the slow first boot happens here, not mid-ingest.
    # The first boot initializes the embedded Postgres and loads the local
    # embedding and reranker models.
    client._ensure_started()

    # Raise the HTTP client timeout above the 300s default. One retain() runs
    # synchronous LLM fact-extraction over every chunk of a session. On a slow
    # model such as gpt-oss-120b a large session can exceed 300s and raise a
    # bare TimeoutError. The underlying client reads self._timeout per request.
    try:
        timeout_s = float(os.environ.get("HINDSIGHT_CLIENT_TIMEOUT", "900"))
        inner = getattr(client, "_client", None) or getattr(client, "client", None)
        if inner is not None and hasattr(inner, "_timeout"):
            inner._timeout = timeout_s
    except Exception:
        pass
    return client


def _retain_one(client, bank_id: str, content: str, timestamp: Optional[datetime],
                context: Optional[str]) -> bool:
    """One synchronous retain() call. Returns True if it succeeded.

    retain_async=False makes the LLM fact-extraction and the storage complete
    before the adapter recalls against the bank. The call is best-effort. One
    failed extraction must not abort the whole persona.
    """
    try:
        client.retain(
            bank_id=bank_id,
            content=content,
            timestamp=timestamp,
            context=context,
            retain_async=False,
        )
        return True
    except Exception as e:  # pragma: no cover
        # A hindsight_client ApiException often has an empty str(). Print the
        # status, the body, and the exception type to keep failures
        # diagnosable.
        detail = str(e) or ""
        status = getattr(e, "status", None)
        body = getattr(e, "body", None) or getattr(e, "data", None)
        print(f"[DEBUG] retain failed for bank={bank_id}: "
              f"type={type(e).__name__} status={status} detail={detail!r} body={str(body)[:400]!r}")
        return False


def _truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


# --------------------------------------------------------------------------
# 'exchange_append' granularity: plugin-faithful stable-document append
# --------------------------------------------------------------------------
# This path mirrors the official Hermes memory plugin
# (NousResearch/hermes-agent plugins/memory/hindsight/__init__.py @ 977884e6).
# It sends one retain per completed user/assistant exchange. Each retain ships
# ONLY the newly-completed turns under the STABLE bare session document_id with
# update_mode="append". Append needs Hindsight >= 0.5.0, and the pinned image
# is hindsight-all==0.8.6.
#
# Each message content string carries the prefix "User: " or "Assistant: " and
# a timestamp. The retain item content is a JSON-array string of the turn
# message dicts. The plugin calls aretain_batch(..., retain_async=True). A
# QUALITY arm must be drained, so this adapter uses the blocking synchronous
# client.retain(retain_async=False).
#
# Runtime fallback. If this client rejects document_id or update_mode with a
# TypeError or an API error, the adapter drops ONCE and run-wide to the
# plugin's legacy behavior for non-append APIs. That behavior resends the FULL
# accumulated session under a per-run unique document id and omits
# update_mode. The adapter records which mode ran, so the manifest and the
# report can cite it.

_APPEND_FALLBACK_LOGGED = False


def _log_append_fallback_once(reason: str) -> None:
    global _APPEND_FALLBACK_LOGGED
    if not _APPEND_FALLBACK_LOGGED:
        _APPEND_FALLBACK_LOGGED = True
        print("[WARN] exchange_append: this Hindsight client rejected stable-document "
              "append (document_id/update_mode); falling back to legacy accumulated-resend "
              f"under a per-run unique document id for the REST of this run. reason: {reason}")


def _format_turn_message(role: str, content: str, ts_iso: Optional[str],
                         max_chars: Optional[int]) -> str:
    """Build one turn message as the pinned plugin ships it.

    The result is a JSON object string with the explicit role, the
    capitalized-role-prefixed content, and a timestamp. The keys are ``role``,
    ``content``, and ``timestamp``, in the same order as
    ``_build_turn_messages()`` in hermes-agent
    plugins/memory/hindsight/__init__.py @ 977884e6. Before 2026-07-21 this
    function dropped the ``role`` key. That was a fidelity bug, because the
    plugin sends the key and Arm C extraction inputs then differed from
    production."""
    prefix = "User: " if role == "user" else "Assistant: "
    text = _truncate(f"{prefix}{content}", max_chars)
    return json.dumps({"role": role, "content": text, "timestamp": ts_iso},
                      ensure_ascii=False)


def _turns_to_content(turn_jsons: List[str]) -> str:
    """Join the turn message objects into the JSON-array string the plugin sends."""
    return "[" + ",".join(turn_jsons) + "]"


def _retain_stable_append(client, bank_id: str, content: str,
                          timestamp: Optional[datetime], context: Optional[str],
                          document_id: str, metadata: Dict[str, Any]) -> None:
    """Send one blocking stable-document append retain. Raises on failure.

    The caller's capability probe decides whether a raise means that the client
    does not support append and the run must go legacy.

    ``metadata`` is best-effort. If the installed client's ``retain`` has no
    ``metadata`` kwarg, it raises TypeError, and this function retries without
    the kwarg. A missing metadata kwarg must NOT look like missing append
    support and force the whole run to legacy.
    """
    try:
        client.retain(
            bank_id=bank_id, content=content, timestamp=timestamp, context=context,
            document_id=document_id, update_mode="append", metadata=metadata,
            retain_async=False,
        )
    except TypeError:
        # metadata= may be the unsupported kwarg, so retry without it. A
        # remaining TypeError means document_id or update_mode is unsupported.
        # That TypeError propagates to the probe.
        client.retain(
            bank_id=bank_id, content=content, timestamp=timestamp, context=context,
            document_id=document_id, update_mode="append", retain_async=False,
        )


def _extract_http_status(e: Exception) -> Optional[int]:
    """Read the HTTP status code from any exception shape the client raises.

    The installed client is hindsight_client 0.8.6, verified against the
    downloaded wheel. It raises
    ``hindsight_client_api.exceptions.ApiException`` and its 4xx and 5xx
    subclasses BadRequestException, NotFoundException, and ServiceException.
    Those carry the code on ``.status``. Other HTTP stacks in the dependency
    tree, such as httpx, aiohttp, and requests-style wrappers, expose it as
    ``.status_code`` or as a nested ``.response.status_code``. This function
    probes all three shapes. A missed attribute would make a real 4xx look
    statusless, that is transient, or the reverse."""
    for attr in ("status", "status_code"):
        value = getattr(e, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    response = getattr(e, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None) or getattr(response, "status", None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _is_capability_error(e: Exception) -> bool:
    """Return True only for errors that PROVE the API lacks append support.

    Two errors prove it. A TypeError means the client wrapper rejects the
    ``document_id`` or ``update_mode`` kwarg. A non-transient 4xx means a
    validation rejection.

    A transient status must NOT latch the run into legacy accumulated-resend
    mode. Review finding B1: a cold-start timeout on the first retain would
    otherwise run the whole arm in the wrong ingestion mode. This function
    treats three cases as transient, and therefore not a capability error.
    Anything statusless is transient, because timeouts and connection resets
    carry no HTTP code. Any 5xx is transient. Two 4xx codes are load or timing
    artifacts rather than a schema rejection:
      * 408 Request Timeout
      * 429 Too Many Requests (rate limit)
    Only a real validation rejection proves the server does not understand
    stable-document append. That is a 4xx other than 408 and 429, for example
    400, 404, or 422. Those return True here and drop the run to legacy. The
    caller re-raises or re-probes everything else as a retryable failure."""
    if isinstance(e, TypeError):
        return True
    status = _extract_http_status(e)
    if status is None:
        return False
    if status in (408, 429):
        return False
    return 400 <= status < 500


def _retain_legacy_accumulated(client, bank_id: str, content: str,
                               timestamp: Optional[datetime], context: Optional[str],
                               document_id: str) -> bool:
    """Run the legacy non-append behavior.

    This resends the FULL accumulated session under a per-run unique
    document_id and omits update_mode. If the client also rejects document_id
    with a TypeError, this degrades to a bare ``_retain_one`` retain."""
    try:
        client.retain(
            bank_id=bank_id, content=content, timestamp=timestamp, context=context,
            document_id=document_id, retain_async=False,
        )
        return True
    except TypeError:
        return _retain_one(client, bank_id, content, timestamp, context)
    except Exception as e:  # pragma: no cover
        detail = str(e) or ""
        status = getattr(e, "status", None)
        print(f"[DEBUG] legacy accumulated retain failed for bank={bank_id}: "
              f"type={type(e).__name__} status={status} detail={detail!r}")
        return False


def _add_session_exchange_append(
    client, bank_id: str, dialogue_messages: List[Dict[str, Any]],
    session_base: Optional[datetime], session_label: str,
    max_chars: Optional[int], append_state: Dict[str, Any],
    strict_quality_run: bool = False,
) -> Tuple[int, int, Optional[str]]:
    """Ingest one session as plugin-faithful stable-document appends.

    This sends one blocking retain per ``Pair_Exchange_Turns`` exchange. A lone
    unpaired message forms its own delta. The append document_id is the STABLE
    per-session id ``f"{bank_id}_doc_{session_label}"``. Each retain ships only
    the newly-completed turns of that exchange with ``update_mode="append"``.

    LOGICAL TIMESTAMP (B4b, intra-session recency parity with Mnemosyne). The
    retain of each exchange uses a MONOTONICALLY ADVANCING logical timestamp,
    ``session_base + timedelta(minutes=exchange_index)``. The Mnemosyne adapter
    does the same per turn with ``session_base + timedelta(minutes=
    turn_index)``. The old behavior stamped every retain in a session with one
    tied session-level timestamp. That gave Hindsight ordered-but-tied recency
    where Mnemosyne got a strict order, and it biased the temporal (Dynamic)
    conflict resolution of MemConflict between the two providers.

    The adapter passes the advancing value as ``retain(timestamp=)``, which is
    the occurred_at and recency anchor of Hindsight. It also embeds the value
    in the ``"timestamp"`` field of each plugin turn message. The retain
    metadata keeps the raw dataset session date untouched as ``session_date``
    (str), so nothing is lost.

    ``strict_quality_run``: when True, a capability fallback off the append
    path raises ``StrictQualityRunError``. Such a fallback means the client of
    this run does not support stable-document append. The alternative is a
    silent downgrade to legacy accumulated-resend, which would invalidate a
    headline Arm-C run.

    ``append_state`` is the RUN-level mode latch and persists across personas.
    It holds ``mode`` in {None (undecided), "append", "accumulated_resend"} and
    a ``run_uid`` for the legacy per-run-unique document id. The capability
    probe is the first exchange retain of the run. Success sets "append". A
    CAPABILITY error (TypeError or 4xx validation, see
    ``_is_capability_error``) sets "accumulated_resend", logs once, and redoes
    this exchange through the legacy path. A TRANSIENT error (timeout, 5xx, or
    connection) counts as a failed retain and leaves the mode undecided, so the
    next exchange probes again. Once "append" is set, later transient failures
    only count and never flip the mode. Returns (retain_ok, retain_failed,
    mode_used).
    """
    ok = 0
    failed = 0
    # The metadata keeps the raw dataset session date verbatim as a str. The
    # advancing per-exchange logical timestamp below therefore never loses the
    # source chronology.
    session_date_iso = session_base.isoformat() if session_base else None
    context = f"MemConflict dialogue session {session_label}"
    document_id = f"{bank_id}_doc_{session_label}"       # STABLE across this session
    run_uid = append_state.setdefault("run_uid", uuid.uuid4().hex[:8])
    legacy_document_id = f"{session_label}-{run_uid}"    # per-run unique (legacy)
    accumulated_turns: List[str] = []                    # for legacy accumulated-resend
    retain_durations_ms: List[float] = []  # timing only, one entry per retain attempt
    exchanges = Pair_Exchange_Turns(dialogue_messages)
    total_calls = len(exchanges)

    for turn_index, exchange in enumerate(exchanges):
        call_n = turn_index + 1
        # Monotonically advancing logical timestamp for THIS exchange (B4b).
        # It is the occurred_at passed to retain() and embedded in each turn
        # message. It mirrors Mnemosyne's session_base +
        # timedelta(minutes=index). It is None-safe: with no parseable session
        # date every exchange stays None, exactly as before, because there is
        # no ordering information to invent.
        ex_ts = session_base + timedelta(minutes=turn_index) if session_base else None
        ex_ts_iso = ex_ts.isoformat() if ex_ts else None
        delta_turns = [
            _format_turn_message(m["role"], m["content"], ex_ts_iso, max_chars)
            for m in exchange
        ]
        accumulated_turns.extend(delta_turns)
        # All values are str(). The MemoryItem of hindsight_client validates
        # metadata as Dict[str, str], so int values fail pydantic validation on
        # the client side. The append smoke caught this: every retain returned
        # 'ValidationError' in about 10ms. session_date keeps the raw dataset
        # date even though the logical retain timestamp advances per exchange.
        metadata = {
            "retained_at": datetime.now(timezone.utc).isoformat(),
            "message_count": str(len(exchange)),  # one full exchange delta = 2 messages
            "turn_index": str(turn_index),
            "session_date": str(session_date_iso),
        }
        delta_content = _turns_to_content(delta_turns)

        mode = append_state.get("mode")
        if mode is None:
            # Capability probe: first exchange retain of the whole run.
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_start call={call_n}/{total_calls} chars={len(delta_content)}")
            _t0 = time.time()
            try:
                _retain_stable_append(client, bank_id, delta_content,
                                      ex_ts, context, document_id, metadata)
                _dur = (time.time() - _t0) * 1000.0
                retain_durations_ms.append(_dur)
                print(f"[DEBUG] bank={bank_id} session {session_label} "
                      f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok=True")
                append_state["mode"] = "append"
                ok += 1
            except Exception as e:
                _dur = (time.time() - _t0) * 1000.0
                retain_durations_ms.append(_dur)
                print(f"[DEBUG] bank={bank_id} session {session_label} "
                      f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok=False")
                if _is_capability_error(e):
                    # B5: a capability fallback silently moves the whole arm
                    # off the plugin-faithful append path. For a headline
                    # quality run that is a mis-measurement worth an abort.
                    # Abort BEFORE the mode latches, so a re-run starts clean.
                    if strict_quality_run:
                        raise StrictQualityRunError(
                            "exchange_append capability fallback: this Hindsight client "
                            "rejected stable-document append (update_mode='append') and "
                            "strict_quality_run is set, so the arm will NOT silently "
                            "downgrade to legacy accumulated-resend. Tripped on "
                            f"bank={bank_id} session={session_label} exchange_index={turn_index}. "
                            f"Underlying error: {type(e).__name__}: {str(e) or '(no detail)'}")
                    _log_append_fallback_once(f"{type(e).__name__}: {str(e) or '(no detail)'}")
                    append_state["mode"] = "accumulated_resend"
                    legacy_content = _turns_to_content(accumulated_turns)
                    print(f"[DEBUG] bank={bank_id} session {session_label} "
                          f"retain_start call={call_n}/{total_calls} chars={len(legacy_content)}")
                    _t1 = time.time()
                    ok_legacy = _retain_legacy_accumulated(client, bank_id,
                                                  legacy_content,
                                                  ex_ts, context, legacy_document_id)
                    _dur1 = (time.time() - _t1) * 1000.0
                    retain_durations_ms.append(_dur1)
                    print(f"[DEBUG] bank={bank_id} session {session_label} "
                          f"retain_done call={call_n}/{total_calls} ms={_dur1:.0f} ok={ok_legacy}")
                    if ok_legacy:
                        ok += 1
                    else:
                        failed += 1
                else:
                    # A transient error is a timeout, a 5xx, or a connection
                    # error. Do NOT decide the run mode from it. Count the
                    # failure and probe again on the next exchange.
                    detail = str(e) or ""
                    print(f"[DEBUG] exchange_append capability probe hit a transient "
                          f"error for bank={bank_id} turn={turn_index} "
                          f"(mode stays undecided): type={type(e).__name__} detail={detail!r}")
                    failed += 1
        elif append_state["mode"] == "append":
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_start call={call_n}/{total_calls} chars={len(delta_content)}")
            _t0 = time.time()
            try:
                _retain_stable_append(client, bank_id, delta_content,
                                      ex_ts, context, document_id, metadata)
                _dur = (time.time() - _t0) * 1000.0
                retain_durations_ms.append(_dur)
                print(f"[DEBUG] bank={bank_id} session {session_label} "
                      f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok=True")
                ok += 1
            except Exception as e:  # transient failure: count it, do NOT flip the mode
                _dur = (time.time() - _t0) * 1000.0
                retain_durations_ms.append(_dur)
                print(f"[DEBUG] bank={bank_id} session {session_label} "
                      f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok=False")
                detail = str(e) or ""
                print(f"[DEBUG] exchange_append retain failed for bank={bank_id} "
                      f"turn={turn_index}: type={type(e).__name__} detail={detail!r}")
                failed += 1
        else:  # accumulated_resend
            legacy_content = _turns_to_content(accumulated_turns)
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_start call={call_n}/{total_calls} chars={len(legacy_content)}")
            _t0 = time.time()
            ok_legacy = _retain_legacy_accumulated(client, bank_id,
                                          legacy_content,
                                          ex_ts, context, legacy_document_id)
            _dur = (time.time() - _t0) * 1000.0
            retain_durations_ms.append(_dur)
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok={ok_legacy}")
            if ok_legacy:
                ok += 1
            else:
                failed += 1

    if retain_durations_ms:
        print(f"[DEBUG] persona bank={bank_id} session {session_label} "
              f"retain_calls={len(retain_durations_ms)} "
              f"retain_mean_ms={(sum(retain_durations_ms) / len(retain_durations_ms)):.0f} "
              f"retain_max_ms={max(retain_durations_ms):.0f}")

    return ok, failed, append_state.get("mode")


def Add_Session_Dialogue_To_Hindsight(
    client,
    bank_id: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
    granularity: str,
    session_label: str,
    max_chars_per_retain: Optional[int] = None,
    append_state: Optional[Dict[str, Any]] = None,
    strict_quality_run: bool = False,
) -> Tuple[float, int, int, Optional[str]]:
    """Ingest the dialogue of one session into Hindsight.

    ``granularity='session'`` is the default. It joins the whole session into
    one role-prefixed transcript and retains it once, so there is one LLM
    extraction call.

    ``granularity='message'`` sends one retain() per INDIVIDUAL message. It
    mirrors the Mnemosyne adapter and makes one extraction call per
    user/assistant turn. It is much slower, and the Hermes plugin does not work
    this way.

    ``granularity='exchange'`` sends one retain() per user and assistant
    EXCHANGE pair, see ``Pair_Exchange_Turns``. This is the cadence of
    ``post_llm_call`` in the official Hindsight-Hermes plugin, which retains
    once per completed exchange. The retains here are INDEPENDENT and lack the
    stable-document append semantics of the plugin.

    ``granularity='exchange_append'`` is plugin-faithful. It sends one retain()
    per exchange under a STABLE per-session document_id with
    update_mode="append" and ships only the new turns, see
    ``_add_session_exchange_append``. It needs the run-level ``append_state``
    latch. On an append-incapable client it falls back to legacy
    accumulated-resend.

    LOGICAL TIMESTAMP (B4b). ``timestamp`` is the dataset ``Date`` of the
    session and is the per-session BASE. Every retain gets a MONOTONICALLY
    ADVANCING logical timestamp ``base + timedelta(minutes=<index>)``. The
    index is the unit that the granularity retains on: the message index for
    'message', and the exchange index for 'exchange' and 'exchange_append'.
    This mirrors the per-turn ``session_base + timedelta(minutes=turn_index)``
    of the Mnemosyne adapter. 'session' granularity is a single retain at index
    0, so it keeps the base date exactly. All providers then get the same
    strict intra-session recency order that the temporal (Dynamic) conflicts of
    MemConflict turn on, instead of tied timestamps for one provider and a
    strict order for another.

    ``max_chars_per_retain`` truncates the content sent to each ``retain()``.
    This is a SMOKE-MODE lever. gpt-oss-120b is very slow on large session
    transcripts, and it often emits huge or malformed extraction output. A
    full-fidelity run on it is therefore impractical. Truncation keeps each
    retain to one small chunk and one fast extraction, so a single persona
    completes end to end. Leave it unset (None) for a full-fidelity run on a
    faster model.

    Returns (add_duration_ms, retain_calls_ok, retain_calls_failed,
    append_mode). ``append_mode`` is the mode that ran under 'exchange_append',
    either "append" or "accumulated_resend". It is None otherwise.
    """
    if not dialogue_messages:
        return 0.0, 0, 0, None
    start = time.time()
    ok = 0
    failed = 0
    append_mode: Optional[str] = None
    context = f"MemConflict dialogue session {session_label}"

    if granularity == "exchange_append":
        if append_state is None:
            append_state = {}
        ok, failed, append_mode = _add_session_exchange_append(
            client, bank_id, dialogue_messages, timestamp, session_label,
            max_chars_per_retain, append_state,
            strict_quality_run=strict_quality_run,
        )
        return (time.time() - start) * 1000.0, ok, failed, append_mode
    elif granularity == "message":
        retain_durations_ms: List[float] = []
        total_calls = len(dialogue_messages)
        for call_n, message in enumerate(dialogue_messages, start=1):
            content = _truncate(f"{message['role']}: {message['content']}", max_chars_per_retain)
            # B4b: advance the logical timestamp per message, with the index
            # call_n-1. This matches the per-turn scheme of Mnemosyne.
            msg_ts = timestamp + timedelta(minutes=call_n - 1) if timestamp else None
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_start call={call_n}/{total_calls} chars={len(content)}")
            _t0 = time.time()
            success = _retain_one(client, bank_id, content, msg_ts, context)
            _dur = (time.time() - _t0) * 1000.0
            retain_durations_ms.append(_dur)
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok={success}")
            if success:
                ok += 1
            else:
                failed += 1
        if retain_durations_ms:
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_calls={len(retain_durations_ms)} "
                  f"retain_mean_ms={(sum(retain_durations_ms) / len(retain_durations_ms)):.0f} "
                  f"retain_max_ms={max(retain_durations_ms):.0f}")
    elif granularity == "exchange":
        retain_durations_ms: List[float] = []
        exchanges = Pair_Exchange_Turns(dialogue_messages)
        total_calls = len(exchanges)
        for call_n, exchange in enumerate(exchanges, start=1):
            content = _truncate(
                "\n".join(f"{m['role']}: {m['content']}" for m in exchange),
                max_chars_per_retain,
            )
            # B4b: advance the logical timestamp per exchange, with the index
            # call_n-1. This matches the per-turn scheme of Mnemosyne.
            ex_ts = timestamp + timedelta(minutes=call_n - 1) if timestamp else None
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_start call={call_n}/{total_calls} chars={len(content)}")
            _t0 = time.time()
            success = _retain_one(client, bank_id, content, ex_ts, context)
            _dur = (time.time() - _t0) * 1000.0
            retain_durations_ms.append(_dur)
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_done call={call_n}/{total_calls} ms={_dur:.0f} ok={success}")
            if success:
                ok += 1
            else:
                failed += 1
        if retain_durations_ms:
            print(f"[DEBUG] bank={bank_id} session {session_label} "
                  f"retain_calls={len(retain_durations_ms)} "
                  f"retain_mean_ms={(sum(retain_durations_ms) / len(retain_durations_ms)):.0f} "
                  f"retain_max_ms={max(retain_durations_ms):.0f}")
    else:  # 'session'
        transcript = _truncate(
            "\n".join(f"{m['role']}: {m['content']}" for m in dialogue_messages),
            max_chars_per_retain,
        )
        print(f"[DEBUG] bank={bank_id} session {session_label} "
              f"retain_start call=1/1 chars={len(transcript)}")
        _t0 = time.time()
        success = _retain_one(client, bank_id, transcript, timestamp, context)
        _retain_ms = (time.time() - _t0) * 1000.0
        print(f"[DEBUG] bank={bank_id} session {session_label} "
              f"retain_done call=1/1 ms={_retain_ms:.0f} ok={success}")
        print(f"[DEBUG] bank={bank_id} session {session_label} "
              f"retain_calls=1 retain_mean_ms={_retain_ms:.0f} retain_max_ms={_retain_ms:.0f}")
        if success:
            ok += 1
        else:
            failed += 1

    return (time.time() - start) * 1000.0, ok, failed, append_mode


# --------------------------------------------------------------------------
# Consolidation-drain polling (Arm B/C: --wait_consolidation)
# --------------------------------------------------------------------------
def _run_sync(coro):
    """Run an async coroutine to completion from synchronous code.

    This mirrors the get-or-create-loop pattern that the synchronous wrappers
    of ``hindsight_client`` use internally, such as ``client.retain`` and
    ``client.recall``. It therefore reuses the same thread-local event loop.
    ``asyncio.run()`` would instead open and close a new loop on every poll.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _list_consolidation_ops(client, bank_id: str) -> List[Any]:
    """List the consolidation ops of a bank, at any status, through the
    Operations API.

    hindsight_client 0.8.6 exposes the async_operations table through the
    low-level async-only ``OperationsApi.list_operations(bank_id, type=,
    status=, limit=)``. That call maps to ``GET
    /v1/default/banks/{bank_id}/operations``. The server route is in
    hindsight_api/api/http.py at operation_id=list_operations. Every retain()
    that lands while auto-consolidation is enabled queues an async
    'consolidation' task through
    MemoryEngine._submit_post_retain_side_effects and
    submit_async_consolidation. A row in the async_operations table of that
    bank tracks the task, with status in {pending, processing, completed,
    failed, cancelled}, see poller.py.

    This function polls without a status filter, which costs one HTTP call.
    Callers then filter or compare by status and id.

    ``client`` is the ``HindsightEmbedded`` wrapper. ``client.client`` makes
    sure the daemon is up and returns the underlying async ``Hindsight``
    client. Its ``.operations`` property is the low-level OperationsApi.
    """
    low_level = client.client
    response = _run_sync(
        low_level.operations.list_operations(bank_id=bank_id, type="consolidation", limit=100)
    )
    return getattr(response, "operations", None) or []


def _non_terminal_op_ids(ops: List[Any]) -> set:
    return {getattr(op, "id", None) for op in ops if getattr(op, "status", None) in ("pending", "processing")}


def Snapshot_Pending_Consolidation_Ops(client, bank_id: str) -> set:
    """Return the ids of the consolidation ops already pending or processing
    for this bank.

    The caller takes this snapshot BEFORE the retain() calls of this session
    queue new work. It scopes ``Wait_For_Consolidation_Drain`` to the ops that
    the retains of THIS session queue. A pre-existing op is then excluded from
    the completion condition of the wait instead of poisoning it. Such an op
    can be permanently wedged from an earlier session, or still draining after
    the wait timeout of the previous session.

    The call is best-effort. A query failure returns an empty set and does not
    raise. The wait of the caller then falls back to waiting on every
    non-terminal op it later sees, which is the pre-hardening behavior.
    """
    try:
        return _non_terminal_op_ids(_list_consolidation_ops(client, bank_id))
    except Exception as e:  # pragma: no cover
        print(f"[WARN] consolidation-drain pre-retain snapshot failed for bank={bank_id}: "
              f"{type(e).__name__}: {e}; wait will not exclude pre-existing ops.")
        return set()


def Wait_For_Consolidation_Drain(
    client,
    bank_id: str,
    pre_existing_pending_ids: Optional[set] = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.5,
) -> Tuple[float, int, bool, bool]:
    """Block until this bank has no NEW pending or processing consolidation
    ops.

    "New" means absent as pending or processing from
    ``pre_existing_pending_ids``. ``Snapshot_Pending_Consolidation_Ops`` takes
    that snapshot immediately before the retain() calls of this session. The
    wait is therefore per-session and tolerant of a pre-existing or stuck op.
    Such an op is excluded from the completion condition by id, however long it
    stays pending or processing, so it can no longer poison the wait of every
    later session into a full timeout. Ops absent from the snapshot are the ops
    that the retains of this session queued, either as brand-new ids or, for
    safety, as ids that were simply not already non-terminal. The function
    waits on those until they reach a terminal state, that is completed,
    failed, or cancelled.

    The function is best-effort by default. A status-query failure logs a
    warning and gives up on this wait. It never raises, because it must never
    abort the persona. A timeout logs a warning and returns instead of blocking
    forever. Under a strict quality run the CALLER ``ingest_session`` turns
    either signalled condition into a hard abort. This function stays
    best-effort and only REPORTS what happened through the returned
    ``timed_out`` and ``poll_failed`` flags.

    Returns (elapsed_seconds, poll_count, timed_out, poll_failed).
    """
    pre_existing_pending_ids = pre_existing_pending_ids or set()
    start = time.time()
    polls = 0
    timed_out = False
    poll_failed = False
    last_progress_s = 0.0
    while True:
        polls += 1
        try:
            ops = _list_consolidation_ops(client, bank_id)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] consolidation-drain status query failed for bank={bank_id}: "
                  f"{type(e).__name__}: {e}; giving up on this wait.")
            poll_failed = True
            break
        outstanding = [
            op for op in ops
            if getattr(op, "status", None) in ("pending", "processing")
            and getattr(op, "id", None) not in pre_existing_pending_ids
        ]
        if not outstanding:
            break
        elapsed = time.time() - start
        if elapsed - last_progress_s >= 30.0:
            print(f"[DEBUG] bank={bank_id} consolidation_waiting "
                  f"elapsed_s={elapsed:.0f} outstanding={len(outstanding)}")
            last_progress_s = elapsed
        if elapsed > timeout_s:
            print(f"[WARN] consolidation drain timed out after {elapsed:.1f}s for bank={bank_id} "
                  f"({len(outstanding)} new op(s) still pending/processing, "
                  f"{len(pre_existing_pending_ids)} pre-existing op(s) excluded); "
                  f"continuing without waiting further.")
            timed_out = True
            break
        time.sleep(poll_interval_s)
    return time.time() - start, polls, timed_out, poll_failed


def _result_created_at(result: Any) -> str:
    """Return the best temporal anchor of a recalled fact for the created_at
    slot of the scorer."""
    for attr in ("occurred_start", "mentioned_at", "occurred_end"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return "Unknown Time"


# The fact types that recall() accepts, copied from the installed daemon.
# hindsight_api 0.8.6 defines
# hindsight_api.engine.response_models.VALID_RECALL_FACT_TYPES = frozenset(
# ["world", "experience", "observation"]). This is a literal instead of an
# import because the adapter process parses the ARGUMENT. That process must not
# depend on the internals of the daemon package only to reject a typo.
VALID_RECALL_TYPES = ("world", "experience", "observation")

# The ctx key that holds the logical date of the CURRENT session as an ISO
# string. It is the recall-time "now" for the questions answered against that
# session. It mirrors the `_QUESTION_DATE_KEY` and `question_date` handling of
# retaindb_server/eval_retaindb_server.py exactly. The shared driver ingests
# session i and then immediately answers the questions of session i, because
# eval_common.Answer_Questions_For_One_Session runs once per session_item
# directly after ingest_session. The "now" of a question is therefore the
# dataset date of that session, not wall-clock. An unparseable or missing date
# gives None, recall() then omits query_timestamp, and Hindsight's
# `_recall_scoring_now` falls back to `utcnow()` (memory_engine.py:743-751).
# That is the earlier wall-clock behavior, for that persona and session only.
_QUESTION_DATE_KEY = "question_date_iso"


def parse_recall_types(value: Optional[str]) -> Optional[List[str]]:
    """Parse ``--recall_types`` to a list or None.

    Accepted values are "observation", "observation,world", "all", and "".

    There are three distinct outcomes. The Arm-C guard in ``__main__`` must
    tell "the operator never chose" apart from "the operator deliberately chose
    unfiltered". A pasted Arm C command with no RECALL_TYPES must refuse to
    run, not silently run unfiltered.

      * ``None`` input, that is the flag was not given, gives ``None``: UNSET.
      * "", "all", "none", or only commas give ``[]``: the DELIBERATELY
        UNFILTERED sentinel. ``__main__`` normalizes it back to ``None`` with
        ``args.recall_types or None`` before it reaches any recall path. The
        wire behavior, that is no ``types`` kwarg, and the recorded
        ``Recall_Types: null`` are then byte-identical to unset. The sentinel
        exists ONLY for the guard. Never pass ``[]`` itself to recall().
        hindsight_api treats a falsy ``types`` as "all types"
        (``request.types if request.types else
        list(VALID_RECALL_FACT_TYPES)``, api/http.py:3852), so an empty list
        means the OPPOSITE of a filter.
      * a real subset gives the canonical list.

    Validation is deliberately strict and runs at PARSE time. An unknown type
    would otherwise appear as a ValueError from the daemon on the very first
    question. That is hours into a sharded ingest, after all the retain GPU
    time is already spent. The parser also normalizes order and removes
    duplicates, so the value recorded in the Results metadata is canonical.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw or raw.lower() in ("all", "none"):
        return []
    parts = [p.strip().lower() for p in raw.split(",")]
    parts = [p for p in parts if p]
    invalid = [p for p in parts if p not in VALID_RECALL_TYPES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid recall type(s): {', '.join(sorted(set(invalid)))}. "
            f"Must be a comma-separated subset of: {', '.join(VALID_RECALL_TYPES)} "
            f"(or 'all' for deliberately-unfiltered recall)")
    # Remove duplicates and keep the canonical VALID_RECALL_TYPES order. An
    # all-commas value such as "," or ",," arrives here with no parts. It then
    # gives the deliberate sentinel, the same as "all".
    return [t for t in VALID_RECALL_TYPES if t in parts] or []


def Search_Hindsight_For_Question(
    client, bank_id: str, question_text: str, top_k: int, budget: str, max_tokens: int,
    prefer_observations: bool = False,
    recall_types: Optional[List[str]] = None,
    query_timestamp: Optional[str] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Recall the facts for one question and map them to the memory shape of
    the scorer.

    The recall() of Hindsight is token-budgeted, not top-K. It returns facts
    ranked by ``scores.final`` until it spends ``max_tokens``. This adapter
    requests a generous budget and slices the top ``top_k``. It keeps up to 5
    for white-box scoring.

    ``recall_types`` becomes the ``types=`` kwarg of the client. The kwarg name
    is verified against the installed hindsight_client 0.8.6:
    ``Hindsight.recall(bank_id, query, types: list[str] | None = None, ...)``.
    The valid members are ``world``, ``experience``, and ``observation``, that
    is ``hindsight_api.engine.response_models.VALID_RECALL_FACT_TYPES``.

    WHY THIS FLAG EXISTS (fairness, "what is under test"). The real Hermes
    memory plugin does NOT recall every fact type. The pinned commit is
    NousResearch/hermes-agent plugins/memory/hindsight/__init__.py @ 977884e6.
    It defaults to observation-only::

        # Default narrows recall to observation-only; pass an explicit
        # `recall_types` list in config.json to broaden (e.g. include
        # "world" / "experience") or to disable the filter entirely.
        self._recall_types: list[str] = ["observation"]

    and then::

        if self._recall_types:
            recall_kwargs["types"] = self._recall_types

    A real Hermes deployment therefore hands the agent only the consolidated
    observation layer at prefetch time. The raw world and experience facts are
    supporting evidence that the observations already summarize, and the agent
    never sees them. The Goal ruling in CLAUDE.md makes the unit of measurement
    what the plugin actually hands Hermes. The plugin-faithful arm must apply
    the same filter. An unset value reproduces the earlier behavior exactly: no
    ``types`` filter gives ``fact_type is None``, and
    MemoryEngine.recall_async then substitutes
    ``list(VALID_RECALL_FACT_TYPES)``, that is all three. Arms A and B ran that
    way.

    ``prefer_observations`` is a recall() kwarg of hindsight_client 0.8.6,
    verified against the installed package. It has an effect only when both
    'observation' and at least one raw type are in scope. It drops a raw fact
    that a returned observation was consolidated from, so the observation
    supersedes that fact instead of duplicating it. The docstring of the
    installed client states this: "no effect unless 'observation' and at least
    one raw type are both in ``types``".

    One consequence is worth a plain statement. With
    ``recall_types=['observation']``, the plugin default, prefer_observations
    is a NO-OP, because no raw facts remain for it to suppress. To pass both is
    therefore harmless and self-consistent, not contradictory. The type filter
    is the stronger form of the same intent. Arm C passes both, so its command
    line stays a strict superset of the arm B command line.

    ``query_timestamp`` is the recall-time "now" as an ISO 8601 string, for
    example ``"2024-03-14T09:00:00+00:00"``. It goes to the client's own
    ``query_timestamp`` kwarg (hindsight_client 0.8.6,
    ``hindsight_client.py:394-450``: "the query-time anchor for relative
    temporal expressions and recency scoring"). On the server it becomes
    ``question_date`` (``hindsight_api/engine/memory_engine.py:4007-4045``).
    ``_recall_scoring_now`` (``memory_engine.py:743-751``) falls back to
    wall-clock ``utcnow()`` only when this value is ``None``.

    Ingestion already anchors ``retain(timestamp=)`` to the logical session
    date of the dataset, see the module docstring above. This value makes the
    recency reference of recall consistent with that anchor. Without it, recall
    compares a logical past against a real wall-clock "now". The effect is a
    soft multiplicative recency boost or decay, not a hard cutoff. This value
    excludes no results.
    """
    start = time.time()
    response = client.recall(
        bank_id=bank_id,
        query=question_text,
        budget=budget,
        max_tokens=max_tokens,
        prefer_observations=prefer_observations,
        # None omits the filter and is the client's own default. Do NOT send an
        # empty list. hindsight_api treats an empty or falsy `types` as "all
        # types" (api/http.py:3852 `request.types if request.types else
        # list(VALID_RECALL_FACT_TYPES)`), so [] would silently mean the
        # opposite of a filter. __main__ collapses the deliberately-unfiltered
        # sentinel of parse_recall_types(), that is [] for "all" or "", back to
        # None. This function therefore only ever sees None or a non-empty list.
        types=recall_types,
        # This anchors recency scoring to the logical session date of the
        # dataset instead of wall-clock. None comes from an unparseable or
        # missing session date. It falls back to the wall-clock default of the
        # client and server, exactly as before this change.
        query_timestamp=query_timestamp,
    )
    duration_ms = (time.time() - start) * 1000.0

    # ZERO RETRIEVAL IS A LEGAL, SCOREABLE OUTCOME. Do not special-case it.
    # An observation-only recall can run against a bank whose consolidation has
    # produced no observation yet. It then returns a well-formed RecallResponse
    # with an EMPTY `results` list. The engine builds
    # RecallResultModel(results=[], ...), raises nothing, and does not fall back
    # to raw facts.
    #
    # The loop below then yields `retrieved == []`, and the shared driver still
    # writes a complete row. Build_Retrieved_Memory_Context([]) renders "No
    # relevant memories found." (benchmark/eval_common.py:172-180), which is by
    # coincidence the exact string the real Hermes plugin returns to the agent
    # in the same situation. The driver still calls the answer LLM and still
    # emits Model_Answer and Retrieved_Memories: [] (eval_common.py:364-387).
    # The judge scores that row normally with Support_Rank 0.
    #
    # Per the project ruling, an observation-coverage gap is the behavior of
    # the shipped product and gets REPORTED, not patched. Never add a "fall
    # back to raw facts when observations are empty" path here. That would
    # measure a configuration no Hermes deployment runs.
    results = getattr(response, "results", None) or []
    retrieved: List[Dict[str, Any]] = []
    for result in results:
        scores = getattr(result, "scores", None)
        final = getattr(scores, "final", None) if scores is not None else None
        retrieved.append({
            "memory": str(getattr(result, "text", "")),
            "created_at": _result_created_at(result),
            "score": round(float(final), 6) if isinstance(final, (int, float)) else final,
            "id": getattr(result, "id", None),
            "type": getattr(result, "type", None),
            "semantic_score": getattr(scores, "semantic", None) if scores is not None else None,
            "keyword_score": getattr(scores, "keyword", None) if scores is not None else None,
            "reranker_score": getattr(scores, "reranker", None) if scores is not None else None,
        })
    # Diagnostic capture. Raw is the one RecallResponse that answered this
    # question. There is one recall call, so no ambiguity about which response
    # represents the question. Ranked is the whole mapped list. Hindsight
    # recall is token-budgeted, this function applies NO slice, and the budget
    # and max_tokens above are exactly what the arm already requests. Nothing
    # is widened for the capture. The fallback of eval_common would infer the
    # same list, but an explicit record keeps the raw and ranked pair in one
    # place.
    record_provider_retrieval(diag, raw=response, ranked=retrieved)
    return retrieved, duration_ms


# --------------------------------------------------------------------------
# Provider binding (the only Hindsight-specific surface the driver sees)
# --------------------------------------------------------------------------
class HindsightBinding(ProviderBinding):
    memory_system = "hindsight"
    store_id_key = "Hindsight_Bank_ID"
    runtime_summary_key = "Hindsight_Runtime_Summary"
    stage_name = "hindsight_answer_generation"
    stage_note = "Hindsight retrieval and question answering"

    def __init__(self, client, budget: str, max_tokens: int, granularity: str,
                 max_chars_per_retain: Optional[int], prefer_observations: bool,
                 wait_consolidation: bool, consolidation_wait_timeout_s: float,
                 consolidation_poll_interval_s: float,
                 strict_quality_run: bool = False,
                 recall_types: Optional[List[str]] = None,
                 plugin_native_recall: bool = False):
        self.client = client
        self.budget = budget
        self.max_tokens = max_tokens
        self.granularity = granularity
        self.max_chars_per_retain = max_chars_per_retain
        self.prefer_observations = prefer_observations
        # None puts no `types` filter on recall(), so recall returns world,
        # experience, and observation. Arms A and B ran that way.
        # ["observation"] reproduces the Hermes plugin's own default. See
        # Search_Hindsight_For_Question.
        self.recall_types = recall_types
        self.wait_consolidation = wait_consolidation
        self.consolidation_wait_timeout_s = consolidation_wait_timeout_s
        self.consolidation_poll_interval_s = consolidation_poll_interval_s
        # B5: when set, a broken quality-arm invariant aborts the shard loudly
        # instead of logging and continuing. The invariants are the
        # consolidation drain timeout, the drain poll failure, and an
        # exchange_append downgrade off the 'append' path. It is off by
        # default, so exploratory arms keep the tolerant behavior. The run plan
        # passes it for the Arm-B and Arm-C headline runs.
        self.strict_quality_run = strict_quality_run
        # PLUGIN-NATIVE RECALL (featured Arm C). When set, the shared driver
        # eval_common hands the answer LLM EVERY recalled item in order and
        # stores them all in Retrieved_Memories, instead of a top_k slice. This
        # mirrors the real Hermes hindsight plugin, which injects its FULL
        # token-budgeted recall result. Its queue_prefetch joins every
        # resp.results item with no top-K slice (NousResearch/hermes-agent
        # plugins/memory/hindsight/__init__.py @ 977884e6, prefetch path). The
        # ProviderBinding-aware eval_common reads this attribute. The bound of
        # the plugin is the recall token budget, that is budget=mid and
        # max_tokens=4096, not a count. Minimal Arm A leaves this False and
        # keeps the historical top_k slice.
        self.plugin_native_recall = plugin_native_recall
        # RUN-level latch for 'exchange_append'. The adapter makes the
        # append-or-legacy capability decision once and shares it across every
        # persona and session of the run. The latch also holds the legacy
        # per-run document-id seed.
        self._append_state: Dict[str, Any] = {}

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        return {
            "store_id": f"mc_{persona_id[-8:]}_{uuid.uuid4().hex[:8]}",
            "persona_tag": persona_id[-8:],
            "total_retain_ok": 0,
            "total_retain_failed": 0,
            "consolidation_wait_ms": 0.0,
            "consolidation_poll_count": 0,
            "consolidation_wait_timeouts": 0,
            "consolidation_sessions_waited": 0,
            _QUESTION_DATE_KEY: None,
        }

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        bank_id = ctx["store_id"]
        timestamp = Parse_Session_Timestamp(session_item)
        session_label = str(session_item.get("Session_ID", session_index))
        # Store the recall-time query_timestamp of this session for the
        # questions answered against it. The shared driver ingests session i
        # and then immediately answers the questions of session i, so the "now"
        # of a question is the date of this session. This is the NOON
        # recall-"now" anchor from eval_common.Parse_Query_Now_Timestamp, the
        # SAME instant the faked OS clock uses. It is NOT the MIDNIGHT
        # ``timestamp`` below, which stays the ingest retain(timestamp=)
        # anchor. Only the recall query_timestamp is noon, and ingest is
        # untouched. An unparseable date gives None, recall() then omits
        # query_timestamp, and the wall-clock fallback applies. This matches
        # the _QUESTION_DATE_KEY handling of retaindb_server.
        query_now = Parse_Query_Now_Timestamp(session_item)
        ctx[_QUESTION_DATE_KEY] = query_now.isoformat() if query_now else None

        # Take the snapshot BEFORE the retain() calls of this session queue new
        # consolidation work. The post-retain wait below then blocks only on
        # the ops this session caused. A pre-existing or stuck op is excluded
        # by id, instead of poisoning the wait of every later session into a
        # full timeout.
        pre_existing_pending_ids = (
            Snapshot_Pending_Consolidation_Ops(self.client, bank_id)
            if self.wait_consolidation else set()
        )

        add_ms, retain_ok, retain_failed, append_mode = Add_Session_Dialogue_To_Hindsight(
            self.client, bank_id, dialogue, timestamp, self.granularity, session_label,
            max_chars_per_retain=self.max_chars_per_retain,
            append_state=self._append_state,
            strict_quality_run=self.strict_quality_run,
        )
        ctx["total_retain_ok"] += retain_ok
        ctx["total_retain_failed"] += retain_failed

        # B5: under strict mode a capability fallback inside exchange_append
        # aborts at the point of the downgrade. But the run-level mode latch
        # can ALSO already hold "accumulated_resend" from an earlier persona or
        # session. Later sessions then take the legacy branch without a new
        # probe. Catch that persisted downgrade here. For a strict
        # exchange_append arm, any Append_Mode other than "append" is a
        # mis-measurement.
        if (self.strict_quality_run and self.granularity == "exchange_append"
                and append_mode is not None and append_mode != "append"):
            raise StrictQualityRunError(
                f"exchange_append ran in Append_Mode={append_mode!r} (not 'append') under "
                f"strict_quality_run on persona-bank={bank_id} session={session_label}: the "
                "arm is no longer on the plugin-faithful stable-document append path. "
                "Drop the run and re-launch on an append-capable client.")

        wait_s = 0.0
        polls = 0
        timed_out = False
        poll_failed = False
        if self.wait_consolidation:
            wait_s, polls, timed_out, poll_failed = Wait_For_Consolidation_Drain(
                self.client, bank_id,
                pre_existing_pending_ids=pre_existing_pending_ids,
                timeout_s=self.consolidation_wait_timeout_s,
                poll_interval_s=self.consolidation_poll_interval_s,
            )
            ctx["consolidation_wait_ms"] += wait_s * 1000.0
            ctx["consolidation_poll_count"] += polls
            ctx["consolidation_sessions_waited"] += 1
            if timed_out:
                ctx["consolidation_wait_timeouts"] += 1
            print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
                  f"consolidation_wait_s={wait_s:.0f} polls={polls} timed_out={timed_out} "
                  f"poll_failed={poll_failed}")
            # B5: an undrained consolidation state means the quality arm did
            # not measure fully-consolidated memory for this session. Under
            # strict mode a drain timeout or a poll failure aborts the shard
            # loudly. The alternative is to answer against half-consolidated
            # memory.
            if self.strict_quality_run and (timed_out or poll_failed):
                _cond = "drain timeout" if timed_out else "drain poll failure"
                raise StrictQualityRunError(
                    f"consolidation {_cond} under strict_quality_run on "
                    f"persona-bank={bank_id} session={session_label} "
                    f"(waited {wait_s:.0f}s over {polls} poll(s), "
                    f"timeout_s={self.consolidation_wait_timeout_s:.0f}): the arm cannot "
                    "claim fully-consolidated memory for this session. Investigate the "
                    "daemon/consolidation load, then re-run.")

        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
              f"retain_ok={retain_ok} ingest_ms={add_ms:.0f}")
        return {
            "Dialogue_Added_To_Memory": retain_ok > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Retain_Call_Count": retain_ok,
            "Retain_Failed_Count": retain_failed,
            "Retain_Granularity": self.granularity,
            "Append_Mode": append_mode,  # 'append' or 'accumulated_resend' under exchange_append, else None
            "Session_Timestamp_Passed": timestamp.isoformat() if timestamp else None,
            "Add_Duration_ms": add_ms,
            "Consolidation_Wait_ms": (wait_s * 1000.0) if self.wait_consolidation else None,
            "Consolidation_Poll_Count": polls if self.wait_consolidation else None,
            "Consolidation_Wait_Timed_Out": timed_out if self.wait_consolidation else None,
        }

    def recall(self, ctx, question_text, top_k):
        return Search_Hindsight_For_Question(
            self.client, ctx["store_id"], question_text, top_k,
            self.budget, self.max_tokens,
            prefer_observations=self.prefer_observations,
            recall_types=self.recall_types,
            query_timestamp=ctx.get(_QUESTION_DATE_KEY),
            diag=ctx,
        )

    def persona_count_extras(self, ctx):
        return {
            "Total_Retain_Calls_OK": ctx["total_retain_ok"],
            "Total_Retain_Calls_Failed": ctx["total_retain_failed"],
        }

    def persona_tail_extras(self, ctx):
        return {
            "Wait_Consolidation_Enabled": self.wait_consolidation,
            "Prefer_Observations_Enabled": self.prefer_observations,
            # Recorded per persona, so a Results file declares its own recall
            # surface. A null value means unfiltered, that is all fact types.
            # Without this field, an observation-only file and an unfiltered
            # file look the same afterward.
            "Recall_Types": self.recall_types,
            # Declares the width of the recall surface. True means
            # plugin-native: the answer saw every recalled item and
            # Retrieved_Memories keeps them all. False means the historical
            # top_k slice. This tells a featured Arm-C file from a minimal one
            # afterward.
            "Plugin_Native_Recall": self.plugin_native_recall,
            "Persona_Consolidation_Wait_Time_ms": ctx["consolidation_wait_ms"],
            "Persona_Consolidation_Poll_Count": ctx["consolidation_poll_count"],
            "Persona_Consolidation_Wait_Timeouts": ctx["consolidation_wait_timeouts"],
            "Persona_Consolidation_Sessions_Waited": ctx["consolidation_sessions_waited"],
        }


def Generate_User_Hindsight_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    budget: str,
    max_tokens: int,
    granularity: str,
    profile: str,
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
    max_chars_per_retain: Optional[int] = None,
    prefer_observations: bool = False,
    wait_consolidation: bool = False,
    consolidation_wait_timeout_s: float = 450.0,
    consolidation_poll_interval_s: float = 1.5,
    strict_quality_run: bool = False,
    recall_types: Optional[List[str]] = None,
    plugin_native_recall: bool = False,
) -> bool:
    print(f"[DEBUG] budget={budget}  granularity={granularity}  "
          f"prefer_observations={prefer_observations}  wait_consolidation={wait_consolidation}  "
          f"strict_quality_run={strict_quality_run}  "
          f"recall_types={recall_types if recall_types else 'ALL (unfiltered)'}  "
          f"plugin_native_recall={plugin_native_recall}")
    binding = HindsightBinding(
        client=None,
        budget=budget,
        max_tokens=max_tokens,
        granularity=granularity,
        max_chars_per_retain=max_chars_per_retain,
        prefer_observations=prefer_observations,
        wait_consolidation=wait_consolidation,
        consolidation_wait_timeout_s=consolidation_wait_timeout_s,
        consolidation_poll_interval_s=consolidation_poll_interval_s,
        strict_quality_run=strict_quality_run,
        recall_types=recall_types,
        plugin_native_recall=plugin_native_recall,
    )

    def setup():
        binding.client = Setup_Hindsight(profile)
        print(f"[DEBUG] Hindsight daemon up at {getattr(binding.client, 'url', '?')}")

    def teardown():
        if binding.client is not None:
            try:
                binding.client.close()
            except Exception:
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
    parser = argparse.ArgumentParser(description="Run Hindsight evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "hindsight_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "hindsight_results.json"),
        top_k_help="How many recalled facts the answer LLM sees (up to 5 are always stored "
                   "for white-box scoring). NOTE the upstream MemConflict adapters answer "
                   "from top-3, so use --top_k 3 for strict answer-accuracy comparability.",
    )
    parser.add_argument("--budget", type=str, default="mid", choices=["low", "mid", "high"],
                        help="Hindsight recall 'thinking budget' (maps to recall_budget_fixed_*).")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Hindsight recall token budget for returned facts.")
    parser.add_argument("--retain_granularity", type=str, default="session",
                        choices=["session", "message", "exchange", "exchange_append"],
                        help="'session' (default): one retain()/extraction per session. "
                             "'message': one retain()/extraction per INDIVIDUAL message (much slower). "
                             "'exchange': cadence only -- one INDEPENDENT retain() per user+assistant "
                             "exchange pair (the cadence of the official Hindsight-Hermes plugin's "
                             "post_llm_call, but without stable-document append semantics). "
                             "'exchange_append': plugin-faithful -- one retain() per exchange under a "
                             "STABLE per-session document_id with update_mode='append' shipping only "
                             "the new turns (falls back to legacy accumulated-resend on an "
                             "append-incapable client; the mode used is recorded per session).")
    parser.add_argument("--prefer_observations", action="store_true",
                        help="Arm B/C: pass prefer_observations=True to recall() so a returned "
                             "'observation' supersedes any raw world/experience fact it was "
                             "consolidated from, instead of both appearing. No effect if "
                             "auto-consolidation never produced observations.")
    parser.add_argument("--recall_types", type=parse_recall_types, default=None,
                        metavar="TYPES",
                        help="Comma-separated Hindsight fact types to recall "
                             "(world|experience|observation); becomes the recall() 'types' "
                             "filter. UNSET (default) = no filter = all three types, which "
                             "is exactly how arms A and B were run — but "
                             "--retain_granularity exchange_append (Arm C) REFUSES to start "
                             "unset: pass 'observation' (plugin-faithful) or the explicit "
                             "'all' (deliberate unfiltered, same wire behaviour as unset). "
                             "'observation' is PLUGIN-FAITHFUL: the real Hermes memory plugin "
                             "(NousResearch/hermes-agent plugins/memory/hindsight/__init__.py "
                             "@ 977884e6) defaults to self._recall_types = ['observation'] "
                             "and only ever hands the agent the consolidated observation "
                             "layer, so ARM C sets RECALL_TYPES=observation. Pair it with "
                             "--wait_consolidation: observations only exist once "
                             "consolidation has drained, and without the wait an "
                             "observation-only recall can legitimately return NOTHING. "
                             "A zero-result recall is kept as-is (scoreable row, "
                             "Support_Rank 0) -- it is the shipped product's behaviour, "
                             "reported not patched.")
    parser.add_argument("--wait_consolidation", action="store_true",
                        help="Arm B/C: after each session's retain() completes, block until the "
                             "daemon reports no pending/processing consolidation ops for this bank "
                             "(polls GET /v1/default/banks/{bank_id}/operations?type=consolidation), "
                             "so every question sees fully-consolidated memory. Best-effort: logs and "
                             "continues on a per-session timeout (--consolidation_wait_timeout_s).")
    parser.add_argument("--consolidation_wait_timeout_s", type=float, default=450.0,
                        help="Max seconds to wait per session for consolidation to drain "
                             "(only used with --wait_consolidation). Default 450s per the Arm-B "
                             "timeout diagnosis: under load the drain regularly exceeds 300s "
                             "(see docs/TROUBLESHOOTING.md).")
    parser.add_argument("--consolidation_poll_interval_s", type=float, default=1.5,
                        help="Seconds between consolidation-drain status polls "
                             "(only used with --wait_consolidation).")
    parser.add_argument("--strict_quality_run", action="store_true",
                        help="HEADLINE Arm-B/C runs only: abort the shard loudly (nonzero exit) "
                             "instead of logging-and-continuing on any silent-degradation path — "
                             "a consolidation drain timeout or poll failure (--wait_consolidation), "
                             "an exchange_append capability fallback to legacy accumulated-resend, "
                             "or any observed Append_Mode != 'append' when --retain_granularity "
                             "exchange_append was requested. Default OFF so exploratory arms keep "
                             "today's tolerant behaviour; the failure message names the exact "
                             "condition and the persona-bank/session it tripped on. Wired from the "
                             "entrypoint via STRICT_QUALITY_RUN=1.")
    parser.add_argument("--plugin_native_k", action="store_true",
                        help="FEATURED Arm C: emit EVERY recalled item (in order) to the answer "
                             "LLM and store them all in Retrieved_Memories, instead of the top_k "
                             "slice. Matches the real Hermes hindsight plugin, which injects its "
                             "FULL token-budgeted recall result (queue_prefetch joins every "
                             "resp.results item — no top-K slice; NousResearch/hermes-agent "
                             "plugins/memory/hindsight/__init__.py @ 977884e6). The plugin's bound "
                             "is the recall token budget (budget/max_tokens), not a count, so the "
                             "returned item count varies per question. SEH@K is unaffected (the "
                             "scorer caps at its own WHITE_BOX_TOP_K_VALUES). Minimal Arm A omits "
                             "this and keeps the top_k slice. Wired from the entrypoint via "
                             "PLUGIN_NATIVE_RECALL=1.")
    parser.add_argument("--profile", type=str,
                        default=os.environ.get("HINDSIGHT_PROFILE", f"memconflict_{uuid.uuid4().hex[:8]}"),
                        help="Embedded Hindsight daemon profile (data isolation). Unique per run "
                             "by default so concurrent shards don't share a daemon.")
    parser.add_argument("--max_chars_per_retain", type=lambda v: opt_int(v), default=None,
                        help="SMOKE-MODE: truncate the content sent to each retain() to this many "
                             "chars, so each session is one small/fast extraction. gpt-oss-120b is "
                             "very slow (and prone to huge/malformed extraction outputs) on full "
                             "session transcripts; leave unset for full fidelity on a faster model.")
    args = parser.parse_args()

    # ARM-C RECALL-SURFACE GUARD. It runs before any client or daemon setup,
    # so it trips instantly. exchange_append IS the plugin-faithful arm, and
    # the pinned plugin (NousResearch/hermes-agent
    # plugins/memory/hindsight/__init__.py @ 977884e6) defaults to
    # observation-only recall. A pasted Arm C command that forgot RECALL_TYPES
    # must therefore not silently measure an UNFILTERED recall surface, that is
    # all fact types, and mislabel the arm. None means the flag was never
    # given, so refuse. [] means an explicit --recall_types all, that is
    # deliberate unfiltered recall, so allow it through. The code below
    # normalizes [] back to None.
    if args.retain_granularity == "exchange_append" and args.recall_types is None:
        print(
            "[FATAL] --retain_granularity exchange_append (Arm C, the plugin-faithful arm) "
            "requires an explicit --recall_types.\n"
            "The pinned Hermes plugin (NousResearch/hermes-agent "
            "plugins/memory/hindsight/__init__.py @ 977884e6) defaults to observation-only "
            "recall (self._recall_types = [\"observation\"]); without RECALL_TYPES this run "
            "would silently recall ALL fact types and mislabel the arm.\n"
            "Pass one of:\n"
            "  --recall_types observation  (env RECALL_TYPES=observation) — plugin-faithful, "
            "the Arm C setting;\n"
            "  --recall_types all          (env RECALL_TYPES=all)         — deliberately "
            "unfiltered recall; NOT plugin-faithful, mislabels Arm C — never for headline runs.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # B2: run_eval() in benchmark/eval_common.py catches fatal exceptions and
    # returns False instead of raising. WITHOUT this exit-code propagation the
    # process would exit 0 on a crash. The `set -e` of the entrypoint would
    # never fire, and STAGE=all would score a partial or half-dead file. A
    # shard that died at persona 2 of 6 would look successful. Pass the boolean
    # to the exit code, so exit status alone judges sharded detached
    # containers.
    ok = Generate_User_Hindsight_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        budget=args.budget,
        max_tokens=args.max_tokens,
        granularity=args.retain_granularity,
        profile=args.profile,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
        max_chars_per_retain=args.max_chars_per_retain,
        prefer_observations=args.prefer_observations,
        wait_consolidation=args.wait_consolidation,
        consolidation_wait_timeout_s=args.consolidation_wait_timeout_s,
        consolidation_poll_interval_s=args.consolidation_poll_interval_s,
        strict_quality_run=args.strict_quality_run,
        # `or None` collapses the deliberately-unfiltered sentinel, that is []
        # from --recall_types all, to None. Every downstream path then stays
        # byte-identical to the historical unset behavior. Those paths are the
        # recall() types kwarg, the Recall_Types row field where null means
        # unfiltered, and the debug banner.
        recall_types=args.recall_types or None,
        plugin_native_recall=args.plugin_native_k,
    )
    raise SystemExit(0 if ok else 1)
