"""Attach-only REST client for the RetainDB server edition (@retaindb/server).

Unlike ``retaindb/_retaindb_server.py``, which spawns and owns a disposable
``@retaindb/local`` Node process, this module never starts a process. The
server edition is a Postgres and pgvector service. Its lifecycle, the build,
``prisma migrate deploy``, and ``node dist/index.js``, is managed entirely
shell-side, by the Docker entrypoint or by ``serve_local.sh`` on a host.
Python only attaches to an already running server over HTTP and waits for it
to become healthy. Keeping server lifecycle out of Python is deliberate.
Migrations and the pnpm and prisma build are node-toolchain concerns, and the
same adapter must run unchanged whether the entrypoint (container) or
``serve_local.sh`` (host) started the server.

The three endpoints the benchmark needs (read-only, matching
``external/RetainDB/packages/server/src``):

  * ``GET  /health``                    -> ``{"status": "ok"}`` (not ``/retaindb/health``)
  * ``POST /v1/memory/ingest/session``  -> ingest a session's dialogue (sync)
  * ``POST /v1/memory/search``          -> retrieve for a question (temporal-aware)

Auth: if the server started with ``RETAINDB_API_KEY`` set, every request must
carry ``Authorization: Bearer <key>``. If unset, the server allows open
access. Pass the key here to attach it to every request; it is omitted when
None or empty, matching open mode.
"""

import time
from typing import Any, Dict, List, Optional

import requests


class RetainDBServerClient:
    """Minimal attach-only REST client over the RetainDB server API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # Set once: an RETAINDB_API_KEY-gated server wants a Bearer token, and
        # an open server ignores the header, so sending it is harmless
        # either way. We omit it entirely when unset, so an open server's
        # request headers stay clean.
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # -- low-level ---------------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Retry 5xx and connection errors a few times, so a single transient
        # server error does not kill a multi-hour generate run. Never retry
        # 4xx errors, because they indicate a request bug, not a transient
        # condition.
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self._session.post(
                    f"{self.base_url}{path}", json=payload, timeout=self.timeout
                )
                if resp.status_code >= 500 and attempt < 2:
                    last_exc = requests.exceptions.HTTPError(
                        f"{resp.status_code} for {path}", response=resp
                    )
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_exc  # unreachable unless every retry is used

    def health(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def wait_healthy(self, ready_timeout: float = 180.0, poll_interval_s: float = 1.0) -> None:
        """Block until ``GET /health`` returns ``{"status": "ok"}`` or time out.

        A shell script starts the server, so this method never inspects a
        process. It only probes over HTTP. On timeout, a clear, actionable
        error tells the operator to start the server (the entrypoint does
        this in Docker, or run ``serve_local.sh`` on a host), instead of
        silently proceeding into a run that would fail on every request.
        """
        deadline = time.time() + ready_timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                h = self.health()
                if h.get("status") == "ok":
                    print(f"[retaindb-server] healthy at {self.base_url}", flush=True)
                    return
            except Exception as e:  # not up yet
                last_err = e
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"RetainDB server at {self.base_url} did not become healthy within "
            f"{ready_timeout}s (last error: {last_err}). Start the server first: "
            f"in Docker the retaindb-server entrypoint launches it; on a host run "
            f"retaindb_server/serve_local.sh."
        )

    # -- benchmark endpoints ----------------------------------------------
    def ingest_session(
        self,
        project: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        write_mode: str = "sync",
        promotion_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/memory/ingest/session.

        ``write_mode="sync"``, our default, makes the server return HTTP 200
        only after LLM extraction and commit finish, with
        ``memories_created``, ``relations_created``, and ``errors`` in the
        body. This means the run never recalls against a half-ingested
        session. ``messages`` carry ``timestamp``, an ISO datetime required
        by the server's zod schema, so memories are genuinely temporal
        (``timestamp`` flows into ``temporal.document_date`` and
        ``event_date``).

        ``promotion_mode`` is a vendor-exposed request field, with values
        ``session_state_v1`` (server default) or ``user_specific_legacy``.
        It selects the server's scope-inference pipeline. Under
        ``user_specific_legacy``, ordinary user facts in the mid-confidence
        band land in SESSION scope, which the session-lifecycle scheduler
        then promotes and summarizes. This is the only path that exercises
        that scheduler on message-only ingest. Omitted, the server default
        applies, matching behavior from before this parameter existed.
        """
        payload: Dict[str, Any] = {
            "project": project,
            "session_id": session_id,
            "user_id": user_id,
            "messages": messages,
            "write_mode": write_mode,
        }
        if promotion_mode:
            payload["promotion_mode"] = promotion_mode
        return self._post("/v1/memory/ingest/session", payload)

    def search(
        self,
        project: str,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        question_date: Optional[str] = None,
        profile: Optional[str] = None,
        include_pending: bool = True,
    ) -> Dict[str, Any]:
        """POST /v1/memory/search.

        ``include_pending`` mirrors the Hermes plugin, which sends
        ``include_pending: True`` on every search
        (``external/hermes-agent/plugins/memory/retaindb/__init__.py:230-238``).
        This is also the current server default. Measured 2026-07-28 on a
        live store: omitted and ``true`` both returned 4 results, while
        ``false`` returned 2. We send it explicitly, so a future default
        change cannot silently move our numbers.

        ``question_date``, an ISO string, drives the server's temporal-aware
        ranking. If unset, the server uses its own "now". The adapter no
        longer sends it (see ``_SEND_QUESTION_DATE`` in
        eval_retaindb_server.py), because the plugin does not, and under
        clock-sync the server's "now" is the session date, so the two forms
        are equivalent. Verified on a live store: both forms returned the
        same row and excluded the same older ones. Kept as a parameter for
        diagnostics.

        ``profile`` is fast, balanced, or quality. If unset, it is omitted,
        and the server default (fast) applies. Only fields the server's
        search schema accepts are sent. There are no hybrid, vector_weight,
        rerank, or threshold knobs, because those live on the unrelated
        document endpoint, not here.
        """
        payload: Dict[str, Any] = {
            "project": project,
            "query": query,
            "top_k": top_k,
            "user_id": user_id,
            "include_pending": include_pending,
        }
        if question_date is not None:
            payload["question_date"] = question_date
        if profile is not None:
            payload["profile"] = profile
        return self._post("/v1/memory/search", payload)
