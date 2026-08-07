"""Lifecycle manager and thin HTTP client for a self-hosted Supermemory server.

Self-hosted Supermemory, like RetainDB, is a local HTTP server rather than a
Python SDK. Unlike RetainDB, it has an **internal LLM** (like Hindsight). The
single ``supermemory-server`` binary (installed by ``npx supermemory local
install`` or ``curl -fsSL https://supermemory.ai/install | bash``) boots an
embedded graph engine, local ONNX embeddings, and the Memory API on
``http://localhost:<port>``, and it drives its own LLM for the "intelligent"
steps (summaries, contextual chunking, **memory extraction**). The pipeline
therefore has TWO distinct LLM roles, which this repo keeps strictly separate
(docs/DECISIONS.md):

  * Supermemory's INTERNAL extraction LLM: a provider knob, configured on the
    spawned server process through its own ``OPENAI_*`` env vars (fed here
    from ``SUPERMEMORY_LLM_*``, falling back to the harness ``OPENAI_*``).
  * The shared ANSWER + JUDGE LLM: the fairness-locked harness model. The
    Python adapter calls it through ``eval_common``, and it reads the
    *harness* ``OPENAI_*`` in the Python process. This module never touches
    it.

The server runs as a separate OS process. Giving it its own ``OPENAI_*`` in
the subprocess env keeps the two roles independent, even when both point at
the same endpoint for a smoke test (both gpt-oss-120b on OpenRouter).

This module owns that server for the benchmark:

  * ``SupermemoryServer`` spawns ``supermemory local start`` (override
    through ``SUPERMEMORY_SERVER_CMD``) with a unique ``PORT`` and
    ``SUPERMEMORY_DATA_DIR``, so runs stay isolated and disposable. It
    captures the API key the server prints on first boot (or uses
    ``SUPERMEMORY_API_KEY`` if preset), polls readiness, and tears the
    process down.
  * ``SupermemoryClient`` wraps the REST endpoints the benchmark needs:
      - POST /v3/documents            (ingest a document -> async processing queue)
      - GET  /v3/documents/processing (drain: how many docs are still processing)
      - GET  /v3/documents/{id}       (per-doc processing status)
      - POST /v4/search               (memories/hybrid recall -- the plugin path)
      - POST /v3/search               (documents recall -- the `documents` arm)

To attach to an already-running server instead of spawning one, pass an
explicit ``base_url`` and ``api_key`` (env ``SUPERMEMORY_BASE_URL`` /
``SUPERMEMORY_API_KEY``).
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# The server prints its generated key as ``sm_...`` on first boot. Keys are
# alphanumeric with - and _. This pattern requires a decent length, so it
# does not match a prefix in prose. It matches the banner line
# ``api key   sm_xxxxxxxx``.
_API_KEY_RE = re.compile(r"\bsm_[A-Za-z0-9_-]{12,}\b")
# The boot banner also prints the URL. This is used only as a secondary
# readiness signal.
_URL_RE = re.compile(r"https?://[0-9A-Za-z_.-]+:\d+")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerDiedError(RuntimeError):
    """The spawned server process exited while a request was in flight.

    The 0.0.5 binary segfaults in its PGlite WASM load path at a measured
    ~0.59% per boot, seconds after boot or mid-session. Every later request
    then raises ConnectionError, and the transport-retry budget spends itself
    against a process that can never answer (30 retries burned ~15 minutes per
    death in the v4minc2 pool). This exception separates "the server is gone"
    from "the socket blipped", so the caller can act at once. The
    ``accepted_doc_ids`` attribute carries the documents this session had
    already handed over before the death, because re-submission is safe only
    when that list is empty.
    """

    def __init__(self, *args: Any):
        super().__init__(*args)
        self.accepted_doc_ids: List[str] = []
        self.dropped: int = 0


class SupermemoryClient:
    """Minimal REST client over the self-hosted Supermemory Memory API."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0,
                 abort_check: Optional[Callable[[], bool]] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        # Transport-retry budget. The 0.0.5 server intermittently resets the
        # client socket ("Connection reset by peer") under multi-shard ingest
        # load. This is a transient blip that the server recovers from, but
        # requests raises ConnectionError for it. Without a retry here, one
        # blip killed a whole shard mid-run (v4min minimal lost 3 of 10
        # shards this way). This retries ONLY connection-level failures,
        # never an HTTP status, because a 400 is deterministic and
        # re-sending it is pointless. So this changes resilience, not what
        # the run requests or measures. Overridable through env vars.
        self._retries = int(os.environ.get("SUPERMEMORY_HTTP_RETRIES", "4"))
        self._retry_backoff = float(
            os.environ.get("SUPERMEMORY_HTTP_RETRY_BACKOFF_S", "2.0"))
        # Liveness probe for the SPAWNED server process, wired by
        # SupermemoryServer._await_ready. It returns True once that process
        # has exited. Attach-mode clients leave it None and keep exactly
        # today's retry behaviour, because this process does not own the
        # server it talks to and cannot tell a crash from a restart.
        self._abort_check = abort_check

    def close(self) -> None:
        """Release this client's connection pool. A respawned server listens on
        a new port, so pooled sockets to the previous one are dead weight."""
        try:
            self._session.close()
        except Exception:
            pass

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _send(self, method: str, path: str, *, json: Optional[Dict[str, Any]] = None,
              params: Optional[Dict[str, Any]] = None,
              timeout: Optional[float] = None) -> Dict[str, Any]:
        """Issue one request, retrying ONLY transport-level errors (connection
        reset, aborted, or read timeout). raise_for_status raises HTTP error
        statuses on the first response, and this never retries them."""
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                resp = self._session.request(
                    method, url, json=json, params=params,
                    headers=self._headers(), timeout=timeout or self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout) as exc:
                # A dead process cannot recover on its own, so the retry
                # budget would only delay the failure by minutes. Fail now and
                # let the caller respawn.
                if self._abort_check is not None and self._abort_check():
                    raise ServerDiedError(
                        f"supermemory server process exited during {method} {path} "
                        f"({type(exc).__name__})") from exc
                attempt += 1
                if attempt > self._retries:
                    raise
                sys.stderr.write(
                    f"[supermemory-client] {method} {path} transport error "
                    f"({type(exc).__name__}); retry {attempt}/{self._retries} "
                    f"after {self._retry_backoff * attempt:.1f}s\n")
                sys.stderr.flush()
                time.sleep(self._retry_backoff * attempt)

    def _post(self, path: str, payload: Dict[str, Any],
              timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._send("POST", path, json=payload, timeout=timeout)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._send("GET", path, params=params, timeout=timeout)

    # -- readiness ---------------------------------------------------------
    def ping(self) -> int:
        """Return the HTTP status of a cheap authed GET, or raise on a
        transport error. A 2xx means the server is up AND this client's
        bearer key is accepted. A 401 or 403 means the server is listening
        but the key is wrong. Readiness treats that as NOT ready, so a stale
        key fails loudly instead of being trusted."""
        resp = self._session.get(
            f"{self.base_url}/v3/documents/processing",
            headers=self._headers(), timeout=10,
        )
        return resp.status_code

    # -- ingest ------------------------------------------------------------
    def add_document(
        self,
        content: str,
        container_tag: str,
        metadata: Optional[Dict[str, Any]] = None,
        custom_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v3/documents. Returns instantly with ``{id, status:"queued"}``.
        Extraction, chunking, embedding, and indexing run in the background
        queue.

        This uses the singular ``containerTag`` (the vendor's recommended,
        higher-performance form; ``containerTags`` is the legacy array).
        ``metadata`` values must be primitives (string, number, or boolean)
        per the API.
        """
        payload: Dict[str, Any] = {"content": content, "containerTag": container_tag}
        if metadata:
            payload["metadata"] = metadata
        if custom_id:
            payload["customId"] = custom_id
        return self._post("/v3/documents", payload)

    def list_processing(self) -> Dict[str, Any]:
        """GET /v3/documents/processing -> ``{documents:[...], total:N}`` for
        documents still in the pipeline (not yet ``done`` or ``failed``)."""
        return self._get("/v3/documents/processing")

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """GET /v3/documents/{id} -> ``{status, content, ...}``."""
        return self._get(f"/v3/documents/{doc_id}")

    def wait_for_drain(
        self,
        doc_ids: Optional[List[str]] = None,
        timeout: float = 600.0,
        poll_interval: float = 1.5,
    ) -> Dict[str, Any]:
        """Block until ingestion has settled, so recall sees the new memories.

        Supermemory's add path is asynchronous: ``POST /v3/documents`` returns
        instantly with a document marked ``queued``, and that document
        becomes searchable only once it reaches ``done``. This function
        therefore drains the queue BEFORE the adapter answers a session's
        questions (this mirrors Hindsight's WAIT_CONSOLIDATION). Without this
        drain, recall races the queue and returns nothing.

        Strategy: prefer per-document polling when ``doc_ids`` is known. This
        is precise and unaffected by other tenants' traffic. The
        account-global ``/v3/documents/processing`` count is the fallback.
        The server pauses ingestion under RAM pressure
        (SUPERMEMORY_EMBEDDING_RAM_LIMIT), so a long queue simply takes
        longer. This function keeps polling until the deadline.

        Returns a small stats dict (drained, failed, timed_out, elapsed_s,
        polls).
        """
        start = time.time()
        polls = 0
        failed: List[str] = []
        pending = list(doc_ids) if doc_ids else None

        while True:
            polls += 1
            if pending is not None:
                still: List[str] = []
                for did in pending:
                    try:
                        status = str(self.get_document(did).get("status", "")).lower()
                    except ServerDiedError:
                        # Nothing will ever reach `done` on a dead process.
                        # Polling to the deadline would add drain_timeout to a
                        # failure the caller must handle now.
                        raise
                    except Exception:
                        # A transient read error: treat this document as
                        # still pending, and retry it.
                        still.append(did)
                        continue
                    if status == "done":
                        continue
                    if status == "failed":
                        failed.append(did)
                        continue
                    still.append(did)
                pending = still
                remaining = len(pending)
            else:
                try:
                    remaining = int(self.list_processing().get("total", 0) or 0)
                except ServerDiedError:
                    raise  # see the per-document branch above
                except Exception:
                    remaining = 1  # unknown count: keep waiting until timeout

            if remaining == 0:
                return {
                    "drained": True, "failed": failed, "timed_out": False,
                    "elapsed_s": round(time.time() - start, 2), "polls": polls,
                }
            if time.time() - start > timeout:
                return {
                    "drained": False, "failed": failed, "timed_out": True,
                    "remaining": remaining,
                    "elapsed_s": round(time.time() - start, 2), "polls": polls,
                }
            time.sleep(poll_interval)

    # -- recall ------------------------------------------------------------
    def search_memories(
        self,
        query: str,
        container_tag: str,
        limit: int = 10,
        threshold: Optional[float] = None,
        rerank: Optional[bool] = None,
        rewrite_query: Optional[bool] = None,
        search_mode: str = "hybrid",
        include: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """POST /v4/search: the conversational memory recall path the Hermes
        supermemory plugin uses (``search_mode`` default ``hybrid``: memories
        first, document chunks as fallback). Returns
        ``{results:[{id, memory|chunk, similarity, updatedAt, metadata, ...}],
        total, timing}``.
        """
        payload: Dict[str, Any] = {
            "q": query, "containerTag": container_tag, "limit": limit,
            "searchMode": search_mode,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        if rerank is not None:
            payload["rerank"] = rerank
        if rewrite_query is not None:
            payload["rewriteQuery"] = rewrite_query
        if include is not None:
            payload["include"] = include
        return self._post("/v4/search", payload)

    def search_documents(
        self,
        query: str,
        container_tag: str,
        limit: int = 10,
        document_threshold: Optional[float] = None,
        chunk_threshold: Optional[float] = None,
        rerank: Optional[bool] = None,
        rewrite_query: Optional[bool] = None,
        only_matching_chunks: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """POST /v3/search: literal document/chunk RAG recall (the
        ``documents`` arm). Returns ``{results:[{documentId,
        chunks:[{content,score}], score, ...}], total, timing}``."""
        payload: Dict[str, Any] = {
            "q": query, "containerTags": [container_tag], "limit": limit,
        }
        if document_threshold is not None:
            payload["documentThreshold"] = document_threshold
        if chunk_threshold is not None:
            payload["chunkThreshold"] = chunk_threshold
        if rerank is not None:
            payload["rerank"] = rerank
        if rewrite_query is not None:
            payload["rewriteQuery"] = rewrite_query
        if only_matching_chunks is not None:
            payload["onlyMatchingChunks"] = only_matching_chunks
        return self._post("/v3/search", payload)

    # -- plugin-faithful ingest + recall (the Hermes supermemory plugin path) --
    def ingest_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        container_tag: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST /v4/conversations: the endpoint the real Hermes plugin uses
        to ingest one full-session conversation at session end
        (plugins/memory/supermemory/__init__.py:397-417). The payload shape
        mirrors the plugin's ``ingest_conversation``: ``conversationId``
        plus ``messages`` (a list of ``{role, content}``), plus
        ``containerTags`` (array), plus an optional ``metadata``. Like
        /v3/documents, this call is ASYNCHRONOUS. It returns instantly with
        ``{id, conversationId, status:"queued"}``, and the returned ``id``
        is a document id, drainable through /v3/documents/{id} (verified on
        server 0.0.5).

        This stamps ``sm_source:"hermes"`` into metadata exactly as the
        plugin's ``_merge_metadata`` does. This is a functional routing key
        for the Supermemory app's Hermes Space, inert on self-hosted, kept
        here for plugin fidelity.
        """
        merged_meta: Dict[str, Any] = {"sm_source": "hermes"}
        if metadata:
            merged_meta.update(metadata)
        payload: Dict[str, Any] = {
            "conversationId": conversation_id,
            "messages": messages,
            "containerTags": [container_tag],
            "metadata": merged_meta,
        }
        return self._post("/v4/conversations", payload)

    def get_profile(
        self,
        query: Optional[str],
        container_tag: str,
    ) -> Dict[str, Any]:
        """POST /v4/profile: the recall endpoint the plugin auto-injects
        through ``prefetch`` -> ``get_profile`` (__init__.py:714-729,
        :356-379). On server 0.0.5 this is a POST (GET returns 404) with
        ``{containerTag, q}``, returning ``{profile:{static:[...],
        dynamic:[...], buckets:{...}}, searchResults:{results:[...], total,
        timing}}`` (camelCase). Returns a normalized dict: ``{static,
        dynamic, buckets, search_results}``, where ``search_results`` is the
        raw list of profile search-result items (the same item schema as
        /v4/search results)."""
        payload: Dict[str, Any] = {"containerTag": container_tag}
        if query:
            payload["q"] = query
        resp = self._post("/v4/profile", payload)
        profile = resp.get("profile") or {}
        search = resp.get("searchResults")
        if search is None:
            search = resp.get("search_results")
        if isinstance(search, dict):
            results = search.get("results") or []
        elif isinstance(search, list):
            results = search
        else:
            results = []
        return {
            "static": profile.get("static") or [],
            "dynamic": profile.get("dynamic") or [],
            "buckets": profile.get("buckets") or {},
            "search_results": results,
        }


class SupermemoryServer:
    """Spawns and owns a disposable self-hosted Supermemory server for one run.

    This module configures the server's INTERNAL extraction LLM through the
    subprocess env (its own ``OPENAI_*``), kept separate from the harness
    answer/judge model. It captures the API key that the server prints on
    first boot from the server's stdout. Pass ``api_key`` (env
    ``SUPERMEMORY_API_KEY``) to preset it instead.
    """

    def __init__(
        self,
        data_dir: str,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        # extraction LLM (server-internal): falls back to harness OPENAI_*,
        # so a single OpenRouter key can drive both roles for a smoke test.
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dimensions: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        log_path: Optional[str] = None,
    ):
        self.port = port or _free_port()
        # A caller-supplied port is fixed. An auto-allocated one is re-drawn on
        # every respawn (see start), so a terminated process that still holds
        # the old port cannot make the next boot fail to bind.
        self._port_explicit = port is not None
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.data_dir = os.path.abspath(data_dir)
        # Use a preset key if one exists. Otherwise, parse the boot banner.
        self.api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY") or None
        # A preset key is authoritative for every boot. Without one, the banner
        # is the only source, and _await_ready re-reads it on each boot.
        self._api_key_preset = bool(self.api_key)
        self.llm_api_key = llm_api_key or os.environ.get("SUPERMEMORY_LLM_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        self.llm_base_url = llm_base_url or os.environ.get("SUPERMEMORY_LLM_BASE_URL") \
            or os.environ.get("OPENAI_BASE_URL")
        self.llm_model = llm_model or os.environ.get("SUPERMEMORY_LLM_MODEL") \
            or os.environ.get("OPENAI_MODEL")
        self.embedding_provider = embedding_provider \
            or os.environ.get("SUPERMEMORY_EMBEDDING_PROVIDER", "local")
        self.embedding_model = embedding_model \
            or os.environ.get("SUPERMEMORY_EMBEDDING_MODEL")
        self.embedding_dimensions = embedding_dimensions \
            or os.environ.get("SUPERMEMORY_EMBEDDING_DIMENSIONS")
        # OPTIONAL OpenAI-compatible remote embedder. The 0.0.5 binary
        # exposes SUPERMEMORY_EMBEDDING_BASE_URL / _API_KEY (provider
        # `openai` or `custom`), so embeddings can point at the shared
        # vllm-embed (bge-small-en-v1.5, 384 dimensions) instead of the local
        # Xenova bge-base default (768 dimensions), for cross-provider
        # embedder parity. When these vars are unset, the local ONNX
        # embedder stays unchanged; this does NOT flip the default. See
        # docs/DECISIONS.md.
        self.embedding_base_url = embedding_base_url \
            or os.environ.get("SUPERMEMORY_EMBEDDING_BASE_URL")
        self.embedding_api_key = embedding_api_key \
            or os.environ.get("SUPERMEMORY_EMBEDDING_API_KEY")
        self.log_path = log_path or os.path.join(self.data_dir, "server.log")
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        # Boots of this instance. Clock-sync arms respawn the server once per
        # session on the SAME data dir, so this is >1 there.
        self.boot_count = 0

    def _spawn_cmd(self) -> List[str]:
        # Default launcher: the globally-installed `supermemory` CLI's
        # `local start`. --no-install ensures a booted image never re-fetches
        # the binary at run time (the Dockerfile bakes it at build time).
        # Override this entirely with SUPERMEMORY_SERVER_CMD (space-split) to
        # point at a raw `supermemory-server` binary or a custom wrapper.
        override = os.environ.get("SUPERMEMORY_SERVER_CMD", "").strip()
        if override:
            return override.split()
        return ["supermemory", "local", "start", "--no-install", "--port", str(self.port)]

    def _env(self) -> Dict[str, str]:
        # Clock-sync arms (BENCH_CLOCKSYNC=1): ONLY this spawned server
        # child preloads libfaketime, so its perceived OS clock tracks the
        # dataset's logical session date (the shared driver rewrites
        # BENCH_CLOCKSYNC_FILE per session). This deliberately leaves the
        # harness Python process un-faked, so the time.time()-based
        # deadlines it owns (wait_for_drain, ~:175, and _await_ready, ~:492)
        # measure real wall-clock time and are safe by construction. This
        # injects LD_PRELOAD into the CHILD env here, never the shell,
        # exactly as benchmark/docker/clock_sync.sh documents for this
        # provider. Everything here is inert unless clock-sync is enabled.
        env = dict(os.environ)
        # Server core.
        env["PORT"] = str(self.port)
        env["SUPERMEMORY_PORT"] = str(self.port)
        env["SUPERMEMORY_DATA_DIR"] = self.data_dir
        env["SUPERMEMORY_DISABLE_TELEMETRY"] = env.get("SUPERMEMORY_DISABLE_TELEMETRY", "1")
        # Extraction LLM: the server reads OPENAI_* for its internal model.
        # This OVERWRITES these vars in the child, so the server's extraction
        # model stays decoupled from whatever the harness answer/judge model
        # is in the parent process.
        if self.llm_api_key:
            env["OPENAI_API_KEY"] = self.llm_api_key
        if self.llm_base_url:
            env["OPENAI_BASE_URL"] = self.llm_base_url
        if self.llm_model:
            env["OPENAI_MODEL"] = self.llm_model
        # Embeddings (local ONNX by default; no key needed).
        env["SUPERMEMORY_EMBEDDING_PROVIDER"] = self.embedding_provider
        if self.embedding_model:
            env["SUPERMEMORY_EMBEDDING_MODEL"] = self.embedding_model
        if self.embedding_dimensions:
            env["SUPERMEMORY_EMBEDDING_DIMENSIONS"] = self.embedding_dimensions
        # Forwarded only when a remote embedder is requested. If absent, the
        # server uses the local ONNX embedder.
        if self.embedding_base_url:
            env["SUPERMEMORY_EMBEDDING_BASE_URL"] = self.embedding_base_url
        if self.embedding_api_key:
            env["SUPERMEMORY_EMBEDDING_API_KEY"] = self.embedding_api_key
        # This also exports a preset key, in case the server honors it as the
        # bearer key instead of generating one (harmless if the server
        # ignores it).
        if self.api_key:
            env["SUPERMEMORY_API_KEY"] = self.api_key
        # Clock-sync: preload libfaketime into the server child ONLY,
        # pointed at the shared per-shard timestamp file. This matches the
        # FAKETIME_* contract that clock_sync.sh / clock_sync.py define
        # (NO_CACHE for live stepping, DONT_FAKE_MONOTONIC so timeouts and
        # sleeps stay real, NO_FAKE_STAT so observed file mtimes stay real).
        # This is a no-op unless BENCH_CLOCKSYNC=1 and the file is set.
        if os.environ.get("BENCH_CLOCKSYNC") == "1" and os.environ.get("BENCH_CLOCKSYNC_FILE"):
            libfaketime = os.environ.get(
                "BENCH_LIBFAKETIME",
                "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1",
            )
            env["LD_PRELOAD"] = libfaketime
            env["FAKETIME_TIMESTAMP_FILE"] = os.environ["BENCH_CLOCKSYNC_FILE"]
            env["FAKETIME_NO_CACHE"] = "1"
            env["FAKETIME_DONT_FAKE_MONOTONIC"] = "1"
            env["NO_FAKE_STAT"] = "1"
        return env

    def start(self, ready_timeout: float = 300.0) -> SupermemoryClient:
        """Boot the server and return a client for THIS boot.

        Safe to call again after ``close(remove_data=False)``: the data dir is
        untouched, so the store carries over. The returned client is a NEW
        object bound to this boot's base_url and bearer key. The caller must
        drop every reference to the previous client.
        """
        if self.boot_count and not self._port_explicit:
            # Fresh ephemeral port per respawn. The terminated process can hold
            # the old port for a moment, and a bind failure would surface only
            # as "server exited early" after the readiness poll.
            self.port = _free_port()
            self.base_url = f"http://127.0.0.1:{self.port}"
        os.makedirs(self.data_dir, exist_ok=True)
        # Keep the log of the boot that just ended. A clock-sync persona
        # respawns once per session, and the truncation below used to leave
        # only the LAST boot's log, so a workflow failure in an earlier boot
        # had no evidence left (2026-08-02: an embedding step failed on boot 1
        # and only boot 2's log survived).
        if self.boot_count:
            try:
                os.replace(self.log_path,
                           f"{self.log_path}.boot{self.boot_count}")
            except OSError:
                pass
        self.boot_count += 1
        # A fresh boot log on each start ensures key-parsing never reads a
        # stale banner.
        self._log_fh = open(self.log_path, "w+", encoding="utf-8")
        cmd = self._spawn_cmd()
        print(
            f"[supermemory] starting server boot={self.boot_count} port={self.port} "
            f"data_dir={self.data_dir} "
            f"embeddings={self.embedding_provider} "
            f"extraction_model={self.llm_model or '(server default)'} "
            f"cmd={' '.join(cmd)}",
            flush=True,
        )
        self._proc = subprocess.Popen(
            cmd, cwd=self.data_dir, env=self._env(),
            stdout=self._log_fh, stderr=subprocess.STDOUT,
        )
        return self._await_ready(ready_timeout)

    def _read_log(self) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""

    def _await_ready(self, ready_timeout: float) -> SupermemoryClient:
        deadline = time.time() + ready_timeout
        client: Optional[SupermemoryClient] = None
        last_code: Optional[int] = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"supermemory server exited early (code={self._proc.returncode}); "
                    f"see {self.log_path}\n--- last log ---\n{self._read_log()[-2000:]}"
                )
            # 1) Learn the API key: use a preset key, or parse the boot banner.
            #    Without a preset key the banner is re-read on EVERY boot, not
            #    only the first. A respawn (clock-sync arms) may print a
            #    different key, and pinging with the previous one returns 401
            #    until the readiness deadline expires. start() truncates the
            #    log, so this only ever reads the current boot's banner.
            if not self._api_key_preset:
                m = _API_KEY_RE.search(self._read_log())
                if m and m.group(0) != self.api_key:
                    self.api_key = m.group(0)
                    # The held client carries the previous bearer key.
                    client = None
                    print(f"[supermemory] captured API key from boot banner "
                          f"({self.api_key[:6]}...)", flush=True)
            # 2) Once a key is known, confirm the HTTP API answers AND
            #    accepts it. This requires 2xx: a 401 or 403 means the held
            #    key (for example a stale preset from a persisted key file
            #    whose data dir was wiped) is not valid, so the server is
            #    NOT ready. Keep waiting, then fail with a key-specific
            #    message rather than trust or republish a bad key.
            if self.api_key:
                if client is None:
                    # abort_check reads THIS server's process handle, so every
                    # client handed out by start() fails fast once its own
                    # process dies (ServerDiedError) instead of retrying a
                    # corpse. close() clears _proc, and the client that
                    # observed it is discarded at the same moment.
                    client = SupermemoryClient(
                        self.base_url, self.api_key,
                        abort_check=lambda: (self._proc is not None
                                             and self._proc.poll() is not None),
                    )
                try:
                    last_code = client.ping()
                    if 200 <= last_code < 300:
                        print(f"[supermemory] ready at {self.base_url} (ping {last_code})",
                              flush=True)
                        return client
                except Exception:
                    pass
            time.sleep(0.75)
        tail = self._read_log()[-2000:]
        hint = ""
        if last_code in (401, 403):
            hint = (f" -- server is up but rejected the bearer key (HTTP {last_code}); "
                    f"a preset/persisted key is likely STALE. If the data dir was reset, "
                    f"clear the published key too (wipe both the data and shared volumes "
                    f"together).")
        raise TimeoutError(
            f"supermemory server not ready within {ready_timeout}s "
            f"(api_key_found={bool(self.api_key)}, last_ping={last_code}){hint}; "
            f"see {self.log_path}\n--- last log ---\n{tail}"
        )

    def close(self, remove_data: bool = False) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        if remove_data and os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir, ignore_errors=True)


if __name__ == "__main__":
    # Tiny self-test: boot, ingest, drain, search, teardown. This needs an
    # extraction LLM configured (OPENAI_* or SUPERMEMORY_LLM_*), because
    # ingest extracts memories.
    run_dir = os.path.join(CURRENT_DIR, ".supermemory_runs", f"selftest_{uuid.uuid4().hex[:8]}")
    srv = SupermemoryServer(data_dir=run_dir)
    cli = srv.start()
    try:
        tag = f"user_{uuid.uuid4().hex[:8]}"
        add = cli.add_document(
            content="User: I moved from Boston to Seattle and joined Amazon as a data scientist.\n"
                    "Assistant: Congrats on the Seattle move and the Amazon role!",
            container_tag=tag, metadata={"session": "s1"},
        )
        print("[selftest] add:", add, file=sys.stderr)
        drain = cli.wait_for_drain([add.get("id")] if add.get("id") else None, timeout=180)
        print("[selftest] drain:", drain, file=sys.stderr)
        res = cli.search_memories("Where does the user live and work?", tag, limit=5)
        print("[selftest] search total:", res.get("total"), file=sys.stderr)
        for r in res.get("results", []):
            print("   -", round(r.get("similarity", 0), 3),
                  r.get("memory") or r.get("chunk"), file=sys.stderr)
    finally:
        srv.close()
