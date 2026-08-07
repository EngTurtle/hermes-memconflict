"""Lifecycle manager and thin HTTP client for a local RetainDB server.

RetainDB differs from Mnemosyne and Hindsight: it is not a Python SDK.
``@retaindb/local`` is a bundled Node HTTP server (one ``dist/cli.js``) that
speaks a small REST API on :3111. RetainDB Local has no LLM inside it.
Memory building is heuristic (role-prefixing, quality gating, dedup,
``inferMemoryType``), and retrieval is a fixed fusion of BM25, vector cosine,
and concept-graph score (RRF, k=60) with a proximity rerank. The only ML
lever is the embedding provider (``hash`` default, or ``local-transformers``,
which is Xenova/all-MiniLM-L6-v2 in-process). See docs/BENCHMARK_MATRIX.md.

This module owns that server for the benchmark, so the adapter is
self-contained. This mirrors how ``eval_hindsight.py`` boots an embedded
Hindsight daemon.

  * ``RetainDBServer`` spawns ``node .../@retaindb/local/dist/cli.js start``
    with a unique RETAINDB_HOME, RETAINDB_STORE, and RETAINDB_PORT, so runs
    stay isolated and disposable. It polls ``/retaindb/health`` until ready,
    then tears the process down.
  * ``RetainDBClient`` wraps the three endpoints the benchmark needs:
      - POST /v1/memory/ingest/session   (ingest a session's dialogue)
      - POST /v1/memory                  (add one memory, message granularity)
      - POST /v1/memory/search           (retrieve for a question)

To attach to an already-running server instead of spawning one, pass an
explicit base_url (RETAINDB_BASE_URL) and manage_server=False.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKG_DIR = os.path.join(CURRENT_DIR, ".retaindb_pkg")


def _find_cli_js(pkg_dir: str) -> str:
    """Find the bundled @retaindb/local CLI entrypoint."""
    candidate = os.path.join(pkg_dir, "node_modules", "@retaindb", "local", "dist", "cli.js")
    if os.path.isfile(candidate):
        return candidate
    # Fallback: an env override can point straight at a cli.js.
    override = os.environ.get("RETAINDB_CLI_JS")
    if override and os.path.isfile(override):
        return override
    raise FileNotFoundError(
        f"Could not find @retaindb/local cli.js under {pkg_dir}. "
        f"Run `npm install @retaindb/local` there, or set RETAINDB_CLI_JS."
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RetainDBClient:
    """Minimal REST client over the RetainDB Local API.

    Timeout and retry exist because embeddings run inside the Node server on
    the CPU (local-transformers). Under multi-shard load, a single
    ``/search`` call can legitimately take minutes. The 2026-07-21 10-shard
    run lost 2 shards (3 personas each) to one 120s read-timeout with no
    retry. Retries cover only transient failures (timeouts, connection
    errors, HTTP 5xx). HTTP 4xx errors are real errors and raise immediately.
    """

    def __init__(self, base_url: str, timeout: Optional[float] = None):
        self.base_url = base_url.rstrip("/")
        # Use `or`, not a default arg to .get, so a set-but-empty env var counts as unset.
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("RETAINDB_HTTP_TIMEOUT") or "300"
        )
        self.max_retries = int(os.environ.get("RETAINDB_HTTP_RETRIES") or "4")
        self._session = requests.Session()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 2):  # first try, then max_retries retries
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code < 500 or attempt > self.max_retries:
                    resp.raise_for_status()  # raises here on 4xx, or 5xx on the final attempt
                    return resp.json()
                reason = f"HTTP {resp.status_code}"
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt > self.max_retries:
                    raise
                reason = type(e).__name__
            delay = min(2.0 * (2 ** (attempt - 1)), 30.0)
            print(
                f"[retaindb] {path} attempt {attempt}/{self.max_retries + 1} failed "
                f"({reason}); retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
        raise RuntimeError(f"unreachable: retry loop exited for {url}")

    def health(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/retaindb/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def ingest_session(
        self,
        project: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        write_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ``messages`` forwards verbatim as {role, content, timestamp} dicts,
        # exactly the shape the plugin-faithful exchange arm needs. No
        # reshaping happens here. ``write_mode`` is optional and omitted
        # unless set, so the existing 'session' and 'message' arms stay
        # byte-identical. The exchange arm passes ``write_mode="sync"`` to
        # mirror the Hermes plugin's ingest payload.
        payload: Dict[str, Any] = {
            "project": project,
            "session_id": session_id,
            "user_id": user_id,
            "messages": messages,
        }
        if write_mode is not None:
            payload["write_mode"] = write_mode
        return self._post("/v1/memory/ingest/session", payload)

    def add_memory(
        self,
        project: str,
        content: str,
        memory_type: str = "factual",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "project": project,
            "content": content,
            "memory_type": memory_type,
            "session_id": session_id,
            "user_id": user_id,
        }
        if importance is not None:
            payload["importance"] = importance
        if metadata is not None:
            payload["metadata"] = metadata
        return self._post("/v1/memory", payload)

    def search(
        self,
        project: str,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._post(
            "/v1/memory/search",
            {
                "project": project,
                "query": query,
                "top_k": top_k,
                "user_id": user_id,
            },
        )


class RetainDBServer:
    """Spawns and owns a disposable RetainDB Local server for one run."""

    def __init__(
        self,
        profile: str,
        pkg_dir: str = DEFAULT_PKG_DIR,
        port: Optional[int] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        home_root: Optional[str] = None,
        log_path: Optional[str] = None,
    ):
        self.profile = profile
        self.cli_js = _find_cli_js(pkg_dir)
        self.port = port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.embedding_provider = embedding_provider or os.environ.get(
            "RETAINDB_EMBEDDING_PROVIDER", "hash"
        )
        self.embedding_model = embedding_model or os.environ.get("RETAINDB_EMBEDDING_MODEL")
        # Isolated, disposable home, so the run never touches ~/.retaindb.
        root = home_root or os.path.join(CURRENT_DIR, ".retaindb_runs")
        self.home = os.path.join(root, profile)
        self.log_path = log_path or os.path.join(self.home, "server.log")
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["RETAINDB_HOME"] = self.home
        env["RETAINDB_STORE"] = os.path.join(self.home, "local-store.json")
        env["RETAINDB_PORT"] = str(self.port)
        # Put the viewer on a unique port too, so concurrent runs never clash.
        env["RETAINDB_VIEWER_PORT"] = str(self.port + 2)
        env["RETAINDB_EMBEDDING_PROVIDER"] = self.embedding_provider
        if self.embedding_model:
            env["RETAINDB_EMBEDDING_MODEL"] = self.embedding_model
        return env

    def start(self, ready_timeout: float = 180.0) -> RetainDBClient:
        os.makedirs(self.home, exist_ok=True)
        self._log_fh = open(self.log_path, "w", encoding="utf-8")
        node_bin = os.environ.get("RETAINDB_NODE_BIN", "node")
        print(
            f"[retaindb] starting server profile={self.profile} port={self.port} "
            f"embeddings={self.embedding_provider} home={self.home}",
            flush=True,
        )
        self._proc = subprocess.Popen(
            [node_bin, self.cli_js, "start"],
            cwd=self.home,
            env=self._env(),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )
        client = RetainDBClient(self.base_url)
        deadline = time.time() + ready_timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"RetainDB server exited early (code={self._proc.returncode}); "
                    f"see {self.log_path}"
                )
            try:
                h = client.health()
                if h.get("status") == "ok":
                    print(f"[retaindb] healthy at {self.base_url}", flush=True)
                    return client
            except Exception as e:  # not up yet
                last_err = e
            time.sleep(0.5)
        raise TimeoutError(
            f"RetainDB server did not become healthy within {ready_timeout}s "
            f"(last error: {last_err}); see {self.log_path}"
        )

    def close(self, remove_home: bool = False) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
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
        if remove_home and os.path.isdir(self.home):
            shutil.rmtree(self.home, ignore_errors=True)


if __name__ == "__main__":
    # Tiny self-test: boot, ingest, search, teardown.
    prof = f"selftest_{uuid.uuid4().hex[:8]}"
    srv = RetainDBServer(prof)
    cli = srv.start()
    try:
        proj = f"p_{uuid.uuid4().hex[:8]}"
        r = cli.ingest_session(
            project=proj,
            session_id="s1",
            messages=[
                {"role": "user", "content": "I moved from Boston to Seattle and joined Amazon as a data scientist."},
                {"role": "assistant", "content": "Congrats on the Seattle move and the Amazon role!"},
            ],
        )
        print("[selftest] ingest:", r, file=sys.stderr)
        s = cli.search(project=proj, query="Where does the user live and work?", top_k=5)
        print("[selftest] search count:", s.get("count"), file=sys.stderr)
        for item in s.get("results", []):
            print("   -", round(item.get("score", 0), 3), item.get("content"), file=sys.stderr)
    finally:
        srv.close()
