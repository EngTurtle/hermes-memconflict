"""Lifecycle manager for a self-hosted OpenViking server (pip ``openviking``).

OpenViking ships the server and its storage in ONE pip package: the
``openviking-server`` console script is a FastAPI/uvicorn app, and the AGFS
content store plus the local vector index live under one
``storage.workspace`` directory. There is no external database, so this
manager is simpler than ``honcho/_honcho_server.py``: it writes one config
file, spawns one process, and waits for ``/ready``.

NEVER ``import openviking`` here. The provider folder that holds this file is
named ``openviking``, so the name would bind to the folder, not the pip
package. Every call goes over HTTP with ``httpx``, the same transport the
Hermes plugin uses.

TWO LLM ROLES, KEPT APART (docs/DECISIONS.md):

  * OpenViking's INTERNAL model — memory extraction at commit, and the
    ``search/search`` intent analysis. It is configured in the ``vlm`` block
    of the config file this module writes, fed from ``OPENVIKING_LLM_*``.
  * The shared ANSWER and JUDGE model — the fairness-locked harness model
    that ``eval_common`` reaches through ``OPENAI_*`` in the parent process.
    This module never reads or writes those.

Two modes:

  * SPAWN (default): write ``ov.conf`` into the run directory, start
    ``openviking-server --config <path>`` on an ephemeral port, poll
    ``/ready``.
  * ATTACH (``OPENVIKING_SERVER_MODE=shared`` or ``OPENVIKING_ENDPOINT``
    set): health-check an already-running server and spawn nothing.

The config schema sets ``extra: "forbid"``, so an unknown key makes the
server exit at startup. Add a key only after checking it against the vendor's
config model.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import httpx

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

#: Vendor default bind port. Used only to build the shared-mode fallback URL.
DEFAULT_PORT = 1933


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def sanitize_run_tag(name: str) -> str:
    """Make a run tag safe as a directory name."""
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))
    return out[:64] or "adhoc"


class OpenVikingServer:
    """Own the ``openviking-server`` process, its config file, and its workspace."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        workspace: Optional[str] = None,
        run_dir: Optional[str] = None,
        server_bin: Optional[str] = None,
        # OpenViking's internal model (extraction, query planning).
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_max_tokens: Optional[int] = None,
        llm_temperature: Optional[float] = None,
        llm_max_concurrent: Optional[int] = None,
        llm_timeout: Optional[int] = None,
        # OpenViking's embedder (the retrieval-embedding surface).
        embedder_model: Optional[str] = None,
        embedder_base_url: Optional[str] = None,
        embedder_api_key: Optional[str] = None,
        embedder_dims: Optional[int] = None,
    ):
        # ATTACH mode: an external server already runs, so this process spawns
        # nothing and only health-checks the URL.
        self.attach_url = base_url or _env("OPENVIKING_ENDPOINT")
        if _env("OPENVIKING_SERVER_MODE", "spawn") == "shared" and not self.attach_url:
            self.attach_url = f"http://127.0.0.1:{DEFAULT_PORT}"
        self.port = port or int(_env("OPENVIKING_SERVER_PORT", "0") or 0) or _free_port()
        self.base_url = (self.attach_url or f"http://127.0.0.1:{self.port}").rstrip("/")

        run_tag = sanitize_run_tag(_env("RUN_TAG", "adhoc"))
        self.run_dir = os.path.abspath(
            run_dir or _env("OPENVIKING_RUN_DIR",
                            os.path.join(CURRENT_DIR, ".openviking_runs", run_tag))
        )
        self.workspace = os.path.abspath(
            workspace or _env("OPENVIKING_WORKSPACE", os.path.join(self.run_dir, "data"))
        )
        self.config_path = os.path.join(self.run_dir, "ov.conf")
        self.log_path = os.path.join(self.run_dir, "server.log")
        self.server_bin = server_bin or _env("OPENVIKING_SERVER_BIN")

        self.llm_model = llm_model or _env("OPENVIKING_LLM_MODEL", "gpt-5.4-mini")
        self.llm_base_url = llm_base_url or _env("OPENVIKING_LLM_BASE_URL")
        self.llm_api_key = (llm_api_key or _env("OPENVIKING_LLM_API_KEY")
                            or _env("OPENROUTER_API_KEY") or "local")
        self.llm_max_tokens = int(
            llm_max_tokens if llm_max_tokens is not None
            else int(_env("OPENVIKING_LLM_MAX_TOKENS", "4096"))
        )
        # 0.0 is the value the vendor's own sample config ships for `vlm`.
        self.llm_temperature = float(
            llm_temperature if llm_temperature is not None
            else float(_env("OPENVIKING_LLM_TEMPERATURE", "0.0"))
        )
        # The vendor default is 64. vllm-gen serves one benchmark at a time,
        # and 64 concurrent extraction calls queue behind each other inside
        # the engine, so every commit reports the queue time as extraction
        # time. 8 keeps the server busy without burying the serving process.
        self.llm_max_concurrent = int(
            llm_max_concurrent if llm_max_concurrent is not None
            else int(_env("OPENVIKING_LLM_MAX_CONCURRENT", "8"))
        )
        self.llm_timeout = int(
            llm_timeout if llm_timeout is not None
            else int(_env("OPENVIKING_LLM_TIMEOUT", "600"))
        )
        # JSON merged into every vlm request (`vlm.extra_request_body`, a
        # vendor-exposed knob). The OpenRouter gpt-oss-20b path needs
        # {"reasoning": {"effort": "low"}}: at the default effort the
        # extraction call burns its whole token budget on reasoning
        # (empty_response in extract_loop) or runs past OpenRouter's
        # keep-alive window, which returns a newline-padded body with no
        # JSON payload. Same mechanism as HONCHO_LLM_THINKING_EFFORT=low.
        raw_extra = _env("OPENVIKING_LLM_EXTRA_BODY", "")
        try:
            self.llm_extra_body = json.loads(raw_extra) if raw_extra else None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OPENVIKING_LLM_EXTRA_BODY is not valid JSON: {exc}") from exc

        self.embedder_model = embedder_model or _env("OPENVIKING_EMBEDDER_MODEL",
                                                     "bge-small-en-v1.5")
        self.embedder_base_url = embedder_base_url or _env("OPENVIKING_EMBEDDER_BASE_URL")
        self.embedder_api_key = (embedder_api_key
                                 or _env("OPENVIKING_EMBEDDER_API_KEY", "local-vllm"))
        self.embedder_dims = int(
            embedder_dims if embedder_dims is not None
            else int(_env("OPENVIKING_EMBEDDER_DIMS", "768"))
        )

        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        #: Version string the server reports at /health or /ready, when it
        #: reports one. It lands in the results row as provenance.
        self.version: Optional[str] = None

    # -- configuration --------------------------------------------------------
    def resolve_server_bin(self) -> str:
        """Find the ``openviking-server`` console script.

        The script sits next to the interpreter that installed the package,
        so look there before ``PATH``: a harness venv and the ambient
        environment can hold different versions.
        """
        if self.server_bin:
            return self.server_bin
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
        for name in ("openviking-server", "openviking-server.exe"):
            candidate = os.path.join(bin_dir, name)
            if os.path.isfile(candidate):
                return candidate
        found = shutil.which("openviking-server")
        if found:
            return found
        raise RuntimeError(
            "openviking-server not found next to the interpreter or on PATH — "
            "run `pip install openviking`, or set OPENVIKING_SERVER_BIN.")

    def build_config(self) -> Dict[str, Any]:
        """The ov.conf body.

        ``rerank`` is OMITTED, not disabled: the section has no enabled flag,
        so absence is how reranking stays off. ``query_planner: null`` makes
        the ``search/search`` intent analysis fall back to ``vlm``, so one
        model serves both internal roles. ``encoding_format: "float"`` is
        explicit because a base64 embedding payload does not survive every
        gateway on the path to vllm-embed.
        """
        return {
            "default_account": _env("OPENVIKING_ACCOUNT", "default"),
            "default_user": "default",
            "storage": {
                "workspace": self.workspace,
                "agfs": {"backend": "local"},
                "vectordb": {"backend": "local"},
            },
            "embedding": {
                "max_concurrent": 10,
                "dense": {
                    "provider": "openai",
                    "api_base": self.embedder_base_url,
                    "api_key": self.embedder_api_key,
                    "model": self.embedder_model,
                    "dimension": self.embedder_dims,
                    "encoding_format": "float",
                    "batch_size": 32,
                },
            },
            "vlm": {
                "provider": "openai",
                "api_base": self.llm_base_url,
                "api_key": self.llm_api_key,
                "model": self.llm_model,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens,
                "max_retries": 3,
                "timeout": self.llm_timeout,
                "max_concurrent": self.llm_max_concurrent,
                "thinking": False,
                "stream": False,
                **({"extra_request_body": self.llm_extra_body}
                   if self.llm_extra_body else {}),
            },
            "query_planner": None,
            "server": {"host": "127.0.0.1", "port": self.port, "auth_mode": "dev"},
            # No telemetry section: 0.4.12's TelemetryConfig is {"tracer": ...}
            # with tracer.enabled False by default, and it rejects unknown
            # fields ("Unknown config field 'telemetry.enabled'" kills the
            # server at boot). server.usage_reporter.enabled also defaults
            # False, so omission is the off state for both.
        }

    def write_config(self) -> str:
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.workspace, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(self.build_config(), fh, indent=2)
        return self.config_path

    def child_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        # The server reads its config only at startup, and only from this
        # path. Both the flag and the variable are set so a child process the
        # server spawns itself resolves the same file.
        env["OPENVIKING_CONFIG_FILE"] = self.config_path

        # Clock-sync arms: preload libfaketime into the SERVER CHILD only, so
        # its perceived OS clock tracks the dataset's logical session date
        # while the harness process keeps real time for its own deadlines.
        # Identical contract to _honcho_server.py and _supermemory_server.py.
        # Inert unless BENCH_CLOCKSYNC=1.
        if os.environ.get("BENCH_CLOCKSYNC") == "1" and os.environ.get("BENCH_CLOCKSYNC_FILE"):
            env["LD_PRELOAD"] = os.environ.get(
                "BENCH_LIBFAKETIME",
                "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1")
            env["FAKETIME_TIMESTAMP_FILE"] = os.environ["BENCH_CLOCKSYNC_FILE"]
            env["FAKETIME_NO_CACHE"] = "1"
            env["FAKETIME_DONT_FAKE_MONOTONIC"] = "1"
            env["NO_FAKE_STAT"] = "1"
        return env

    # -- lifecycle ------------------------------------------------------------
    def start(self, ready_timeout: float = 180.0) -> str:
        if self.attach_url:
            self._await_ready(ready_timeout)
            print(f"[openviking] attached to {self.base_url} "
                  f"(version={self.version})", flush=True)
            return self.base_url

        if not self.embedder_base_url:
            # A broken embedder does not stop OpenViking. The API keeps
            # answering, commits keep returning `accepted`, and the failure
            # surfaces only as error_count on the embedding queue in
            # POST /api/v1/system/wait. Fail here instead, where the message
            # names the missing setting.
            raise RuntimeError(
                "OPENVIKING_EMBEDDER_BASE_URL is not set. OpenViking needs an "
                "OpenAI-compatible embeddings endpoint: point it at vllm-embed, "
                "or start honcho/_local_embed_server.py for a host smoke.")

        self.write_config()
        cmd = [self.resolve_server_bin(), "--config", self.config_path]
        self._log_fh = open(self.log_path, "w+", encoding="utf-8")
        print(f"[openviking] starting server on port {self.port} "
              f"(workspace={self.workspace}, model={self.llm_model}, "
              f"embedder={self.embedder_model}/{self.embedder_dims}d)", flush=True)
        # start_new_session puts the server in its own process group. The
        # server spawns an `agfs` child of its own, and terminating the group
        # is what stops that child too.
        self._proc = subprocess.Popen(
            cmd, cwd=self.run_dir, env=self.child_env(),
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._await_ready(ready_timeout)
        print(f"[openviking] ready at {self.base_url} (version={self.version})", flush=True)
        return self.base_url

    def _read_log(self) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""

    def _probe(self, path: str) -> Optional[httpx.Response]:
        try:
            return httpx.get(f"{self.base_url}{path}", timeout=5.0)
        except Exception:
            return None

    def _capture_version(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        body = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        for key in ("version", "server_version", "openviking_version"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                self.version = value.strip()
                return

    def _await_ready(self, ready_timeout: float) -> None:
        deadline = time.time() + ready_timeout
        last_error = "no response"
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"openviking server exited early (code={self._proc.returncode}); "
                    f"see {self.log_path}\n--- last log ---\n{self._read_log()[-2500:]}")
            # /ready reports storage and index readiness; /health answers 200
            # as soon as the HTTP app binds. Poll /ready, and accept /health
            # only when the build has no /ready route.
            resp = self._probe("/ready")
            if resp is not None and resp.status_code == 404:
                resp = self._probe("/health")
            if resp is not None:
                if 200 <= resp.status_code < 300:
                    try:
                        payload = resp.json()
                    except Exception:
                        payload = None
                    self._capture_version(payload)
                    status = ""
                    if isinstance(payload, dict):
                        body = (payload.get("result")
                                if isinstance(payload.get("result"), dict) else payload)
                        status = str(body.get("status") or "").lower()
                    if status in ("", "ready", "ok", "healthy", "up"):
                        return
                    last_error = f"status={status}"
                else:
                    last_error = f"HTTP {resp.status_code}"
            time.sleep(1.0)
        tail = self._read_log()[-2500:]
        raise TimeoutError(
            f"openviking server not ready within {ready_timeout}s at {self.base_url} "
            f"(last: {last_error})\n--- server log ---\n{tail}")

    def alive(self) -> bool:
        # A spawned child that exited is dead regardless of what the port
        # answers (another process could own it). For a live child, and in
        # attach mode, only an HTTP probe can tell.
        if self._proc is not None and self._proc.poll() is not None:
            return False
        resp = self._probe("/health")
        if resp is None:
            resp = self._probe("/ready")
        return resp is not None and 200 <= resp.status_code < 300

    def close(self, remove_run_dir: bool = False) -> None:
        if self._proc is not None:
            try:
                # Signal the whole process group: the `agfs` child outlives a
                # bare terminate() on the parent and keeps the workspace lock
                # file (.openviking.pid), which blocks the next run.
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                except (AttributeError, ProcessLookupError, PermissionError, OSError):
                    self._proc.terminate()
                try:
                    self._proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print("[openviking] server did not stop in 20s; killing", flush=True)
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except (AttributeError, ProcessLookupError, PermissionError, OSError):
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
        if remove_run_dir and os.path.isdir(self.run_dir):
            shutil.rmtree(self.run_dir, ignore_errors=True)


if __name__ == "__main__":
    # Self-test: boot, ready check, tear down. This needs an embedder at
    # OPENVIKING_EMBEDDER_BASE_URL and an internal model at OPENVIKING_LLM_*.
    os.environ.setdefault("RUN_TAG", "selftest")
    server = OpenVikingServer()
    try:
        url = server.start()
        print(f"[selftest] up at {url}; version={server.version}", file=sys.stderr)
    finally:
        server.close()
