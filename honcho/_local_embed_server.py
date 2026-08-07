"""OpenAI-compatible ``/v1/embeddings`` shim over fastembed bge-small-en-v1.5.

Honcho REQUIRES an embedder. It has no local-model transport: the only
self-hosted option is ``EMBEDDING_MODEL_CONFIG__TRANSPORT=openai`` pointed at
an OpenAI-compatible endpoint. Offline and Docker runs point that at the
shared ``vllm-embed`` service (bge-small-en-v1.5, dim 384), the same embedding
surface Mnemosyne, Hindsight, and mem0 use. A HOST smoke has no vllm-embed, so
this module serves the same model family from fastembed instead, on the same
384 dimensions, so a host smoke and a Docker run agree on the vector width and
the pgvector column type.

This is smoke-test infrastructure. It is never the headline serving path: a
banked run points ``HONCHO_EMBEDDER_BASE_URL`` at vllm-embed.

Run it standalone::

    python honcho/_local_embed_server.py --port 8099

or spawn it in-process::

    srv = LocalEmbedServer(port=0); srv.start(); srv.base_url  # http://127.0.0.1:<port>/v1
"""

import argparse
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

# fastembed downloads the ONNX weights on first use. HF's Xet backend is not
# reachable through the egress proxy, so force the classic CDN (mem0
# precedent, CLAUDE.md).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMS = 384


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Embedder:
    """Lazy fastembed wrapper.

    The model loads on the first request, not at import, so a caller that
    only wants the port can start the server without paying the weight
    download.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.load()
        assert self._model is not None
        return [list(map(float, v)) for v in self._model.embed(texts)]


class _Handler(BaseHTTPRequestHandler):
    embedder: _Embedder = _Embedder()
    served_model: str = "bge-small-en-v1.5"
    verbose: bool = False

    def log_message(self, fmt: str, *args: Any) -> None:
        # BaseHTTPRequestHandler logs every request to stderr. A drain loop
        # embeds thousands of chunks, so the default would bury the run log.
        if self.verbose:
            sys.stderr.write("[embed] " + (fmt % args) + "\n")

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  (stdlib handler naming)
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok", "model": self.served_model})
            return
        if path in ("/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": [
                {"id": self.served_model, "object": "model", "owned_by": "local"},
            ]})
            return
        self._send_json(404, {"error": {"message": f"no route {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/v1/embeddings", "/embeddings"):
            self._send_json(404, {"error": {"message": f"no route {path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._send_json(400, {"error": {"message": f"bad request body: {e}"}})
            return

        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list):
            # An OpenAI client may send pre-tokenized integer arrays. This
            # shim cannot detokenize them, so reject that shape loudly
            # instead of embedding a stringified list.
            if raw_input and not all(isinstance(t, str) for t in raw_input):
                self._send_json(400, {"error": {
                    "message": "this shim accepts string input only, not token ids"}})
                return
            texts = [str(t) for t in raw_input]
        else:
            self._send_json(400, {"error": {"message": "missing 'input'"}})
            return

        if not texts:
            self._send_json(200, {"object": "list", "data": [],
                                  "model": self.served_model,
                                  "usage": {"prompt_tokens": 0, "total_tokens": 0}})
            return

        try:
            vectors = self.embedder.embed(texts)
        except Exception as e:  # pragma: no cover
            self._send_json(500, {"error": {"message": f"embed failed: {e}"}})
            return

        # `dimensions` is accepted and ignored: bge-small has exactly one
        # output width (384), so honoring or dropping the parameter gives the
        # same vector. vLLM instead 400s on the parameter, which is why the
        # mem0 adapter strips it; this shim absorbs it so no caller-side shim
        # is needed here.
        approx_tokens = sum(max(1, len(t) // 4) for t in texts)
        self._send_json(200, {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)],
            "model": self.served_model,
            "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens},
        })


class LocalEmbedServer:
    """Own the shim as a background thread inside the calling process."""

    def __init__(self, port: Optional[int] = None, host: str = "127.0.0.1",
                 model_name: str = DEFAULT_MODEL, served_model: str = "bge-small-en-v1.5",
                 preload: bool = True, verbose: bool = False):
        self.host = host
        self.port = port or _free_port()
        self.model_name = model_name
        self.served_model = served_model
        self.preload = preload
        self.verbose = verbose
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> str:
        handler = type("_BoundHandler", (_Handler,), {
            "embedder": _Embedder(self.model_name),
            "served_model": self.served_model,
            "verbose": self.verbose,
        })
        if self.preload:
            # Load weights BEFORE the first Honcho request. The deriver's
            # embed call has its own timeout, and a cold first-call download
            # can exceed it.
            handler.embedder.load()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="local-embed", daemon=True)
        self._thread.start()
        print(f"[embed] serving {self.model_name} as '{self.served_model}' "
              f"at {self.base_url}", flush=True)
        return self.base_url

    def close(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI-compatible embeddings shim (fastembed)")
    parser.add_argument("--host", default=os.environ.get("HONCHO_EMBED_SHIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HONCHO_EMBED_SHIM_PORT", "8099")))
    parser.add_argument("--model", default=os.environ.get("HONCHO_EMBED_SHIM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--served_model", default=os.environ.get("HONCHO_EMBEDDER_MODEL", "bge-small-en-v1.5"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    server = LocalEmbedServer(port=args.port, host=args.host, model_name=args.model,
                              served_model=args.served_model, verbose=args.verbose)
    server.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.close()
