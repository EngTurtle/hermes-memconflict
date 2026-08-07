"""Truncating pass-through proxy: Honcho -> OpenAI /v1/embeddings on vllm-embed.

Adapted from ``retaindb_server/embed_proxy.py`` (same stdlib-only
``http.server`` + ``urllib`` shape, same env-var naming, same entrypoint
health gate). RetainDB's proxy translates a bespoke protocol into the
OpenAI one. Honcho already speaks OpenAI, so this one changes ONE field:
it adds ``truncate_prompt_tokens`` to every upstream request.

WHY Honcho needs it
-------------------
``vllm-embed`` serves ``BAAI/bge-small-en-v1.5``, whose window is 512
tokens PER INPUT. vLLM answers 400 to any longer input:

    This model's maximum context length is 512 tokens. However, you
    requested 0 output tokens and your prompt contains at least 513
    input tokens ... (parameter=input_tokens, value=513)

Honcho's representation path calls
``EmbeddingClient.simple_batch_embed`` (``external/honcho/src/embedding_client.py:251``),
which does NO chunking and NO length check. Chunking exists only on the
other path, ``batch_embed`` -> ``_prepare_chunks`` (line 370). So one long
observation 400s the WHOLE save, for both observers at once
(``external/honcho/src/crud/representation.py:111``). Measured on smoke
``hn_smkmin_p0b``: 14 dropped saves against 11 completed deriver batches
in persona 0, sessions 0-2.

No vendor knob fixes it. ``EMBEDDING_MAX_INPUT_TOKENS`` (default 8192,
``src/config.py:705``) is not consulted by ``simple_batch_embed`` at all,
and lowering it to 512 would still not help, for two reasons: that path
skips the check, and Honcho measures length with tiktoken ``cl100k_base``
while bge-small tokenizes with BERT WordPiece, so the two token counts
disagree. Honcho also exposes no way to forward
``truncate_prompt_tokens`` to the embedding server. ``external/`` is a
pinned submodule and must not be edited, so the fix goes in front of the
server instead.

WHY truncation loses nothing
----------------------------
bge-small-en-v1.5 has a 512-position encoder. It cannot attend past
position 512 under any configuration. Text beyond that point never
reaches the model, whether it is cut here or the request is refused.
Truncating turns a dropped save into a save of exactly the vector the
model would have produced. The alternative on offer is not a longer
vector, it is no vector.

vLLM truncates with the SERVED model's own tokenizer, so the cut lands at
the real window boundary. A local tokenizer would need the bge vocabulary
in the image and would still only approximate what the server does.

Routes:
  * ``POST /v1/embeddings`` (or ``/embeddings``)  inject truncate -> upstream, body returned verbatim
  * ``GET  /v1/models`` (or ``/models``)          plain pass-through
  * ``GET  /health``                              {"status":"ok"}, local, does not touch upstream

Env:
  HONCHO_EMBED_PROXY_PORT       listen port (default 3198)
  HONCHO_EMBED_PROXY_UPSTREAM   OpenAI-compatible base (default http://vllm-embed:8000/v1)
  HONCHO_EMBED_PROXY_TRUNCATE   value sent as truncate_prompt_tokens (default -1)
  HONCHO_EMBED_PROXY_TIMEOUT_S  upstream request timeout seconds (default 120)

Upstream errors are returned with the upstream status code and body, so a
real failure still reaches Honcho's retry and log path unchanged.

Clock note: this process runs in the entrypoint's own clock domain, NOT
under the libfaketime LD_PRELOAD that ``honcho/_honcho_server.py`` injects
into its children. Either way it is safe. It opens plain HTTP to
``vllm-embed`` on the compose network and never does TLS, so a perceived
year in the dataset past cannot produce a "certificate not yet valid"
failure -- the failure mode that forced the build-time tiktoken cache in
``Dockerfile.honcho``.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("HONCHO_EMBED_PROXY_PORT", "3198"))
UPSTREAM = os.environ.get(
    "HONCHO_EMBED_PROXY_UPSTREAM", "http://vllm-embed:8000/v1"
).rstrip("/")
# -1 tells vLLM "truncate to this model's max_model_len", so the cut follows
# the served embedder instead of a number hardcoded here. bge-small-en-v1.5
# resolves it to 512. Set a literal to pin a smaller window.
TRUNCATE = int(os.environ.get("HONCHO_EMBED_PROXY_TRUNCATE", "-1"))
# 120s, not RetainDB's 55s: Honcho batches up to 2048 inputs per call
# (max_batch_size, embedding_client.py:180), which is a much larger unit of
# work than one RetainDB extraction batch.
TIMEOUT_S = float(os.environ.get("HONCHO_EMBED_PROXY_TIMEOUT_S", "120"))
API_KEY = os.environ.get("HONCHO_EMBED_PROXY_UPSTREAM_API_KEY", "local-vllm")


def _forward(path, body=None, method="GET"):
    """Send one request upstream. Returns (status, raw_bytes, content_type)."""
    req = urllib.request.Request(
        f"{UPSTREAM}{path}",
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")


class Handler(BaseHTTPRequestHandler):
    # The default per-request stderr line is noise at batch scale, and the
    # deriver's log is what the run is read from.
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    def _send(self, status, raw, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, status, obj):
        # Compact separators: the entrypoint health gate greps the literal
        # '"status":"ok"'. json.dumps' default '{"status": "ok"}' has a space
        # after the colon and never matches that grep.
        self._send(status, json.dumps(obj, separators=(",", ":")).encode("utf-8"))

    def _relay_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            raw = e.read()
            # Return the upstream status and body unchanged. A 400 that is
            # NOT a length overflow must still reach Honcho intact.
            self._send(e.code, raw or b'{"error":"upstream error"}')
        elif isinstance(e, urllib.error.URLError):
            self._send_json(502, {"error": f"upstream unreachable: {e.reason}"})
        else:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _norm(self):
        """Strip the optional /v1 prefix, so both /v1/x and /x route the same."""
        p = self.path.split("?", 1)[0].rstrip("/")
        return p[3:] if p.startswith("/v1/") else p

    def do_GET(self):
        p = self._norm()
        if p == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if p == "/models":
            try:
                status, raw, ctype = _forward("/models")
                self._send(status, raw, ctype)
            except Exception as e:  # noqa: BLE001 - always answer with JSON
                self._relay_error(e)
            return
        self._send_json(404, {"error": f"not found: {self.path}"})

    def do_POST(self):
        if self._norm() != "/embeddings":
            self._send_json(404, {"error": f"not found: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
            if not isinstance(req, dict):
                self._send_json(400, {"error": "body must be a JSON object"})
                return
            # The only mutation. An explicit caller value wins, so a future
            # Honcho release that sets the field keeps control of it.
            req.setdefault("truncate_prompt_tokens", TRUNCATE)
            status, out, ctype = _forward(
                "/embeddings",
                body=json.dumps(req).encode("utf-8"),
                method="POST",
            )
            self._send(status, out, ctype)
        except Exception as e:  # noqa: BLE001 - always answer with JSON
            self._relay_error(e)


def main():
    print(
        f"[honcho-embed-proxy] listening on 127.0.0.1:{PORT} -> {UPSTREAM} "
        f"truncate_prompt_tokens={TRUNCATE} timeout={TIMEOUT_S}s",
        flush=True,
    )
    # 127.0.0.1 only: every client is a child process of this container
    # (spawned API + deriver). Nothing outside needs to reach it.
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
