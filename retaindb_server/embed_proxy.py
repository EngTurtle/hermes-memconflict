"""Protocol-translation proxy: RetainDB remote-inference embeddings -> OpenAI /v1/embeddings.

RetainDB's server edition (``EMBEDDING_MODE=remote``) speaks a bespoke
embedding protocol, not the OpenAI one, verified from
``external/RetainDB/packages/server/src/engine/inference-client.ts``:

    POST {base}/v1/inference/embeddings   body {"inputs": ["text", ...]}
    ->                                     resp {"embeddings": [[floats...], ...]}
    (row count matches inputs count, all numeric; optional Authorization: Bearer)

vLLM, and every OpenAI-compatible server, instead speaks:

    POST {upstream}/embeddings            body {"model": M, "input": [...]}
    ->                                     resp {"data": [{"index": i, "embedding": [...]}], ...}

This tiny stdlib-only proxy, using only ``http.server`` and ``urllib``,
bridges the two, so RetainDB's remote embeddings are served by the same
embedder the other providers use: the shared ``vllm-embed`` service
(contract v5: ``Alibaba-NLP/gte-modernbert-base``, served name
``gte-modernbert-base``, 768 dimensions padded to 1024 here; contract v4
served bge-small-en-v1.5 at 384). The extraction LLM path is unaffected. It talks
OpenAI-compatible directly to vllm-gen, with no proxy.

The proxy sends ``truncate_prompt_tokens`` (default -1, vLLM's
"truncate to the served model's window") on every upstream call. Without
it, an input over the served window returns a vLLM 400, which this proxy
surfaces as a 502 that RetainDB's client never retries — under contract
v4's 512-token cap that silently starved extraction of over-length
exchanges.

Zero-padding is schema compliance, not a semantic change. RetainDB's schema
hardcodes ``vector(1024)``, from a vendor migration in the pinned,
unmodifiable submodule, while the contract embedder produces 768 dimensions.
Each upstream vector is right-padded with zeros to ``EMBED_PROXY_PAD_DIM``
(default 1024) before it is returned. Zero-padding preserves both the L2
norm and every pairwise dot product, because the appended zeros contribute
nothing to either, so cosine similarity, and therefore RetainDB's ranking,
stays exactly unchanged. This is pure schema compliance, in territory the
deployer controls, since RetainDB's remote-inference protocol deliberately
leaves the model and service choice to the deployer. The proxy errors out
with a non-200 response if an upstream vector is longer than PAD_DIM, since
that signals a misconfigured upstream and truncation would not preserve
norms.

Routes:
  * ``POST /v1/inference/embeddings``  translate {inputs} -> upstream -> pad -> {embeddings}
  * ``GET  /health``                   {"status": "ok"}

Env:
  EMBED_PROXY_PORT              listen port (default 3199)
  EMBED_PROXY_UPSTREAM_BASE_URL OpenAI-compatible base (default http://vllm-embed:8000/v1)
  EMBED_PROXY_UPSTREAM_MODEL    served-model-name (default gte-modernbert-base)
  EMBED_PROXY_UPSTREAM_API_KEY  Bearer for upstream (default local-vllm; vLLM ignores it)
  EMBED_PROXY_TIMEOUT_S         upstream request timeout seconds (default 55)
  EMBED_PROXY_MAX_BATCH         chunk upstream calls above this many inputs (default 128)
  EMBED_PROXY_PAD_DIM           right-pad each vector with zeros to this dim (default 1024)
  EMBED_PROXY_TRUNCATE_TOKENS   truncate_prompt_tokens sent upstream (default -1 =
                                the served window; 0 = send nothing)
  EMBED_PROXY_RETRIES           attempts per upstream chunk on 5xx/unreachable (default 3)

Any upstream or translation failure returns a non-200 JSON body
``{"error": "..."}``, so RetainDB's client surfaces it. With
REMOTE_INFERENCE_REQUIRED=true, this makes the server fail loudly, instead
of silently falling back to an OpenAI path that serves no embeddings on
OpenRouter or vLLM-gen.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("EMBED_PROXY_PORT", "3199"))
UPSTREAM_BASE_URL = os.environ.get(
    "EMBED_PROXY_UPSTREAM_BASE_URL", "http://vllm-embed:8000/v1"
).rstrip("/")
UPSTREAM_MODEL = os.environ.get("EMBED_PROXY_UPSTREAM_MODEL", "gte-modernbert-base")
UPSTREAM_API_KEY = os.environ.get("EMBED_PROXY_UPSTREAM_API_KEY", "local-vllm")
# 120, not 55: a near-window input queues behind concurrent traffic, and 55s
# timed out in a live probe under three-provider load (measured 2026-08-02
# against the interim decoder embedder and kept as headroom).
TIMEOUT_S = float(os.environ.get("EMBED_PROXY_TIMEOUT_S", "120"))
MAX_BATCH = int(os.environ.get("EMBED_PROXY_MAX_BATCH", "128"))
# RetainDB's vector(1024) schema requires 1024-dimension rows, while the
# contract embedder produces 768 dimensions. Right-pad with zeros. This
# preserves the norm and dot product, so cosine similarity and ranking stay
# exactly unchanged; see the module docstring.
PAD_DIM = int(os.environ.get("EMBED_PROXY_PAD_DIM", "1024"))
# -1 = vLLM's "truncate to the served model's max_model_len". A 400 on an
# over-length input becomes a 502 here, and RetainDB's client retries
# nothing, so truncation is the only path that keeps the row. 0 disables
# sending the parameter (for a non-vLLM upstream that rejects it).
TRUNCATE_TOKENS = int(os.environ.get("EMBED_PROXY_TRUNCATE_TOKENS", "-1"))
# RetainDB's client treats any non-200 as terminal for that batch, so
# transient upstream failures (a vllm-embed restart, a socket drop) must be
# absorbed here or the rows are lost.
RETRIES = max(1, int(os.environ.get("EMBED_PROXY_RETRIES", "3")))


def _upstream_embed(inputs):
    """Call the upstream OpenAI /embeddings, chunked to MAX_BATCH, order preserved.

    OpenAI's response ``data[]`` carries a per-row ``index``. We sort by it
    within each chunk, so the returned rows line up with the inputs
    regardless of upstream ordering. Chunks are concatenated in request
    order, so the full result matches the input order one to one (RetainDB
    asserts that row count equals input count).
    """
    out = []
    for start in range(0, len(inputs), MAX_BATCH):
        chunk = inputs[start:start + MAX_BATCH]
        body_obj = {"model": UPSTREAM_MODEL, "input": chunk}
        if TRUNCATE_TOKENS != 0:
            body_obj["truncate_prompt_tokens"] = TRUNCATE_TOKENS
        payload = json.dumps(body_obj).encode("utf-8")
        req = urllib.request.Request(
            f"{UPSTREAM_BASE_URL}/embeddings",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {UPSTREAM_API_KEY}",
            },
        )
        body = None
        for attempt in range(1, RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                # 4xx is deterministic (bad request either way); only 5xx
                # and transport errors are worth another attempt.
                if e.code < 500 or attempt == RETRIES:
                    raise
                time.sleep(attempt)
            except (urllib.error.URLError, TimeoutError):
                # TimeoutError covers socket read timeouts, which urlopen can
                # raise bare (not wrapped in URLError) on Python 3.10+.
                if attempt == RETRIES:
                    raise
                time.sleep(attempt)
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(chunk):
            raise ValueError(
                f"upstream returned {len(data) if isinstance(data, list) else 'no'} "
                f"rows for {len(chunk)} inputs"
            )
        rows = sorted(data, key=lambda d: d.get("index", 0))
        for row in rows:
            emb = row.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("upstream row missing 'embedding' list")
            if len(emb) > PAD_DIM:
                raise ValueError(
                    f"upstream vector dim {len(emb)} exceeds EMBED_PROXY_PAD_DIM {PAD_DIM} "
                    f"(truncation would not preserve norms — misconfigured upstream model?)"
                )
            # Right-pad with zeros to the schema dimension. This preserves
            # the norm and dot product.
            if len(emb) < PAD_DIM:
                emb = emb + [0.0] * (PAD_DIM - len(emb))
            out.append(emb)
    return out


class Handler(BaseHTTPRequestHandler):
    # Quiet the default per-request stderr logging, which is noisy at batch scale.
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    def _send_json(self, status, obj):
        # Use compact separators with no spaces. The entrypoint health-gate
        # greps the literal '"status":"ok"', matching the node server's
        # JSON.stringify output. Python's default json.dumps emits
        # '{"status": "ok"}' with a space after the colon, which never
        # matches that grep and causes a false 60-second health-timeout
        # failure.
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": f"not found: {self.path}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/inference/embeddings":
            self._send_json(404, {"error": f"not found: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
            inputs = req.get("inputs")
            if not isinstance(inputs, list):
                self._send_json(400, {"error": "body must be {\"inputs\": [str, ...]}"})
                return
            if not inputs:
                self._send_json(200, {"embeddings": []})
                return
            embeddings = _upstream_embed([str(t) for t in inputs])
            self._send_json(200, {"embeddings": embeddings})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"upstream HTTP {e.code}: {detail[:500]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"upstream unreachable: {e.reason}"})
        except Exception as e:  # noqa: BLE001 - always answer with a JSON error
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    print(
        f"[embed-proxy] listening on :{PORT} -> upstream {UPSTREAM_BASE_URL} "
        f"model={UPSTREAM_MODEL} pad_dim={PAD_DIM} max_batch={MAX_BATCH} "
        f"timeout={TIMEOUT_S}s truncate_prompt_tokens={TRUNCATE_TOKENS} retries={RETRIES}",
        flush=True,
    )
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
