"""A FAITHFUL LOCAL MOCK of the self-hosted Supermemory Memory API. This is
a TEST DOUBLE for validating this repo's adapter wiring WITHOUT the vendor
binary.

WHY THIS EXISTS: the real ``supermemory-server`` binary distributes only
from GitHub Releases (supermemoryai/supermemory). This session's GitHub
repo-scope policy blocks that, and a cross-tier ``add_repo`` call is
unsupported, so the live server cannot boot here. This mock implements the
*documented* HTTP contract exactly (endpoints, request fields, response
shapes, the async processing queue, bearer auth, and the first-boot
banner). This lets the ENTIRE integration run unchanged against it:
``_supermemory_server.py`` spawn plus banner key-parse plus readiness,
``SupermemoryClient`` add/drain/search, ``eval_supermemory``
ingest->drain->recall->row-mapping, the shared answer LLM, and the
upstream scorer contract.

WHAT IT VALIDATES: the wiring. WHAT IT DOES NOT VALIDATE: Supermemory's
real memory extraction, hybrid retrieval, or graph quality. Its
"extraction" is a naive per-utterance split, and its "search" is lexical
token overlap, deliberately simple. **Never use output produced against
this mock for any headline number.** The real quality is measured by the
offline run against the vendor binary.

Run standalone (this prints the boot banner, then serves):
    PORT=8899 python supermemory/_mock_supermemory_server.py
Or let ``_supermemory_server.py`` spawn it:
    SUPERMEMORY_SERVER_CMD="python <abs>/_mock_supermemory_server.py" <driver...>
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# How long a document sits in the async queue before it becomes "done"
# and searchable. Kept non-zero so the adapter's drain loop polls at
# least once, exercising it.
_PROCESS_DELAY_S = float(os.environ.get("MOCK_SM_PROCESS_DELAY_S", "0.8"))
_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "sm_mock" + uuid.uuid4().hex)

_LOCK = threading.Lock()
# containerTag -> list[doc]. doc = {id, status, created_ts, content, metadata, memories:[...]}
_STORE = {}
_DOCS = {}  # id -> doc, for /v3/documents/{id}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_WORD_RE.findall(str(text).lower()))


def _extract_memories(content):
    """Naive "extraction": one memory per non-empty utterance line, with the
    role prefix stripped, plus the whole line as a fallback. This mimics a
    document fanning out into several extracted memories, enough to make
    lexical recall non-trivial."""
    mems = []
    for line in str(content).splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading "User:" or "Assistant:" role tag, if present.
        stripped = re.sub(r"^(user|assistant)\s*:\s*", "", line, flags=re.IGNORECASE)
        if stripped:
            mems.append(stripped)
    return mems or [str(content).strip()]


def _advance(doc, now):
    if doc["status"] != "done" and (now - doc["created_ts"]) >= _PROCESS_DELAY_S:
        doc["status"] = "done"


def _advance_all():
    now = time.time()
    with _LOCK:
        for doc in _DOCS.values():
            _advance(doc, now)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def _authed(self):
        # Faithful to the real server: the bearer must be THE key, not just
        # any non-empty string. This exercises the adapter's stale/wrong-key
        # handling (401 causes a loud failure). MOCK_SM_ANY_KEY=1 relaxes
        # this check.
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        if os.environ.get("MOCK_SM_ANY_KEY") == "1":
            return True
        return auth[len("Bearer "):].strip() == _API_KEY

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            return {}

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if not self._authed():
            return self._send(401, {"error": "missing bearer"})
        if path == "/v3/documents/processing":
            _advance_all()
            with _LOCK:
                pending = [
                    {"id": d["id"], "status": d["status"],
                     "created_at": d["created_at"], "updated_at": d["created_at"],
                     "container_tags": [d["container_tag"]], "metadata": d["metadata"]}
                    for d in _DOCS.values() if d["status"] not in ("done", "failed")
                ]
            return self._send(200, {"documents": pending, "total": len(pending)})
        m = re.match(r"^/v3/documents/([^/]+)$", path)
        if m:
            _advance_all()
            with _LOCK:
                doc = _DOCS.get(m.group(1))
            if not doc:
                return self._send(404, {"error": "not found"})
            return self._send(200, {
                "id": doc["id"], "status": doc["status"], "content": doc["content"],
                "container_tags": [doc["container_tag"]], "metadata": doc["metadata"],
                "created_at": doc["created_at"], "updated_at": doc["created_at"],
            })
        return self._send(404, {"error": "unknown route"})

    def do_POST(self):
        path = urlparse(self.path).path
        if not self._authed():
            return self._send(401, {"error": "missing bearer"})
        payload = self._read_json()
        if path == "/v3/documents":
            return self._add_document(payload)
        if path == "/v4/search":
            return self._search_memories(payload)
        if path == "/v3/search":
            return self._search_documents(payload)
        return self._send(404, {"error": "unknown route"})

    # -- handlers ----------------------------------------------------------
    def _add_document(self, payload):
        content = payload.get("content", "")
        tag = payload.get("containerTag") or (payload.get("containerTags") or ["sm_project_default"])[0]
        metadata = payload.get("metadata") or {}
        doc_id = "doc_" + uuid.uuid4().hex[:16]
        created_at = metadata.get("session_date") or "2024-01-01T00:00:00+00:00"
        doc = {
            "id": doc_id, "status": "queued", "created_ts": time.time(),
            "content": content, "metadata": metadata, "container_tag": tag,
            "created_at": created_at, "memories": _extract_memories(content),
        }
        with _LOCK:
            _STORE.setdefault(tag, []).append(doc)
            _DOCS[doc_id] = doc
        return self._send(200, {"id": doc_id, "status": "queued"})

    def _gather_memories(self, tag):
        """Return all DONE memories for a container tag, as (text, updatedAt,
        metadata, doc) tuples."""
        out = []
        with _LOCK:
            for doc in _STORE.get(tag, []):
                if doc["status"] != "done":
                    continue
                for mem in doc["memories"]:
                    out.append((mem, doc["created_at"], doc["metadata"], doc))
        return out

    def _search_memories(self, payload):
        q = payload.get("q", "")
        tag = payload.get("containerTag") or (payload.get("containerTags") or ["sm_project_default"])[0]
        limit = int(payload.get("limit", 10) or 10)
        threshold = payload.get("threshold")
        mode = payload.get("searchMode", "memories")
        qtok = _tokens(q)
        scored = []
        for mem, updated_at, metadata, doc in self._gather_memories(tag):
            mtok = _tokens(mem)
            if not mtok:
                continue
            overlap = len(qtok & mtok)
            sim = overlap / (len(qtok | mtok) or 1)  # Jaccard, deterministic
            scored.append((sim, mem, updated_at, metadata))
        scored.sort(key=lambda x: x[0], reverse=True)
        if threshold is not None:
            scored = [s for s in scored if s[0] >= float(threshold)]
        results = []
        for sim, mem, updated_at, metadata in scored[:limit]:
            results.append({
                "id": "mem_" + uuid.uuid4().hex[:12], "memory": mem,
                "similarity": round(sim, 4), "metadata": metadata,
                "updatedAt": updated_at, "version": 1,
            })
        # Hybrid mode: if memory hits are thin, append a 'chunk' fallback.
        # This exercises the adapter's memory-vs-chunk handling.
        if mode == "hybrid" and len(results) < limit:
            with _LOCK:
                docs = list(_STORE.get(tag, []))
            for doc in docs:
                if doc["status"] != "done":
                    continue
                ctok = _tokens(doc["content"])
                sim = len(qtok & ctok) / (len(qtok | ctok) or 1)
                if sim <= 0:
                    continue
                results.append({
                    "id": "chunk_" + uuid.uuid4().hex[:12], "chunk": doc["content"],
                    "similarity": round(sim, 4), "metadata": doc["metadata"],
                    "updatedAt": doc["created_at"], "version": 1,
                })
                if len(results) >= limit:
                    break
        return self._send(200, {"results": results, "total": len(results), "timing": 3})

    def _search_documents(self, payload):
        q = payload.get("q", "")
        tags = payload.get("containerTags") or ([payload["containerTag"]] if payload.get("containerTag") else [])
        limit = int(payload.get("limit", 10) or 10)
        qtok = _tokens(q)
        with _LOCK:
            docs = [d for t in tags for d in _STORE.get(t, []) if d["status"] == "done"]
        scored = []
        for doc in docs:
            ctok = _tokens(doc["content"])
            sim = len(qtok & ctok) / (len(qtok | ctok) or 1)
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, doc in scored[:limit]:
            results.append({
                "documentId": doc["id"], "title": None, "type": "text",
                "score": round(sim, 4),
                "chunks": [{"content": doc["content"], "score": round(sim, 4), "isRelevant": sim > 0.1}],
                "metadata": doc["metadata"],
                "createdAt": doc["created_at"], "updatedAt": doc["created_at"],
            })
        return self._send(200, {"results": results, "total": len(results), "timing": 3})


def main():
    port = int(os.environ.get("PORT") or os.environ.get("SUPERMEMORY_PORT") or 8787)
    # Boot banner in the documented format, so this exercises
    # _supermemory_server.py's key-parse and readiness path exactly as it
    # would run against the real binary.
    print("  ┌──────────────────────────────────────────────────┐", flush=True)
    print(f"  │  url       http://127.0.0.1:{port}", flush=True)
    print(f"  │  database  ./.supermemory", flush=True)
    print(f"  │  api key   {_API_KEY}", flush=True)
    print(f"  │  org id    org_{uuid.uuid4().hex[:16]}", flush=True)
    print("  └──────────────────────────────────────────────────┘", flush=True)
    print(f"[mock-supermemory] listening on :{port} (process_delay={_PROCESS_DELAY_S}s) "
          f"— TEST DOUBLE, not the real server", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
