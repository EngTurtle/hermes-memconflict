"""Minimal mem0 wiring check: init, ingest (raw, no LLM), recall.

This script needs no OpenRouter and no internal LLM. It runs mem0 with
``infer=False``, so the add path stores the raw turns WITHOUT the
extraction/update-memory LLM calls. It exercises only the embedder and
vector store, the parts that must be installed locally. It mirrors
_smoke_retaindb_min.py: confirm the mem0 SDK, the local HuggingFace
embedder, and the embedded vector-store search all work end-to-end before
spending any LLM tokens.

    mem0/run_mem0.sh python mem0/_smoke_mem0_min.py
    # or, with no venv/env needed at all (uses the local HF embedder):
    python mem0/_smoke_mem0_min.py
"""
import argparse
import os
import sys
import tempfile
import time
import uuid

os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mem0 import Memory  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--embedder_provider", default=os.environ.get("MEM0_EMBEDDER_PROVIDER", "huggingface"))
parser.add_argument("--embedder_model", default=os.environ.get("MEM0_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
parser.add_argument("--embedder_dims", type=int, default=int(os.environ.get("MEM0_EMBEDDER_DIMS", "384")))
args = parser.parse_args()

tmp = tempfile.mkdtemp(prefix="mem0_smoke_")
embedder_cfg = {"model": args.embedder_model, "embedding_dims": args.embedder_dims}
if os.environ.get("MEM0_EMBEDDER_BASE_URL"):
    embedder_cfg["openai_base_url"] = os.environ["MEM0_EMBEDDER_BASE_URL"]

config = {
    # The LLM is constructed but never CALLED, because infer=False below.
    # A dummy key keeps the OpenAI client constructor happy with no network use.
    "llm": {"provider": "openai", "config": {"model": "unused", "api_key": "sk-unused"}},
    "embedder": {"provider": args.embedder_provider, "config": embedder_cfg},
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": "mem0smoke",
        "path": os.path.join(tmp, "qd"),
        "embedding_model_dims": args.embedder_dims,
        "on_disk": False,
    }},
    "version": "v1.1",
}

print(f"[min] init mem0 (embedder={args.embedder_provider}/{args.embedder_model}, dims={args.embedder_dims}) ...", flush=True)
t0 = time.time()
memory = Memory.from_config(config)
print(f"[min] Memory up in {time.time()-t0:.1f}s (store={tmp})", flush=True)

user_id = f"smoke_{uuid.uuid4().hex[:6]}"
# This is a tiny temporal-conflict scenario, like the MemConflict dynamic questions.
sessions = [
    ("s1", [
        {"role": "user", "content": "I just relocated to Melbourne and started an internship at Northern Logistics."},
        {"role": "assistant", "content": "Nice, Melbourne! How is the internship at Northern Logistics going?"},
    ], "2024-01-05"),
    ("s2", [
        {"role": "user", "content": "Update: I left Melbourne and now live in Seattle, working as a data scientist at Amazon."},
        {"role": "assistant", "content": "Congrats on the move to Seattle and the data scientist role at Amazon!"},
    ], "2024-03-10"),
]
for sid, messages, date in sessions:
    t = time.time()
    # infer=False stores raw turns, with no extraction/update LLM calls.
    # Never pass timestamp= or reference_date=: they are managed-platform
    # only and raise in OSS.
    # 2.x DOES honor metadata.created_at as the stored payload's created_at.
    r = memory.add(messages, user_id=user_id,
                   metadata={"timestamp": date, "created_at": date, "session_id": sid},
                   infer=False)
    n = len(r.get("results", [])) if isinstance(r, dict) else len(r or [])
    print(f"[min] add {sid}: stored={n} ({time.time()-t:.2f}s)", flush=True)

for q in ["Where does the user currently live?", "What is the user's current job?",
          "Did the user's residence change recently?"]:
    t = time.time()
    # mem0ai 2.x uses a keyword-only signature. `limit` is gone, replaced
    # by `top_k`. A top-level `user_id=` argument raises, so the tenant ID
    # must travel inside `filters`.
    resp = memory.search(query=q, filters={"user_id": user_id}, top_k=5, threshold=0.0)
    results = resp.get("results", []) if isinstance(resp, dict) else (resp or [])
    print(f"\n[min] Q: {q}  ({(time.time()-t)*1000:.0f}ms, {len(results)} memories)", flush=True)
    for r in results[:5]:
        print(f"   - score={r.get('score')} | [{(r.get('metadata') or {}).get('timestamp')}] {r.get('memory')}", flush=True)

print("\n[min] DONE", flush=True)
