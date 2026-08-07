"""Minimal Supermemory wiring check: boot the server, ingest a session,
drain, and recall.

Unlike RetainDB's LLM-free smoke, Supermemory has an INTERNAL extraction
LLM, so ingestion spends LLM calls. This smoke therefore needs an
extraction model configured (``OPENAI_*`` or ``SUPERMEMORY_LLM_*``; for the
offline run, point it at a local OpenAI-compatible endpoint, and for
validation, at OpenRouter gpt-oss-120b). It does NOT touch the harness
answer/judge LLM. It only exercises the memory system (ingest -> async
drain -> hybrid retrieval), mirroring _smoke_retaindb_min.py.

    supermemory/run_supermemory.sh python supermemory/_smoke_supermemory_min.py
"""
import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _supermemory_server import SupermemoryServer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--search_mode", default=os.environ.get("SUPERMEMORY_SEARCH_MODE", "hybrid"),
                    choices=["hybrid", "memories"])
parser.add_argument("--embedding_provider", default=os.environ.get("SUPERMEMORY_EMBEDDING_PROVIDER", "local"))
parser.add_argument("--drain_timeout", type=float, default=float(os.environ.get("SUPERMEMORY_DRAIN_TIMEOUT", "300")))
parser.add_argument("--base_url", default=os.environ.get("SUPERMEMORY_BASE_URL"))
parser.add_argument("--api_key", default=os.environ.get("SUPERMEMORY_API_KEY"))
args = parser.parse_args()

run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       ".supermemory_runs", f"smoke_{uuid.uuid4().hex[:8]}")

server = None
if args.base_url:
    from _supermemory_server import SupermemoryClient
    if not args.api_key:
        raise SystemExit("--base_url set but no --api_key/SUPERMEMORY_API_KEY")
    print(f"[min] attaching to {args.base_url}", flush=True)
    client = SupermemoryClient(args.base_url, args.api_key)
    client.ping()
else:
    print(f"[min] booting supermemory server (data_dir={run_dir}, "
          f"embeddings={args.embedding_provider}) ...", flush=True)
    t0 = time.time()
    server = SupermemoryServer(data_dir=run_dir, embedding_provider=args.embedding_provider)
    client = server.start()
    print(f"[min] server up in {time.time()-t0:.1f}s at {server.base_url}", flush=True)

tag = f"user_{uuid.uuid4().hex[:8]}"
# A tiny temporal-conflict scenario, like the MemConflict dynamic questions.
sessions = [
    ("s1", "User: I just relocated to Melbourne and started an internship at a legal firm "
           "called Northern Logistics.\nAssistant: Nice, Melbourne! How is the internship going?"),
    ("s2", "User: Update: I left Melbourne and now live in Seattle, working as a data scientist "
           "at Amazon.\nAssistant: Congrats on the move to Seattle and the data scientist role!"),
]
try:
    for sid, content in sessions:
        t = time.time()
        add = client.add_document(content=content, container_tag=tag,
                                  metadata={"session_id": sid, "session_date": "2024-01-01"})
        did = add.get("id")
        drain = client.wait_for_drain([did] if did else None, timeout=args.drain_timeout)
        print(f"[min] ingest {sid}: id={did} status={add.get('status')} "
              f"drained={drain.get('drained')} drain_s={drain.get('elapsed_s')} "
              f"failed={len(drain.get('failed', []))} ({time.time()-t:.2f}s)", flush=True)

    for q in ["Where does the user currently live?", "What is the user's current job?",
              "Did the user's residence change recently?"]:
        t = time.time()
        resp = client.search_memories(q, tag, limit=5, search_mode=args.search_mode)
        results = resp.get("results", [])
        print(f"\n[min] Q: {q}  ({(time.time()-t)*1000:.0f}ms, {len(results)} results)", flush=True)
        for r in results[:5]:
            text = r.get("memory") or r.get("chunk")
            kind = "memory" if r.get("memory") is not None else "chunk"
            print(f"   - sim={r.get('similarity')} [{kind}] {text}", flush=True)
finally:
    if server is not None:
        server.close()
print("\n[min] DONE", flush=True)
