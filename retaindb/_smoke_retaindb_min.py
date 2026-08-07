"""Minimal RetainDB wiring check: boot local server, ingest a session, recall.

This needs no OpenRouter and no LLM. It exercises only the RetainDB memory
system (ingest plus hybrid retrieval), mirroring _smoke_hindsight_min.py.
Run it to confirm the Node server, the REST client, and the search path all
work end to end, before spending any LLM tokens.

    benchmark/run_retaindb.sh python benchmark/_smoke_retaindb_min.py
    # or, no venv/env needed at all:
    python benchmark/_smoke_retaindb_min.py --embedding_provider local-transformers
"""
import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _retaindb_server import RetainDBServer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_provider", default=os.environ.get("RETAINDB_EMBEDDING_PROVIDER", "hash"),
                    choices=["hash", "local-transformers"])
args = parser.parse_args()

profile = f"smoketest_{uuid.uuid4().hex[:8]}"
print(f"[min] booting RetainDB Local (profile={profile}, embeddings={args.embedding_provider}) ...", flush=True)
t0 = time.time()
server = RetainDBServer(profile, embedding_provider=args.embedding_provider)
client = server.start()
print(f"[min] server up in {time.time()-t0:.1f}s at {server.base_url}", flush=True)

project = f"bank_{uuid.uuid4().hex[:6]}"
user_id = "smoke_user"
# A small temporal-conflict scenario, similar to the MemConflict dynamic questions.
sessions = [
    ("s1", [
        {"role": "user", "content": "I just relocated to Melbourne and started an internship at a legal firm called Northern Logistics."},
        {"role": "assistant", "content": "Nice, Melbourne! How is the internship at Northern Logistics going?"},
    ]),
    ("s2", [
        {"role": "user", "content": "Update: I left Melbourne and now live in Seattle, working as a data scientist at Amazon."},
        {"role": "assistant", "content": "Congrats on the move to Seattle and the data scientist role at Amazon!"},
    ]),
]
for sid, messages in sessions:
    t = time.time()
    r = client.ingest_session(project=project, session_id=sid, messages=messages, user_id=user_id)
    print(f"[min] ingest {sid}: created={r.get('memories_created')} skipped={r.get('skipped')} "
          f"({time.time()-t:.2f}s)", flush=True)

for q in ["Where does the user currently live?", "What is the user's current job?",
          "Did the user's residence change recently?"]:
    t = time.time()
    resp = client.search(project=project, query=q, top_k=5, user_id=user_id)
    results = resp.get("results", [])
    print(f"\n[min] Q: {q}  ({(time.time()-t)*1000:.0f}ms, {len(results)} memories)", flush=True)
    for r in results[:5]:
        sc = r.get("scores", {})
        print(f"   - score={r.get('score')} bm25={sc.get('bm25')} vec={sc.get('vector')} "
              f"graph={sc.get('graph')} | {r.get('content')}", flush=True)

server.close()
print("\n[min] DONE", flush=True)
