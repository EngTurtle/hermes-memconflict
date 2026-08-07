"""Targeted test. It retains specific real MemConflict sessions and runs a recall.

This script isolates fact-extraction behavior for one session, with detailed
error reports.
Usage: python _smoke_retain_one.py [comma-separated session indices, default 5]
This script honors HINDSIGHT_API_* environment variables (for example,
HINDSIGHT_API_LLM_STRICT_SCHEMA=1)."""
import json, os, sys, time, uuid, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hindsight import (Setup_Hindsight, Build_Session_Dialogue_List,
                            Parse_Session_Timestamp, Search_Hindsight_For_Question)

DATA = os.path.join(os.path.dirname(__file__), "..", "external", "MemConflict", "Data", "Step4_4.jsonl")
idxs = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["5"])]
row = json.loads(open(DATA).readline())
chain = row["Full_Session_Chain"]
print(f"strict_schema={os.environ.get('HINDSIGHT_API_LLM_STRICT_SCHEMA')!r} sessions={idxs}", flush=True)

client = Setup_Hindsight(f"retainone_{uuid.uuid4().hex[:8]}")
print(f"daemon up url={getattr(client,'url','?')}", flush=True)
bank = f"b_{uuid.uuid4().hex[:6]}"

for i in idxs:
    sess = chain[i]
    dialogue = Build_Session_Dialogue_List(sess.get("Session_Dialogue", {}))
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in dialogue)
    ts = Parse_Session_Timestamp(sess)
    t = time.time()
    try:
        client.retain(bank_id=bank, content=transcript, timestamp=ts,
                      context=f"session {i}", retain_async=False)
        print(f"[sess {i}] RETAIN OK in {time.time()-t:.1f}s (chars={len(transcript)}, msgs={len(dialogue)})", flush=True)
    except Exception as e:
        print(f"[sess {i}] RETAIN FAILED in {time.time()-t:.1f}s "
              f"type={type(e).__name__} status={getattr(e,'status',None)} "
              f"str={str(e)[:300]!r} body={str(getattr(e,'body',None) or getattr(e,'data',None))[:400]!r}", flush=True)

last = chain[idxs[-1]]
qs = last.get("Session_Questions") or []
if qs:
    q = str(qs[0]["question"])
    retrieved, ms = Search_Hindsight_For_Question(client, bank, q, top_k=5, budget="low", max_tokens=2048)
    print(f"\nQ: {q}\nrecall {ms:.0f}ms, {len(retrieved)} facts:", flush=True)
    for r in retrieved[:5]:
        print(f"  - score={r['score']} [{r['created_at']}] {r['memory'][:130]}", flush=True)
client.close()
print("\nRETAIN-ONE DONE", flush=True)
