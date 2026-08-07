"""Minimal Hindsight wiring check. It boots the embedded daemon, retains data, and runs a recall."""
import os, sys, time, uuid
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_hindsight import Setup_Hindsight, Search_Hindsight_For_Question

profile = f"smoketest_{uuid.uuid4().hex[:8]}"
print(f"[min] booting embedded Hindsight (profile={profile}) ...", flush=True)
t0 = time.time()
client = Setup_Hindsight(profile)
print(f"[min] daemon up in {time.time()-t0:.1f}s url={getattr(client,'url','?')}", flush=True)

bank = f"bank_{uuid.uuid4().hex[:6]}"
facts = [
    ("user: I moved from Boston to Seattle last month.", datetime(2022, 3, 1, tzinfo=timezone.utc)),
    ("user: Actually I now work as a data scientist at Amazon.", datetime(2022, 4, 1, tzinfo=timezone.utc)),
    ("assistant: Congrats on the new role in Seattle!", datetime(2022, 4, 1, tzinfo=timezone.utc)),
]
for i, (content, ts) in enumerate(facts):
    t = time.time()
    client.retain(bank_id=bank, content=content, timestamp=ts,
                  context="smoke", retain_async=False)
    print(f"[min] retain {i+1}/{len(facts)} ok in {time.time()-t:.1f}s", flush=True)

for q in ["Where does the user live?", "What is the user's job?"]:
    t = time.time()
    retrieved, ms = Search_Hindsight_For_Question(client, bank, q, top_k=5, budget="low", max_tokens=2048)
    print(f"\n[min] Q: {q}  ({ms:.0f}ms, {len(retrieved)} facts)", flush=True)
    for r in retrieved[:5]:
        print(f"   - score={r['score']} [{r['created_at']}] {r['memory']}", flush=True)

client.close()
print("\n[min] DONE", flush=True)
