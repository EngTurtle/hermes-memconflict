"""Minimal Honcho wiring check: spawn, ingest, drain, recall in every mode.

This script answers no benchmark question and calls no answer model. It
proves the parts a Honcho run needs before any tokens are spent on a real
persona:

  * ``import honcho`` binds the SDK, not this provider folder;
  * the embed shim, the run database, the vector-dimension fix, the API, and
    the deriver all come up;
  * the plugin-faithful per-exchange ingest reaches the deriver;
  * ``queue_status()`` drains to zero;
  * every recall mode returns something a benchmark row could carry.

It ingests a two-session temporal-conflict scenario, the same shape as a
MemConflict dynamic question: a fact stated in session 1 and changed in
session 2. It ASSERTS that hybrid recall is non-empty, because an empty
hybrid payload means the deriver produced no peer model, which no later stage
can recover from.

    honcho/run_honcho.sh python honcho/_smoke_honcho_min.py
"""

import os
import sys
import time
import uuid

# sys.path[0] is <repo>/honcho, which holds no `honcho/` subpackage, so this
# resolves the installed SDK. Failing loudly here is the point: if this
# import ever binds the provider folder, every later Honcho call would fail
# in a much less obvious way.
import honcho as _honcho_sdk  # noqa: E402

from _honcho_server import HonchoServer  # noqa: E402
from _local_embed_server import LocalEmbedServer  # noqa: E402
from eval_honcho import (  # noqa: E402
    Add_Session_Dialogue_To_Honcho,
    Drain_Honcho_Queue,
    HonchoRecall,
    OBSERVATION_PRESETS,
    Schedule_Dream_And_Drain,
    sanitize_id,
)
from honcho import Honcho  # noqa: E402

RECALL_MODES = ("hybrid", "base", "dialectic", "conclusions", "search")

# A preference stated, then changed. Hybrid recall must show the change, and
# the conclusions arm must hold both facts as separate rows.
SCENARIO = [
    ("s1", "2024-01-05", [
        {"role": "user", "content": "I just relocated to Melbourne and started an "
                                    "internship at Northern Logistics."},
        {"role": "assistant", "content": "Nice, Melbourne! How is the internship at "
                                         "Northern Logistics going?"},
        {"role": "user", "content": "It is going well. I bike to the office every "
                                    "morning and I only drink oat milk lattes."},
        {"role": "assistant", "content": "A bike commute and an oat milk latte sounds "
                                         "like a good routine."},
    ]),
    ("s2", "2024-03-10", [
        {"role": "user", "content": "Update: I left Melbourne. I now live in Seattle "
                                    "and work as a data scientist at Amazon."},
        {"role": "assistant", "content": "Congratulations on the Seattle move and the "
                                         "data scientist role at Amazon."},
        {"role": "user", "content": "I also stopped drinking oat milk. I switched to "
                                    "black coffee, and I take the light rail now."},
        {"role": "assistant", "content": "Black coffee and the light rail it is."},
    ]),
]

QUESTIONS = [
    "Where does the user currently live and work?",
    "What does the user drink in the morning now?",
    "Describe how the user commutes today and what changed about it since the "
    "start of the year, including any details about the city they live in.",
]


def _preview(text: str, width: int = 220) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[:width] + " ..."


def _conclusion_count(ctx, helper) -> int:
    """Count the conclusions Honcho holds for the observer-to-observed pair.

    A dream consolidates observations into deductive and inductive
    conclusions, so the count RISES when the dreamer runs. The level itself
    is not observable here: v3.0.9's conclusion API response schema omits the
    `level` field (``src/schemas/api.py:435-449``), so the SDK reports
    "explicit" for every row regardless. The level split lives only in the
    server's `documents` table.
    """
    observer, observed = helper.dream_pairs(ctx)[0]
    try:
        page = ctx["client"].peer(observer).conclusions_of(observed).list(size=100)
        return len(list(page.items))
    except Exception as e:
        print(f"[min] conclusion listing failed: {e}", flush=True)
        return -1


def main() -> int:
    print(f"[min] honcho SDK at {_honcho_sdk.__file__}", flush=True)
    if os.path.dirname(os.path.abspath(_honcho_sdk.__file__)) == os.path.dirname(
            os.path.abspath(__file__)):
        raise SystemExit("[min] FAIL: `import honcho` resolved to this provider folder, "
                         "not the SDK. Run this file as `python honcho/_smoke_honcho_min.py`.")

    tag = f"minsmoke_{uuid.uuid4().hex[:6]}"
    os.environ.setdefault("RUN_TAG", tag)
    os.environ.setdefault("HONCHO_PG_CREATE_DB", "1")

    embed = None
    if not os.environ.get("HONCHO_EMBEDDER_BASE_URL"):
        # No vllm-embed on a host smoke: serve bge-small-en-v1.5 locally, at
        # the same 384 dimensions the Docker path uses.
        embed = LocalEmbedServer(port=0)
        os.environ["HONCHO_EMBEDDER_BASE_URL"] = embed.start()
        os.environ.setdefault("HONCHO_EMBEDDER_MODEL", "bge-small-en-v1.5")
        os.environ.setdefault("HONCHO_EMBEDDER_DIMS", "384")

    server = HonchoServer()
    try:
        base_url = server.start()
        print(f"[min] server up at {base_url}; dim_fix={server.dim_fix_report}", flush=True)

        workspace = sanitize_id(f"minsmoke_{uuid.uuid4().hex[:8]}")
        client = Honcho(base_url=base_url, workspace_id=workspace,
                        api_key=os.environ.get("HONCHO_API_KEY", "local"),
                        timeout=float(os.environ.get("HONCHO_TIMEOUT", "300")))
        observation_mode = os.environ.get("HONCHO_OBSERVATION_MODE", "directional")
        ctx = {
            "store_id": workspace,
            "client": client,
            "user_peer_id": "user",
            "ai_peer_id": "hermes",
            "observation": OBSERVATION_PRESETS[observation_mode],
            "current_session_id": None,
        }

        for session_id, date, messages in SCENARIO:
            ctx["current_session_id"] = session_id
            timestamp = None
            add_ms, added, exchanges = Add_Session_Dialogue_To_Honcho(
                ctx, session_id, messages, timestamp,
                message_max_chars=25000, send_created_at=False,
            )
            print(f"[min] {session_id} ({date}): added={added} exchanges={exchanges} "
                  f"add_ms={add_ms:.0f}", flush=True)
            drain = Drain_Honcho_Queue(ctx, timeout_s=1800.0, poll_s=2.0)
            print(f"[min] {session_id} drained in {drain['Drain_Duration_ms'] / 1000:.1f}s "
                  f"(polls={drain['Drain_Polls']}, "
                  f"work_units={drain['Queue_Work_Units']})", flush=True)

        # Dream once, after the last session, and show what it added. The
        # conclusion count for the observer-to-observed pair rises when the
        # dreamer derives new facts from the existing observations.
        dream_helper = HonchoRecall(mode="conclusions", observation_mode=observation_mode)
        before_count = _conclusion_count(ctx, dream_helper)
        dream = Schedule_Dream_And_Drain(
            ctx, dream_helper.dream_pairs(ctx), timeout_s=1800.0, poll_s=2.0)
        print(f"\n[min] dream: requests={dream['Dream_Requests']} "
              f"errors={dream['Dream_Errors']} "
              f"work_units={dream['Dream_Work_Units']} "
              f"({dream['Dream_Duration_ms'] / 1000:.1f}s)", flush=True)
        if dream["Dream_Errors"]:
            print("[min] FAIL: schedule_dream returned an error", flush=True)
            return 1
        after_count = _conclusion_count(ctx, dream_helper)
        observer, observed = dream_helper.dream_pairs(ctx)[0]
        print(f"[min] conclusions {observer} -> {observed}: before={before_count} "
              f"after={after_count} (a rise means the dreamer derived new facts)",
              flush=True)

        failures = []
        for mode in RECALL_MODES:
            helper = HonchoRecall(mode=mode, observation_mode=observation_mode)
            for question in QUESTIONS:
                start = time.time()
                ctx["_raw_recall"] = {}
                items = helper.recall(ctx, question, top_k=5)
                elapsed = (time.time() - start) * 1000.0
                print(f"\n[min] mode={mode} ({elapsed:.0f}ms, {len(items)} items)\n"
                      f"      Q: {question}", flush=True)
                for item in items:
                    print(f"      - [{item.get('source')}] {_preview(item.get('memory', ''))}",
                          flush=True)
                if not items:
                    failures.append((mode, question))

        # Only hybrid is a hard gate. An empty base, dialectic, conclusions,
        # or search result on ONE question is a finding to report, not a
        # wiring fault; an empty hybrid means no peer model exists at all.
        hybrid_failures = [q for m, q in failures if m == "hybrid"]
        if hybrid_failures:
            print(f"\n[min] FAIL: hybrid recall returned nothing for "
                  f"{len(hybrid_failures)} question(s)", flush=True)
            return 1
        if failures:
            print(f"\n[min] NOTE: {len(failures)} empty result(s) outside hybrid: "
                  f"{[m for m, _ in failures]}", flush=True)
        print("\n[min] DONE", flush=True)
        return 0
    finally:
        server.close()
        if embed is not None:
            embed.close()


if __name__ == "__main__":
    sys.exit(main())
