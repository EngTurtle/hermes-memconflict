"""Pull representative qualitative examples from scored Mnemosyne results.

For each conflict type, this shows one example where Mnemosyne answered
correctly (AA at least 0.5, with the gold evidence retrieved) and one where it
failed. The report can then show the cause: retrieval competition or answer
synthesis.
"""

import argparse
import json
import os
from typing import Any, Dict, List

CT_AA = {
    "dynamic_conflict": "dynamic_answer_accuracy",
    "static_conflict": "static_answer_accuracy",
    "conditional_conflict": "conditional_answer_accuracy",
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_file", required=True,
                    help="Provider Scores/<...>_eval_scores.jsonl to pull examples from.")
    ap.add_argument("--results_file", required=True,
                    help="Matching provider Results/<...>_results.jsonl; used to recover the "
                         "retrieved-memory text for each question.")
    ap.add_argument("--per_bucket", type=int, default=1)
    args = ap.parse_args()

    # Maps question_id to retrieved memories, read from the results file.
    retr: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(args.results_file):
        for p in load_jsonl(args.results_file):
            for s in p.get("Full_Session_Chain", []):
                for q in s.get("Session_Questions", []):
                    retr[str(q.get("question_id")) + "|" + str(q.get("question"))] = q.get("Retrieved_Memories", [])

    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {ct: {"hit": [], "miss": []} for ct in CT_AA}
    for p in load_jsonl(args.scores_file):
        for s in p.get("Full_Session_Chain", []):
            for q in s.get("Session_Questions", []):
                ct = q.get("conflict_type")
                if ct not in CT_AA:
                    continue
                m = q.get("Evaluation_Result", {}).get("Metrics", {})
                aa = float(m.get(CT_AA[ct], 0) or 0)
                bucket = "hit" if aa >= 0.5 else "miss"
                rec = {
                    "question": q.get("question"),
                    "gold": q.get("answer"),
                    "model_answer": q.get("Model_Answer"),
                    "aa": aa,
                    "retrieved": [str(r.get("memory", ""))[:120] for r in
                                  retr.get(str(q.get("question_id")) + "|" + str(q.get("question")), [])[:3]],
                }
                if len(buckets[ct][bucket]) < args.per_bucket:
                    buckets[ct][bucket].append(rec)

    for ct in CT_AA:
        print(f"\n{'='*70}\n{ct}\n{'='*70}")
        for bucket in ("hit", "miss"):
            for rec in buckets[ct][bucket]:
                print(f"\n[{bucket.upper()}] AA={rec['aa']}")
                print(f"  Q:     {rec['question']}")
                print(f"  Gold:  {rec['gold']}")
                print(f"  Model: {rec['model_answer']}")
                print("  Top retrieved:")
                for r in rec["retrieved"]:
                    print(f"    - {r}")


if __name__ == "__main__":
    main()
