"""Aggregate per-persona MemConflict scores into headline metrics.

Every number here derives from fields that the **upstream** scorer
(`external/MemConflict/Evaluation/eval_scoring.py`) wrote into the score
file. This module does no judging and no re-derivation of per-question
metrics. It only aggregates. We verified the fidelity rules below
line-by-line against upstream on 2026-07-21. Keep the citations current if
you touch this file, because a silent aggregation drift makes our
published numbers incomparable to the benchmark we claim to run.

Upstream facts this file is bound to
-----------------------------------
* **Per-question metric derivation** is upstream's, not ours. The judge
  returns a 1-based `*_first_support_rank` (0 means "no supporting evidence
  in Top-K"), and `derive_white_box_metrics_from_rank`
  (eval_scoring.py:552-568) turns it into `hit@k = 1.0 iff 1 <= rank <= k`
  and `log_rank@k = 1/log2(rank+1)`, else both 0.0. `parse_support_rank`
  (eval_scoring.py:542-549) already clamps out-of-range or non-integer
  ranks to 0. So this file reads the stored `hit_at_*` and
  `log_rank_score_at_*` values verbatim and never re-derives them from a
  rank, which would duplicate, and risk diverging from, upstream logic.
* **Denominators are "all questions of that conflict type"**, including
  questions the judge could not score. `accumulate_question_metrics`
  (eval_scoring.py:851-857) bumps `Question_Count` for every question whose
  `conflict_type` is known, then reads `float(metrics.get(key, 0) or 0)`,
  and `build_conflict_metric_summary` (eval_scoring.py:860-869) divides by
  that same count for every metric. So `missing_answer` and `rule_based`
  rows count as zeros, or as their rule-based partial credit, and are never
  excluded. Log-rank is emphatically not averaged over hits only. This file
  mirrors both choices exactly: `counts[ct]` increments before any metric
  is read.
* **Cross-persona pooling is question-weighted.** Upstream's own reporting
  helper `diagnose_failures.Weighted_Summary_Metric`
  (diagnose_failures.py:52-70) recombines per-persona summaries as
  `sum(metric_p * n_p) / sum(n_p)`, which is algebraically identical to
  pooling raw questions across personas, the approach this file takes.
  We verified this numerically to 1e-9, on the full 30-persona baseline and
  on a file containing 28 `missing_answer` rows.
* **Partial credit** (1.0, 0.5, or 0.0) exists only for
  `dynamic_answer_accuracy` and `static_answer_accuracy`
  (`PARTIAL_CREDIT_BLACK_BOX_METRICS`, eval_scoring.py:522-539). The scorer
  binarizes `conditional_answer_accuracy`, UOCS, and CRS. This file sums
  whatever is stored, so a 0.5 score propagates identically into AA, macro
  or micro AA, and EUG.
* **UOCS and CRS** are ordinary black-box metrics in upstream's schema
  (eval_scoring.py:41-74) with the same all-questions-of-that-type
  denominator, not "of the questions answered correctly". This file uses
  the same denominator.

Aggregations we add on top (upstream ships no cross-persona aggregator)
----------------------------------------------------------------------
  * **Micro** (question-weighted): total credit divided by total questions.
    The huge Dynamic category (2,946 of 3,750 questions) dominates this
    metric.
  * **Macro** (unweighted mean of the three conflict-type scores): the
    aggregation MemConflict uses for its published system comparison, so
    the category imbalance does not swamp the headline. Upstream's only
    macro-style combination in code (`Average_EUG`, diagnose_failures.py:95)
    is likewise an unweighted mean over the three types. The conventions
    differ: upstream always divides by 3, and this file divides by the
    number of conflict types actually present. On a full run, all three
    types are present, so the two agree exactly. We verified that our macro
    `EUG_gap@3` reproduces upstream `Average_EUG` to 0 ulp on all 20
    committed multi-type score files. On a partial or smoke file, upstream's
    fixed divide-by-3 would silently report a mean one-third of the real
    score. For example, `hindsight_armB_smoke2` (7 questions, dynamic only)
    gives upstream 0.0714 versus our 0.2143, the latter being the actual
    dynamic gap. This is a documented, deliberate deviation. It cannot
    affect any full-run number, and every headline in the reports comes
    from a full run.

Two EUG columns, and why
------------------------
Upstream's README lists EUG as "whether retrieved gold memories are
converted into correct answers". The only implementation of it in the repo
defines it as a **gap between rates**: `EUG = SEH@3 - AA`, per conflict
type, plus an unweighted 3-type mean `Average_EUG` (diagnose_failures.py:
73-96). `EUG_gap@3` reports exactly that, computed from those two upstream
metrics.

`EUG@5` (kept under its original key because reports and prior summaries
cite it) is **our own** conditional statistic: mean AA restricted to
questions whose gold evidence reached rank <= 5. It answers "when the
answerer did see the gold, how often did it use it", a question the gap
form cannot answer, since a gap of 0.06 is consistent with wildly different
utilization rates. Two notes: it is conditional, so its denominator is only
the hit@5 subset, and because AA carries partial credit, a 0.5 answer
contributes 0.5, so it is a mean AA, not a "fraction fully correct". It is
strictly additive: no upstream-comparable metric is computed from it.

SEH@5 and the per-K denominator
-------------------------------
The judge scores each question once at Top-5 (`MAX_WHITE_BOX_TOP_K`,
eval_scoring.py:1023-1037), and upstream then re-derives the white-box
metrics at k=2/3/5 into `Evaluation_Result.White_Box_By_K`. SEH@3 and
log-rank@3 come from the primary `Metrics` block (k=3, upstream's headline
K). SEH@5 comes from `White_Box_By_K["5"]`. Upstream's k-wise aggregator
(`build_white_box_summary_by_k`, eval_scoring.py:930-948) skips a question
entirely when that block is absent, rather than counting it as a miss, so
the @5 denominator can legitimately differ from the @3 one. This file
reproduces that skip with a separate `counts_k5`. On every score file
committed to this repo, the two denominators are equal, since no row is
missing the block, so this is a robustness guard, not a number change.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

# These are per-conflict-type metric key names in the scored output. These
# strings must exactly match upstream's METRIC_SCHEMAS and
# WHITE_BOX_METRIC_CONFIGS (eval_scoring.py:41-95), including the
# conflict_type spellings. Upstream treats these as a closed set and raises
# on an unknown one (eval_scoring.py:1019-1021). A typo here would silently
# drop a whole category instead of raising an error. So this file counts
# and reports any row it cannot route, rather than ignoring it (see
# `unroutable`).
CONFLICTS = {
    "dynamic_conflict": {
        "name": "Dynamic",
        "aa": "dynamic_answer_accuracy",
        "diag": ("update_awareness_and_order_consistency_score", "UOCS"),
        "hit3": "updated_evidence_hit_at_3", "hit5": "updated_evidence_hit_at_5",
        "logrank3": "updated_evidence_log_rank_score_at_3",
    },
    "static_conflict": {
        "name": "Static",
        "aa": "static_answer_accuracy",
        "diag": ("conflict_recognition_score", "CRS"),
        "hit3": "truth_evidence_hit_at_3", "hit5": "truth_evidence_hit_at_5",
        "logrank3": "truth_evidence_log_rank_score_at_3",
    },
    "conditional_conflict": {
        "name": "Conditional",
        "aa": "conditional_answer_accuracy",
        "diag": (None, None),
        "hit3": "correct_condition_evidence_hit_at_3", "hit5": "correct_condition_evidence_hit_at_5",
        "logrank3": "correct_condition_evidence_log_rank_score_at_3",
    },
}
ORDER = ["dynamic_conflict", "static_conflict", "conditional_conflict"]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--scores_file", default=os.path.join(here, "Scores", "mnemosyne_eval_scores.jsonl"))
    ap.add_argument("--system", default="mnemosyne")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--track", default=None,
                    choices=["capacity-ceiling", "native-automation", "plugin-faithful"],
                    help="Provenance label: how the arm's automation compares across providers.")
    ap.add_argument("--lifecycle_provenance", default=None,
                    choices=["stock", "configured-stock", "custom-adapter", "oracle"],
                    help="Provenance label: where the arm's memory-lifecycle logic comes from.")
    args = ap.parse_args()

    # Accumulators, per conflict type.
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)
    # This is a separate @5 denominator. Upstream's
    # build_white_box_summary_by_k skips a question that carries no
    # White_Box_By_K["5"] block, instead of scoring it 0.
    counts_k5: Dict[str, int] = defaultdict(int)
    # EUG@5 accumulators, over questions whose gold was in the answerer's top-5.
    eug_num: Dict[str, float] = defaultdict(float)   # sum of AA where hit5==1
    eug_den: Dict[str, int] = defaultdict(int)       # count where hit5==1
    judge_methods: Dict[str, int] = defaultdict(int)
    # These are audit counters: total question rows seen versus rows this
    # file could route. Upstream raises on an unknown conflict_type, so a
    # nonzero `unroutable` means the file was not produced by the standard
    # scorer. Surface it, never swallow it.
    total_rows = 0
    unroutable: Dict[str, int] = defaultdict(int)

    for persona in load_jsonl(args.scores_file):
        for session in persona.get("Full_Session_Chain", []):
            for q in session.get("Session_Questions", []):
                total_rows += 1
                ct = q.get("conflict_type")
                if ct not in CONFLICTS:
                    unroutable[str(ct)] += 1
                    continue
                cfg = CONFLICTS[ct]
                ev = q.get("Evaluation_Result", {})
                m = ev.get("Metrics", {})
                by_k = ev.get("White_Box_By_K", {})
                m5 = by_k.get("5", {}).get("Metrics", {})
                counts[ct] += 1
                judge_methods[ev.get("Judge_Method", "unknown")] += 1

                # `.get(key, 0) or 0` mirrors upstream
                # accumulate_question_metrics verbatim: an absent metric key
                # contributes 0 to the numerator, while still counting in
                # the denominator.
                aa = float(m.get(cfg["aa"], 0) or 0)
                hit3 = float(m.get(cfg["hit3"], 0) or 0)
                logrank3 = float(m.get(cfg["logrank3"], 0) or 0)
                sums[ct]["aa"] += aa
                sums[ct]["hit3"] += hit3
                sums[ct]["logrank3"] += logrank3
                if cfg["diag"][0]:
                    sums[ct]["diag"] += float(m.get(cfg["diag"][0], 0) or 0)
                if isinstance(by_k.get("5"), dict):
                    hit5 = float(m5.get(cfg["hit5"], 0) or 0)
                    counts_k5[ct] += 1
                    sums[ct]["hit5"] += hit5
                    if hit5 >= 1.0:
                        eug_num[ct] += aa
                        eug_den[ct] += 1

    def cat_avg(ct: str, key: str) -> float:
        return sums[ct][key] / counts[ct] if counts[ct] else 0.0

    def cat_avg_k5(ct: str) -> float:
        return sums[ct]["hit5"] / counts_k5[ct] if counts_k5[ct] else 0.0

    def eug(ct: str) -> float:
        return eug_num[ct] / eug_den[ct] if eug_den[ct] else 0.0

    def eug_gap(ct: str) -> float:
        """Upstream EUG: SEH@3 - AA (diagnose_failures.py:83-90).

        Both rates use the same K and the same denominators, so they are
        directly subtractable.
        """
        return cat_avg(ct, "hit3") - cat_avg(ct, "aa")

    present = [ct for ct in ORDER if counts[ct]]
    total_q = sum(counts.values())
    total_q5 = sum(counts_k5[ct] for ct in present)

    # Micro = question-weighted; Macro = unweighted mean of the per-type scores.
    def micro(key: str) -> float:
        return sum(sums[ct][key] for ct in present) / total_q if total_q else 0.0

    def macro(key: str) -> float:
        return sum(cat_avg(ct, key) for ct in present) / len(present) if present else 0.0

    micro_hit5 = sum(sums[ct]["hit5"] for ct in present) / total_q5 if total_q5 else 0.0
    macro_hit5 = sum(cat_avg_k5(ct) for ct in present) / len(present) if present else 0.0
    micro_eug = (sum(eug_num[ct] for ct in present) / sum(eug_den[ct] for ct in present)
                 if sum(eug_den[ct] for ct in present) else 0.0)
    macro_eug = sum(eug(ct) for ct in present) / len(present) if present else 0.0
    micro_eug_gap = micro("hit3") - micro("aa")
    # This is upstream's Average_EUG (diagnose_failures.py:95): an unweighted
    # 3-type mean of the per-type gaps, over the types actually present (see
    # the module docstring).
    macro_eug_gap = sum(eug_gap(ct) for ct in present) / len(present) if present else 0.0

    L = []
    L.append(f"# MemConflict results — {args.system}")
    L.append("")
    if args.track or args.lifecycle_provenance:
        L.append(f"Track: **{args.track or '—'}**  |  Lifecycle provenance: **{args.lifecycle_provenance or '—'}**")
        L.append("")
    L.append(f"Scored questions: **{total_q}**  |  Judge methods: {dict(judge_methods)}")
    if unroutable:
        L.append("")
        L.append(f"> **WARNING — {sum(unroutable.values())} of {total_rows} question rows had an "
                 f"unrecognised `conflict_type` and were excluded: {dict(unroutable)}. "
                 f"The upstream scorer raises on unknown types, so this file was not produced by it.**")
    L.append("")
    L.append("## Per conflict type")
    L.append("")
    L.append("| Conflict type | N | AA | SEH@3 | SEH@5 | Log-rank@3 | Diagnostic | EUG-gap@3 | EUG-cond@5 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ct in present:
        cfg = CONFLICTS[ct]
        diag = f"{cat_avg(ct,'diag'):.3f} ({cfg['diag'][1]})" if cfg["diag"][1] else "—"
        L.append(f"| {cfg['name']} | {counts[ct]} | {cat_avg(ct,'aa'):.3f} | "
                 f"{cat_avg(ct,'hit3'):.3f} | {cat_avg_k5(ct):.3f} | {cat_avg(ct,'logrank3'):.3f} | "
                 f"{diag} | {eug_gap(ct):+.3f} | {eug(ct):.3f} |")
    L.append("")
    L.append("## Overall")
    L.append("")
    L.append("| Aggregation | AA | SEH@3 | SEH@5 | Log-rank@3 | EUG-gap@3 | EUG-cond@5 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    L.append(f"| Micro (question-weighted) | {micro('aa'):.3f} | {micro('hit3'):.3f} | "
             f"{micro_hit5:.3f} | {micro('logrank3'):.3f} | {micro_eug_gap:+.3f} | {micro_eug:.3f} |")
    L.append(f"| **Macro (MemConflict protocol)** | **{macro('aa'):.3f}** | **{macro('hit3'):.3f}** | "
             f"{macro_hit5:.3f} | {macro('logrank3'):.3f} | **{macro_eug_gap:+.3f}** | {macro_eug:.3f} |")
    L.append("")
    L.append("AA = Answer Accuracy (1.0/0.5/0.0 partial credit for dynamic & static; binary for "
             "conditional). SEH@K = judge-assessed supporting evidence in top-K (semantic, not "
             "literal gold matching -- the judge is asked whether a retrieved item semantically "
             "supports the reference answer, not whether it is the dataset's exact gold turn). "
             "Log-rank@3 = 1/log2(rank+1), 0 when no retrieved item semantically supports the "
             "answer within the top-3, averaged over ALL questions. "
             "UOCS/CRS = per-conflict diagnostics. **Macro** (unweighted type mean) is MemConflict's "
             "headline aggregation; **micro** is question-weighted and dominated by the Dynamic "
             "category. **EUG-gap@3** = SEH@3 - AA, MemConflict's own utilization-gap definition "
             "(`diagnose_failures.py`); its macro value equals upstream's `Average_EUG` whenever all "
             "three conflict types are present, i.e. on every full run. **EUG-cond@5** "
             "is this repo's extra diagnostic — mean AA restricted to questions whose gold reached "
             "rank <= 5 — and is NOT the upstream EUG; do not compare it to published EUG numbers.")
    report = "\n".join(L)
    print(report)
    if unroutable:
        print(f"[summarize] WARNING: {sum(unroutable.values())} unroutable rows: {dict(unroutable)}",
              file=sys.stderr)

    if args.out_json:
        summary = {
            "system": args.system,
            "scored_questions": total_q,
            # These are audit fields: question rows present in the file
            # versus rows this file could route. These must be equal.
            # Anything else means categories were dropped, and every metric
            # below covers an incomplete set.
            "question_rows_in_file": total_rows,
            "unroutable_rows": dict(unroutable),
            "judge_methods": dict(judge_methods),
            **({"track": args.track} if args.track else {}),
            **({"lifecycle_provenance": args.lifecycle_provenance} if args.lifecycle_provenance else {}),
            "overall": {
                "micro": {"AA": micro("aa"), "SEH@3": micro("hit3"), "SEH@5": micro_hit5,
                          "log_rank@3": micro("logrank3"),
                          "EUG_gap@3": micro_eug_gap, "EUG@5": micro_eug},
                "macro_memconflict_protocol": {
                    "AA": macro("aa"), "SEH@3": macro("hit3"), "SEH@5": macro_hit5,
                    "log_rank@3": macro("logrank3"),
                    # This is upstream diagnose_failures.Average_EUG.
                    "EUG_gap@3": macro_eug_gap, "EUG@5": macro_eug},
            },
            "by_conflict_type": {
                ct: {
                    "N": counts[ct], "AA": cat_avg(ct, "aa"),
                    "SEH@3": cat_avg(ct, "hit3"), "SEH@5": cat_avg_k5(ct),
                    "N_at_5": counts_k5[ct],
                    "log_rank@3": cat_avg(ct, "logrank3"),
                    "EUG_gap@3": eug_gap(ct), "EUG@5": eug(ct), "EUG@5_N": eug_den[ct],
                    "diagnostic": {CONFLICTS[ct]["diag"][1]: cat_avg(ct, "diag")} if CONFLICTS[ct]["diag"][1] else None,
                }
                for ct in present
            },
            # This records which EUG is which, next to the numbers, so a
            # future reader of the JSON cannot mistake our conditional
            # statistic for upstream's.
            "metric_notes": {
                "EUG_gap@3": "upstream MemConflict EUG: SEH@3 - AA per type; macro value = Average_EUG",
                "EUG@5": "this repo only: mean AA over questions whose gold evidence reached rank <= 5",
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
