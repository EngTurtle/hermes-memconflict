"""Build a penalty-judge copy of the upstream MemConflict Evaluation package.

WHY THIS EXISTS. The standard judge rubric collapses two different failures
into the same 0.0: an answer that states a WRONG fact, and an answer that
states no fact at all (a refusal, or an "I am not sure"). Answer accuracy
therefore cannot separate a provider that hallucinates a stale value from
one that abstains. This generator writes an ALTERNATE judge arm that scores
a wrong or contradictory answer -1 and keeps 0.0 for a missing or uncertain
one. The two arms are NOT comparable; -1 is outside upstream's metric range.

WHY A COPY AND NOT AN EDIT. `external/` holds pinned submodules, the exact
code under test, and CLAUDE.md forbids modifying anything under it. So this
script copies `external/MemConflict/Evaluation/*.py` to
`benchmark/penalty_judge_eval/` and applies exact string replacements to the
copy. `MEMCONFLICT_EVAL_DIR` then points `benchmark/score_resumable.py` and
`benchmark/llm_reasoning.py` at the copy. The copy is gitignored: this
generator is the committed artifact, the copy is derived.

    python benchmark/make_penalty_judge_evaldir.py
    MEMCONFLICT_EVAL_DIR=benchmark/penalty_judge_eval \
      benchmark/score_files.sh --temperature 1.0 --top_p 0.95 --top_k 64 \
        --suffix gj12pen <results.jsonl>

WHAT CHANGES, AND NOTHING ELSE. Each replacement below must match exactly
once, or the script exits 1 rather than write a half-patched copy. Run with
--verify to diff the copy against upstream and confirm that only the
intended lines differ.

  1-3. The three `Score 0.0 if ...` sentences in `build_llm_judge_prompt`
       (eval_scoring.py:456, 478, 500), one per conflict type. Every 1.0 and
       0.5 sentence, the diagnostic metric definitions, the support-rank
       definitions, and the JSON response schemas stay byte-identical, so
       update order recognition, contradiction recognition, and supporting
       evidence hit at K are unaffected.

  4. `parse_trinary_score_value` (eval_scoring.py:528-539) floors every value
     below 0.25 to 0.0. A judge-returned -1 would silently become 0.0 and the
     whole arm would measure nothing. A `numeric <= -0.5` branch runs BEFORE
     the existing thresholds, so the 1.0 / 0.5 / 0.0 boundaries are untouched.

  5. `conditional_answer_accuracy` is NOT in
     `PARTIAL_CREDIT_BLACK_BOX_METRICS`, so `parse_llm_metric_result`
     (eval_scoring.py:600-604) routes it through `parse_binary_value`, which
     returns `1 if numeric != 0 else 0`. A -1 would arrive as +1.0, the exact
     opposite of the intent. A third dispatch branch sends only
     `conditional_answer_accuracy` through a penalty-aware parser.
     `parse_binary_value` itself is NOT changed, so update order recognition
     and contradiction recognition keep upstream's binarization.

NOT PATCHED, ON PURPOSE. `build_rule_based_result` (eval_scoring.py:412-420)
and `build_missing_answer_result` still score 0.0, never -1. Those are the
no-LLM fallbacks, and a judge that never ran cannot assert that an answer is
wrong. Check `Judge_Method_Statistics` in the score file: any count outside
`llm_judge` is a question the penalty rubric never reached.
"""

import argparse
import difflib
import os
import shutil
import sys
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.abspath(os.path.join(
    CURRENT_DIR, "..", "external", "MemConflict", "Evaluation"))
DEFAULT_OUT_DIR = os.path.join(CURRENT_DIR, "penalty_judge_eval")

# (description, exact old text, new text). Each must occur EXACTLY once in
# eval_scoring.py. Whitespace and punctuation are significant: these strings
# are the judge prompt the model reads.
REPLACEMENTS: List[Tuple[str, str, str]] = [
    (
        "dynamic rubric: split wrong from absent",
        "Score 0.0 if it is wrong, contradictory, does not contain the key updated fact, or is overly uncertain.",
        "Score 0.0 if it does not contain the key updated fact, or is uncertain. Score -1 if the answer is wrong or contradictory.",
    ),
    (
        "static rubric: split wrong from absent",
        "Score 0.0 if it is wrong, contradictory, does not contain the key true fact, or is overly uncertain.",
        "Score 0.0 if it does not contain the key true fact, or is uncertain. Score -1 if the answer is wrong or contradictory.",
    ),
    (
        "conditional rubric: split wrong from absent",
        "Score 0.0 if the condition is wrong, contradictory, absent, or overly uncertain.",
        "Score 0.0 if the condition is absent or uncertain. Score -1 if the condition is wrong or contradictory.",
    ),
    (
        "parse_trinary_score_value: let -1 survive the floor",
        """    if numeric >= 0.75:
        return 1.0
    if numeric >= 0.25:
        return 0.5
    return 0.0""",
        """    # PENALTY JUDGE ARM. Upstream floors everything below 0.25 to 0.0, so a
    # judge-returned -1 would be indistinguishable from an abstention and the
    # arm would measure nothing. This branch runs first and leaves the 0.75
    # and 0.25 boundaries below byte-identical.
    if numeric <= -0.5:
        return -1.0
    if numeric >= 0.75:
        return 1.0
    if numeric >= 0.25:
        return 0.5
    return 0.0""",
    ),
    (
        "penalty-aware parser for the binarized conditional metric",
        """PARTIAL_CREDIT_BLACK_BOX_METRICS = {
    "dynamic_answer_accuracy",
    "static_answer_accuracy",
}""",
        """PARTIAL_CREDIT_BLACK_BOX_METRICS = {
    "dynamic_answer_accuracy",
    "static_answer_accuracy",
}

# PENALTY JUDGE ARM. conditional_answer_accuracy is binary upstream, so it
# runs through parse_binary_value, where `-1 != 0` returns 1 and a penalty
# becomes full credit. Only this metric moves to the parser below. The
# diagnostics (update_awareness_and_order_consistency_score,
# conflict_recognition_score) keep parse_binary_value unchanged, because
# their rubric lines are not patched and they never see a -1.
PENALTY_BINARY_BLACK_BOX_METRICS = {
    "conditional_answer_accuracy",
}


def parse_penalty_binary_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if numeric <= -0.5:
        return -1.0
    return 1.0 if numeric >= 0.5 else 0.0""",
    ),
    (
        "parse_llm_metric_result: dispatch the conditional metric",
        """        if metric_key in PARTIAL_CREDIT_BLACK_BOX_METRICS:
            metrics[metric_key] = parse_trinary_score_value(parsed_result.get(metric_key, 0))
        else:""",
        """        if metric_key in PARTIAL_CREDIT_BLACK_BOX_METRICS:
            metrics[metric_key] = parse_trinary_score_value(parsed_result.get(metric_key, 0))
        elif metric_key in PENALTY_BINARY_BLACK_BOX_METRICS:
            metrics[metric_key] = parse_penalty_binary_value(parsed_result.get(metric_key, 0))
        else:""",
    ),
]

TARGET_FILE = "eval_scoring.py"


def build(out_dir: str) -> None:
    if not os.path.isdir(UPSTREAM_DIR):
        sys.exit(f"FATAL: upstream Evaluation dir not found: {UPSTREAM_DIR}\n"
                 f"  run: git submodule update --init --recursive")

    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for name in sorted(os.listdir(UPSTREAM_DIR)):
        if not name.endswith(".py"):
            continue
        shutil.copy2(os.path.join(UPSTREAM_DIR, name), os.path.join(out_dir, name))
        copied += 1
    print(f"[penalty-judge] copied {copied} python files -> {out_dir}")

    target = os.path.join(out_dir, TARGET_FILE)
    # utf-8 keeps the upstream BOM as a leading ﻿ character, so the
    # written file is byte-identical outside the replacements below.
    with open(target, "r", encoding="utf-8") as f:
        text = f.read()

    for label, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"FATAL: '{label}' matched {n} times, expected exactly 1.\n"
                     f"  Upstream changed. Re-read {UPSTREAM_DIR}/{TARGET_FILE} "
                     f"and update REPLACEMENTS. Refusing to write a half-patched copy.")
        text = text.replace(old, new)
        print(f"[penalty-judge] applied: {label}")

    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"[penalty-judge] wrote {target}")


def verify(out_dir: str) -> int:
    """Diff every copied file against upstream and print what differs."""
    rc = 0
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".py"):
            continue
        up = os.path.join(UPSTREAM_DIR, name)
        cp = os.path.join(out_dir, name)
        with open(up, "r", encoding="utf-8") as f:
            a = f.read().splitlines()
        with open(cp, "r", encoding="utf-8") as f:
            b = f.read().splitlines()
        diff = list(difflib.unified_diff(a, b, fromfile=f"upstream/{name}",
                                         tofile=f"copy/{name}", lineterm=""))
        if not diff:
            print(f"[verify] {name}: identical")
            continue
        if name != TARGET_FILE:
            print(f"[verify] FAIL {name}: differs but must not")
            rc = 1
        added = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
        print(f"[verify] {name}: {len(removed)} lines removed, {len(added)} added")
        print("\n".join(diff))
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--verify", action="store_true",
                    help="Only diff an existing copy against upstream; do not rebuild.")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if args.verify:
        sys.exit(verify(out_dir))
    build(out_dir)
    sys.exit(verify(out_dir))


if __name__ == "__main__":
    main()
