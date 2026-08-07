#!/usr/bin/env python3
"""Pre-score row gate: refuse to launch the judge stage on a dead generate stage.

The judge stage is expensive: a full LLM pass over every question, at
SCORE_WORKERS concurrency, running for hours. A generate stage that died
early, hung, or silently produced empty answers must not burn that judge
time on a partial or garbage Results file (the qwen "rumination" empty-answer
spiral is a known contract-v2 failure mode; see CLAUDE.md). This script is
the guard. Run it immediately before score_resumable.py, wired into
run_score() in benchmark/docker/answer_env.sh, against the same Results
JSONL the judge is about to read.

This script is provider-agnostic by construction. It reads exactly the row
schema benchmark/eval_common.py's run_eval() writes for every provider
(mnemosyne, hindsight, retaindb): one JSON object per line, one persona per
row, with persona identity in the ``ID`` field and per-question rows nested
at ``Full_Session_Chain[*].Session_Questions[*]``. We verified these field
names against real committed rows, for example
hindsight/Results/hindsight_results_armB_qwen.jsonl and
mnemosyne/Results/*.jsonl; see ProviderBinding.begin_persona's ctx contract
and Build_Compact_Question/Build_Compact_Session in eval_common.py.

Checks (all run every time):
  1. FILE PRESENCE / ROW COUNT -- a missing file, or zero persona rows, fails.
  2. EMPTY-ANSWER FRACTION -- among question rows that were actually
     attempted (a "Model_Answer" key is present at all; run_eval() never
     sets it on a question that max_questions_per_session skipped), the
     fraction whose Model_Answer is null or whitespace-only. This fails when
     it exceeds EMPTY_ANSWER_MAX_FRAC (default 0.02).
  3. ZERO ATTEMPTS -- persona rows exist but not one question carries a
     Model_Answer key. This fails. An earlier version passed this case
     vacuously: empty_frac was 0.0 when attempted_questions was 0, and the
     empty-frac gate armed itself only when attempted_questions was above 0.
     So a generate stage that ingested 30 personas and answered nothing used
     to sail into the judge.
  4. PERSONA COUNT -- distinct persona IDs present versus the count implied
     by NUM_PERSONAS or START_IDX+END_IDX, when env lets us derive it. Fewer
     than expected fails, and more than expected also fails (an earlier
     version failed only on fewer, so scoring a concatenation of the wrong
     shards used to pass). When not derivable, for example a plain
     STAGE=score re-run with none of those vars set, this only warns, per
     the caller's explicit instruction not to guess.
  5. DUPLICATE PERSONAS -- the same persona ID on more than one row fails (a
     double-merged shard file). Rows with no ID at all only warn.
  6. DATASET CROSS-CHECK, best-effort, never a spurious failure -- when env
     lets us derive the persona slice and the MemConflict dataset is
     readable, we load the expected slice from the dataset itself (env
     MEMCONFLICT_DATASET, default ../external/MemConflict/Data/
     Step4_4.jsonl, the same resolution write_manifest.py uses):
       * persona IDs outside the expected dataset slice fail (this means
         scoring the wrong shard's rows against this run's tag);
       * the expected eligible question count (question items whose
         "question" string is non-empty, exactly what run_eval()'s answer
         loop attempts; see eval_common.py Answer_Questions_For_One_Session)
         must match the attempted count, unless MAX_SESSIONS or
         MAX_QUESTIONS_PER_SESSION is set (a capped smoke run legitimately
         attempts fewer, so this only prints the numbers as info).
     A missing or unreadable dataset, or an underivable slice, only warns
     and skips the check. This check must never invent a failure out of
     environmental bad luck.

Escape hatch: SKIP_ROW_GATE=1 still runs every check and prints the summary,
but never exits nonzero. This is a documented bypass for a deliberately
partial re-score.

Usage:
    python benchmark/preflight_rows.py --results_file hindsight/Results/hindsight_results_full_s0.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, Optional


def _iter_question_items(persona: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every question-item dict in a persona row (chain, then session, then question).

    Mirrors benchmark/replay_answers.py's ``_iter_question_items``: same
    Results JSONL shape, same traversal.
    """
    for session in persona.get("Full_Session_Chain", []) or []:
        if not isinstance(session, dict):
            continue
        for question_item in session.get("Session_Questions", []) or []:
            if isinstance(question_item, dict):
                yield question_item


def _load_personas(path: str):
    """Load one persona dict per JSONL line.

    A malformed line is itself a fail-fast signal of a truncated or crashed
    write mid-line, so this raises SystemExit(1) directly instead of
    returning a partial list.
    """
    personas = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                personas.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[preflight_rows] FAIL: {path}:{line_no} is not valid JSON "
                    f"(truncated write?): {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
    return personas


def _int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _expected_persona_count() -> Optional[int]:
    """Derive the expected persona count from env, or None if not derivable.

    START_IDX+END_IDX (Hindsight/RetainDB sharding) wins when both are set.
    Otherwise this falls back to NUM_PERSONAS, which is Mnemosyne's
    whole-run count and the default persona cap for the other two providers.
    This never guesses from just one of START_IDX/END_IDX: a lone START_IDX
    with no END_IDX tells us where the personas start, not how many were
    requested.
    """
    start_idx = _int_env("START_IDX")
    end_idx = _int_env("END_IDX")
    if start_idx is not None and end_idx is not None:
        return end_idx - start_idx
    return _int_env("NUM_PERSONAS")


def _caps_env_set() -> bool:
    """True when a session or question cap is in effect, that is, a smoke-shaped run."""
    return bool(os.environ.get("MAX_SESSIONS", "").strip()) or bool(
        os.environ.get("MAX_QUESTIONS_PER_SESSION", "").strip()
    )


def _dataset_path() -> str:
    """Dataset path, using the same resolution as write_manifest.py `_dataset_info()`."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get(
        "MEMCONFLICT_DATASET",
        os.path.join(root, "external", "MemConflict", "Data", "Step4_4.jsonl"),
    )


def _persona_slice_bounds() -> Optional[tuple]:
    """(start, end) persona slice implied by env, or None if underivable.

    Mirrors eval_common.run_eval(): `all_personas[start_idx:end_idx]` for
    Hindsight/RetainDB shards (START_IDX+END_IDX), or the first
    NUM_PERSONAS personas for a Mnemosyne whole-run (`[0:NUM_PERSONAS]`).
    """
    start_idx = _int_env("START_IDX")
    end_idx = _int_env("END_IDX")
    if start_idx is not None and end_idx is not None:
        return (start_idx, end_idx)
    num = _int_env("NUM_PERSONAS")
    if num is not None:
        return (0, num)
    return None


def _dataset_expectations() -> Optional[Dict[str, Any]]:
    """Expected persona IDs and eligible question count for the env-derived
    dataset slice. Returns None, with a warning, when that is not derivable.

    "Eligible" means a question item dict whose `question` string is
    non-empty after strip(). run_eval()'s answer loop attempts exactly
    these items: it skips past empty question text without ever setting
    Model_Answer (eval_common.py Answer_Questions_For_One_Session). We
    validated this against the committed v1 full runs: 30 personas gave
    3,750 eligible questions, matching the dataset's documented size.

    This is best-effort by contract. Any problem, such as a missing file,
    bad JSON, or an unexpected shape, returns None after a warning, and the
    caller skips the cross-check instead of failing the gate on
    environmental bad luck.
    """
    bounds = _persona_slice_bounds()
    if bounds is None:
        return None
    path = _dataset_path()
    try:
        personas = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    personas.append(json.loads(line))
        start, end = bounds
        selected = personas[start:end]
        expected_ids = []
        eligible_questions = 0
        for persona in selected:
            pid = persona.get("ID")
            if pid is not None:
                expected_ids.append(str(pid))
            for question_item in _iter_question_items(persona):
                if str(question_item.get("question", "")).strip():
                    eligible_questions += 1
        return {
            "path": path,
            "slice": (start, end),
            "ids": expected_ids,
            "eligible_questions": eligible_questions,
        }
    except Exception as exc:
        print(
            f"[preflight_rows] WARN: could not derive dataset expectations "
            f"from {path}: {exc} -- skipping the dataset cross-check",
            file=sys.stderr,
        )
        return None


def _empty_answer_max_frac() -> float:
    raw = os.environ.get("EMPTY_ANSWER_MAX_FRAC", "0.02")
    try:
        return float(raw)
    except ValueError:
        print(
            f"[preflight_rows] WARN: EMPTY_ANSWER_MAX_FRAC={raw!r} is not a "
            "number; using the default 0.02",
            file=sys.stderr,
        )
        return 0.02


def run(results_file: str) -> int:
    skip = os.environ.get("SKIP_ROW_GATE", "0") == "1"
    max_frac = _empty_answer_max_frac()

    if not os.path.isfile(results_file):
        print(f"[preflight_rows] rows=0 personas=0/? questions=0 empty_answers=0 "
              f"-- FILE NOT FOUND: {results_file}")
        if skip:
            print("[preflight_rows] SKIP_ROW_GATE=1 -- bypassing "
                  "(would have FAILED: missing results file)")
            return 0
        print(f"[preflight_rows] FAIL: results file does not exist: {results_file}",
              file=sys.stderr)
        return 1

    personas = _load_personas(results_file)
    n_rows = len(personas)

    persona_ids = set()
    duplicate_ids = set()
    rows_without_id = 0
    attempted_questions = 0
    empty_answers = 0
    for persona in personas:
        pid = persona.get("ID")
        if pid is None:
            rows_without_id += 1
        else:
            pid = str(pid)
            if pid in persona_ids:
                duplicate_ids.add(pid)
            persona_ids.add(pid)
        for question_item in _iter_question_items(persona):
            if "Model_Answer" not in question_item:
                # Never attempted, due to a max_questions_per_session cap or
                # a resumed row's untouched tail. Not an "empty answer".
                continue
            attempted_questions += 1
            answer = question_item.get("Model_Answer")
            if answer is None or str(answer).strip() == "":
                empty_answers += 1

    empty_frac = (empty_answers / attempted_questions) if attempted_questions else 0.0
    n_personas = len(persona_ids)
    expected_personas = _expected_persona_count()

    print(
        f"[preflight_rows] rows={n_rows} "
        f"personas={n_personas}" + (f"/{expected_personas}" if expected_personas is not None else "/?")
        + f" questions_attempted={attempted_questions} empty_answers={empty_answers} "
        f"({empty_frac:.1%}, max={max_frac:.1%})"
    )
    if rows_without_id:
        print(
            f"[preflight_rows] WARN: {rows_without_id} persona row(s) have no "
            "ID field -- they are excluded from persona-identity checks",
            file=sys.stderr,
        )

    failures = []
    if n_rows == 0:
        failures.append("results file has zero persona rows")
    if n_rows > 0 and attempted_questions == 0:
        failures.append(
            "persona rows exist but zero questions were ever attempted "
            "(no question item carries a Model_Answer key -- dead generate stage "
            "or a retained MAX_SESSIONS/MAX_QUESTIONS_PER_SESSION cap)"
        )
    if attempted_questions > 0 and empty_frac > max_frac:
        failures.append(
            f"empty-answer fraction {empty_frac:.1%} exceeds "
            f"EMPTY_ANSWER_MAX_FRAC={max_frac:.1%} ({empty_answers}/{attempted_questions})"
        )
    if duplicate_ids:
        sample = ", ".join(sorted(duplicate_ids)[:3])
        failures.append(
            f"{len(duplicate_ids)} persona ID(s) appear on more than one row "
            f"(double-merged shard file?): e.g. {sample}"
        )
    if expected_personas is not None:
        if n_personas < expected_personas:
            failures.append(
                f"only {n_personas} distinct persona ID(s) present, expected {expected_personas} "
                "(derived from NUM_PERSONAS/START_IDX/END_IDX)"
            )
        elif n_personas > expected_personas:
            failures.append(
                f"{n_personas} distinct persona ID(s) present, expected only {expected_personas} "
                "(derived from NUM_PERSONAS/START_IDX/END_IDX) -- extra/unexpected personas"
            )
    elif n_rows > 0:
        print(
            "[preflight_rows] WARN: persona-count expectation not derivable "
            "from env (need NUM_PERSONAS or START_IDX+END_IDX) -- skipping "
            "that check, not failing on it",
            file=sys.stderr,
        )

    # Best-effort dataset cross-check: persona identity and question totals.
    expectations = _dataset_expectations() if n_rows > 0 else None
    if expectations is not None:
        expected_ids = set(expectations["ids"])
        eligible = expectations["eligible_questions"]
        start, end = expectations["slice"]
        unexpected = persona_ids - expected_ids
        if unexpected:
            sample = ", ".join(sorted(unexpected)[:3])
            failures.append(
                f"{len(unexpected)} persona ID(s) not in the expected dataset "
                f"slice [{start},{end}) (wrong shard scored under this tag?): e.g. {sample}"
            )
        if _caps_env_set():
            print(
                f"[preflight_rows] INFO: dataset slice [{start},{end}) has "
                f"{eligible} eligible question(s); {attempted_questions} attempted "
                "-- MAX_SESSIONS/MAX_QUESTIONS_PER_SESSION set, so the equality "
                "check is skipped (capped run)"
            )
        elif attempted_questions != eligible:
            failures.append(
                f"attempted question count {attempted_questions} != {eligible} "
                f"eligible question(s) in dataset slice [{start},{end}) "
                "(truncated generate stage, or rows from a different slice)"
            )
        else:
            print(
                f"[preflight_rows] dataset cross-check OK: slice [{start},{end}) "
                f"eligible={eligible} == attempted={attempted_questions}"
            )

    if failures:
        reason = "; ".join(failures)
        if skip:
            print(f"[preflight_rows] SKIP_ROW_GATE=1 -- bypassing (would have FAILED: {reason})")
            return 0
        print(f"[preflight_rows] FAIL: {reason}", file=sys.stderr)
        return 1

    print("[preflight_rows] OK -- proceeding to score")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-score row gate: fail fast on a truncated/empty generate "
                     "stage before it reaches the judge stage."
    )
    ap.add_argument("--results_file", required=True,
                     help="Provider Results JSONL about to be scored (same file "
                          "score_resumable.py's --input_file points at).")
    args = ap.parse_args(argv)
    return run(args.results_file)


if __name__ == "__main__":
    sys.exit(main())
