"""Retrieval-frozen answer replay for the MemConflict benchmark.

Completed provider runs stored, per question, the top-K ``Retrieved_Memories``
and a ``Model_Answer``. The answer-decoding config (temperature, max_tokens,
thinking) differed between providers, a fairness bug. This tool re-asks the
answer LLM for every already-answered question using the stored retrievals,
so retrieval stays frozen, under whatever decoding env is currently set. It
writes a new Results JSONL that flows unchanged through the shared scorer.

Because retrieval is frozen, SEH@K and log-rank@K stay unchanged by
construction. Only ``Model_Answer`` changes, so any AA delta isolates the
decoding effect.

This tool reuses the answer prompt, system prompt, and LLM call verbatim from
``eval_common``. It must never re-implement them, or it would stop being a
faithful replay of the shared answer contract.

Usage (decoding is env-driven, so set OPENAI_TEMPERATURE / OPENAI_MAX_TOKENS /
MEMCONFLICT_ENABLE_THINKING / OPENAI_MODEL before running):

    python benchmark/replay_answers.py \
        --input_file  hindsight/Results/hindsight_results_full_s0.jsonl \
        --output_file hindsight/Results/hindsight_results_full_s0_replay.jsonl

    # smoke test, makes no LLM calls:
    python benchmark/replay_answers.py --input_file <in> --output_file <out> --dry_run
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Import the shared answer contract from eval_common no matter the launch cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from eval_common import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    Build_Retrieved_Memory_Context,
    Generate_Answer_With_Retrieved_Memory,
    load_jsonl_items,
    write_jsonl_items,
)

# The presence of this key forbids replay: it marks arms that fed the answer
# context beyond the stored Retrieved_Memories (Mnemosyne arm D / canonical).
# This tool cannot reconstruct that appended context, so replaying would
# silently answer from a smaller context than the original run, and the
# comparison would be invalid.
_FORBIDDEN_QUESTION_KEY = "Canonical_Context"

#: Question-row key holding the pre-replay answer, so old and new answers stay
#: auditable and resume can skip already-replayed questions.
_ORIGINAL_ANSWER_KEY = "Replayed_Original_Answer"

#: Question-row key holding the SHA-256 of the replay inputs that produced the
#: stored replayed answer (see `replay_input_fingerprint`). Resume reuses a
#: stored answer only when this hash matches the one recomputed from the
#: current --input_file, so answers replayed from different retrievals can
#: never be silently carried over.
_REPLAY_FP_KEY = "Replay_Input_FP"


def replay_input_fingerprint(question_item: Dict[str, Any], top_k: int) -> str:
    """Hash exactly the inputs that determine one replayed answer, for resume
    validation.

    Resume used to match only on persona ID and question count, so an output
    file produced from a different Results file (same personas, different
    retrievals) could silently donate its answers. This closes that hole the
    same way score_resumable.py's `judge_input_fingerprint` closes it for the
    judge checkpoint: a stored answer is reused only if the answer LLM would
    have seen the identical prompt inputs.

    The hashed tuple is exactly what `_replay_one_question` feeds the LLM:
    the stripped question text, the rendered memory context, and K. The
    memory context comes from the shared `Build_Retrieved_Memory_Context`
    over `Retrieved_Memories[:top_k]`, so a change to memory text, created_at,
    or K flows into the hash, while non-rendered fields like `score` cannot
    cause a spurious re-replay. The decoding env is not hashed on purpose: it
    is recorded per persona in Replay_Manifest, and the whole point of this
    tool is to replay the same inputs under a different decoding env.

    This function does not share code with
    score_resumable.judge_input_fingerprint, on purpose. Importing
    score_resumable has module-level side effects (sys.path
    inserts, `llm_reasoning.install_as_llm_request()`, an upstream
    `eval_scoring` import), and a live run's score stage may execute that
    file while it must stay frozen. Both functions share the same semantics,
    a stable JSON list payload hashed to sha256 hex. Unify them later if
    score_resumable's helpers are ever factored out.
    """
    retrieved = question_item.get("Retrieved_Memories", []) or []
    payload = json.dumps(
        [
            str(question_item.get("question", "")).strip(),
            Build_Retrieved_Memory_Context(retrieved[:top_k]),
            top_k,
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Row navigation helpers
# --------------------------------------------------------------------------
def _iter_question_items(persona: Dict[str, Any]):
    """Yield every question-item dict in a persona row (chain, then session, then question)."""
    for session in persona.get("Full_Session_Chain", []) or []:
        if not isinstance(session, dict):
            continue
        for question_item in session.get("Session_Questions", []) or []:
            if isinstance(question_item, dict):
                yield question_item


def _has_model_answer(question_item: Dict[str, Any]) -> bool:
    return question_item.get("Model_Answer") not in (None, "")


def _current_decoding_env() -> Dict[str, Optional[str]]:
    """The decoding env as currently set. Stamped into every persona manifest,
    so a replay file records the config under which it was regenerated."""
    keys = (
        "OPENAI_TEMPERATURE",
        "OPENAI_MAX_TOKENS",
        "MEMCONFLICT_ENABLE_THINKING",
        "OPENAI_MODEL",
    )
    return {k: os.environ.get(k) for k in keys}


# --------------------------------------------------------------------------
# Pre-flight: refuse Canonical_Context arms; warn on top-K mismatch
# --------------------------------------------------------------------------
def _preflight(personas: List[Dict[str, Any]], top_k: int) -> None:
    """Refuse forbidden arms by raising SystemExit. Warn on Actual_Top_K drift."""
    mismatched_k = 0
    seen_k = set()
    for p_idx, persona in enumerate(personas):
        for question_item in _iter_question_items(persona):
            if _FORBIDDEN_QUESTION_KEY in question_item:
                sys.stderr.write(
                    f"REFUSING to replay: question item in persona index {p_idx} "
                    f"(ID={persona.get('ID')!r}) contains a {_FORBIDDEN_QUESTION_KEY!r} "
                    "key. Such arms answered from context beyond the stored "
                    "Retrieved_Memories, so a retrieval-frozen replay would be "
                    "invalid. Aborting without writing.\n"
                )
                sys.exit(2)
            actual_k = question_item.get("Actual_Top_K")
            if actual_k is not None:
                seen_k.add(actual_k)
                if actual_k != top_k:
                    mismatched_k += 1
    if mismatched_k:
        sys.stderr.write(
            f"WARNING: --top_k={top_k} but {mismatched_k} question(s) carry a "
            f"different Actual_Top_K (seen values: {sorted(seen_k)}). The replay "
            f"will build context from Retrieved_Memories[:{top_k}], which may not "
            "match the top-K the original run actually used.\n"
        )


# --------------------------------------------------------------------------
# Resume: index already-written output personas by ID
# --------------------------------------------------------------------------
def _load_resume_index(output_file: str) -> Dict[str, Dict[str, Any]]:
    """Map persona ID to its already-written output persona, for resume."""
    if not os.path.exists(output_file):
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for persona in load_jsonl_items(output_file):
        index[str(persona.get("ID"))] = persona
    return index


def _merge_resumed_answers(
    working: Dict[str, Any], resumed: Dict[str, Any], top_k: int
) -> Tuple[int, int, int]:
    """Copy already-replayed answers from a resumed persona onto ``working``.

    This matches question items positionally within the persona, since the
    same input file gives the same chain shape. It then validates each
    candidate's stored ``Replay_Input_FP`` against the fingerprint
    recomputed from the current input question (`replay_input_fingerprint`):

      * match    -> carry the answer, and the replay scan skips it;
      * mismatch -> drop the answer and re-replay the question, exactly what
                    a fresh run against this input would do;
      * absent   -> the row predates fingerprinting. Accept it unverified,
                    and leave it without a fingerprint so it stays visibly
                    legacy on any later resume. This mirrors
                    score_resumable's checkpoint policy.

    Returns ``(carried, legacy, mismatched)``. A carried question has
    ``Replayed_Original_Answer`` set, and is not re-asked.
    """
    carried = legacy = mismatched = 0
    resumed_questions = list(_iter_question_items(resumed))
    working_questions = list(_iter_question_items(working))
    if len(resumed_questions) != len(working_questions):
        # Shape drift should not happen for the same input. Ignore resume for
        # this persona and replay it fresh, instead of misaligning answers.
        return 0, 0, 0
    for w_q, r_q in zip(working_questions, resumed_questions):
        if _ORIGINAL_ANSWER_KEY not in r_q:
            continue
        want = replay_input_fingerprint(w_q, top_k)
        stored_fp = r_q.get(_REPLAY_FP_KEY)
        if stored_fp is not None and stored_fp != want:
            mismatched += 1
            continue
        if stored_fp is None:
            legacy += 1
        else:
            w_q[_REPLAY_FP_KEY] = stored_fp
        w_q["Model_Answer"] = r_q.get("Model_Answer")
        w_q[_ORIGINAL_ANSWER_KEY] = r_q.get(_ORIGINAL_ANSWER_KEY)
        if "Response_Duration_ms" in r_q:
            w_q["Response_Duration_ms"] = r_q.get("Response_Duration_ms")
        carried += 1
    return carried, legacy, mismatched


# --------------------------------------------------------------------------
# The replay itself
# --------------------------------------------------------------------------
def _replay_one_question(question_item: Dict[str, Any], top_k: int) -> None:
    """Re-ask the answer LLM for one question using its stored retrievals.

    This mutates ``question_item`` in place: it swaps Model_Answer, refreshes
    Response_Duration_ms, and records the pre-replay answer and the input
    fingerprint that resume validates against. It runs inside a worker
    thread. Each call owns a distinct dict, so it needs no shared-state
    locking.
    """
    # Compute the fingerprint first. The hashed fields (question,
    # Retrieved_Memories, K) are not mutated below, but computing the hash
    # before any mutation keeps that invariant obvious.
    input_fp = replay_input_fingerprint(question_item, top_k)
    retrieved = question_item.get("Retrieved_Memories", []) or []
    context_text = Build_Retrieved_Memory_Context(retrieved[:top_k])
    question_text = str(question_item.get("question", "")).strip()
    old_answer = question_item.get("Model_Answer")
    answer_text, _cost_info, duration_ms = Generate_Answer_With_Retrieved_Memory(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        context_text=context_text,
        question_text=question_text,
    )
    # Stamp the audit keys first, then overwrite the live answer.
    question_item[_ORIGINAL_ANSWER_KEY] = old_answer
    question_item[_REPLAY_FP_KEY] = input_fp
    question_item["Model_Answer"] = answer_text
    question_item["Response_Duration_ms"] = duration_ms


class Counts:
    def __init__(self) -> None:
        self.total_questions = 0
        self.untouched = 0          # empty or absent Model_Answer, never replayed
        self.already_replayed = 0   # carried over from a resumed output file
        self.would_replay = 0       # non-empty answer, needs a fresh LLM call
        self.replayed = 0           # actually re-asked this run (0 in dry_run)


def replay_persona(
    working: Dict[str, Any],
    top_k: int,
    workers: int,
    remaining_limit: Optional[int],
    dry_run: bool,
    counts: Counts,
) -> None:
    """Replay every eligible question in one persona, in place."""
    pending: List[Dict[str, Any]] = []
    for question_item in _iter_question_items(working):
        counts.total_questions += 1
        if not _has_model_answer(question_item):
            counts.untouched += 1
            continue
        if _ORIGINAL_ANSWER_KEY in question_item:
            counts.already_replayed += 1
            continue
        counts.would_replay += 1
        if remaining_limit is not None and len(pending) >= remaining_limit:
            continue
        pending.append(question_item)

    if dry_run:
        # Exercise the context-rebuild path, which catches malformed
        # retrievals, but make no LLM calls.
        for question_item in pending:
            retrieved = question_item.get("Retrieved_Memories", []) or []
            Build_Retrieved_Memory_Context(retrieved[:top_k])
        return

    if not pending:
        return
    if workers <= 1:
        for question_item in pending:
            _replay_one_question(question_item, top_k)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda q: _replay_one_question(q, top_k), pending))
    counts.replayed += len(pending)


def run(args: argparse.Namespace) -> int:
    personas = load_jsonl_items(args.input_file)
    print(f"Loaded {len(personas)} persona rows from {args.input_file}")

    # Pre-flight: hard refusal on forbidden arms, plus a top-K mismatch warning.
    _preflight(personas, args.top_k)

    resume_index = _load_resume_index(args.output_file)
    if resume_index:
        print(
            f"Resuming: {len(resume_index)} persona row(s) already in "
            f"{args.output_file} — already-replayed questions will be skipped."
        )

    manifest = {
        "source_file": os.path.abspath(args.input_file),
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "decoding_env": _current_decoding_env(),
    }

    counts = Counts()
    remaining_limit = args.limit_questions
    resumed_legacy = 0   # resumed answers with no stored fingerprint (rows predating fingerprinting)
    resumed_stale = 0    # resumed answers whose inputs changed, so they were re-replayed
    results: List[Dict[str, Any]] = []

    for persona in personas:
        working = copy.deepcopy(persona)
        working["Replay_Manifest"] = copy.deepcopy(manifest)

        # Carry already-replayed answers from a prior, crashed output run
        # onto this fresh copy. The per-question scan below is the single
        # source of truth for the already_replayed tally: carried questions
        # now carry the audit key, so the scan counts and skips them. Each
        # candidate is fingerprint-validated against this input file (see
        # _merge_resumed_answers). Stale candidates re-replay.
        resumed = resume_index.get(str(persona.get("ID")))
        if resumed is not None:
            _c, _legacy, _stale = _merge_resumed_answers(working, resumed, args.top_k)
            resumed_legacy += _legacy
            resumed_stale += _stale

        # Under --limit_questions, budget fresh replays across personas.
        before_replayed = counts.replayed
        replay_persona(
            working=working,
            top_k=args.top_k,
            workers=args.workers,
            remaining_limit=remaining_limit,
            dry_run=args.dry_run,
            counts=counts,
        )
        if remaining_limit is not None:
            remaining_limit -= (counts.replayed - before_replayed)
            if remaining_limit < 0:
                remaining_limit = 0

        results.append(working)
        # Write after each persona, the same crash-safe pattern as run_eval.
        write_jsonl_items(args.output_file, results)

    # Same reporting policy as score_resumable.load_checkpoint. This accepts
    # legacy rows but flags them loudly, and drops and re-replays stale rows.
    if resumed_legacy:
        print(f"[replay] WARNING: {resumed_legacy} resumed answer(s) predate input "
              f"fingerprinting and were accepted UNVERIFIED (cannot prove they "
              f"were replayed from this exact --input_file)")
    if resumed_stale:
        print(f"[replay] {resumed_stale} resumed answer(s) had stale inputs "
              f"(question/retrieval changed) and were dropped for re-replay")

    _print_summary(args, counts)
    return 0


def _print_summary(args: argparse.Namespace, counts: Counts) -> None:
    print("")
    print("=== Replay summary ===")
    print(f"  input_file      : {args.input_file}")
    print(f"  output_file     : {args.output_file}")
    print(f"  top_k           : {args.top_k}")
    print(f"  dry_run         : {args.dry_run}")
    print(f"  total questions : {counts.total_questions}")
    print(f"  untouched (no Model_Answer) : {counts.untouched}")
    print(f"  already replayed (resumed)  : {counts.already_replayed}")
    print(f"  would replay    : {counts.would_replay}")
    if args.dry_run:
        print(f"  replayed (this run): 0 (dry run — no LLM calls)")
    else:
        print(f"  replayed (this run): {counts.replayed}")
    if args.limit_questions is not None:
        print(f"  (limited to {args.limit_questions} fresh replays)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_answers.py",
        description=(
            "Retrieval-frozen answer replay for MemConflict: re-ask the answer "
            "LLM for every already-answered question using its STORED retrievals, "
            "under the decoding env currently set. Isolates the answer-decoding "
            "effect on AA; SEH/log-rank are untouched by construction. Decoding "
            "is env-driven (OPENAI_TEMPERATURE / OPENAI_MAX_TOKENS / "
            "MEMCONFLICT_ENABLE_THINKING / OPENAI_MODEL)."
        ),
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Completed provider Results JSONL (one persona per line).",
    )
    parser.add_argument(
        "--output_file", required=True,
        help="New Results JSONL to write (resumable; existing rows are reused).",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Retrievals to feed the answer LLM; should match Actual_Top_K in "
             "the rows (a mismatch is warned, not fatal). Default 5.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Thread-pool size for answer LLM calls within a persona. Default 8.",
    )
    parser.add_argument(
        "--limit_questions", type=int, default=None,
        help="Cap the number of fresh replays (for smoke tests). Budgeted "
             "across personas in order. Default: no cap.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Rebuild contexts and count questions but make NO LLM calls.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        sys.stderr.write("--top_k must be a positive integer.\n")
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
