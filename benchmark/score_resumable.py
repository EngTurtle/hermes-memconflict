"""Restart-proof, resumable scoring for the MemConflict benchmark, for any provider.

The upstream scorer (`eval_scoring`) holds every persona's judge results in
memory and writes a persona only when it finishes. So a container restart
mid-run loses everything. This driver fixes that with two phases:

  Phase 1 (parallel, checkpointed): flatten all questions, judge each one
    with the upstream `Evaluate_Single_Question` in a flat thread pool, and
    append every verdict to a JSONL checkpoint keyed by a stable
    per-question id. A restart reloads the checkpoint and skips
    already-judged questions, losing at most the few in-flight calls.

  Phase 2 (no LLM): monkeypatch `Evaluate_Single_Question` to return the
    cached verdict, then call the upstream `Generate_User_Evaluation`
    unchanged. The upstream code does all metric derivation, white-box@K,
    and output formatting. So results are byte-for-byte what the standard
    scorer produces. Only the judging was parallelized and cached.

Run with MEMCONFLICT_JSON_MODE=1, so the judge returns valid JSON (see
benchmark/docker/answer_env.sh: bench_judge_env).

--------------------------------------------------------------------------
FIDELITY CONTRACT (audited 2026-07-21 against external/MemConflict/Evaluation/
eval_scoring.py). The whole cross-paper comparison depends on our judging
being the same judging the benchmark defines. So this section records
exactly what is reused and what is ours:

REUSED VERBATIM from upstream — never reimplemented in this file:
  * the judge prompt          `es.build_llm_judge_prompt` (per-type templates,
                              `white_box_config` support_rank_field/description,
                              top-K interpolation, memory rendering)
  * the judge system prompt   `es.LLM_JUDGE_SYSTEM_PROMPT`
  * the judge call itself     `es.evaluate_question_with_llm` -> `llm_request`
                              (incl. its `json_markers`)
  * response parsing          `es.parse_llm_metric_result`, `parse_binary_value`,
                              `parse_trinary_score_value`, `parse_support_rank`
  * white-box derivation      `es.derive_white_box_metrics_from_rank`,
                              `build_white_box_result_by_k`
  * fallbacks                 `es.build_rule_based_result`,
                              `es.build_missing_answer_result`
  * aggregation + output      `es.Generate_User_Evaluation` and everything it
                              calls, run unmodified in Phase 2
  * question eligibility      `es.METRIC_SCHEMAS` membership
  * the judged top-K          `es.MAX_WHITE_BOX_TOP_K` (5), the same value
                              upstream's `Generate_Single_Persona_Evaluation`
                              passes at eval_scoring.py:1027. So the prompt says
                              "Top-5" and the rank scale is 0 to 5. Judging once
                              at MAX and re-deriving hit@{2,3,5} from that one
                              rank is upstream's own design, not ours.
  Verified 2026-07-21: we captured the prompt both paths build, for one real
  dataset question of each conflict type, and diffed them. They are
  byte-identical, even after this file stamps `_qkey` onto the question dict.
  That field is not in `prediction_fields` and no prompt builder reads it, so
  it cannot leak.

OURS (the resumability layer). Each piece below is argued to be
results-neutral, because "faster or restartable" must never mean
"different":
  * per-question checkpointing + content fingerprint (see `load_checkpoint`)
  * a flat thread pool over questions instead of upstream's per-persona loop
  * qkey assignment + collision disambiguation (see `assign_qkeys`)
  * run-scoped default paths for the intermediate files (see `main`)

KNOWN, DELIBERATE, DOCUMENTED behavior differences:
  1. `assign_qkeys` skips a question whose `conflict_type` is not in
     `es.METRIC_SCHEMAS`. Upstream instead raises `ValueError`
     (eval_scoring.py:1021). The net effect is identical: neither path ever
     judges such a question, and Phase 2 runs the same upstream code that
     raises on it. So the run still fails, just after Phase 1 rather than
     during it. This file never silently scores anything upstream would
     refuse.
  2. A judge failure is cached. Upstream also retries nothing (a failed
     judge becomes `rule_based` in the same run), so a single run is
     faithful. But a resumed run reuses the cached `rule_based` verdict
     instead of retrying the judge. That is the price of resumability, and
     it is conservative: it can only reproduce, never improve on, what a
     single upstream run would have scored. `Judge_Method_Statistics` in the
     output makes the rate auditable. Check it before publishing any number.
  3. Concurrency cannot reorder anything that matters. Each verdict is a
     pure function of one question's own prompt, because upstream holds no
     cross-question state. Phase 2 re-runs the upstream aggregator
     single-threaded (`parallel_workers = 1`, upstream's own default). So
     persona order in the output file matches input order, regardless of
     the order Phase 1 finished in.
--------------------------------------------------------------------------
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMCONFLICT_EVAL_DIR = os.path.abspath(os.environ.get(
    "MEMCONFLICT_EVAL_DIR",
    os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Evaluation"),
))
for _p in (MEMCONFLICT_EVAL_DIR, CURRENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Route the judge through the reasoning/JSON-mode wrapper.
# IMPORT ORDER MATTERS: eval_scoring binds `from llm_request import llm_request`
# at import time (eval_scoring.py:25-28). So this file must install the alias
# before the import below, or the judge would run on the raw upstream client
# with none of the canonical decoding config from answer_env.sh.
import llm_reasoning  # noqa: E402
llm_reasoning.install_as_llm_request()

import eval_scoring as es  # noqa: E402


QKEY_FIELD = "_qkey"


def assign_qkeys(personas: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Stamp a stable id on every scorable question and return [(qkey, question)].

    The id is positional (`<persona ID>|<session idx>|<question idx>`)
    rather than content-derived, so a resumed run lines up with a checkpoint
    written before the crash. Positional ids are unique only if persona IDs
    are unique within the file. A Results file assembled by concatenating
    shards can break that. Two questions sharing a qkey would then share a
    cached verdict, meaning one question would be scored with another
    question's judgment. That is a correctness bug, not just wasted work, so
    an occurrence suffix disambiguates collisions. The first occurrence
    keeps the historic bare key, so every checkpoint written before this
    guard existed still resumes.

    (No committed Results file under mnemosyne/, hindsight/, or retaindb/
    has a duplicate persona ID, verified 2026-07-21. So this guard is
    precautionary and changes nothing about any existing artifact.)
    """
    flat: List[Tuple[str, Dict[str, Any]]] = []
    seen: Dict[str, int] = {}
    for p_idx, persona in enumerate(personas):
        pid = str(persona.get("ID", p_idx))
        for s_idx, session in enumerate(persona.get("Full_Session_Chain", [])):
            questions = session.get("Session_Questions", [])
            if not isinstance(questions, list):
                continue
            for q_idx, q in enumerate(questions):
                if not isinstance(q, dict):
                    continue
                # Upstream raises on an unknown conflict_type (eval_scoring.py:1021).
                # Skipping it here only avoids judging it. Phase 2 still runs
                # that upstream check, so the run fails exactly as upstream's
                # would.
                if q.get("conflict_type") not in es.METRIC_SCHEMAS:
                    continue
                qkey = f"{pid}|{s_idx}|{q_idx}"
                n = seen.get(qkey, 0)
                seen[qkey] = n + 1
                if n:
                    qkey = f"{qkey}#{n}"
                    print(f"[resumable] WARNING: duplicate question id, disambiguated -> {qkey}")
                q[QKEY_FIELD] = qkey
                flat.append((qkey, q))
    return flat


def judge_input_fingerprint(question_item: Dict[str, Any], prediction_fields: List[str]) -> str:
    """Hash exactly the inputs that determine a verdict, for checkpoint validation.

    A qkey identifies a question's position, not its content, and nothing
    in the checkpoint file records which Results file produced it. Two real
    hazards follow: pointing `--checkpoint` at another arm's checkpoint,
    which the tag-derived default in answer_env.sh makes one typo away, and
    re-scoring a Results file that was regenerated with different answers
    or retrieval. Either hazard would silently replay stale verdicts and
    publish them as this run's numbers.

    Fingerprinting closes that gap: a cached verdict is reused only if the
    judge would have seen identical inputs. The hashed tuple is exactly the
    set `Evaluate_Single_Question` consumes: conflict type, question,
    reference answer, the extracted model answer and the field it came
    from, the rendered Top-K memory block, and K. This is everything
    `build_llm_judge_prompt`, `build_rule_based_result`, and
    `build_missing_answer_result` read. Nothing outside that tuple can
    change a verdict. So hashing more would only cause spurious cache
    misses, meaning needless GPU use, and hashing less would leave a hole.
    """
    model_answer, used_field = es.extract_model_answer(question_item, prediction_fields)
    retrieved = es.extract_top_k_retrieved_memories(question_item, es.MAX_WHITE_BOX_TOP_K)
    payload = json.dumps(
        [
            question_item.get("conflict_type", "unknown"),
            question_item.get("question", ""),
            question_item.get("answer", ""),
            model_answer,
            used_field,
            es.format_retrieved_memories_for_prompt(retrieved),
            es.MAX_WHITE_BOX_TOP_K,
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_checkpoint(path: str, fingerprints: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Reload judged verdicts, dropping any whose inputs no longer match.

    This returns {qkey: {"judged": ..., "cost": ...}}. It accepts a record
    only if its stored fingerprint equals the one recomputed from the
    current input file (see `judge_input_fingerprint`). A mismatch is
    dropped and re-judged, exactly what a fresh run would do.

    Checkpoints written before fingerprinting existed carry no `fp` field.
    This function accepts them, because rejecting them would force a full,
    expensive re-judge of runs whose committed scores are already
    published. It counts and reports them loudly, because for those files
    the stale-verdict hazard above is unproven rather than excluded.
    Malformed lines, such as the torn last line of a checkpoint that was
    being appended to when the container died, are skipped. The affected
    question is then simply re-judged.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    legacy = mismatched = malformed = stale_key = 0
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qkey = rec["qkey"]
                judged = rec["judged"]
            except Exception:
                malformed += 1
                continue
            want = fingerprints.get(qkey)
            if want is None:
                # This is a verdict for a question this input file does not
                # contain, for example a checkpoint from a larger or older
                # run. Dropping it keeps the "N judged / M remaining" counts
                # honest. It is never looked up.
                stale_key += 1
                continue
            fp = rec.get("fp")
            if fp is None:
                legacy += 1
            elif fp != want:
                mismatched += 1
                cache.pop(qkey, None)
                continue
            cache[qkey] = {"judged": judged, "cost": rec.get("cost")}
    if stale_key:
        print(f"[resumable] {stale_key} checkpoint entries reference questions "
              f"absent from --input_file and were ignored")
    if legacy:
        print(f"[resumable] WARNING: {legacy} checkpoint entries predate input "
              f"fingerprinting and were accepted UNVERIFIED (cannot prove they "
              f"were judged from this exact --input_file)")
    if mismatched:
        print(f"[resumable] {mismatched} checkpoint entries had stale inputs "
              f"(answers/retrieval changed) and will be re-judged")
    if malformed:
        print(f"[resumable] {malformed} unparseable checkpoint lines skipped "
              f"(expected after a crash mid-append); those questions re-judge")
    return cache


ZERO_COST: Dict[str, Any] = {
    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    "total_cost_usd": 0.0, "model": None, "pricing_available": False,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", default=os.path.join(CURRENT_DIR, "Results", "mnemosyne_results.jsonl"))
    ap.add_argument("--output_file", default=os.path.join(CURRENT_DIR, "Scores", "mnemosyne_eval_scores.jsonl"))
    # These two default to None and are derived from --output_file below,
    # rather than being fixed paths. They used to be constants under
    # benchmark/Scores/. answer_env.sh's run_score() never overrides them,
    # so every provider and every RUN_TAG wrote the same two files. The
    # Phase 2 aggregates whatever is at that path. Two score stages often run concurrently here, since
    # the runs are sharded and long, and could have made Phase 2 aggregate
    # the other provider's questions under this run's name. Deriving these
    # paths from --output_file scopes them by provider and tag for free,
    # with no change to any entrypoint. Explicit values still win.
    ap.add_argument("--output_perfect_file", default=None)
    ap.add_argument("--checkpoint", default=os.path.join(CURRENT_DIR, "Scores", "judged_checkpoint.jsonl"))
    ap.add_argument("--annotated_input", default=None)
    # This fallback matches benchmark/docker/answer_env.sh
    # (SCORE_WORKERS:-40), the source of truth coupled to RETRY_TIMES and
    # timeout. It applies only when SCORE_WORKERS is unset.
    ap.add_argument("--workers", type=int, default=int(os.getenv("SCORE_WORKERS", "40")))
    ap.add_argument("--prediction_fields", default="Model_Answer,Predicted_Answer,Generated_Answer,memory_answer,model_answer,predicted_answer")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.output_file)
    out_stem = os.path.splitext(os.path.basename(args.output_file))[0]
    if args.output_perfect_file is None:
        args.output_perfect_file = os.path.join(out_dir, f"{out_stem}.json")
    if args.annotated_input is None:
        args.annotated_input = os.path.join(out_dir, f"_annotated_{out_stem}.jsonl")

    prediction_fields = [x.strip() for x in args.prediction_fields.split(",") if x.strip()]
    if os.path.dirname(args.checkpoint):
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    personas = es.load_jsonl_items(args.input_file)
    flat = assign_qkeys(personas)
    print(f"[resumable] {len(personas)} personas, {len(flat)} scorable questions")

    # This persists the qkey-annotated input, so Phase 2 loads identical
    # question ids. The round trip is lossless: these values came from
    # JSON, dict order is preserved (upstream's
    # place_session_evaluation_before_event_types depends on key order),
    # and Python's float repr round-trips exactly. So Phase 2 sees the same
    # objects Phase 1 judged.
    es.write_jsonl_items(args.annotated_input, personas)

    fingerprints = {qkey: judge_input_fingerprint(q, prediction_fields) for qkey, q in flat}
    cache = load_checkpoint(args.checkpoint, fingerprints)
    print(f"[resumable] checkpoint has {len(cache)} usable judged questions; "
          f"{len(flat) - len(cache)} remaining")

    todo = [(k, q) for (k, q) in flat if k not in cache]
    ckpt_lock = threading.Lock()
    ckpt_fh = open(args.checkpoint, "a", encoding="utf-8")
    done_counter = {"n": len(cache)}
    failures = {"n": 0}
    total = len(flat)
    start = time.time()

    def judge(item: Tuple[str, Dict[str, Any]]):
        qkey, q = item
        judged, cost = es.Evaluate_Single_Question(
            question_item=q,
            prediction_fields=prediction_fields,
            enable_llm_judge=True,
            top_k=es.MAX_WHITE_BOX_TOP_K,
        )
        # This checkpoints the judge cost alongside the verdict and replays
        # it in Phase 2. Without it, Observable_Token_Cost_Summary and
        # token_cost in every score file would report a zero-token judge
        # stage, a silent divergence from upstream, which reports the real
        # usage. Metrics never read cost, so this changes only reported
        # cost, never AA, SEH, EUG, or log-rank.
        with ckpt_lock:
            ckpt_fh.write(json.dumps(
                {"qkey": qkey, "fp": fingerprints.get(qkey), "judged": judged, "cost": cost},
                ensure_ascii=False) + "\n")
            ckpt_fh.flush()
            cache[qkey] = {"judged": judged, "cost": cost}
            done_counter["n"] += 1
            n = done_counter["n"]
        if n % 50 == 0:
            print(f"[resumable] judged {n}/{total} (+{time.time()-start:.0f}s)")
        return qkey

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(judge, it) for it in todo]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:  # Keep going. Missing verdicts default to rule-based in Phase 2.
                    failures["n"] += 1
                    print(f"[resumable] judge error: {e}")
    ckpt_fh.close()
    print(f"[resumable] Phase 1 complete: {len(cache)}/{total} judged in {time.time()-start:.0f}s"
          + (f" ({failures['n']} raised and will fall back to rule-based)" if failures["n"] else ""))

    # ---- Phase 2: aggregate via the unmodified upstream scorer, cache-backed ----
    misses = {"n": 0}

    def cached_evaluate(question_item, prediction_fields, enable_llm_judge=True, top_k=es.WHITE_BOX_TOP_K):
        # This signature mirrors es.Evaluate_Single_Question exactly,
        # including the top_k default, so upstream's call site, which
        # always passes MAX_WHITE_BOX_TOP_K explicitly (eval_scoring.py:1027),
        # behaves the same.
        qkey = question_item.get(QKEY_FIELD)
        record = cache.get(qkey)
        if record is not None:
            return record["judged"], dict(record.get("cost") or ZERO_COST)
        # This is the fallback for any question missing a verdict, because
        # Phase 1 raised, or the process died between the pool and here.
        # This is byte-identical to the branch upstream takes when its own
        # judge call fails. So such a question is scored the way upstream
        # would have scored it: not skipped, not dropped, and visible
        # afterward as rule_based or missing_answer in
        # Judge_Method_Statistics.
        misses["n"] += 1
        model_answer, used_field = es.extract_model_answer(question_item, prediction_fields)
        if not model_answer:
            return es.build_missing_answer_result(question_item.get("conflict_type", "unknown"), used_field, top_k), dict(ZERO_COST)
        return es.build_rule_based_result(question_item, model_answer, used_field, top_k), dict(ZERO_COST)

    es.Evaluate_Single_Question = cached_evaluate

    class _Args:
        input_file = args.annotated_input
        output_file = args.output_file
        output_perfect_file = args.output_perfect_file
        prediction_fields = args.prediction_fields
        enable_llm_judge = True
        # 1 is upstream's own default (eval_scoring.py:1287). This keeps
        # persona order in the output file equal to input order. Phase 2
        # makes no LLM calls, so there is nothing to parallelize anyway.
        parallel_workers = 1
    ok = es.Generate_User_Evaluation(_Args())
    if misses["n"]:
        print(f"[resumable] WARNING: {misses['n']} questions had no cached verdict and "
              f"were scored by upstream's non-LLM fallback")
    print(f"[resumable] Phase 2 aggregation {'OK' if ok else 'FAILED'} -> {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[resumable] fatal: {e}:{traceback.format_exc()}")
        sys.exit(1)
