#!/usr/bin/env python
"""Build two result CSVs from the banked v5 featured penalty-rubric scores.

Classification (penalty rubric, from parse_penalty_binary_value /
parse_trinary_score_value in benchmark/penalty_judge_eval/eval_scoring.py):
  answer_accuracy == 1.0  -> correct
  answer_accuracy == 0.5  -> partial (dynamic/static only)
  answer_accuracy == 0.0  -> blank   (absent/uncertain, incl. missing_answer)
  answer_accuracy == -1.0 -> incorrect (wrong/contradictory)

Token columns: generation-side vLLM counter deltas. cached input =
prefix_cache_hits; uncached = prompt_tokens - hits; output = generation_tokens.
Per-turn = divided by 71,060 dialogue turns (dataset property, same for all).
"""
import json, csv, os, collections

# Repo root is the parent of this script's benchmark/ directory, so the script
# runs from any checkout without a hardcoded path.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
TURNS = 71060  # dialogue turns across the 30 personas (BENCHMARK_MATRIX 1675-1677)

# (label, provider, tag, scores_file)
WAVES = [
    ("Honcho",              "honcho",          "v5ftc",    "honcho/Scores/honcho_v5ftc_gj12pen_eval_scores.jsonl"),
    ("mem0",                "mem0",            "v5ftc",    "mem0/Scores/mem0_v5ftc_gj12pen_eval_scores.jsonl"),
    ("Supermemory",         "supermemory",     "v5ftc",    "supermemory/Scores/supermemory_v5ftc_gj12pen_eval_scores.jsonl"),
    ("Hindsight (2nd arm)", "hindsight",       "v5ftcall", "hindsight/Scores/hindsight_v5ftcall_gj12pen_eval_scores.jsonl"),
    ("RetainDB server",     "retaindb_server", "v5ftc",    "retaindb_server/Scores/retaindb_server_v5ftc_gj12pen_eval_scores.jsonl"),
    ("Hindsight (featured)","hindsight",       "v5ftc086", "hindsight/Scores/hindsight_v5ftc086_gj12pen_eval_scores.jsonl"),
    ("OpenViking",          "openviking",      "v5ftovk",  "openviking/Scores/openviking_v5ftovk_gj12pen_eval_scores.jsonl"),
    ("Mnemosyne",           "mnemosyne",       "v5ftc",    "mnemosyne/Scores/mnemosyne_v5ftc_gj12pen_eval_scores.jsonl"),
]

# Generation tokens. 7 waves from the reviewed BENCHMARK_MATRIX table (lines
# 1691-1699); Hindsight v5ftcall computed from its token_usage sidecar because
# the matrix lists only the featured Hindsight arm. (cached, uncached, output)
# Each value matches the wave's VALID whole-window sidecar: the base
# token_usage_<tag>.json for honcho, hindsight v5ftc086, and mnemosyne, and
# token_usage_<tag>_all30.json where the pool supervisor closed its window
# before hand-relaunched personas finished (mem0, supermemory, retaindb_server,
# openviking, hindsight v5ftcall).
TOK = {
    ("honcho","v5ftc"):          (392886912, 539248157, 42546584),
    ("mem0","v5ftc"):            (532169088, 130026454, 17148227),
    ("supermemory","v5ftc"):     (127627104,  41927976, 18328682),
    ("retaindb_server","v5ftc"): ( 33278784, 224111680, 52781967),
    ("hindsight","v5ftc086"):    ( 31144608, 155567746, 21967822),
    ("openviking","v5ftovk"):    ( 25868832,  73741805, 19345376),
    ("mnemosyne","v5ftc"):       (        0,   8262116,  9825512),
}

def hindsight_v5ftcall_tokens():
    # The base token_usage_v5ftcall.json carries valid:false ("SUPERSEDED:
    # partial window" -- the pool supervisor closed it at 11:54 UTC while the
    # hand-relaunched persona 26 ran until 12:53). The _all30 file is the valid
    # window over all 30 personas and is what BENCHMARK_MATRIX banks.
    g = json.load(open("hindsight/Results/token_usage_v5ftcall_all30.json"))["servers"]["vllm_gen"]
    cached = g["prefix_cache_hits"]
    uncached = g["prompt_tokens"] - cached
    return (cached, uncached, g["generation_tokens"])
TOK[("hindsight","v5ftcall")] = hindsight_v5ftcall_tokens()

# Restart caveat per wave. The three token columns are whole-window vLLM
# counter deltas, so a persona attempt that failed inside the window and was
# relaunched leaves its tokens in the numerator while the 71,060-turn
# denominator counts each turn once. Concurrent pool runs share one vLLM, so
# a failed attempt's tokens cannot be subtracted; where a wave had in-window
# failed attempts the label carries a bounded inflation estimate built from
# the committed persona_pool_<tag>.log failure lines and the session depths in
# docs/BENCHMARK_MATRIX.md.
CAVEAT = {
    ("honcho","v5ftc"):          "clean (pool log: all 30 exited 0)",
    ("mem0","v5ftc"):            "inflated ~1% (persona 1 failed at ~session 20, relaunched from session 0)",
    ("supermemory","v5ftc"):     "inflated ~10-20% (9 failed persona attempts in window, bounded estimate)",
    ("hindsight","v5ftcall"):    "inflated ~3% (persona 26 failed near-complete, relaunched; 30-wide false start excluded by window start)",
    ("retaindb_server","v5ftc"): "no restarts; ~0.6% foreign traffic in window (documented, left in)",
    ("hindsight","v5ftc086"):    "clean (pool log: all 30 exited 0)",
    ("openviking","v5ftovk"):    "inflated ~15-20% (9 failed attempts across 8 personas, most past session 38; bounded estimate)",
    ("mnemosyne","v5ftc"):       "clean (single container, no relaunches)",
}

def classify(v):
    if v == 1.0:  return "correct"
    if v == 0.5:  return "partial"
    if v == 0.0:  return "blank"
    if v == -1.0: return "incorrect"
    return "other"   # should never happen

def aa_value(er):
    """Return the conflict-type answer_accuracy value, or None."""
    m = er.get("Metrics") or {}
    for k, v in m.items():
        if k.endswith("answer_accuracy"):
            return v
    # missing_answer rows may carry no Metrics; they are scored 0.0 = blank
    if er.get("Judge_Method") == "missing_answer":
        return 0.0
    return None

def bin_label(session_idx):  # session_idx is 1-based
    lo = ((session_idx - 1) // 5) * 5 + 1
    return f"{lo}-{lo+4}"

summary_rows = []
bin_rows = []

for label, prov, tag, sf in WAVES:
    tot = collections.Counter()
    per_bin = collections.defaultdict(collections.Counter)
    other = 0
    with open(sf, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            persona = json.loads(line)
            for si, sess in enumerate(persona["Full_Session_Chain"], start=1):
                b = bin_label(si)
                for q in sess.get("Session_Questions") or []:
                    er = q.get("Evaluation_Result") or {}
                    v = aa_value(er)
                    if v is None:
                        other += 1
                        continue
                    c = classify(v)
                    if c == "other":
                        other += 1
                        continue
                    tot[c] += 1
                    per_bin[b][c] += 1
    n = tot["correct"] + tot["partial"] + tot["blank"] + tot["incorrect"]
    cached, uncached, output = TOK[(prov, tag)]
    summary_rows.append({
        "provider": label,
        "tag": tag,
        "number_of_questions": n,
        "correct": tot["correct"],
        "partial_correct": tot["partial"],
        "blank": tot["blank"],
        "incorrect": tot["incorrect"],
        "cached_input_per_turn": round(cached / TURNS, 1),
        "uncached_input_per_turn": round(uncached / TURNS, 1),
        "output_tokens_total": output,
        "token_restart_caveat": CAVEAT[(prov, tag)],
    })
    if other:
        print(f"WARN {label}: {other} unclassified questions")
    # bins sorted numerically by lower bound
    for b in sorted(per_bin, key=lambda x: int(x.split("-")[0])):
        cc = per_bin[b]
        bn = cc["correct"] + cc["partial"] + cc["blank"] + cc["incorrect"]
        bin_rows.append({
            "provider": label,
            "tag": tag,
            # Excel reads "6-10" as a date; the ="..." text-formula form forces
            # it to render the literal string instead.
            "session_bin": f'="{b}"',
            "number_of_questions": bn,
            "correct": cc["correct"],
            "partial_correct": cc["partial"],
            "blank": cc["blank"],
            "incorrect": cc["incorrect"],
        })
    print(f"{label:22} n={n} correct={tot['correct']} partial={tot['partial']} "
          f"blank={tot['blank']} incorrect={tot['incorrect']}")

OUT = os.path.join(REPO, "benchmark", "Scores")
os.makedirs(OUT, exist_ok=True)
p1 = os.path.join(OUT, "v5_featured_results_summary.csv")
p2 = os.path.join(OUT, "v5_featured_results_by_session_bin.csv")

with open(p1, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader(); w.writerows(summary_rows)

with open(p2, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
    w.writeheader(); w.writerows(bin_rows)

print("\nwrote:", p1)
print("wrote:", p2)
