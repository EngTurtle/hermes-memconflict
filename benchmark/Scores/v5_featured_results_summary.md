# `v5_featured_results_summary.csv`: how each column is derived

One row per scored wave in the contract-v5 featured comparison. The judge is
gemma-4-12b under the penalty rubric (suffix `gj12pen`). All values are produced
by `benchmark/make_result_csvs.py`. That script is the authoritative derivation;
this file explains it.

## Sources

- **Answer classification** reads each wave's
  `<provider>/Scores/<provider>_<tag>_gj12pen_eval_scores.jsonl`. Each line is one
  persona; each question sits under `Full_Session_Chain[*].Session_Questions[*]`
  with an `Evaluation_Result` block.
- **Token columns** read the generation-server counter deltas in each wave's
  `token_usage_<tag>.json` sidecar (`servers.vllm_gen`), the same numbers banked
  in `docs/BENCHMARK_MATRIX.md`.

## Columns

| column | derivation |
|---|---|
| `provider`, `tag` | The wave. The two Hindsight arms are separate rows: `v5ftcall` (unfiltered recall) and `v5ftc086` (featured). |
| `number_of_questions` | Count of scored questions in the wave. 3,750 for every wave (30 personas × the dataset's questions). |
| `correct` | Questions whose conflict-type answer accuracy is **1.0**. |
| `partial_correct` | Answer accuracy **0.5**. Only dynamic and static conflicts can score 0.5; conditional is binary. |
| `blank` | Answer accuracy **0.0**, the judge's "absent or uncertain" bucket. Includes empty answers (`Judge_Method: missing_answer`, scored 0.0 without a judge call). Not the same as a wrong answer. |
| `incorrect` | Answer accuracy **−1.0**, the model committed to a wrong or contradictory answer. |
| `cached_input_per_turn` | `prefix_cache_hits` ÷ 71,060 dialogue turns. |
| `uncached_input_per_turn` | (`prompt_tokens` − `prefix_cache_hits`) ÷ 71,060. |
| `output_tokens_total` | `generation_tokens` for the wave (total, not per turn). |
| `token_restart_caveat` | Per-wave note on restart inflation of the token columns (see "Token caveat"). |

## The four answer categories

The value is the conflict-type accuracy metric in
`Evaluation_Result.Metrics`: `dynamic_answer_accuracy`,
`static_answer_accuracy`, or `conditional_answer_accuracy`. The penalty rubric
maps a question to exactly one of four values through
`parse_trinary_score_value` (dynamic, static) and `parse_penalty_binary_value`
(conditional) in `benchmark/penalty_judge_eval/eval_scoring.py`:

- **1.0** correct · **0.5** partial · **0.0** absent/uncertain (blank) ·
  **−1.0** wrong (incorrect).

The four counts partition the 3,750 questions with no remainder.

**Validation.** The mean of the raw values,
`(correct·1.0 + partial·0.5 + blank·0.0 + incorrect·−1.0) ÷ 3750`, equals each
wave's micro answer accuracy in its `summary_<tag>_gj12pen.json`. Checked:
Honcho 0.595, mem0 0.304.

Retrieval is sliced to the top 5 memories before the judge sees it
(`extract_top_k_retrieved_memories`), so the classification measures the answer
model working from the same top-5 evidence for every provider.

## Token columns

`71,060` is the number of dialogue turns across the 30 personas, a property of
the dataset (`docs/BENCHMARK_MATRIX.md`, "1,579 sessions and 71,060 dialogue
turns"), identical for every provider, so the per-turn columns are comparable.
Cost is reported per turn, not per question, because a Hermes deployment ingests
conversation rather than answering quiz items.

Cached input, uncached input, and output are kept separate because a hosted API
bills them at different rates. The counter reports logical prompt tokens
regardless of cache hits, so `cached_input` is prefix-cache hits and
`uncached_input` is the remainder that had to be prefilled.

## Token caveat

The token columns are whole-wave vLLM counter deltas: every generation-server
token during the wave's window is attributed to the run. A persona attempt that
failed partway and was relaunched leaves its tokens in the numerator, while the
71,060-turn denominator counts each turn once, so per-turn cost is overstated for
any wave that had restarts. The pool ran all personas concurrently through one
vLLM, so a failed attempt's tokens cannot be subtracted, so the caveat values are
bounded estimates from the committed `persona_pool_<tag>.log` failure lines and
the session depths in `docs/BENCHMARK_MATRIX.md`, not measurements. The answer
classification is unaffected: it reads the final banked answers.

Clean: Honcho, Hindsight `v5ftc086`, Mnemosyne. Inflated (bounded): Supermemory
~10-20%, OpenViking ~15-20%, mem0 ~1%, Hindsight `v5ftcall` ~3%. RetainDB had no
restarts but ~0.6% foreign traffic in its window.

## Regenerate

```
.venv/Scripts/python.exe benchmark/make_result_csvs.py
```

Rebuilds this CSV and `v5_featured_results_by_session_bin.csv` from the committed
scores and token sidecars.
