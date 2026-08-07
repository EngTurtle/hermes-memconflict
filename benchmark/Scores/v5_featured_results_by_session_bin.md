# `v5_featured_results_by_session_bin.csv`: how each column is derived

The same answer classification as `v5_featured_results_summary.csv`, split by
where in each persona's conversation the question was asked. One row per
(wave, session bin). Produced by `benchmark/make_result_csvs.py`.

## Source

Each wave's `<provider>/Scores/<provider>_<tag>_gj12pen_eval_scores.jsonl`. This
CSV has no token columns.

## Columns

| column | derivation |
|---|---|
| `provider`, `tag` | The wave. The two Hindsight arms are separate. |
| `session_bin` | The 1-based position of the session in the persona's `Full_Session_Chain`, grouped in fives: `1-5`, `6-10`, `11-15`, … The deepest personas reach `51-55`. |
| `number_of_questions` | Questions in that bin, summed across all 30 personas. |
| `correct`, `partial_correct`, `blank`, `incorrect` | The four penalty-rubric categories (1.0 / 0.5 / 0.0 / −1.0), counted within the bin. Definitions are identical to the summary CSV (see `v5_featured_results_summary.md`). |

## Session binning

Sessions are numbered by their order in the persona's dialogue, counting every
session including the question-light early ones (session 1 is the persona's
first). The bin is `((session_index − 1) // 5) × 5 + 1` to that plus four. A
question asked in any persona's session 7 lands in bin `6-10`, regardless of
which persona.

The counts are aggregated **across personas by session index**: a bin's
`number_of_questions` is the total from every persona at that depth, so bins with
more personas reaching that depth hold more questions. Early bins are small
because early sessions carry few questions; late bins taper because not every
persona runs to 55 sessions. Each wave's bins sum to its 3,750 questions.

## Excel note

`session_bin` is written as `="1-5"` (a text-formula wrapper) so Excel keeps it
as text. Without it, Excel reads `6-10` as June 10. Tools other than Excel
(pandas, plain readers) see the literal string `="1-5"` and can strip the
wrapper.

## What the split shows

Comparing a wave's later bins to its earlier ones shows how answer quality moves
as the memory store grows over the conversation. Bins hold different numbers of
questions (`number_of_questions`), so compare each category as a share of that
count, not as a raw count: a rising `incorrect` share in the later bins indicates
accuracy decaying as the store grows.

## Regenerate

```
.venv/Scripts/python.exe benchmark/make_result_csvs.py
```
