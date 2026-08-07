# Troubleshooting

This file lists symptom, cause, and fix for problems that cost real time on this
benchmark. Each entry records what did **not** work, so you do not repeat the
same dead ends.

Contract versions: **v1** = `gemma-4-e2b`, **v3** = `qwen3.5-4b`
(`RedHatAI/Qwen3.5-4B-quantized.w4a16`). v2 served the same alias from
`AxionML/Qwen3.5-4B-NVFP4`. The project abandoned v2 with zero personas banked.
Entries tagged v1 describe the gemma era and may not apply to current serving.

---

## Serving (vLLM)

### The silent engine wedge, flashinfer top-k sampler race

**Symptom.** `Avg generation throughput: 0.0 tokens/s` with `Running: 28 reqs`,
and no change over time. `/health` still returns 200. No exception, no CUDA
fault, no Xid, nothing in any log. Only a process restart clears it.

**Detector.** Compare `nvidia-smi` GPU utilization against power draw. Health
endpoints and the `Running:` count both read normal throughout.

| State | GPU util | Power | Memory controller load |
|---|---|---|---|
| Wedged | ~100% | ~64W / 300W | 1-3% |
| Healthy | 95-97% | 192-222W | 67-81% |

The GPU clock runs at full speed, utilization looks full, and memory traffic is
zero. This signature means a kernel busy-waits on a sync that never completes.
The combination rules out three other causes: thermal throttle (PerfCap stays
unchanged), bandwidth starvation (would show high traffic), and slow compute
(would draw power).

**Cause.** `flashinfer-ai/flashinfer#3615`. In the multi-CTA radix top-k sampler,
the leading CTA zeroes the software barrier's `arrival_counter` with no
synchronization against peers' final `wait_ge`. A peer CTA then reads zero and
spins forever. The named platform is consumer Blackwell SM120/SM121. This host is
an RTX 5070 Ti at compute capability 12.0. The bug fires on every decode step
because the Qwen3.5 card sets `top_k=20`, and 28-30 shard concurrency is the
multi-CTA regime.

**Fix.** Set `VLLM_USE_FLASHINFER_SAMPLER=0` on `vllm-gen`. This keeps the same
decoding contract but runs it through a different kernel. Verify the fix with the
engine's own log lines:

- must be present: `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0`
- must be absent: `Using FlashInfer for top-p & top-k sampling.`

Declare the fix confirmed only after a **completed run** (ROWS=30, exit 0). Do
not declare it confirmed from elapsed clean minutes alone. Mean time to wedge
ranged from 3 to 68 minutes, so any config can look cured for an hour.

**What did not work.** Do not repeat these:

| Variable | Substitution | Result | Why it missed |
|---|---|---|---|
| Concurrency | 30 → 8 shards | wedged (once at 7) | reduces frequency, not the race |
| CUDA graphs | `FULL_AND_PIECEWISE` → `PIECEWISE` | wedged (verified active: `PIECEWISE=51`, no `FULL=`) | sampler runs outside the graph |
| Image | custom `vllm-x86_64-cu13` → official `v0.25.1` | wedged | both pin flashinfer ≤ 0.6.13 |
| Attention backend | FLASH_ATTN ↔ FLASHINFER | wedged under both | wrong subsystem |
| Quantization | NVFP4 → Marlin W4A16 | wedged (~15 min) | changes the GEMM, never the sampler. This is why the v3 checkpoint swap did not help |
| Async scheduling | on → `--no-async-scheduling` | wedged at 68 min | narrows the race window only. Kept as a free aggravator removal |

Mean time to wedge by config: NVFP4+async 7.5 min, W4A16+async 15 min, W4A16
no-async 68 min.

**Do not regress the vLLM version.** The fix is flashinfer 0.6.14, added by vLLM
PR #47669, merged 2026-07-16. That is two days after `v0.25.1` was cut, so no
release tag contains the fix. `v0.25.1` pins 0.6.13. `v0.24.0` pins 0.6.12 (same
bug) and also predates the SM120 startup fix (#47164), so it may not boot here.
If the sampler fix ever fails, escalate as follows: run
`pip install flashinfer-python==0.6.14 flashinfer-cubin==0.6.14`, then build a
**pinned** `nightly-<sha>` image. `--enforce-eager` is not the answer. It is
ineffective per vllm#49203 and costs 20-40% throughput.

`vllm#48718`, long recorded here as the open NVFP4 suspect, was **closed
2026-07-19**. The cause was flashinfer, not quantization.

**Resolved under contract v4 (2026-07-22).** The `nightly` image (engine
`v0.23.1rc1.dev1373+g387189c42`) ships flashinfer 0.6.14. rrsmoke4 ran the full
10-shard saturation profile with `Using FlashInfer for top-p & top-k sampling.`
ACTIVE. It produced zero wedges, 10 of 10 shards clean. The compose file now
retires all three workarounds (`VLLM_USE_FLASHINFER_SAMPLER=0`,
`cudagraph_mode=PIECEWISE`, `--no-async-scheduling`) as comments. Everything
above stays as the recovery playbook if the signature ever reappears on a future
image.

**Open question (v1-era, may not apply to current serving).** The contract-v1
gemma hang (2026-07-19, 727.8 to 0.0 tok/s, 100% GPU, no error) may have been
this same bug, blamed at the time on CUDA graph deadlock and "fixed" by an image
swap. Gemma 4's `generation_config.json` also sets `top_k=64`, so flashinfer
top-k was active there too, but a different image build also bundles a different
flashinfer version, which equally explains the cure. v1 results are unaffected
either way. To settle it, check the flashinfer version inside `gemma4-cu130`, or
grep archived v1 logs for `Using FlashInfer for top-p & top-k sampling.`

### A judge that generates to the cap and returns an empty string

**Symptom.** Judge requests never complete. The engine looks healthy:
`Running: 16 reqs`, `Waiting: 0 reqs`, `Avg generation throughput: 265 tokens/s`,
GPU KV cache usage climbing 69.7% to 75.8%, `/health` 200. The resumable
checkpoint stops growing. `vllm:request_success_total` stays flat for hours while
`vllm:generation_tokens_total` keeps rising.

**Detector.** One request, one trivial prompt, against the idle server:

```bash
curl -s http://localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":64,"temperature":0}'
```

A healthy server answers in under a second. The broken local judge returned
`finish_reason: length`, `completion_tokens: 64`, empty `content` and empty
`reasoning_content`. At `max_tokens: 2048` it burned all 2,048 tokens the same
way. With `enable_thinking: false` it returned repeated Tamil glyphs
(`ண…ட…ம…`) instead of an empty string. Temperature 0 gives the same result, so
this is neither sampling nor the chat template. The model emits noise for every
input.

**Distinguish it from the flashinfer wedge.** That one shows
`Avg generation throughput: 0.0 tokens/s` at ~64 W. This one generates at full
rate and draws ~108 W. Both keep `/health` at 200, so only the single-request
probe separates them.

**Cause: not established.** The local judge served 5,035 successful completions
first, then degraded. It broke between 17:45 and 18:50 UTC on 2026-08-05, after
an episode of 4,097 preemptions at 97-98% KV cache saturation under 40 judge
workers. `--kv-cache-dtype fp8` is the suspect, because a preemption discards and
recomputes KV blocks, but this is untested.

**Fix.** Restart the container. Do not lower `max_tokens` first. The cap is not
the cause, and a lower cap only makes each empty result arrive sooner.

**A degraded judge does write rows, by two paths.** Empty content makes
`_extract_json_from_content` raise ValueError
(`benchmark/penalty_judge_eval/llm_request.py:104-140`, called at line 266).
Tenacity retries `RETRY_TIMES` times (40, from `answer_env.sh:74`), re-raises
(`llm_request.py:196-202`), `evaluate_question_with_llm` returns None
(`eval_scoring.py:690-692`), `Evaluate_Single_Question` returns
`build_rule_based_result` (`eval_scoring.py:731`), and `score_resumable.judge()`
writes that rule-based row with a valid fingerprint (`score_resumable.py:350-355`).
Timing protected this wave: each degraded call spends 40 retries at ~60 s, roughly
an hour per question, and the run stopped inside that window. The second path:
under `MEMCONFLICT_JSON_MODE=1` (`answer_env.sh:137`) decoding follows a guided
JSON grammar, so a degraded model can emit garbage that still parses.
`parse_llm_metric_result` then defaults every missing metric to 0 and stamps
`Judge_Method: "llm_judge"` (`eval_scoring.py:624-647`), a row indistinguishable
from a real verdict by sampling.

**Check every row, not a sample.** The degradation starts mid-file, so reading
the first, middle, and last row misses it by construction. Scan the whole
checkpoint for the `Judge_Method` histogram, empty reasoning text, and reasoning
whose non-ASCII fraction is above 1%:

```bash
python3 - mnemosyne/Scores/v5ftc_gj12pen_judged_checkpoint.jsonl <<'PY'
import json, sys
from collections import Counter
for path in sys.argv[1:]:
    hist, flagged = Counter(), []
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        j = json.loads(line)["judged"]
        hist[j.get("Judge_Method")] += 1
        text = (j.get("Reasoning") or "").strip()
        if j.get("Judge_Method") == "llm_judge" and not text:
            flagged.append((i, "empty reasoning"))
        elif text and sum(ord(c) > 127 for c in text) / len(text) > 0.01:
            flagged.append((i, "non-ASCII over 1%"))
    print(path, dict(hist), flagged)
PY
```

**Audit result, 2026-08-05.** This judge wrote eight non-empty checkpoints that
day, 5,044 rows:

| checkpoint | rows |
|---|---|
| `mnemosyne/Scores/v5ftc_gj12pen_judged_checkpoint.jsonl` | 3,750 |
| `hindsight/Scores/v5ftc086_rest_gj12pen_judged_checkpoint.jsonl` | 686 |
| `hindsight/Scores/v5ftc086_p{0,1,10,11,12,13}_gj12pen_judged_checkpoint.jsonl` | 608 |

Ten of the 5,044 rows are `missing_answer`, which returns before the judge call
(`eval_scoring.py:717`). That leaves 5,034 judge calls, consistent with the 5,035
successful completions above. The per-persona wave stopped at 14:03 EDT, so `p13`
holds 33 rows rather than a full persona and `p14` is 0 bytes.

The full scan found zero rule-based rows. Eleven Mnemosyne rows and two Hindsight
rows trip the non-ASCII flag at 0.010 to 0.018, but every one is coherent English
whose short reasoning string holds an en-dash or a currency sign, so read the
flagged text before acting on the flag. Exactly one row is degraded: an
`llm_judge` verdict with empty reasoning and all-zero metrics at line 1287 of
`mnemosyne/Scores/v5ftc_gj12pen_judged_checkpoint.jsonl`, a `dynamic_conflict`
question that spent 2,382 output tokens. One row of 2,946 dynamic questions moves
dynamic answer accuracy by 0.0003 and macro answer accuracy by 0.0001, so the
banked Mnemosyne `v5ftc` number stands. Every other file is clean.

### Apply serving changes only from a quiet server

**Symptom.** 22 of 30 shards died with `APIConnectionError` immediately after the
team applied the correct flashinfer fix. Zero rows banked.

**Cause.** From a shard's view, the server was unusable across one continuous
~8-minute gap made of three stacked events: wedge, then watchdog auto-restart,
then operator `--force-recreate`. Each event alone fits inside the retry budget.
Clustered together, they exceed it.

**Fix.** Set `RETRY_TIMES=40` (~13 min) in `answer_env.sh`. This size covers the
worst realistic *cluster* of outages, not just one clean restart. Do not raise it
further: the retry window is also how long a fatal misconfiguration stays
invisible. Operating rule: never apply a serving change during recovery.

A restart costs ~8 minutes, not ~4. `llm_request`'s 300s timeout must expire on
each in-flight dead socket first. Do not lower that timeout. At 27-way
concurrency a legitimate 16k-token thinking answer can exceed 200s.

### `docker compose restart` does not apply compose file changes

This command reuses the existing container spec. It never re-reads the compose
file, but it still reports "healthy". Use
`docker compose up -d --force-recreate <svc>` instead. Then verify the flag
landed by reading the engine's own startup "non-default args" log line. Related
issue: `cd benchmark/docker` fails silently when the shell is already there. This
once produced a "verification" that pointed at the stale container.

### A soak test must reproduce the real generation profile

A 30-minute soak at 8 workers, 2048-token cap, one repeated prompt passed with
zero wedges. The real workload (27-way, 16384 cap, diverse prompts) wedged within
minutes. An unrepresentative soak is worse than none, because it creates false
confidence.

### Startup and capacity failures (v1-era, may not apply to current serving)

These three were seen only on the contract-v1 gemma stack. `--gpu-memory-utilization`
was raised for gemma's heterogeneous KV heads and does not bind the qwen
checkpoints. `VLLM_WSL2_ENABLE_PIN_MEMORY=1` stays required on this WSL2 host or
the runner dies at GPU worker init with `RuntimeError: UVA is not available`. The
disk-exhaustion recovery (Docker Desktop refuses to start, both vLLM endpoints
HTTP 000, VRAM falls to 1709 MiB) is: kill Docker, `wsl --shutdown`, restart,
prune build cache. The 178GB VHDX does not shrink after internal cleanup.

---

## Silent success, the recurring failure mode

This project has seven separate instances of this pattern. Each one showed up as
the **absence** of a symptom, never as an error message.

| # | What reported success | Actual state | Fix |
|---|---|---|---|
| 1 | Shard-wait loop logged "COMPLETED WITH FAILURES", returned 0 | 0 rows, all 30 shards dead, exit 0 | `do_generate` returns 1 on any shard failure or short row count |
| 2 | Manifest declared the run's decoding config | Hardcoded contract-v1 values into every v2 manifest | `_canonical_config()` is now a live `os.environ` read. Unset vars record as `null` |
| 3 | Adapters exited 0 on fatal errors | Partial/empty results scored as if complete | All six adapters `raise SystemExit(0 if ok else 1)`. `benchmark/preflight_rows.py`'s gate refuses to spend judge GPU on a truncated file |
| 4 | Mnemosyne auto-sleep fired 290 times | Did nothing, both layers reported success | See below |
| 5 | `docker compose restart` reported healthy | Ran the old container spec | `--force-recreate` plus verify the engine log |
| 6 | `/proc` entry count showed 37 → 35 processes | Real shard count fell 30 → 18 | Count `python*eval_*.py*` cmdlines, not `/proc` entries |
| 7 | `mnemosyne/Results/shards/<tag>/shard_*.jsonl` empty | Rows go to a container-local path | See below |

**Rule: verify by what a thing DID.** Check counters, status fields, and explicit
log lines. Never verify by the absence of an error.

### Mnemosyne auto-sleep fired 290 times with zero effect

**Symptom.** 290 consolidation ticks fired 0 sleeps. After a partial fix, 159
ticks fired, but every one returned `status=no_op`, 0 summaries, 0 proposals.

**Cause, layer 1.** The harness inflates `MNEMOSYNE_WM_TTL_HOURS` to ~1000 years
so that backdated 2022 rows survive the working-memory TTL. But the plugin
derives its consolidation eligibility cutoff from that *same* knob (`now -
TTL/2`), putting the cutoff at year ~1526. Nothing was ever eligible. A code
comment asserted the opposite, so the code read as correct on review.

**Cause, layer 2.** Even after fixing layer 1, `sleep_all_sessions` independently
recomputes the same cutoff from the same inflated constant (`beam.py:8369`).

**Fix.** Derive the cutoff from the plugin's real 168h default, and call with
`force=True`. This fix was verified: 158 sleeps fired, all `status=consolidated`,
158 summaries, 552 model-refresh proposals, **276 applied**, exit 0.
`mr_applied > 0` also confirms that `MNEMOSYNE_LLM_MAX_TOKENS >= 2048` is
genuinely in effect.

Without this fix, the auto-sleep arm would have produced a byte-equivalent
duplicate of the baseline run, and the project would have published it as a
consolidation result.

### Mnemosyne shard rows unreadable at the expected path

`entrypoint.mnemosyne.sh` uses two different directories that share a
`shards/$TAG` suffix: `LOGDIR="$RESDIR/shards/$TAG"` (bind-mounted, host-visible,
holds `shard_N.log`) and `SHARDDIR="$SCRATCH/shards/$TAG"` (container-local,
holds the `.jsonl` rows). Counting rows on the host path always returns 0. This
looks identical to "still ingesting" and to "everything died". Read the rows
with `docker exec <ctr> sh -c 'cat /scratch/shards/<tag>/shard_*.jsonl | wc -l'`.

---

## Methodology traps

### Scoring dies on a database the scorer never uses

**Symptom.** `STAGE=score` for hindsight fails in about 88 s with
`[hindsight] FATAL: cannot reach postgres at hindsight-pg:5432 (db='postgres')
after 10 tries: [Errno -2] Name or service not known`, then
`FATAL: per-run database creation failed`. Zero questions judged.

**Cause, two parts.** `docker-compose.yml` declares `depends_on: hindsight-pg`,
and `score_with_judge.sh` runs providers with `--no-deps`, which suppresses it.
That alone would be harmless, except `entrypoint.hindsight.sh:74-178` creates the
per-run database at TOP LEVEL, gated only on `HINDSIGHT_PG_MODE`. `shared` is the
only accepted value and every other value exits 2, so there is no env-level
escape. `STAGE` is not read until line 148, and only to decide whether an
EXISTING database is fatal. The score path is `run_stage` → `run_score` →
`score_resumable.py`, which touches no database and starts no daemon.

The gating is inconsistent across providers, so this is a missed gate and not a
design requirement:

| entrypoint | generate-time infra gated on STAGE? |
|---|---|
| supermemory | yes: `entrypoint.supermemory.sh:94,156` |
| retaindb-server | partly: clocksync at line 96, per-run DB NOT |
| hindsight | no |
| mem0 | qdrant wait not gated |
| mnemosyne | n/a, no infra |

**Fix.** Score through `benchmark/score_files.sh`, which takes result paths and
judge server details and never enters a provider entrypoint.

**Do not gate the entrypoints instead.** Scoring needs none of that
infrastructure, so a gate would only make an unnecessary code path fail more
quietly. The gating table above explains why the failure looks inconsistent
across providers. It is not a list of fixes to apply.

**What did not work.** Starting `hindsight-pg` first does clear the error, but it
keeps a Postgres dependency for a stage that never queries it. The driver already
does this for retaindb-server, which is why that provider did not fail.

### The judge's `max_tokens` is subtracted from its input budget

**Symptom.** Every judge call for one provider returns HTTP 400, and the scorer
logs `[DEBUG] LLM judge failed, fallback to rule-based scoring`. Measured on the
Honcho `v5ftc` wave: 3,057 rejections in 11 minutes.

**Cause.** vLLM refuses a request when prompt plus `max_tokens` exceeds the
served window. The harness asks for `max_tokens=16384` and the judge was served
at `--max-model-len 32768`, so the real input budget is 16,384 tokens, half the
window:

```
This model's maximum context length is 32768 tokens. However, you requested
16384 output tokens and your prompt contains at least 16385 input tokens,
for a total of at least 32769 tokens. (parameter=input_tokens, value=16385)
```

Only Honcho hits this, and the top-K slice is not the reason:
`build_llm_judge_prompt` (`benchmark/penalty_judge_eval/eval_scoring.py:434`)
is fed by `extract_top_k_retrieved_memories` at line 335 for every provider
alike, Honcho included. Honcho's individual items are simply about 5,000 tokens
each, because the plugin injects one markdown block split across named sections. Built through that
exact code path and tokenized with the served checkpoint's own `tokenizer.json`,
all 3,750 judge prompts measure: minimum 17,095 tokens, median 24,602, p90
27,612, p99 29,297, maximum 30,812. **100% of the wave exceeds the 16,384-token
budget**, so no part of it is judgeable at this serving. The size is by design:
`presets.sh:448` sets `HONCHO_CONTEXT_TOKENS=32768`, a 131,072-character cap
applied by `truncate_items_to_budget` (`honcho/eval_honcho.py:270-289`, called at
line 960). The stored retrieval is therefore the block the answer model actually
received, so widening the judge shows it no extra evidence.

**Two measurement traps here.** The shipped `tokenizer.json` carries an embedded
truncation config that clips every encode to exactly 16,384 ids, which reports
every prompt as 16,404 tokens until truncation is disabled. And estimating tokens
as characters/4 OVER-states gemma tokens on this markdown by 12-14% (measured
chars-per-token is 4.56), so a window sized from a chars/4 estimate has more
margin than it appears, not less.

**Lowering `max_tokens` does not fix it.** At 8,192, input headroom becomes
24,576 and 1,898 of the 3,750 prompts (50.6%) still reject. The cap is also not
free: the judge thinks, and the recorded `completion_tokens` includes the
reasoning trace (`external/MemConflict/Evaluation/llm_request.py:250-251` counts
sampled tokens before the reasoning parser splits them). A full wave recorded a
maximum of 14,438 output tokens against stored verdicts of ~180 tokens, so an
8,192 cap would truncate real judge calls mid-thought, convert them to
rule-based fallbacks, and change judge behaviour for every wave.

**Why this is invisible until it bites.** The failure is per-provider, and the
rule-based fallback is cached by `score_resumable.py:350-355` with a valid
fingerprint, so `load_checkpoint` replays every one on the next resume. Deleting
the output rows is not enough: the checkpoint file itself must go. Gate on
`judge_methods` before banking: `grep -c rule_based <scores>.jsonl` must be 0.

**Two things zero `rule_based` cannot see**, both auditable without re-judging. A
question absent from the Results file never reaches the scorer and lands in no
category. `preflight_rows.py`'s dataset cross-check catches it only when the
persona slice is derivable from `NUM_PERSONAS` or `START_IDX`+`END_IDX`, and
`score_files.sh` sets neither, so a host re-score only warns
(`preflight_rows.py:332-338`). An empty `Model_Answer` is scored 0 with no LLM
call under `Judge_Method: "missing_answer"` (`eval_scoring.py:385-396`), a
separate counter. So audit two checks per summary: total judged == 3,750, and
`missing_answer` accounted for. The five v5 waves read 3,750 each with 0 to 6
missing answers.

**The first judge has the same arithmetic but does not hit it.** `bench_judge_env`
sets `OPENAI_MAX_TOKENS=16384` unconditionally (`answer_env.sh:154`) with no
window awareness. Contract v5 serves the qwen judge at 131,072, leaving 114,688
tokens of headroom, so the ~30k Honcho prompts fit with about 4x margin. The trap
goes live only if a featured Honcho wave is scored against a 32,768-token server.

**Fix.** Serve the judge at `--max-model-len 49152` for that wave, keeping
`max_tokens=16384` and sampling 1.0/0.95/64 identical. 49,152 = 32,768 + 16,384
clears the worst measured prompt (30,812 + 16,384 = 47,196) with about 1,950
tokens spare, and the structural bound too: the plugin's 131,072-character budget
is about 28,700 gemma tokens at 4.56 chars/token, plus answer and rubric, roughly
31k. The judge model's `config.json` gives `max_position_embeddings = 262144` with
no `rope_scaling`, and vLLM applied none at boot, so 32,768 is an arbitrary
serve-time cap and raising it changes no positional encoding within the first
32,768 positions. That is why the other waves need no re-score.

**Drop the workers to 12.** The KV pool is a fixed token budget (350,811 tokens
on the VM judge) and the window raise barely moves it, since the activation peak
scales with `--max-num-batched-tokens`, not the window. Each Honcho request holds
prompt plus completion, roughly 26,000-36,000 tokens, so the pool sustains 10-13
concurrent. 96 workers would queue about 84 of them, multiply per-call wall time,
and drive calls into the 600 s `MEMCONFLICT_REQUEST_TIMEOUT`, the
timeout-retry-more-load cascade described at `answer_env.sh:182-193`.

### 96 judge workers exhaust the container's 1024 file descriptors

**Symptom.** Scoring runs fast, then hangs. The scorer container sits at ~1% CPU,
the judge sits idle with no requests running, and the checkpoint stops growing.
Everything looks healthy from outside: the judge answers a trivial prompt in
0.49 s. Measured on the Hindsight `v5ftcall` wave: 93 questions/minute for the
first 1,100 questions, then nothing.

**Cause.** The image ships the Docker default `ulimit -n 1024`. `_get_client`
(`benchmark/penalty_judge_eval/llm_request.py:73`, reached through
`llm_reasoning.py:163`) constructs a NEW OpenAI client per call, and each one
builds its own httpx connection pool and SSL context. At 96 workers the socket
count reaches the limit and every later call raises:

```
OSError: [Errno 24] Too many open files
  File ".../httpx/_config.py", line 40, in create_ssl_context
```

`evaluate_question_with_llm` treats that like any judge failure and returns a
rule-based verdict, which `score_resumable.py` then caches with a valid
fingerprint. So the failure both stalls the wave AND writes results that leave
the penalty rubric.

**Fix.** `--ulimit nofile=65536:65536` on the scorer's `docker run`. Confirm it
took effect with `docker exec <container> sh -c 'ulimit -n'`. Lowering the worker
count also avoids it but costs throughput for no reason.

**Repair, do not restart from zero.** Drop only the poisoned rows and resume.
`score_resumable.py` re-judges whatever is missing. On this wave 1 row of 1,148
was rule-based, so 1,147 good judgements were kept:

```bash
python3 - <<'PY'
import json
ck = "hindsight/Scores/v5ftcall_gj12pen_judged_checkpoint.jsonl"
keep = [l for l in open(ck, encoding="utf-8")
        if l.strip() and json.loads(l)["judged"].get("Judge_Method") != "rule_based"]
open(ck, "w", encoding="utf-8").writelines(keep)
PY
```

Filter on the field, not on the substring. A rule-based row always contains the
string `rule_based` (`eval_scoring.py:425`, plus `Error_Tags:
["rule_based_fallback"]` at line 429), so the substring test has no false
negatives, but a judge's free-text reasoning can contain that string and cost a
good row. `missing_answer` rows contain neither and are kept, which is correct:
they depend only on the Results row's `Model_Answer` (`eval_scoring.py:717`), not
on judge health.

**Gate every wave on it.** Count rule-based rows in the checkpoint after each
wave, not only on the final scores file. A scores JSONL holds one line per
persona, at most 30, so `grep -c rule_based` on a scores file is a valid
zero/non-zero test and not a count of affected questions. Two unrelated causes
have now produced cached rule-based verdicts in one evening, this one and the
judge window entry above, and neither announced itself in the run's exit code.

### A judge sampling default silently changes what a score means

**Symptom.** None at run time. The only trace is the `[answer_env] SCORE ...`
line reading `temp=0.6` where the rest of the arm reads `temp=1.0`.

**Cause.** `bench_judge_env` defaults `BENCH_JUDGE_TEMPERATURE` to 0.6 and
`BENCH_JUDGE_TOP_K` to 20, the qwen3.5-4b contract. A caller that does not pass
the overrides inherits them. `score_with_judge.sh` passes 1.0 / 0.95 / 64 for the
gemma judge. The first version of `score_files.sh` passed nothing, so hindsight
judged 61 questions under different sampling than the four providers it would be
compared against.

**Fix.** `score_files.sh` requires `--temperature`, `--top_p`, and `--top_k`, and
exits 2 rather than fall back. It logs the sampling on every run. The tainted
checkpoint was deleted and the provider re-judged from zero.

**Detection.** Compare the `temp=` field in the `[answer_env] SCORE` line across
every provider in an arm. They must match. The manifest records the judge
decoding too, but only after the stage has already run.

### A capped smoke cannot produce a meaningful score

**Symptom.** `NUM_PERSONAS=1 MAX_SESSIONS=6` exits 0 with every metric at 0.000.

**This is correct, not a defect.** Persona 0's only in-range questions sit in
session 5, and all of them are `dynamic_conflict` ("how did X change?"). Session
5 holds only the *before* state. The *after* state ingests in session 6. The
adapter ingests session N, then answers session N's questions, so at answer time
the store genuinely contains no evidence of change. Raising `MAX_SESSIONS` does
not fix this: only questions whose evidence arrives before the question (session
9 or later, for this persona) can score.

**Judge a smoke on plumbing only.** Check that rows are present, that questions
attempted is greater than 0, that answers are non-empty, that `Retrieved_Memories`
is populated with memory/created_at/score, and that provider ingest counters look
sane. Never judge a smoke on AA. This rule applies to every provider. A smoke
also never validates scaling: RetainDB Local's 6-session smoke ran fast and
predicted nothing, because at ~540 memories the quadratic search term is
invisible and at ~4,700 it dominates.

### Sharded merges must be numeric, not lexicographic

`cat ..._s*.jsonl` gives the order s0, s1, s10..s14, s2, which scrambles persona
order. Use a numeric loop, and use `>` not `>>`, because a second append silently
duplicates every row:

```bash
for i in $(seq 0 14); do cat <provider>_results_<tag>_s$i.jsonl; done > <provider>_results_<tag>.jsonl
```

### The persona-pool supervisor has no resume logic, kept container names are the resume

**Symptom.** A relaunched wave logs `LAUNCH FAILED (stale container name)` for
personas that already finished.

**Cause and fix, together.** The supervisor keeps no state, so a relaunch tries
every persona in the range. Docker refuses the name of a kept exited container,
so a completed persona is skipped and the line is cosmetic. That is the resume
path: do not remove the exited containers to "clean up" before a relaunch.

**Kill the supervisor BEFORE shedding containers.** It backfills any freed slot,
so removing containers under a live supervisor relaunches those personas.

**Measured pool sizing for the two Hindsight arms.** Concurrency 7 fits about
30 GB of system RAM for the minimal arm. The pg0 featured arm budgets about
2.5 GB per container, which is why its launcher pool defaults to 4.

**A pg0 shard's steady-state RSS is about 1.1 GiB, not the 2.5 GB budget above**
(measured 2026-08-05 on a 49.37 GiB VM over the whole `v5ftcall` wave, sampler
every 120 s). At concurrency 15 the total sat 15.0 to 16.0 GiB for 5.5 hours,
median 15.43 GiB, so 1.03 to 1.07 GiB per shard; the last shard running alone
read 1.106 to 1.113 GiB. Growth is a startup transient, not a trend: the first
8 minutes at concurrency 30 rose 29.72 to 30.12 GiB in decelerating steps of
0.16, 0.10, 0.10, then 0.04 GiB per sample.

**Do not extrapolate those first minutes linearly.** Doing that during the
`v5ftcall` launch projected about 55 GiB against 49.37 GiB of RAM and killed a
healthy 30-wide pool to restart it at 15. The measured plateau says 30-wide would
have used about 32 GiB and finished in roughly half the 7 h 26 m the 15-wide wave
took. Size from a plateau, or from the 2.5 GB budget as a ceiling, and treat the
first ten minutes as warm-up.

### Unfiltered recall can exceed Hindsight's 600-second Postgres statement timeout

**Symptom.** A shard exits 1 mid-wave with
`ServiceException: (500) ... {"detail":"Failed to search memories
(QueryCanceledError): QueryCanceledError('canceling statement due to statement
timeout')"}`, raised from `Search_Hindsight_For_Question`
(`hindsight/eval_hindsight.py:1092`) through the client's `recall_memories`.

**Cause.** `hindsight_api/config.py:1191` sets
`DEFAULT_DB_STATEMENT_TIMEOUT = 600` seconds and
`engine/memory_engine.py:3151` applies it as `SET statement_timeout` on every
pool connection. 600 s is not a tight limit, and one recall query still ran past
it.

**When it happens.** Seen once in 30 personas on the `v5ftcall` arm
(`RECALL_TYPES=all`, 2026-08-05, persona 26), and never on the `v5ftc086` arm
(`RECALL_TYPES=observation`) over the same 30 personas. Over all 3,750 questions
in each wave, unfiltered recall returns a median of 126 memories per question
against 115, and a mean of 114.5 against 111.8, about 10% more by median, 2.4% by
mean. So it scans somewhat more rows, but that difference alone does not explain a
query running past 600 s.

**Contention matters more than store size.** The identical persona passed on a
near-solo retry with no configuration change, so treat the timeout as
contention-dependent rather than as a fixed property of unfiltered recall.

**Fix, and why we did not apply it.** The timeout is a vendor-exposed knob,
`HINDSIGHT_API_DB_STATEMENT_TIMEOUT`, which the harness does not currently set.
The persona was relaunched UNCHANGED instead. 600 s is the vendor default, so it
is a vendor-endorsed value under ruling 2, and evenness across the 30 personas
outranks rescuing one query. Raising it for all 30 from the start would have meant
re-running the whole arm over one failure in thirty. Report the failure with the
arm.

### An alternate judge rubric outside 0.0 to 1.0 is destroyed by two parsers

**Symptom.** The alternate arm runs, exits 0, and produces numbers identical to
or better than the standard arm. Nothing warns, and no count of the new score
value appears anywhere.

**Cause, two independent parsers in
`external/MemConflict/Evaluation/eval_scoring.py`.**
`parse_trinary_score_value` floors every value below 0.25 to 0.0, so a
judge-returned −1 becomes an abstention on the dynamic and static metrics.
`conditional_answer_accuracy` is not in `PARTIAL_CREDIT_BLACK_BOX_METRICS`, so it
routes through `parse_binary_value`, where `numeric != 0` returns 1 and a −1
becomes full credit, so the sign inverts. The first parser hides the value, the
second flips it. Either one alone makes the arm measure nothing real.

**Fix.** Patch both in the generated copy.
`benchmark/make_penalty_judge_evaldir.py` does this for the `_gj12pen` arm. Then
confirm on the LIVE checkpoint that the new value appears for all three conflict
types before spending judge GPU on a full wave. A per-type count of the new value
is the only proof. A completed run and a plausible mean are not.

### Score stage needs `NUM_PERSONAS` passed explicitly

The `retaindb`/`hindsight`/`mem0`/`supermemory` compose services default to
`NUM_PERSONAS: 1`. With that default, a merged 30-persona file trips
`preflight_rows.py`'s "more personas than expected" branch and fails the gate.
This is the gate working correctly. Do **not** reach for `SKIP_ROW_GATE=1` to
route around it. Pass `-e NUM_PERSONAS=30` instead. Host-side runs need this
setting too, or the check degrades to warn-only and skips the dataset
cross-check.

### A retry-backoff sleep raised OSError under clock-sync, killing one shard

**Symptom.** One persona shard exited 1 during the generate stage. The
persona-pool supervisor logged `persona 1 FAILED: mem0_v5ftc_p1 exited 1 --
pool continues`. The other shards kept running.

```
Error processing mem0 evaluation: [Errno 22] Invalid argument
  File "/app/benchmark/eval_common.py", line 919, in run_eval
  File "/app/benchmark/eval_common.py", line 796, in Generate_Single_Persona_Eval
  File "/app/benchmark/eval_common.py", line 596, in Answer_Questions_For_One_Session
  File "/app/benchmark/eval_common.py", line 246, in Generate_Answer_With_Retrieved_Memory
    answer_text, cost_info = llm_request(
  File "tenacity/__init__.py", line 480, in __call__
    self.sleep(do)
  File "tenacity/nap.py", line 31, in sleep
    time.sleep(seconds)
OSError: [Errno 22] Invalid argument
```

Seen 2026-08-03 09:49 UTC on the mem0 featured wave, run tag `v5ftc`, persona 1,
session 20 of 51. Contract v5, `BENCH_CLOCKSYNC=1`. All four traceback frames sit
in the shared `benchmark/eval_common.py`, not in mem0-specific code.

**Cause, INFERRED, not proven.** `time.sleep` raises `OSError: [Errno 22]` on a
negative duration. The failure sits inside tenacity's retry backoff in
`llm_request` (`eval_common.py:246`), so the computed retry wait was negative.
Under `BENCH_CLOCKSYNC=1`, libfaketime steps the container clock to each session's
dataset date, so a clock step during a pending retry wait is the plausible source
of a negative interval. Not reproduced. The container was removed during
recovery, so its logs did not survive to confirm this.

**Frequency.** One occurrence across 60 persona containers: the 30-persona
Hindsight `v5ftc086` wave plus the 30-persona mem0 `v5ftc` wave. Every other
container logged zero.

**Fix.** Relaunch the failed persona as its own container, after freeing the name.
The supervisor's own failure message names this path:

```
docker rm mem0_v5ftc_p1
docker compose run -d --name mem0_v5ftc_p1 -e RUN_TAG=v5ftc_p1 -e STAGE=generate \
  -e START_IDX=1 -e END_IDX=2 -e NUM_PERSONAS=30 -e PRESET=mem0_featured_clocksync mem0
```

The relaunched shard re-ingests persona 1 from scratch. Generate passes
`--reset_collection`, so the partly filled qdrant collection is cleared, not
appended to.

**What did not work / not done.** No code change and no tenacity pin. One event in
60 containers does not justify changing the shared harness mid-comparison, because
a harness change that moves one provider's numbers would invalidate the
comparison.

---

## Provider: Mnemosyne

| Symptom | Cause | Fix |
|---|---|---|
| SEH@3 drops to 0.002, AA to 0.02 after backdating | Backdated timestamps trip `_trim_working_memory()`'s 168h TTL, deleting ~99% of raw turns right after ingest | Raise `MNEMOSYNE_WM_TTL_HOURS` past the dataset time span before ingesting. The entrypoint now defaults it to 8760000 whenever `LIFECYCLE=1` / `USE_DATASET_TIME=1` / `PLUGIN_CONFIG!=off` |
| SEH@3 0.031, top-5 dominated by fact rows | `fact_recall` auto-enables when extraction is on and floods recall with ~10 lossy fact strings that outscore raw turns | Keep `MNEMOSYNE_FACT_RECALL_ENABLED=0` (entrypoint default). Note: even with it off, once `extract=True` has run, recall boosts source memories on fact matches while displaying raw content, invisible unless provenance fields are logged |
| SEH@3 ~0.4, ~30% of retrieved rows are summaries | `sleep()` writes lossy episodic summaries and `[MODEL_REFRESH_PROPOSAL]` rows into the recall path | Drop `sleep()` from the retirement path. Retire via `resolve_conflict()` plus `invalidate()`. If sleep is wanted, suppress with `MNEMOSYNE_EP_LIMIT=0` plus source filtering |
| Retirement picks the wrong side of a conflict | Winner selection by `updated_at` fails. All rows get `updated_at` ≈ import time regardless of dataset dates | Pick the winner by the backdated source-row timestamp |
| Sleep's model-refresh yields zero proposals, no error | `MNEMOSYNE_LLM_MAX_TOKENS=512` caps both per-message extraction *and* sleep's whole-session model-refresh JSON. 512 suits the former but truncates the latter mid-JSON | Raise to ≥ 2048. Entrypoint sets 3072 under `CANONICAL=1` / `PLUGIN_AUTO_SLEEP=1` |
| Model-refresh conflict-supersession rarely fires on small models | The gate requires confidence ≥ 0.98 **and** ≥ 3 cited evidence ids. Small models clear the confidence bar but rarely cite 3 | Lower `MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE` (entrypoint sets 2 under `CANONICAL=1`) |
| `403 Forbidden` loading the embedding model (bge-small under contract v4, `Alibaba-NLP/gte-modernbert-base` under v5) | `*.cdn.hf.co` unreachable, or Xet backend in use | Whitelist `us.aws.cdn.hf.co`, set `HF_HUB_DISABLE_XET=1`. Fallback `MNEMOSYNE_EMBEDDINGS_VIA_API=1` |
| `recall_enhanced` crashes | `Mnemosyne.recall()` forwards `None` scoring weights for non-"general" query intents | Open upstream bug. Affects no committed run (none used `MNEMOSYNE_ENHANCED_RECALL=1`) |
| Dimension-mismatch warning | Stale `HERMES_HOME`/`MNEMOSYNE_DATA_DIR` from a run at a different embedding dim | Use an isolated home per run. Contract v5 moved the shared embedder from 384 to 768 dims, so every v4 store is stale |
| Featured-arm recall sparse (~2 real memories/question) despite a 16-candidate over-fetch (ftsmoke_mn) | `sleep_model_refresh_proposal` bookkeeping rows semantically match user questions and rank into recall. 85% of over-fetch candidates in the smoke (113/132) were proposal rows that the prefetch overlay then drops. Distilled `sleep_consolidation`/canonical rows also get the plugin's 1.12 quality boost and outrank raw `[USER]` turns | Plugin-faithful product behavior, not an adapter bug. Do not engineer it away. Expect the featured number to lean on consolidation-summary quality. Report it as a finding |
| On the featured clock-sync arm, recall returns zero candidates and the answers read as refusals, 27 of 122 questions in the persona-27 smoke `ft27mn` | The shipped 168-hour working-memory TTL and the plugin's auto-sleep gate work against each other under a compressed session cadence. The trim deletes only unconsolidated rows (`beam.py:3836-3849`, `consolidated_at IS NULL`), but the gate needs `working_count > 50` accumulated across sessions, and the faked clock's median inter-session gap is 29 days, so each session's first write deletes the previous session's rows before the gate can fire. Measured in `ft27mn`: 2 auto-sleep invocations in 277 ticks, 12 episodic rows | `PLUGIN_SESSION_SLEEP=1` (`--plugin_session_sleep`), one forced drained `sleep(force=True)` per session before that session's questions, the vendor's own consolidation at a benchmark cadence (user ruling 2026-08-02, DECISIONS "Mnemosyne featured arm"). `ft27mn` → `ft27mn2`: questions with zero retrieval 27 → 4, recall candidates 1,421 → 1,871, mean fill 3.34 → 3.65 of 5, 51 of 51 session sleeps ran, 2,266 items consolidated, 0 unconsolidated rows left. Do NOT fix this by raising `MNEMOSYNE_WM_TTL_HOURS`: the entrypoint refuses an explicit TTL on this arm, because the shipped TTL is part of what the arm measures |

`MNEMOSYNE_ENHANCED_RECALL=1` alone is not a meaningful test. Its Weibull recency
stage operates on ingest wall-clock timestamps, which sit only seconds apart, not
on the dataset's multi-year timeline. Backdating (`--use_dataset_time`) is a
precondition for a meaningful test.

---

## Provider: Hindsight

| Symptom | Cause | Fix |
|---|---|---|
| pg0 reports every instance stopped, `uri=None`. Daemon dies with "Database URL is required for migrations" | `python:3.12-slim` has no real `/bin/kill`. pg0's `is_process_running` spawns a standalone `kill -0 <pid>`, which fails to spawn | Install `procps`, already in `Dockerfile.hindsight` |
| Embedded Postgres `initdb` refuses to run as root | Root container | Run as non-root (`bench` user with writable `$HOME`), baked into the Dockerfile |
| Consolidation 400s at every budget setting under an 8192 window | Consolidation renders token budgets as ~3× JSON (indented, with `source_memories`) on top of a ~2.7k-token fixed template. Measured 17.5-18.5k chars of real request body at budgets nominally summing to far less | `--max-model-len 32768`. At 8192, retain also lost ~7.6-11% of sessions to overflow |
| vLLM 400 "at least N input tokens" does not match the real prompt | The number is the *threshold* (window − max_tokens + 1), not actual size | Measure real prompts via Hindsight's `llm_requests` Postgres table |
| ~3% of retains repetition-loop to the token cap, truncate, fail JSON parse, retry | Shipped `HINDSIGHT_API_LLM_TEMPERATURE_RETAIN=0.1` combined with the unbounded `facts` array. 54.5% of retain GPU time wasted over 269 calls | Set `0.7` (see DECISIONS) plus a 4096 retain cap |
| Arm C silently runs the wrong ingestion mode | `--retain_granularity exchange_append` with `--recall_types` unset recalls all fact types and mislabels the arm | Adapter refuses: `SystemExit(2)`. Pass `RECALL_TYPES=observation`. `all` is an explicit opt-out, never for headline runs |
| Under `HINDSIGHT_PG_MODE=pg0` the daemon exits part-way through a persona | The daemon's idle checker compares `time.time()`, which libfaketime fakes. A faked forward jump of weeks between sessions reads as idleness | `HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=0`. `daemon.py:59` returns before the checker loop when the value is `<= 0`. The vendor docstring at `cli.py:20` says "default: 300" and is stale, the code default is already 0. Never declare this key in compose: `daemon_embed_manager.py:495` int-parses it, and the entrypoint's empty-var guard matches `HINDSIGHT_API_` only |
| Two `HINDSIGHT_PG_MODE=pg0` containers fight over one Postgres data directory | pg0 hardcodes `~/.pg0/instances/<name>/data`, and compose mounts the SHARED `hindsight_state` volume at `/home/bench`, which is `$HOME` | The entrypoint exports `HOME=/tmp/hs_home_${TAG}` for this arm, onto the container's own filesystem. `ALLOW_EXISTING_PG0=1` is required to relaunch onto an existing cluster |
| Recall returns items dated after the session that asked the question, even under clock-sync, 6 items from 2 memories in the persona-27 featured smoke `ft27hs2` | The deriver stamps a stated future plan with the PLANNED date rather than the date the user said it. "I start the new job in March" stored in January carries March. Recall then ranks and returns it on that date | Vendor extraction behaviour, so report it rather than filter it (ruling 3). The consequence to state with any clock-sync result: a small amount of post-question content reaches the answer model. Measured rate at persona 27 is 6 recall items across 122 questions |
| Near-duplicate items fill a recall result, and the vendor docstrings promise a diversity stage that removes them | The documented MMR (maximal marginal relevance) diversity stage does not exist. Two docstrings say "Apply MMR for diversity" (`hindsight_api/memory_engine.py:4342` and `:4649`) and no implementation is present anywhere in the package | No knob, because there is no stage to enable. Near-duplicate flooding at recall is unmitigated, report it (ruling 3) |
| `--prefer_observations` changes nothing on the featured arm | The flag is inert under `RECALL_TYPES=observation`. Its dedup guard runs only when the raw fact types `world` and `experience` are in the request (`hindsight_api/memory_engine.py:5196-5197`), and it only ever removes raw facts that duplicate an observation. It never deduplicates observations against each other | Keep the flag set for plugin fidelity and treat it as inert on this arm. Do not read a `prefer_observations` result as deduplicated |
| The same fact comes back several times in one recall result, each time in different wording | Near-duplicate observations are a property of the STORE, not of recall. Consolidation merges two observations only at cosine similarity ≥ 0.97 (`hindsight_api/config.py:1152-1156`), so paraphrases stay below the threshold and all persist | Vendor default, kept under ruling 2 (DECISIONS, "Hindsight featured ranking left at vendor defaults"). No recall-side setting removes rows that are already distinct in the store |

### The append path stamped every fact with the wall clock (0.8.4)

**Symptom.** The featured arm (`RETAIN_GRANULARITY=exchange_append`) stored 2026
dates on a 2022 dataset, and nothing reported an error. Run `ftclk1_p0`,
2026-07-31, exit 0: 2,381 retains succeeded, 0 failed, 53 sessions drained with 0
consolidation timeouts, `run_contract_missing_required` empty, strict gate passed.
Every `memory_units` row carried a real-clock `mentioned_at` (world 2,327 rows,
experience 9, observation 1,429), all 2026-07-30 22:47 to 2026-07-31 02:34, zero
2022 dates. Measured cost on persona 0 against the banked minimal run, same 122
questions: update order recognition 0.547 → 0.305, micro answer accuracy 0.475 →
0.344.

**Cause.** `hindsight_api/engine/retain/orchestrator.py`, version 0.8.4, read from
inside `memconflict-hindsight:latest` and from the extracted wheel:

| line | what it does |
|---|---|
| 824 | `if update_mode == "append" and effective_doc_id and is_first_batch:`, entered only for append |
| 852 | `contents_dicts = [{"content": json.dumps(_merged, …)}]`, the JSON-array merge collapses every item into one dict carrying only `context` and `tags`. **The incoming item's `event_date` is dropped here** |
| 2296 | `event_date_value = utcnow()`, `_build_contents` finds no `event_date` and falls back to the wall clock |

The merge runs whenever the retained content parses as a JSON array of objects.
The harness sends exactly that shape, so it always succeeds and the date is always
lost. `fact_extraction.py:1560-1562` and `:2304-2306` then set `mentioned_at` from
that value, and the same value anchors the extraction model's relative-date
resolution, which is why the few populated `occurred_start` values read 2026-07-24
and 2026-07-30.

**Ruled out, do not re-investigate.**

- **The adapter.** `eval_hindsight.py:1162` parses the session date, `:439`
  computes the advancing per-exchange value, `:257` `_retain_stable_append`
  passes it as `timestamp=`. The stored document text still holds
  `"timestamp": "2022-01-03T00:00:00+00:00"`, so the value was correct at the
  wire.
- **The client.** `hindsight_client.py` builds the item dict with
  `"timestamp": timestamp` and wraps it as `Timestamp(actual_instance=raw_ts)`.
  The minimal arm proves the same call delivers the date.
- **The server request mapping.** `api/http.py:6706-6726` maps `item.timestamp`
  to `content_dict["event_date"]` for every item, with no append special case.
- **Consolidation.** `consolidation/consolidator.py:521-543`
  `_aggregate_source_fields` propagates min `occurred_start`, max `occurred_end`,
  and max `mentioned_at`. Its wall-clock fallback at `:2329-2333` fires only when
  every source fact has a null `mentioned_at`, which was not the case.
  Consolidation carried forward dates that were already wrong. Separate property
  worth knowing: the update path at `:1874-1876` uses `GREATEST`, so a later
  `mentioned_at` cannot be lowered by an earlier source fact.

**The minimal arm is not affected.** Session granularity goes through
`_retain_one`, which sends no `document_id` and no `update_mode`, so it never
enters the branch at `:824`. The banked minimal run retrieved memories dated
`2022-02-25T00:00:00.010000+00:00`, with the same 10 ms per-fact spacing as the
featured run: same server code, different base.

**Fix, two parts, both needed.** Upstream PR **#2684** (0.8.5) restores
`event_date` through the merge. The image pins **0.8.6**, where the restored fix
reads at `orchestrator.py:982-983` and `:1009-1010`. Two caveats stay live in that
same file, which is why libfaketime stays too: issue **#3010** still collapses
every item's date to the first item's (`first.get("event_date")`), and the
`utcnow()` fallback has only moved, to `:2621`. The featured arm therefore runs
`HINDSIGHT_PG_MODE=pg0`, a per-container embedded cluster inside the faked clock
domain, so whatever the vendor still discards falls back to a date inside the
logical domain. See `docs/DECISIONS.md`, "Hindsight featured arm moves to embedded
pg0 under libfaketime".

**`pg_trgm` is present in pg0 0.15.0, so no migration fallback is needed.**
Recorded because the risk was real: hindsight's migrations create `vector` and
`pg_trgm`, and a bundled PostgreSQL without contrib would fail the boot. Measured
in the `ftsmk086_p0` smoke through `bench_hs_pg0_report`: `vector` 0.8.5,
`pg_trgm` 1.6, `plpgsql` 1.0.

**Severity outside this benchmark is small.** In ordinary Hermes use the
conversation happens now, so a wall-clock stamp matches reality. The defect only
appears when the retained content is backdated: importing history, replaying a
transcript, or running this benchmark. Append is also not configurable away:
`_resolve_retain_target` (`plugins/memory/hindsight/__init__.py:1181-1200`) probes
the live API and returns `(session_id, "append")` whenever the server supports it
(Hindsight 0.5.0 and later), and the plugin's config schema at `:992-1010`
exposes no key for it.

### Discriminating load latency from the engine wedge

Both problems look like a stalled server. Under load latency, throughput never
reaches 0.0 and power stays high. The wedge sits at 0.0 tok/s at idle power.

### Grammar enforcement is contract-dependent

Under contract v3 (qwen), the qwen3 parser closes an empty `<think>` block
immediately when thinking is off, so grammar engages from token 0. Extraction runs
`enable_thinking:false` with `HINDSIGHT_API_LLM_STRICT_SCHEMA=1`. Thinking-on
extraction cost 384s per session on Qwen, versus ~14s with thinking off. On
contract v1 (gemma, v1-era) the opposite held: `--reasoning-parser` combined with
`enable_thinking=false` silently disabled xgrammar structured output for the whole
generation (vllm#39130), which is why v1 extraction had to run thinking-ON and v1
judge JSON-mode was soft-enforced. Do not retro-rescore v1 files under v3 serving.

### `presence_penalty` cannot be a server-side default

vLLM 0.25.1's `get_diff_sampling_param` allowlists only `[repetition_penalty,
temperature, top_k, top_p, min_p, max_new_tokens]` (`vllm/config/model.py:1506`).
Provider-internal calls run the model card's set minus `presence_penalty`.
`--generation-config vllm` does not change that.

### `entrypoint.hindsight.sh` unsets empty `HINDSIGHT_API_*` vars

`HindsightConfig.from_env()` int-parses strings that are set but empty. Keep the
guard that unsets them.

### A reranker 429 or restart kills shards, the TEI client does not retry 4xx

**Symptom.** A shard exits 1 with `Failed to search memories ... TEI rerank
request failed: Client error '429 Too Many Requests'` (or
`All connection attempts failed`).

**Cause.** `hindsight_api`'s `RemoteTEICrossEncoder` retries only on
connect/timeout/5xx errors, 3 attempts, seconds apart. A 4xx error is instantly
fatal to the recall call, which is fatal to the shard. Measured 2026-07-22 on the
first GPU reranker smoke: a `--max-concurrent-requests 16` cap on
`hindsight-rerank` returned 429 under 10 shards times 8 client-concurrent, and
killed 2 shards. The reranker container recreate that fixed this killed a third
shard, because connection-refused outlasts the client's retry window.

**Fix.** Never cap TEI permits below worst-case shard concurrency (see the
per-item permit entry below for what "worst case" means, which is much larger than
the request count). Treat `hindsight-rerank` restarts like vllm-gen restarts: only
from a quiet server, never while shards are live. Dead shards relaunch safely with
fresh bank IDs plus `ALLOW_EXISTING_DB=1` (consolidation-off arms, recall is
bank-scoped).

### TEI permits are per ITEM, not per request, fast aligned shards 429 at the 512 default

**Symptom** (rrsmoke4, 2026-07-22, the first stress run on the contract-v4 serving
stack). One shard died on the fatal-4xx path above with **no** permit cap set. The
TEI log showed 13 `try_acquire_permit: no permits available` errors, all within
one millisecond (21:32:18.586Z).

**Cause.** TEI acquires its admission permit at finer-than-request granularity
(`core/src/infer.rs` `try_acquire_permit`), so one 128-document `/rerank` call
consumes ~128 of the default 512 permits. The default pool covers only ~4
concurrent full batches, not 512 requests. `hindsight_api` defaults
(`DEFAULT_RERANKER_TEI_BATCH_SIZE=128`, `DEFAULT_RERANKER_TEI_MAX_CONCURRENT=8`,
per shard) put worst-case in-flight items at `shards × 8 × 128`. The old serving
stack never hit this limit because slow generation staggered shard phases. The v4
stack runs ~2.4x faster, so it landed all 10 shards' answer, recall, and rerank
phases in the same instant (~525 in-flight items observed: 512 admitted plus 13
rejected).

**Fix.** Set `--max-concurrent-requests 4096` on `hindsight-rerank` (~32
concurrent full batches, ~8x the observed peak). Permits are a free semaphore for
admission control, not memory. `--max-batch-tokens` remains the VRAM bound. Two
rules follow: never cap permits below worst-case in-flight items, and raising the
cap above the default is free and correct for many-shard bursts. Verified: s4
relaunched clean, 0 permit errors.

---

## Provider: mem0

| Symptom | Cause | Fix |
|---|---|---|
| `import mem0` resolves to the wrong module | The provider folder `mem0/` collides with the pip package name. `eval_mem0.py` imports `from mem0 import Memory` at the top of the file, before any `sys.path` insert | Run `python mem0/eval_mem0.py` (sys.path[0] = `<repo>/mem0`, which has no `mem0` subpackage). Never `python -m mem0.eval_mem0` from the repo root, and never add the repo root to `sys.path` |
| stderr flooded with 403s | mem0 fires anonymous PostHog telemetry. The egress proxy returns 403 for it | `MEM0_TELEMETRY=False` (the adapter `setdefault`s it before importing mem0) |
| Sharded runs cannot share storage | mem0's embedded qdrant locks its on-disk path to one process | `MEM0_VECTOR_MODE=server` (compose default) points every shard at the shared `qdrant` service. Each shard gets its own collection `mem0_<sanitized RUN_TAG>`, and personas are isolated by `user_id` |
| Every embed call 400s against vllm-embed; ingest stores zero memories, then the first `search()` crashes the run | mem0ai 0.1.118's `OpenAIEmbedding.embed` (`mem0/embeddings/openai.py:46`) unconditionally sends `dimensions=<config>`. vLLM's pooling server rejects any explicit `dimensions` for a model with no matryoshka support, which contract v4's bge-small-en-v1.5 was, whatever the value. `add()` catches embed failures and only logs them, so the run dies later, at the first unguarded search | A monkeypatch in `eval_mem0.py` (right after `from mem0 import Memory`) re-issues `embeddings.create()` without the kwarg. Under contract v4 the result was numerically identical: bge-small's only output size is 384, the same as the configured dims. Seen 2026-07-22 (`mem0_itsmoke` exit 1). The patch stays under contract v5, where the embedder's native width is 1024 |
| Generate finishes (`Successfully processed`) but the container stays running forever | mem0ai/qdrant-client leave non-daemon threads running (connection/thread-pool workers). The interpreter never exits, so control never returns to the entrypoint | `eval_mem0.py` ends with a flush plus `os._exit(code)` instead of `SystemExit`. Output files are already closed by then. Seen 2026-07-22 (`mem0_itsmoke2` needed `docker stop`) |
| `ValueError: Top-level entity parameters ... are not supported` on `search()` (mem0ai 2.x) | 2.x moved entity scoping into a `filters` dict and dropped `limit=`. A 0.1.x-shaped `search(query, user_id=…, limit=…)` call raises before any retrieval | `search(query, filters={"user_id": …}, top_k=20, threshold=0.0)`. `top_k` replaces `limit`. `threshold=0.0` is deliberate (see DECISIONS) |
| Event split is 100% ADD, zero UPDATE/DELETE/NONE, looks like a broken decision path | Not broken: **mem0 2.0.14 is ADD-only by construction.** `memory/main.py:1165-1168` stamps `"event": "ADD"` as a string literal. `infer=True` makes ONE extraction call on a prompt reading "Your sole operation is ADD" (`configs/prompts.py:472`). The two-phase update machinery is dead code referenced only by its own `def`, and no flag revives it (`version` defaults to `"v1.1"` and feeds telemetry only) | Nothing to fix. Report it. Do not look for a wiring fault: the project checked and ruled out adapter counting, candidate-search scoping, parse-failure-defaults, embed truncation, and clocksync on 2026-07-28 (see BENCHMARK_MATRIX v4minc) |
| mem0's own `add()` docstring says `infer` decides add/update/delete | Upstream stale doc: `main.py:765-767` still describes the 0.1.x two-phase behaviour that 2.0.14 removed | Do not trust it. Read `main.py:1165-1168` instead. Same rule as this repo's own docs: a doc describing behaviour is not evidence the behaviour is live |
| A single bad extraction kills the shard (`LLMError`) instead of skipping (mem0ai 2.x) | 0.1.118 logged and swallowed extraction failures. 2.x raises `LLMError` out of `add()` | `eval_mem0.py` catches it per `add()` and counts it in `Total_Add_Calls_Failed`. A failed extraction is a data point, not a run abort |
| Every embed 400s again after the 2.x upgrade, even with the 0.1.x `dimensions` shim in place | 2.x batches embeddings through a separate `embed_batch` path that also sends `dimensions=` | The shim covers `embed_batch` as well as `embed` |
| Ingest silently goes to a remote endpoint; internal LLM calls never reach vllm-gen | 2.x `OpenAILLM` redirects `base_url` when `OPENROUTER_API_KEY` is present in the env | `eval_mem0.py` pops `OPENROUTER_API_KEY` before constructing `Memory` |
| Hybrid search degrades silently to semantic-only; stderr shows blocked downloads at first `add()` | 2.x wants `en_core_web_sm` (entity extraction) and a fastembed BM25 model at runtime. The egress proxy blocks both | Both are baked in at image build time in `Dockerfile.mem0` with `FASTEMBED_CACHE_PATH` pinned |
| `BENCH_CLOCKSYNC=1` exits before generate on mem0 | Fail-closed guard: clock-sync relies on 2.x resolving the extraction prompt's dates per `add()` from the process clock. 0.1.x does not do this, it freezes the dates into the prompt at import | Expected. Run clock-sync only on mem0ai 2.x. The old 0.1.118 monkeypatch is deleted, not disabled |
| `UserWarning: Qdrant client version 1.18.0 is incompatible with server version 1.12.4` every run | `qdrant-client==1.18.0` arrives transitively from `mem0ai==0.1.118`. `Dockerfile.mem0` pins the server (`qdrant/qdrant:v1.12.4`) but not the client | Was benign (add/search worked in `ftsmoke_m0`). Closed 2026-07-26: the server is now `qdrant/qdrant:v1.18.3`, matching what the 2.x client resolves. 2.x creates an IDF-modifier BM25 sparse-vector collection, so sparse support can no longer be left to a skewed server |

qdrant has no container healthcheck (its image ships no shell or curl), so
`entrypoint.mem0.sh` does a bounded qdrant-client readiness wait before generate.

---

## Provider: Supermemory

| Symptom | Cause | Fix |
|---|---|---|
| Recall returns nothing right after ingest | Ingestion is async: `POST /v3/documents` returns `queued`, and a memory is only searchable at `done` | The adapter drains via `wait_for_drain` after each session's ingest, before answering. This is the single biggest correctness requirement of the adapter |
| Cannot install the server | The binary ships only via GitHub Releases. Some environments block that egress (it built fine locally on 2026-07-22) | `Dockerfile.supermemory` bakes it in at build time with a pinned `SUPERMEMORY_SERVER_VERSION` (pinning also skips the rate-limited `api.github.com` "latest" lookup) |
| Every `POST /v3/documents` sits at `queued` forever; drain times out at 600s with zero extraction calls | Server versions 0.0.6 and 0.0.7-rc.2 ship a broken linux-x64 bundle: `Cannot find module '@rivetkit/rivetkit-wasm'` at boot (0.0.6 logs it, 0.0.7-rc.2 fails silently). The Rivet workflow engine that drives async ingest never initializes | Pin `SUPERMEMORY_SERVER_VERSION=0.0.5` (this version boots `workflow rivet`, and a doc reaches `done` in ~5s). Before bumping the version, verify a document reaches `done` on the new tag. We isolated this issue 2026-07-22 across all three versions |
| Extraction and answer/judge configs cross-contaminate | Both are `OPENAI_*`-shaped | `_supermemory_server.py` gives the spawned server its own `OPENAI_*` from `SUPERMEMORY_LLM_*` |
| Every document logs `memory agent failed (Nms)` at ~10-25 ms then `finalized: N chunks, 0 memories`; recall silently runs on chunk-RAG fallback (itsmoke2: 470/470 retrieved items were chunks) | The memory agent is a **tool-calling** agent (7 function tools, `tool_choice:auto`, max_tokens 12000). A vLLM server without `--enable-auto-tool-choice --tool-call-parser` returns 400 on the request before inference runs. Supermemory swallows the error into (disabled) telemetry, so the log shows only the fast failure. The signature is the ~10-25 ms timing plus a green `[Extraction] Extracted with text` line | vllm-gen runs `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (contract v4). The parser must be `qwen3_coder` for Qwen3.5: its chat template emits the coder-style `<function=...><parameter=...>` XML, while `hermes` expects JSON inside `<tool_call>` and would silently fail to parse. Verify with a probe: send a `tools` plus `tool_choice:"auto"` request and confirm a 200 response with a parsed `tool_calls` array (not tool-call text in `content`). After the fix (itsmoke3): 469/470 retrieved items were memory-type, and the agent completed 42/53 calls, 11/53 still fail (open) |
| New documents sit at `queued` forever; every session drain would time out at 600s; HTTP stays 200, crons report "0 stuck queued documents" | 0.0.5's Rivet workflow dispatcher died silently. It was alive through 18:41:59Z and confirmed dead by 21:49:48Z (~80 min after boot, coinciding with its last real document), with **zero** error or exception lines in 9,000 lines of server.log. Detection is behavioural only: watch for a doc stuck at `queued` past its normal ~5s processing time | Restart `supermemory-server`. A restart alone does NOT re-dispatch already-queued docs. The ~30-minute "stuck queued document retry" cron sweep does that instead (observed: 3 orphans rescued ~26 min post-restart). Operational rule for full runs: probe ingest liveness (send 1 tiny doc, confirm `done`) before each wave. A mid-run dispatcher death causes a drain timeout on every subsequent session. It is unknown whether 0.0.6+ fixes this (0.0.6 is unusable anyway, see the pin row) |
| `/v4/search` dies with repeated 120s `ReadTimeout`s while the SAME session's ingest had just drained normally, `server.log` floods with `[NODE-CRON] [WARN] missed execution at <date>!` walking forward in 30-min slots from the boot date, and if the run keeps going, server RSS goes from ~0.75 GB above a 1.2 GB baseline to over 20 GB in about 4 minutes and the host OOM-kills it. NOT the dispatcher-death row above: that one hangs ingest with a quiet log, this one hangs search, then the host, with a 30k-line log naming the cause | node-cron v4's missed-execution replay inside supermemory-server 0.0.5, proven on an EMPTY store (zero documents) by an isolation probe on 2026-07-29. `Runner.start()` replays missed slots in a synchronous while-loop that never yields; each iteration allocates about 0.46 MB (log strings re-reading `FAKETIME_NO_CACHE`, a `Date`, an un-awaited promise), and GC cannot run because the loop holds one stack frame throughout. Six crons register unconditionally: `retryStuckQueuedDocuments`, `retryStuckProcessingDocuments`, `retryStuckDreamingJobs` (all `*/30 * * * *`), `cleanupExpiredCache` and a telemetry heartbeat (`0 */6 * * *`), and a daily update check (`0 12 * * *`). The replayed executions do NO work (`onMissedExecution` is a no-op), so the cost is purely the loop. node-cron computes missed executions from the **wall clock**, which clocksync fakes (`FAKETIME_DONT_FAKE_MONOTONIC=1` keeps timers real), so the first real cron heartbeat after boot compares the faked clock and finds the whole faked span missing. Heartbeats are 30 REAL minutes apart; the first only seeds the tracking date, so the replay fires at the SECOND heartbeat, exactly the measured 35-54-minute death window. A persona's ~3-year fake span is about 163,000 slots across six crons, about 73 GB of allocation, more than any host absorbs. Replay size equals the fake-time span traveled since the previous heartbeat, so anchoring the dataset nearer real time does not help (span, not era, drives it), and spreading the travel across more heartbeats does not help either (same total span). No vendor knob exists: the full `SUPERMEMORY_*` env surface has nothing matching CRON/SCHEDULE/DISABLE/TASKS/SWEEP; `SUPERMEMORY_RUN_CRONS_AT_BOOT` only adds a boot-time run, `SUPERMEMORY_NO_UPDATE_CHECK` only gates the update check's network call, and `--help` is ignored (the binary boots the server regardless). Measured (`smk_sm_clk`, 2026-07-27): 29,596 slots in 649 s = 45.6/s, before the allocation mechanism was known; a full 2022-2025 replay is about 51,000 slots per cron, about 19 minutes | Retries alone do not fix this. `SUPERMEMORY_HTTP_RETRIES=30` (~75 min stall budget) covers ordinary transport blips, but it is what `smtest1g` (one quiet container, no pool-sizing confound) OOM-killed anyway. The fix (commit `241f71c`): under clock-sync the adapter respawns the spawned server once per session on the SAME data dir (`supermemory/eval_supermemory.py:_respawn_server`, `SupermemoryServer.start()` made re-callable). The driver steps the clock BEFORE each boot, so a server never observes a forward jump, the replay span is structurally zero, and a server also almost never lives the 30 real minutes a heartbeat needs. Knob: `SUPERMEMORY_RESPAWN_PER_SESSION` (default 1 under clock-sync, 0 reproduces the OOM, declared in `docker-compose.yml`). The Bun faketime honor probe still runs once, on the first boot only. Validated on `clkfix_p9` (persona 9, `supermemory_minimal_clocksync`, strict quality on, watermark 1gb): 54/54 documents, 113 questions, 0 drain timeouts, 0 failed, 0 dropped, 54 respawns, 44.3 min total, 0 missed-execution lines, where all four earlier clocksync attempts died with zero complete personas. Cost is a wash, not a tax: the no-clocksync discriminator `smnoclk_p9` (same config minus clock-sync, same day) took 45.2 min with 1 drain timeout. Two alternatives were verified but NOT taken, open for a user ruling (see DECISIONS): a one-byte patch of the vendor's replay guard, and compressing the fake timeline to a short span |
| `<data_dir>/server.log` looked unrecoverable across five OOM waves, blocking any direct evidence of the cron replay | Wrong assumption that `.supermemory_runs/` was not on the host bind mount | It is bind-mounted. The OOM run's log survived at `run_14b0fa90/server.log` (33,706 missed-execution lines, ends mid-replay). Under `SUPERMEMORY_RESPAWN_PER_SESSION`, `SupermemoryServer.start()` truncates the log on each boot, so only the newest boot's log persists per data dir, read it immediately after a failure, before the next respawn overwrites it |
| A clock-synced run banks a complete-looking result (`drained=True` on every session) whose memories were never actually ingested | A DROPPED document (one that never got a document id: empty content, a failed POST) never entered `doc_ids`, so it never reached `wait_for_drain`, and an all-dropped session short-circuited straight to `drained=True` at the empty-`doc_ids` branch. `SUPERMEMORY_STRICT_QUALITY` checked only `timed_out`/`failed`, not this path | Commit `241f71c` counts a DROPPED document as `Total_Drain_Dropped` and adds it to the strict-quality guard alongside `timed_out`/`failed`, so an all-dropped session now aborts the shard instead of banking a false-complete result. `SUPERMEMORY_STRICT_QUALITY=1` is now the `supermemory_minimal_clocksync` preset default (previously 0 under every minimal arm, 1 under featured only) |
| Host RAM exhausted and the SSD pegged at 100% within ~55 min of launching a 6-wide spawn-mode pool; `docker stats` itself times out at 120 s, the Docker engine 500s on `containers/json`, one persona exits with an unknown code | `SUPERMEMORY_EMBEDDING_RAM_LIMIT` is **misnamed**. It is not an embedding allocation. It is the ingest **backpressure watermark**. The server pauses ingest only when `RSS > boot_baseline + limit` (binary: `fK0()` compares `process.memoryUsage.rss()` against `iH + jW0()`). At the preset's `2gb`, each spawned server was free to grow to ~4 GiB before throttling. 6 × 4 GiB overruns a 24.8 GiB VM that already holds ~8.8 GiB of vLLM. The vendor default is 1gb. Our 8gb compose default exists for SHARED mode (one server), where it fixed 8-minute ingest idles (commit `0a3187c`) | Under spawn mode, set `SUPERMEMORY_EMBEDDING_RAM_LIMIT=512mb` and size the pool from a MEASURED single-persona RSS, not from a shared-mode number. Measured 2026-07-28 at 512mb: one persona plateaus at ~1.85 GiB, 5-wide runs at 16.0/24.84 GiB with 8.8 GiB free, and drains stay ~12 s. The preset's `bench_preset_set ... 2gb 8gb` correctly KEEPS an explicit override (logs `512mb KEPT`, explicit override), so pass it on the launch line rather than editing the preset |
| Reasoning about spawn-mode ingest parallelism from `SUPERMEMORY_INGEST_CONCURRENCY` | That var is declared ONLY on the two central `supermemory-server` services (compose lines 788, 857). The spawn-mode run-service does not declare it, so it never reaches a spawned server. Spawn mode has always run at the **vendor default of 2**, so setting it on a spawn launch line is a no-op | Do not credit or blame it for spawn-mode behaviour. If an explicit value is ever wanted there, add it to the run-service. Do this only between providers, never while containers are live (bind-mount plus `depends_on` recreate hazard) |
| Featured arm's `/v4/profile` search subject to the server's default threshold, unlike the `/v4/search` arm (forced 0.0) | `/v4/profile` is POST-only on 0.0.5 (GET 404s) and accepts no threshold/limit params of its own | Plugin-faithful property, not a bug. Disclose the asymmetry when reading featured-arm scores |
| Under `BENCH_CLOCKSYNC=1`, a persona aborts with `server died during drain … N document(s) already accepted, NOT re-submitting` followed by `STRICT_QUALITY … timed_out=True failed=0 dropped=0`. It struck 5 of 30 personas in `v4minc3` (17%), at session index 21, 22, 32, 41, and 47, so it tracks neither store size nor boot count | The spawned 0.0.5 server takes a **Bun segfault** while embedding a document that `/v3/documents` had already accepted, so the drain poll finds a dead process. `<data_dir>/server.log` ends in `panic(main thread): Segmentation fault at address 0x….46505845`. One crash site, not ASLR noise: across the two `v4minc2` kills and the four `v4minc3` mode-A kills the fault low word is `0x46505845` every time and the `bun.report` trace prefix is byte-identical. Every crash sits in a **2.39-2.49 GB RSS** band, which is the 2 GB PGlite WASM ceiling the binary declares (`maximum: 32768` pages) plus runtime overhead. An earlier reading of these crashes as a 0.59%-per-boot random hazard is superseded: the RSS band is far too tight for that | There is no vendor fix. Recover at the harness level and rerun the persona. Fail-fast (commit `58b4507`) turns the loss from ~15 minutes of futile retries into an immediate abort: `ServerDiedError` is raised on the first refused request when `self._proc.poll()` is non-None, and the adapter does NOT re-submit after acceptance, because that would duplicate the document. Strict quality then aborts the persona rather than answering against a store that lost a document. In `v4minc3` all 5 aborted personas passed on a rerun at the same tag and the same `code_sha`. Untried candidate, cheaper than a rerun: on a drain-phase death, respawn on the same data dir and RE-DRAIN the accepted document id, since the PGlite write is probably committed, so this would recover the persona instead of discarding ~50 minutes of work |
| A persona dies at recall time with `requests.exceptions.HTTPError: 500 Server Error … /v4/search`, mid-run, after ingest drained normally | `_supermemory_server.py` `_send` raises on any non-2xx and there is no 5xx retry on the recall path, so one server-side 500 kills the persona. In the `v4minc3` p18 case `server.log` showed an internal fetch returning a 169-byte 500 body; `vllm-embed` logged zero non-200 in the window while `vllm-gen` logged 44 400s in the same hour, which is the known stochastic memory-agent failure. So the likely chain is a search-time LLM call taking a vllm-gen 400 that Supermemory re-throws as a 500 | Rerun the persona. The p18 rerun passed the same session, so the 500 is load-induced, not deterministic for a given question. A 5xx retry on the recall path is still missing and would remove this as a persona-killer |
| Recall returns nothing on questions worded "recent…" while the same store answers other questions with a full inject. 29 of 122 questions in the contract-v5 featured smoke `ft27sm2`, all of them among the 31 questions worded that way | The vendor's **14-day recency window** on temporally worded queries, unchanged at contract v5 and on the `/v4/profile` featured path. It filters on the memory agent's extracted `temporalEventStartMs`, not on `created_at`, which is why clock-sync does not rescue it (full evidence in BENCHMARK_MATRIX, "Supermemory's 14-day window") | No vendor knob exists. Report it, and say what it costs: in `ft27sm2`, **27 of the 29 empty questions had their supporting evidence in the store**, and **22 of the 26 questions whose gold answer is a bare "Yes." returned empty**. The featured dynamic-conflict number therefore measures the query filter as much as retrieval |
| `/v4/profile` returns every fact in the `dynamic` section; `static` stays empty, on ANY store size (0 static facts across 122 ft27sm2 questions; 0 across four textbook identity facts in a controlled probe, 2026-08-02) | NOT consolidation-gated (an earlier version of this row said so; wrong). `static` is fed solely by the `addToStaticProfile` boolean the memory agent may pass to `CreateMemory`, and the SELF-HOSTED agent's system prompt never mentions the static profile or the flag, the cloud extraction prompt in the same binary carries a full placement section the self-hosted prompt lacks. An optional flag with no instruction is never set | Product property, report it: the featured arm's profile recall is the dynamic section only. (These are profile SECTIONS, unrelated to MemConflict's `dynamic`/`static` conflict types.) Related: the "dreaming" lane is ALSO unreachable in self-hosted 0.0.5, the `selfHosted` feature flag routes memory extraction inline at ingest, `dreamingJob` rows are never written, and `retryStuckDreamingJobs` is a permanent no-op ("Found 0 stale processing dreaming batches" on every heartbeat). `SUPERMEMORY_RUN_CRONS_AT_BOOT=true` verifiably runs all four retry sweeps at every boot, but they fire into an empty table. Cross-session reconciliation is NOT lost: the inline memory agent searches prior sessions and supersedes contradicted memories (`updates` relation, `isLatest:false`), verified by probe, clock-independent |
| Every document drains cleanly and reaches status `done`, but recall returns zero results and the store holds no memories. `SUPERMEMORY_STRICT_QUALITY=1` does not abort | `SUPERMEMORY_EMBEDDING_DIMENSIONS` does not match the embedder's real output width. The server reshapes both pgvector columns to the DECLARED width at boot, so the vector upsert then fails on every chunk. `server.log` shows `VectorDB upsert failed … Failed to upsert chunk embeddings` and `finalized: 1 chunks, 0 memories`. The workflow still marks the document `done`, so the drain succeeds and strict quality sees nothing wrong. Measured 2026-08-01 on 0.0.5: declared 1024 against a 384-wide embedder | Before each wave, read the boot log's `reshaped "chunk"."embedding" → vector(N)` lines and confirm N equals the embedder's real width (768 for contract v5's `gte-modernbert-base`). The persisted plan is at `<data_dir>/embedding-plan.json`. A width change needs a fresh `supermemory_data` volume, the store is dimension-bound |
| A document sits at status `embedding` forever and every drain times out. Same observable shape as the silent dispatcher death above | A bad or unknown `SUPERMEMORY_EMBEDDING_MODEL`. The embedding workflow step fails and rolls back, `server.log` shows `embeddings-batch-1 ✗ Rollback traversal halted`, which leaves the document permanently non-terminal. Found 2026-08-01 | Distinguish it from dispatcher death by the log line `[embeddings] primary failed after <=20000ms, falling back: The model '…' does not exist.` Dispatcher death logs nothing. Fix the model name to the served alias (`gte-modernbert-base` under contract v5) and restart |
| Under contract v5, a session-granularity document wedges at status `embedding` and the drain times out (`timed_out=True failed=0 dropped=0`); the very first session of the `ft27sm` featured smoke hit it. `server.log` shows `embeddings-batch-1 ✗ Step "embeddings-batch-1" failed (attempt 1)` then `napi run handler failed … internal_error`, no rollback line, no model-not-found line | **0.0.5's embedding workflow step has an internal payload cap of ~256 KiB.** The vendor batches 15 chunks per embedding call; one batch carrying more than ~12,288 floats fails the step. At 1024 dims (contract v5) that is 13+ chunks ≈ 9,500+ characters of dialogue, and 1,577 of the 1,579 dataset sessions (99.9%) are larger (median 17,377 chars), so session-granularity ingest cannot run at 1024 dims at all. At v4's 384 dims the same 15-chunk batch is 5,760 floats, far under the cap, which is why v4 never saw it. Reproduced 2026-08-01 with no clock-sync and no respawn; size sweep: 12×1024 `done`, 13×1024 wedged, 18×384 `done` (identical document, only the width changed). The upstream embed call itself SUCCEEDS (vllm-embed returns 200 in ~1 s); the failure is inside the vendor's workflow-step messaging. Shrinking the wire body by rounding floats did not help, the cap is on the step payload, not the HTTP response. No vendor knob: the `SUPERMEMORY_LOCAL_EMBEDDING_BATCH_SIZE` family applies only to the in-process Xenova embedder, not the `openai` provider path | A vendor product property under ruling 3, report it. Arm-posture options recorded in DECISIONS: exchange-granularity documents (1-2 chunks) verified to reach `done`; a labelled cadence deviation is the only way the arm runs at 1024 dims. The boot-time clock-sync honor probe is single-chunk and proves nothing about real session ingest, do not read `probe OK` as ingest health. Resolved 2026-08-02 by the contract v5 amendment: the shared embedder is now `gte-modernbert-base` at 768 dims, where a full 15-chunk batch is 11,520 floats and fits, so session granularity runs again. The 1024-dim numbers above stay as the evidence for the cap |
| A Supermemory shard exits 1 mid-run. The container log ends `[supermemory] server died during drain, persona <id> session <n>: supermemory server process exited during GET /v3/documents/<id> (ConnectionError); 1 document(s) already accepted, NOT re-submitting`, then `RuntimeError: STRICT_QUALITY: persona <id> session <n> ingest degraded (timed_out=True failed=0 dropped=0); aborting rather than answering against missing memories.` The pool supervisor logs `persona N FAILED: <container> exited 1 -- pool continues`. `docker inspect` shows `OOMKilled=false` (the host did not kill the container; the spawned server child process died) | Different presentation from the Bun-segfault row above (same 2.39-2.49 GB RSS ceiling): that one dies mid-embed. This one dies during the per-session document drain, under `BENCH_CLOCKSYNC=1`. Clock-sync forces spawn mode with a server respawn every session, because one shared server can hold only one perceived clock, this arm boots a new server about once per session, about 19 boots by session 17 on one persona. That respawn rate is a property of the clock-sync arm, not of a normal deployment, which runs one long-lived server. Measured on the contract-v5 featured wave `v5ftc`, pool 15: four personas lost in the first two and a half hours (9, 14, 12, and one more), at sessions 17, 24, and 37, the longer a persona runs, the more exposed it is | `SUPERMEMORY_STRICT_QUALITY=1` is working as intended: it aborts the persona rather than answer against a store it knows is incomplete. Keep it on, `SUPERMEMORY_STRICT_QUALITY=0` downgrades this to a warning and produces scoreable but wrong numbers. Relaunch the persona as its own container: `docker rm sm_v5ftc_p<i>` then `docker compose run -d --no-deps --name sm_v5ftc_p<i> -e RUN_TAG=v5ftc_p<i> -e STAGE=generate -e START_IDX=<i> -e END_IDX=<i+1> -e NUM_PERSONAS=30 -e PRESET=supermemory_featured_clocksync supermemory`. The persona restarts from session 0. Relaunching every failure immediately competes for host memory with the pool's own unstarted personas, and each restart is worth the same one persona as a fresh one, so under memory pressure, let the pool drain first and rerun the failed set afterward at lower concurrency |

---

## Provider: Honcho

Three vendor words appear below. The **deriver** is the extraction worker that
turns messages into stored observations. The **dialectic** is Honcho's internal
question-answering call, `POST /peers/{id}/chat`. A **dream** is the consolidation
job that rewrites stored observations into higher-level conclusions.

| Symptom | Cause | Fix |
|---|---|---|
| API and deriver both refuse to boot below 1536-dim embeddings | Migrations hardcode `Vector(1536)` (`migrations/versions/917195d9b5e9:31`, `119a52b73c60:45,53`, `a1b2c3d4e5f6_initial_schema.py:366`) while `src/models.py` sizes columns from `EMBEDDING_VECTOR_DIMENSIONS`. `provision_db.py` only replays migrations, so the columns stay at 1536 | Run the vendor's `scripts/configure_embeddings.py --yes` after provisioning, it drops and rebuilds the two HNSW indexes and ALTERs `documents.embedding` and `message_embeddings.embedding`. `_honcho_server.py`'s `apply_embedding_dim_fix()` runs it and verifies `atttypmod` |
| A run exits 0 with a workspace holding zero conclusions; hybrid recall reads only the dialectic's "I have no information" | An unreachable embedder does not stop Honcho. Every "save representation" call fails on the embed step (observed: 518 representation tasks processed, 954 x "Failed to save representation ... 401 Missing Authentication header", 0 rows in `documents`), and the deriver keeps draining work units regardless, so nothing signals failure | `HonchoServer.start()` refuses spawn mode with no `HONCHO_EMBEDDER_BASE_URL` set, so the run fails at boot instead of finishing silently empty |
| Deriver logs `Llm Call Duration 122131 ms` with `Observation Count 0`; user-side conclusions stay 0 while assistant-side looks healthy | A reasoning model at its own default reasoning effort spends the whole `MAX_OUTPUT_TOKENS` budget (8192) reasoning and returns empty content; JSON repair then fails on the empty string | `HONCHO_LLM_THINKING_EFFORT=low` (maps to `{MODULE}__THINKING_EFFORT` -> `reasoning_effort`, `src/llm/backends/openai.py:297`). Drain per session dropped 125 s -> 4 s. Leave unset for a non-reasoning model (qwen3.5-4b, contract v4) |
| `external/honcho`'s `uv.lock` changes after a routine dependency sync | A plain `uv sync` re-resolves and drops `exclude-newer` | `uv sync --frozen` (verified rc=0 on Python 3.11). `Dockerfile.honcho` builds with `--frozen` against the pristine lock |
| `schedule_dream` returns 2xx but no conclusions consolidate | The POST always 2xxs. Failures land only in the deriver log, e.g. "Error processing dream task DreamType.OMNI ... Collection not found" when the pair has no conclusions yet | Check the deriver log, not the HTTP response, before trusting a dream fired. The adapter's per-persona summary records `Total_Dream_Errors` |
| An OpenRouter model id resolves to a name the endpoint does not serve (`gpt-oss-20b` instead of `openai/gpt-oss-20b`) | `_normalize_model_transport` (`src/config.py:262-275`) splits a model id at the first `/` when the prefix is anthropic/openai/gemini AND transport is unset | Set `{MODULE}__TRANSPORT=openai` next to every `{MODULE}__MODEL` |
| Deriver startup adds up to 30 s of latency before the first drain can begin | `DERIVER_POLLING_STARTUP_JITTER_SECONDS` defaults to 30 s so co-started deriver instances do not poll in lockstep | Set 0.0, one dedicated deriver has no peers to collide with |
| Deriver batches complete, but many "save representation" calls drop and `documents` grows far slower than the observation counts imply | `vllm-embed` answers 400 to any input above its served window. Contract v4's bge-small-en-v1.5 window is 512 tokens PER INPUT; contract v5's `gte-modernbert-base` is served at 8192 (`VLLM_EMBED_MAX_LEN`), so this failure is far rarer under v5 but not impossible. Honcho's representation path calls `simple_batch_embed` (`external/honcho/src/embedding_client.py:251`), which does no chunking and no length check, so one long observation fails the whole save for both observers (`src/crud/representation.py:111`). Measured on smoke `hn_smkmin_p0b`: 14 dropped saves against 11 completed deriver batches in persona 0, sessions 0-2. `EMBEDDING_MAX_INPUT_TOKENS` (default 8192, `src/config.py:705`) does not fix it: that path never reads it, and Honcho counts tokens with tiktoken `cl100k_base` while bge tokenizes with WordPiece, so the two counts disagree | `honcho/embed_proxy.py`, started by `entrypoint.honcho.sh` in the spawn-generate path, adds `truncate_prompt_tokens=-1` to every upstream embedding request, which tells vLLM to cut at the served model's own window. The cut therefore removes only text the encoder could not attend to. The precedent is `retaindb_server/embed_proxy.py`. Set `HONCHO_EMBED_PROXY=0` to measure the unproxied rate again |
| Stored conclusions reach 50,000 characters, one sentence repeated hundreds of times | The deriver repetition-loops on qwen3.5-4b and runs to the output cap. Measured on smoke `hn_smkft_p0`: 18 of 79 documents (23%) sat within 1% of the 8192-token cap, mean 41,189 chars, unique-sentence ratio 0.181, worst row one sentence 341 times. The rate ran 23% to 33% of observations across the two smokes. `openai.py:148` repairs the truncated JSON, so the runaway is STORED instead of discarded. `HONCHO_LLM_MAX_OUTPUT_TOKENS` (`_honcho_server.py:386`) is a global: it feeds the deriver, summary, five dialectic levels, and both dream specialists | `HONCHO_DERIVER_MAX_OUTPUT_TOKENS=2048` plus `HONCHO_DERIVER_PRESENCE_PENALTY=1.5`, both applied to `DERIVER_MODEL_CONFIG` ONLY, so the dialectic and dream budgets do not move. Real observations have a median length of 241 chars, so 2048 tokens cannot truncate a healthy one. vLLM's `get_diff_sampling_param` does not allowlist `presence_penalty`, so the Qwen card value reaches an internal call only as a per-request kwarg |
| The answer prompt overflows the 32768-token window; vLLM 400s on nearly every question after the store fills | Uncapped injection. The plugin's `_truncate_to_budget` (`__init__.py:870-883`) is shipped OFF, `_parse_context_tokens` (`client.py:145-153`) returns None when `contextTokens` is unset, and the adapter had no equivalent. Measured at persona 0 session 5 of 53: the featured hybrid block reached 254k tokens, and the minimal conclusions arm reached 32,890 prompt tokens on a top-5 slice. Runaway documents (row above) are the input | `HONCHO_CONTEXT_TOKENS=8192` (`eval_honcho.py` `truncate_items_to_budget`), a port of the plugin's own cut: `context_tokens * 4` chars, word boundary kept only past 80% of the budget, `" …"` appended. It runs on the final joined block of BOTH recall paths, after the top-K slice. 0 restores the shipped uncapped behavior |
| The adapter logs `dialectic failed (level=low): An unexpected error occurred` on 61 of 122 questions (50%) in the featured contract-v4 smoke `smkft3`, first at session 9 of 53, at a rate that rises with store size. The HTTP status is 500, from Honcho's generic handler (`src/main.py:233-243`) | A vLLM 400 context overflow on tool-loop **iteration 1**, before any tool result exists. `src/dialectic/core.py:170` hardcodes a 25-observation prefetch (10 when `reasoning_level=minimal`), and `_prefetch_relevant_observations` (`core.py:271-287`) pastes 25 explicit plus 25 derived observations unabridged into the user message. `search_memory` (`src/utils/agent_tools.py:1062-1105`) limits result COUNT and has no token or character budget. Measured prefetch alone: **20,739-27,197 tokens**. `DIALECTIC_MAX_INPUT_TOKENS=20000` fires but reduces nothing, 209 of 210 truncation events removed 0 tokens, because `truncate_messages_to_fit` (`src/llm/conversation.py:120-171`) keeps the system message and never drops the last unit, and at iteration 1 the conversation is exactly `[system, user]`. The counter also ignores tool schemas and `tool_calls` (`conversation.py:18-33`): a measured ~2,324 uncounted tokens per request, of which the tool schemas alone are 1,162 | **No fix on contract v4's 32,768-token window.** Contract v5 serves 131072, which is past the measured prefetch, so the featured arm does not hit this. Report the failure rate with any v4 numbers (see DECISIONS "The dialectic and dream overflows are reported, not configured away"). Keep `HONCHO_DIALECTIC_MAX_INPUT_TOKENS=20000`: it bounds later tool-loop iterations, and it costs nothing. The earlier claim here that 20000 removes these 400s is superseded, it structurally cannot reach iteration 1 |
| Dream specialist calls 400 on context length even after the deriver is capped: in `smkft3`, deduction failed 97 of 104 and induction 71 of 102. The run summary shows no sign of it, the orchestrator logs `Dream cycle completed` and the adapter's `Total_Dream_Errors` stays 0 | No input bound EXISTS on the specialist path. `src/dreamer/specialists.py:267-286` omits `max_input_tokens`, and `tool_loop.py:344` truncates only when it is set, zero `Truncating:` lines in `deriver.log` against 210 in `api.log`. `DREAM_HISTORY_TOKEN_LIMIT` (16,384) bounds `get_recent_history` only, which is in neither specialist's tool set (`agent_tools.py:1616`). Tool results dominate the prompt about 20:1 over the prompt text: mean `search_memory` result ~7,060 tokens, max ~31,500, so ONE maximal result exceeds the whole 24,576-token input budget (32,768 minus the 8,192-token output reservation) | **No fix.** No `DREAM_*_MAX_INPUT_TOKENS` exists in `src/config.py`, and `DREAM_MAX_TOOL_ITERATIONS` is never read by the specialists, which hardcode 12 and 10. Read `deriver.log` for the true rate; the summary counters do not see these failures |
| The deriver still repetition-loops with `HONCHO_DERIVER_MAX_OUTPUT_TOKENS=2048` AND `HONCHO_DERIVER_PRESENCE_PENALTY=1.5` both applied | 228 of 4,346 explicit documents (5.2%) in `smkft3` sit pinned at 1,877-2,028 tokens, the cap, with a tail sentence such as "The user resides in a location where they have a backyard suitable for organizing casual games." repeated to the cap. 128 of the 394 documents over 2,000 chars are detected loops, and they hold 1.27M of those documents' 3.04M chars | The two mitigations cut the rate from 23-33% to 5.2% and bound each runaway to a quarter of its old size. No further vendor knob exists. These inflated documents are the single upstream defect feeding BOTH overflow rows above: they take 8 to 10 of every 25 dialectic prefetch slots, and they are inside every specialist `search_memory` result. A deriver fix is the only one change that would shrink both prompts |
| A dream specialist logs `Tool create_observations_inductive failed unexpectedly: TypeError: 'str' object does not support item assignment` (also seen on `create_observations_deductive`). 11 of about 204 specialist tool calls in the contract-v5 featured smoke `ft27hn2`. The dream still reports success and `Total_Dream_Errors` stays 0 | The model returns a plain string where the tool in `src/utils/agent_tools.py` expects a dict and assigns into it by key. The tool has no type guard, so the argument shape the model chose becomes an exception | Vendor defect, no knob, report it (ruling 3). It only became visible under contract v5: on v4's 32,768-token window the specialists 400'd on context length before they ever reached a tool call, so this rate is new information, not a regression. Read `deriver.log` for it, the run summary does not count it |
| Two config names from the design draft turn out to have no effect | `{MODULE}__STRUCTURED_OUTPUT_MODE` does not exist in v3.0.9, the openai backend tries strict `json_schema` via `chat.completions.parse` and falls back automatically on `BadRequestError`/`JSONDecodeError`/`ValidationError` with JSON repair (`src/llm/backends/openai.py:144-199`). `DERIVER_REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS` is also not a real field, the real one is `DERIVER_REPRESENTATION_BATCH_MAX_TOKENS` (default 1024) | Config `Settings` classes are `extra="ignore"`, so both were silent no-ops rather than errors. Removed from the adapter and server manager instead of left as dead config |

---

## Provider: OpenViking

| Symptom | Cause | Fix |
|---|---|---|
| `openviking-server` exits 1 at boot with `Unknown config field 'telemetry.enabled'` | 0.4.12's `TelemetryConfig` is `{"tracer": {...}}` and every config model rejects unknown keys (`extra: "forbid"`). The vendor docs' `telemetry: {"enabled": false}` shape does not exist in this version | Omit the `telemetry` section from `ov.conf`. `tracer.enabled` and `server.usage_reporter.enabled` both default to `False`, so omission is the off state. `_openviking_server.py` writes no telemetry section |
| Session commit task ends `failed` with `Expecting value: line NNNN column 1` after ~12 min, or `extract_loop` logs `failure_kind=empty_response` | gpt-oss-20b at its default reasoning effort burns the whole `vlm.max_tokens` budget on reasoning (empty content), or the extraction call outlives OpenRouter's keep-alive window, OpenRouter pads the pending response with newlines and closes with no JSON payload, which the OpenAI client fails to parse (smoke run 2, 2026-08-03: 3,364 newlines then EOF at char 18502) | `OPENVIKING_LLM_EXTRA_BODY='{"reasoning": {"effort": "low"}}'` (merged into `vlm.extra_request_body`) plus `OPENVIKING_LLM_MAX_TOKENS=8192`. Same mechanism as `HONCHO_LLM_THINKING_EFFORT=low`. `run_openviking.sh` sets both for the OpenRouter path; local qwen3.5-4b is non-reasoning and needs neither |
| A question's recall is empty (only the session-start block in `prefetch` mode) while sibling questions retrieve 6 scored entries | `/api/v1/search/search` runs LLM intent analysis and can emit zero queries: the smoke's `query_plan` reasoned "the session context already states..." and returned `queries: []`, `total: 0`. The plugin does not fall back to `find` on an empty result, only on request failure | None, plugin-faithful behavior, report it (ruling 3). Quantify with the `find` diagnostic arm, which skips intent analysis. Per-question evidence is in `Provider_Raw_Retrieval.response.result.query_plan` |
| Lane exits 1 at boot; server log ends `DataDirectoryLocked: Another OpenViking process (PID N) is already using the data directory` | `docker rm -f` leaves the vendor's `.openviking.pid` lock on the per-run workspace. When the workspace is reused, the recorded PID can name a live process in the new container, so the lock reads as held (`process_lock.py:133` in the 0.4.12 wheel) | Delete `openviking/.openviking_runs/<run_tag>/` before relaunching that persona. The failure is loud, exit 1 with the lock path named |
| Adapter aborts with `extraction task ... still 'pending' after 1800s`; server log has one `[QueueManager] Concurrent worker error for SessionCommit: ...`, either `lock I/O error: failed to read lock token at '....path.ovlock' ... No such file or directory` or `Expecting value: line 1 column 1 (char 0)` | An unhandled queue-worker exception records the error and never acks; the item re-queues only via RecoverStale at the next server start, so the commit task stays `pending` until the adapter's task timeout (wheel `queue_manager.py:236-240`). The trigger on this Windows host is workspace I/O on the Docker Desktop bind mount: a just-written file not yet readable. 8 casualties in ~45 min across 20 lane-starts (2026-08-04, v5minovk wave), in two clusters, one at a 12-server start burst, one at steady state, so the bind mount is unsafe at any load | Put the workspace on container-local storage: `-e OPENVIKING_RUN_DIR=/tmp/ovk_run` (`MSYS_NO_PATHCONV=1` on the docker command, or Git Bash rewrites the path). Results, manifest, and sidecars still land on the bind-mounted `Results/`. **A native-Linux host DOES reproduce this, the earlier "no bind-mount layer on Linux" claim is wrong** (2026-08-05, `v5ftovk` wave on the rented VM, 30 shards on a bind-mounted workspace): personas 16, 10, and 21 aborted with the same 1800 s timeout at 23:58, 00:05, and 00:51 UTC, about one per 50 minutes, and p21's `server.log` carries the `.ovlock` read error. Docker Desktop's filesystem translation makes it worse, it does not cause it, concurrent workspace I/O at 30 shards is enough. Apply `OPENVIKING_RUN_DIR=/tmp/ovk_run` on every host, not only Windows. Detection: any lane silent >8 min is stuck (normal commits run 30-90 s); kill and relaunch that persona rather than waiting out the 1800 s timeout. A relaunch restarts the persona from session 0, so on a deadline compute the last recoverable moment as `expiry − full-persona wall` and bank N of 30 after it |

## Provider: RetainDB server edition

This is a distinct product from the ruled-out local edition. The project found
seven upstream defects. Five carry patches in `retaindb_server/server_patches/`.
`Dockerfile.retaindb-server` references those patches, so removing one breaks the
image build.

| Symptom | Cause | Fix |
|---|---|---|
| Recall returns ZERO memories for "has X changed recently?" questions while other questions on the same store return a full top-5 | A **~7-day recency window** on temporally-worded queries, evaluated against `question_date`. Both conditions are required: temporal wording AND `question_date` sent. Measured 2026-07-28 on `rdbs_diag2_p2`, same store, `question_date=2022-11-20`: "Has the user work status changed recently?" returned 0, "What is the user work status?" returned 5. The boundary sits at 7 days (`11-12` returned 2, `11-13` returned 0) | No vendor knob found. This is provider behaviour, so **report it, do not engineer around it**. Dropping `question_date` would trade it for wall-clock recall, which is worse. This is NOT the `validUntil` supersession filter: `validUntil` is set on zero rows |
| Store is nearly empty after many sessions; `created=0` per session with `errors=0` | RetainDB's extraction yields almost nothing on MemConflict dialogue: 2 memories from 11 sessions on persona 2, 10 by session ~13, with degenerate content ("Correction: it monthly so it's ready for reviews"). `chunks`/`documents`/`entities`/`embeddings` all show 0 rows | Unresolved. This is why EARLY sessions return empty recall on every question. It is distinct from the recency window, which bites later sessions. Report it as a provider characteristic |
| A store-state hypothesis cannot be checked after a run | Each persona container owns its own Postgres, and **the database dies with the container** | Query the FIRST completed persona while the rest of the pool still runs: `docker exec <c> psql -U postgres -d retaindb_<tag>_p<i>`. The v4minc run had a 3.5-hour window for this, and we missed it |
| `prisma.$use is not a function` at boot | `@prisma/client@6.19.3` removed the `$use` middleware API | Patch 0001 ports it to `$extends`/`$allOperations` |
| `Unknown argument conversationId` on `message.createMany` | The shipped `schema.prisma` describes only 27 of the 67 tables the migrations create | The build swaps in `schema.introspected.prisma` (a full `prisma db pull`, 67 models) |
| First ingest 500s on a pristine deploy | The server auto-creates a project FK'd to `organizations` then `users`. Both are empty after `migrate deploy` | `seed.sql` inserts the default org and owner user idempotently |
| Server refuses to boot | `ENCRYPTION_KEY` is required, at least 32 characters | Entrypoints default it to a documented non-secret dev value |
| Featured arm: EVERY session's `lifecycle_wait` ends `status=timeout session_scoped=0` after the full 150 s, though the scheduler log shows `promoted=N skipped=0 summary=skipped` about 1 s after its tick | This is not a promotion failure. `session_scoped=0` is the LAST-poll value, read after promotion already flipped every SESSION row to USER. Three things interact: `generateSessionSummary` returns null **silently** on short LLM output (`if (!summary \|\| summary.length < 20) return null`, `session-lifecycle.ts:217-218`, an error would log, this does not); promotion runs concurrently (`Promise.allSettled`, :253-256) and consumes the SESSION rows that `findStaleSessions` selects on (:79), so the summary is never retried; and the release table had no exit for "promoted, summary-eligible, summary impossible" | Fixed in `8b957af` plus `f3c0fc5`, then superseded by the log-marker release: the barrier now keys on the scheduler's OWN per-pass completion line, `[session-lifecycle] <sid>: promoted=N skipped=M summary=<uuid\|skipped>`. `summary=skipped` is definitive for that pass, so the session releases IMMEDIATELY. `summary=<uuid>` means keep polling for the DB row. **No marker yet means the pass is still running, so KEEP WAITING.** The entrypoint file-redirects the node server's output to `$RETAINDB_SERVER_LOG` and tails it back to stdout. This uses redirection, not a pipe: `node … \| tee f &` would make `$!` the tee pid and break the liveness/kill logic. The old no-progress grace remains only as a fallback when the log is unreadable. Every release records `release_signal=log_marker\|db_grace\|timeout`. Measured: 150 s timeouts fell to 35-70 s under the grace, then to **5.0 s** on the marker. Why the grace alone was not enough (external review, 2026-07-27): we measured 15 s on a near-idle 1-persona server, but under a full wave a VALID summary call can outlast it. The grace would then declare a skip, answer that session without the summary, and undercount the summary rate |
| One session re-enters the stale list forever, logging `promoted=0 skipped=1 summary=skipped` every tick (74 such lines in one 53-session run, ALL from a single session) | The promoter refuses one row by TYPE: `PROMOTABLE_TYPES` (`session-lifecycle.ts:28`) excludes `project_state`. That session is left with 1 memory, below `SUMMARY_MIN_MEMORIES=2`, so it never gets a summary either, and the SESSION row keeps it eligible for reselection indefinitely. A second, different orphan case: 10 rows written AFTER a session's single lifecycle pass took its `findMany` snapshot become permanently invisible, because the summary that pass wrote makes `findStaleSessions` (:81-88) exclude the session forever | Nothing to fix server-side. This is vendor behaviour, and `external/` is pinned. The harness watchdog above releases on lack of progress, which covers both cases. Signature of a stuck row: no `promoted_at` / `promoted_from_session` / `source_session_id` metadata. All 817 of 817 promoted rows carry all three fields; 0 of 11 stuck rows do |
| Clocksync SQL assert reports rows "leaking" to real time that are actually correct | The assert threshold was written as `"createdAt" >= '2026-01-01'`. But **MemConflict's logical session dates run 2022 through 2026-01**, so a persona whose last session is 2026-01-15 legitimately stores rows in Jan 2026 | Use the run's REAL wall-clock month as the floor: `WHERE "createdAt" >= '2026-07-01' OR "updatedAt" >= '2026-07-01'` must return 0. The old query passed early smokes only because their rows stopped at 2025-11. On a 30-persona wave it would have failed falsely on every persona reaching Jan 2026 |
| Every search response has `temporal.document_date: null` despite correct DB data | `api/memory.ts` reads `r.memory.documentDate` at top level, but the engine returns it nested under `memory.temporal`. Verified: 176 of 176 memories had the field in the DB, but all showed null in responses | Patch 0003 |
| Queries return another persona's results | The semantic cache keys only on query-embedding similarity (threshold 0.85), with no project/user/date scoping. The live log showed hits at 0.915-0.986 across personas | Patch 0004 adds `RETAINDB_DISABLE_SEARCH_CACHE`. Entrypoints export `true` |
| Re-asked questions get stale-era results | The exact-key cache omits `question_date`, and the benchmark compresses months into seconds inside the 300s TTL | Same patch 0004 knob |
| micro AA 0.004 despite completing cleanly | ivfflat indexes (`lists=100`) built on **empty** tables at migration time produce degenerate centroids. At `probes=1`, search scans one near-random cluster. True top-2 evidence (cosine 0.657/0.629) never appeared. The top hit scored only 0.149-0.595 | `post_migrate.sql` drops the three ivfflat indexes. Exact KNN runs sub-ms at per-run-DB scale. This is not the production answer: rebuild after bulk load, or use HNSW |
| A 500 roughly every 500 searches kills a long run | A path emits a result row with no `.memory` object | Patch 0005 keeps the response alive instead of throwing, and the client retries 5xx (never 4xx). The patch originally DROPPED such rows; since `48967b3` it wraps and keeps them, see the next row for why dropping was wrong |
| Recall returns zero candidates on exactly the highest-confidence questions, 11 of 122 in the persona-27 featured smoke `ft27rdb`, and the log pairs an `Early exit at 0.9xx` line with `dropping N malformed` | The vendor's early-exit path (similarity ≥ 0.92) returns raw rows without the memory envelope. Patch 0005 dropped any row with no `.memory` object, so the best hits were the ones discarded | Patch 0005 now wraps such a row in the envelope, mirroring `injectSourceChunks`' chunk-less shape (`48967b3`). Verified in `ft27rdb2`: 0 empty retrievals, fill 4.87 of 5, the wrap fired 200 times. The patch file is CRLF, to match the Windows build context |
| Every search reports fast-mode degradation, 117 of 122 in `ft27rdb`, so results carry no graph traversal and no temporal scoring | `MEMORY_SEARCH_POST_VECTOR_BUDGET_MS` defaults to 120 ms (`search.ts:31`). A remote contract embedder's round trip exceeds that on nearly every query, and the server then skips the post-vector stages | Set `MEMORY_SEARCH_POST_VECTOR_BUDGET_MS=2000`, a vendor-exposed knob (compose default since `48967b3`, justified under ruling 2). `ft27rdb2`: 0 degradations |
| `pnpm install` fails on `sharp` | This is a native libvips build, on the image-ingest path only, never the memory path | Patch 0002 marks it never-built |
| Eval logs `done` but the container never exits (its exit-monitor never fires) | The server registers `process.on('SIGTERM', stopCacheCleanup)` (`engine/compressor.ts:369`), which overrides Node's default terminate-on-SIGTERM behaviour. The entrypoint's plain `kill` ran that handler and nothing else, then `wait` blocked PID 1 forever | `stop_server` in `entrypoint.retaindb-server.sh` sends SIGTERM, waits through an 8s grace loop, then sends `kill -9` and calls `wait` (server and embed_proxy). This is safe: ingest runs `write_mode=sync`, so all memories are committed before shutdown. Seen 2026-07-22 (`rdbs_itsmoke2`) |
| `start_embed_proxy` FATALs after 60s though the proxy is healthy | The health-gate greps the literal compact string `'"status":"ok"'`. `embed_proxy.py` used the default `json.dumps`, which emits `{"status": "ok"}` with a space | `_send_json` now uses `separators=(",", ":")`. If adding health endpoints, match the exact bytes the gate greps |
| `EXTRACTION_MODEL` has no effect | The vendor's `.env.example` advertises the wrong name. `engine/memory/extractor.ts` reads `EXTRACTOR_MODEL` | Set `EXTRACTOR_MODEL` |
| Scoped searches report `fallback:"lexical"`, `embed_ms:0` | This is a labeling quirk on the `rerankByScope` path. Vector search did run | Advisory, not diagnostic |
| Ops-maintenance tick fails every 60s with `PrismaClientValidationError: Unknown field 'source' for select on SyncJob` | `getConnectorHealthSummary`, exercised only with `DISABLE_SCHEDULER=false` (the featured arm), selects a field the schema does not have | Caught and non-fatal. It never touches memory or lifecycle. Unique to the featured arm. Candidate for a server patch if it ever starts mattering |
| Under `BENCH_CLOCKSYNC=1`, a promoted dataset-year memory ranks as maximally recent and dominates conflict-candidate windows; `createdAt` is correct but `updatedAt` reads real-2026 | `Memory.updatedAt` is `@updatedAt` with no DB default (`schema.prisma:248`), so Prisma stamps it client-side through the query engine's vDSO clock. `libfaketime` cannot intercept this. Promotion UPDATEs the row (`session-lifecycle.ts:147-158`), and `updatedAt` drives `fetchComparableMemories` ordering (`write.ts:475`), keyword search (`api/memory.ts:389-391`), and the recency anchor `eventDate \|\| documentDate \|\| updatedAt \|\| createdAt` (`search.ts:58-66`) | `clocksync_created_at.sql` runs `BEFORE INSERT OR UPDATE`: on INSERT it forces `createdAt` and `updatedAt` to the faked `now()`; on UPDATE it forces `updatedAt` only. Verify with `SELECT count(*) FROM memories WHERE "createdAt" >= '2026-01-01' OR "updatedAt" >= '2026-01-01'`, which must equal 0 |
| Featured-arm sessions report `lifecycle_wait status=done_promotion` but no session summary ever exists | The vendor runs promotion and the summary concurrently (`session-lifecycle.ts:252-256`). Promotion is a DB update; the summary needs an LLM round-trip (`:198-215`). The adapter released on the promotion alone, so the driver moved on mid-summary | The barrier now requires `has_summary` for summary-eligible sessions (`total_active >= SESSION_SUMMARY_MIN_MEMORIES`, default 2, `session-lifecycle.ts:23,170-183`). `done_promotion` is accepted only for ineligible sessions. Watch `Lifecycle_Eligible_Without_Summary` |
| Featured arm with the scheduler on reports `Lifecycle_Status_Counts {"skipped_no_session_scope": N}` for every session | Extraction routes almost all facts straight to USER scope, so `findStaleSessions` (which requires `scope='SESSION'`) has nothing to promote | `RETAINDB_SERVER_PROMOTION_MODE=user_specific_legacy` routes mid-confidence facts to SESSION scope. Yield is stochastic (gated by confidence and type). This is a vendor limitation, reported not fixed. Fidelity-label the arm (see DECISIONS) |
| `/v1/memory/search` results show recently-ingested rows at similarity 1.0 with wall-clock `created_at` | `include_pending` (default true) merges recent raw content into results outside the normal scoring and timestamp path | Plugin-faithful: the plugin gets the same overlay. Kept for the featured arm. Disclose it when reading scores |

---

## Provider: RetainDB local edition, why it was ruled out

**Ruled out 2026-07-22.** This is a product-level scaling defect, not a wiring
problem. The adapter is correct and is kept for the record.

**Cause.** In `@retaindb/local@0.2.1`, `dist/cli.js`:
`LocalMemoryRuntime.search()` (`:405`) maps over every candidate. For each
candidate it calls `relatedConceptBoost` (`:503`), which does
`this.graph(project).edges`, a full concept-graph rebuild that scans every active
memory and sorts the whole edge map (`:512-533`). The code has no memoization,
even though the graph is identical for every candidate in one query. The rebuild
runs *before* the relevance filter, so discarded candidates pay for it too, though
only the top-300 edges are ever used.

**Measured** against a copy of a live store: n=2,897 active memories, 51,015
edges. One rebuild takes 84.5 ms. 2,897 x 84.5 ms gives **~245 s per `/search`**
on an idle core, before BM25 or cosine scoring even runs.

Two further store-wide costs exist. Every `/search` call bumps
`access_count`/`strength`, then persists a 24.6 MB pretty-printed JSON file.
`addMemory` rewrites the whole store per memory, about 2.2 GB of writes per
90-memory session at that size. The product has no vector index and no inverted
index.

**No fix available.** The full env surface is
`RETAINDB_{PORT,PROJECT,HOME,STORE,EMBEDDING_PROVIDER,EMBEDDING_MODEL,VIEWER_PORT,BASE_URL,AGENT_ID,SESSION_ID}`.
`top_k` slices *after* all n candidates are scored. Memoizing `graph()` would mean
patching vendor code, which is out of bounds, since the vendor's code is the
product under test.

A full run projected at 46 hours or more. Zero personas were banked. No
contract-v3 number exists.

This finding resolved three earlier confusions. 8-shard and 15-shard configs both
landed near 0.57 sessions/min because the bound is quadratic compute, not
scheduling. Cost exploded with session number because n grows ~90 per session, so
cost grows ~n². The 6-session smoke predicted nothing because smokes validate
plumbing, not scaling.

**Sharding history, for the record.** 15 shards ran 4.6x *slower* in aggregate
than 6 (0.54 vs 2.5 sessions/min), while CPU usage went from 507% to 1505% of
1600% and GPU usage fell to 15%. Shard s0's `/search` time grew from 99s (session
8) to 191s (12) to 380s (15) to 1,171s (18). Adding shards to speed this up is
what turned a projected ~10 hours into never-finishes.

Also note: `embedText()` wraps the `local-transformers` path in a bare
`catch { return hashEmbedding(text) }`, so a load failure silently degrades to
lexical-only retrieval and still exits 0. Check for this directly: MiniLM vectors
are 384-dim, hash vectors are 96-dim, so inspect a stored embedding.

## Mnemosyne featured: `auto_sleep ... SKIP ... eligible=0` on every cadence tick

**Symptom.** A featured clock-sync shard log shows the cadence auto-sleep skipping
continuously, 243 consecutive `auto_sleep persona <id> trigger=cadence exch=<n>
SKIP working=2146 eligible=0` lines in one persona, which reads as consolidation
never running.

**Cause: it already ran.** The arm sets `PLUGIN_SESSION_SLEEP=1`, so a forced
sleep fires at every session end and stamps `consolidated_at` on everything:
`session_sleep ... FIRED status=consolidated items=45 summaries=1 mr_proposals=5
mr_applied=1 wm_total=2150 wm_consolidated=2150 wm_unconsolidated=0`. The cadence
gate then finds nothing unconsolidated, so `eligible=0` is the *consequence* of
consolidation working, not its absence.

**How to tell the two apart.** `grep` the shard log for `session_sleep` and check
`status` and `invocations`. Healthy: `FIRED status=consolidated` roughly once per
session with `wm_unconsolidated=0` and non-zero `items`. Broken would be no
`session_sleep` lines at all, or `wm_unconsolidated` climbing while `items=0`. Do
not diagnose from the `auto_sleep` lines alone.

**Not a fix, by design.** `MNEMOSYNE_WM_TTL_HOURS` stays unset on this arm and
`entrypoint.mnemosyne.sh` exits 2 if an explicit TTL reaches it, the shipped
168 h TTL under the faked clock is the property being measured.
