"""Measure the context length and output size of the MemConflict run.

This script measures three LLM workloads in a memory-provider benchmark
run. Use the results to set serving parameters, such as max context and
max output tokens, for self-hosted inference.

  1. INGESTION material: the dialogue a framework must feed to its
     extractor. Each framework uses its own granularity, for example
     per-message, per-8-batch, per-session, or whole-history. The
     script reports all of these.
  2. ANSWER generation: the shared MemConflict QA call. It sends a
     system prompt, the retrieved top-k memories, and the question,
     then returns a short answer. This is a small RAG call.
  3. JUDGE: the shared scorer call. It sends a judge prompt: the
     question, gold answer, model answer, top-5 retrieved memories,
     and rubric. It returns a small JSON verdict.

This script cannot use tiktoken, because its vocab download host is
blocked. Also, a single generic tokenizer would not match an arbitrary
self-hosted model. Instead, the script reports exact character counts,
which do not depend on a tokenizer. It converts characters to tokens
using a calibrated ratio. This run's real token usage sets the ratio,
from the mimo-v2.5 tokenizer in the answer-generation prompt accounting.
Other model tokenizers give different ratios. Treat the token numbers
as accurate to about plus or minus 15%, and add headroom.
"""

import json
import os
import sys
from statistics import mean, median

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CUR, "..", "external", "MemConflict", "Evaluation"))
sys.path.insert(0, CUR)

from eval_mnemosyne import (MNEMOSYNE_ANSWER_SYSTEM_PROMPT, Build_Session_Dialogue_List,
                            Build_Retrieved_Memory_Context)
import eval_scoring as es

ANSWER_TOP_K = 5  # This run generated answers using top-5 retrieval.

DATA = os.path.join(CUR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl")
RES = os.path.join(CUR, "Results", "mnemosyne_results.jsonl")
CKPT = os.path.join(CUR, "Scores", "judged_checkpoint.jsonl")


def clen(s: str) -> int:
    return len(s or "")


def pctl(v, p):
    if not v:
        return 0
    s = sorted(v)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def stats(v):
    return {"n": len(v), "median": int(median(v)) if v else 0, "mean": int(mean(v)) if v else 0,
            "p95": pctl(v, 95), "p99": pctl(v, 99), "max": max(v) if v else 0}


# ---- 1. Ingestion material (dataset dialogue) ----------------------------
msg, batch8, sess, persona = [], [], [], []
for line in open(DATA):
    p = json.loads(line)
    ptot = 0
    for s in p["Full_Session_Chain"]:
        d = Build_Session_Dialogue_List(s.get("Session_Dialogue", {}))
        lens = [clen(f"{m['role']}: {m['content']}") for m in d]
        msg.extend(lens)
        if d:
            sess.append(sum(lens))
        ptot += sum(lens)
        for i in range(0, len(lens), 8):
            batch8.append(sum(lens[i:i + 8]))
    persona.append(ptot)

# ---- 2. Answer generation (run's stored contexts) ------------------------
sys_ans = clen(MNEMOSYNE_ANSWER_SYSTEM_PROMPT)
ans_in, ans_out, ctx_only = [], [], []
# These variables track real mimo token counts against character counts, for the input side.
tot_in_tokens = tot_in_chars = tot_out_tokens = tot_out_chars = 0
for line in open(RES):
    p = json.loads(line)
    obs = p.get("Observable_Token_Cost_Summary", {})
    tot_in_tokens += obs.get("Input_Tokens", 0) or 0
    tot_out_tokens += obs.get("Output_Tokens", 0) or 0
    for s in p["Full_Session_Chain"]:
        for q in s.get("Session_Questions", []):
            if not q.get("Model_Answer"):
                continue
            # The compact output drops the context string. This code rebuilds
            # it exactly as the adapter did, using the stored top-k Retrieved_Memories.
            retrieved = q.get("Retrieved_Memories", [])[:ANSWER_TOP_K]
            ctx = Build_Retrieved_Memory_Context(retrieved)
            user = f"Retrieved Memory Context:\n{ctx}\n\nQuestion:\n{q.get('question','')}\n\nAnswer:"
            in_chars = sys_ans + clen(user)
            ans_in.append(in_chars)
            ans_out.append(clen(q.get("Model_Answer", "")))
            ctx_only.append(clen(ctx))
            tot_in_chars += in_chars
            tot_out_chars += clen(q.get("Model_Answer", ""))

# This ratio is chars-per-token, calibrated from real input accounting.
# The input has no hidden reasoning tokens.
CPT = (tot_in_chars / tot_in_tokens) if tot_in_tokens else 4.0

# ---- 3. Judge (reconstruct exact upstream prompt) ------------------------
sys_judge = clen(es.LLM_JUDGE_SYSTEM_PROMPT)
judge_in, judge_out = [], []
for line in open(RES):
    p = json.loads(line)
    for s in p["Full_Session_Chain"]:
        for q in s.get("Session_Questions", []):
            if q.get("conflict_type") not in es.METRIC_SCHEMAS:
                continue
            retr = es.extract_top_k_retrieved_memories(q, es.MAX_WHITE_BOX_TOP_K)
            prompt = es.build_llm_judge_prompt(q, q.get("Model_Answer", ""), retr, es.MAX_WHITE_BOX_TOP_K)
            judge_in.append(sys_judge + clen(prompt))
if os.path.exists(CKPT):
    for line in open(CKPT):
        rec = json.loads(line).get("judged", {})
        obj = dict(rec.get("Metrics", {}))
        obj.update(rec.get("White_Box_Metadata", {}))
        obj["reasoning"] = rec.get("Reasoning", "")
        judge_out.append(clen(json.dumps(obj)))


def tok(chars):
    return int(round(chars / CPT))


def row(label, s):
    return (f"| {label} | {s['n']:,} | {s['median']:,} / {tok(s['median']):,} | "
            f"{s['p95']:,} / {tok(s['p95']):,} | {s['max']:,} / **{tok(s['max']):,}** |")


H = "| stage | n | median (chars/tok) | p95 (chars/tok) | max (chars/**tok**) |"
SEP = "|---|--:|--:|--:|--:|"
print("# MemConflict — context & output size envelope\n")
print(f"Calibration: {tot_in_tokens:,} real input tokens (mimo-v2.5) over "
      f"{tot_in_chars:,} chars => **~{CPT:.2f} chars/token**. Token columns use this ratio.\n")

print("## 1. Ingestion material (dialogue an extractor must read)\n")
print(H); print(SEP)
print(row("per message", stats(msg)))
print(row("per 8-message batch", stats(batch8)))
print(row("per session (all turns)", stats(sess)))
print(row("per persona (whole history)", stats(persona)))

print("\n## 2. Answer generation (RAG QA call — shared harness)\n")
print(H); print(SEP)
print(row("retrieved context only (top-5)", stats(ctx_only)))
print(row("INPUT  (sys+ctx+question)", stats(ans_in)))
print(row("OUTPUT (answer, visible)", stats(ans_out)))

print("\n## 3. Judge (scorer call — shared harness)\n")
print(H); print(SEP)
print(row("INPUT  (judge prompt)", stats(judge_in)))
print(row("OUTPUT (JSON verdict, visible)", stats(judge_out)))

ai, ao = stats(ans_in), stats(ans_out)
ji, jo = stats(judge_in), stats(judge_out)
print("\n## Serving recommendation (tokens, this run's top-5 config)\n")
print("| workload | context (input) | generation (output) | notes |")
print("|---|--:|--:|---|")
print(f"| Answer (QA) | ~{tok(ai['p99']):,} (p99) / {tok(ai['max']):,} (max) | "
      f"~{tok(ao['p99']):,} (p99) / {tok(ao['max']):,} (max) | +hidden reasoning if reasoning model |")
print(f"| Judge | ~{tok(ji['p99']):,} (p99) / {tok(ji['max']):,} (max) | "
      f"~{tok(jo['p99']):,} (p99); one 3750th outlier hit ~{tok(jo['max']):,} | we served 4096 out, 0 failures |")
print(f"| Ingest, per-message | ~{tok(pctl(msg,99)):,} (p99) / {tok(max(msg)):,} (max) | framework-defined | + extractor prompt |")
print(f"| Ingest, per-8-batch | ~{tok(pctl(batch8,99)):,} (p99) / {tok(max(batch8)):,} (max) | framework-defined | Mem0-style batching |")
print(f"| Ingest, per-session | ~{tok(pctl(sess,99)):,} (p99) / {tok(max(sess)):,} (max) | framework-defined | |")
print(f"| Ingest, whole-history | ~{tok(pctl(persona,99)):,} (p99) / {tok(max(persona)):,} (max) | framework-defined | needs chunking on most servers |")
print("\nRule of thumb: an 8K context covers per-message/per-batch ingestion + all "
      "QA/judge; 32K covers per-session ingestion too. Whole-history ingestion "
      "(~325K tok) exceeds typical windows and must be chunked/summarized by the framework.")
print("\nCaveats:")
print("- Tokens estimated at ~%.2f chars/tok (mimo-v2.5, calibrated). Llama/Qwen/Mistral "
      "tokenizers are less dense (~3.6-4.0), so they yield ~10-20%% FEWER tokens — these "
      "numbers are a safe (slightly high) upper bound." % CPT)
print("- OUTPUT is VISIBLE text only. Reasoning models emit extra hidden reasoning tokens "
      "against the generation budget (mimo used ~100-300+); non-reasoning models do not.")
print("- The small retrieved-context (~%d tok max) reflects storing raw dialogue turns; a "
      "framework that stores larger memory units (session summaries) will retrieve bigger "
      "contexts." % tok(stats(ctx_only)['max']))
print("- Embedding retrieval (e.g. bge-small) truncates each memory at the embedder's max "
      "sequence length (512 tok) independent of the LLM context above.")
