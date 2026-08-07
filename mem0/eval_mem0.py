"""MemConflict evaluation adapter for the mem0 memory system.

The shared ``benchmark/eval_common.py`` driver runs the provider-agnostic
pipeline: dataset iteration, dialogue flattening, the answer prompt and answer
LLM call, results-row emission, and compaction. This pipeline is identical by
construction for every provider. This file supplies only the mem0-specific
binding: store setup, ingestion, and retrieval.

WHAT IS DIFFERENT ABOUT mem0 (vs. Mnemosyne / Hindsight / RetainDB)
------------------------------------------------------------------
mem0 is a Python SDK (``from mem0 import Memory``). It runs entirely
self-hosted and in-process: it owns an internal LLM for fact extraction, an
embedder, and a vector store. This adapter targets **mem0ai 2.x** (pinned to
``mem0ai[nlp]==2.0.14`` in ``benchmark/docker/Dockerfile.mem0``). This pipeline
differs from the 0.1.x line this adapter first targeted:

  * INGESTION IS SINGLE-PASS ADDITIVE EXTRACTION. ``Memory.add(messages,
    user_id=...)`` makes ONE internal LLM call. That call runs an
    additive-extraction prompt that shows the model the memories already
    stored for that ``user_id`` plus the new messages, and returns the facts
    worth adding (``configs/prompts.py:generate_additive_extraction_prompt``).
    Every emitted row carries ``event: "ADD"`` (``memory/main.py:1165-1168``).
    2.x REMOVES the 0.1.x second phase: a per-fact ADD / UPDATE / DELETE /
    NONE decision against the existing store. This algorithm change is itself
    a finding to report for this contract. Conflicting facts now coexist as
    separate rows, and conflict resolution shifts to RETRIEVAL and the answer
    model instead of happening at write time. In 2.x, extraction failure
    RAISES ``mem0.exceptions.LLMError`` (``main.py:1267``) instead of quietly
    returning no facts, so the adapter counts a failed add instead of
    silently absorbing it (``Total_Add_Calls_Failed``).

  * RETRIEVAL IS HYBRID: it blends semantic dense-embedding search, BM25
    lexical search, and entity linking into one ``score``. ``Memory.search(
    query, filters={"user_id": ...}, top_k=N, threshold=T)`` uses a 2.x
    keyword-only signature. ``limit`` is gone; use ``top_k`` instead. A
    top-level ``user_id=`` argument RAISES via
    ``_reject_top_level_entity_params``, so the tenant ID must travel inside
    ``filters``. Entity linking auto-creates a SECOND vector collection,
    ``<collection_name>_entities`` (``main.py:303-304``,
    ``_entity_collection_name``), and the reset path must drop this
    collection alongside the main one. BM25 and entity lemmatization need
    optional extras: spacy plus ``en_core_web_sm`` (``mem0ai[nlp]``) and
    ``fastembed`` for qdrant sparse vectors. The image bakes both in at
    BUILD time, because runtime egress is blocked. Without them, mem0
    silently degrades to semantic-only search, which would be a different
    arm.

BEST-EFFORT CONFIG (per the project's best-effort ruling): mem0's internal
LLM points at the SAME serving model the harness uses to answer (qwen3.5-4b
via vllm-gen for offline runs). This matches what a real Hermes deployment
self-hosting mem0 would run. ``infer=True`` (mem0's extraction) stays ON:
turning it off would store raw turns and discard the feature this benchmark
exists to measure. The embedder is bge-small-en-v1.5 offline, identical to
Mnemosyne and Hindsight, so the retrieval-embedding surface is shared and not
a mem0 advantage. The adapter requests search at mem0's own provider-default
width (``top_k=20``) and slices to the shared top-K for the answer context.
This matches the shape of the RetainDB (10 -> 5) and Supermemory (10 -> 5)
arms. It sets ``threshold=0.0`` so the vendor's 0.1 blended-score cut cannot
make mem0 answer from fewer memories than every other provider's shared
top-K.

TIME: the adapter passes the dataset's simulated session ``Date`` as each
memory's ``metadata.timestamp`` AND as ``metadata.created_at``. 2.x honors a
caller-supplied ``created_at`` in metadata and stamps the stored payload's
``created_at``/``updated_at`` from it (``main.py:1003-1005``), so logical time
reaches the store even without libfaketime. The ``timestamp=`` kwarg on
``add`` (and ``reference_date=`` on ``search``) is a MANAGED-platform-only
feature. It raises ``ValueError`` in OSS (``main.py:1066``, ``main.py:1552``),
so the adapter never passes it. Under ``BENCH_CLOCKSYNC=1``, libfaketime also
moves the process clock, so the extraction prompt's per-call Observation/
Current Date lines (``prompts.py:_resolve_dates``, lines 297-303) land on the
session's logical date.

Isolation is per-persona: each persona gets a unique mem0 ``user_id`` (mem0's
native tenancy boundary) inside one disposable, per-run vector store. This
mirrors how the RetainDB adapter isolates personas by ``project`` inside one
disposable server.
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Set MEM0_TELEMETRY to False before importing mem0, because the module reads
# this variable at import time. mem0 fires anonymous PostHog telemetry events;
# the egress proxy blocks us.i.posthog.com and returns 403, so leaving
# telemetry on floods stderr with harmless ProxyError tracebacks.
os.environ.setdefault("MEM0_TELEMETRY", "False")

# Import the mem0 PACKAGE here, while sys.path[0] is still this script's own
# directory (``<repo>/mem0``) and nothing has put the repo root on the path.
# This provider folder is itself named ``mem0``. If the repo root were ever on
# sys.path, ``import mem0`` could resolve to THIS folder (a namespace package)
# instead of the installed SDK, and shadow it. Importing here, before any
# sys.path inserts, and never inserting the repo root below, guarantees the
# adapter binds the real SDK.
from mem0 import Memory  # noqa: E402

# In mem0ai 2.x, extraction failure raises an EXCEPTION (``raise LLMError(
# f"LLM extraction failed: {e}") from e``, mem0/memory/main.py:1267); 0.1.x
# returned an empty fact list instead. Import LLMError here, next to the
# Memory import, for the same sys.path reason. The fallback class keeps the
# adapter importable, and the except clause harmless, if mem0 ever moves the
# real class.
try:
    from mem0.exceptions import LLMError  # noqa: E402
except Exception:  # pragma: no cover
    class LLMError(Exception):  # type: ignore[no-redef]
        """Use this placeholder when mem0.exceptions.LLMError is unavailable."""

# --- Compat shim: mem0's OpenAI embedder sends `dimensions` -----------------
# mem0ai 2.0.14's OpenAIEmbedding (mem0/embeddings/openai.py) sets
# ``self._pass_dimensions_to_api = self.config.embedding_dims is not None``
# and then adds ``kwargs["dimensions"] = self.config.embedding_dims`` in BOTH
# ``embed()`` and ``embed_batch()``. This adapter DOES set embedding_dims
# (384, to pin the qdrant collection dimension), so mem0 still sends the
# param. vLLM's pooling endpoint returns 400 on ANY explicit `dimensions`
# param for a model that does not declare matryoshka support ("does not
# support matryoshka representation, changing output dimensions will lead to
# poor results"). vLLM rejects the param outright, not just a mismatched
# value. A 1-persona Docker smoke (2026-07-22) confirmed this: every add()
# embed call returned 400 (caught and logged, so ingest silently produced
# zero memories), and the first search() call, not wrapped in a try/except
# upstream, then crashed the whole run.
# mem0 has no config knob to omit the parameter while keeping embedding_dims
# set, so this shim monkeypatches both methods to drop `dimensions` from the
# request. The output is numerically identical to what mem0 intended, because
# bge-small's only output size IS 384, matching the configured embedder_dims,
# so this changes no retrieval semantics.
# Patching ``embed_batch`` is not optional: 2.x calls it on the add path
# (main.py:964) and for entity embeddings (main.py:1078), and it raises
# ValueError on any returned/requested count mismatch. So the shim preserves
# the 100-per-request chunking, the index sort, and that count check exactly.
#
# --- Second reason the shim exists: the 512-token embedder cap ---------------
# 2.x embeds the ENTIRE add() input as ONE related-memory search query BEFORE
# extraction (Memory._add_to_vector_store:47,
# ``self.embedding_model.embed(parsed_messages, "search")``). 0.1.118
# extracted first and only embedded the short per-fact strings, so this
# ceiling is new in 2.x. The shared embedder bge-small-en-v1.5 caps at 512
# tokens by architecture. Measured against Step4_4.jsonl (5 personas,
# tokenized on vllm-embed), the share of add() windows over the 512-token cap
# is:
#   whole session     100%  (261/261, median 4087, max 9180)
#   8 msgs / 4 turns   8.8% (267/3032, median 358, p95 564)
#   6 msgs / 3 turns   1.9% (75/3997,  median 268, p95 429)  <- the arm's cadence
#   2 msgs / 1 turn    0.1% (8/11713,  median  90, p95 156)
# A 1-persona session-granularity smoke (2026-07-26) confirmed the first row:
# every one of 53 add() calls returned 400 in 2-3ms and stored ZERO memories.
# So the arm runs `batch` at 3 dialogue turns (MEM0_ADD_BATCH_SIZE=6), and
# this shim sends vLLM's `truncate_prompt_tokens` so the residual 1.9%
# truncate server-side instead of returning 400 and silently dropping ingest
# windows. The rate cannot reach 0% at ANY window size, because single
# messages run to 1585 tokens, so this shim is required at every cadence.
# The truncated text is mem0's INTERNAL related-memory pre-filter query, its
# ADD-vs-skip dedup decision. It never affects the retrieval surface under
# test: extracted facts are one sentence each and never approach 512 tokens.
# The adapter counts and reports truncations, so the rate is visible in the
# run summary rather than inferred.
_MEM0_EMBED_TRUNCATE_TOKENS = int(os.environ.get("MEM0_EMBED_TRUNCATE_TOKENS") or "512")
MEM0_EMBED_TRUNCATED_CALLS = {"embed": 0, "embed_batch": 0, "texts": 0}
try:
    import mem0.embeddings.openai as _mem0_openai_embedding

    def _vllm_extra_body():
        """Build the vLLM-only param. Omit it entirely when the knob is <= 0."""
        if _MEM0_EMBED_TRUNCATE_TOKENS > 0:
            return {"truncate_prompt_tokens": _MEM0_EMBED_TRUNCATE_TOKENS}
        return None

    def _vllm_compatible_embed(self, text, memory_action=None):  # noqa: ARG001
        text = text.replace("\n", " ")
        kwargs = {}
        extra = _vllm_extra_body()
        if extra is not None:
            kwargs["extra_body"] = extra
        resp = self.client.embeddings.create(
            input=[text], model=self.config.model, encoding_format="float", **kwargs,
        )
        # If usage.prompt_tokens equals the cap, vLLM clipped this input.
        used = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        if used is not None and _MEM0_EMBED_TRUNCATE_TOKENS > 0 and used >= _MEM0_EMBED_TRUNCATE_TOKENS:
            MEM0_EMBED_TRUNCATED_CALLS["embed"] += 1
            MEM0_EMBED_TRUNCATED_CALLS["texts"] += 1
        return resp.data[0].embedding

    def _vllm_compatible_embed_batch(self, texts, memory_action="add"):  # noqa: ARG001
        MAX_BATCH = 100
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i:i + MAX_BATCH]
            kwargs = {}
            extra = _vllm_extra_body()
            if extra is not None:
                kwargs["extra_body"] = extra
            response = self.client.embeddings.create(
                input=chunk, model=self.config.model, encoding_format="float", **kwargs,
            )
            used = getattr(getattr(response, "usage", None), "prompt_tokens", None)
            if (used is not None and _MEM0_EMBED_TRUNCATE_TOKENS > 0
                    and used >= _MEM0_EMBED_TRUNCATE_TOKENS * len(chunk)):
                # The whole chunk hit the cap; count it conservatively as one clipped call.
                MEM0_EMBED_TRUNCATED_CALLS["embed_batch"] += 1
            all_embeddings.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )
        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"embed_batch() returned {len(all_embeddings)} embeddings for "
                f"{len(texts)} texts using model '{self.config.model}'"
            )
        return all_embeddings

    _mem0_openai_embedding.OpenAIEmbedding.embed = _vllm_compatible_embed
    _mem0_openai_embedding.OpenAIEmbedding.embed_batch = _vllm_compatible_embed_batch
except Exception as _e:  # pragma: no cover
    print(f"[mem0] WARN: could not apply vLLM embed-dimensions compat shim: {_e}", flush=True)

# --- Clock-sync arm: nothing to patch on mem0ai 2.x -------------------------
# 0.1.118 froze "Today's date is <YYYY-MM-DD>." into FACT_RETRIEVAL_PROMPT as
# a MODULE-LEVEL f-string at import time. So the adapter had to rewrite that
# line per add() for the clock-sync arm. mem0ai 2.x removes that path: the
# additive extraction prompt resolves its "## Observation Date" and
# "## Current Date" sections PER CALL via ``_resolve_dates``
# (mem0/configs/prompts.py:297-303, rendered at :348-349), which defaults to
# ``datetime.now(timezone.utc)``. libfaketime therefore covers the whole
# surface on its own, so this file deletes the monkeypatch instead of
# porting it.
#
# A fail-closed version check replaces the monkeypatch. The patch's absence
# is safe only on the 2.x codebase. If BENCH_CLOCKSYNC=1 ever runs against a
# 0.1.x install (frozen prompt, no patch), the arm would silently reason
# about the real 2026 date. This check cannot assert the faked YEAR, because
# setup runs at real time and the clock steps per session. So it asserts the
# CODEBASE instead: package version 2.x, and the absence of 0.1.x's
# ``get_fact_retrieval_messages`` symbol in mem0.memory.main (the
# frozen-prompt entry point). Either failure exits.
if os.environ.get("BENCH_CLOCKSYNC") == "1":
    from importlib.metadata import version as _pkg_version  # noqa: E402
    import mem0.memory.main as _mem0_main  # noqa: E402

    _mem0_version = _pkg_version("mem0ai")
    if not _mem0_version.startswith("2."):
        raise SystemExit(
            f"[mem0] BENCH_CLOCKSYNC=1 requires mem0ai 2.x (per-call prompt dates); "
            f"installed mem0ai=={_mem0_version} freezes the extraction prompt date at "
            f"import and needs the deleted monkeypatch. Refusing to run a mislabelled arm."
        )
    if hasattr(_mem0_main, "get_fact_retrieval_messages"):
        raise SystemExit(
            "[mem0] BENCH_CLOCKSYNC=1: mem0.memory.main still exposes "
            "get_fact_retrieval_messages (the 0.1.x frozen-prompt path). Refusing to run: "
            "the per-add() 'Today's date' rewrite this adapter used to apply is gone."
        )
    print(f"[mem0] BENCH_CLOCKSYNC=1: mem0ai=={_mem0_version}, extraction-prompt dates "
          f"resolve per add() from the faked clock (prompts.py:_resolve_dates)", flush=True)

# The 0.1.x frozen-prompt monkeypatch that used to live here is DELETED, not
# disabled. On 2.x, ``mem0.memory.main`` no longer exposes
# ``get_fact_retrieval_messages`` at all, so the patch could only fall into
# its own except branch and print "WARN: could not apply clock-sync
# fact-prompt patch" on every clocksync run (observed in the 2026-07-26
# smoke). The fail-closed version guard above now carries the guarantee: it
# refuses to run BENCH_CLOCKSYNC=1 against any codebase where that symbol
# still exists.

# --- Wire in the upstream MemConflict Evaluation helpers -------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMCONFLICT_EVAL_DIR = os.environ.get(
    "MEMCONFLICT_EVAL_DIR",
    os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Evaluation"),
)
MEMCONFLICT_EVAL_DIR = os.path.abspath(MEMCONFLICT_EVAL_DIR)
if MEMCONFLICT_EVAL_DIR not in sys.path:
    sys.path.insert(0, MEMCONFLICT_EVAL_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# The shared, provider-agnostic harness modules (eval_common, llm_reasoning,
# the scorers) live in ../benchmark. This makes them importable regardless of
# launch cwd.
_SHARED_HARNESS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark"))
if _SHARED_HARNESS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_HARNESS_DIR)

from dotenv import load_dotenv

import eval_common  # noqa: E402
from eval_common import (  # noqa: E402  (re-exports keep old imports working)
    Pair_Exchange_Turns,
    Parse_Session_Timestamp,
    ProviderBinding,
    add_common_eval_args,
    opt_int,
    record_provider_retrieval,
)

load_dotenv()
load_dotenv(os.path.join(CURRENT_DIR, ".env"))


# --------------------------------------------------------------------------
# mem0 backend
# --------------------------------------------------------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


# --- Recall width / threshold (mem0 2.x search knobs) -----------------------
# MEM0_SEARCH_TOP_K: what the adapter ASKS mem0 for. 20 is mem0 2.x's own
#   provider default (``Memory.search(..., top_k: int = 20)``, main.py:1539).
#   The adapter requests the provider-native width and slices to the shared
#   --top_k for the answer context, the same shape as the RetainDB (10 -> 5)
#   and Supermemory (10 -> 5) arms. The extra rows are recorded only in the
#   diagnostic ranked list; they never reach the answer prompt, so the
#   configuration under test stays unchanged.
# MEM0_SEARCH_THRESHOLD: set explicitly to 0.0, NOT the vendor default 0.1.
#   2.x's threshold cuts on a BLENDED semantic+BM25+entity score, so 0.1 can
#   drop memories that sit inside the shared top-K every other provider gets.
#   That is a harness asymmetry, not a quality signal (same reasoning as the
#   Supermemory SUPERMEMORY_SEARCH_THRESHOLD decision, CLAUDE.md). Set
#   MEM0_SEARCH_THRESHOLD=0.1 to run the vendor default as a LABELED
#   sensitivity arm; it is not the headline config.
MEM0_SEARCH_TOP_K = int(_env("MEM0_SEARCH_TOP_K", "20"))
MEM0_SEARCH_THRESHOLD = float(_env("MEM0_SEARCH_THRESHOLD", "0.0"))


def Build_Mem0_Config(
    llm_provider: str,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    llm_temperature: float,
    llm_max_tokens: int,
    embedder_provider: str,
    embedder_model: str,
    embedder_base_url: Optional[str],
    embedder_dims: int,
    vector_store_provider: str,
    vector_store_path: Optional[str],
    collection_name: str,
    vector_store_host: Optional[str] = None,
    vector_store_port: Optional[int] = None,
    vector_store_url: Optional[str] = None,
    vector_store_on_disk: bool = False,
) -> Dict[str, Any]:
    """Assemble the mem0 ``Memory.from_config`` dict from resolved knobs.

    This function surfaces every mem0-tunable setting here, so a manifest or
    log records the exact config. The LLM block points mem0's INTERNAL
    extraction and update-memory calls at the serving model (best-effort
    ruling). The embedder block is the shared retrieval-embedding surface
    (bge-small-en-v1.5 offline).

    VECTOR STORE has two modes (mirrors Hindsight's embedded-pg0 vs
    shared-pg):
      * SERVER (``vector_store_url`` or ``vector_store_host``+``port`` set):
        connect to a CENTRAL qdrant server that every shard shares. This mode
        is what enables sharded runs. mem0's embedded qdrant locks its
        on-disk ``path`` to a single process, so N shard processes cannot
        share one embedded store; they must point at a server instead.
        Personas stay isolated by ``user_id``. A run or shard is isolated by
        its ``collection_name``, the qdrant analog of Hindsight's per-run
        database inside the shared postmaster.
      * EMBEDDED (only ``vector_store_path`` set): a disposable per-process
        on-disk store, the host-smoke default. mem0 wipes the path on init,
        so each run starts fresh.
    """
    llm_config: Dict[str, Any] = {
        "model": llm_model,
        "temperature": llm_temperature,
        "max_tokens": llm_max_tokens,
    }
    if llm_api_key:
        llm_config["api_key"] = llm_api_key
    if llm_base_url:
        # mem0's OpenAI LLM reads openai_base_url, or the OPENAI_BASE_URL env var.
        llm_config["openai_base_url"] = llm_base_url

    embedder_config: Dict[str, Any] = {
        "model": embedder_model,
        "embedding_dims": embedder_dims,
    }
    if embedder_base_url:
        # For provider="openai", this points at an OpenAI-compatible embeddings
        # endpoint, for example the shared vllm-embed server running bge-small-en-v1.5.
        embedder_config["openai_base_url"] = embedder_base_url

    vector_config: Dict[str, Any] = {
        "collection_name": collection_name,
        "embedding_model_dims": embedder_dims,
        "on_disk": vector_store_on_disk,
    }
    server_mode = bool(vector_store_url or (vector_store_host and vector_store_port))
    if server_mode:
        # CENTRAL server: connect over the network. Every shard shares this server.
        if vector_store_url:
            vector_config["url"] = vector_store_url
        else:
            vector_config["host"] = vector_store_host
            vector_config["port"] = int(vector_store_port)
    else:
        # EMBEDDED: a plain on-disk path, with no server process. mem0 wipes it fresh on init.
        vector_config["path"] = vector_store_path

    return {
        "llm": {"provider": llm_provider, "config": llm_config},
        "embedder": {"provider": embedder_provider, "config": embedder_config},
        "vector_store": {"provider": vector_store_provider, "config": vector_config},
        # v1.1 is mem0's current add/search response shape: {"results": [...]}.
        "version": "v1.1",
    }


def _vector_store_is_server(config: Dict[str, Any]) -> bool:
    vc = config.get("vector_store", {}).get("config", {})
    return bool(vc.get("url") or (vc.get("host") and vc.get("port")))


def _wait_for_qdrant(config: Dict[str, Any], timeout_s: float = 120.0) -> None:
    """Wait, with a bound, for the central qdrant server (server mode only).

    depends_on only orders container START, not readiness. So, like the
    Hindsight entrypoint's bounded DB retry, this function polls qdrant
    before the run touches it. This function does nothing in embedded mode.
    """
    if not _vector_store_is_server(config):
        return
    vc = config["vector_store"]["config"]
    from qdrant_client import QdrantClient  # mem0 already depends on this; safe to import
    deadline = time.time() + timeout_s
    target = vc.get("url") or f"{vc.get('host')}:{vc.get('port')}"
    attempt = 0
    while True:
        attempt += 1
        try:
            client = QdrantClient(url=vc["url"]) if vc.get("url") else QdrantClient(host=vc["host"], port=int(vc["port"]))
            client.get_collections()
            client.close()
            print(f"[mem0] qdrant server reachable at {target} (attempt {attempt})", flush=True)
            return
        except Exception as e:  # pragma: no cover
            if time.time() >= deadline:
                raise RuntimeError(f"qdrant server {target} not reachable after {timeout_s:.0f}s: {e}")
            time.sleep(2.0)


def _reset_qdrant_collection(config: Dict[str, Any]) -> None:
    """Delete this run's or shard's collections so a GENERATE stage starts fresh.

    This function applies to server mode only. Unlike the embedded path,
    which mem0 wipes on init, a server collection persists. So a re-run
    under the same RUN_TAG would ingest on top of stale points. Deleting the
    per-run collection first mirrors the Hindsight entrypoint's
    fresh-per-run-database guarantee. This function is idempotent: a missing
    collection is fine.

    mem0 2.x's entity linking lazily creates a SECOND collection named
    ``<collection_name>_entities`` (``_entity_collection_name``,
    mem0/memory/main.py:303-304, used by the ``entity_store`` property at
    :425-432). The function must drop both collections. Leaving stale entity
    rows behind would let a previous run's entity graph steer this run's
    hybrid ranking.
    """
    if not _vector_store_is_server(config):
        return
    vc = config["vector_store"]["config"]
    from qdrant_client import QdrantClient
    client = QdrantClient(url=vc["url"]) if vc.get("url") else QdrantClient(host=vc["host"], port=int(vc["port"]))
    name = vc["collection_name"]
    try:
        existing = {c.name for c in client.get_collections().collections}
        for target in (name, f"{name}_entities"):
            if target in existing:
                client.delete_collection(collection_name=target)
                print(f"[mem0] reset: deleted existing collection '{target}' for a fresh generate", flush=True)
            else:
                print(f"[mem0] reset: collection '{target}' did not exist (fresh)", flush=True)
    finally:
        client.close()


def Setup_Mem0(config: Dict[str, Any], reset_collection: bool = False) -> Memory:
    """Build a mem0 ``Memory`` from the resolved config dict.

    In server mode, this function waits for the central qdrant to be
    reachable. When ``reset_collection`` is set, during the GENERATE stage,
    it drops this run's collection first so ingestion starts clean.
    """
    _wait_for_qdrant(config)
    if reset_collection:
        _reset_qdrant_collection(config)
    # mem0 2.x's OpenAILLM silently REDIRECTS to openrouter.ai when
    # OPENROUTER_API_KEY is present in the env (mem0/llms/openai.py:43-48),
    # overriding the adapter's resolved base_url. That would send mem0's
    # internal extraction off-host and off-contract. Drop the key before
    # constructing Memory. The harness answer/judge path reads its own key
    # from its own config, and the --llm_api_key default already captured
    # any value at argparse time.
    os.environ.pop("OPENROUTER_API_KEY", None)
    memory = Memory.from_config(config)
    vc = config["vector_store"]["config"]
    where = vc.get("url") or (f"{vc.get('host')}:{vc.get('port')}" if vc.get("host") else f"embedded:{vc.get('path')}")
    print(f"[mem0] Memory initialised "
          f"(llm={config['llm']['provider']}/{config['llm']['config'].get('model')}, "
          f"embedder={config['embedder']['provider']}/{config['embedder']['config'].get('model')}, "
          f"vector_store={config['vector_store']['provider']}@{where} "
          f"collection={vc.get('collection_name')})", flush=True)
    return memory


def _iso(timestamp: Optional[datetime]) -> Optional[str]:
    return timestamp.isoformat() if timestamp else None


def _tally_events(add_result: Any, tally: Dict[str, int]) -> int:
    """Count mem0 add-result events (ADD/UPDATE/DELETE/NONE) and return net created.

    ``Memory.add`` returns ``{"results": [{"id","memory","event"}, ...]}``.

    NOTE (mem0ai 2.x): the split is STRUCTURALLY all-ADD. 2.x's pipeline runs
    single-pass additive extraction, and every emitted row is stamped
    ``event: "ADD"`` (mem0/memory/main.py:1165-1168). The 0.1.x per-fact
    ADD/UPDATE/DELETE/NONE decision phase no longer exists, so UPDATE, DELETE,
    and NONE can never be nonzero. This function keeps the four fields for
    row-schema continuity with the v3-contract mem0 files, and as evidence of
    the all-ADD shape in the run's own summary. Report the algorithm change as
    a finding; do not treat it as a bug to work around.
    """
    results = []
    if isinstance(add_result, dict):
        results = add_result.get("results") or []
    elif isinstance(add_result, list):
        results = add_result
    created = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event") or "").upper()
        if event == "ADD":
            tally["ADD"] += 1
            created += 1
        elif event == "UPDATE":
            tally["UPDATE"] += 1
        elif event == "DELETE":
            tally["DELETE"] += 1
        else:
            tally["NONE"] += 1
    return created


def _chunk(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i:i + size] for i in range(0, len(seq), max(1, size))]


def Add_Session_Dialogue_To_Mem0(
    memory: Memory,
    user_id: str,
    session_label: str,
    dialogue_messages: List[Dict[str, Any]],
    timestamp: Optional[datetime],
    granularity: str,
    event_tally: Dict[str, int],
    batch_size: int = 8,
) -> Tuple[float, int, int, int]:
    """Ingest one session's dialogue into mem0.

    ``granularity='batch'`` (default): one ``add`` per fixed-size window of
    ``batch_size`` messages. This is the cadence the MemConflict authors' own
    mem0 runner uses (``Evaluation/eval_memzero.py`` ingests in 8-message
    batches). Small windows keep each extraction focused; conflict resolution
    accumulates across batches within and across sessions. The reference
    targets the HOSTED mem0 platform (``MemoryClient``, ``api.mem0.ai``),
    which this project cannot use, because it selects self-hostable
    providers only. So this adapter runs the same cadence against the
    self-hosted ``Memory`` SDK.

    ``granularity='session'``: one ``add`` carrying every message. mem0
    extracts across the whole session in a single pass, using a big
    extraction prompt. This mode is an alternative arm, not the reference
    cadence.

    ``granularity='exchange'``: one ``add`` per user/assistant exchange.
    This is mem0's canonical per-turn ingestion; production integrations
    call ``add`` after each completed turn. It gives the finest-grained
    conflict resolution and makes the most internal LLM calls. It uses the
    shared ``Pair_Exchange_Turns`` grouping every provider uses.

    NOTE on time: the reference passes a real ``timestamp`` to the HOSTED
    ``add``. In OSS mem0 2.x, that kwarg is a managed-platform feature and
    raises ``ValueError`` (``main.py:1066``), so this function never passes
    it. Instead the session date goes into metadata TWICE: as
    ``metadata.timestamp``, which ``_result_created_at`` surfaces as the
    scorer's ``created_at`` unchanged from the earlier contract, and as
    ``metadata.created_at``, which 2.x honors as the stored payload's own
    ``created_at``/``updated_at`` (``main.py:1003-1005`` fills them only
    when the caller did not). That makes logical time correct in mem0's own
    records too, independent of libfaketime. This is belt and braces with
    the clock-sync arm, and it stays correct even in a default,
    real-clock run.

    All paths run ``infer=True`` (mem0's extraction), which is the whole
    point of the system. Returns (add_duration_ms, memories_created_net,
    adds_seen, adds_failed).
    """
    if not dialogue_messages:
        return 0.0, 0, 0, 0
    ts_iso = _iso(timestamp)
    metadata = {"timestamp": ts_iso, "session_id": session_label}
    if ts_iso:
        # mem0 2.x honors this field as the stored memory's created_at/updated_at.
        metadata["created_at"] = ts_iso
    start = time.time()
    created = 0
    add_calls = 0
    add_failures = 0

    def _add_one(messages: List[Dict[str, Any]], label: str) -> None:
        """Run one infer=True add() and count failures instead of swallowing them.

        In 2.x, extraction failure RAISES ``mem0.exceptions.LLMError``
        (``main.py:1267``), where 0.1.x returned an empty fact list instead.
        So a wedge or an unparseable extraction is now visible. This
        function uses the same log-and-continue idiom as before, because a
        lost window must not kill a 30-persona shard, but it surfaces the
        count as ``Total_Add_Calls_Failed`` so a strict wave can gate on 0.
        """
        nonlocal created, add_calls, add_failures
        try:
            result = memory.add(messages, user_id=user_id, metadata=dict(metadata), infer=True)
            created += _tally_events(result, event_tally)
            add_calls += 1
        except LLMError as e:  # extraction failed inside mem0's internal LLM call
            add_failures += 1
            print(f"[DEBUG] add({label}) LLM extraction FAILED user={user_id} "
                  f"session={session_label}: {e}", flush=True)
        except Exception as e:  # pragma: no cover
            add_failures += 1
            print(f"[DEBUG] add({label}) failed user={user_id} session={session_label}: {e}", flush=True)

    if granularity == "exchange":
        exchanges = Pair_Exchange_Turns(dialogue_messages)
        total = len(exchanges)
        cap_mode = total > 40
        for exch_idx, group in enumerate(exchanges, start=1):
            if not group:
                continue
            messages = [{"role": m["role"], "content": m["content"]} for m in group]
            if (not cap_mode) or (exch_idx % 10 == 0):
                print(f"[DEBUG] user={user_id} session {session_label} "
                      f"add exchange={exch_idx}/{total} msgs={len(messages)}", flush=True)
            _add_one(messages, "exchange")
    elif granularity == "batch":
        flat = [{"role": m["role"], "content": m["content"]} for m in dialogue_messages]
        batches = _chunk(flat, batch_size)
        total = len(batches)
        for b_idx, messages in enumerate(batches, start=1):
            print(f"[DEBUG] user={user_id} session {session_label} "
                  f"add batch={b_idx}/{total} msgs={len(messages)} (batch_size={batch_size})", flush=True)
            _add_one(messages, "batch")
    else:  # 'session'
        messages = [{"role": m["role"], "content": m["content"]} for m in dialogue_messages]
        print(f"[DEBUG] user={user_id} session {session_label} add_call msgs={len(messages)}", flush=True)
        _add_one(messages, "session")

    return (time.time() - start) * 1000.0, created, add_calls, add_failures


def _result_created_at(result: Dict[str, Any]) -> str:
    """Return the best temporal anchor for a recalled memory, for the scorer's created_at.

    Prefer the dataset session date this adapter stored in
    metadata.timestamp. Fall back to mem0's own wall-clock created_at.
    """
    metadata = result.get("metadata") or {}
    for candidate in (metadata.get("timestamp"), result.get("created_at")):
        if candidate:
            return str(candidate)
    return "Unknown Time"


def Search_Mem0_For_Question(
    memory: Memory, user_id: str, question_text: str, top_k: int,
    diag: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Recall memories for one question and map them to the scorer's memory shape.

    mem0 2.x signature notes (mem0/memory/main.py:1536-1551): the signature
    is keyword-only, ``limit`` is gone and replaced by ``top_k``, and the
    tenant MUST travel inside ``filters``. A top-level ``user_id=`` argument
    raises via ``_reject_top_level_entity_params``. ``reference_date=`` is a
    managed-platform-only argument and raises, so this function never passes
    it.

    WIDTH: this function asks mem0 for its own provider-default ``top_k``
    (20, ``MEM0_SEARCH_TOP_K``). The ANSWER CONTEXT is the top-``top_k``
    slice of that result, the shared K (5) — exactly the shape of the
    RetainDB (10 -> 5) and Supermemory (10 -> 5) minimal arms: request the
    provider's native width, then cut to the fairness-locked K. THRESHOLD:
    0.0 (``MEM0_SEARCH_THRESHOLD``), not the vendor default 0.1. See the
    constants above.

    ``diag`` (the per-persona ctx) receives the diagnostic capture: the raw
    ``memory.search`` response and the FULL mapped ranked list, up to 20
    rows. The returned list, what the answer model sees, is the top-5
    slice. So an offline depth curve on a mem0 file now runs to depth 20
    without changing the configuration under test.
    """
    start = time.time()
    response = memory.search(
        query=question_text,
        filters={"user_id": user_id},
        top_k=MEM0_SEARCH_TOP_K,
        threshold=MEM0_SEARCH_THRESHOLD,
    )
    duration_ms = (time.time() - start) * 1000.0

    if isinstance(response, dict):
        results = response.get("results") or []
    elif isinstance(response, list):
        results = response
    else:
        results = []

    ranked: List[Dict[str, Any]] = []
    for result in results:
        ranked.append({
            "memory": str(result.get("memory", "")),
            "created_at": _result_created_at(result),
            "score": result.get("score"),
            "id": result.get("id"),
        })
    record_provider_retrieval(diag, raw=response, ranked=ranked)
    return ranked[:top_k], duration_ms


# --------------------------------------------------------------------------
# Provider binding (the only mem0-specific surface the driver sees)
# --------------------------------------------------------------------------
class Mem0Binding(ProviderBinding):
    memory_system = "mem0"
    store_id_key = "Mem0_User_ID"
    runtime_summary_key = "Mem0_Runtime_Summary"
    stage_name = "mem0_answer_generation"
    stage_note = "mem0 retrieval and question answering"

    def __init__(self, memory: Optional[Memory], granularity: str, batch_size: int = 8):
        self.memory = memory
        self.granularity = granularity
        self.batch_size = batch_size

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown")
        # A unique user_id is mem0's native tenancy boundary, so it gives per-persona isolation.
        return {
            "store_id": f"mc_{persona_id[-8:]}_{uuid.uuid4().hex[:8]}",
            "persona_tag": persona_id[-8:],
            "total_created": 0,
            "total_add_calls": 0,
            "total_add_failed": 0,
            "events": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0},
        }

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        timestamp = Parse_Session_Timestamp(session_item)
        session_label = str(session_item.get("Session_ID", session_index))

        add_ms, created, add_calls, add_failed = Add_Session_Dialogue_To_Mem0(
            self.memory, ctx["store_id"], session_label, dialogue, timestamp,
            self.granularity, ctx["events"], batch_size=self.batch_size,
        )
        ctx["total_created"] += created
        ctx["total_add_calls"] += add_calls
        ctx["total_add_failed"] += add_failed
        print(f"[DEBUG] persona {ctx['persona_tag']} session {session_label} "
              f"add_calls={add_calls} add_failed={add_failed} created_net={created} "
              f"events={ctx['events']} ingest_ms={add_ms:.0f}", flush=True)
        return {
            "Dialogue_Added_To_Memory": created > 0,
            "Dialogue_Message_Count": len(dialogue),
            "Memories_Created_Net": created,
            "Add_Calls": add_calls,
            "Add_Calls_Failed": add_failed,
            "Retain_Granularity": self.granularity,
            "Session_Timestamp_Passed": _iso(timestamp),
            "Add_Duration_ms": add_ms,
        }

    def recall(self, ctx, question_text, top_k):
        return Search_Mem0_For_Question(self.memory, ctx["store_id"], question_text, top_k,
                                        diag=ctx)

    def end_persona(self, ctx):
        # Best-effort per-persona cleanup: drop this user's memories so a
        # shared store never leaks facts across personas. The per-run store
        # is disposable anyway, but the unique user_id plus this delete call
        # gives belt-and-braces isolation.
        try:
            if self.memory is not None:
                self.memory.delete_all(user_id=ctx["store_id"])
        except Exception as e:  # pragma: no cover
            print(f"[DEBUG] delete_all failed user={ctx.get('store_id')}: {e}", flush=True)

    def persona_count_extras(self, ctx):
        events = ctx["events"]
        return {
            "Total_Memories_Created_Net": ctx["total_created"],
            "Total_Add_Calls": ctx["total_add_calls"],
            # mem0 2.x raises LLMError on extraction failure. Expect 0 here.
            # A strict wave gates on this count, because a nonzero value means
            # memories were never written for that window, so recall for it
            # cannot succeed.
            "Total_Add_Calls_Failed": ctx["total_add_failed"],
            # Embedder-cap truncations (see the embed shim above): mem0 2.x
            # embeds the whole add() input as one related-memory search
            # query, and about 1.9% of 3-turn windows exceed bge-small's
            # 512-token cap. vLLM clips these server-side instead of
            # returning 400. This count makes the rate visible per run
            # rather than inferred from the measurement above.
            "Total_Embed_Truncated_Calls": (
                MEM0_EMBED_TRUNCATED_CALLS["embed"]
                + MEM0_EMBED_TRUNCATED_CALLS["embed_batch"]
            ),
            "Embed_Truncate_Tokens": _MEM0_EMBED_TRUNCATE_TOKENS,
            # This split is structurally all-ADD under mem0ai 2.x. See _tally_events.
            "Total_Event_ADD": events["ADD"],
            "Total_Event_UPDATE": events["UPDATE"],
            "Total_Event_DELETE": events["DELETE"],
            "Total_Event_NONE": events["NONE"],
        }


def Generate_User_Mem0_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    top_k: int,
    granularity: str,
    batch_size: int,
    config: Dict[str, Any],
    start_idx: int,
    end_idx: Optional[int],
    max_sessions: Optional[int],
    max_questions_per_session: Optional[int],
    overwrite_existing_answers: bool,
    reset_collection: bool = False,
) -> bool:
    print(f"[DEBUG] granularity={granularity}  "
          f"llm={config['llm']['config'].get('model')}  "
          f"embedder={config['embedder']['config'].get('model')}", flush=True)
    binding_holder: Dict[str, Any] = {}

    def setup():
        memory = Setup_Mem0(config, reset_collection=reset_collection)
        binding_holder["binding"].memory = memory

    def teardown():
        # mem0's vector store and history DB have no explicit close method.
        # The per-run store path is disposable scratch, so this is fine.
        return None

    binding = Mem0Binding(memory=None, granularity=granularity, batch_size=batch_size)
    binding_holder["binding"] = binding
    return eval_common.run_eval(
        binding=binding,
        input_jsonl_path=input_jsonl_path,
        output_jsonl_path=output_jsonl_path,
        output_json_path=output_json_path,
        top_k=top_k,
        start_idx=start_idx,
        end_idx=end_idx,
        max_sessions=max_sessions,
        max_questions_per_session=max_questions_per_session,
        overwrite_existing_answers=overwrite_existing_answers,
        setup=setup,
        teardown=teardown,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mem0 evaluation on the MemConflict dataset")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(CURRENT_DIR, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(CURRENT_DIR, "Results", "mem0_results.jsonl"),
        default_output_json_path=os.path.join(CURRENT_DIR, "Results", "mem0_results.json"),
        top_k_help="How many recalled memories the answer LLM sees (up to 5 are always stored "
                   "for white-box scoring). NOTE the upstream MemConflict adapters answer "
                   "from top-3, so use --top_k 3 for strict answer-accuracy comparability.",
        default_start_idx=int(_env("START_IDX", "0")),
        default_end_idx=opt_int(_env("END_IDX")),
    )
    parser.add_argument("--retain_granularity", type=str,
                        default=_env("MEM0_RETAIN_GRANULARITY", "batch"),
                        choices=["batch", "session", "exchange"],
                        help="'batch' (default): one add() per --mem0_add_batch_size-message window — "
                             "the cadence the MemConflict authors' own mem0 runner uses (eval_memzero.py "
                             "ingests in 8-message batches). 'session': one add() per whole session. "
                             "'exchange': one add() per user/assistant exchange (mem0's canonical "
                             "per-turn ingestion; finest-grained conflict resolution, most LLM calls).")
    parser.add_argument("--mem0_add_batch_size", type=int,
                        default=int(_env("MEM0_ADD_BATCH_SIZE", "8")),
                        help="Messages per add() in the 'batch' arm (default 8, matching the MemConflict "
                             "authors' eval_memzero.py). Ignored by the 'session'/'exchange' arms.")
    # --- mem0 internal LLM (fact extraction and update-memory decision) ------
    parser.add_argument("--llm_provider", type=str, default=_env("MEM0_LLM_PROVIDER", "openai"))
    parser.add_argument("--llm_model", type=str,
                        default=_env("MEM0_LLM_MODEL", _env("OPENAI_MODEL", "openai/gpt-oss-120b")))
    parser.add_argument("--llm_base_url", type=str,
                        default=_env("MEM0_LLM_BASE_URL", _env("OPENAI_BASE_URL")))
    parser.add_argument("--llm_api_key", type=str,
                        default=_env("MEM0_LLM_API_KEY", _env("OPENAI_API_KEY", _env("OPENROUTER_API_KEY"))))
    parser.add_argument("--llm_temperature", type=float,
                        default=float(_env("MEM0_LLM_TEMPERATURE", "0.7")))
    parser.add_argument("--llm_max_tokens", type=int,
                        default=int(_env("MEM0_LLM_MAX_TOKENS", "2048")))
    # --- mem0 embedder (the shared retrieval-embedding surface) ---------------
    parser.add_argument("--embedder_provider", type=str,
                        default=_env("MEM0_EMBEDDER_PROVIDER", "huggingface"),
                        help="'huggingface' (local sentence-transformers, fully offline — the smoke "
                             "default) or 'openai' (an OpenAI-compatible embeddings endpoint, e.g. the "
                             "shared vllm-embed serving bge-small-en-v1.5 for offline/Docker runs).")
    parser.add_argument("--embedder_model", type=str,
                        default=_env("MEM0_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--embedder_base_url", type=str, default=_env("MEM0_EMBEDDER_BASE_URL"))
    parser.add_argument("--embedder_dims", type=int, default=int(_env("MEM0_EMBEDDER_DIMS", "384")))
    # --- mem0 vector store ----------------------------------------------------
    # Two modes: EMBEDDED (default: on-disk path, no server, the host-smoke
    # path) or SERVER (a central qdrant server shared by every shard,
    # required for sharded runs; see Build_Mem0_Config). Passing
    # --vector_store_host/--vector_store_url (or MEM0_QDRANT_HOST /
    # MEM0_QDRANT_URL) switches to server mode.
    parser.add_argument("--vector_store_provider", type=str,
                        default=_env("MEM0_VECTOR_STORE", "qdrant"))
    parser.add_argument("--vector_store_path", type=str,
                        default=_env("MEM0_VECTOR_STORE_PATH",
                                     os.path.join("/tmp", f"mem0_qdrant_{uuid.uuid4().hex[:8]}")),
                        help="Embedded mode: on-disk path for the local vector store (disposable scratch).")
    parser.add_argument("--vector_store_host", type=str, default=_env("MEM0_QDRANT_HOST"),
                        help="Server mode: central qdrant host (e.g. the 'qdrant' compose service). "
                             "Set with --vector_store_port to share one store across shards.")
    parser.add_argument("--vector_store_port", type=lambda v: opt_int(v),
                        default=opt_int(_env("MEM0_QDRANT_PORT", "6333")),
                        help="Server mode: central qdrant REST port (default 6333).")
    parser.add_argument("--vector_store_url", type=str, default=_env("MEM0_QDRANT_URL"),
                        help="Server mode: full qdrant URL (alternative to host/port).")
    parser.add_argument("--vector_store_on_disk", action="store_true",
                        default=_env("MEM0_QDRANT_ON_DISK", "").lower() in ("1", "true", "yes", "on"),
                        help="Store vectors on disk in the server (persistent/mmap) rather than in RAM.")
    parser.add_argument("--reset_collection", action="store_true",
                        default=_env("MEM0_RESET_COLLECTION", "").lower() in ("1", "true", "yes", "on"),
                        help="Server mode: delete this run's collection before ingest so a GENERATE stage "
                             "starts fresh (embedded mode is always fresh). Set by the entrypoint for generate.")
    parser.add_argument("--collection_name", type=str, default=_env("MEM0_COLLECTION", "memconflict"))
    args = parser.parse_args()

    config = Build_Mem0_Config(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        embedder_provider=args.embedder_provider,
        embedder_model=args.embedder_model,
        embedder_base_url=args.embedder_base_url,
        embedder_dims=args.embedder_dims,
        vector_store_provider=args.vector_store_provider,
        vector_store_path=os.path.abspath(args.vector_store_path),
        collection_name=args.collection_name,
        vector_store_host=args.vector_store_host,
        vector_store_port=args.vector_store_port,
        vector_store_url=args.vector_store_url,
        vector_store_on_disk=args.vector_store_on_disk,
    )

    # eval_common.run_eval() returns False, not an exception, on a fatal
    # error, so per-persona incremental output survives a mid-run crash.
    # This block propagates that as a nonzero exit, so the entrypoint's
    # `set -e` (and preflight_rows.py) actually stop the run instead of
    # scoring a partial file. This is the same contract as the other
    # adapters' __main__.
    ok = Generate_User_Mem0_Eval(
        input_jsonl_path=os.path.abspath(args.input_jsonl_path),
        output_jsonl_path=os.path.abspath(args.output_jsonl_path),
        output_json_path=os.path.abspath(args.output_json_path),
        top_k=args.top_k,
        granularity=args.retain_granularity,
        batch_size=args.mem0_add_batch_size,
        config=config,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
        reset_collection=args.reset_collection,
    )
    # This code calls os._exit, not SystemExit, because mem0ai and
    # qdrant-client leave non-daemon threads (connection-pool and
    # thread-pool workers) that keep the interpreter alive after main
    # returns. Without os._exit, the process never hands control back to
    # the entrypoint, and the container hangs "running" forever (itsmoke2
    # needed a docker stop). All output files are already flushed and
    # closed by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)
