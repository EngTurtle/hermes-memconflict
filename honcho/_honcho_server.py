"""Lifecycle manager for a self-hosted Honcho server (pinned tag v3.0.9).

Self-hosted Honcho is not a Python SDK with an embedded store, and not a
single binary either. It is THREE cooperating pieces:

  * a FastAPI HTTP API (``src/main.py``), the surface the ``honcho-ai`` SDK
    talks to;
  * a SEPARATE deriver worker process (``python -m src.deriver``) that drains
    the work queue and calls the internal LLM to build peer representations;
  * PostgreSQL with pgvector, which holds messages, conclusions, and
    embeddings.

So this module is a hybrid of the Hindsight (external database) and
Supermemory (spawned server with an internal LLM) managers. It has TWO LLM
roles, kept strictly apart (docs/DECISIONS.md):

  * Honcho's INTERNAL models — deriver, dialectic (five reasoning levels),
    summary, dream, peer card. They are configured on the SPAWNED CHILDREN
    through Honcho's own namespaced env vars, fed here from ``HONCHO_LLM_*``.
  * The shared ANSWER and JUDGE model — the fairness-locked harness model
    the adapter reaches through ``eval_common`` and ``OPENAI_*`` in THIS
    process. This module never reads or writes those in the parent.

Two modes:

  * SPAWN (default): create a per-run database, provision it, fix the vector
    dimension, then launch the API and the deriver as child processes with
    ephemeral ports and per-run logs.
  * ATTACH (``HONCHO_SERVER_MODE=shared`` or ``HONCHO_BASE_URL`` set): health
    check an already-running API. Every shard of a sharded run attaches to
    one central API and deriver pair.

THE VECTOR-DIMENSION FIX IS MANDATORY BELOW 1536. The alembic migrations
hardcode ``Vector(1536)`` (``migrations/versions/917195d9b5e9_...:31``,
``119a52b73c60_...:45,53``, ``a1b2c3d4e5f6_initial_schema.py:366``), while
``src/models.py`` sizes the same columns from ``EMBEDDING_VECTOR_DIMENSIONS``.
``scripts/provision_db.py`` only replays migrations, so a 384-dim embedder
leaves ``public.documents.embedding`` and ``public.message_embeddings.embedding``
at ``vector(1536)``. Both the API lifespan and the deriver then refuse to
start (``src/startup/embedding_validator.py``). The vendor ships the repair:
``scripts/configure_embeddings.py --yes`` drops the two HNSW indexes, runs
``ALTER COLUMN embedding TYPE vector(<dim>) USING NULL``, and recreates the
indexes from their snapshotted definitions. This module runs that script; the
raw-SQL fallback below exists only for an install where the script is absent.
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SERVER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "external", "honcho"))

#: The two pgvector columns v3.0.9's startup validator checks
#: (``src/startup/embedding_validator.py:38``).
EMBEDDING_TABLES: Tuple[str, ...] = ("documents", "message_embeddings")

_SAFE_DB_NAME = re.compile(r"[^a-zA-Z0-9_]")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def sanitize_db_name(name: str) -> str:
    """Make a run tag safe as a bare Postgres identifier."""
    return _SAFE_DB_NAME.sub("_", name).lower()[:48] or "run"


def _dsn_to_libpq(dsn: str) -> str:
    """Strip SQLAlchemy's ``+psycopg`` driver tag so psycopg accepts the DSN."""
    return dsn.replace("postgresql+psycopg://", "postgresql://", 1)


def _dsn_with_database(dsn: str, database: str) -> str:
    """Swap the database name in a DSN, keeping user, host, port, and query."""
    head, _, tail = dsn.rpartition("/")
    query = ""
    if "?" in tail:
        _, _, query = tail.partition("?")
        query = "?" + query
    return f"{head}/{database}{query}"


# --------------------------------------------------------------------------
# Database bootstrap
# --------------------------------------------------------------------------
def create_database(dsn: str, database: str, drop_existing: bool = True) -> None:
    """Create a per-run database on the server the DSN points at.

    A host smoke and a spawn-mode run each want a disposable database, so
    a re-run under the same tag can never answer from the previous run's
    conclusions. CREATE DATABASE cannot run inside a transaction, so this
    uses autocommit.
    """
    import psycopg

    admin_dsn = _dsn_to_libpq(_dsn_with_database(dsn, "postgres"))
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        if drop_existing:
            # A previous run may still hold sessions open. Terminate them
            # first: DROP DATABASE fails while any backend is connected.
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (database,),
            )
            conn.execute(f'DROP DATABASE IF EXISTS "{database}"')
        conn.execute(f'CREATE DATABASE "{database}"')
    print(f"[honcho] created database '{database}'", flush=True)


def read_vector_dims(dsn: str, schema: str = "public") -> Dict[str, int]:
    """Return ``{table: declared dimension}`` for the embedding columns.

    pgvector stores the declared dimension directly in ``atttypmod``, with no
    header offset, which is the same read the server's own validator makes
    (``embedding_validator.py:119-135``). A value of -1 means the column has
    no declared dimension.
    """
    import psycopg

    query = """
        SELECT c.relname AS table_name, a.atttypmod AS typmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = %s AND c.relkind = 'r'
          AND c.relname = ANY(%s) AND a.attname = 'embedding'
    """
    with psycopg.connect(_dsn_to_libpq(dsn)) as conn:
        rows = conn.execute(query, (schema, list(EMBEDDING_TABLES))).fetchall()
    return {row[0]: int(row[1]) for row in rows}


def _apply_dim_fix_sql(dsn: str, target_dim: int, schema: str = "public") -> None:
    """Raw-SQL fallback for the vector-dimension fix.

    This runs only when ``scripts/configure_embeddings.py`` is missing or
    fails. It performs the same three steps in one transaction: snapshot and
    drop the HNSW indexes, retype the columns, recreate the indexes from the
    snapshotted definitions. ``USING NULL`` is safe because this only ever
    runs on a freshly provisioned, empty database; the function refuses
    otherwise, exactly as the vendor script does.
    """
    import psycopg

    with psycopg.connect(_dsn_to_libpq(dsn)) as conn:
        with conn.cursor() as cur:
            for table in EMBEDDING_TABLES:
                cur.execute(f'SELECT count(*) FROM "{schema}"."{table}" '
                            "WHERE embedding IS NOT NULL")
                populated = cur.fetchone()[0]
                if populated:
                    raise RuntimeError(
                        f"refusing to retype {schema}.{table}.embedding: "
                        f"{populated} rows already carry vectors")
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = %s AND tablename = ANY(%s) "
                "AND indexdef ILIKE '%%hnsw%%'",
                (schema, list(EMBEDDING_TABLES)),
            )
            index_defs = cur.fetchall()
            for index_name, _ in index_defs:
                cur.execute(f'DROP INDEX "{schema}"."{index_name}"')
            for table in EMBEDDING_TABLES:
                cur.execute(f'ALTER TABLE "{schema}"."{table}" '
                            f"ALTER COLUMN embedding TYPE vector({target_dim}) USING NULL")
            for _, index_def in index_defs:
                cur.execute(index_def)
        conn.commit()
    print(f"[honcho] dim fix applied by fallback SQL (target={target_dim})", flush=True)


def apply_embedding_dim_fix(
    dsn: str,
    target_dim: int,
    server_dir: str = DEFAULT_SERVER_DIR,
    python_exe: Optional[str] = None,
    schema: str = "public",
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Bring the pgvector columns to ``target_dim``. Idempotent.

    Returns a small report dict, so a caller can log what actually changed.
    A compose topology that provisions the database outside this process must
    call this function too; otherwise both children abort at startup.
    """
    before = read_vector_dims(dsn, schema)
    if before and all(v == target_dim for v in before.values()):
        return {"changed": False, "before": before, "after": before, "method": "none"}

    script = os.path.join(server_dir, "scripts", "configure_embeddings.py")
    method = "vendor_script"
    if os.path.isfile(script):
        child_env = dict(env or os.environ)
        child_env["DB_CONNECTION_URI"] = dsn
        child_env["DB_SCHEMA"] = schema
        child_env["EMBEDDING_VECTOR_DIMENSIONS"] = str(target_dim)
        proc = subprocess.run(
            [python_exe or sys.executable, script, "--yes"],
            cwd=server_dir, env=child_env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"[honcho] configure_embeddings.py failed (rc={proc.returncode}); "
                  f"falling back to raw SQL\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}",
                  flush=True)
            _apply_dim_fix_sql(dsn, target_dim, schema)
            method = "fallback_sql"
    else:
        _apply_dim_fix_sql(dsn, target_dim, schema)
        method = "fallback_sql"

    after = read_vector_dims(dsn, schema)
    missing = [t for t in EMBEDDING_TABLES if t not in after]
    wrong = {t: d for t, d in after.items() if d != target_dim}
    if missing or wrong:
        raise RuntimeError(
            f"embedding dim fix incomplete: missing={missing} wrong={wrong} "
            f"(target={target_dim}). The API and the deriver both refuse to boot "
            f"on a mismatch (src/startup/embedding_validator.py).")
    print(f"[honcho] pgvector columns at dim {target_dim} "
          f"(before={before}, method={method})", flush=True)
    return {"changed": True, "before": before, "after": after, "method": method}


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
class HonchoServer:
    """Own the API process, the deriver process, and the run database."""

    def __init__(
        self,
        server_dir: Optional[str] = None,
        port: Optional[int] = None,
        base_url: Optional[str] = None,
        pg_dsn: Optional[str] = None,
        create_db: Optional[bool] = None,
        db_name: Optional[str] = None,
        run_dir: Optional[str] = None,
        # Honcho's internal LLM (deriver, dialectic, summary, dream).
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_max_output_tokens: Optional[int] = None,
        llm_thinking_effort: Optional[str] = None,
        deriver_max_output_tokens: Optional[int] = None,
        deriver_presence_penalty: Optional[float] = None,
        # Honcho's embedder (the shared retrieval-embedding surface).
        embedder_model: Optional[str] = None,
        embedder_base_url: Optional[str] = None,
        embedder_api_key: Optional[str] = None,
        embedder_dims: Optional[int] = None,
        deriver_workers: Optional[int] = None,
        deriver_flush: Optional[bool] = None,
        summary_enabled: Optional[bool] = None,
        peer_card_enabled: Optional[bool] = None,
    ):
        self.server_dir = os.path.abspath(server_dir or _env("HONCHO_SERVER_DIR", DEFAULT_SERVER_DIR))
        # ATTACH mode: an external API already runs, so this process spawns
        # nothing and only health checks the URL.
        self.attach_url = base_url or _env("HONCHO_BASE_URL")
        if _env("HONCHO_SERVER_MODE", "spawn") == "shared" and not self.attach_url:
            self.attach_url = "http://honcho-api:8000"
        self.port = port or int(_env("HONCHO_SERVER_PORT", "0") or 0) or _free_port()
        self.base_url = self.attach_url or f"http://127.0.0.1:{self.port}"

        self.pg_dsn = pg_dsn or _env(
            "HONCHO_PG_DSN",
            "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
        )
        self.create_db = _truthy(_env("HONCHO_PG_CREATE_DB"), False) if create_db is None else create_db
        run_tag = _env("RUN_TAG", "run")
        self.db_name = db_name or _env("HONCHO_PG_DB") or f"honcho_{sanitize_db_name(run_tag)}"
        if self.create_db:
            self.pg_dsn = _dsn_with_database(self.pg_dsn, self.db_name)

        self.run_dir = os.path.abspath(
            run_dir or _env("HONCHO_RUN_DIR", os.path.join(CURRENT_DIR, ".honcho_runs", run_tag))
        )
        self.api_log_path = os.path.join(self.run_dir, "api.log")
        self.deriver_log_path = os.path.join(self.run_dir, "deriver.log")

        self.llm_model = llm_model or _env("HONCHO_LLM_MODEL", "gpt-5.4-mini")
        self.llm_base_url = llm_base_url or _env("HONCHO_LLM_BASE_URL")
        self.llm_api_key = llm_api_key or _env("HONCHO_LLM_API_KEY") \
            or _env("OPENROUTER_API_KEY") or "local"
        self.llm_max_output_tokens = int(
            llm_max_output_tokens if llm_max_output_tokens is not None
            else int(_env("HONCHO_LLM_MAX_OUTPUT_TOKENS", "8192"))
        )
        # Reasoning budget for the internal roles. Honcho maps this onto the
        # request's `reasoning_effort` (src/llm/backends/openai.py:297). It
        # is unset by default, so the model's own default applies. On a
        # reasoning model it is not optional: gpt-oss-20b at the default
        # effort spent the ENTIRE 8192-token budget on reasoning for the
        # deriver's observation call and returned empty content, which
        # Honcho logged as "Deriver generated zero observations" after a
        # 122-second call. The user's messages then produced no conclusions
        # at all, so the whole user representation was empty. Same reason the
        # harness answer role runs MEMCONFLICT_REASONING_EFFORT=low.
        self.llm_thinking_effort = llm_thinking_effort or _env("HONCHO_LLM_THINKING_EFFORT")
        # DERIVER-ONLY output cap, separate from llm_max_output_tokens because
        # HONCHO_LLM_MAX_OUTPUT_TOKENS feeds EVERY role (deriver, summary, the
        # five dialectic levels, both dream specialists) through
        # _model_config_env below. The deriver is the one role that measurably
        # runs away on qwen3.5-4b: smoke hn_smkft_p0 stored 18 of 79 documents
        # pinned at the 8192-token cap, mean 41,189 chars, unique-sentence
        # ratio 0.181 (one sentence repeated 341 times in the worst row). Those
        # rows then overflowed the 32,768-token serving window at recall and
        # 400-killed Honcho's own dialectic. A real observation has a median
        # length of 241 chars, so 2048 tokens cannot truncate a healthy one and
        # bounds a residual runaway to a quarter of the old damage. Precedent:
        # the Hindsight retain cap of 4096. Lowering the GLOBAL knob instead
        # would also cut the dialectic answer and the dream specialists, whose
        # budgets are not the defect.
        self.deriver_max_output_tokens = int(
            deriver_max_output_tokens if deriver_max_output_tokens is not None
            else int(_env("HONCHO_DERIVER_MAX_OUTPUT_TOKENS", "2048"))
        )
        # Vendor-exposed sampling knob (ModelConfig.presence_penalty,
        # src/config.py:253) that reaches the request body through
        # build_config_extra_params (src/llm/request_builder.py:33) and
        # backends/openai.py:316-324. vLLM cannot serve presence_penalty as a
        # server-side default -- get_diff_sampling_param allowlists only
        # repetition_penalty, temperature, top_k, top_p, min_p, and
        # max_new_tokens -- so every provider-internal call runs the Qwen card
        # set MINUS presence_penalty unless it is sent per request. 1.5 is the
        # card value the harness answer role already uses
        # (benchmark/docker/answer_env.sh:111). Set empty to send nothing.
        _pp = (str(deriver_presence_penalty) if deriver_presence_penalty is not None
               else _env("HONCHO_DERIVER_PRESENCE_PENALTY", "1.5"))
        self.deriver_presence_penalty = float(_pp) if _pp else None
        # frequency_penalty is the only count-scaling repetition knob the
        # OpenAI backend forwards (openai.py:316-324). presence_penalty is a
        # flat one-time -1.5 per already-seen token, so it saturates after the
        # first occurrence and cannot stop a sustained verbatim loop; measured
        # 5.2% of stored documents were repetition loops pinned at the 2048
        # cap WITH presence 1.5 active (smkft3). At 0.3, a repeated 30-token
        # sentence loses ~9 logits by its 5th repetition. Not a Qwen-card
        # value; recorded in docs/DECISIONS.md with the probe measurement.
        # Set empty to send nothing.
        _fp = _env("HONCHO_DERIVER_FREQUENCY_PENALTY", "0.3")
        self.deriver_frequency_penalty = float(_fp) if _fp else None
        # `DIALECTIC_MAX_INPUT_TOKENS`, the vendor's own bound on the dialectic
        # prompt. Its default of 100,000 (`src/config.py:936`) assumes a
        # 128k-window model; vllm-gen serves 32,768. Honcho does not error on
        # the excess, it TRUNCATES the message list
        # (`truncate_messages_to_fit`, reached through
        # `honcho_llm_call(max_input_tokens=...)`, `src/llm/api.py:333-341`,
        # `src/dialectic/core.py:450`) — but only when the bound is set below
        # the window. Left at 100,000 the dialectic 400s on every question
        # once the representation grows: measured on smoke `hn_smkft2_p0`,
        # every dialectic call from persona 0 session 5 onward failed with
        # "requested 8192 output tokens and your prompt contains at least
        # 24577 input tokens". 20000 leaves the 8192-token dialectic answer
        # room inside the window, with margin: Honcho counts tokens with
        # tiktoken `cl100k_base` and the server tokenizes with Qwen's
        # tokenizer, so the two counts disagree.
        self.dialectic_max_input_tokens = int(
            _env("HONCHO_DIALECTIC_MAX_INPUT_TOKENS", "20000")
        )
        self.embedder_model = embedder_model or _env("HONCHO_EMBEDDER_MODEL", "bge-small-en-v1.5")
        self.embedder_base_url = embedder_base_url or _env("HONCHO_EMBEDDER_BASE_URL")
        self.embedder_api_key = embedder_api_key or _env("HONCHO_EMBEDDER_API_KEY", "local")
        self.embedder_dims = int(
            embedder_dims if embedder_dims is not None else int(_env("HONCHO_EMBEDDER_DIMS", "384"))
        )
        self.deriver_workers = int(
            deriver_workers if deriver_workers is not None else int(_env("HONCHO_DERIVER_WORKERS", "4"))
        )
        self.deriver_flush = _truthy(_env("HONCHO_DERIVER_FLUSH"), True) if deriver_flush is None else deriver_flush
        # Summary and peer-card generation each spend internal LLM calls on
        # every ingest. The 'conclusions' recall arm reads neither, so the
        # minimal preset turns both off and buys ingest throughput without
        # changing what that arm retrieves. The vendor default is on for both
        # (SummarySettings.ENABLED, PeerCardSettings.ENABLED).
        self.summary_enabled = (_truthy(_env("HONCHO_SUMMARY_ENABLED"), True)
                                if summary_enabled is None else summary_enabled)
        self.peer_card_enabled = (_truthy(_env("HONCHO_PEER_CARD_ENABLED"), True)
                                  if peer_card_enabled is None else peer_card_enabled)

        self._api: Optional[subprocess.Popen] = None
        self._deriver: Optional[subprocess.Popen] = None
        self._api_log = None
        self._deriver_log = None
        self.dim_fix_report: Dict[str, Any] = {}

    # -- child configuration -------------------------------------------------
    @property
    def python_exe(self) -> str:
        """The interpreter that owns Honcho's own dependency set.

        The server is a uv project. ``uv sync`` in ``HONCHO_SERVER_DIR``
        creates ``.venv`` there. The harness venv does NOT carry the server's
        dependencies (fastapi, alembic, uvloop), so this never falls back to
        ``sys.executable`` silently: it reports the missing venv instead.
        """
        override = _env("HONCHO_SERVER_PYTHON")
        if override:
            return override
        candidate = os.path.join(self.server_dir, ".venv", "bin", "python")
        if os.path.isfile(candidate):
            return candidate
        candidate_win = os.path.join(self.server_dir, ".venv", "Scripts", "python.exe")
        if os.path.isfile(candidate_win):
            return candidate_win
        raise RuntimeError(
            f"no server venv at {self.server_dir}/.venv — run `uv sync` there, "
            f"or set HONCHO_SERVER_PYTHON")

    def _model_config_env(self, prefix: str) -> Dict[str, str]:
        """Map one Honcho model-config block onto the internal LLM.

        TRANSPORT is always set alongside MODEL, and that is not cosmetic.
        ``_normalize_model_transport`` (src/config.py:262-275) splits a model
        id at the first ``/`` when the prefix is anthropic, openai, or gemini
        AND transport is unset. An OpenRouter id such as
        ``openai/gpt-oss-20b`` would silently become model ``gpt-oss-20b``,
        which OpenRouter does not serve. Setting TRANSPORT explicitly keeps
        the id intact.
        """
        env: Dict[str, str] = {
            f"{prefix}__TRANSPORT": "openai",
            f"{prefix}__MODEL": str(self.llm_model),
            f"{prefix}__MAX_OUTPUT_TOKENS": str(self.llm_max_output_tokens),
        }
        if self.llm_thinking_effort:
            env[f"{prefix}__THINKING_EFFORT"] = self.llm_thinking_effort
        if self.llm_base_url:
            env[f"{prefix}__OVERRIDES__BASE_URL"] = self.llm_base_url
        if self.llm_api_key:
            env[f"{prefix}__OVERRIDES__API_KEY"] = self.llm_api_key
        return env

    def child_env(self) -> Dict[str, str]:
        """Build the env both children run under.

        The harness answer and judge model lives in the PARENT's ``OPENAI_*``
        and is never copied here. Honcho reads its own namespaced variables,
        so the two roles cannot collide.
        """
        env = dict(os.environ)
        env["DB_CONNECTION_URI"] = str(self.pg_dsn)
        env["DB_SCHEMA"] = _env("HONCHO_DB_SCHEMA", "public")
        # Redis is optional and absent on this host. Auth off keeps the SDK's
        # placeholder bearer key acceptable.
        env["CACHE_ENABLED"] = "false"
        env["AUTH_USE_AUTH"] = "false"

        # Deriver: FLUSH_ENABLED bypasses the token-threshold batching
        # (DERIVER_REPRESENTATION_BATCH_MAX_TOKENS, default 1024). Without it
        # a work unit can wait for more messages that never come, so the
        # drain after a session would block on a batch that is under
        # threshold. WORKERS raises ingest throughput; it changes latency,
        # not what is derived.
        env["DERIVER_FLUSH_ENABLED"] = "true" if self.deriver_flush else "false"
        env["DERIVER_WORKERS"] = str(self.deriver_workers)
        env["DERIVER_POLLING_SLEEP_INTERVAL_SECONDS"] = _env(
            "HONCHO_DERIVER_POLL_S", "1.0")
        # The vendor jitters the first poll by up to 30s so co-started
        # instances do not poll in lockstep. One local deriver has no peers
        # to collide with, and that jitter is pure latency on every drain.
        env["DERIVER_POLLING_STARTUP_JITTER_SECONDS"] = _env(
            "HONCHO_DERIVER_STARTUP_JITTER_S", "0.0")
        env["DERIVER_POLLING_SLEEP_MAX_INTERVAL_SECONDS"] = _env(
            "HONCHO_DERIVER_POLL_MAX_S", "2.0")

        # Feature toggles for the two derived artifacts the 'conclusions'
        # recall arm never reads. Off means fewer internal LLM calls per
        # ingest; the hybrid and base arms need both ON, because they inject
        # the session summary and the peer card as sections.
        env["SUMMARY_ENABLED"] = "true" if self.summary_enabled else "false"
        env["PEER_CARD_ENABLED"] = "true" if self.peer_card_enabled else "false"

        # Every internal LLM role points at the same model. Dialectic scales
        # its reasoning level per query, so all five levels must resolve, not
        # only the default 'low'.
        for prefix in ("DERIVER_MODEL_CONFIG", "SUMMARY_MODEL_CONFIG",
                       "DREAM_DEDUCTION_MODEL_CONFIG", "DREAM_INDUCTION_MODEL_CONFIG"):
            env.update(self._model_config_env(prefix))
        for level in ("minimal", "low", "medium", "high", "max"):
            env.update(self._model_config_env(f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG"))

        # Deriver-only overlay, applied AFTER the loop so the other roles keep
        # the global budget and send no penalty. See the two constructor
        # comments for the measured runaway this bounds.
        env["DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS"] = str(self.deriver_max_output_tokens)
        if self.deriver_presence_penalty is not None:
            env["DERIVER_MODEL_CONFIG__PRESENCE_PENALTY"] = str(self.deriver_presence_penalty)
        if self.deriver_frequency_penalty is not None:
            env["DERIVER_MODEL_CONFIG__FREQUENCY_PENALTY"] = str(self.deriver_frequency_penalty)

        # Fits the dialectic prompt inside the served window. See the
        # constructor comment: without it every dialectic call 400s once the
        # representation passes about 24,577 tokens.
        env["DIALECTIC_MAX_INPUT_TOKENS"] = str(self.dialectic_max_input_tokens)

        # LLMSettings holds the transport-level credentials. Any module
        # override above wins, but these keep an unmapped role working.
        if self.llm_api_key:
            env["LLM_OPENAI_API_KEY"] = self.llm_api_key
        if self.llm_base_url:
            env["LLM_OPENAI_BASE_URL"] = self.llm_base_url

        # Embedder. DIMENSIONS_MODE=never stops the OpenAI client sending
        # `dimensions=`: bge-small has one output width, and vLLM's pooling
        # endpoint 400s on the parameter for a model with no matryoshka
        # support (the same failure the mem0 adapter shims around).
        env["EMBEDDING_MODEL_CONFIG__TRANSPORT"] = "openai"
        env["EMBEDDING_MODEL_CONFIG__MODEL"] = str(self.embedder_model)
        env["EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE"] = _env(
            "HONCHO_EMBEDDER_DIMENSIONS_MODE", "never")
        if self.embedder_base_url:
            env["EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL"] = self.embedder_base_url
        if self.embedder_api_key:
            env["EMBEDDING_MODEL_CONFIG__OVERRIDES__API_KEY"] = self.embedder_api_key
        env["EMBEDDING_VECTOR_DIMENSIONS"] = str(self.embedder_dims)

        # Clock-sync arms: preload libfaketime into the SERVER CHILDREN only,
        # so their perceived OS clock tracks the dataset's logical session
        # date while the harness process keeps real time for its own
        # deadlines. Identical contract to _supermemory_server.py. Inert
        # unless BENCH_CLOCKSYNC=1.
        if os.environ.get("BENCH_CLOCKSYNC") == "1" and os.environ.get("BENCH_CLOCKSYNC_FILE"):
            env["LD_PRELOAD"] = os.environ.get(
                "BENCH_LIBFAKETIME",
                "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1")
            env["FAKETIME_TIMESTAMP_FILE"] = os.environ["BENCH_CLOCKSYNC_FILE"]
            env["FAKETIME_NO_CACHE"] = "1"
            env["FAKETIME_DONT_FAKE_MONOTONIC"] = "1"
            env["NO_FAKE_STAT"] = "1"
        return env

    # -- lifecycle -----------------------------------------------------------
    def provision(self) -> None:
        """Create the run database, replay migrations, fix the vector dim."""
        if self.create_db:
            create_database(self.pg_dsn, self.db_name,
                            drop_existing=_truthy(_env("HONCHO_PG_DROP_DB"), True))
        env = self.child_env()
        script = os.path.join(self.server_dir, "scripts", "provision_db.py")
        print(f"[honcho] provisioning {self.pg_dsn}", flush=True)
        proc = subprocess.run([self.python_exe, script], cwd=self.server_dir,
                              env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"provision_db.py failed (rc={proc.returncode})\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        self.dim_fix_report = apply_embedding_dim_fix(
            dsn=str(self.pg_dsn), target_dim=self.embedder_dims,
            server_dir=self.server_dir, python_exe=self.python_exe,
            schema=env["DB_SCHEMA"], env=env,
        )

    def start(self, ready_timeout: float = 180.0) -> str:
        """Boot (or attach to) the API, start the deriver, return the base URL."""
        if self.attach_url:
            self._await_health(ready_timeout)
            print(f"[honcho] attached to {self.base_url}", flush=True)
            return self.base_url

        if not self.embedder_base_url:
            # An unreachable embedder does NOT stop Honcho. The API keeps
            # answering, the deriver keeps draining work units, and every
            # "save representation" call fails on the embed step, so the
            # workspace ends with ZERO conclusions and every recall section
            # comes back empty. A probe run hit exactly this: 518
            # representation tasks processed, 954 embed 401s, 0 documents,
            # and a hybrid payload that carried only the dialectic's "I have
            # no information" answer. Fail here instead.
            raise RuntimeError(
                "HONCHO_EMBEDDER_BASE_URL is not set. Honcho requires an "
                "OpenAI-compatible embeddings endpoint: point it at vllm-embed, "
                "or let the adapter start honcho/_local_embed_server.py "
                "(HONCHO_EMBED_SHIM=1).")
        os.makedirs(self.run_dir, exist_ok=True)
        self.provision()
        env = self.child_env()

        self._api_log = open(self.api_log_path, "w+", encoding="utf-8")
        api_cmd = [self.python_exe, "-m", "uvicorn", "src.main:app",
                   "--host", "127.0.0.1", "--port", str(self.port)]
        print(f"[honcho] starting API on port {self.port} "
              f"(model={self.llm_model}, embedder={self.embedder_model}/"
              f"{self.embedder_dims}d)", flush=True)
        self._api = subprocess.Popen(api_cmd, cwd=self.server_dir, env=env,
                                     stdout=self._api_log, stderr=subprocess.STDOUT)

        self._deriver_log = open(self.deriver_log_path, "w+", encoding="utf-8")
        print(f"[honcho] starting deriver (workers={self.deriver_workers}, "
              f"flush={self.deriver_flush}, summary={self.summary_enabled}, "
              f"peer_card={self.peer_card_enabled})", flush=True)
        self._deriver = subprocess.Popen(
            [self.python_exe, "-m", "src.deriver"], cwd=self.server_dir, env=env,
            stdout=self._deriver_log, stderr=subprocess.STDOUT)

        self._await_health(ready_timeout)
        # The deriver owns no port, so liveness is the only check available.
        # It must be running BEFORE the first ingest: a queue that nothing
        # drains would make every drain time out instead of failing here.
        if self._deriver.poll() is not None:
            raise RuntimeError(
                f"honcho deriver exited early (code={self._deriver.returncode}); "
                f"see {self.deriver_log_path}\n--- last log ---\n"
                f"{self._read_log(self.deriver_log_path)[-2500:]}")
        print(f"[honcho] ready at {self.base_url}", flush=True)
        return self.base_url

    def _read_log(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""

    def _await_health(self, ready_timeout: float) -> None:
        deadline = time.time() + ready_timeout
        last_error = ""
        while time.time() < deadline:
            for proc, name, log_path in self._children():
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"honcho {name} exited early (code={proc.returncode}); "
                        f"see {log_path}\n--- last log ---\n"
                        f"{self._read_log(log_path)[-2500:]}")
            try:
                with urlopen(f"{self.base_url}/health", timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        return
                    last_error = f"HTTP {resp.status}"
            except URLError as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
            time.sleep(1.0)
        tail = self._read_log(self.api_log_path)[-2500:]
        raise TimeoutError(
            f"honcho API not healthy within {ready_timeout}s at {self.base_url} "
            f"(last error: {last_error})\n--- api log ---\n{tail}")

    def _children(self) -> List[Tuple[subprocess.Popen, str, str]]:
        out: List[Tuple[subprocess.Popen, str, str]] = []
        if self._api is not None:
            out.append((self._api, "API", self.api_log_path))
        if self._deriver is not None:
            out.append((self._deriver, "deriver", self.deriver_log_path))
        return out

    def alive(self) -> bool:
        """True when this process spawned children and both still run."""
        if self.attach_url:
            return True
        return all(proc.poll() is None for proc, _, _ in self._children())

    def close(self, remove_run_dir: bool = False) -> None:
        for proc, name, _ in self._children():
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(f"[honcho] {name} did not stop in 15s; killing", flush=True)
                    proc.kill()
            except Exception:
                pass
        self._api = None
        self._deriver = None
        for handle in (self._api_log, self._deriver_log):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self._api_log = None
        self._deriver_log = None
        if remove_run_dir and os.path.isdir(self.run_dir):
            shutil.rmtree(self.run_dir, ignore_errors=True)


if __name__ == "__main__":
    # Self-test: boot, health check, tear down. This needs Postgres reachable
    # at HONCHO_PG_DSN and an internal LLM configured through HONCHO_LLM_*.
    os.environ.setdefault("HONCHO_PG_CREATE_DB", "1")
    os.environ.setdefault("RUN_TAG", "selftest")
    server = HonchoServer()
    try:
        url = server.start()
        print(f"[selftest] up at {url}; dim_fix={server.dim_fix_report}", file=sys.stderr)
    finally:
        server.close()
