#!/usr/bin/env python3
"""Write a best-effort run manifest for a provider benchmark run.

The manifest file lets a reader reproduce and audit a run. It records the
repo and submodule SHAs, dataset identity, a redacted env snapshot, and the
canonical answer/judge decoding config, so a reader can diff fairness parity
across providers.

Call this script from each entrypoint at generate-start and score-start:

    python benchmark/write_manifest.py --provider_dir /app/mnemosyne \
        --run_tag local --stage generate

The environment SNAPSHOT is best-effort. Each entrypoint tolerates a snapshot
failure, and this script catches its own errors, so a snapshot problem never
aborts a run. Missing or unavailable optional data appears as null.

The REQUIRED RUN CONTRACT is not best-effort (added 2026-07-26, from the
2026-07-24 upstream review's "Make run identity fail closed"). Provenance
splits into two parts:

  * ``run_contract`` holds the fields that DEFINE what the run measured: code
    SHA, dataset identity and hash, provider, preset, serving envelope
    summary, answer/judge model and decoding, temporal contract and provider
    temporal capability, retain cadence, retrieval surface, and prompt
    hashes. ``run_contract_hash`` is the sha256 of its canonical JSON. Under
    ``STRICT_RUN_CONTRACT=1`` (or ``BENCH_CLOCKSYNC=1``, the clock-normalized
    wave), a missing required field aborts the GENERATE stage with a nonzero
    exit instead of a warning.
  * ``env`` / ``repo`` / ``serving_envelope`` / ``token_usage`` hold the
    best-effort snapshot. A missing field only warns.

Scope note: this file makes the hash available and records it. Validating the
hash at shard-merge or score time is out of scope here.

The canonical_config block is not hardcoded. An earlier version hardcoded it
and baked in the contract-v1 answer temperature 0.2 and max_tokens 3072, and
the judge's thinking-off/temp 0.2/4096 settings, long after the harness moved
to contract v2. Every manifest since that switch silently misreported the run
config. The block now reads os.environ live for the exact decoding vars
answer_env.sh exports (bench_answer_env / bench_judge_env), at whatever point
in the run this script runs. See _canonical_config()'s docstring for why that
read is trustworthy without this file knowing answer_env.sh's defaults.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

# This file lives at <root>/benchmark/, so the repo root is one level up.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Env vars captured in the snapshot, by exact name or prefix.
#
# This list must include every arm-selecting var that has no provider prefix,
# beyond the run-identity vars THINKING/TOP_K/.../STAGE. Otherwise two
# manifests from DIFFERENT arms can be byte-identical on the one axis that
# matters. We verified this: manifest_armB_qwen_generate.json and
# manifest_append_qwen_generate.json were indistinguishable on the
# granularity/wait flags before this list grew. We built the list by reading
# every entrypoint under benchmark/docker/ and benchmark/docker/run_shards.sh
# for env vars that select an arm. Provider-prefixed vars (MNEMOSYNE_/
# HINDSIGHT_/RETAINDB_/...) are already covered by _ENV_PREFIXES below, so we
# do not repeat them here, even when they also select an arm (for example
# HINDSIGHT_PG_MODE, HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION,
# MNEMOSYNE_ENHANCED_RECALL).
_ENV_EXACT = {
    "THINKING", "TOP_K", "NUM_PERSONAS", "START_IDX", "END_IDX",
    "RUN_TAG", "STAGE",
    # Hindsight and RetainDB ingest and consolidation arm selectors.
    "RETAIN_GRANULARITY", "PREFER_OBSERVATIONS", "WAIT_CONSOLIDATION",
    # Unprefixed, so HINDSIGHT_ in _ENV_PREFIXES does not cover it. Without
    # this, an arm-C manifest cannot say which recall surface produced the
    # Results file: observation-only (plugin default) or unfiltered (all
    # fact types).
    "RECALL_TYPES",
    # Unprefixed, like RECALL_TYPES. Selects the featured plugin-native
    # recall surface width (the answer LLM and Retrieved_Memories get every
    # recalled item, with no top_k slice) versus the minimal top_k slice. A
    # manifest must record this to distinguish a featured Arm-C file from a
    # minimal one.
    "PLUGIN_NATIVE_RECALL",
    "CONSOLIDATION_WAIT_TIMEOUT_S",
    # Mnemosyne arm selectors, plugin-fidelity and lifecycle family.
    "PLUGIN_CONFIG", "USE_DATASET_TIME", "EXTRACT", "LIFECYCLE", "CANONICAL",
    "ORACLE",
    # PLUGIN_AUTO_SLEEP was missing until 2026-07-21. This was a pre-existing
    # bug: every auto-sleep manifest written before then cannot say whether
    # the arm was even enabled, the one fact that distinguishes it from the
    # plain plugin arm. PLUGIN_PREFETCH_OVERLAY selects the plugin's real
    # prefetch() read path (filters, canonical merge, semantic dedup)
    # instead of raw recall, so it also changes what a run measured. Both
    # are unprefixed, so no _ENV_PREFIXES entry covers them.
    # PLUGIN_SESSION_SLEEP selects the featured arm's manually cadenced
    # consolidation (one sleep(force=True) per session). It decides whether
    # the prior sessions' rows survive the shipped working-memory TTL, so a
    # manifest without it cannot say what the run measured.
    "PLUGIN_AUTO_SLEEP", "PLUGIN_SESSION_SLEEP", "PLUGIN_PREFETCH_OVERLAY",
    # Smoke and shard-shape caps: these change what a run actually covers.
    "MAX_SESSIONS", "MAX_QUESTIONS_PER_SESSION", "NUM_SHARDS", "SHARD_IDX",
    # Retrieval and scoring knobs that stay the same across providers but
    # still shape the run. Top-K is also captured above via TOP_K; kept here
    # too for clarity.
    "JUDGE_THINKING", "JUDGE_REASONING_EFFORT", "SCORE_WORKERS",
    # Shared retry policy (answer_env.sh). We widened it so a shard rides
    # out a vllm-gen restart instead of dying. These vars are unprefixed, so
    # without them the snapshot would silently omit the retry envelope a run
    # actually used.
    "RETRY_TIMES", "WAIT_TIME_LOWER", "WAIT_TIME_UPPER",
    # STRICT_QUALITY_RUN was missing until 2026-07-21, the same defect class
    # as the RECALL_TYPES gap above. It decides whether a Hindsight shard
    # aborts on silent-degradation paths (drain timeout, append downgrade)
    # or absorbs them, so without it a manifest cannot say whether the
    # strict guard protected the run. SKIP_ROW_GATE and
    # EMPTY_ANSWER_MAX_FRAC also shape what the score stage judged: an audit
    # needs to see a bypassed or loosened row gate. All three are
    # unprefixed.
    "STRICT_QUALITY_RUN", "SKIP_ROW_GATE", "EMPTY_ANSWER_MAX_FRAC",
    # Run-identity vars added with the required run contract (2026-07-26).
    # PRESET names the launch bundle (benchmark/docker/presets.sh) and is
    # part of the contract hash. STRICT_RUN_CONTRACT says whether the
    # fail-closed gate was armed, which an audit must be able to see. Both
    # are unprefixed.
    "PRESET", "STRICT_RUN_CONTRACT",
    # RetainDB-server cadence and lifecycle selectors use the RETAINDB_
    # prefix (covered below), but the scheduler and lifecycle-threshold
    # knobs the server itself reads use bare names, so nothing else
    # captures them.
    "DISABLE_SCHEDULER", "SESSION_INACTIVITY_THRESHOLD_MS",
    "SESSION_LIFECYCLE_INTERVAL_MS", "EXTRACTOR_MODEL",
}
_ENV_PREFIXES = (
    "OPENAI_", "MEMCONFLICT_", "SCORE_",
    "MNEMOSYNE_", "HINDSIGHT_", "RETAINDB_", "SUPERMEMORY_", "BENCH_",
    # MEM0_ was missing until 2026-07-26 (2026-07-24 featured-run audit, P0
    # #2): the ftsmoke_m0 manifest recorded almost only RETAIN_GRANULARITY,
    # so nobody could audit mem0's effective internal LLM, embedder,
    # collection, or vector mode from the manifest. EMBED_PROXY_ is the
    # RetainDB server's embedding translation layer, the same class of
    # unauditable gap.
    "MEM0_", "EMBED_PROXY_",
    # Honcho adapter + server-manager knobs (recall arm, dialectic shape,
    # internal-LLM/embedder wiring, spawn-vs-shared server selection).
    "HONCHO_",
    # OpenViking adapter + server-manager knobs (recall arm, recall shaping,
    # ingest cadence, internal-LLM/embedder wiring, workspace and user-prefix
    # isolation).
    "OPENVIKING_",
)
# A var whose name contains one of these strings (case-insensitive) has its
# value redacted in the snapshot.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
# Exception: token-count config such as OPENAI_MAX_TOKENS, SCORE_MAX_TOKENS,
# and HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS. "TOKENS" trips the "TOKEN"
# marker, but these are numeric decoding budgets, never secrets, and must
# stay visible for fairness diffing. This allowlist wins over the markers.
_NON_SECRET_SUBSTRINGS = ("MAX_TOKENS", "COMPLETION_TOKENS")


def _redact(name):
    upper = name.upper()
    if any(safe in upper for safe in _NON_SECRET_SUBSTRINGS):
        return False
    return any(marker in upper for marker in _SECRET_MARKERS)


def _env_snapshot():
    snap = {}
    for name, value in os.environ.items():
        if name in _ENV_EXACT or name.startswith(_ENV_PREFIXES):
            snap[name] = "<redacted>" if _redact(name) else value
    return dict(sorted(snap.items()))


def _git(args):
    """Run a git command at the repo root. Return stripped stdout, or None on failure."""
    try:
        out = subprocess.run(
            ["git", "-C", _ROOT] + args,
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


# Env vars that may carry the launching repo commit, checked in this order.
# GIT_SHA is the var benchmark/docker/run_shards.sh exports and every
# provider service declares in docker-compose.yml. The rest are conventional
# CI names, accepted so a future launcher need not learn this file's own
# spelling.
_GIT_SHA_ENV_VARS = ("GIT_SHA", "BENCH_GIT_SHA", "GIT_COMMIT", "SOURCE_COMMIT")


def _git_sha_stamp_path():
    """Path of the optional host-written SHA stamp file.

    The provider containers bind-mount ``benchmark/`` but not ``.git`` (the
    image bakes in source, not history), so a container cannot derive the
    commit on its own. ``GIT_SHA`` is the primary channel. This file is the
    fallback for launch paths other than run_shards.sh: a manual
    ``docker compose run ... mnemosyne`` inherits nothing, which is why the
    mnemosyne v4-minimal manifest recorded head_sha:null while every sharded
    provider recorded a real SHA. Write this file on the host with
    ``benchmark/docker/stamp_git_sha.sh``, or any command piping
    ``git rev-parse HEAD`` into it.
    """
    override = os.environ.get("BENCH_GIT_SHA_FILE", "").strip()
    if override:
        return override
    return os.path.join(_ROOT, "benchmark", ".git_sha")


def _read_git_sha_stamp():
    path = _git_sha_stamp_path()
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                value = fh.read().strip().split()[0]
            if re.fullmatch(r"[0-9a-fA-F]{7,40}", value or ""):
                return value, path
    except Exception:
        pass
    return None, path


def _repo_info():
    """Resolve the code SHA, and say where it came from.

    Check order: explicit env passthrough, then live git (whenever a .git
    directory or file is present, so linked worktrees also work), then the
    host-written stamp file. Every manifest records ``head_sha_source``, so
    an audit can tell a real checkout apart from a passthrough, instead of
    trusting a bare hex string.
    """
    env_sha = None
    env_var = None
    for var in _GIT_SHA_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            env_sha, env_var = value, var
            break
    # .git is a directory in a normal clone, but a file in a linked worktree
    # or submodule. An older `isdir` check silently skipped both of those.
    dot_git = os.path.join(_ROOT, ".git")
    has_git = os.path.exists(dot_git)
    head = _git(["rev-parse", "HEAD"]) if has_git else None
    submodules = _git(["submodule", "status"]) if has_git else None
    dirty = None
    if has_git:
        status = _git(["status", "--porcelain"])
        if status is not None:
            dirty = bool(status.strip())
    stamp_sha, stamp_path = _read_git_sha_stamp()

    if env_sha:
        head_sha, source = env_sha, "env:%s" % env_var
    elif head:
        head_sha, source = head, "git:rev-parse"
    elif stamp_sha:
        head_sha, source = stamp_sha, "stamp:%s" % stamp_path
    else:
        head_sha, source = None, "unavailable"

    info = {
        "head_sha": head_sha,
        "head_sha_source": source,
        "git_sha_env": env_sha,
        "git_rev_parse": head,
        "git_sha_stamp_file": stamp_path,
        "git_sha_stamp": stamp_sha,
        "worktree_dirty": dirty,
        "submodule_status": submodules,
    }
    if head is None and submodules is None:
        info["note"] = ("no .git in this container: head_sha comes from %s "
                        "(export GIT_SHA before `docker compose run`, or write "
                        "%s on the host, to keep it populated)"
                        % (source, stamp_path))
    return info


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _dataset_info():
    """Dataset identity: path, line count, and a content hash.

    The hash makes "same dataset" checkable instead of assumed, per the
    review's required-contract list. The file is about 40 MB, so hashing
    costs well under a second and runs twice per run. ``BENCH_DATASET_SHA256``
    skips the hashing when a caller already computed it.
    """
    path = os.environ.get(
        "MEMCONFLICT_DATASET",
        os.path.join(_ROOT, "external", "MemConflict", "Data", "Step4_4.jsonl"),
    )
    info = {"path": path, "line_count": None, "sha256": None, "bytes": None}
    override = os.environ.get("BENCH_DATASET_SHA256", "").strip()
    try:
        if os.path.isfile(path):
            info["bytes"] = os.path.getsize(path)
            with open(path, "r", encoding="utf-8") as fh:
                info["line_count"] = sum(1 for _ in fh)
            info["sha256"] = override or _sha256_file(path)
    except Exception as exc:
        info["error"] = "%s: %s" % (type(exc).__name__, exc)
    return info


# --- temporal contract ---------------------------------------------------------
# The 2026-07-24 upstream review asks every manifest to name the temporal
# policy as a contract, plus each provider's declared temporal capability
# (the mechanism that actually carries logical time to that provider). The
# capability map below is the review's own classification table. It is the
# FALLBACK, hardcoded here because for every provider except Hindsight the
# mechanism is a property of the adapter-provider pair, not of a run's env:
#   hindsight       native — retain timestamps plus an explicit recall
#                   query_timestamp; no OS clock preload anywhere in its
#                   path. True for the minimal arm only, see below.
#   mnemosyne       controlled_process_clock — libfaketime on the shard
#                   python process.
#   mem0            controlled_process_clock — libfaketime plus a per-add
#                   patch of its import-frozen prompt date.
#   supermemory     controlled_process_clock — libfaketime on the spawned
#                   server.
#   retaindb_server controlled_process_clock+postgres — node and its own
#                   postmaster share one faked clock domain.
#   honcho          controlled_process_clock+postgres — the spawn arm's
#                   adapter-managed API+deriver children and, under
#                   BENCH_CLOCKSYNC=1, its own in-container Postgres, all
#                   share one libfaketime clock domain (honcho/_honcho_server.py).
#   openviking      controlled_process_clock — libfaketime on the spawned
#                   server child only (openviking/_openviking_server.py).
#                   Storage is that server's own local workspace, so no
#                   second clock domain exists to fake.
#
# Hindsight runs two arms on two mechanisms, so the map alone misreports one
# of them:
#   * minimal (session granularity) keeps `native`. `_retain_one` sends no
#     document_id and no update_mode, so it never enters the append merge
#     that drops event_date, and nothing on its path preloads libfaketime.
#   * featured (exchange_append) does enter that merge, so it runs embedded
#     pg0 under libfaketime and declares
#     controlled_process_clock+postgres through BENCH_TEMPORAL_CAPABILITY.
# entrypoint.hindsight.sh sets that override in its pg0 branch — never a
# preset and never an operator — because only the branch that starts the
# faked daemon knows which mechanism engaged, and the recorded value must
# describe what actually ran. A wrong value hashes into run_contract_hash
# and makes two different measurements look like one run.
TEMPORAL_CONTRACT_NAME = "logical_session_noon_v1"
TEMPORAL_CAPABILITY = {
    "hindsight": "native",
    "mnemosyne": "controlled_process_clock",
    "mem0": "controlled_process_clock",
    "supermemory": "controlled_process_clock",
    "retaindb_server": "controlled_process_clock+postgres",
    "honcho": "controlled_process_clock+postgres",
    "openviking": "controlled_process_clock",
    # Ruled out 2026-07-22 (npm local edition). Kept so a legacy re-score of
    # its artifacts still produces a complete manifest instead of "unknown".
    "retaindb": "controlled_process_clock+postgres",
}
# The override is validated against this set instead of being passed through.
# An unvalidated typo would hash into run_contract_hash and read as a real
# mechanism forever.
_TEMPORAL_CAPABILITY_ALLOWED = frozenset({
    "native",
    "controlled_process_clock",
    "controlled_process_clock+postgres",
})


def _temporal_contract():
    return (TEMPORAL_CONTRACT_NAME
            if os.environ.get("BENCH_CLOCKSYNC", "0") == "1" else "none")


def _temporal_capability(provider):
    """The mechanism that carried logical time, override first, then the map.

    An unrecognised override returns None, which _missing_required counts as
    a missing required field (temporal_capability is in
    _REQUIRED_CONTRACT_FIELDS). Under the strict gate the generate stage then
    aborts, instead of banking artifacts stamped with a typo.
    """
    override = os.environ.get("BENCH_TEMPORAL_CAPABILITY", "").strip()
    if override:
        if override in _TEMPORAL_CAPABILITY_ALLOWED:
            return override
        print("[write_manifest] WARN: BENCH_TEMPORAL_CAPABILITY=%r is not one "
              "of %s; recording temporal_capability as missing"
              % (override, ", ".join(sorted(_TEMPORAL_CAPABILITY_ALLOWED))),
              file=sys.stderr)
        return None
    return TEMPORAL_CAPABILITY.get(provider, "unknown")


# The full sampling set that controls answer and judge decoding, per
# answer_env.sh's bench_answer_env / bench_judge_env and eval_common's
# llm_request wrapper. This is one flat list, not split into answer vars and
# judge vars, because both stages read every one of these vars. Only the
# value each stage exports differs, not which vars exist.
_DECODING_ENV_KEYS = (
    "OPENAI_TEMPERATURE", "OPENAI_MAX_TOKENS",
    "MEMCONFLICT_ENABLE_THINKING", "MEMCONFLICT_JSON_MODE",
    "MEMCONFLICT_JSON_THINKING", "MEMCONFLICT_REASONING_EFFORT",
    "MEMCONFLICT_TOP_P", "MEMCONFLICT_TOP_K", "MEMCONFLICT_MIN_P",
    "MEMCONFLICT_PRESENCE_PENALTY",
)


def _canonical_config(stage):
    """Live snapshot of the decoding config in effect for this stage.

    This function does not hardcode or rebuild the config from defaults. It
    reads os.environ directly for _DECODING_ENV_KEYS. That read is
    trustworthy because of when write_manifest.py runs relative to
    answer_env.sh:
      * generate stage: each entrypoint calls `bench_answer_env` immediately
        before this script (see entrypoint.*.sh `do_generate`), and the
        shell has not exported any judge var yet, so os.environ already
        holds exactly this run's answer decoding, nothing more.
      * score stage: `run_score()` in answer_env.sh calls `bench_judge_env`
        immediately before this script and always re-exports every judge
        var, so a STAGE=score-only run never keeps generate-stage leftovers.
        os.environ then holds exactly the judge decoding.
    A var that is unset, for example MEMCONFLICT_JSON_MODE during the answer
    stage (bench_answer_env explicitly unsets it), comes back as None here.
    We do not default it, so a genuinely missing export stays visible in the
    manifest diff instead of being hidden behind a guessed value.
    """
    return {
        "stage": stage,
        "decoding": {key: os.environ.get(key) for key in _DECODING_ENV_KEYS},
    }


def _serving_envelope(provider_dir, run_tag):
    """Best-effort ingest of a serving-envelope sidecar into the manifest.

    From inside a provider container, the manifest's env snapshot cannot see
    the vLLM serving side: image digest, checkpoint, effective server flags.
    The 2026-07-21 review confirmed that OPENAI_MODEL records only the
    served alias, so recovering the checkpoint that produced a result
    otherwise needed git archaeology. A host-side sidecar written at wave
    start closes that gap (see mnemosyne/Scores/v3/serving_envelope_v2_
    baseline.json for the reference shape, captured live via `docker inspect
    vllm-gen`). This function folds that sidecar into every manifest for the
    same RUN_TAG, so the result artifact carries its own provenance.

    Lookup order: the explicit SERVING_ENVELOPE_FILE env var, then
    <provider_dir>/Scores/serving_envelope_<run_tag>.json. Returns the
    parsed JSON, or {"note": ...} when the file is absent or unreadable.
    This function never raises, because manifest writing is best-effort by
    contract.
    """
    candidates = []
    env_path = os.environ.get("SERVING_ENVELOPE_FILE", "").strip()
    if env_path:
        candidates.append(env_path)
    if run_tag:
        candidates.append(os.path.join(provider_dir, "Scores",
                                       "serving_envelope_%s.json" % run_tag))
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return {"source_file": path, "data": data}
        except Exception as exc:
            return {"note": "sidecar unreadable: %s (%s)" % (path, exc)}
    return {"note": "no serving-envelope sidecar found (checked: %s)"
                    % (", ".join(candidates) or "nothing")}


# --- prompt hashes -------------------------------------------------------------
# The two system prompts decide what we ask the answerer and the judge to do.
# We read them as text and hash them, rather than importing them: importing
# eval_common pulls in the whole harness, and eval_scoring lives under
# external/. An unrecorded prompt edit is exactly the kind of silent contract
# change the review's required-contract list is meant to catch. Extraction
# failure records "n/a" explicitly, never a guessed or omitted value.
_PROMPT_SOURCES = (
    ("answer_system_prompt",
     os.path.join(_ROOT, "benchmark", "eval_common.py"),
     "ANSWER_SYSTEM_PROMPT"),
    ("judge_system_prompt",
     None,  # Resolve this path from MEMCONFLICT_EVAL_DIR at call time.
     "LLM_JUDGE_SYSTEM_PROMPT"),
)


def _eval_dir():
    return os.environ.get(
        "MEMCONFLICT_EVAL_DIR",
        os.path.join(_ROOT, "external", "MemConflict", "Evaluation"),
    )


def _prompt_hash(path, const_name):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return "n/a"
    match = re.search(re.escape(const_name) + r'\s*=\s*"""(.*?)"""', text, re.S)
    if not match:
        return "n/a"
    return "sha256:" + hashlib.sha256(
        match.group(1).encode("utf-8")).hexdigest()


def _prompt_hashes():
    out = {}
    for name, path, const_name in _PROMPT_SOURCES:
        if path is None:
            path = os.path.join(_eval_dir(), "eval_scoring.py")
        out[name] = _prompt_hash(path, const_name)
    return out


def _token_usage(provider_dir, run_tag):
    """Fold the token-accounting sidecar into the manifest when one exists.

    benchmark/token_usage.py writes this sidecar two ways: per shard
    (``token_usage_<RUN_TAG>``, scope=shard) from the entrypoint, and per
    wave (``token_usage_<tag>``, scope=run) from run_shards.sh, with the
    ``_s<k>`` range-shard suffix or ``_p<i>`` per-persona suffix stripped.
    This function checks both, and prefers the shard file because this
    container produced it.
    """
    candidates = []
    override = os.environ.get("TOKEN_USAGE_FILE", "").strip()
    if override:
        candidates.append(override)
    results_dir = os.path.join(provider_dir, "Results")
    if run_tag:
        candidates.append(os.path.join(results_dir,
                                       "token_usage_%s.json" % run_tag))
        wave_tag = re.sub(r"_[sp][0-9]+$", "", run_tag)
        if wave_tag != run_tag:
            candidates.append(os.path.join(results_dir,
                                           "token_usage_%s.json" % wave_tag))
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    return {"source_file": path, "data": json.load(fh)}
        except Exception as exc:
            return {"note": "sidecar unreadable: %s (%s)" % (path, exc)}
    return {"note": "no token-usage sidecar found (checked: %s)"
                    % (", ".join(candidates) or "nothing")}


# --- required run contract -----------------------------------------------------
# Dotted leaf paths that must be non-null and non-empty. Everything else in
# the contract dict is recorded but may legitimately be absent: `preset` is
# empty for a hand-launched run, and the serving image digest is host-only
# information a manual `docker compose run` cannot supply.
_REQUIRED_CONTRACT_FIELDS = (
    "code_sha",
    "provider",
    "dataset.path", "dataset.sha256", "dataset.line_count",
    "temporal_contract", "temporal_capability",
    "serving.model_alias", "serving.base_url",
    "answer_judge.model",
    "answer_judge.decoding.OPENAI_TEMPERATURE",
    "answer_judge.decoding.OPENAI_MAX_TOKENS",
    "retrieval.top_k",
    "prompt_hashes.answer_system_prompt",
    "prompt_hashes.judge_system_prompt",
)


def _serving_summary(serving_envelope):
    """Served alias, image digest, and engine version, from the sidecar if present.

    Falls back to the env-configured alias and base URL, so the contract
    still pins which endpoint and alias a run used, even when the sidecar
    capture failed. That failure is itself loud: the entrypoints run the
    capture with --strict under the clock contract.
    """
    data = (serving_envelope or {}).get("data") or {}
    summary = data.get("summary") or {}
    image = data.get("image") or {}
    return {
        "model_alias": os.environ.get("OPENAI_MODEL") or None,
        "base_url": os.environ.get("OPENAI_BASE_URL") or None,
        "served_ids": summary.get("gen_served_ids"),
        "checkpoint_roots": summary.get("gen_checkpoint_roots"),
        "engine_version": summary.get("gen_engine_version"),
        "image": image.get("vllm_gen_image")
        or os.environ.get("BENCH_SERVING_IMAGE") or None,
        "image_digest": image.get("vllm_gen_image_digest")
        or os.environ.get("BENCH_SERVING_IMAGE_DIGEST") or None,
        "embed_model_alias": (data.get("configured") or {}).get("embed_model_alias"),
    }


def _retain_contract():
    """Ingest cadence and lifecycle selectors, in every provider's own spelling."""
    keys = (
        "RETAIN_GRANULARITY",                 # hindsight, mem0
        "SUPERMEMORY_RETAIN_GRANULARITY",
        "RETAINDB_RETAIN_GRANULARITY",
        # OpenViking accepts both spellings; entrypoint.openviking.sh resolves
        # them to one value, so the two keys can never disagree in a manifest.
        # SEND_CREATED_AT selects whether messages carry a dataset timestamp,
        # which changes what a run ingested, so it belongs with the cadence.
        "OPENVIKING_RETAIN_GRANULARITY", "OPENVIKING_SEND_CREATED_AT",
        "MEM0_ADD_BATCH_SIZE",
        "PLUGIN_CONFIG", "PLUGIN_AUTO_SLEEP", "PLUGIN_SESSION_SLEEP",
        "PLUGIN_PREFETCH_OVERLAY",
        "HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "WAIT_CONSOLIDATION",
        "PREFER_OBSERVATIONS", "CONSOLIDATION_WAIT_TIMEOUT_S",
        "DISABLE_SCHEDULER", "RETAINDB_SERVER_WAIT_LIFECYCLE",
        "RETAINDB_SERVER_PROMOTION_MODE", "RETAINDB_SERVER_PROFILE",
        "SUPERMEMORY_INGEST_ENDPOINT",
        "EXTRACT", "LIFECYCLE", "CANONICAL", "ORACLE", "USE_DATASET_TIME",
    )
    return {key: os.environ.get(key) for key in keys}


def _retrieval_contract():
    """The retrieval surface: how many items, of what kind, from where."""
    keys = (
        "RECALL_TYPES", "PLUGIN_NATIVE_RECALL",
        "SUPERMEMORY_SEARCH_MODE", "SUPERMEMORY_SEARCH_THRESHOLD",
        "SUPERMEMORY_RECALL_ENDPOINT", "SUPERMEMORY_DOCUMENTS_ARM",
        "SUPERMEMORY_RERANK", "SUPERMEMORY_REWRITE_QUERY",
        "MNEMOSYNE_FACT_RECALL_ENABLED", "MNEMOSYNE_ENHANCED_RECALL",
        "RETAINDB_SERVER_PLUGIN_OVERLAY",
        # OpenViking's recall-surface selector, the analog of
        # SUPERMEMORY_SEARCH_MODE above: prefetch (the plugin's block plus
        # search entries, no top-K slice) | search | find decide which items
        # reach the answer model.
        "OPENVIKING_RECALL_MODE",
    )
    out = {key: os.environ.get(key) for key in keys}
    out["top_k"] = os.environ.get("TOP_K")
    return out


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _missing_required(contract):
    missing = []
    for dotted in _REQUIRED_CONTRACT_FIELDS:
        node = contract
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if node is None or (isinstance(node, str) and not node.strip()):
            missing.append(dotted)
    return missing


def strict_enabled(stage):
    """Is the fail-closed gate armed for this stage?

    STRICT_RUN_CONTRACT=1 arms it explicitly. BENCH_CLOCKSYNC=1 also arms
    it, because the clock-normalized wave is the wave whose artifacts must
    be reproducible. This applies to the generate stage only: an abort
    during the score stage, which runs from inside answer_env.sh's
    run_score, would cost the run its judge manifest and protect nothing.
    A wrong contract produces unusable data at generation time.
    """
    if stage != "generate":
        return False
    return (os.environ.get("STRICT_RUN_CONTRACT", "0") == "1"
            or os.environ.get("BENCH_CLOCKSYNC", "0") == "1")


def build_run_contract(provider, stage, repo, dataset, serving_envelope):
    """The required half of provenance: the fields that define the measurement.

    This omits RUN_TAG, START_IDX/END_IDX, NUM_SHARDS, and `stage` on
    purpose. Every shard of one wave must produce the same hash, so shard
    geometry cannot appear in it. Answer decoding and judge decoding do
    differ by stage, which is the whole point of answer_env.sh, so a
    generate-stage hash and a score-stage hash for one run are legitimately
    different values. Compare hashes only within a stage.
    """
    return {
        "code_sha": repo.get("head_sha"),
        "provider": provider,
        "preset": os.environ.get("PRESET", ""),
        "dataset": {
            "path": dataset.get("path"),
            "sha256": dataset.get("sha256"),
            "line_count": dataset.get("line_count"),
        },
        "temporal_contract": _temporal_contract(),
        "temporal_capability": _temporal_capability(provider),
        "serving": _serving_summary(serving_envelope),
        "answer_judge": {
            "model": os.environ.get("OPENAI_MODEL") or None,
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "decoding": {key: os.environ.get(key) for key in _DECODING_ENV_KEYS},
            "thinking": os.environ.get("THINKING"),
            "judge_thinking": os.environ.get("JUDGE_THINKING"),
            "judge_reasoning_effort": os.environ.get("JUDGE_REASONING_EFFORT"),
            "retry": {key: os.environ.get(key) for key in
                      ("RETRY_TIMES", "WAIT_TIME_LOWER", "WAIT_TIME_UPPER")},
        },
        "retain": _retain_contract(),
        "retrieval": _retrieval_contract(),
        "prompt_hashes": _prompt_hashes(),
    }


def build_manifest(provider_dir, run_tag, stage):
    provider = os.path.basename(os.path.normpath(provider_dir))
    repo = _repo_info()
    dataset = _dataset_info()
    serving_envelope = _serving_envelope(provider_dir, run_tag)
    contract = build_run_contract(provider, stage, repo, dataset, serving_envelope)
    contract_json = _canonical_json(contract)
    contract_hash = "sha256:" + hashlib.sha256(
        contract_json.encode("utf-8")).hexdigest()
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": stage,
        "run_tag": run_tag,
        "provider": provider,
        "temporal_contract": _temporal_contract(),
        "temporal_capability": _temporal_capability(provider),
        "preset": os.environ.get("PRESET", ""),
        "run_contract": contract,
        "run_contract_hash": contract_hash,
        "run_contract_missing_required": _missing_required(contract),
        "run_contract_strict": strict_enabled(stage),
        "repo": repo,
        "dataset": dataset,
        "canonical_config": _canonical_config(stage),
        "serving_envelope": serving_envelope,
        "token_usage": _token_usage(provider_dir, run_tag),
        "env": _env_snapshot(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a best-effort run manifest.")
    ap.add_argument("--provider_dir", required=True,
                    help="Provider dir, e.g. /app/mnemosyne (its ./Scores gets the manifest).")
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--stage", required=True)
    args = ap.parse_args(argv)

    manifest = build_manifest(args.provider_dir, args.run_tag, args.stage)
    scores_dir = os.path.join(args.provider_dir, "Scores")
    # The stage name goes in the filename because generate and score can run
    # in the same container under STAGE=all, and each stage's env snapshot
    # differs (answer decoding versus judge decoding). One file per stage
    # keeps both auditable, instead of the score write clobbering the
    # generate one.
    out_path = os.path.join(scores_dir, "manifest_%s_%s.json" % (args.run_tag, args.stage))
    os.makedirs(scores_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("[write_manifest] wrote %s (stage=%s)" % (out_path, args.stage))
    # Echo the contract identity to stdout so every shard log carries it. An
    # operator often has only the shard log open while a wave is in flight,
    # so checking which contract a shard is on must not require opening the
    # manifest.
    print("[write_manifest] run_contract_hash=%s temporal_contract=%s "
          "temporal_capability=%s preset=%s code_sha=%s (%s)"
          % (manifest["run_contract_hash"], manifest["temporal_contract"],
             manifest["temporal_capability"], manifest["preset"] or "(none)",
             (manifest["repo"]["head_sha"] or "UNKNOWN")[:12],
             manifest["repo"]["head_sha_source"]))
    missing = manifest["run_contract_missing_required"]
    if missing:
        # Fail closed under the strict gate (STRICT_RUN_CONTRACT=1 or
        # BENCH_CLOCKSYNC=1, generate stage). Write the manifest first, so
        # the artifact itself shows the failure, then abort the stage.
        message = ("required run-contract fields missing: %s" % ", ".join(missing))
        if manifest["run_contract_strict"]:
            print("[write_manifest] FATAL: %s" % message, file=sys.stderr)
            print("[write_manifest]        STRICT_RUN_CONTRACT/BENCH_CLOCKSYNC is "
                  "armed, so this run must not generate artifacts it cannot "
                  "identify. Fix the missing fields (code_sha: export GIT_SHA or "
                  "write benchmark/.git_sha; serving.*: let the serving-envelope "
                  "capture succeed) and relaunch.", file=sys.stderr)
            return 3
        print("[write_manifest] WARN: %s (best-effort mode)" % message,
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except SystemExit:
        raise
    except Exception as exc:
        # Best-effort for everything except the strict contract gate. A
        # crash here must not abort an ordinary run, but under the strict
        # gate, an unwritten manifest is itself a missing run contract.
        strict = (os.environ.get("STRICT_RUN_CONTRACT", "0") == "1"
                  or os.environ.get("BENCH_CLOCKSYNC", "0") == "1")
        generate = "generate" in (sys.argv or [])
        print("[write_manifest] %s: %s"
              % ("FATAL" if (strict and generate) else "WARN", exc),
              file=sys.stderr)
        sys.exit(3 if (strict and generate) else 0)
