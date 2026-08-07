#!/usr/bin/env bash
# Host lifecycle for the RetainDB server edition (@retaindb/server).
#
# Builds the server from a copy of the external/RetainDB submodule. Never
# builds inside external/, because that tree is read-only pinned code. Runs
# prisma migrations, then execs the node server with the benchmark's
# best-effort environment. This script is idempotent: it reuses an existing
# build. Set RETAINDB_SERVER_BUILD_DIR to point at a build elsewhere. This is
# the host analog of entrypoint.retaindb-server.sh's server bring-up. The
# Python adapter only attaches, over HTTP.
#
# Prereqs: node 20 or later, pnpm (corepack enable), and a reachable Postgres
# with the pgvector extension available (DATABASE_URL). LLM extraction needs
# a non-empty OPENAI_API_KEY, or it silently degrades to regex and
# pattern-only matching.
#
# Usage:
#   DATABASE_URL=postgresql://u:p@localhost:5432/retaindb_server \
#   OPENAI_API_KEY=$OPENROUTER_API_KEY OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
#   EXTRACTOR_MODEL=openai/gpt-oss-120b \
#     retaindb_server/serve_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/retaindb_server"
SUBMODULE="$ROOT/external/RetainDB"

# --- Build dir (gitignored copy of the submodule, reused if present) --------
BUILD_DIR="${RETAINDB_SERVER_BUILD_DIR:-$HERE/.server_build}"
SERVER_PKG="$BUILD_DIR/packages/server"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL is not set (postgresql://... with pgvector available)" >&2
  exit 1
fi

# --- Best-effort server config (documented in docs/DECISIONS.md) -------------
# Embedding mode. The contract config is remote, the same embedder as the
# other providers: the shared bge-small, through embed_proxy.py, zero-padded
# to 1024 dimensions. On a host with no GPU, the default here is local for
# convenience. Local is off contract: it uses an in-process
# Xenova/bge-large-en-v1.5, a different embedder, so its numbers are not
# comparable to the other providers. Use it only to prove the wiring works.
# RETAINDB_EMBEDDING_MODE=remote points the server at a bespoke inference
# service through EMBEDDING_INFERENCE_BASE_URL, which the operator must
# export, for example a running embed_proxy.py. We then set the two
# required-strictness env vars, for loud failure and batch timeout. Never
# leave EMBEDDING_MODE unset, because the server default, 'remote', needs a
# base URL and would otherwise error.
RETAINDB_EMBEDDING_MODE="${RETAINDB_EMBEDDING_MODE:-local}"
case "$RETAINDB_EMBEDDING_MODE" in
  local)
    export EMBEDDING_MODE="local"
    ;;
  remote)
    if [[ -z "${EMBEDDING_INFERENCE_BASE_URL:-}" ]]; then
      echo "ERROR: RETAINDB_EMBEDDING_MODE=remote requires EMBEDDING_INFERENCE_BASE_URL exported" >&2
      echo "       (point it at a running embed_proxy.py, e.g. http://127.0.0.1:3199)" >&2
      exit 1
    fi
    export EMBEDDING_MODE="remote"
    export REMOTE_INFERENCE_REQUIRED="true"
    export INFERENCE_TIMEOUT_MS="${INFERENCE_TIMEOUT_MS:-60000}"
    ;;
  *)
    echo "ERROR: unknown RETAINDB_EMBEDDING_MODE=$RETAINDB_EMBEDDING_MODE (expected local|remote)" >&2
    exit 1
    ;;
esac
# DISABLE_SCHEDULER=true turns off the 60-second session-lifecycle
# consolidation job, so the baseline arm stays deterministic and
# reproducible. A scheduler-on arm B is planned for later.
export DISABLE_SCHEDULER="${DISABLE_SCHEDULER:-true}"
# We pass write_mode explicitly per ingest, so leave MEMORY_WRITE_MODE_DEFAULT unset.
export PORT="${PORT:-3000}"
# EXTRACTOR_MODEL is the variable the code reads. EXTRACTION_MODEL, from the
# vendor .env.example, is a dead no-op. Default to the answer model if the
# caller passed OPENAI_MODEL.
export EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-${OPENAI_MODEL:-openai/gpt-oss-120b}}"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARN: OPENAI_API_KEY is empty — RetainDB extraction will silently degrade to regex/pattern-only." >&2
fi
# The BGE model downloads from the HF CDN on first use. Disable Xet, so it
# comes over the classic CDN, matching the local edition's runner.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# ENCRYPTION_KEY, fresh-deploy fix #4 (see server_patches/README.md): the
# server refuses to boot without a non-empty key of at least 32 characters.
# This guards only agent-task connector credential encryption, not the
# memory path. This is a documented, non-secret dev default. Override it for
# a real deployment.
export ENCRYPTION_KEY="${ENCRYPTION_KEY:-retaindb-benchmark-dev-encryption-key-0123456789}"
# RETAINDB_DISABLE_SEARCH_CACHE=true (patch 0004) bypasses both search
# caches. The semantic cache leaks across scopes, because it is keyed only
# on embedding similarity, and the exact-key cache omits question_date, so
# the benchmark's re-asked dynamic questions get stale results inside the
# 300-second TTL. This is disabled for correctness, and is env-overridable
# (the vendor default keeps both caches on).
export RETAINDB_DISABLE_SEARCH_CACHE="${RETAINDB_DISABLE_SEARCH_CACHE:-true}"

# --- pnpm on PATH ------------------------------------------------------------
if ! command -v pnpm >/dev/null 2>&1; then
  if command -v corepack >/dev/null 2>&1; then
    corepack enable >/dev/null 2>&1 || true
    corepack prepare pnpm@9.15.0 --activate >/dev/null 2>&1 || true
  fi
fi
if ! command -v pnpm >/dev/null 2>&1; then
  echo "ERROR: pnpm not found (install node>=20 + corepack, or npm i -g pnpm@9.15.0)" >&2
  exit 1
fi

# --- Copy submodule out + build (idempotent) ---------------------------------
if [[ ! -f "$SERVER_PKG/dist/index.js" ]]; then
  echo "[retaindb-server] building from submodule copy -> $BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  # Copy the submodule contents. Never build in external/. Exclude any
  # nested git metadata and node_modules, so the copy stays clean and
  # reproducible.
  #   -a preserves structure. The trailing /. copies contents, including dotfiles.
  cp -a "$SUBMODULE/." "$BUILD_DIR/"
  rm -rf "$BUILD_DIR/.git" "$BUILD_DIR/node_modules" "$SERVER_PKG/node_modules" 2>/dev/null || true
  # Fresh-deploy patch layer. The vendor server does not run from pristine
  # source (see server_patches/README.md). Apply each .patch, then swap the
  # generator schema for the introspected one, before install and generate.
  # The `prisma migrate deploy` step below still uses the original
  # migrations directory. Only the generator input schema changes.
  ( cd "$BUILD_DIR"
    for p in "$HERE"/server_patches/[0-9]*.patch; do
      echo "[retaindb-server] applying $(basename "$p")"
      git apply -p1 "$p" || patch -p1 < "$p"
    done
    cp "$HERE/server_patches/schema.introspected.prisma" packages/server/prisma/schema.prisma
    pnpm install --frozen-lockfile --filter @retaindb/server...
    pnpm --filter @retaindb/server run build
    pnpm --filter @retaindb/server exec prisma generate )
else
  echo "[retaindb-server] reusing existing build at $SERVER_PKG (dist/index.js present)"
fi

# --- Migrate + post-migrate SQL + serve --------------------------------------
cd "$SERVER_PKG"
echo "[retaindb-server] prisma migrate deploy"
pnpm exec prisma migrate deploy --schema=./prisma/schema.prisma
# Post-migrate SQL, fixes #3 and #7, in order: seed.sql (the default org and
# owner user that the server's project auto-creation references by foreign
# key) then post_migrate.sql (drops the empty-table ivfflat indexes, so
# retrieval uses exact KNN). Both are idempotent. The host path uses psql, a
# documented prereq. The container path uses apply_seed.py (asyncpg). Run
# both only after migrate deploy.
if command -v psql >/dev/null 2>&1; then
  for _sql in seed.sql post_migrate.sql; do
    echo "[retaindb-server] applying $_sql via psql"
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$HERE/server_patches/$_sql"
  done
elif python3 -c "import asyncpg" >/dev/null 2>&1; then
  echo "[retaindb-server] psql absent; applying seed + post_migrate via asyncpg"
  python3 "$HERE/server_patches/apply_seed.py"
else
  echo "ERROR: need psql or python3+asyncpg to apply server_patches/*.sql" >&2
  exit 1
fi
echo "[retaindb-server] starting node dist/index.js on :$PORT (EMBEDDING_MODE=$EMBEDDING_MODE, DISABLE_SCHEDULER=$DISABLE_SCHEDULER, EXTRACTOR_MODEL=$EXTRACTOR_MODEL)"
exec node dist/index.js
