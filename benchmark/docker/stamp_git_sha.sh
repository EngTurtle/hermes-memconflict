#!/usr/bin/env bash
# Stamp the launching repo commit where a provider CONTAINER can read it.
#
# The provider images bake in the source and bind-mount `benchmark/` and the
# provider directory, but never `.git`. So a container cannot derive the
# commit itself. GIT_SHA is the primary channel. Every provider service
# declares it in docker-compose.yml, and benchmark/docker/run_shards.sh
# exports it. But a hand-launched `docker compose run ... mnemosyne` inherits
# nothing. That is exactly why the mnemosyne v4-minimal manifest recorded
# head_sha:null, while every sharded provider recorded a real SHA.
#
# This script writes benchmark/.git_sha. write_manifest.py reads that file as
# its last resort (BENCH_GIT_SHA_FILE overrides the path). benchmark/ IS
# bind-mounted, so the stamp reaches every provider container with no compose
# change. Run it once per checkout state before a wave:
#
#   benchmark/docker/stamp_git_sha.sh
#
# The output is an untracked run artifact, like the Results/Scores files. Do
# not commit it. Re-run this script after any commit so the stamp cannot go
# stale silently. The manifest records head_sha_source, so a stamp-sourced SHA
# is always distinguishable from a git-read one.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$REPO_ROOT/benchmark/.git_sha}"

SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
if [ -z "$SHA" ]; then
  echo "stamp_git_sha.sh: not a git checkout at $REPO_ROOT — nothing stamped" >&2
  exit 1
fi

printf '%s\n' "$SHA" > "$OUT"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null || true)" ]; then
  echo "stamp_git_sha.sh: WARNING — worktree is DIRTY, so $SHA does not describe" \
       "the code that will actually run." >&2
fi
echo "stamp_git_sha.sh: wrote $OUT ($SHA)"
