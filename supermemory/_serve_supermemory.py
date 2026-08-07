"""Boot ONE long-running self-hosted Supermemory server as a shared central
store, publish its API key to a shared file, and block until it exits.

This is the "central DB" for sharded runs (the analog of Hindsight's shared
``hindsight-pg`` Postgres service). Many shard containers each run
``eval_supermemory.py`` in ATTACH mode against this one server, isolated per
run by containerTag namespace. Hindsight had to externalize an in-process DB
to Postgres to share it. Supermemory's server is already an HTTP service that
owns an embedded graph engine, so sharing it only means "run one, point the
shards at it".

Key handoff (robust across restarts with a persistent data dir):
  * If the shared key-file already exists (a prior boot wrote it), or
    ``SUPERMEMORY_API_KEY`` is preset, this script passes that key to the
    server as its bearer key and uses it for the readiness ping. A restart
    with a persisted ``SUPERMEMORY_DATA_DIR`` (which keeps the same key)
    therefore just works, and the server never needs to reprint it.
  * Otherwise, on a true first boot, the server generates a key and prints
    it on its boot banner. ``SupermemoryServer.start()`` captures it, and
    this script persists it to the key-file for the shards to read.

Env vars (set by entrypoint.supermemory-server.sh):
  PORT / SUPERMEMORY_PORT        fixed port the shards dial (e.g. 8787)
  SUPERMEMORY_DATA_DIR           persistent embedded-store dir (named volume)
  SUPERMEMORY_SHARED_DIR         dir on the shared volume for the key-file (default /shared)
  SUPERMEMORY_API_KEY            optional preset bearer
  SUPERMEMORY_LLM_* / OPENAI_*   internal extraction LLM (see _supermemory_server.py)
  SUPERMEMORY_EMBEDDING_*        embeddings
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _supermemory_server import SupermemoryServer  # noqa: E402


def main() -> int:
    port = int(os.environ.get("PORT") or os.environ.get("SUPERMEMORY_PORT") or 8787)
    data_dir = os.environ.get("SUPERMEMORY_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".supermemory_runs", "central")
    shared_dir = os.environ.get("SUPERMEMORY_SHARED_DIR", "/shared")
    key_file = os.path.join(shared_dir, "api_key")
    os.makedirs(shared_dir, exist_ok=True)

    # Prefer a persisted or preset key, so a restart against the same data
    # dir reuses the stable bearer key (the server does not reprint it on a
    # non-first boot).
    preset = None
    if os.path.isfile(key_file):
        preset = open(key_file, encoding="utf-8").read().strip() or None
        if preset:
            print(f"[serve] reusing persisted API key from {key_file} "
                  f"({preset[:6]}...)", flush=True)
    if not preset:
        env_key = os.environ.get("SUPERMEMORY_API_KEY", "").strip()
        preset = env_key or None

    server = SupermemoryServer(data_dir=data_dir, port=port, api_key=preset)
    server.start(ready_timeout=float(os.environ.get("SUPERMEMORY_READY_TIMEOUT", "600")))

    # Publish the captured or preset key for the shards. This writes the
    # file only AFTER the server answers its readiness ping, so the compose
    # healthcheck (key-file exists and port responds) is a true readiness
    # gate for `depends_on`.
    key = server.api_key or ""
    tmp = key_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(key)
    os.replace(tmp, key_file)
    try:
        os.chmod(key_file, 0o644)
    except Exception:
        pass
    print(f"[serve] central Supermemory server READY at http://0.0.0.0:{port} "
          f"— key published to {key_file}", flush=True)

    # Block on the server process. Propagate its exit code, so the container
    # dies with it (the compose restart policy then applies).
    proc = server._proc  # noqa: SLF001 (intentional: this script owns this process)
    if proc is None:
        print("[serve] FATAL: server process handle is None", file=sys.stderr, flush=True)
        return 1
    try:
        return proc.wait()
    except KeyboardInterrupt:
        server.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
