#!/usr/bin/env python3
"""Capture a serving-envelope sidecar for a run. Run this script inside the run container.

Why this script exists: write_manifest.py always looks for
``<provider>/Scores/serving_envelope_<RUN_TAG>.json``. It writes "no
serving-envelope sidecar found" when the file is absent. All five v4-minimal
runs got this message, because no run had ever written the file. The
manifest's environment snapshot sees only ``OPENAI_MODEL``, the served alias.
The alias stayed the same across contracts v2, v3, and v4. The underlying
checkpoint changed twice in that time. So the only way to recover a result's
real weights was to search the compose file history. This script records
that information at generate time.

What it records:
  * ``/v1/models`` from the generation server and the embedding server. This
    gives the served alias and any metadata vLLM attaches. On several builds,
    ``root`` is the checkpoint path. This script uses it as the real identity.
  * ``/version``, when the build exposes it. This gives the engine banner
    version.
  * Environment passthroughs for facts a container cannot see on its own:
    ``BENCH_SERVING_IMAGE`` and ``BENCH_SERVING_IMAGE_DIGEST``. The host
    launcher sets these from ``docker inspect vllm-gen``. The script also
    records the served alias and base URL the harness was configured with.

Usage:
    python benchmark/capture_serving_envelope.py \
        --provider_dir /app/mnemosyne --run_tag v4min_cs [--strict]

By default this script is best-effort. It prints a warning and exits 0. The
entrypoints set ``--strict`` when ``STRICT_RUN_CONTRACT=1`` or
``BENCH_CLOCKSYNC=1``. With ``--strict``, an unreachable server is fatal and
exits nonzero. A clock-normalized run must not produce artifacts with no
record of the serving side.
"""

import argparse
import datetime
import json
import os
import sys
import urllib.request

_TIMEOUT_S = float(os.environ.get("BENCH_SERVING_PROBE_TIMEOUT_S", "15"))


def _base(url):
    return (url or "").strip().rstrip("/")


def _get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def probe_server(base_url):
    """Probe one OpenAI-compatible server. This function never raises."""
    out = {"base_url": base_url, "models": None, "version": None, "errors": {}}
    if not base_url:
        out["errors"]["base_url"] = "unset"
        return out
    root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
    try:
        out["models"] = _get_json(root + "/v1/models")
    except Exception as exc:
        out["errors"]["models"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        # vLLM's OpenAI server exposes /version on most builds. Older builds return 404.
        out["version"] = _get_json(root + "/version")
    except Exception as exc:
        out["errors"]["version"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def _served_ids(probe):
    data = ((probe or {}).get("models") or {}).get("data") or []
    return [row.get("id") for row in data if isinstance(row, dict)]


def _checkpoint_roots(probe):
    data = ((probe or {}).get("models") or {}).get("data") or []
    return sorted({row.get("root") for row in data
                   if isinstance(row, dict) and row.get("root")})


def _embed_base_url():
    for var in ("BENCH_SERVING_EMBED_BASE_URL",
                "MNEMOSYNE_EMBEDDING_API_URL",
                "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL",
                "MEM0_EMBEDDER_BASE_URL",
                "SUPERMEMORY_EMBEDDING_BASE_URL",
                "EMBED_PROXY_UPSTREAM_BASE_URL"):
        value = _base(os.environ.get(var, ""))
        if value:
            return value
    return ""


def build_envelope(provider, run_tag):
    gen_url = _base(os.environ.get("BENCH_SERVING_GEN_BASE_URL", "")
                    or os.environ.get("OPENAI_BASE_URL", ""))
    embed_url = _embed_base_url()
    gen = probe_server(gen_url)
    embed = probe_server(embed_url)
    envelope = {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "captured_by": "benchmark/capture_serving_envelope.py (in-container probe)",
        "provider": provider,
        "run_tag": run_tag,
        "configured": {
            # This is the value the harness was configured to use. It may not match what actually answered.
            "answer_model_alias": os.environ.get("OPENAI_MODEL"),
            "gen_base_url": gen_url or None,
            "embed_base_url": embed_url or None,
            "embed_model_alias": (os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
                                  or os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL")
                                  or os.environ.get("MEM0_EMBEDDER_MODEL")
                                  or os.environ.get("SUPERMEMORY_EMBEDDING_MODEL")
                                  or os.environ.get("EMBED_PROXY_UPSTREAM_MODEL")),
        },
        # Host-only facts. A container cannot read the compose file or another
        # service's image digest. The launcher passes these values in.
        "image": {
            "vllm_gen_image": os.environ.get("BENCH_SERVING_IMAGE") or None,
            "vllm_gen_image_digest": os.environ.get("BENCH_SERVING_IMAGE_DIGEST") or None,
            "vllm_embed_image": os.environ.get("BENCH_SERVING_EMBED_IMAGE") or None,
            "vllm_embed_image_digest": os.environ.get("BENCH_SERVING_EMBED_IMAGE_DIGEST") or None,
            "note": ("null means the launcher did not export BENCH_SERVING_IMAGE* "
                     "— benchmark/docker/run_shards.sh exports them via "
                     "`docker inspect`; a manual `docker compose run` does not."),
        },
        "vllm_gen": gen,
        "vllm_embed": embed,
        "summary": {
            "gen_served_ids": _served_ids(gen),
            "gen_checkpoint_roots": _checkpoint_roots(gen),
            "gen_engine_version": (gen.get("version") or {}).get("version")
            if isinstance(gen.get("version"), dict) else gen.get("version"),
            "embed_served_ids": _served_ids(embed),
            "embed_checkpoint_roots": _checkpoint_roots(embed),
        },
    }
    envelope["ok"] = bool(envelope["summary"]["gen_served_ids"])
    return envelope


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider_dir", required=True)
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--out", default=None,
                    help="Default: <provider_dir>/Scores/serving_envelope_<run_tag>.json "
                         "— exactly where write_manifest.py looks.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit nonzero when the generation server could not be probed.")
    args = ap.parse_args(argv)

    provider = os.path.basename(os.path.normpath(args.provider_dir))
    out_path = args.out or os.path.join(
        args.provider_dir, "Scores", "serving_envelope_%s.json" % args.run_tag)
    envelope = build_envelope(provider, args.run_tag)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, sort_keys=True)
        fh.write("\n")

    summary = envelope["summary"]
    print("[serving_envelope] wrote %s (gen_ids=%s checkpoint=%s engine=%s)"
          % (out_path, summary["gen_served_ids"] or "NONE",
             summary["gen_checkpoint_roots"] or "unknown",
             summary["gen_engine_version"] or "unknown"))
    if not envelope["ok"]:
        msg = ("could not read /v1/models from the generation server (%s): %s"
               % (envelope["configured"]["gen_base_url"],
                  envelope["vllm_gen"]["errors"]))
        if args.strict:
            print("[serving_envelope] FATAL: %s" % msg, file=sys.stderr)
            return 1
        print("[serving_envelope] WARN: %s" % msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        # A capture crash is fatal only in strict mode. The script checks --strict on argv here.
        strict = "--strict" in (sys.argv or [])
        print("[serving_envelope] %s: %s" % ("FATAL" if strict else "WARN", exc),
              file=sys.stderr)
        sys.exit(1 if strict else 0)
