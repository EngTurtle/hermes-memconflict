#!/usr/bin/env python3
"""Per-run vLLM token accounting from the Prometheus /metrics endpoint.

WHY THIS EXISTS: the harness sees only the tokens it spends itself (answer and
judge calls). Each provider also spends serving tokens the harness never
observes: Mnemosyne's sleep and model-refresh, Hindsight's retain and
consolidation, mem0's ADD/UPDATE decisions, Supermemory's memory agent, and
RetainDB's extractor. These calls go straight from the provider process to
vllm-gen, so only the server itself counts all of them. vLLM exports
cumulative counters on ``/metrics``. A snapshot before and after a run turns
them into a per-run bill.

Usage (both subcommands are best-effort by contract — see EXIT CODES):

    python benchmark/token_usage.py snapshot --out /tmp/tok_start.json
    python benchmark/token_usage.py finish  --start /tmp/tok_start.json \
        --out <provider>/Results/token_usage_<RUN_TAG>.json --scope run

Endpoints default to the generation server derived from ``OPENAI_BASE_URL`` and
the embedding server derived from the first set provider embedding variable
(see _default_embed_url). ``--gen_url`` / ``--embed_url`` override these, and
``BENCH_TOKENS_GEN_URL`` / ``BENCH_TOKENS_EMBED_URL`` override the same
defaults from the environment (the host-side launcher uses the published
ports).

EXIT CODES: always 0, unless the arguments themselves are wrong. A snapshot
that failed is written as ``{"ok": false, "error": ...}``, and a delta built
over such a snapshot is written with ``"valid": false`` plus a reason. Token
accounting must never fail a benchmark run. It must only report that it does
not know.

COUNTER SEMANTICS. vLLM's counters are cumulative for the life of the engine
process, so a delta is meaningful only if the same process served the whole
window. Two guards apply:
  * ``process_start_time_seconds`` (prometheus_client's own gauge) is recorded
    in every snapshot. If it moves, the server restarted, so mark
    ``valid: false``.
  * Any counter that went down also means a restart or a metrics reset, so
    mark ``valid: false``.
Metric NAMES are matched by a suffix regex, not an exact string. vLLM has
renamed and re-labelled these counters across versions (``vllm:prompt_tokens_total``
is current, and label sets vary by model and engine). We sum every label
series whose name ends in the target suffix, and accept both the
``vllm:``-prefixed and bare forms.

ATTRIBUTION CAVEAT (recorded in every sidecar): the delta attributes all
server traffic in the window to this run. Two concurrent runs against one
vllm-gen, or a stray probe, land in the same counters. The sidecar states this
caveat rather than hiding it.
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.request

_TIMEOUT_S = float(os.environ.get("BENCH_TOKENS_TIMEOUT_S", "10"))

# Maps a name suffix to a canonical field. The regex matches the metric NAME
# only, and we sum across labels, so `vllm:prompt_tokens_total{model_name="qwen3.5-4b"}`
# and a bare `prompt_tokens_total` both land in prompt_tokens.
_COUNTER_PATTERNS = (
    ("prompt_tokens", re.compile(r"(?:^|:)prompt_tokens(?:_total)?$")),
    ("generation_tokens", re.compile(r"(?:^|:)generation_tokens(?:_total)?$")),
    ("requests_success", re.compile(r"(?:^|:)request_success(?:_total)?$")),
    # Prefix-cache counters, both in TOKENS (vLLM's own unit for these). With
    # prefix caching off, prefix_cache_queries stays at 0. This is how we tell
    # "the feature never engaged" from "engaged but missed" (hit_rate is then
    # null, never 0.0). Both counters are present on vllm-gen and vllm-embed,
    # but keep the fields None-safe downstream. A server that omits them must
    # not invalidate an otherwise good token snapshot.
    ("prefix_cache_queries", re.compile(r"(?:^|:)prefix_cache_queries(?:_total)?$")),
    ("prefix_cache_hits", re.compile(r"(?:^|:)prefix_cache_hits(?:_total)?$")),
)
# Engine-identity and restart markers: single-valued gauges, so the last value read wins.
_MARKER_PATTERNS = (
    ("process_start_time_seconds",
     re.compile(r"(?:^|:)process_start_time_seconds$")),
)

_METRIC_LINE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>[-+0-9.eEnaN]+)$")


def _metrics_url(base):
    """Convert a base URL to its /metrics form.

    Accepts http://h:8000, .../v1, .../v1/, or .../metrics as input."""
    if not base:
        return None
    url = base.strip().rstrip("/")
    if url.endswith("/metrics"):
        return url
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url + "/metrics"


def _default_gen_url():
    return _metrics_url(os.environ.get("BENCH_TOKENS_GEN_URL")
                        or os.environ.get("OPENAI_BASE_URL"))


def _default_embed_url():
    # Reads whichever provider-specific embedding endpoint this container was
    # given. All of them point at the shared vllm-embed service in the Docker
    # contract.
    for var in ("BENCH_TOKENS_EMBED_URL",
                "MNEMOSYNE_EMBEDDING_API_URL",
                "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL",
                "MEM0_EMBEDDER_BASE_URL",
                "SUPERMEMORY_EMBEDDING_BASE_URL",
                "EMBED_PROXY_UPSTREAM_BASE_URL"):
        value = os.environ.get(var, "").strip()
        if value:
            return _metrics_url(value)
    return None


def _fetch(url):
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "replace")


def _parse(text):
    counters = {field: 0.0 for field, _ in _COUNTER_PATTERNS}
    seen = {field: False for field in counters}
    markers = {}
    matched_names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        for field, pattern in _COUNTER_PATTERNS:
            if pattern.search(name):
                counters[field] += value
                seen[field] = True
                matched_names.add(name)
        for field, pattern in _MARKER_PATTERNS:
            if pattern.search(name):
                markers[field] = value
                matched_names.add(name)
    out = {field: (int(counters[field]) if seen[field] else None)
           for field in counters}
    out["markers"] = markers
    out["matched_metric_names"] = sorted(matched_names)
    return out


def snapshot(url):
    """Snapshot one server's cumulative counters.

    Always returns a dict. Sets ``ok: false`` with an ``error`` string when
    the endpoint was unreachable or served no recognized counters.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if not url:
        return {"ok": False, "url": None, "at": now,
                "error": "no metrics URL resolved"}
    try:
        parsed = _parse(_fetch(url))
    except Exception as exc:
        return {"ok": False, "url": url, "at": now,
                "error": "%s: %s" % (type(exc).__name__, exc)}
    if parsed["prompt_tokens"] is None and parsed["generation_tokens"] is None:
        return {"ok": False, "url": url, "at": now,
                "error": "no prompt/generation token counters found at endpoint",
                "matched_metric_names": parsed["matched_metric_names"]}
    return {
        "ok": True,
        "url": url,
        "at": now,
        "prompt_tokens": parsed["prompt_tokens"],
        "generation_tokens": parsed["generation_tokens"],
        "requests": parsed["requests_success"],
        "prefix_cache_queries": parsed["prefix_cache_queries"],
        "prefix_cache_hits": parsed["prefix_cache_hits"],
        # Restart detector: prometheus_client's process start gauge. The name
        # is generic because this marks whether the same engine served the
        # whole window; it is not a token count.
        "engine_start_marker": parsed["markers"].get("process_start_time_seconds"),
        "matched_metric_names": parsed["matched_metric_names"],
    }


def snapshot_all(gen_url, embed_url):
    return {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "servers": {
            "vllm_gen": snapshot(gen_url),
            "vllm_embed": snapshot(embed_url),
        },
    }


def delta(start, end):
    """Per-server delta with an explicit validity verdict.

    Sets ``valid: false`` plus ``invalid_reason`` whenever the numbers cannot
    be trusted: a failed snapshot on either side, a moved engine-start marker
    (a restart), or a counter that decreased.
    """
    out = {}
    for server in ("vllm_gen", "vllm_embed"):
        s = (start.get("servers") or {}).get(server) or {}
        e = (end.get("servers") or {}).get(server) or {}
        row = {"start_at": s.get("at"), "end_at": e.get("at")}
        if not s.get("ok") or not e.get("ok"):
            row.update({
                "valid": False,
                "invalid_reason": "snapshot failed (start=%s, end=%s)"
                                  % (s.get("error") or "ok", e.get("error") or "ok"),
                "prompt_tokens": None, "generation_tokens": None, "requests": None,
                "prefix_cache_queries": None, "prefix_cache_hits": None,
                "prefix_cache_hit_rate": None,
            })
            out[server] = row
            continue
        if (s.get("engine_start_marker") is not None
                and e.get("engine_start_marker") is not None
                and s["engine_start_marker"] != e["engine_start_marker"]):
            row.update({
                "valid": False,
                "invalid_reason": "server restarted during the window "
                                  "(process_start_time_seconds %s -> %s)"
                                  % (s["engine_start_marker"], e["engine_start_marker"]),
                "prompt_tokens": None, "generation_tokens": None, "requests": None,
                "prefix_cache_queries": None, "prefix_cache_hits": None,
                "prefix_cache_hit_rate": None,
            })
            out[server] = row
            continue
        regressed = []
        values = {}
        for field in ("prompt_tokens", "generation_tokens", "requests",
                      "prefix_cache_queries", "prefix_cache_hits"):
            a, b = s.get(field), e.get(field)
            if a is None or b is None:
                values[field] = None
                continue
            if b < a:
                regressed.append("%s %s -> %s" % (field, a, b))
            values[field] = b - a
        if regressed:
            row.update({
                "valid": False,
                "invalid_reason": "counter regressed, server restarted or metrics "
                                  "reset: " + "; ".join(regressed),
                "prompt_tokens": None, "generation_tokens": None, "requests": None,
                "prefix_cache_queries": None, "prefix_cache_hits": None,
                "prefix_cache_hit_rate": None,
            })
        else:
            row.update({"valid": True, "invalid_reason": None})
            row.update(values)
            if values.get("prompt_tokens") is not None \
                    and values.get("generation_tokens") is not None:
                row["total_tokens"] = values["prompt_tokens"] + values["generation_tokens"]
            # Computes the hit rate over this run's window, from the deltas,
            # not from the cumulative totals (which a long-lived server lets
            # earlier runs dominate). Returns None, not 0.0, when the counters
            # are absent or the window queried nothing, so "no data" stays
            # distinguishable from a genuine 0% hit rate.
            queries = values.get("prefix_cache_queries")
            hits = values.get("prefix_cache_hits")
            if queries is None or hits is None or queries <= 0:
                row["prefix_cache_hit_rate"] = None
            else:
                row["prefix_cache_hit_rate"] = round(hits / queries, 6)
        row["cumulative_end"] = {
            "prompt_tokens": e.get("prompt_tokens"),
            "generation_tokens": e.get("generation_tokens"),
            "requests": e.get("requests"),
            "prefix_cache_queries": e.get("prefix_cache_queries"),
            "prefix_cache_hits": e.get("prefix_cache_hits"),
        }
        out[server] = row
    return out


def _write(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gen_url", default=None,
                        help="vllm-gen metrics URL (default: from OPENAI_BASE_URL).")
    common.add_argument("--embed_url", default=None,
                        help="vllm-embed metrics URL (default: provider embed var).")
    common.add_argument("--out", required=True)

    p_snap = sub.add_parser("snapshot", parents=[common],
                            help="write a start snapshot.")
    p_snap.set_defaults(cmd="snapshot")

    p_fin = sub.add_parser("finish", parents=[common],
                           help="take an end snapshot and write the delta sidecar.")
    p_fin.add_argument("--start", required=True, help="the snapshot file to diff against.")
    p_fin.add_argument("--run_tag", default=os.environ.get("RUN_TAG", ""))
    p_fin.add_argument("--provider", default="")
    p_fin.add_argument("--scope", default="run", choices=["run", "shard"],
                       help="'run' = whole wave (launcher), 'shard' = one container.")
    p_fin.set_defaults(cmd="finish")

    args = ap.parse_args(argv)
    gen_url = _metrics_url(args.gen_url) if args.gen_url else _default_gen_url()
    embed_url = _metrics_url(args.embed_url) if args.embed_url else _default_embed_url()

    if args.cmd == "snapshot":
        snap = snapshot_all(gen_url, embed_url)
        _write(args.out, snap)
        for name, row in sorted(snap["servers"].items()):
            print("[token_usage] snapshot %s: %s" % (
                name, "prompt=%s gen=%s requests=%s" % (
                    row.get("prompt_tokens"), row.get("generation_tokens"),
                    row.get("requests")) if row.get("ok")
                else "UNAVAILABLE (%s)" % row.get("error")))
        return 0

    end = snapshot_all(gen_url, embed_url)
    start = None
    start_error = None
    try:
        with open(args.start, "r", encoding="utf-8") as fh:
            start = json.load(fh)
    except Exception as exc:
        start_error = "%s: %s" % (type(exc).__name__, exc)
    if start is None:
        sidecar = {
            "run_tag": args.run_tag, "provider": args.provider, "scope": args.scope,
            "valid": False,
            "invalid_reason": "start snapshot unreadable (%s): %s"
                              % (args.start, start_error),
            "end_snapshot": end,
        }
    else:
        per_server = delta(start, end)
        sidecar = {
            "run_tag": args.run_tag,
            "provider": args.provider,
            "scope": args.scope,
            "valid": all(row.get("valid") for row in per_server.values()),
            "invalid_reason": "; ".join(
                "%s: %s" % (name, row["invalid_reason"])
                for name, row in sorted(per_server.items())
                if row.get("invalid_reason")) or None,
            "window": {"start": start.get("captured_at"),
                       "end": end.get("captured_at")},
            "servers": per_server,
            "start_snapshot_file": os.path.abspath(args.start),
        }
    sidecar["concurrent_use_caveat"] = (
        "These deltas are vLLM's own cumulative counters differenced over the run "
        "window, so they attribute ALL traffic that any client sent to the server "
        "during that window to this run — including a concurrent run, a probe, or "
        "another provider's shards. Per-shard files (scope=shard) each cover the "
        "WHOLE server for their window and must NOT be summed; use the scope=run "
        "file for a wave total."
    )
    sidecar["metric_source"] = (
        "vLLM Prometheus /metrics: prompt_tokens_total, generation_tokens_total, "
        "request_success_total, prefix_cache_queries_total, "
        "prefix_cache_hits_total (name-suffix matched, all label series summed); "
        "restart detected via process_start_time_seconds. prefix_cache_* are in "
        "TOKENS; prefix_cache_hit_rate is hits/queries over THIS run's window, "
        "null when the counters are absent (vllm-embed) or queries==0 (prefix "
        "caching disabled or never consulted). NOTE: prompt_tokens_total counts "
        "logical prompt tokens regardless of cache hits, so it stays comparable "
        "across providers but does NOT reflect prefill compute saved by the cache."
    )
    _write(args.out, sidecar)
    print("[token_usage] wrote %s (valid=%s%s)" % (
        args.out, sidecar["valid"],
        "" if sidecar["valid"] else ": " + str(sidecar.get("invalid_reason"))))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # never fail a run over accounting
        print("[token_usage] WARN: %s" % exc, file=sys.stderr)
        sys.exit(0)
