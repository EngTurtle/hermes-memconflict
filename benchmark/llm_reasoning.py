"""Drop-in wrapper around MemConflict's ``llm_request`` that adds OpenRouter
reasoning-effort control.

``xiaomi/mimo-v2.5`` is a reasoning model. Left uncapped, it emits long
hidden reasoning traces, making every answer or judge call take about 13
seconds. That turns a 3,750-question benchmark into a multi-hour run.
Setting ``MEMCONFLICT_REASONING_EFFORT`` (for example ``low``) forwards
``{"reasoning": {"effort": ...}}`` to OpenRouter through the OpenAI SDK's
``extra_body``. This cuts per-call latency about 9x while keeping answers
correct on these factual questions.

The module re-exports ``llm_request`` and ``calculate_cumulative_cost`` with
the same signatures as the upstream module. This lets it swap in for both
the answer generator (``eval_mnemosyne``) and the LLM judge
(``eval_scoring``) by aliasing ``sys.modules['llm_request']`` to this
module.
"""

import importlib.util
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEMCONFLICT_EVAL_DIR = os.environ.get(
    "MEMCONFLICT_EVAL_DIR",
    os.path.join(_CURRENT_DIR, "..", "external", "MemConflict", "Evaluation"),
)
_MEMCONFLICT_EVAL_DIR = os.path.abspath(_MEMCONFLICT_EVAL_DIR)

# Load the upstream llm_request.py under a private module name. This avoids
# a clash with the ``llm_request`` name this module may alias itself to.
_UP_PATH = os.path.join(_MEMCONFLICT_EVAL_DIR, "llm_request.py")
_spec = importlib.util.spec_from_file_location("_memconflict_llm_request", _UP_PATH)
_up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_up)

# Re-export the cost helper unchanged.
calculate_cumulative_cost = _up.calculate_cumulative_cost

_REASONING_EFFORT = os.environ.get("MEMCONFLICT_REASONING_EFFORT", "").strip()
_JSON_MODE = os.environ.get("MEMCONFLICT_JSON_MODE", "").strip().lower() in ("1", "true", "yes", "on")
# Native thinking. When on, this forwards
# chat_template_kwargs={"enable_thinking": true}, so the template emits a
# thinking trace. The vLLM server must run with a matching
# --reasoning-parser (gemma4 for contract v1, qwen3 for v2), so the trace
# splits into reasoning_content and never leaks into the answer text.
_ENABLE_THINKING = os.environ.get("MEMCONFLICT_ENABLE_THINKING", "").strip().lower() in ("1", "true", "yes", "on")
# Contract v1 (gemma4 parser) had to suppress thinking under JSON mode.
# Guided-JSON constrained output from the first token, which is
# incompatible with a leading thought channel (vLLM #39130). Contract v2
# (qwen3 parser) enforces the grammar on the post-reasoning content, so the
# judge can think first and then emit schema-clamped JSON. Grey-area
# correctness judgments benefit from this short deliberation.
# MEMCONFLICT_JSON_THINKING=1 opts in to thinking under JSON mode (the v2
# judge default set by answer_env.sh, paired with a low or medium
# reasoning-effort budget). Leaving it unset or 0 keeps the legacy suppress
# behavior.
_JSON_THINKING = os.environ.get("MEMCONFLICT_JSON_THINKING", "").strip().lower() in ("1", "true", "yes", "on")

# Sampling knobs the upstream llm_request does not plumb, because it only
# sends temperature and max_tokens. Qwen3.5's model card prescribes full
# sampling sets per mode (for example thinking-general: temp 1.0, top_p
# 0.95, top_k 20, min_p 0, presence_penalty 1.5). Near-greedy decoding, the
# old temp 0.2, made the think channel loop or ruminate. We measured this as
# empty, truncated answers. top_p and presence_penalty are standard OpenAI
# request parameters. top_k and min_p are vLLM extensions sent through
# extra_body. Each injects only when its env var is set, so non-canonical
# callers stay untouched.
def _opt_float(name):
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


_TOP_P = _opt_float("MEMCONFLICT_TOP_P")
_TOP_K = _opt_float("MEMCONFLICT_TOP_K")
_MIN_P = _opt_float("MEMCONFLICT_MIN_P")
_PRESENCE_PENALTY = _opt_float("MEMCONFLICT_PRESENCE_PENALTY")

# Per-request timeout. Upstream hardcodes `timeout: int = 300` as a default
# argument (llm_request.py:209) and always passes it explicitly into
# client.chat.completions.create. So no environment variable, client
# default, or anything short of editing external/ (pinned and off-limits)
# can raise it. Overriding the kwarg here is the only lever.
#
# This value must move with judge concurrency. Per-call latency scales
# roughly with worker count, so a concurrency raise without a matching
# timeout raise converts "slower" into "TimeoutError, then retry, then more
# load", the Class C cascade that stopped Arm B. Measured 2026-07-21: about
# 146 s per call at 24 workers, projecting about 243 s at 40. So 300 s
# leaves no margin at all for tail latency.
# Cost of raising it: a hung call holds its worker for this long, and a
# container restart waits this long for in-flight sockets to expire.
# CLAUDE.md's roughly 8-minute recovery grows proportionally longer.
_REQUEST_TIMEOUT = _opt_float("MEMCONFLICT_REQUEST_TIMEOUT")


def _wrap_client_with_reasoning(client):
    """Wrap client.chat.completions.create to carry reasoning effort, thinking, and JSON mode.

    JSON mode (``MEMCONFLICT_JSON_MODE=1``) forwards
    ``response_format={"type": "json_object"}``. For xiaomi/mimo-v2.5, this
    both forces valid JSON and suppresses the model's runaway reasoning
    trace. The LLM judge is fast and reliable with it. Without it, mimo
    frequently burns the whole token budget on reasoning and returns empty
    content, which then falls back to rule-based scoring. Enable this only
    for JSON-output calls, such as the judge, never for free-text answer
    generation.
    """
    effort = _REASONING_EFFORT
    thinking = _ENABLE_THINKING and (not _JSON_MODE or _JSON_THINKING)
    sampling = any(v is not None for v in (_TOP_P, _TOP_K, _MIN_P, _PRESENCE_PENALTY))
    if (not effort and not _JSON_MODE and not thinking and not sampling
            and _REQUEST_TIMEOUT is None):
        return client
    orig_create = client.chat.completions.create

    def create_with_reasoning(**kwargs):
        if effort or thinking or _TOP_K is not None or _MIN_P is not None:
            extra_body = dict(kwargs.get("extra_body") or {})
            if effort:
                # This sends both shapes: OpenRouter reads reasoning.effort,
                # and vLLM's OpenAI server reads top-level reasoning_effort.
                # Either server ignores unknown extra fields and logs them,
                # so sending both is harmless and keeps one code path for
                # both backends.
                extra_body.setdefault("reasoning", {"effort": effort})
                extra_body.setdefault("reasoning_effort", effort)
            if thinking:
                ctk = dict(extra_body.get("chat_template_kwargs") or {})
                ctk.setdefault("enable_thinking", True)
                extra_body["chat_template_kwargs"] = ctk
            if _TOP_K is not None:
                extra_body.setdefault("top_k", int(_TOP_K))
            if _MIN_P is not None:
                extra_body.setdefault("min_p", _MIN_P)
            kwargs["extra_body"] = extra_body
        if _TOP_P is not None:
            kwargs.setdefault("top_p", _TOP_P)
        if _PRESENCE_PENALTY is not None:
            kwargs.setdefault("presence_penalty", _PRESENCE_PENALTY)
        if _JSON_MODE and "response_format" not in kwargs:
            kwargs["response_format"] = {"type": "json_object"}
        if _REQUEST_TIMEOUT is not None:
            # This assigns directly rather than using setdefault, because
            # upstream always passes timeout=300 explicitly, so a
            # setdefault would never fire.
            kwargs["timeout"] = _REQUEST_TIMEOUT
        return orig_create(**kwargs)

    client.chat.completions.create = create_with_reasoning
    return client


# Patch the upstream module's client factory. llm_request() calls the
# module-global _get_client(), so this replacement takes effect on every call.
_orig_get_client = _up._get_client


def _get_client():
    return _wrap_client_with_reasoning(_orig_get_client())


_up._get_client = _get_client

# Re-export the (now reasoning-aware) request function unchanged.
llm_request = _up.llm_request


def install_as_llm_request():
    """Alias this module as ``llm_request`` in sys.modules.

    Call this before importing ``eval_scoring``, so the upstream judge picks
    up the reasoning-aware ``llm_request`` and ``calculate_cumulative_cost``.
    """
    sys.modules["llm_request"] = sys.modules[__name__]
