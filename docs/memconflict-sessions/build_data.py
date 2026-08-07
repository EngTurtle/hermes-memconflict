#!/usr/bin/env python3
"""Build web/data.js from Data/Step4_4.jsonl.

The conversation browser opens directly from disk (file://), where fetch() is
blocked by the browser's same-origin policy. So instead of fetching the JSONL at
runtime, we ship the data as a JS file that assigns a global via a <script src>
tag (which file:// allows).

This script performs the sole build step: it reads the released benchmark file,
flattens/normalizes each session's dialogue into an ordered message list, trims
the payload to what the UI needs, and writes:

    web/data.js  ->  window.MEMCONFLICT_DATA = JSON.parse("<escaped json>");

JSON.parse of a string literal is both simpler to emit and faster for the JS
engine to parse than a giant inline object literal.

Re-run only when Data/Step4_4.jsonl changes:

    python3 web/build_data.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "Data" / "Step4_4.jsonl"
OUT = REPO_ROOT / "web" / "data.js"

_TURN_RE = re.compile(r"dialogue_turn_(\d+)$")


def turn_num(key: str) -> int:
    """Numeric index of a dialogue_turn_N key (so turn_10 sorts after turn_9)."""
    m = _TURN_RE.match(key)
    return int(m.group(1)) if m else 0


def normalize_message(msg):
    """Return (role, content) for a message object, tolerating malformed shapes.

    Observed shapes in the dataset:
      - {"role": "user", "content": "..."}                (normal)
      - {"role": "user", "content": "...", "name": "..."} (harmless extra key)
      - {"role": "assistant"}                             (content missing)
      - {"assistant": "content", "content": "..."}        (bogus sentinel key)
      - {"assistant": {"content": "..."}}                 (role missing, nested)
    """
    if not isinstance(msg, dict):
        return "unknown", str(msg) if msg is not None else ""

    if "role" in msg:
        content = msg.get("content")
        if isinstance(content, str):
            return msg["role"], content
        if isinstance(content, dict):  # defensive: nested content under content
            return msg["role"], content.get("content", "") or ""
        return msg["role"], content or ""

    for role_key in ("assistant", "user", "system"):
        if role_key in msg:
            val = msg[role_key]
            if isinstance(val, dict):  # {"assistant": {"content": "..."}}
                return role_key, val.get("content", "") or ""
            # {"assistant": "content", "content": "..."} -> real text is in content
            return role_key, msg.get("content", "") or ""

    return "unknown", ""


def flatten_dialogue(session_dialogue):
    """Flatten {dialogue_turn_N: [msg, ...]} into an ordered message list.

    Turns are NOT reliably [user, assistant] pairs (some have 1 or 3 messages),
    so we iterate every message in numeric turn order rather than indexing.
    """
    messages = []
    if not isinstance(session_dialogue, dict):
        return messages
    for key in sorted(session_dialogue.keys(), key=turn_num):
        turn = turn_num(key)
        turn_msgs = session_dialogue[key]
        if not isinstance(turn_msgs, list):
            turn_msgs = [turn_msgs]
        for msg in turn_msgs:
            role, content = normalize_message(msg)
            messages.append({"turn": turn, "role": role, "content": content})
    return messages


def build_session(s):
    return {
        "session_id": s.get("Session_ID"),
        "date": s.get("Date"),
        "session_type": s.get("Session_Type"),
        "outline": s.get("Session_Outline", ""),
        "event_types": s.get("Event_Types", []),
        "messages": flatten_dialogue(s.get("Session_Dialogue", {})),
        "static_conflicts": s.get("Static_Conflict_Information", []),
        "conditional_conflicts": s.get("Conditional_Conflict_Information", []),
        "others_dynamic": s.get("Others_Dynamic_Information", []),
        "revealed_attributes": s.get("Revealed_Attributes", {}),
        "session_questions": s.get("Session_Questions", []),
    }


def build_persona(idx, obj):
    fixed = obj.get("Fixed_Profile", {})
    personality = obj.get("Personality", {})
    sessions = [build_session(s) for s in obj.get("Full_Session_Chain", [])]
    return {
        "idx": idx,
        "id": obj.get("ID"),
        "name": fixed.get("Name", f"Persona {idx}"),
        "persona_seed": obj.get("metadata", {}).get("persona_seed", ""),
        "gender": fixed.get("Gender", ""),
        "mbti": personality.get("MBTI", ""),
        # Profile blocks surfaced in the details panel (kept verbatim).
        "fixed_profile": fixed,
        "dynamic_profile": obj.get("Dynamic_Profile", {}),
        "preference_profile": obj.get("Preference_Profile", {}),
        "personality": personality,
        "life_goal": obj.get("Life_Goal", {}),
        "others_profile": obj.get("Others_Profile", {}),
        "sessions": sessions,
    }


def main():
    if not SRC.exists():
        sys.exit(f"Source data not found: {SRC}")

    personas = []
    total_sessions = 0
    total_messages = 0
    with SRC.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ! skipping line {idx}: JSON error: {e}", file=sys.stderr)
                continue
            persona = build_persona(idx, obj)
            personas.append(persona)
            total_sessions += len(persona["sessions"])
            total_messages += sum(len(s["messages"]) for s in persona["sessions"])

    payload = json.dumps(personas, ensure_ascii=False, separators=(",", ":"))
    # Outer dumps escapes the whole payload into a single JS/JSON string literal.
    js = "window.MEMCONFLICT_DATA = JSON.parse(" + json.dumps(payload) + ");\n"
    OUT.write_text(js, encoding="utf-8")

    size_mb = OUT.stat().st_size / 1e6
    print(
        f"Wrote {len(personas)} personas, {total_sessions} sessions, "
        f"{total_messages} messages to {OUT} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
