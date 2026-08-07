"""Build oracle canonical slots for the MemConflict --oracle arm.

This module converts a persona's own gold annotations into canonical
identity slots with versions. The history-aware retrieval path can then
run without Mnemosyne's LLM model-refresh step. This is the upper-bound
arm. The module reads gold data only at build time. It never reads the
question answers.

Gold sources:
  * Dynamic: ``Revealed_Attributes`` (initial_reveal sessions) give the
              version-1 value of each attribute. ``Updated_Attributes``
              rows (``{Attribute, Before, After}``) give each later
              version, effective at their session. The code splits
              nested-dict attributes (for example Career_Status) into
              one slot per sub-field. It compares Before and After to
              find which sub-fields changed, because sub-fields change
              on their own.
  * Static:   ``Static_Conflict_Information`` gives one fixed version
              per ``Conflict_ID``, taken from the ``Point_A`` row. The
              gold data says Point_A stays correct. Point_B is a
              contradicting claim, so a version for Point_B would
              contradict the dataset's own answers. The code drops
              ``Distractor`` rows (other people or false leads).
  * Conditional: ``Conditional_Conflict_Information`` ``Point_*`` rows
              hold values tied to a condition. These values coexist,
              they do not replace each other. Each new point rewrites
              the slot body to the full accumulated list, in the form
              "Item (when: Condition); ...". The current body always
              holds every active preference, and the history shows how
              the list grew.

Return shape: ``{session_position: [{"category","name","body"}, ...]}``.
The key is the session's position in ``Full_Session_Chain``, which is
what the eval loop enumerates. The key is not the ``Session_ID`` field.
"""

from typing import Any, Dict, List

# CanonicalStore.remember() checks the body. The same body is a no-op.
# A new body replaces the old one.
DYNAMIC_CATEGORY = "dynamic"
STATIC_CATEGORY = "static"
CONDITIONAL_CATEGORY = "conditional"

_MAX_BODY = 400


def _stringify(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"{k}: {_stringify(v)}" for k, v in value.items()]
        return "; ".join(parts)
    if isinstance(value, list):
        return "; ".join(_stringify(v) for v in value)
    return str(value).strip()


def _emit(out: Dict[int, List[Dict[str, str]]], pos: int,
          category: str, name: str, body: Any) -> None:
    text = _stringify(body)[:_MAX_BODY]
    if not text or not str(name).strip():
        return
    out.setdefault(pos, []).append({
        "category": category, "name": str(name).strip(), "body": text,
    })


def _emit_dynamic_value(out: Dict[int, List[Dict[str, str]]], pos: int,
                        attribute: str, value: Any) -> None:
    """A scalar attribute makes one slot. A dict attribute makes one slot for each sub-field."""
    if isinstance(value, dict):
        for sub_key, sub_val in value.items():
            _emit(out, pos, DYNAMIC_CATEGORY, f"{attribute}.{sub_key}", sub_val)
    else:
        _emit(out, pos, DYNAMIC_CATEGORY, attribute, value)


def _emit_dynamic_update(out: Dict[int, List[Dict[str, str]]], pos: int,
                         update_row: Dict[str, Any]) -> None:
    attribute = update_row.get("Attribute")
    before = update_row.get("Before")
    after = update_row.get("After")
    if not attribute or after in (None, ""):
        return
    if isinstance(after, dict):
        before_dict = before if isinstance(before, dict) else {}
        # Only changed sub-fields become new slot versions.
        # Rewriting an unchanged sub-field would be a no-op in the store.
        # The diff step keeps the slot list correct for diagnostics.
        for sub_key, sub_val in after.items():
            if before_dict.get(sub_key) != sub_val:
                _emit(out, pos, DYNAMIC_CATEGORY, f"{attribute}.{sub_key}", sub_val)
    else:
        _emit(out, pos, DYNAMIC_CATEGORY, attribute, after)


def Build_Oracle_Slots_For_Persona(
    persona_item: Dict[str, Any],
) -> Dict[int, List[Dict[str, str]]]:
    chain = persona_item.get("Full_Session_Chain") or []
    out: Dict[int, List[Dict[str, str]]] = {}

    # --- Conditional accumulation state: Conflict_ID maps to an ordered point list ---
    conditional_points: Dict[str, List[str]] = {}
    conditional_slot_name: Dict[str, str] = {}

    for pos, session in enumerate(chain):
        if not isinstance(session, dict):
            continue

        # Dynamic version-1 baselines. Only initial_reveal sessions set these.
        revealed = session.get("Revealed_Attributes")
        if isinstance(revealed, dict):
            for attribute, value in revealed.items():
                _emit_dynamic_value(out, pos, attribute, value)

        for row in session.get("Updated_Attributes") or []:
            if isinstance(row, dict):
                _emit_dynamic_update(out, pos, row)

        # Static: use only the Point_A rows. They hold the canonical value.
        for row in session.get("Static_Conflict_Information") or []:
            if not isinstance(row, dict) or row.get("Role") != "Point_A":
                continue
            _emit(out, pos, STATIC_CATEGORY,
                  row.get("Target_Field_Path"), row.get("Value"))

        # Conditional: build up the list of coexisting Item/Condition points.
        for row in session.get("Conditional_Conflict_Information") or []:
            if not isinstance(row, dict):
                continue
            role = str(row.get("Role", ""))
            if not role.startswith("Point"):
                continue  # Distractor rows describe other people.
            conflict_id = str(row.get("Conflict_ID") or "")
            item = row.get("Item")
            condition = row.get("Condition")
            if not conflict_id or not item:
                continue
            slot_name = str(row.get("Preference_Type") or conflict_id)
            conditional_slot_name.setdefault(conflict_id, slot_name)
            entry = f"{item} (when: {condition})" if condition else str(item)
            points = conditional_points.setdefault(conflict_id, [])
            if entry not in points:
                points.append(entry)
            _emit(out, pos, CONDITIONAL_CATEGORY,
                  conditional_slot_name[conflict_id], "; ".join(points))

    return out
