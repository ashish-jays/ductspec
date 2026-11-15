"""What the sheet says, as the lines a reviewer actually reads.

The structured object knows where every value came from. A reviewer does not:
they have the lettering on the sheet and a takeoff value somebody carried
forward, and have to work out whether the first supports the second. This
module writes that lettering, so the model is given the same thing.

A value the drawing shows can be lettered three ways: on the duct's own tag,
in a note covering a run of ducts, or in a general note for a system. A value
the drawing does not show is simply absent for this duct, and the same value
may still appear on the sheet against something else. Telling those apart is
the task.

`shown_for` reads the lines back. The dataset tests use it to hold the
lettering to the object's provenance, so the two cannot drift apart.
"""

from __future__ import annotations

import re

from .objects import DuctObject, _material_abbrev

SYSTEM_ABBREV = {
    "supply_air": "SA",
    "return_air": "RA",
    "general_exhaust": "EA",
    "kitchen_exhaust": "KEF",
    "lab_exhaust": "LEF",
    "relief_air": "RLA",
    "shower_exhaust": "SHX",
    "dishwasher_exhaust": "DWX",
}
SYSTEM_NAME = {
    "supply_air": "SUPPLY AIR",
    "return_air": "RETURN AIR",
    "general_exhaust": "GENERAL EXHAUST",
    "kitchen_exhaust": "KITCHEN EXHAUST",
    "lab_exhaust": "LAB EXHAUST",
    "relief_air": "RELIEF AIR",
    "shower_exhaust": "SHOWER EXHAUST",
    "dishwasher_exhaust": "DISHWASHER EXHAUST",
}
STYLES = ("tag", "range", "general_note")
DISTRACTORS = ("none", "other_duct", "other_range", "other_system")


def _inches(value: object) -> str:
    v = float(value)
    whole, half = int(v), v - int(v) == 0.5
    if half:
        return f'{whole}-1/2"' if whole else '1/2"'
    return f'{whole}"' if v == whole else f'{v}"'


def value_token(prop: str, value: object, element: str) -> str:
    """The value as a drafter would letter it."""
    if prop == "material":
        return _material_abbrev(value)
    if prop == "insulation_thickness_in":
        return f"{_inches(value)} {'LINER' if element == 'acoustical_liner' else 'INSUL'}"
    if prop == "pressure_rating_in_wg":
        return f"{_inches(value)} WG"
    raise ValueError(f"no lettering for property {prop!r}")


def number(duct_id: str) -> int:
    return int(duct_id.split("-")[1])


def _tag(duct: DuctObject, duct_id: str, token: str = "") -> str:
    parts = [f"{duct_id}: {duct.width_in}x{duct.height_in}"]
    if token:
        parts.append(token)
    parts.append(SYSTEM_ABBREV[duct.system])
    return " ".join(parts)


def _other_system(system: str) -> str:
    names = list(SYSTEM_NAME)
    return names[(names.index(system) + 1) % len(names)]


def lines(duct: DuctObject, prop: str, elsewhere: object) -> list[dict]:
    """The sheet lines relevant to one duct and one property.

    `elsewhere` is the value lettered against something other than this duct.
    For a back-filled value the caller passes the same value, so the sheet
    carries the right number in the wrong place.
    """
    n = number(duct.id)
    sheet = duct.sheet
    own_tag_id = f"{sheet}-tag-{duct.id}"
    observed = duct.properties.get(prop)
    out: list[dict] = []

    if observed is not None and observed.value is not None and observed.is_independent:
        token = value_token(prop, observed.value, duct.element)
        if observed.origin == "DRAWING_SCHEDULE":
            out.append({"id": own_tag_id, "text": _tag(duct, duct.id)})
            out.append({
                "id": observed.source_id,
                "text": f"{duct.id} | {duct.width_in}x{duct.height_in} | {token}",
            })
        else:
            style = STYLES[n % len(STYLES)]
            if style == "tag":
                out.append({"id": observed.source_id, "text": _tag(duct, duct.id, token)})
            else:
                out.append({"id": own_tag_id, "text": _tag(duct, duct.id)})
                if style == "range":
                    text = f"D-{n - 2} THRU D-{n + 3}: {token}"
                else:
                    text = f"ALL {SYSTEM_NAME[duct.system]} DUCTS: {token} U.N.O."
                out.append({"id": observed.source_id, "text": text})
    else:
        out.append({"id": own_tag_id, "text": _tag(duct, duct.id)})

    distractor = DISTRACTORS[(n // len(STYLES)) % len(DISTRACTORS)]
    if distractor != "none" and elsewhere is not None:
        token = value_token(prop, elsewhere, duct.element)
        note_id = f"{sheet}-note{n % 9 + 11}"
        if distractor == "other_duct":
            other = f"D-{n + 7}"
            line = {"id": f"{sheet}-tag-{other}", "text": _tag(duct, other, token)}
        elif distractor == "other_range":
            line = {"id": note_id, "text": f"D-{n + 10} THRU D-{n + 15}: {token}"}
        else:
            name = SYSTEM_NAME[_other_system(duct.system)]
            line = {"id": note_id, "text": f"ALL {name} DUCTS: {token} U.N.O."}
        out.insert(0 if n % 2 else len(out), line)

    ids = [item["id"] for item in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate drawing line ids for {duct.id}: {ids}")
    return out


_TAG = re.compile(r"^(D-\d+): \d+x\d+ (.+)$")
_RANGE = re.compile(r"^D-(\d+) THRU D-(\d+): .+$")
# A general note may carry a further remark after it, two spaces on.
_GENERAL = re.compile(r"^ALL (.+?) DUCTS: .+? U\.N\.O\.(?:  .*)?$")
_SCHEDULE = re.compile(r"^(D-\d+) \| \d+x\d+ \| .+$")


def shown_for(drawing: list[dict], duct_id: str, system: str) -> list[str]:
    """Ids of the lines that letter a value against this duct."""
    n = number(duct_id)
    hits = []
    for item in drawing:
        text = item["text"]
        if m := _SCHEDULE.match(text):
            applies = m.group(1) == duct_id
        elif m := _TAG.match(text):
            # A bare tag ends in the system abbreviation and letters no value.
            applies = m.group(1) == duct_id and m.group(2) not in SYSTEM_ABBREV.values()
        elif m := _RANGE.match(text):
            applies = int(m.group(1)) <= n <= int(m.group(2))
        elif m := _GENERAL.match(text):
            applies = m.group(1) == SYSTEM_NAME[system]
        else:
            raise ValueError(f"unreadable drawing line {text!r}")
        if applies:
            hits.append(item["id"])
    return hits
