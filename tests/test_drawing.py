"""The model is shown the sheet, not the answer key.

Three things are held here. The prompt never carries a value's origin. The
lettering on the sheet agrees with the origin it stands in for, on every row.
And a silent drawing leads to the specification, not to an RFI.
"""

import pytest

from ductspec.derive import SILENT_DRAWING_ACTION, Packet, derive
from ductspec.drawing import shown_for, value_token
from ductspec.generate import VALUES, build
from ductspec.objects import DuctObject, copied_from_spec, observed
from ductspec.prompt import render, view
from ductspec.requirements import Requirement
from ductspec.schema import ACTIONS, INDEPENDENT_ORIGINS, ORIGINS


@pytest.fixture(scope="module")
def rows():
    return [r for split in build().values() for r in split]


def _carried(row):
    return row["input"]["object"]["properties"].get(row["input"]["property"]) or {}


def _is_shown(row):
    prop = _carried(row)
    return prop.get("origin") in INDEPENDENT_ORIGINS and prop.get("value") is not None


def test_the_prompt_never_states_where_a_value_came_from(rows):
    for row in rows:
        prompt = render(row["input"])
        packet = prompt.split("PACKET:\n", 1)[1]
        assert '"origin"' not in packet, row["id"]
        assert '"properties"' not in packet, row["id"]
        for origin in ORIGINS:
            assert origin not in packet, f"{row['id']} leaks {origin}"


def test_the_view_carries_nothing_that_separates_a_shown_value_from_a_filled_one(rows):
    """Outside the drawing lines, a shown value and a back-filled one look alike.

    If any other field differed between them, the model could read the answer
    off that field and ignore the sheet.
    """
    def outside_drawing(row):
        v = view(row["input"])
        return (
            v["takeoff"]["property"], repr(v["takeoff"]["value"]), v["duct"]["system"],
            v["duct"]["element"], tuple(sorted(v["duct"].get("facts", {}).items())),
            tuple(e["id"] for e in v["evidence"]), v["policy"]["text"],
        )

    shown = {outside_drawing(r) for r in rows if _is_shown(r)}
    filled = {
        outside_drawing(r) for r in rows
        if _carried(r).get("value") is not None and not _is_shown(r)
    }
    assert filled, "the data must contain back-filled values"
    assert filled <= shown, "every back-filled situation has a shown twin that looks the same"


def test_the_lettering_agrees_with_the_origin_on_every_row(rows):
    for row in rows:
        inp = row["input"]
        hits = shown_for(inp["drawing"], inp["object"]["id"], inp["object"]["system"])
        if _is_shown(row):
            assert hits == [_carried(row)["source_id"]], row["id"]
        else:
            assert hits == [], f"{row['id']}: the sheet shows a value the object says it lacks"


def test_a_definitive_verdict_cites_a_line_that_is_on_the_sheet(rows):
    for row in rows:
        if row["target"]["verdict"] in ("COMPLIANT", "NON_COMPLIANT"):
            on_sheet = {item["id"]: item["text"] for item in row["input"]["drawing"]}
            (cited,) = row["target"]["observation_ids"]
            token = value_token(
                row["input"]["property"], _carried(row)["value"], row["input"]["object"]["element"])
            assert token in on_sheet[cited], row["id"]


def test_finding_the_value_anywhere_on_the_sheet_is_not_enough(rows):
    """A back-filled value is lettered against something else often enough that
    matching the number without matching the duct gets it wrong."""
    fooled = 0
    filled = 0
    for row in rows:
        prop = _carried(row)
        if prop.get("value") is None or _is_shown(row):
            continue
        filled += 1
        token = value_token(row["input"]["property"], prop["value"], row["input"]["object"]["element"])
        if any(token in item["text"] for item in row["input"]["drawing"]):
            fooled += 1
    assert filled and fooled / filled > 0.5, (fooled, filled)


def test_a_shown_value_is_lettered_in_more_than_one_way(rows):
    kinds = set()
    for row in rows:
        if not _is_shown(row):
            continue
        text = next(i["text"] for i in row["input"]["drawing"] if i["id"] == _carried(row)["source_id"])
        kinds.add("range" if " THRU " in text else "general" if text.startswith("ALL ")
                  else "schedule" if " | " in text else "tag")
    assert kinds == {"tag", "range", "general", "schedule"}


# --- a silent drawing is not a discrepancy ----------------------------------

REQ = Requirement(
    id="R-1", source="UFGS-23-30-00", clause_id="C-1", page=1, property="material",
    predicate={"op": "in", "values": ["galvanized_steel"]}, quote="Galvanized.",
)
PACKET = Packet(requirements=(REQ,), sources=("UFGS-23-30-00",), policy_id="P-1")


def test_a_silent_drawing_follows_the_specification():
    silent = DuctObject("D-1", "supply_air", 20, 12, {})
    out = derive(silent, "material", PACKET)
    assert out.missing_evidence == ("MISSING_OBSERVATION",)
    assert out.action == "FOLLOW_SPEC_RECORD_AS_SPEC_DERIVED"
    assert out.governing_ids == ("C-1",)


def test_a_back_filled_value_follows_the_specification():
    filled = DuctObject("D-1", "supply_air", 20, 12,
                        {"material": copied_from_spec("galvanized_steel", "C-1")})
    out = derive(filled, "material", PACKET)
    assert out.missing_evidence == ("INFERRED_VALUE_ONLY",)
    assert out.action == "FOLLOW_SPEC_RECORD_AS_SPEC_DERIVED"


def test_a_real_difference_still_goes_to_an_rfi():
    wrong = DuctObject("D-1", "supply_air", 20, 12,
                       {"material": observed("aluminum", "D-1")})
    out = derive(wrong, "material", PACKET)
    assert out.verdict == "NON_COMPLIANT"
    assert out.action == "RFI_DRAWING_SPEC_DISCREPANCY"


def test_the_silent_drawing_action_is_one_the_schema_accepts():
    assert SILENT_DRAWING_ACTION in ACTIONS


def test_every_value_has_lettering():
    for prop, values in VALUES.items():
        for value in values:
            assert value_token(prop, value, "duct")
