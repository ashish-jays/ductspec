"""Every label in the dataset comes out of derive(). These are its rules."""

import pytest

from ductspec.derive import Packet, derive
from ductspec.objects import DuctObject, copied_from_spec, observed
from ductspec.requirements import Requirement, load

ALL = {r.id: r for r in load()}
KITCHEN_INSULATION = ALL["R-UFGS-2.4.5-insulation"]      # kitchen exhaust, >= 2 inch
EXPOSED_MATERIAL = ALL["R-UFGS-3.2.7.2-material"]        # kitchen exhaust, 304/316 SS
POINTER = ALL["R-UFGS-3.2.11-reference"]                 # defers to 23 07 00
VA_LINER = ALL["R-VA-2.3.B-liner"]                       # only where lining indicated


def packet(*reqs: Requirement, sources=()) -> Packet:
    return Packet(requirements=tuple(reqs), sources=tuple(sources), policy_id="P-1")


def kitchen_duct(**props) -> DuctObject:
    return DuctObject("D-401", "kitchen_exhaust", 24, 14, props)


# --- 4. comparison ----------------------------------------------------------

def test_meeting_the_requirement_is_compliant_and_cites_the_clause():
    duct = kitchen_duct(insulation_thickness_in=observed(3.0, "D-401", note=4))
    out = derive(duct, "insulation_thickness_in", packet(KITCHEN_INSULATION))
    assert out.verdict == "COMPLIANT"
    assert out.governing_ids == ("2.4.5",)
    assert out.observation_ids == ("M101-note4",)


def test_falling_short_is_non_compliant_and_raises_an_rfi():
    duct = kitchen_duct(insulation_thickness_in=observed(1.0, "D-401", note=4))
    out = derive(duct, "insulation_thickness_in", packet(KITCHEN_INSULATION))
    assert out.verdict == "NON_COMPLIANT"
    assert out.action == "RFI_DRAWING_SPEC_DISCREPANCY"


def test_boundary_value_counts_as_met():
    duct = kitchen_duct(insulation_thickness_in=observed(2.0, "D-401", note=4))
    assert derive(duct, "insulation_thickness_in", packet(KITCHEN_INSULATION)).verdict == "COMPLIANT"


# --- 3. evidence ------------------------------------------------------------

def test_the_same_number_back_filled_from_the_spec_stops_the_verdict():
    """The project in one test: identical value, different provenance."""
    shown = kitchen_duct(insulation_thickness_in=observed(3.0, "D-401", note=4))
    filled = kitchen_duct(insulation_thickness_in=copied_from_spec(3.0, "2.4.5"))

    assert derive(shown, "insulation_thickness_in", packet(KITCHEN_INSULATION)).verdict == "COMPLIANT"

    out = derive(filled, "insulation_thickness_in", packet(KITCHEN_INSULATION))
    assert out.verdict == "INSUFFICIENT_EVIDENCE"
    assert out.missing_evidence == ("INFERRED_VALUE_ONLY",)
    assert out.observation_ids == ()
    # It still knows which clause governs — it just cannot certify against it.
    assert out.governing_ids == ("2.4.5",)


def test_a_silent_drawing_is_not_a_failure():
    out = derive(kitchen_duct(), "insulation_thickness_in", packet(KITCHEN_INSULATION))
    assert out.missing_evidence == ("MISSING_OBSERVATION",)
    assert out.verdict != "NON_COMPLIANT"


# --- 2. applicability -------------------------------------------------------

def test_a_requirement_for_another_system_does_not_govern():
    """3.2.7.2 reaches kitchen exhaust. It says nothing about a supply duct."""
    supply = DuctObject(
        "D-101", "supply_air", 20, 12,
        {"material": observed("galvanized_steel", "D-101", note=7)},
        element="exposed_ductwork",
    )
    out = derive(supply, "material", packet(EXPOSED_MATERIAL))
    assert out.verdict == "INSUFFICIENT_EVIDENCE"
    assert out.missing_evidence == ("MISSING_AUTHORITY",)


def test_a_requirement_for_another_element_does_not_govern():
    """A rule about plenums and casings is not a rule about a duct."""
    duct = kitchen_duct(material=observed("galvanized_steel", "D-401", note=7))
    out = derive(duct, "material", packet(ALL["R-UFGS-2.12.10.1-material"]))
    assert out.missing_evidence == ("MISSING_AUTHORITY",)


def test_an_unestablished_condition_is_unknown_not_false():
    """VA 2.3.B binds only where the drawings indicate lining."""
    liner = DuctObject("D-501", "supply_air", 20, 12,
                       {"insulation_thickness_in": observed(0.5, "D-501", note=2)},
                       element="acoustical_liner")
    out = derive(liner, "insulation_thickness_in", packet(VA_LINER))
    assert out.missing_evidence == ("APPLICABILITY_UNKNOWN",)
    # Once the drawing establishes it, the comparison runs and this one fails.
    liner.facts["lining_indicated_on_drawings"] = True
    assert derive(liner, "insulation_thickness_in", packet(VA_LINER)).verdict == "NON_COMPLIANT"


# --- 1. authority -----------------------------------------------------------

def test_a_pointer_to_a_document_nobody_supplied_is_missing_authority():
    duct = DuctObject("D-601", "supply_air", 20, 12,
                      {"insulation_thickness_in": observed(2.0, "D-601", note=9)})
    out = derive(duct, "insulation_thickness_in", packet(POINTER, sources=["UFGS-23-30-00"]))
    assert out.missing_evidence == ("MISSING_AUTHORITY",)
    assert "23-07-00" in out.reason


def test_an_unresolved_pointer_beside_a_real_requirement_is_not_a_dead_end():
    """A dangling cross-reference does not block a question another supplied
    clause already settles."""
    duct = kitchen_duct(insulation_thickness_in=observed(3.0, "D-401", note=4))
    out = derive(duct, "insulation_thickness_in",
                 packet(POINTER, KITCHEN_INSULATION, sources=["UFGS-23-30-00"]))
    assert out.verdict == "COMPLIANT"
    assert out.governing_ids == ("2.4.5",)


def test_an_empty_packet_cannot_produce_a_verdict():
    duct = kitchen_duct(insulation_thickness_in=observed(3.0, "D-401", note=4))
    assert derive(duct, "insulation_thickness_in", packet()).verdict == "INSUFFICIENT_EVIDENCE"


# --- several requirements at once -------------------------------------------

def test_where_two_requirements_apply_both_must_be_met():
    """A 1-inch liner minimum and a 2-inch kitchen minimum are compatible
    obligations, so 1.5 inches satisfies one and fails the other. The failure
    is cited to the clause that was not met, not to both."""
    liner = DuctObject("D-701", "kitchen_exhaust", 24, 14,
                       {"insulation_thickness_in": observed(1.5, "D-701", note=3)},
                       element="acoustical_liner")
    kitchen_liner = Requirement(
        id="R-TEST", source="UFGS-23-07-00", clause_id="2.4.5", page=24,
        property="insulation_thickness_in", predicate={"op": "at_least", "value": 2.0},
        quote="", systems=("kitchen_exhaust",), element="acoustical_liner",
    )
    out = derive(liner, "insulation_thickness_in",
                 packet(ALL["R-UFGS-2.12.11.3-liner"], kitchen_liner))
    assert out.verdict == "NON_COMPLIANT"
    # Both clauses still govern — failing one does not make the other stop
    # applying — but only the unmet one is recorded as failed.
    assert set(out.governing_ids) == {"2.12.11.3", "2.4.5"}
    assert out.failed_ids == ("2.4.5",)


# --- evaluation-only metadata ----------------------------------------------

def test_answerability_is_recorded_so_a_lucky_guess_cannot_look_supported():
    answerable = kitchen_duct(insulation_thickness_in=observed(3.0, "D-401", note=4))
    silent = kitchen_duct()

    assert derive(answerable, "insulation_thickness_in", packet(KITCHEN_INSULATION)).answer_available
    assert not derive(silent, "insulation_thickness_in", packet(KITCHEN_INSULATION)).answer_available


def test_the_reason_names_the_clauses_it_rests_on():
    duct = kitchen_duct(insulation_thickness_in=observed(1.0, "D-401", note=4))
    assert "clause 2.4.5" in derive(duct, "insulation_thickness_in", packet(KITCHEN_INSULATION)).reason
