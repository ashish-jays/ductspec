from pathlib import Path

import pytest

from ductspec.clauses import parse_clauses, strip_designer_notes
from ductspec.schema import SchemaError, validate_record

SOURCES = Path(__file__).resolve().parents[1] / "data" / "sources"


def test_designer_notes_are_stripped():
    text = (
        "**********\n"
        "NOTE:  Choose one of the following for the project.\n"
        "**********\n"
        "Supply ducts must be galvanized steel.\n"
    )
    assert "NOTE:" not in strip_designer_notes(text)
    assert "galvanized steel" in strip_designer_notes(text)


@pytest.mark.skipif(not (SOURCES / "UFGS-23-30-00.pdf").exists(), reason="run ductspec.sources first")
def test_parses_real_ufgs_with_provenance():
    clauses = parse_clauses(SOURCES / "UFGS-23-30-00.pdf", "UFGS-23-30-00")
    assert len(clauses) > 100
    assert {c.part for c in clauses} == {
        "PART 1 GENERAL",
        "PART 2 PRODUCTS",
        "PART 3 EXECUTION",
    }
    # Every clause must be locatable in the PDF, or its citation is unverifiable.
    assert all(c.page_start > 0 for c in clauses)
    # The body starts well past the table of contents.
    assert max(c.page_start for c in clauses) > 10
    # A guide spec always ships some unresolved choices.
    assert any(not c.operative for c in clauses)
    assert all(c.non_operative_reasons for c in clauses if not c.operative)


def _record(**overrides):
    rec = {
        "id": "T-1",
        "input": {
            "question": "Does D-101's shown material meet the supply-duct requirement?",
            "object": {
                "id": "D-101",
                "properties": {
                    "material": {
                        "value": "galvanized_steel",
                        "origin": "DRAWING_ANNOTATION",
                        "source_id": "M101-callout7",
                    }
                },
            },
            "evidence": [{"id": "C-1", "text": "Supply-air ducts shall be galvanized steel."}],
            "policy": {"id": "P-1", "text": "Compare required material with the shown material."},
        },
        "target": {
            "verdict": "COMPLIANT",
            "scope": "material",
            "governing_ids": ["C-1"],
            "missing_evidence": [],
            "action": "NONE_FOR_THIS_CHECK",
        },
    }
    rec["target"].update(overrides)
    return rec


def test_valid_record_passes():
    assert validate_record(_record())["id"] == "T-1"


def test_backfilled_value_cannot_support_a_definitive_verdict():
    """The central claim of the project, enforced at the schema boundary."""
    rec = _record()
    rec["input"]["object"]["properties"]["material"]["origin"] = "COPIED_FROM_SPEC"
    with pytest.raises(SchemaError, match="not independent evidence"):
        validate_record(rec)


def test_abstention_requires_a_reason_and_an_action():
    with pytest.raises(SchemaError, match="requires at least one missing_evidence"):
        validate_record(_record(verdict="INSUFFICIENT_EVIDENCE", action="REQUEST_MISSING_DETAIL"))


def test_citation_must_resolve_to_supplied_evidence():
    with pytest.raises(SchemaError, match="not in the supplied evidence"):
        validate_record(_record(governing_ids=["C-99"]))
