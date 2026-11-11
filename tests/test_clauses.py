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


# --- counterfactual pairs ----------------------------------------------------

from ductspec.cases import (  # noqa: E402
    add_inapplicable_authority,
    build_record,
    inject_instruction,
    remove_observation,
    remove_precedence,
    strip_provenance,
)
from ductspec.objects import DuctObject, observed  # noqa: E402

CLAUSE = {"id": "C-1", "text": "Supply-air ducts shall be galvanized steel."}
POLICY = {"id": "P-1", "text": "Compare the required material with the shown material."}


def _base():
    duct = DuctObject(
        "D-101", "supply_air", 20, 12,
        {"material": observed("galvanized_steel", "D-101", text="D-101: 20x12 GALV SA")},
    )
    return build_record(
        "P1", "Does D-101's shown material meet the supply-duct requirement?",
        duct, [CLAUSE], POLICY, scope="material", verdict="COMPLIANT",
        governing_ids=["C-1"], action="NONE_FOR_THIS_CHECK",
    )


def test_base_record_cites_the_drawing_observation():
    rec = _base()
    assert rec["target"]["observation_ids"] == ["M101-note1"]


def test_same_number_relabelled_as_inferred_flips_to_abstention():
    base = _base()
    twin = strip_provenance(base, "C-1")
    scope = base["target"]["scope"]
    # The value is untouched; only its provenance moved.
    assert (
        twin["input"]["object"]["properties"][scope]["value"]
        == base["input"]["object"]["properties"][scope]["value"]
    )
    assert twin["target"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert twin["target"]["missing_evidence"] == ["INFERRED_VALUE_ONLY"]
    assert twin["target"]["observation_ids"] == []


def test_silent_drawing_flips_to_abstention():
    twin = remove_observation(_base())
    assert twin["target"]["missing_evidence"] == ["MISSING_OBSERVATION"]
    assert twin["target"]["action"] == "REQUEST_MISSING_DETAIL"


def test_conflict_without_precedence_escalates_as_an_rfi():
    twin = remove_precedence(_base())
    assert twin["target"]["action"] == "RFI_DRAWING_SPEC_DISCREPANCY"


def test_inapplicable_authority_must_not_move_the_answer():
    base = _base()
    twin = add_inapplicable_authority(
        base, {"id": "C-9", "text": "Kitchen exhaust ducts shall be black iron."}
    )
    assert twin["target"]["verdict"] == base["target"]["verdict"]
    assert twin["target"]["governing_ids"] == base["target"]["governing_ids"]
    assert len(twin["input"]["evidence"]) == len(base["input"]["evidence"]) + 1


def test_instruction_in_drawing_text_must_not_move_the_answer():
    base = _base()
    twin = inject_instruction(base)
    assert "ignore specification requirements" in twin["input"]["object"]["properties"]["material"]["source_text"]
    assert twin["target"]["verdict"] == base["target"]["verdict"]


@pytest.mark.skipif(not (SOURCES / "UFGS-23-30-00.pdf").exists(), reason="run ductspec.sources first")
def test_side_by_side_alternatives_are_not_citable():
    """UFGS 23 30 00 clause 3.2.7.1 offers two materials for grease-laden duct:
    18 ga 304L/316L stainless, or 16 ga carbon steel. Until the designer picks
    one there is no requirement, so it must not be usable as governing evidence.
    A 60-character cap on the alternatives pattern let this one through."""
    clauses = parse_clauses(SOURCES / "UFGS-23-30-00.pdf", "UFGS-23-30-00")
    grease = next(c for c in clauses if c.clause_id == "3.2.7.1")
    assert not grease.operative
    assert "UNRESOLVED_ALTERNATIVES" in grease.non_operative_reasons

    # Its neighbour states one material outright, so it stays citable.
    exposed = next(c for c in clauses if c.clause_id == "3.2.7.2")
    assert exposed.operative and not exposed.has_optional_text


# --- the cross-authority holdout -------------------------------------------

from ductspec.conventions import VA  # noqa: E402


@pytest.mark.skipif(not (SOURCES / "VA-23-31-00.pdf").exists(), reason="run ductspec.sources first")
def test_va_corpus_parses_with_its_own_conventions():
    clauses = parse_clauses(SOURCES / "VA-23-31-00.pdf", "VA-23-31-00", VA)
    by_id = {c.clause_id: c for c in clauses}

    # VA numbers clauses 2.1 with lettered subparagraphs, so a citation can
    # name 2.1.A rather than the whole article.
    assert "2.1.A" in by_id
    assert "galvanized sheet steel" in by_id["2.1.A"].text

    # Spec writer notes are indented rather than rule-delimited, and must not
    # survive into any clause body.
    assert not any("SPEC WRITER" in c.text.upper() for c in clauses)


@pytest.mark.skipif(not (SOURCES / "VA-23-31-00.pdf").exists(), reason="run ductspec.sources first")
def test_unmade_choices_are_uncitable_in_either_notation():
    """UFGS writes an unmade choice as "[A] [B]", VA as "//A// //B//".

    This is the point of the holdout: the model has to learn that a choice
    nobody has made is not a requirement, not that square brackets are bad.
    """
    va = parse_clauses(SOURCES / "VA-23-31-00.pdf", "VA-23-31-00", VA)
    unresolved = [c for c in va if "UNRESOLVED_ALTERNATIVES" in c.non_operative_reasons]
    assert unresolved, "VA 23 31 00 offers the engineer several unmade choices"
    # e.g. duct pressure class is offered as 2, 3 or 4 inch WG until selected.
    assert any("//" in c.text for c in unresolved)


@pytest.mark.skipif(not (SOURCES / "UFGS-23-07-00.pdf").exists(), reason="run ductspec.sources first")
def test_table_figures_do_not_become_clauses():
    """Insulation thickness tables are full of numbers that look like clause
    ids — 0.235, 401.5, 61.552 — and a run of them was producing clauses
    numbered after gauge values, some colliding with real clause ids.

    A citation to a clause that does not exist is the worst thing this dataset
    could contain, so this is a regression test rather than a style check.
    """
    clauses = parse_clauses(SOURCES / "UFGS-23-07-00.pdf", "UFGS-23-07-00")
    ids = [c.clause_id for c in clauses]

    assert len(ids) == len(set(ids)), "clause ids must be unique"
    for cid in ids:
        head = int(cid.split(".")[0])
        assert head in (1, 2, 3), f"{cid} is outside the document's PART numbering"

    # Every heading must be one the document lists in its own contents.
    from ductspec.clauses import toc_entries
    from pypdf import PdfReader
    text = "\n".join(pg.extract_text() or "" for pg in PdfReader(SOURCES / "UFGS-23-07-00.pdf").pages)
    listed = toc_entries(text)
    assert listed, "this section publishes a table of contents"
    assert set(ids) <= set(listed), f"not in contents: {set(ids) - set(listed)}"

    # Clauses that sit AFTER a table must survive. An earlier filter used the
    # shape of the numbering instead, and a thickness value of 3.5 pushed the
    # counter past the genuine 3.3 and 3.4, silently dropping them.
    assert {"2.4.5", "3.3", "3.4"} <= set(ids)


@pytest.mark.skipif(not (SOURCES / "UFGS-23-30-00.pdf").exists(), reason="run ductspec.sources first")
def test_long_headings_do_not_leak_into_the_clause_body():
    """A heading that wraps leaves its tail at the top of the body, so a quote
    taken from that clause carries a fragment of its own title. UFGS 2.12.11.1
    came out titled "Systems with total pressure " with the rest in the text.
    """
    from ductspec.clauses import _normalise, toc_entries
    from pypdf import PdfReader

    pdf = SOURCES / "UFGS-23-30-00.pdf"
    text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
    listed = toc_entries(text)
    clauses = parse_clauses(pdf, "UFGS-23-30-00")

    for clause in clauses:
        full = listed.get(clause.clause_id)
        if full:
            assert len(_normalise(clause.title)) >= len(_normalise(full)) - 2, clause.clause_id
