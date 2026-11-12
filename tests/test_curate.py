"""The curation gates, and what they catch.

These are not hypothetical faults. Every one of them was in the requirement
file at some point today, and two were put there by bugs in our own parser.
"""

from pathlib import Path

import pytest

from ductspec import curate as curate_module
from ductspec.curate import (
    CURATED_GATES, RAW_GATES, ClauseExistsGate, Corpus, CorpusMissing, InheritedScopeGate,
    OperativeClauseGate, QuoteFidelityGate, arms, curate, to_samples,
)
from ductspec.requirements import load

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "requirements.raw.yaml"
AUDITED = ROOT / "data" / "requirements.yaml"
HAVE_CORPUS = (ROOT / "data" / "sources" / "UFGS-23-30-00.pdf").exists()

pytestmark = pytest.mark.skipif(not HAVE_CORPUS, reason="run ductspec.sources first")


def test_the_raw_arm_keeps_everything():
    """It is the unaudited extraction, not a degraded copy."""
    kept, rejected = curate(load(RAW), RAW_GATES)
    assert len(kept) == len(load(RAW))
    assert rejected == []


def test_a_citation_to_the_wrong_clause_is_caught_by_its_quote():
    """The VA liner rule was recorded against clause 2.2.B. That id exists —
    it is the pressure-class paragraph — so checking the id alone passes it.
    The quote is what gives it away."""
    _, rejected = curate(load(RAW), CURATED_GATES)
    by_id = {r.metadata["requirement_id"]: r for r in rejected}

    bad = by_id["R-VA-2.2.B-liner"]
    assert bad.rejecting_step == "quote_fidelity"
    assert "quote_not_in_clause" in bad.rejection_reason


def test_a_requirement_resting_on_an_unmade_choice_is_caught():
    _, rejected = curate(load(RAW), CURATED_GATES)
    by_id = {r.metadata["requirement_id"]: r for r in rejected}

    bad = by_id["R-UFGS-3.2.7.1-material"]
    assert bad.rejecting_step == "clause_operative"
    assert "UNRESOLVED_ALTERNATIVES" in bad.rejection_reason


def test_adoption_is_a_property_of_the_sentence_not_the_article():
    """UFGS 3.2.11 opens with an unconditional cross-reference and later offers
    the engineer a choice about outdoor air intakes. Rejecting the article
    would discard a sentence that binds because a later one does not."""
    kept, _ = curate(load(RAW), CURATED_GATES)
    assert any(r.id == "R-UFGS-3.2.11-reference" for r in kept)


def test_nothing_is_dropped_without_a_reason():
    """CuratorKIT's contract: silent drops are bugs."""
    kept, rejected = curate(load(RAW), CURATED_GATES)
    assert len(kept) + len(rejected) == len(load(RAW))
    for r in rejected:
        assert r.rejection_reason and r.rejecting_step


def test_the_audited_set_passes_its_own_gates():
    """Whatever the gates reject, the audited file should already be clean of."""
    kept, rejected = curate(load(AUDITED), CURATED_GATES)
    assert rejected == [], [r.rejection_reason for r in rejected]
    assert len(kept) == len(load(AUDITED))


def test_gates_and_audit_disagree_and_that_is_the_finding():
    """The gates drop a faulty requirement. The audit repaired it instead.

    The VA liner rule is real and binding — it was recorded against the wrong
    clause with its condition missing. An automated gate can only remove it;
    a reader with the PDF open fixed the citation and recovered the condition.
    """
    out = arms(RAW, AUDITED)
    assert len(out["raw"]["requirements"]) == 11
    assert len(out["curated"]["requirements"]) == 7
    assert len(out["audited"]["requirements"]) == 10

    curated_ids = {r.id for r in out["curated"]["requirements"]}
    audited_ids = {r.id for r in out["audited"]["requirements"]}
    assert "R-VA-2.3.B-liner" in audited_ids      # repaired, with its condition
    assert not {i for i in curated_ids if i.startswith("R-VA-2.")} - {"R-VA-2.1.A-material"}


# --- scope, and the headings a clause sits under -----------------------------

def _rejections():
    _, rejected = curate(load(RAW), CURATED_GATES)
    return {r.metadata["requirement_id"]: r for r in rejected}


def test_the_four_faults_in_the_raw_extraction_are_each_caught_for_their_own_reason():
    steps = {rid: r.rejecting_step for rid, r in _rejections().items()}
    assert steps == {
        "R-VA-2.2.B-liner": "quote_fidelity",
        "R-UFGS-3.2.7.1-material": "clause_operative",
        "R-UFGS-3.2.7.2-material": "inherited_scope",
        "R-UFGS-2.12.10.1-material": "inherited_scope",
    }


def test_a_rule_under_a_kitchen_exhaust_heading_cannot_be_recorded_for_every_system():
    """3.2.7 "Kitchen Exhaust Ductwork" has no body of its own, so the parser
    does not keep it and only the table of contents names it. The gate used to
    look for it among the parsed clauses, find nothing, and pass."""
    bad = _rejections()["R-UFGS-3.2.7.2-material"]
    assert "3.2.7_'Kitchen Exhaust Ductwork'" in bad.rejection_reason
    assert bad.rejection_reason.endswith("needs_system=kitchen_exhaust")


def test_a_casing_gauge_cannot_be_recorded_as_a_duct_material():
    bad = _rejections()["R-UFGS-2.12.10.1-material"]
    assert bad.rejection_reason.endswith("element=duct")
    assert "Plenum" in bad.rejection_reason


def test_the_same_two_clauses_pass_once_their_scope_is_right():
    audited = {r.id: r for r in load(AUDITED)}
    fixed = [audited["R-UFGS-3.2.7.2-material"], audited["R-UFGS-2.12.10.1-material"]]
    passed, rejected = InheritedScopeGate().run(to_samples(fixed))
    assert len(passed) == 2 and rejected == []


def test_a_clause_is_read_under_every_heading_above_it():
    lineage = curate_module.CORPUS.lineage("UFGS-23-30-00", "3.2.7.2")
    assert lineage == [
        ("3.2.7.2", "Exposed Ductwork"),
        ("3.2.7", "Kitchen Exhaust Ductwork"),
        ("3.2", "INSTALLATION"),
    ]


def test_a_heading_that_cannot_be_found_is_a_rejection_not_a_pass(monkeypatch):
    real = curate_module.CORPUS.headings("UFGS-23-30-00")
    without = {k: v for k, v in real.items() if k != "3.2.7"}
    monkeypatch.setattr(curate_module.CORPUS, "headings", lambda source: without)
    audited = {r.id: r for r in load(AUDITED)}
    passed, rejected = InheritedScopeGate().run(to_samples([audited["R-UFGS-3.2.7.2-material"]]))
    assert passed == []
    assert rejected[0].rejection_reason == "heading_above_not_found:UFGS-23-30-00#3.2.7.2"


@pytest.mark.parametrize("gate", [
    ClauseExistsGate, QuoteFidelityGate, OperativeClauseGate, InheritedScopeGate])
def test_no_gate_passes_a_clause_it_cannot_find(gate):
    """Each gate is safe run alone, not only behind the one that checks ids."""
    raw = {r.id: r for r in load(RAW)}
    sample = to_samples([raw["R-UFGS-3.2.7.2-material"]])[0]
    sample.metadata["clause_id"] = "9.9.9"
    passed, rejected = gate().run([sample])
    assert passed == []
    assert rejected[0].rejection_reason == "clause_not_found:UFGS-23-30-00#9.9.9"


def test_curating_without_the_documents_fails_instead_of_passing_everything(monkeypatch, tmp_path):
    """With no PDF on disk the gates used to pass every requirement, and the
    curated arm came out identical to the raw one with nothing to say so."""
    monkeypatch.setattr(curate_module, "SOURCES", tmp_path)
    monkeypatch.setattr(curate_module, "CORPUS", Corpus())
    with pytest.raises(CorpusMissing, match="python -m ductspec.sources"):
        curate(load(RAW), CURATED_GATES)


def test_a_document_that_parses_to_nothing_fails_too(monkeypatch):
    monkeypatch.setattr(curate_module, "CORPUS", Corpus())
    monkeypatch.setattr(curate_module, "parse_clauses", lambda *a, **k: [])
    with pytest.raises(CorpusMissing, match="parsed to no clauses"):
        curate(load(RAW), CURATED_GATES)
