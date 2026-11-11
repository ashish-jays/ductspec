"""The curation gates, and what they catch.

These are not hypothetical faults. Every one of them was in the requirement
file at some point today, and two were put there by bugs in our own parser.
"""

from pathlib import Path

import pytest

from ductspec.curate import CURATED_GATES, RAW_GATES, arms, curate
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
    assert len(out["curated"]["requirements"]) == 9
    assert len(out["audited"]["requirements"]) == 10

    curated_ids = {r.id for r in out["curated"]["requirements"]}
    audited_ids = {r.id for r in out["audited"]["requirements"]}
    assert "R-VA-2.3.B-liner" in audited_ids      # repaired, with its condition
    assert not {i for i in curated_ids if i.startswith("R-VA-2.")} - {"R-VA-2.1.A-material"}
