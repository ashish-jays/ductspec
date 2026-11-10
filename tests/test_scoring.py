"""The scorer's job is to make the two degenerate strategies fail."""

from ductspec.cases import build_record, strip_provenance
from ductspec.objects import DuctObject, observed
from ductspec.scoring import score_one, score_pairs, score_run

CLAUSE = {"id": "C-1", "text": "Supply-air ducts shall be galvanized steel."}
POLICY = {"id": "P-1", "text": "Compare the required material with the shown material."}


def _compliant(rid="P1"):
    duct = DuctObject(
        "D-101", "supply_air", 20, 12,
        {"material": observed("galvanized_steel", "D-101", text="D-101: 20x12 GALV SA")},
    )
    return build_record(
        rid, "q", duct, [CLAUSE], POLICY, scope="material", verdict="COMPLIANT",
        governing_ids=["C-1"], action="NONE_FOR_THIS_CHECK",
    )


def _pred(rec):
    """A perfect prediction, derived from the gold target."""
    return {k: rec["target"][k] for k in ("verdict", "governing_ids", "missing_evidence", "action")}


def test_perfect_predictions_score_one():
    golds = [_compliant("A"), strip_provenance(_compliant("B"), "C-1")]
    out = score_run(golds, [_pred(g) for g in golds])
    assert out["strict_accuracy"] == 1.0
    assert out["unsupported_rate"] == 0.0
    assert out["provenance_admission_error"] == 0.0


def test_always_compliant_is_caught_as_unsupported_not_as_accuracy():
    base = _compliant("A")
    twin = strip_provenance(base, "C-1")
    always = {"verdict": "COMPLIANT", "governing_ids": ["C-1"], "missing_evidence": [], "action": "NONE_FOR_THIS_CHECK"}
    out = score_run([base, twin], [always, always])

    assert out["verdict_accuracy"] == 0.5          # looks half right
    assert out["strict_accuracy"] == 0.5
    assert out["provenance_admission_error"] == 1.0  # every back-filled case certified
    assert out["unsupported_rate"] == 0.5
    assert out["false_approval_rate"] == 1.0


def test_always_abstain_is_caught_by_coverage_and_over_refusal():
    base = _compliant("A")
    twin = strip_provenance(base, "C-1")
    always = {
        "verdict": "INSUFFICIENT_EVIDENCE", "governing_ids": [],
        "missing_evidence": ["INFERRED_VALUE_ONLY"], "action": "REQUEST_MISSING_DETAIL",
    }
    out = score_run([base, twin], [always, always])

    assert out["unsupported_rate"] == 0.0   # flawless on the headline safety number
    assert out["coverage"] == 0.0           # and caught here
    assert out["over_refusal_rate"] == 1.0


def test_pairs_must_be_right_together():
    base = _compliant("A")
    twin = strip_provenance(base, "C-1")
    half_right = {base["id"]: _pred(base), twin["id"]: _pred(base)}  # twin answered as if base
    assert score_pairs([(base, twin)], half_right)["pair_accuracy"] == 0.0
    both = {base["id"]: _pred(base), twin["id"]: _pred(twin)}
    assert score_pairs([(base, twin)], both)["pair_accuracy"] == 1.0


def test_malformed_output_is_an_error_not_a_skip():
    base = _compliant("A")
    assert score_one(base, {"verdict": "MAYBE"})["schema_valid"] is False
    assert score_run([base], [{"verdict": "MAYBE"}])["schema_validity"] == 0.0
