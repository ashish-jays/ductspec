"""The test set asks the question the raw arm was taught the wrong answer to.

The raw arm trains on UFGS 3.2.7.1 as settled: an unmade choice between two
materials that the raw extraction recorded as a requirement. An evaluation in
which that clause never appears cannot show what that did.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from ductspec.conventions import BY_NAME, UFGS
from ductspec.derive import Packet, derive
from ductspec.generate import PROBE_PANEL, build
from ductspec.objects import DuctObject, copied_from_spec, observed
from ductspec.requirements import Requirement, load, load_unadopted

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "requirements.raw.yaml"
PROBE_CLAUSE = ("UFGS-23-30-00", "3.2.7.1")


def _shows(row, clause=PROBE_CLAUSE):
    return any((e["source"], e["id"]) == clause for e in row["input"]["evidence"])


@pytest.fixture(scope="module")
def splits():
    return build()


@pytest.fixture(scope="module")
def probes(splits):
    return [r for r in splits["test"] if r["panel"] == PROBE_PANEL]


def test_the_probe_panel_exists_and_is_only_in_the_test_set(splits, probes):
    assert len(probes) >= 16
    for split in ("train", "dev"):
        assert not [r for r in splits[split] if r["panel"] == PROBE_PANEL]


def test_every_probe_shows_the_clause_that_separates_the_arms(probes):
    for row in probes:
        assert _shows(row), row["id"]


def test_the_probe_clause_is_never_trained_on_as_audited(splits):
    """Not in the curated arm's training data in any role."""
    for split in ("train", "dev"):
        assert not [r["id"] for r in splits[split] if _shows(r)]


def test_the_right_answer_to_every_probe_is_that_nothing_is_required_yet(probes):
    for row in probes:
        t = row["target"]
        assert t["verdict"] == "INSUFFICIENT_EVIDENCE", row["id"]
        assert t["missing_evidence"] == ["APPLICABILITY_UNKNOWN"], row["id"]
        assert t["action"] == "REQUEST_APPLICABILITY_CLARIFICATION", row["id"]
        # An unmade choice is not citable as governing.
        assert t["governing_ids"] == [], row["id"]
        assert row["meta"]["answer_available"] is False


def test_the_probes_cover_a_shown_value_a_filled_value_and_a_silent_drawing(probes):
    kinds = set()
    for row in probes:
        prop = row["input"]["object"]["properties"].get(row["input"]["property"])
        kinds.add("silent" if prop is None else prop["origin"])
    assert {"silent", "DRAWING_ANNOTATION", "DRAWING_SCHEDULE", "COPIED_FROM_SPEC"} <= kinds


def test_the_raw_arm_is_trained_to_answer_the_probe_clause_definitively(probes):
    """The conflict the panel measures, checked rather than assumed.

    Built from the unaudited requirements, the training split shows the same
    clause text and labels it COMPLIANT or NON_COMPLIANT.
    """
    raw_train = build(requirements=load(RAW))["train"]
    taught = [r for r in raw_train if _shows(r)]
    assert taught, "the raw arm must train on the probe clause"
    assert {r["target"]["verdict"] for r in taught} & {"COMPLIANT", "NON_COMPLIANT"}

    probe_text = {e["text"] for r in probes for e in r["input"]["evidence"]}
    raw_text = {e["text"] for r in taught for e in r["input"]["evidence"]
                if (e["source"], e["id"]) == PROBE_CLAUSE}
    assert probe_text == raw_text, "the probe shows the clause exactly as the raw arm saw it"


def test_both_arms_can_learn_the_convention_from_other_clauses(splits):
    """Unmade choices other than the probe are in training, so declining to
    treat one as settled is learnable and not a property of one clause id."""
    seen = set()
    for row in splits["train"]:
        if row["target"]["missing_evidence"] == ["APPLICABILITY_UNKNOWN"]:
            for e in row["input"]["evidence"]:
                if UFGS.alternatives.search(e["text"]):
                    seen.add((e["source"], e["id"]))
    assert PROBE_CLAUSE not in seen
    assert len(seen) >= 3, seen

    raw_train = build(requirements=load(RAW))["train"]
    raw_seen = {
        (e["source"], e["id"]) for r in raw_train
        if r["target"]["missing_evidence"] == ["APPLICABILITY_UNKNOWN"]
        for e in r["input"]["evidence"]
    }
    assert seen <= raw_seen, "the raw arm sees the same unmade choices"


def test_a_bracketed_clause_in_training_is_never_given_a_definitive_answer(splits):
    for split in ("train", "dev", "test"):
        for row in splits[split]:
            if any(UFGS.alternatives.search(e["text"]) for e in row["input"]["evidence"]):
                assert row["target"]["verdict"] == "INSUFFICIENT_EVIDENCE", row["id"]


# --- derive -----------------------------------------------------------------

CHOICE = Requirement(
    id="U-1", source="S", clause_id="C-9", page=1, property="material",
    predicate={"op": "unadopted", "drawn_values": ["galvanized_steel", "aluminum"]},
    quote="Provide [galvanized steel] [aluminum].",
)
RULE = Requirement(
    id="R-1", source="S", clause_id="C-1", page=1, property="material",
    predicate={"op": "in", "values": ["galvanized_steel"]}, quote="Galvanized.",
)


@pytest.mark.parametrize("props", [
    {"material": observed("galvanized_steel", "D-1")},
    {"material": copied_from_spec("galvanized_steel", "C-9")},
    {},
])
def test_an_unmade_choice_requires_nothing_whatever_the_drawing_shows(props):
    duct = DuctObject("D-1", "supply_air", 20, 12, props)
    out = derive(duct, "material", Packet((CHOICE,), ("S",), "P-1"))
    assert out.verdict == "INSUFFICIENT_EVIDENCE"
    assert out.missing_evidence == ("APPLICABILITY_UNKNOWN",)
    assert out.governing_ids == ()
    assert out.action == "REQUEST_APPLICABILITY_CLARIFICATION"


def test_a_settled_rule_beside_an_unmade_choice_still_governs():
    duct = DuctObject("D-1", "supply_air", 20, 12, {"material": observed("galvanized_steel", "D-1")})
    out = derive(duct, "material", Packet((CHOICE, RULE), ("S",), "P-1"))
    assert out.verdict == "COMPLIANT"
    assert out.governing_ids == ("C-1",)


# --- the quotes, against the document ---------------------------------------

@pytest.mark.parametrize("req", [r for r, _ in load_unadopted()], ids=lambda r: r.id)
def test_each_unmade_choice_is_quoted_from_a_clause_the_document_marks_unsettled(req):
    pdf = ROOT / "data" / "sources" / f"{req.source}.pdf"
    if not pdf.exists():
        pytest.skip("corpus not fetched: run `python -m ductspec.sources`")
    from ductspec.clauses import parse_clauses

    from ductspec.sources import convention_of

    convention = BY_NAME[convention_of(req.source)]
    clauses = {c.clause_id: c for c in parse_clauses(pdf, req.source, convention)}
    assert req.clause_id in clauses, f"{req.source} has no clause {req.clause_id}"
    clause = clauses[req.clause_id]
    assert " ".join(req.quote.split()) in " ".join(clause.text.split())
    # In the issuer's own convention: brackets, or a sentence handing the
    # choice to the designer.
    shows_choice = convention.alternatives.search(req.quote) or (
        convention.deferral is not None and convention.deferral.search(req.quote))
    assert shows_choice, "the quote itself must show the choice"
    assert not clause.operative
    assert req.page == clause.page_start


def test_no_unmade_choice_is_also_recorded_as_an_audited_requirement():
    """A clause can hold a settled sentence beside an unmade one: Dartmouth
    3.2.B settles galvanized steel for supply, return and general exhaust and
    leaves kitchen and lab ductwork to the designer. What must never happen is
    a settled requirement and an unmade choice for the same property reaching
    the same system, which would give one duct two contradictory answers."""
    from ductspec.objects import SYSTEMS

    def reach(r):
        return set(r.systems) or set(SYSTEMS)

    for req, _ in load_unadopted():
        for settled in load():
            if (settled.source, settled.clause_id, settled.property) != (req.source, req.clause_id, req.property):
                continue
            assert settled.quote != req.quote, req.id
            assert settled.element != req.element or not reach(settled) & reach(req), (req.id, settled.id)


def test_exactly_the_clause_that_separates_the_arms_is_the_probe():
    raw_only = {(r.source, r.clause_id) for r in load(RAW) if r.source.startswith("UFGS")} - {
        (r.source, r.clause_id) for r in load()
    }
    probe = {(r.source, r.clause_id) for r, is_probe in load_unadopted() if is_probe}
    assert probe == raw_only == {PROBE_CLAUSE}


# --- the report that reads the panel ----------------------------------------

def _probe_report():
    spec = importlib.util.spec_from_file_location("probe_report", ROOT / "scripts" / "probe_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer(row):
    return json.dumps({k: row["target"][k] for k in (
        "verdict", "governing_ids", "observation_ids", "missing_evidence", "action")})


def test_the_report_separates_a_right_answer_from_the_raw_arms_answer(splits, probes):
    report = _probe_report().report
    right = report(splits["test"], {r["id"]: _answer(r) for r in probes})
    assert right["n"] == len(probes)
    assert right["strict_accuracy"] == 1.0
    assert right["unsupported_rate"] == 0.0

    as_taught = json.dumps({
        "verdict": "COMPLIANT", "governing_ids": ["3.2.7.1"], "observation_ids": [],
        "missing_evidence": [], "action": "NONE_FOR_THIS_CHECK"})
    wrong = report(splits["test"], {r["id"]: as_taught for r in probes})
    assert wrong["strict_accuracy"] == 0.0
    assert wrong["unsupported_rate"] == 1.0
    assert wrong["false_approval_rate"] == 1.0
    assert wrong["verdicts"] == {"COMPLIANT": len(probes)}


def test_the_report_refuses_to_score_a_panel_with_missing_generations(splits, probes):
    outputs = {r["id"]: _answer(r) for r in probes[1:]}
    # The count of missing rows is the point: a bare lookup failure names one id
    # and hides how much of the panel went unscored.
    with pytest.raises(KeyError, match="1 probe rows have no generation"):
        _probe_report().report(splits["test"], outputs)
