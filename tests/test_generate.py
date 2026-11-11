"""Properties the built dataset must have, checked against the files on disk."""

import json
from collections import Counter
from pathlib import Path

import pytest

from ductspec.generate import build
from ductspec.schema import INDEPENDENT_ORIGINS

BUILD = Path(__file__).resolve().parents[1] / "data" / "build"


@pytest.fixture(scope="module")
def splits():
    return build()


def test_no_family_straddles_a_split(splits):
    """The leakage test. Members of a family differ by one controlled change,
    so a twin in the test set whose sibling was trained on measures recall
    rather than reasoning."""
    owner: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            seen = owner.setdefault(row["family"], split)
            assert seen == split, f"family {row['family']} appears in {seen} and {split}"


def test_the_transfer_panel_is_never_trained_on(splits):
    for split in ("train", "dev"):
        assert not [r for r in splits[split] if r["panel"] == "transfer"]
    assert [r for r in splits["test"] if r["panel"] == "transfer"]


def test_held_out_sources_appear_in_no_training_evidence(splits):
    """Not even as a distractor."""
    for split in ("train", "dev"):
        for row in splits[split]:
            for item in row["input"]["evidence"]:
                assert not item["source"].startswith("VA-"), row["id"]


def test_every_definitive_verdict_rests_on_the_drawing(splits):
    for rows in splits.values():
        for row in rows:
            if row["target"]["verdict"] in ("COMPLIANT", "NON_COMPLIANT"):
                prop = row["input"]["object"]["properties"][row["target"]["scope"]]
                assert prop["origin"] in INDEPENDENT_ORIGINS, row["id"]
                assert row["target"]["observation_ids"], row["id"]
                assert row["meta"]["answer_available"], row["id"]


def test_every_abstention_says_why_and_what_to_do(splits):
    for rows in splits.values():
        for row in rows:
            if row["target"]["verdict"] == "INSUFFICIENT_EVIDENCE":
                assert row["target"]["missing_evidence"], row["id"]
                assert row["target"]["action"] != "NONE_FOR_THIS_CHECK", row["id"]
                assert not row["meta"]["answer_available"], row["id"]


def test_every_citation_resolves_to_supplied_evidence(splits):
    for rows in splits.values():
        for row in rows:
            available = {e["id"] for e in row["input"]["evidence"]}
            for cid in row["target"]["governing_ids"]:
                assert cid in available, f"{row['id']} cites {cid}, which is not in the packet"


def test_the_model_never_sees_the_answerability_metadata(splits):
    for rows in splits.values():
        for row in rows:
            assert "answer_available" not in json.dumps(row["input"])


def test_all_three_verdicts_and_all_four_reasons_are_represented(splits):
    verdicts = Counter(r["target"]["verdict"] for rows in splits.values() for r in rows)
    assert len(verdicts) == 3 and min(verdicts.values()) > 20

    reasons = Counter(
        code for rows in splits.values() for r in rows
        for code in r["target"]["missing_evidence"]
    )
    assert {"INFERRED_VALUE_ONLY", "MISSING_OBSERVATION",
            "MISSING_AUTHORITY", "APPLICABILITY_UNKNOWN"} <= set(reasons)


def test_the_provenance_twins_differ_only_in_origin(splits):
    """Somewhere in the data there is a pair with the same value, one read off
    the drawing and one back-filled, with different answers. That pair is the
    experiment."""
    rows = [r for rows in splits.values() for r in rows]
    by_situation = {}
    for r in rows:
        prop = r["input"]["object"]["properties"].get(r["target"]["scope"])
        if not prop or prop.get("value") is None:
            continue
        # Everything about the situation except where the value came from.
        key = (
            r["family"],
            prop["value"],
            tuple(sorted(e["id"] for e in r["input"]["evidence"])),
            tuple(sorted(r["input"]["object"].get("facts", {}).items())),
        )
        by_situation.setdefault(key, []).append((prop["origin"], r["target"]))

    found = 0
    for members in by_situation.values():
        shown = [t for o, t in members if o in INDEPENDENT_ORIGINS]
        filled = [t for o, t in members if o not in INDEPENDENT_ORIGINS]
        if not shown or not filled:
            continue
        # Where no authority settles the question, both members abstain for
        # that reason and provenance never comes up — correct, and covered by
        # its own test below.
        if not all(t["verdict"] in ("COMPLIANT", "NON_COMPLIANT") for t in shown):
            continue
        for t in filled:
            assert t["verdict"] == "INSUFFICIENT_EVIDENCE"
            assert t["missing_evidence"] == ["INFERRED_VALUE_ONLY"]
            assert t["observation_ids"] == []
            # It still knows which clause governs.
            assert t["governing_ids"]
        for t in shown:
            assert t["verdict"] in ("COMPLIANT", "NON_COMPLIANT")
        found += 1

    assert found, "the dataset must contain same-value, different-provenance pairs"


def test_authority_is_settled_before_provenance_is_asked(splits):
    """Where nothing supplied governs the property, the answer is
    MISSING_AUTHORITY whether or not the value was read off the drawing.

    Reporting INFERRED_VALUE_ONLY there would tell the reviewer to go and find
    a drawing detail, when the real problem is that the specification section
    is missing from the packet. Different finding, different person to ask.
    """
    rows = [r for rows in splits.values() for r in rows]
    blocked = [
        r for r in rows
        if r["target"]["missing_evidence"] == ["MISSING_AUTHORITY"]
        and (r["input"]["object"]["properties"].get(r["target"]["scope"]) or {}).get("value")
        is not None
    ]
    assert blocked, "the data must contain cases where no supplied clause governs"

    origins = {
        r["input"]["object"]["properties"][r["target"]["scope"]]["origin"] for r in blocked
    }
    assert origins & set(INDEPENDENT_ORIGINS), "including some read off the drawing"
    for r in blocked:
        assert r["target"]["action"] != "NONE_FOR_THIS_CHECK"


# --- the two training arms --------------------------------------------------

def test_the_arms_differ_only_in_which_requirements_were_sound():
    """The ablation. Same generator, same seed, same family-to-split
    assignment, same test set, same number of training records — the only
    difference is that the raw arm was built from an unaudited requirement set.
    """
    from ductspec.generate import build_arms

    arms = build_arms()
    raw, curated = arms["raw"], arms["curated"]

    assert raw["test"] == curated["test"], "the evaluation cannot depend on the arm"
    assert raw["dev"] == curated["dev"]
    assert len(raw["train"]) == len(curated["train"]), "token exposure must match"

    raw_reqs = {r["meta"]["requirement"] for r in raw["train"]}
    cur_reqs = {r["meta"]["requirement"] for r in curated["train"]}

    # Exactly one faulty requirement separates them: UFGS 3.2.7.1 offers the
    # engineer two materials for grease-laden duct and asks them to choose.
    assert raw_reqs - cur_reqs == {"R-UFGS-3.2.7.1-material"}
    assert cur_reqs - raw_reqs == set()


def test_a_familys_split_does_not_depend_on_what_else_was_generated():
    """A family lands in the same split whichever arm is being built. Otherwise
    the ablation measures the shuffle."""
    from ductspec.generate import build, build_arms
    from ductspec.requirements import load

    arms = build_arms()
    where = {}
    for arm, splits in arms.items():
        for split, rows in splits.items():
            for row in rows:
                key = (arm, row["family"])
                assert where.setdefault(key, split) == split

    full = build(requirements=load())
    full_where = {r["family"]: s for s, rows in full.items() for r in rows}
    for (arm, family), split in where.items():
        if family in full_where and arm == "curated":
            assert full_where[family] == split, family
