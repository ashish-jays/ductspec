"""Nothing scored can be learned from, whichever way the arms are built."""

import copy
from pathlib import Path

import pytest

from ductspec import leakage
from ductspec.generate import build

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def splits():
    return build()


def test_the_audited_build_has_no_leak(splits):
    assert leakage.check(splits) == {}


@pytest.mark.parametrize("size", [None, 600])
def test_both_arms_have_no_leak_at_any_size(size):
    if not (ROOT / "data" / "sources" / "UFGS-23-30-00.pdf").exists():
        pytest.skip("run ductspec.sources first")
    pytest.importorskip("curatorkit")
    from ductspec.generate import build_arms

    for arm, arm_splits in build_arms(train_size=size).items():
        assert leakage.check(arm_splits) == {}, arm


def _moved(splits, predicate, to):
    bad = copy.deepcopy(splits)
    row = next(r for s in bad.values() for r in s if predicate(r))
    for rows in bad.values():
        if row in rows:
            rows.remove(row)
    bad[to].append(row)
    return bad, row["id"]


def test_a_test_row_copied_into_training_is_caught(splits):
    bad = copy.deepcopy(splits)
    row = next(r for r in bad["test"] if r["panel"] == "transfer")
    bad["train"].append(copy.deepcopy(row))
    found = leakage.check(bad)
    assert row["id"] in found["holdout_never_trained"]
    assert row["id"] in found["no_identical_prompt"]
    assert row["family"] in found["family_in_one_split"]


def test_a_known_authority_test_row_copied_into_training_is_caught(splits):
    """The panel no authority rule covers: only the prompt and family checks
    stand between a copied known-authority test row and training."""
    bad = copy.deepcopy(splits)
    row = next(r for r in bad["test"] if r["panel"] == "known_authority")
    bad["train"].append(copy.deepcopy(row))
    found = leakage.check(bad)
    assert row["id"] in found["no_identical_prompt"]
    assert row["family"] in found["family_in_one_split"]


def test_a_validation_row_in_training_is_caught(splits):
    bad, rid = _moved(splits, lambda r: r["panel"] == "authority_validation", "train")
    assert rid in leakage.check(bad)["validation_never_trained"]


def test_a_probe_row_in_training_is_caught(splits):
    bad, rid = _moved(splits, lambda r: r["panel"] == "curation_probe", "train")
    assert rid in leakage.check(bad)["probe_only_in_test"]


def test_a_training_only_row_scored_is_caught(splits):
    bad, rid = _moved(splits, lambda r: r["meta"]["requirement"].startswith(("R-DART", "U-UH")), "dev")
    assert rid in leakage.check(bad)["train_only_never_scored"]


def test_a_validation_unmade_choice_in_training_is_caught(splits):
    bad, rid = _moved(splits, lambda r: r["meta"]["requirement"] == "U-UH-1.2.L-material", "train")
    assert rid in leakage.check(bad)["validation_never_trained"]
