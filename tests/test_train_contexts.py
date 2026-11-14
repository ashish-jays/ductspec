"""Train-only runs add drawings to training and change nothing that is scored.

A model that has seen four sheets per clause can recall a family. More sheets
make reading the lettering the cheaper thing to learn. The runs added for that
must not touch the dev or test set, or every number reported before and after
them is measured on a different exam.
"""

import json

import pytest

from ductspec import generate
from ductspec.generate import CONTEXTS, PROBE_PANEL, TRAIN_CONTEXTS, build
from ductspec.requirements import load, load_unadopted

TRAIN_SHEETS = {c["sheet"] for c in TRAIN_CONTEXTS}


def _sheet(row):
    """The sheet the row's drawing lines are lettered on."""
    sheets = {line["id"].split("-")[0] for line in row["input"]["drawing"]}
    assert len(sheets) == 1, (row["id"], sheets)
    return sheets.pop()


@pytest.fixture(scope="module")
def with_runs():
    return build()


@pytest.fixture(scope="module")
def without_runs():
    return build(train_contexts=())


def test_the_dev_and_test_sets_are_the_same_records_with_or_without_them(with_runs, without_runs):
    for split in ("dev", "test"):
        assert json.dumps(with_runs[split]) == json.dumps(without_runs[split])


def test_every_row_trained_on_before_is_still_trained_on_unchanged(with_runs, without_runs):
    after = {r["id"]: r for r in with_runs["train"]}
    assert len(after) == len(with_runs["train"]), "duplicate training ids"
    for row in without_runs["train"]:
        assert after[row["id"]] == row


def test_they_add_training_rows(with_runs, without_runs):
    added = [r for r in with_runs["train"] if _sheet(r) in TRAIN_SHEETS]
    assert len(added) == len(with_runs["train"]) - len(without_runs["train"])
    assert len(added) > len(without_runs["train"])


def test_no_train_only_sheet_is_ever_scored(with_runs):
    for split in ("dev", "test"):
        assert not [r["id"] for r in with_runs[split] if _sheet(r) in TRAIN_SHEETS]


def test_no_family_on_a_train_only_sheet_has_a_member_that_is_scored(with_runs):
    added = {r["family"] for r in with_runs["train"] if _sheet(r) in TRAIN_SHEETS}
    scored = {r["family"] for split in ("dev", "test") for r in with_runs[split]}
    assert not added & scored


def test_held_out_sources_and_the_probe_clause_stay_untrained(with_runs):
    probe_ids = {req.id for req, is_probe in load_unadopted() if is_probe}
    held_out = {req.id for req in load() if req.source.startswith("VA-")}
    assert probe_ids and held_out
    for row in with_runs["train"]:
        owner = row["id"].rsplit("-", 2)[0]
        assert owner not in probe_ids | held_out, row["id"]
    assert not [r for r in with_runs["train"] if r["panel"] == PROBE_PANEL]


def test_a_train_only_run_on_an_evaluated_sheet_is_refused():
    with pytest.raises(ValueError, match="reuse evaluated sheets"):
        build(train_contexts=(dict(TRAIN_CONTEXTS[0], sheet=CONTEXTS[0]["sheet"]),))


def test_the_generator_default_is_the_declared_runs():
    assert generate.build.__defaults__[-1] is TRAIN_CONTEXTS


def test_the_added_unmade_choices_are_trained_on_and_never_scored(with_runs):
    from ductspec.requirements import train_only_unadopted

    added = train_only_unadopted()
    # 5 UFGS clauses, and 3 from Dartmouth and Houston (dataset v2); a fourth,
    # Houston 1.2.L, is held out for validation.
    assert len(added) == 8
    owner = lambda row: row["id"].rsplit("-", 2)[0]
    assert {owner(r) for r in with_runs["train"]} >= added
    for split in ("dev", "test"):
        assert not [r["id"] for r in with_runs[split] if owner(r) in added]


def test_a_probe_cannot_also_be_train_only(tmp_path, monkeypatch):
    import yaml
    from ductspec import requirements

    data = yaml.safe_load(requirements.UNADOPTED.read_text(encoding="utf-8"))
    probe = next(e for e in data["unadopted"] if e["probe"])
    probe["train_only"] = True
    bad = tmp_path / "unadopted.yaml"
    bad.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(requirements, "UNADOPTED", bad)
    monkeypatch.setattr(generate, "train_only_unadopted", lambda: requirements.train_only_unadopted(bad))
    monkeypatch.setattr(generate, "load_unadopted", lambda: requirements.load_unadopted(bad))
    with pytest.raises(AssertionError):
        build(train_contexts=())


def test_the_probe_twin_is_held_out_for_validation_and_never_trained(with_runs):
    owner = lambda row: row["id"].rsplit("-", 2)[0]
    assert not [r for r in with_runs["train"] if owner(r) == "U-UH-1.2.L-material"]
    dev = [r for r in with_runs["dev"] if owner(r) == "U-UH-1.2.L-material"]
    assert dev and {r["panel"] for r in dev} == {"authority_validation"}
