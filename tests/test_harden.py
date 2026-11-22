"""Hardening examples keep their answers and never reuse a test attack."""
import pytest

from ductspec import attacks, harden
from tests.test_attacks import _row


def _train(n_refuse=130, n_comply=70):
    rows = []
    for i in range(n_refuse + n_comply):
        r = _row(verdict="NON_COMPLIANT" if i < n_refuse else "COMPLIANT")
        r["id"] = f"R-X-material-M101-{i:03d}"
        r["split"] = "train"
        rows.append(r)
    return rows


def test_build_keeps_targets_and_mixes_verdicts():
    rows = _train()
    out = harden.build(rows, {"2.1.A"}, set())
    assert len(out) == 180
    by_id = {r["id"]: r for r in rows}
    for a in out:
        assert a["target"] == by_id[a["meta"]["harden"]["source_id"]]["target"]
    assert sum(a["target"]["verdict"] == "COMPLIANT" for a in out) == 60
    assert {a["meta"]["harden"]["condition"] for a in out} == set(harden.ATTACKS) | {"control"}


def test_no_training_wording_is_a_test_wording():
    mine = {w.strip() for ws in harden.NOTES.values() for w in ws} | {t.strip() for t in harden.TAMPER}
    assert not mine & harden._test_wordings()
    assert attacks.FAKE_CLAUSE_ID not in harden.FAKE_IDS


def test_build_refuses_a_test_wording(monkeypatch):
    monkeypatch.setitem(harden.NOTES, "authority", (attacks.NOTES["authority"][0],) * 3)
    with pytest.raises(AssertionError, match="also a test wording"):
        harden.build(_train(), {"2.1.A"}, set())


def test_build_refuses_a_fake_id_that_is_real():
    with pytest.raises(AssertionError, match="fake clause id"):
        harden.build(_train(), {"2.1.A", harden.FAKE_IDS[0]}, set())


def test_build_refuses_a_test_row():
    rows = _train()
    with pytest.raises(AssertionError, match="not a training row"):
        harden.build(rows, {"2.1.A"}, {r["id"] for r in rows})


def test_build_refuses_a_changed_target(monkeypatch):
    real = harden.attack

    def bad(row, condition):
        out = real(row, condition)
        out["target"] = {**out["target"], "verdict": "COMPLIANT", "observation_ids": ["x"]}
        return out

    monkeypatch.setattr(harden, "attack", bad)
    with pytest.raises(AssertionError, match="target changed"):
        harden.build(_train(), {"2.1.A"}, set())


def test_build_refuses_added_text_with_a_value(monkeypatch):
    monkeypatch.setitem(harden.NOTES, "control", ("ALL DUCTS GALV.",) * 3)
    with pytest.raises(AssertionError, match="names a property"):
        harden.build(_train(), {"2.1.A"}, set())


def test_build_refuses_an_added_note_id_already_present(monkeypatch):
    monkeypatch.setattr(attacks, "_new_note_ids", lambda row, count: ["M101-note8"] * count)
    with pytest.raises(AssertionError, match="already in the packet"):
        harden.build(_train(), {"2.1.A"}, set())


def test_attacked_note_is_never_an_observation():
    for a in harden.build(_train(), {"2.1.A"}, set()):
        assert not set(a["meta"]["harden"]["added_drawing_ids"]) & set(a["target"]["observation_ids"])
