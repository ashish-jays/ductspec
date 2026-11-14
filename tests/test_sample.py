"""The training set is chosen by stated rules, not by whatever comes first."""

from collections import Counter

import pytest

from ductspec import sample
from ductspec.sample import ANSWER_MIX, _clause, _Rotation, answer_type, balanced


def _row(i, kind, req="R-UFGS-1.1-material", sheet="M101", system="supply_air"):
    verdict, missing = (kind, []) if kind in ("COMPLIANT", "NON_COMPLIANT") else ("INSUFFICIENT_EVIDENCE", [kind])
    return {
        "id": f"{req}-{sheet}-{i:03d}",
        "target": {"verdict": verdict, "missing_evidence": missing},
        "meta": {"requirement": req},
        "input": {"object": {"system": system}, "drawing": [{"id": f"{sheet}-tag-D-{i}"}]},
    }


def _pool(per_type=200):
    rows, i = [], 0
    for kind in ANSWER_MIX:
        for k in range(per_type):
            rows.append(_row(i, kind, req=f"R-UFGS-{k % 7}.1-material", sheet=f"M{101 + k % 5}"))
            i += 1
    return rows


def test_the_mix_is_the_stated_one():
    out = balanced(_pool(), 500)
    counts = Counter(answer_type(r) for r in out)
    assert len(out) == 500
    for kind, share in ANSWER_MIX.items():
        assert abs(counts[kind] - share * 500) <= 1, (kind, counts[kind])


def test_applicability_unknown_is_capped():
    assert ANSWER_MIX["APPLICABILITY_UNKNOWN"] <= 0.20
    assert abs(sum(ANSWER_MIX.values()) - 1.0) < 1e-9


def test_a_short_answer_type_gives_all_it_has_and_the_rest_make_up_the_size():
    rows = [r for r in _pool() if answer_type(r) != "MISSING_AUTHORITY"]
    rows += [_row(9000 + i, "MISSING_AUTHORITY") for i in range(5)]
    out = balanced(rows, 600)
    assert len(out) == 600
    assert Counter(answer_type(r) for r in out)["MISSING_AUTHORITY"] == 5


def test_the_same_seed_gives_the_same_rows_and_another_seed_does_not():
    pool = _pool()
    ids = lambda rows: [r["id"] for r in rows]
    assert ids(balanced(pool, 300)) == ids(balanced(pool, 300))
    assert set(ids(balanced(pool, 300, seed=1))) != set(ids(balanced(pool, 300, seed=2)))


def test_every_clause_family_is_used_before_any_is_used_twice():
    rows = [_row(i, "COMPLIANT", req=f"R-UFGS-{i % 6}.1-material") for i in range(60)]
    out = sample._spread(rows, 6, seed=0)
    assert len({_clause(r) for r in out}) == 6


def test_requirements_from_one_clause_share_its_turn():
    """Four pressure classes listed in one sentence are one family, so a fifth
    clause is not starved by them."""
    rows = [_row(i, "COMPLIANT", req=f"R-DART-3.2.A-pressure-{k}") for k in range(4) for i in range(10 * k, 10 * k + 10)]
    rows += [_row(100 + i, "COMPLIANT", req="R-UFGS-2.4.5-insulation") for i in range(10)]
    out = sample._spread(rows, 2, seed=0)
    assert {_clause(r) for r in out} == {("DART", "3.2.A"), ("UFGS", "2.4.5")}


def test_sheets_within_a_requirement_take_turns():
    """The first version drained one sheet before touching the next."""
    rows = [_row(i, "COMPLIANT", sheet=sheet) for sheet in ("M101", "M102", "M201") for i in range(20)]
    out = sample._spread(rows, 3, seed=0)
    assert {r["input"]["drawing"][0]["id"].split("-")[0] for r in out} == {"M101", "M102", "M201"}


def test_rotation_drops_empty_groups_and_takes_everything():
    rot = _Rotation({"a": {"x": [{"id": "1"}], "y": [{"id": "2"}, {"id": "3"}]}, "b": {"z": [{"id": "4"}]}}, seed=0)
    taken = []
    while not rot.empty():
        taken.append(rot.take()["id"])
    assert sorted(taken) == ["1", "2", "3", "4"]


def test_impossible_requests_are_refused():
    with pytest.raises(ValueError):
        balanced(_pool(1), 10_000)
    with pytest.raises(ValueError):
        balanced(_pool(), 0)
    with pytest.raises(ValueError, match="no share"):
        bad = _row(1, "COMPLIANT")
        bad["target"] = {"verdict": "INSUFFICIENT_EVIDENCE", "missing_evidence": ["UNRESOLVED_CONFLICT"]}
        balanced([bad], 1)


def test_both_arms_get_the_same_mix_and_never_a_scored_row():
    from pathlib import Path

    if not (Path(__file__).resolve().parents[1] / "data" / "sources" / "UFGS-23-30-00.pdf").exists():
        pytest.skip("run ductspec.sources first")
    pytest.importorskip("curatorkit")
    from ductspec.generate import build_arms

    arms = build_arms(train_size=400)
    mixes = [Counter(answer_type(r) for r in arms[a]["train"]) for a in ("raw", "curated")]
    assert mixes[0] == mixes[1]
    scored = {r["id"] for r in arms["curated"]["dev"] + arms["curated"]["test"]}
    for arm in ("raw", "curated"):
        assert len(arms[arm]["train"]) == 400
        assert not scored & {r["id"] for r in arms[arm]["train"]}
