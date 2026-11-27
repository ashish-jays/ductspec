"""The single-rule arm differs from curated only by the rows of one rejected rule."""
import collections
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_single_rule", ROOT / "scripts/build_single_rule.py")
bsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsr)

from ductspec.sample import _authority, answer_type  # noqa: E402


def rows(name):
    return [json.loads(l) for l in open(ROOT / f"data/build/{name}/train.jsonl", encoding="utf-8")]


def test_only_the_rule_rows_are_new_and_the_mix_is_unchanged():
    curated, matched = rows("curated"), rows("raw-matched")
    out, dropped = bsr.build(curated, matched)
    cur_ids = {r["id"] for r in curated}
    new = [r for r in out if r["id"] not in cur_ids]
    assert len(new) == len(dropped) == 44
    assert {r["meta"]["requirement"] for r in new} == {bsr.RULE}
    for a in {_authority(r) for r in curated}:
        assert collections.Counter(answer_type(r) for r in out if _authority(r) == a) == \
            collections.Counter(answer_type(r) for r in curated if _authority(r) == a)


def test_another_rejected_rule_in_the_input_is_refused():
    curated, matched = rows("curated"), rows("raw-matched")
    bad = copy.deepcopy(curated)
    bad[0]["meta"]["requirement"] = "R-UFGS-2.12.10.1-material"
    with pytest.raises(AssertionError, match="another rejected rule"):
        bsr.build(bad, matched)


def test_a_matched_arm_without_the_rule_is_refused():
    curated, matched = rows("curated"), rows("raw-matched")
    with pytest.raises(AssertionError, match="no rows of the rule"):
        bsr.build(curated, [r for r in matched if r["meta"]["requirement"] != bsr.RULE])
