"""The note splitter keeps every word, changes nothing it need not, and maps ids back."""
import json
from pathlib import Path

import pytest

from ductspec import notes


def test_splits_a_multi_sentence_note_and_keeps_every_word():
    text = ("3. DUCTWORK INSULATION TO MEET CURRENT 2015 IECC REQUIRMENTS. RECTANGULAR SUPPLY AIR DUCTWORK IS TO BE "
            "EXTERNALLY INSULATED WITH ONE AND ONE-HALF (1-1/2\") THICK INSULATION. RECTANGULAR RETURN AND TRANSFER "
            "AIR DUCTS SHALL BE LINED WITH 1\" LINER ONLY.")
    parts = notes.sentences(text)
    assert len(parts) == 3
    assert parts[0].startswith("3. DUCTWORK")
    assert " ".join(parts).split() == text.split()


@pytest.mark.parametrize("text", ["ALL SUPPLY AIR DUCTS: GALV U.N.O.", "D-698 THRU D-703: 2\" LINER",
                                  "D-700: 20x12 SA", "ALL RETURN AIR DUCTS: 1\" WG U.N.O."])
def test_one_sentence_lines_are_unchanged(text):
    line = {"id": "M101-note8", "text": text}
    assert notes.split_lines([line]) == [line]


def test_ids_map_back_to_the_note():
    lines = notes.split_lines([{"id": "ME00-note3", "text": "A ONE. B TWO."}])
    assert [l["id"] for l in lines] == ["ME00-note3.s1", "ME00-note3.s2"]
    assert {notes.parent(l["id"]) for l in lines} == {"ME00-note3"}
    assert notes.parent("M101-note8") == "M101-note8"


def test_split_packet_leaves_target_and_meta_alone():
    row = {"input": {"drawing": [{"id": "n1", "text": "X ONE. Y TWO."}]}, "target": {"verdict": "COMPLIANT"},
           "meta": {"a": 1}}
    out = notes.split_packet(row)
    assert out["target"] == row["target"] and out["meta"] == row["meta"]
    assert len(out["input"]["drawing"]) == 2 and len(row["input"]["drawing"]) == 1


def test_frozen_test_set_is_unchanged_by_splitting():
    path = Path(__file__).resolve().parents[1] / "data/build/curated/test.jsonl"
    if not path.exists():
        pytest.skip("test set not built")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    changed = [r["id"] for r in rows if notes.split_packet(r)["input"] != r["input"]]
    assert changed == []
