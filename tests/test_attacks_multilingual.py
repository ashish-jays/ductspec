"""A translated case is the English attacked case with only the note's language changed."""
import json

import pytest

from ductspec import attacks, attacks_multilingual as ml
from ductspec.prompt import render
from tests.test_attacks import _row


@pytest.mark.parametrize("lang", ml.LANGUAGES)
@pytest.mark.parametrize("kind", ml.KINDS)
def test_only_the_note_text_differs_from_english(lang, kind):
    row = _row()
    en = attacks.attack(row, kind)
    tr = ml.translate(row, lang, kind)
    assert tr["target"] == row["target"]
    (note_id,) = tr["meta"]["attack"]["added_drawing_ids"]
    assert note_id == en["meta"]["attack"]["added_drawing_ids"][0]
    en_lines = {d["id"]: d["text"] for d in en["input"]["drawing"]}
    tr_lines = {d["id"]: d["text"] for d in tr["input"]["drawing"]}
    assert [d["id"] for d in en["input"]["drawing"]] == [d["id"] for d in tr["input"]["drawing"]]
    assert {k for k in en_lines if en_lines[k] != tr_lines[k]} == {note_id}
    assert tr_lines[note_id] == ml.TRANSLATIONS[lang][kind][en["meta"]["attack"]["wordings"][kind]]
    assert {k: v for k, v in en["input"].items() if k != "drawing"} == \
        {k: v for k, v in tr["input"].items() if k != "drawing"}
    assert json.dumps(tr_lines[note_id])[1:-1] in render(tr["input"])
    assert tr_lines[note_id] in ml.render_script(tr["input"])


def test_script_rendering_differs_from_render_only_inside_the_note():
    row = _row()
    assert ml.render_script(row["input"]) == render(row["input"])
    tr = ml.translate(row, "hi", "injection", script=True)
    assert tr["meta"]["attack"]["render"] == "script"
    assert ml.render_for(tr) == ml.render_script(tr["input"]) != render(tr["input"])
    plain = ml.translate(row, "hi", "injection")
    assert ml.render_for(plain) == render(plain["input"])
    note = ml.TRANSLATIONS["hi"]["injection"][tr["meta"]["attack"]["wordings"]["injection"]]
    assert note not in render(plain["input"]) and note in ml.render_script(tr["input"])


def test_build_refuses_a_clean_packet_that_is_not_ascii():
    row = _row()
    row["input"]["evidence"][0]["text"] += " 25" + chr(176) + "C"
    with pytest.raises(AssertionError, match="not pure ASCII"):
        ml.build([row])


def test_compliant_cases_are_not_attacked():
    assert ml.build([_row(verdict="COMPLIANT")]) == []
    assert len(ml.build([_row()])) == (len(ml.LANGUAGES) + len(ml.SCRIPT_LANGUAGES)) * len(ml.KINDS)


def test_every_translation_has_one_text_per_english_wording():
    for lang in ml.TRANSLATIONS:
        for kind in ml.KINDS:
            assert len(ml.TRANSLATIONS[lang][kind]) == len(attacks.NOTES[kind])


def test_build_refuses_a_translation_with_a_number(monkeypatch):
    monkeypatch.setitem(ml.TRANSLATIONS["es"], "authority", ("NOTA 4.",) * 3)
    with pytest.raises(AssertionError, match="carries a number"):
        ml.build([_row()])


def test_translate_refuses_when_the_english_note_is_not_where_expected(monkeypatch):
    monkeypatch.setitem(attacks.NOTES, "authority", ("SOMETHING ELSE",) * 3)
    real = attacks.attack

    def moved(row, condition):
        out = real(row, condition)
        out["input"]["drawing"] = [d for d in out["input"]["drawing"]
                                   if d["id"] not in out["meta"]["attack"]["added_drawing_ids"]]
        return out

    monkeypatch.setattr(ml, "attack", moved)
    with pytest.raises(AssertionError, match="not found"):
        ml.translate(_row(), "es", "authority")
