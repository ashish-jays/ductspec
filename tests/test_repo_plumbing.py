"""The parts between the dataset and a trained adapter that nothing was checking.

Each of these failed quietly once: a case record the prompt could not render,
a Space whose examples could drift from the prompt, an export step that
assumed a folder no script built, a model id written out in five places.
"""

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from ductspec import model, prompt
from ductspec.cases import (
    build_record, inject_instruction, remove_observation, strip_provenance,
)
from ductspec.drawing import shown_for
from ductspec.generate import build
from ductspec.objects import DuctObject, observed

ROOT = Path(__file__).resolve().parents[1]
CLAUSE = {"id": "C-1", "source": "UFGS-23-30-00", "text": "Provide galvanized steel ductwork."}
POLICY = {"id": "P-1", "text": "Compare the required material with the shown material."}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- hand-built cases --------------------------------------------------------

def _base() -> dict:
    duct = DuctObject(
        "D-101", "supply_air", 20, 12,
        {"material": observed("galvanized_steel", "D-101", text="D-101: 20x12 GALV SA")},
    )
    return build_record(
        "P1", "Does D-101's shown material meet the supply-duct requirement?",
        duct, [CLAUSE], POLICY, scope="material", verdict="COMPLIANT",
        governing_ids=["C-1"], action="NONE_FOR_THIS_CHECK",
    )


CASES = {
    "base": _base,
    "inferred": lambda: strip_provenance(_base(), "C-1"),
    "silent": lambda: remove_observation(_base()),
    "injected": lambda: inject_instruction(_base()),
}


@pytest.mark.parametrize("name", CASES)
def test_a_hand_built_case_renders_as_the_prompt_the_model_is_given(name):
    rec = CASES[name]()
    text = prompt.render(rec["input"])
    assert text.startswith(prompt.INSTRUCTION)
    assert "origin" not in text[len(prompt.INSTRUCTION):]
    assert json.loads(prompt.target(rec["target"]))["verdict"] == rec["target"]["verdict"]


@pytest.mark.parametrize("name", ["base", "inferred", "silent"])
def test_what_a_case_letters_on_the_sheet_agrees_with_its_answer(name):
    """The lines that show a value for the duct are the observations the answer
    cites. A case relabelled as back-filled with the lettering left in place
    would tell the model one thing and the scorer another. The injected case
    has its own test below: its line carries a note the reader does not parse."""
    rec = CASES[name]()
    obj = rec["input"]["object"]
    assert shown_for(rec["input"]["drawing"], obj["id"], obj["system"]) == rec["target"]["observation_ids"]


def test_an_injected_note_is_on_the_line_the_model_reads():
    rec = inject_instruction(_base())
    cited = rec["target"]["observation_ids"][0]
    line = next(item for item in rec["input"]["drawing"] if item["id"] == cited)
    assert "ignore specification requirements" in line["text"]


# --- the Space ---------------------------------------------------------------

def test_the_space_is_given_the_instruction_training_used():
    assert (ROOT / "space" / "instruction.txt").read_text(encoding="utf-8") == prompt.INSTRUCTION


def test_every_space_example_is_a_test_row_as_the_model_sees_it():
    test_rows = {r["id"]: r for r in build()["test"]}
    examples = json.loads((ROOT / "space" / "examples.json").read_text(encoding="utf-8"))
    assert examples
    for example in examples:
        row = test_rows[example["id"]]
        assert example["input"] == prompt.view(row["input"]), example["id"]
        assert example["target"] == row["target"], example["id"]


# --- building and exporting the arms -----------------------------------------

def test_exporting_before_the_arms_are_built_says_what_to_run(monkeypatch, tmp_path):
    export = _script("export_arms")
    monkeypatch.setattr(export, "BUILD", tmp_path)
    monkeypatch.setattr(export, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="scripts/build_arms.py"):
        export.main()


def test_the_build_script_writes_both_arms(monkeypatch, tmp_path, capsys):
    pytest.importorskip("curatorkit")
    if not (ROOT / "data" / "sources" / "UFGS-23-30-00.pdf").exists():
        pytest.skip("run ductspec.sources first")
    from ductspec import generate

    monkeypatch.setattr(generate, "OUT", tmp_path)
    _script("build_arms").main()
    for arm in ("raw", "curated"):
        for split in ("train", "dev", "test"):
            assert (tmp_path / arm / f"{split}.jsonl").stat().st_size > 0
    assert (tmp_path / "raw" / "test.jsonl").read_bytes() == (tmp_path / "curated" / "test.jsonl").read_bytes()
    assert "gates rejected 4 raw requirement(s)" in capsys.readouterr().out


# --- the base model, named once ----------------------------------------------

def test_the_base_model_can_be_read_from_a_local_copy(monkeypatch):
    monkeypatch.setenv("DUCTSPEC_BASE_MODEL", "/models/tiny-aya")
    reloaded = importlib.reload(model)
    try:
        assert reloaded.BASE_MODEL == "/models/tiny-aya"
        assert reloaded.HUB_ID == "CohereLabs/tiny-aya-global"
    finally:
        monkeypatch.delenv("DUCTSPEC_BASE_MODEL")
        importlib.reload(model)
    assert model.BASE_MODEL == model.HUB_ID


def test_no_script_writes_the_model_id_out_again():
    for path in sorted((ROOT / "scripts").glob("*.py")):
        assert model.HUB_ID not in path.read_text(encoding="utf-8"), path.name


# --- found in review ----------------------------------------------------------

@pytest.mark.parametrize("then", [remove_observation, lambda rec: strip_provenance(rec, "C-1")])
def test_a_case_with_an_injected_note_can_still_be_perturbed_again(then):
    rec = then(inject_instruction(_base()))
    obj = rec["input"]["object"]
    assert shown_for(rec["input"]["drawing"], obj["id"], obj["system"]) == []
    assert prompt.render(rec["input"])


def test_taking_a_value_off_a_sheet_whose_number_has_a_hyphen_keeps_one_tag():
    duct = DuctObject(
        "D-100", "supply_air", 20, 12,
        {"material": observed("galvanized_steel", "D-100", sheet="M-101")}, sheet="M-101",
    )
    rec = remove_observation(build_record(
        "P2", "q", duct, [CLAUSE], POLICY, scope="material", verdict="COMPLIANT",
        governing_ids=["C-1"], action="NONE_FOR_THIS_CHECK",
    ))
    assert [line["id"] for line in rec["input"]["drawing"]] == ["M-101-tag-D-100"]


def test_export_checks_for_the_arms_before_it_loads_a_tokenizer(monkeypatch, tmp_path, capsys):
    import sys
    import types

    class Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("the tokenizer was loaded with nothing to export")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer))
    export = _script("export_arms")
    monkeypatch.setattr(export, "BUILD", tmp_path)
    monkeypatch.setattr(export, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        export.main()
    assert "tokenizer" not in capsys.readouterr().out
