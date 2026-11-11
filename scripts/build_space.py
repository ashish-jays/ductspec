"""Assemble the Space: examples, the instruction, and a replay file.

The replay exists so the demo still works where no GPU does, and so the thing
a visitor sees is a generation we actually produced and scored rather than a
fresh one nobody checked.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec.prompt import INSTRUCTION, view  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
PREDICTIONS = ROOT / "data" / "predictions"


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/build/curated/test.jsonl").open()]
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    # Prefer families that contain a provenance pair: the same situation with a
    # value read off the drawing and the same value back-filled.
    chosen: list[dict] = []
    for family, members in by_family.items():
        shown = [m for m in members if m["target"]["verdict"] in ("COMPLIANT", "NON_COMPLIANT")]
        filled = [m for m in members
                  if m["target"]["missing_evidence"] == ["INFERRED_VALUE_ONLY"]]
        if shown and filled:
            chosen += [shown[0], filled[0]]
        if len(chosen) >= 6:
            break

    # The Space edits and submits the packet as the model sees it, so the
    # examples carry the view, not the record with its answer key.
    shown = [{**row, "input": view(row["input"])} for row in chosen]
    (SPACE / "examples.json").write_text(json.dumps(shown, indent=1))
    (SPACE / "instruction.txt").write_text(INSTRUCTION)

    replay = {}
    saved = PREDICTIONS / "curated.jsonl"
    if saved.exists():
        outputs = {json.loads(l)["id"]: json.loads(l)["output"] for l in saved.open()}
        for row in chosen:
            if row["id"] in outputs:
                replay[json.dumps(view(row["input"]), sort_keys=True)] = outputs[row["id"]]
    (SPACE / "replay.json").write_text(json.dumps(replay, indent=1))

    print(f"examples: {len(chosen)}  replay entries: {len(replay)}")
    for row in chosen:
        origin = row["input"]["object"]["properties"][row["target"]["scope"]]["origin"]
        print(f"  {row['id']:40} {origin:20} -> {row['target']['verdict']}")


if __name__ == "__main__":
    main()
