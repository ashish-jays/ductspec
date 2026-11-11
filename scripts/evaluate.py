"""Score every model state through AuditKIT, on one frozen test set.

The generations are read from disk, so this is deterministic and costs
nothing. AuditKIT replays them as a callable model, which means the run carries
a fingerprint, the comparison is graded in the right sense per metric, and
pairwise significance is computed for us.

The baselines are included on purpose. A number is only readable against what
an oracle, a model that certifies everything, and a model that certifies
nothing would score on the same rows.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import auditkit as ak  # noqa: E402

from ductspec.audit import scorers, to_samples  # noqa: E402
from ductspec.prompt import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "predictions"

ALWAYS_COMPLIANT = json.dumps({
    "verdict": "COMPLIANT", "governing_ids": [], "observation_ids": [],
    "missing_evidence": [], "action": "NONE_FOR_THIS_CHECK"})
ALWAYS_ABSTAIN = json.dumps({
    "verdict": "INSUFFICIENT_EVIDENCE", "governing_ids": [], "observation_ids": [],
    "missing_evidence": ["MISSING_OBSERVATION"], "action": "REQUEST_MISSING_DETAIL"})


def replay(name: str, rows: list[dict]):
    """A model that answers from a saved generation file."""
    path = PREDICTIONS / f"{name}.jsonl"
    saved = {json.loads(line)["id"]: json.loads(line)["output"] for line in path.open()}
    by_prompt = {render(r["input"]): r["id"] for r in rows}
    missing = [r["id"] for r in rows if r["id"] not in saved]
    if missing:
        raise SystemExit(
            f"{name}: no generation for {len(missing)} rows (first: {missing[0]}). "
            "Scoring a partial run would silently compare different row sets."
        )

    def model(prompts):
        return [saved[by_prompt[p]] for p in prompts]

    model.__name__ = name
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/build/curated/test.jsonl")
    ap.add_argument("--states", nargs="*", default=["base", "raw", "curated"])
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / args.split).open()]
    samples = to_samples(rows)

    models, names = [], []
    for state in args.states:
        if (PREDICTIONS / f"{state}.jsonl").exists():
            models.append(replay(state, rows))
            names.append(state)
        else:
            print(f"(skipping {state}: no generations yet)")

    by_prompt = {render(r["input"]): r for r in rows}
    models += [
        lambda ps: [json.dumps(by_prompt[p]["target"]) for p in ps],
        lambda ps: [ALWAYS_COMPLIANT] * len(ps),
        lambda ps: [ALWAYS_ABSTAIN] * len(ps),
    ]
    names += ["oracle", "always-compliant", "always-abstain"]

    result = ak.compare_models(
        models, samples, scorers=scorers(), model_names=names,
        experiment_name="ductspec-conformance", tags=["ductspec", "track1"],
    )
    print(result.summary())

    out = ROOT / "data" / "build" / "comparison.txt"
    out.write_text(result.summary())
    print(f"\n-> {out.relative_to(ROOT)}")

    # Per-panel, so transfer to the held-out authority is reported separately
    # from behaviour under authorities seen in training.
    for panel in ("known_authority", "transfer"):
        subset = [r for r in rows if r["panel"] == panel]
        if not subset or len(models) <= 3:
            continue
        sub = ak.compare_models(
            models[: len(args.states)], to_samples(subset),
            scorers=scorers(), model_names=names[: len(args.states)],
            experiment_name=f"ductspec-{panel}",
        )
        path = ROOT / "data" / "build" / f"comparison-{panel}.txt"
        path.write_text(sub.summary())
        print(f"-> {path.relative_to(ROOT)}  ({len(subset)} rows)")


if __name__ == "__main__":
    main()
