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

from ductspec.audit import scorers, with_outputs  # noqa: E402
from ductspec.prompt import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "predictions"

ALWAYS_COMPLIANT = json.dumps({
    "verdict": "COMPLIANT", "governing_ids": [], "observation_ids": [],
    "missing_evidence": [], "action": "NONE_FOR_THIS_CHECK"})
ALWAYS_ABSTAIN = json.dumps({
    "verdict": "INSUFFICIENT_EVIDENCE", "governing_ids": [], "observation_ids": [],
    "missing_evidence": ["MISSING_OBSERVATION"], "action": "REQUEST_MISSING_DETAIL"})


def saved_outputs(name: str) -> dict[str, str]:
    path = PREDICTIONS / f"{name}.jsonl"
    return {json.loads(line)["id"]: json.loads(line)["output"] for line in path.open()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/build/curated/test.jsonl")
    ap.add_argument("--states", nargs="*", default=["base", "raw", "curated"])
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / args.split).open()]

    # Each model state is scored from its own saved generations, attached to
    # the samples. Replaying them through callables instead would make every
    # state share one cache entry — see ductspec.audit.with_outputs.
    states: dict[str, dict[str, str]] = {}
    for state in args.states:
        if (PREDICTIONS / f"{state}.jsonl").exists():
            states[state] = saved_outputs(state)
        else:
            print(f"(skipping {state}: no generations yet)")

    states["oracle"] = {r["id"]: json.dumps(r["target"]) for r in rows}
    states["always-compliant"] = {r["id"]: ALWAYS_COMPLIANT for r in rows}
    states["always-abstain"] = {r["id"]: ALWAYS_ABSTAIN for r in rows}

    runs = {}
    for name, saved in states.items():
        runs[name] = ak.evaluate(
            with_outputs(rows, saved), model="precomputed", scorers=scorers(),
            experiment_name="ductspec-conformance", tags=["ductspec", "track1", name],
        )

    lines = [f"DuctSpec — {len(rows)} samples, {len(scorers())} metrics", ""]
    metrics = [m.name for m in scorers()]
    width = max(len(m) for m in metrics) + 2
    lines.append("metric".ljust(width) + "".join(n.rjust(20) for n in runs))
    lines.append("-" * (width + 20 * len(runs)))
    for metric in metrics:
        row = metric.ljust(width)
        for run in runs.values():
            value = run.headline.get(metric)
            row += (f"{value:.4f}" if value is not None else "n/a").rjust(20)
        lines.append(row)
    lines.append("")
    for name, run in runs.items():
        lines.append(f"{name}: run_id={run.run_id}")
    summary = "\n".join(lines)
    print(summary)

    out = ROOT / "data" / "build" / "comparison.txt"
    # Scoring saved generations needs no build; the folder may not exist yet.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary + "\n")
    print(f"\n-> {out.relative_to(ROOT)}")
    return

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
