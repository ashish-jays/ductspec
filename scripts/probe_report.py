"""Score each model state on the curation probe panel.

The probe panel shows the one clause that separates the two training arms: an
unmade choice the raw extraction recorded as a requirement. The headline
evaluation averages it into 300-odd rows; this reads it on its own, per state,
so the effect of curation is visible where it is expected to be.

Needs only the saved generations, not AuditKIT.

    python scripts/probe_report.py                 # base, raw, curated
    python scripts/probe_report.py --states base
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec.generate import PROBE_PANEL  # noqa: E402
from ductspec.parsing import parse_prediction  # noqa: E402
from ductspec.scoring import score_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "predictions"


def report(rows: list[dict], outputs: dict[str, str]) -> dict:
    """Scores and the verdicts given, for the probe rows only."""
    probes = [r for r in rows if r["panel"] == PROBE_PANEL]
    missing = [r["id"] for r in probes if r["id"] not in outputs]
    if missing:
        raise KeyError(f"{len(missing)} probe rows have no generation, e.g. {missing[0]}")
    preds = [parse_prediction(outputs[r["id"]]) or {} for r in probes]
    scores = score_run(probes, preds)
    scores["verdicts"] = dict(Counter(p.get("verdict") or "UNPARSED" for p in preds))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/build/curated/test.jsonl")
    ap.add_argument("--states", nargs="*", default=["base", "raw", "curated"])
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / args.split).open()]
    n = sum(r["panel"] == PROBE_PANEL for r in rows)
    if not n:
        raise SystemExit(f"{args.split} has no {PROBE_PANEL} rows; rebuild the dataset")
    print(f"{PROBE_PANEL}: {n} rows. Correct answer on every one: "
          "INSUFFICIENT_EVIDENCE / APPLICABILITY_UNKNOWN.\n")

    for state in args.states:
        path = PREDICTIONS / f"{state}.jsonl"
        if not path.exists():
            print(f"{state}: no generations at {path.relative_to(ROOT)}")
            continue
        outputs = {json.loads(l)["id"]: json.loads(l)["output"] for l in path.open()}
        scores = report(rows, outputs)
        print(f"{state}")
        for key in ("strict_accuracy", "unsupported_rate", "false_approval_rate"):
            print(f"  {key:22} {scores[key]}")
        print(f"  verdicts given         {scores['verdicts']}\n")


if __name__ == "__main__":
    main()
