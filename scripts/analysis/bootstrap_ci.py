"""95% paired bootstrap confidence intervals on the 333 test cases.

Reads the committed generations in results/ and the frozen test set built by
scripts/build_arms.py. Per-case scores are recomputed here and checked against
each folder's AuditKIT metrics.txt before any interval is drawn; the script
stops if they differ.

    PYTHONPATH=src python scripts/analysis/bootstrap_ci.py
"""
import json
import random
import re
from pathlib import Path

from ductspec.parsing import parse_prediction

ROOT = Path(__file__).resolve().parents[2]
ROWS = [json.loads(l) for l in open(ROOT / "data/build/curated/test.jsonl", encoding="utf-8")]
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")
STATES = {
    "untrained": "aya-expanse-8b-base",
    "v1 raw": "aya-expanse-8b-raw-r64",
    "v2 raw": "aya-expanse-8b-v2-raw-r64",
    "v2 curated": "aya-expanse-8b-v2-curated-r64",
}
METRICS = ("fully correct", "wrong approvals", "unneeded refusals")
AUDITKIT_KEYS = ("strict_accuracy", "false_approval_rate", "over_refusal_rate")


def norm(field, value):
    return sorted(value or []) if field.endswith("ids") or field == "missing_evidence" else value


def auditkit_values(folder):
    text = (ROOT / "results" / folder / "metrics.txt").read_text(encoding="utf-8")
    return tuple(float(re.search(rf"^{k}\s+([0-9.]+)", text, re.M).group(1)) for k in AUDITKIT_KEYS)


def per_case(folder):
    preds = {}
    for line in open(ROOT / "results" / folder / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        preds[p["id"]] = parse_prediction(p["output"]) or {}
    out = []
    for r in ROWS:
        p, t = preds[r["id"]], r["target"]
        out.append((
            all(norm(f, p.get(f)) == norm(f, t.get(f)) for f in FIELDS),
            p.get("verdict") == "COMPLIANT" and t["verdict"] != "COMPLIANT",
            p.get("verdict") == "INSUFFICIENT_EVIDENCE" and t["verdict"] != "INSUFFICIENT_EVIDENCE",
        ))
    return out


scores = {}
for label, folder in STATES.items():
    s = per_case(folder)
    mine = tuple(round(sum(x[i] for x in s) / len(s), 4) for i in range(3))
    kit = auditkit_values(folder)
    ok = all(abs(a - b) < 0.0006 for a, b in zip(mine, kit))
    print(f"{label:11} recomputed {mine}  AuditKIT {kit}  {'match' if ok else 'MISMATCH'}")
    assert ok, f"{folder}: per-case scores do not reproduce AuditKIT; not drawing intervals"
    scores[label] = s

random.seed(20261115)
B, n = 10_000, len(ROWS)
samples = [[random.randrange(n) for _ in range(n)] for _ in range(B)]


def interval(values):
    values = sorted(values)
    return values[int(0.025 * B)], values[int(0.975 * B) - 1]


print("\n95% CI per model")
for label, s in scores.items():
    parts = []
    for m in range(3):
        lo, hi = interval([sum(s[i][m] for i in idx) / n for idx in samples])
        parts.append(f"{METRICS[m]} {sum(x[m] for x in s) / n:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  {label:11} " + " | ".join(parts))

print("\nPaired differences, v2 curated minus other (interval excluding 0 = significant)")
for other in ("untrained", "v1 raw", "v2 raw"):
    a, b, parts = scores["v2 curated"], scores[other], []
    for m in range(3):
        point = sum(a[i][m] - b[i][m] for i in range(n)) / n
        lo, hi = interval([sum(a[i][m] - b[i][m] for i in idx) / n for idx in samples])
        parts.append(f"{METRICS[m]} {point:+.1%} [{lo:+.1%}, {hi:+.1%}] sig={'yes' if lo > 0 or hi < 0 else 'no'}")
    print(f"  vs {other:9} " + " | ".join(parts))
