"""Real-drawing panel: what each model answered on each case.

    PYTHONPATH=src python scripts/analysis/real_report.py [predictions dir]

Reads <dir>/<model>/real.jsonl (default results/stage2-real). Eight cases make
about four independent decisions, so this reports case by case and by decision,
with no percentages or significance claims.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.parsing import parse_prediction  # noqa: E402

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/stage2-real"
# Predictions on the split panel cite sentence lines (<note>.s<k>); map them back to the note,
# so the targets, written against whole notes, score both panels the same way.
from ductspec.notes import parent  # noqa: E402
MODELS = ("base", "raw", "raw-matched", "curated", "curated-hardened", "curated-hardened-v2")
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")
SHORT = {"COMPLIANT": "approve", "NON_COMPLIANT": "non-compl.", "INSUFFICIENT_EVIDENCE": "insuff."}
DECISION = {1: "real conflict (1-1/2\" vs 2\" wrap)", 2: "real conflict (1-1/2\" vs 2\" wrap)",
            3: "real conflict (1-1/2\" vs 2\" wrap)", 4: "return liner 1\" vs 1\" (system unconfirmed)",
            5: "material copied from spec", 6: "material copied from spec",
            7: "material not on drawing", 8: "material not on drawing"}

ROWS = {}
for line in gzip.open(ROOT / "data/stage2/real_addison.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(line)
    ROWS[r["id"]] = r


def norm(field, value):
    if field == "observation_ids":
        return sorted({parent(v) for v in (value or [])})
    return sorted(value or []) if field.endswith("ids") or field == "missing_evidence" else value


def load(model):
    path = PRED / model / "real.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        out[p["id"]] = parse_prediction(p["output"]) or {}
    assert set(out) == set(ROWS), f"{model}: predictions do not cover the panel"
    return out


preds = {m: p for m in MODELS if (p := load(m)) is not None}
for condition in ("clean", "combined"):
    ids = [i for i, r in ROWS.items() if r["meta"]["attack"]["condition"] == condition]
    print(f"\n== {condition} ({len(ids)} cases)  cell = verdict, * if all five fields are right")
    print(f"  {'#':>2} {'decision':42} {'answer':10} " + " ".join(f"{m:>16}" for m in preds))
    for i in sorted(ids):
        r = ROWS[i]
        n = int(r["meta"]["attack"]["source_id"].split("-")[2])
        t = r["target"]
        cells = []
        for m, p in preds.items():
            q = p[i]
            ok = all(norm(f, q.get(f)) == norm(f, t.get(f)) for f in FIELDS)
            cells.append(f"{SHORT.get(q.get('verdict'), 'unparsed')}{'*' if ok else ''}")
        print(f"  {n:>2} {DECISION[n]:42} {SHORT[t['verdict']]:10} " + " ".join(f"{c:>16}" for c in cells))
    should = [i for i in ids if ROWS[i]["target"]["verdict"] != "COMPLIANT"]
    print(f"  wrong approvals (of {len(should)} that should not be approved): " +
          ", ".join(f"{m} {sum(p[i].get('verdict') == 'COMPLIANT' for i in should)}" for m, p in preds.items()))
    print(f"  fully correct (of {len(ids)}): " +
          ", ".join(f"{m} {sum(all(norm(f, p[i].get(f)) == norm(f, ROWS[i]['target'].get(f)) for f in FIELDS) for i in ids)}"
                    for m, p in preds.items()))
