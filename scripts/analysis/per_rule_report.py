"""Where the unsound rules hurt: clean test results by the rule governing each case.

    PYTHONPATH=src python scripts/analysis/per_rule_report.py [predictions dir]

Groups the 333 clean test cases into: the three UFGS requirements the curation
gates reject (the only training difference between raw-matched and curated),
the same clauses where the test labels them as unresolved templates,
the other UFGS requirements, and the other issuers. Reads <dir>/<model>/clean.jsonl
(default results/stage2-attacks).
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.parsing import parse_prediction  # noqa: E402

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/stage2-attacks"
MODELS = ("raw", "raw-matched", "curated", "curated-hardened")
REJECTED = ("R-UFGS-2.12.10.1-material", "R-UFGS-3.2.7.1-material", "R-UFGS-3.2.7.2-material")
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")

ROWS = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/build/curated/test.jsonl", encoding="utf-8")}
assert len(ROWS) == 333


def norm(field, value):
    return sorted(value or []) if field.endswith("ids") or field == "missing_evidence" else value


def load(model):
    out = {}
    for line in open(PRED / model / "clean.jsonl", encoding="utf-8"):
        p = json.loads(line)
        sid = p["id"].split("::")[0]
        t, q = ROWS[sid]["target"], parse_prediction(p["output"]) or {}
        out[sid] = (all(norm(f, q.get(f)) == norm(f, t.get(f)) for f in FIELDS),
                    q.get("verdict") == "COMPLIANT" and t["verdict"] != "COMPLIANT")
    assert set(out) == set(ROWS), f"{model}: clean predictions do not cover the 333 cases"
    return out


def group(row):
    req = row["meta"]["requirement"]
    if req in REJECTED:
        return f"rejected: {req}"
    # Same clause text as a rejected rule, labelled as an unresolved template
    # (bracketed options, no choice made). Not a different rule.
    if any(req.endswith(r[1:]) for r in REJECTED):
        return f"same clause, unresolved: {req}"
    return "other UFGS rules" if "-UFGS-" in req else "other issuers"


models = [m for m in MODELS if (PRED / m / "clean.jsonl").exists()]
scores = {m: load(m) for m in models}
groups = collections.defaultdict(list)
for sid, row in ROWS.items():
    groups[group(row)].append(sid)

print("Clean test cases by governing rule: fully correct (all cases) and wrong approvals (should-not-approve cases)\n")
print(f"  {'group':38} {'cases':>5} {'should not':>10}   " + "  ".join(f"{m:>18}" for m in models))
for g in sorted(groups):
    ids = groups[g]
    sn = [i for i in ids if ROWS[i]["target"]["verdict"] != "COMPLIANT"]
    cells = [f"{sum(scores[m][i][0] for i in ids):3} ok, {sum(scores[m][i][1] for i in sn):2} WA" for m in models]
    print(f"  {g:38} {len(ids):5} {len(sn):10}   " + "  ".join(f"{c:>18}" for c in cells))
