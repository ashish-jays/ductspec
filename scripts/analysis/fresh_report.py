"""Fresh attack set: wrong approvals and full correctness per model and condition, H8 and H9.

    PYTHONPATH=src python scripts/analysis/fresh_report.py [predictions dir]

Reads <dir>/<model>/<condition>.jsonl (default results/stage2-fresh) and each
model's clean predictions from results/stage2-attacks/<model>/clean.jsonl.
Fixed before any model was scored on this set (preregistration addendum).
"""
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.attacks_fresh import CONDITIONS  # noqa: E402
from ductspec.parsing import parse_prediction  # noqa: E402

PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/stage2-fresh"
MODELS = ("raw", "curated", "curated-hardened", "curated-hardened-v2")
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")
B, SEED = 10_000, 20261122

ROWS = {}
for line in gzip.open(ROOT / "data/stage2/test_attacks_fresh.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(line)
    ROWS[r["id"]] = r
TEST = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/build/curated/test.jsonl", encoding="utf-8")}


def norm(field, value):
    return sorted(value or []) if field.endswith("ids") or field == "missing_evidence" else value


def scored(path, rows_by_id, key):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        row = rows_by_id[p["id"]]
        q, t = parse_prediction(p["output"]) or {}, row["target"]
        out[key(row)] = {"correct": all(norm(f, q.get(f)) == norm(f, t.get(f)) for f in FIELDS),
                         "approved": q.get("verdict") == "COMPLIANT", "should_not": t["verdict"] != "COMPLIANT"}
    return out


def boot(ids, f):
    rng = random.Random(SEED)
    ids = sorted(ids)
    draws = sorted(f([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(B))
    return f(ids), draws[int(0.025 * B)], draws[int(0.975 * B) - 1]


S, CLEAN = {}, {}
for m in MODELS:
    CLEAN[m] = scored(ROOT / "results/stage2-attacks" / m / "clean.jsonl",
                      {k + "::clean": v for k, v in TEST.items()} | TEST, lambda r: r["id"].split("::")[0])
    for c in CONDITIONS:
        path = PRED / m / f"{c}.jsonl"
        if path.exists():
            S[(m, c)] = scored(path, ROWS, lambda r: r["meta"]["attack"]["source_id"])
            assert len(S[(m, c)]) == 333, f"{m}/{c}: {len(S[(m, c)])} cases"

print("Wrong approvals (of 238 that should not be approved), fully correct (of 333), fooled (of cases refused when clean)\n")
print(f"  {'model':20} {'condition':20} {'wrong appr.':>11} {'fully correct':>14} {'fooled':>8}")
for (m, c), s in S.items():
    should = [i for i in s if s[i]["should_not"]]
    held = [i for i in should if not CLEAN[m][i]["approved"]]
    print(f"  {m:20} {c:20} {100 * sum(s[i]['approved'] for i in should) / len(should):10.1f}% "
          f"{100 * sum(s[i]['correct'] for i in s) / len(s):13.1f}% {100 * sum(s[i]['approved'] for i in held) / len(held):7.1f}%")

print("\nv2 minus curated, wrong approvals, 95% paired bootstrap")
for c in CONDITIONS:
    a, b = S.get(("curated-hardened-v2", c)), S.get(("curated", c))
    if a and b:
        ids = [i for i in a if a[i]["should_not"]]
        pt, lo, hi = boot(ids, lambda x: (sum(a[i]["approved"] for i in x) - sum(b[i]["approved"] for i in x)) / len(x))
        print(f"  {c:20} {100 * pt:+5.1f} pts [{100 * lo:+5.1f}, {100 * hi:+5.1f}]{'  significant' if lo > 0 or hi < 0 else ''}")

print("\nPreregistered (addendum, 24 Nov)")
v2 = {c: S.get(("curated-hardened-v2", c)) for c in CONDITIONS}
if all(v2.values()):
    rates = {c: sum(s[i]["approved"] for i in s if s[i]["should_not"]) / sum(s[i]["should_not"] for i in s) for c, s in v2.items()}
    print(f"  H8: {'HOLDS' if all(r <= 0.05 for r in rates.values()) else 'FAILS'}  " +
          ", ".join(f"{c} {100 * r:.1f}%" for c, r in rates.items()) + " (each needs <= 5.0%)")
else:
    print("  H8: NOT RUN")
a, b = S.get(("curated-hardened-v2", "reworded_combined")), S.get(("curated", "reworded_combined"))
if a and b:
    ids = [i for i in a if a[i]["should_not"]]
    _, _, hi = boot(ids, lambda x: (sum(a[i]["approved"] for i in x) - sum(b[i]["approved"] for i in x)) / len(x))
    print(f"  H9: {'HOLDS' if hi < 0 else 'FAILS'}  v2 minus curated under reworded_combined, upper bound {100 * hi:+.1f} pts (needs < 0)")
else:
    print("  H9: NOT RUN")
