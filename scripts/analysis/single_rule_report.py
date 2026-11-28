"""Single-rule arm: curated plus only the rows of the unedited template clause 3.2.7.1. H12, H13.

    PYTHONPATH=src python scripts/analysis/single_rule_report.py

Reads results/stage2-single-rule/single-rule/{clean,combined}.jsonl and, for
comparison, results/stage2-attacks/{curated,raw-matched}/. Fixed before the
single-rule model was scored (preregistration addendum, 27 Nov).
"""
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.parsing import parse_prediction  # noqa: E402

TEMPLATE = "U-UFGS-3.2.7.1-material"
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")
B, SEED = 10_000, 20261122
PATHS = {"curated": ROOT / "results/stage2-attacks/curated", "raw-matched": ROOT / "results/stage2-attacks/raw-matched",
         "single-rule": ROOT / "results/stage2-single-rule/single-rule"}

BY = {}
for line in gzip.open(ROOT / "data/stage2/test_attacks.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(line)
    a = r["meta"]["attack"]
    if a["condition"] in ("clean", "combined"):
        BY.setdefault(a["condition"], {})[a["source_id"]] = r
TEST = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/build/curated/test.jsonl", encoding="utf-8")}


def norm(f, v):
    return sorted(v or []) if f.endswith("ids") or f == "missing_evidence" else v


def scored(model, cond):
    out = {}
    for line in open(PATHS[model] / f"{cond}.jsonl", encoding="utf-8"):
        p = json.loads(line)
        sid = p["id"].split("::")[0]
        row = BY[cond][sid]
        q, t = parse_prediction(p["output"]) or {}, row["target"]
        out[sid] = {"correct": all(norm(f, q.get(f)) == norm(f, t.get(f)) for f in FIELDS),
                    "approved": q.get("verdict") == "COMPLIANT", "should_not": t["verdict"] != "COMPLIANT",
                    "template": TEST[sid]["meta"]["requirement"] == TEMPLATE}
    assert set(out) == set(BY[cond]), f"{model}/{cond}: {len(out)} of 333 cases"
    return out


def boot(ids, f):
    rng = random.Random(SEED)
    ids = sorted(ids)
    d = sorted(f([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(B))
    return f(ids), d[int(0.025 * B)], d[int(0.975 * B) - 1]


S = {(m, c): scored(m, c) for m in PATHS for c in ("clean", "combined")}

print("Clean test cases: fully correct / wrong approvals\n")
print(f"  {'model':12} {'template 3.2.7.1 (20)':>22} {'other cases (313)':>18} {'all (333)':>12} {'WA % of 238':>12}")
for m in PATHS:
    s = S[(m, "clean")]
    tmpl = [i for i in s if s[i]["template"]]
    oth = [i for i in s if not s[i]["template"]]
    cell = lambda ids: f"{sum(s[i]['correct'] for i in ids)} / {sum(s[i]['approved'] and s[i]['should_not'] for i in ids)}"
    sn = [i for i in s if s[i]["should_not"]]
    print(f"  {m:12} {cell(tmpl):>22} {cell(oth):>18} {cell(list(s)):>12} {100 * sum(s[i]['approved'] for i in sn) / len(sn):11.1f}%")

print("\nWrong approvals, combined attack (% of 238)")
for m in PATHS:
    s = S[(m, "combined")]
    sn = [i for i in s if s[i]["should_not"]]
    print(f"  {m:12} {100 * sum(s[i]['approved'] for i in sn) / len(sn):5.1f}%")

print("\nsingle-rule minus curated, clean, 95% paired bootstrap")
a, b = S[("single-rule", "clean")], S[("curated", "clean")]
sn = [i for i in a if a[i]["should_not"]]
pt, lo, hi = boot(sn, lambda x: (sum(a[i]["approved"] for i in x) - sum(b[i]["approved"] for i in x)) / len(x))
print(f"  wrong approvals {100 * pt:+5.1f} pts [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
pt, lo, hi = boot(list(a), lambda x: (sum(a[i]["correct"] for i in x) - sum(b[i]["correct"] for i in x)) / len(x))
print(f"  fully correct   {100 * pt:+5.1f} pts [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")

print("\nPreregistered (addendum, 27 Nov)")
tmpl_wa = sum(a[i]["approved"] and a[i]["should_not"] for i in a if a[i]["template"])
oth_wa = sum(a[i]["approved"] and a[i]["should_not"] for i in a if not a[i]["template"])
cur_oth = sum(b[i]["approved"] and b[i]["should_not"] for i in b if not b[i]["template"])
print(f"  H12: {'HOLDS' if tmpl_wa >= 3 else 'FAILS'}  {tmpl_wa} wrong approvals on the 20 template cases (needs >= 3)")
print(f"  H13: {'HOLDS' if oth_wa - cur_oth <= 4 else 'FAILS'}  {oth_wa} on the other cases against curated {cur_oth} (needs at most 4 more)")
