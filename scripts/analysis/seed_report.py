"""Second training seed: curated and hardening v2 retrained with seed 7, H10 and H11.

    PYTHONPATH=src python scripts/analysis/seed_report.py

Reads results/stage2-seed7/<model>-seed7/<condition>.jsonl and, for the seed-42
comparison, results/stage2-attacks/<model>/ (clean, combined) and
results/stage2-fresh/<model>/ (the five fresh conditions).
"""
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.attacks_fresh import CONDITIONS as FRESH  # noqa: E402
from ductspec.parsing import parse_prediction  # noqa: E402

CONDS = ("clean", "combined") + tuple(FRESH)
MODELS = ("curated", "curated-hardened-v2")
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")
B, SEED = 10_000, 20261122

BY = {}
for name in ("test_attacks.jsonl.gz", "test_attacks_fresh.jsonl.gz"):
    for line in gzip.open(ROOT / "data/stage2" / name, "rt", encoding="utf-8"):
        r = json.loads(line)
        a = r["meta"]["attack"]
        if a["condition"] in CONDS:
            BY.setdefault(a["condition"], {})[a["source_id"]] = r
assert all(len(BY[c]) == 333 for c in CONDS), {c: len(BY.get(c, {})) for c in CONDS}


def norm(field, value):
    return sorted(value or []) if field.endswith("ids") or field == "missing_evidence" else value


def path(model, seed, cond):
    if seed == 7:
        return ROOT / "results/stage2-seed7" / f"{model}-seed7" / f"{cond}.jsonl"
    return ROOT / ("results/stage2-attacks" if cond in ("clean", "combined") else "results/stage2-fresh") / model / f"{cond}.jsonl"


def scored(model, seed, cond):
    out = {}
    for line in open(path(model, seed, cond), encoding="utf-8"):
        p = json.loads(line)
        sid = p["id"].split("::")[0]
        row = BY[cond][sid]
        q, t = parse_prediction(p["output"]) or {}, row["target"]
        out[sid] = {"correct": all(norm(f, q.get(f)) == norm(f, t.get(f)) for f in FIELDS),
                    "approved": q.get("verdict") == "COMPLIANT", "should_not": t["verdict"] != "COMPLIANT",
                    "answerable": t["verdict"] != "INSUFFICIENT_EVIDENCE",
                    "refused": q.get("verdict") == "INSUFFICIENT_EVIDENCE"}
    assert set(out) == set(BY[cond]), f"{model} seed {seed} {cond}: {len(out)} of 333 cases"
    return out


S = {(m, s, c): scored(m, s, c) for m in MODELS for s in (42, 7) for c in CONDS}


def wa(s):
    ids = [i for i in s if s[i]["should_not"]]
    return sum(s[i]["approved"] for i in ids) / len(ids)


def fc(s):
    return sum(s[i]["correct"] for i in s) / len(s)


def boot(ids, f):
    rng = random.Random(SEED)
    ids = sorted(ids)
    draws = sorted(f([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(B))
    return f(ids), draws[int(0.025 * B)], draws[int(0.975 * B) - 1]


print("Wrong approvals (of 238) and fully correct (of 333), seed 42 against seed 7\n")
print(f"  {'model':20} {'condition':20} {'WA s42':>7} {'WA s7':>7} {'FC s42':>7} {'FC s7':>7}")
for m in MODELS:
    for c in CONDS:
        a, b = S[(m, 42, c)], S[(m, 7, c)]
        print(f"  {m:20} {c:20} {100 * wa(a):6.1f}% {100 * wa(b):6.1f}% {100 * fc(a):6.1f}% {100 * fc(b):6.1f}%")

print("\nUnneeded refusals on clean (of 158 answerable)")
for m in MODELS:
    for s in (42, 7):
        x = S[(m, s, "clean")]
        ans = [i for i in x if x[i]["answerable"]]
        print(f"  {m:20} seed {s:2}: {100 * sum(x[i]['refused'] for i in ans) / len(ans):5.1f}%")

print("\nv2 minus curated, wrong approvals, same seed, 95% paired bootstrap")
for s in (42, 7):
    for c in CONDS:
        a, b = S[("curated-hardened-v2", s, c)], S[("curated", s, c)]
        ids = [i for i in a if a[i]["should_not"]]
        pt, lo, hi = boot(ids, lambda x: (sum(a[i]["approved"] for i in x) - sum(b[i]["approved"] for i in x)) / len(x))
        print(f"  seed {s:2} {c:20} {100 * pt:+5.1f} pts [{100 * lo:+5.1f}, {100 * hi:+5.1f}]{'  significant' if lo > 0 or hi < 0 else ''}")

print("\nPreregistered (addendum, 24 Nov, second seed)")
v2 = {c: wa(S[("curated-hardened-v2", 7, c)]) for c in ("combined",) + tuple(FRESH)}
print(f"  H10: {'HOLDS' if all(r <= 0.05 for r in v2.values()) else 'FAILS'}  " +
      ", ".join(f"{c} {100 * r:.1f}%" for c, r in v2.items()) + " (each needs <= 5.0%)")
a, b = S[("curated-hardened-v2", 7, "combined")], S[("curated", 7, "combined")]
ids = [i for i in a if a[i]["should_not"]]
_, _, hi = boot(ids, lambda x: (sum(a[i]["approved"] for i in x) - sum(b[i]["approved"] for i in x)) / len(x))
print(f"  H11: {'HOLDS' if hi < 0 else 'FAILS'}  seed-7 v2 minus seed-7 curated under combined, upper bound {100 * hi:+.1f} pts (needs < 0)")
