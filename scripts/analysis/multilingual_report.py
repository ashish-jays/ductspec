"""Stage 2 multilingual test: wrong approvals per language, paired with English.

    PYTHONPATH=src python scripts/analysis/multilingual_report.py <multilingual preds dir> <english preds dir>

Both dirs hold <model>/<condition>.jsonl. Every translated case is the English
attacked case with only the note's language changed, so each language is
compared with English on the same 238 should-not-approve cases.
"""
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.parsing import parse_prediction  # noqa: E402

ML, EN = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = ("raw", "curated")
KINDS = ("authority", "injection")
VARIANTS = (("es", "Spanish"), ("hi", "Hindi, as our pipeline sends it (escape codes)"),
            ("hi_script", "Hindi, shown as script"), ("ar", "Arabic, as our pipeline sends it (escape codes)"),
            ("ar_script", "Arabic, shown as script"))
B, SEED = 10_000, 20261122


ROWS = {}
for line in gzip.open(ROOT / "data/stage2/test_attacks_multilingual.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(line)
    ROWS[r["id"]] = r


CASES = {r["meta"]["attack"]["source_id"] for r in ROWS.values()}  # the 238 should-not-approve cases
assert len(CASES) == 238, len(CASES)


def approvals(path):
    """Per-case outcome for one prediction file, keyed by the clean case id.

    Refuses a file that does not cover exactly the 238 cases, so a truncated
    file cannot silently shrink a denominator. An English file (333 cases) is
    cut to the 238 after the check that it covers them.
    """
    out = {}
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        q = parse_prediction(p["output"]) or {}
        out[p["id"].split("::")[0]] = {"approved": q.get("verdict") == "COMPLIANT",
                                       "obs": set(q.get("observation_ids") or []), "id": p["id"]}
    missing = CASES - set(out)
    if missing:
        raise SystemExit(f"{path}: {len(missing)} of the 238 cases have no prediction")
    if len(out) not in (238, 333):  # a translated file has the 238; an English file all 333
        raise SystemExit(f"{path}: unexpected case count {len(out)}")
    return {k: v for k, v in out.items() if k in CASES}


def boot(ids, f):
    rng = random.Random(SEED)  # fresh per interval, as in attack_report.py
    ids = sorted(ids)
    draws = sorted(f([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(B))
    return f(ids), draws[int(0.025 * B)], draws[int(0.975 * B) - 1]


def diff(a, b):
    return lambda x: (sum(a[i]["approved"] for i in x) - sum(b[i]["approved"] for i in x)) / len(x)


def fmt(pt, lo, hi):
    return f"{100 * pt:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]{' *' if lo > 0 or hi < 0 else '  '}"


print("Wrong approvals on the 238 should-not-approve cases; [95% paired bootstrap]; * = significant\n")
for kind in KINDS:
    print(f"== {kind}")
    print(f"  {'variant':48} {'model':8} {'wrong appr.':>11} {'cited note':>11}   {'vs English, same model':>24}")
    for m in MODELS:
        en = approvals(EN / m / f"{kind}.jsonl")
        clean = approvals(EN / m / "clean.jsonl")
        ids = sorted(next(iter([approvals(ML / m / f"{kind}_es.jsonl")])))
        en_rate = sum(en[i]["approved"] for i in ids) / len(ids)
        clean_rate = sum(clean[i]["approved"] for i in ids) / len(ids)
        print(f"  {'clean (no note)':48} {m:8} {100 * clean_rate:10.1f}%")
        print(f"  {'English':48} {m:8} {100 * en_rate:10.1f}%")
    for suffix, label in VARIANTS:
        for m in MODELS:
            en = approvals(EN / m / f"{kind}.jsonl")
            tr = approvals(ML / m / f"{kind}_{suffix}.jsonl")
            ids = sorted(tr)
            rate = sum(tr[i]["approved"] for i in ids) / len(ids)
            cited = sum(bool(tr[i]["obs"] & set(ROWS[tr[i]["id"]]["meta"]["attack"]["added_drawing_ids"]))
                        for i in ids) / len(ids)
            print(f"  {label:48} {m:8} {100 * rate:10.1f}% {100 * cited:10.1f}%   {fmt(*boot(ids, diff(tr, en)))}")
    print()

print("Curated minus raw, same variant")
for kind in KINDS:
    for suffix, label in (("", "English"),) + VARIANTS:
        a = approvals((EN if not suffix else ML) / "curated" / (f"{kind}.jsonl" if not suffix else f"{kind}_{suffix}.jsonl"))
        b = approvals((EN if not suffix else ML) / "raw" / (f"{kind}.jsonl" if not suffix else f"{kind}_{suffix}.jsonl"))
        ids = sorted(approvals(ML / "curated" / f"{kind}_es.jsonl"))
        print(f"  {kind:10} {label:48} {fmt(*boot(ids, diff(a, b)))}")
