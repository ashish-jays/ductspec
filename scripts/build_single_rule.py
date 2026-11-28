"""Build the single-rule arm: the curated arm plus one rejected rule, nothing else.

    PYTHONPATH=src python scripts/build_single_rule.py

The matched raw arm adds 127 rows from three rejected UFGS rules and traces its
extra wrong approvals to one of them, the unedited template clause 3.2.7.1.
This arm adds only that rule's rows (the 44 the matched arm used), so the test
is whether one rule learned as binding is enough to produce the effect, and
whether it lands where the matched arm's did.

- every row is a curated row, except the 44 rows of R-UFGS-3.2.7.1-material,
  taken unchanged from the matched raw arm;
- they replace 44 curated UFGS rows of the same answer types, chosen among the
  rows the matched arm also dropped, in a fixed hash order;
- issuers, answer mix per issuer and size equal the curated arm's, checked.

Writes data/build/single-rule/train.jsonl and manifest.json.
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.sample import _authority, _key, answer_type  # noqa: E402

RULE = "R-UFGS-3.2.7.1-material"
REJECTED = {"R-UFGS-2.12.10.1-material", "R-UFGS-3.2.7.1-material", "R-UFGS-3.2.7.2-material"}
SAMPLE_SEED = 20261114
ORDER_SEED = 20261111


def digest(rows):
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest()


def build(curated, matched):
    add = [r for r in matched if r["meta"]["requirement"] == RULE]
    assert add, "matched arm has no rows of the rule"
    matched_ids = {r["id"] for r in matched}
    need = collections.Counter(answer_type(r) for r in add)
    dropped = set()
    for kind, n in need.items():
        cands = [r for r in curated if _authority(r) == "UFGS" and answer_type(r) == kind and r["id"] not in matched_ids]
        cands.sort(key=lambda r: _key(SAMPLE_SEED, "single-rule", r["id"]))
        assert len(cands) >= n, f"{kind}: need {n}, only {len(cands)} curated UFGS rows to swap"
        dropped |= {r["id"] for r in cands[:n]}
    out = [r for r in curated if r["id"] not in dropped] + add
    out.sort(key=lambda r: _key(SAMPLE_SEED, "order", r["id"]))
    random.Random(ORDER_SEED).shuffle(out)

    # The design, checked rather than assumed.
    cur_ids = {r["id"] for r in curated}
    assert len(out) == len(curated) == len({r["id"] for r in out})
    foreign = [r for r in out if r["id"] not in cur_ids]
    assert len(foreign) == len(add) and all(r["meta"]["requirement"] == RULE for r in foreign)
    assert not any(r["meta"]["requirement"] in REJECTED - {RULE} for r in out), "another rejected rule leaked in"
    for a in {_authority(r) for r in curated} | {_authority(r) for r in out}:
        mine = collections.Counter(answer_type(r) for r in out if _authority(r) == a)
        theirs = collections.Counter(answer_type(r) for r in curated if _authority(r) == a)
        assert mine == theirs, f"{a}: answer mix differs {mine} vs {theirs}"
    return out, sorted(dropped)


def main() -> None:
    curated = [json.loads(l) for l in open(ROOT / "data/build/curated/train.jsonl", encoding="utf-8")]
    matched = [json.loads(l) for l in open(ROOT / "data/build/raw-matched/train.jsonl", encoding="utf-8")]
    out, dropped = build(curated, matched)
    dest = ROOT / "data/build/single-rule"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    manifest = {"rows": len(out), "rows_sha256": digest(out), "curated_rows_sha256": digest(curated),
                "rule": RULE, "rule_rows": len(out) - len(set(r["id"] for r in out) & set(r["id"] for r in curated)),
                "replaced_curated_ids": dropped}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "replaced_curated_ids"}, indent=2))


if __name__ == "__main__":
    main()
