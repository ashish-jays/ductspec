"""CuratorKIT filtering: what each gate removed, and what that did to training.

Run from the ductspec repo with PYTHONPATH=src, after build_arms --train-size 1200.
"""
import json
from collections import Counter
from pathlib import Path

from ductspec.curate import CURATED_GATES, RAW_GATES, to_samples
from ductspec.requirements import ROOT, load

reqs = load(ROOT / "data" / "requirements.raw.yaml")


def walk(gates, label):
    samples = to_samples(reqs)
    print(f"\n{label}: {len(samples)} extracted requirements in")
    removed = []
    for gate in gates:
        before = len(samples)
        samples, rejected = gate.run(samples)
        print(f"  {type(gate).__name__:22} {before:3} -> {len(samples):3}  (removed {len(rejected)})")
        for r in rejected:
            removed.append(r.metadata["requirement_id"])
            print(f"      - {r.metadata['requirement_id']:34} {r.rejection_reason}")
    print(f"  kept {len(samples)}")
    return set(removed)


raw_removed = walk(RAW_GATES, "RAW arm gates")
cur_removed = walk(CURATED_GATES, "CURATED arm gates")
only_curated = cur_removed - raw_removed
print(f"\nRemoved by curation but kept in raw: {len(only_curated)} requirements: {sorted(only_curated)}")

for arm in ("raw", "curated"):
    rows = [json.loads(l) for l in open(ROOT / f"data/build/{arm}/train.jsonl", encoding="utf-8")]
    from_removed = sum(r["meta"]["requirement"] in only_curated for r in rows)
    kinds = Counter(r["target"]["verdict"] if r["target"]["verdict"] != "INSUFFICIENT_EVIDENCE"
                    else r["target"]["missing_evidence"][0] for r in rows)
    print(f"{arm:8} train rows {len(rows)}; rows built on curation-removed requirements: {from_removed}")
    print(f"         answer mix: {dict(kinds.most_common())}")
