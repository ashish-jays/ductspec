"""Build the matched raw arm: the curated arm's mix, with the gates switched off.

    PYTHONPATH=src python scripts/build_matched_raw.py

The Stage 1 raw and curated arms differ in two ways: the curation gates, and
(because the sampler rotates over whatever requirements exist) the issuer mix,
raw 609/390/146/55 UFGS/Dartmouth/Houston/NYCHA against curated 500/464/169/67.
So a difference between them could be the gates or the mix.

This arm removes the second difference:
- every non-UFGS row is the curated arm's own row, unchanged. The gates also
  reject one non-UFGS requirement (R-VA-2.2.B-liner), but VA is not a training
  issuer, so the non-UFGS training pools of the two arms are identical (checked:
  1,815 rows each);
- the UFGS rows are drawn from the raw arm's UFGS pool, in exactly the
  curated arm's count per answer type, by the same selection rule
  (ductspec.sample._spread, same seed);
- the final order is made the way build_arms makes it.

What is left is the gates: the UFGS rows may come from the requirements the
gates rejected. The script reports how many do.

Writes data/build/raw-matched/train.jsonl and the conversational copy in
data/build/messages/raw-matched/train.jsonl.
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.curate import CURATED_GATES, RAW_GATES, curate  # noqa: E402
from ductspec.generate import build  # noqa: E402
from ductspec.prompt import to_messages  # noqa: E402
from ductspec.requirements import load  # noqa: E402
from ductspec.sample import _authority, _key, _spread, answer_type  # noqa: E402

SAMPLE_SEED = 20261114  # ductspec.sample.balanced default
ORDER_SEED = 20261111   # ductspec.generate.build_arms default


def digest(rows):
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest()


def main() -> None:
    curated = [json.loads(l) for l in open(ROOT / "data/build/curated/train.jsonl", encoding="utf-8")]
    assert len(curated) == 1200, len(curated)

    raw_reqs, _ = curate(load(ROOT / "data" / "requirements.raw.yaml"), RAW_GATES)
    cur_reqs, rejected = curate(load(ROOT / "data" / "requirements.raw.yaml"), CURATED_GATES)
    rejected_ids = {r.metadata["requirement_id"] for r in rejected}
    pool = build(requirements=raw_reqs)["train"]

    keep = [r for r in curated if _authority(r) != "UFGS"]
    want = collections.Counter(answer_type(r) for r in curated if _authority(r) == "UFGS")
    ufgs_pool = [r for r in pool if _authority(r) == "UFGS"]
    picked = []
    for kind, count in want.items():
        rows = [r for r in ufgs_pool if answer_type(r) == kind]
        chosen = _spread(rows, count, SAMPLE_SEED)
        assert len(chosen) == count, f"{kind}: wanted {count}, pool has {len(rows)}"
        picked += chosen

    out = keep + picked
    out.sort(key=lambda r: _key(SAMPLE_SEED, "order", r["id"]))
    random.Random(ORDER_SEED).shuffle(out)

    # The match, checked rather than assumed.
    assert len(out) == len(curated) == len({r["id"] for r in out})
    for a in {_authority(r) for r in curated} | {_authority(r) for r in out}:
        mine = collections.Counter(answer_type(r) for r in out if _authority(r) == a)
        theirs = collections.Counter(answer_type(r) for r in curated if _authority(r) == a)
        assert mine == theirs, f"{a}: answer mix differs {mine} vs {theirs}"
    assert [r for r in out if _authority(r) != "UFGS"] and \
        sorted(r["id"] for r in out if _authority(r) != "UFGS") == sorted(r["id"] for r in keep)

    from_rejected = collections.Counter(r["meta"]["requirement"] for r in out if r["meta"]["requirement"] in rejected_ids)
    reqs = collections.Counter(r["meta"]["requirement"] for r in out if _authority(r) == "UFGS")
    cur_reqs_used = collections.Counter(r["meta"]["requirement"] for r in curated if _authority(r) == "UFGS")
    print("issuers:", dict(collections.Counter(_authority(r) for r in out)))
    print(f"UFGS requirements used: {len(reqs)} (curated arm: {len(cur_reqs_used)})")
    print(f"rows from gate-rejected requirements: {sum(from_rejected.values())} of 1200")
    for rid, n in sorted(from_rejected.items()):
        print(f"  {rid:32} {n}")
    print(f"rows identical to the curated arm: {len({r['id'] for r in out} & {r['id'] for r in curated})}")

    dest = ROOT / "data/build/raw-matched"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    msg = ROOT / "data/build/messages/raw-matched"
    msg.mkdir(parents=True, exist_ok=True)
    (msg / "train.jsonl").write_text("".join(json.dumps(to_messages(r)) + "\n" for r in out), encoding="utf-8")
    manifest = {"rows": len(out), "rows_sha256": digest(out), "curated_rows_sha256": digest(curated),
                "from_rejected": dict(from_rejected)}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
