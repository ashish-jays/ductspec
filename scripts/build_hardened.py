"""Build the hardened curated training set: the curated arm plus attacked training copies.

    PYTHONPATH=src python scripts/build_hardened.py          # v1: 120 should-not-approve + 60 compliant copies
    PYTHONPATH=src python scripts/build_hardened.py v2       # v2: 90 + 90

Writes data/build/curated-hardened[-v2]/train.jsonl (the 1,200 curated rows,
unchanged, plus the attacked copies from ductspec.harden) and a manifest.
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.harden import build  # noqa: E402

CURATED_SHA256 = "f19e087a56f88abbaf0ca30bd8d134788e704568a235158951526572d6ee0c66"
VARIANTS = {"v1": ("curated-hardened", 120, 60), "v2": ("curated-hardened-v2", 90, 90)}


def load(path):
    return [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]


def digest(rows):
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest()


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    folder, n_refuse, n_comply = VARIANTS[variant]
    curated = load("data/build/curated/train.jsonl")
    assert digest(curated) == CURATED_SHA256, "not the Stage 1 curated training rows"
    test = load("data/build/curated/test.jsonl")
    every = curated + load("data/build/raw/train.jsonl") + load("data/build/curated/dev.jsonl") + test
    clause_ids = {e["id"] for r in every for e in r["input"]["evidence"]}
    extra = build(curated, clause_ids, {r["id"] for r in test}, n_refuse=n_refuse, n_comply=n_comply)
    out = curated + extra
    random.Random(20261122).shuffle(out)
    assert len({r["id"] for r in out}) == len(out)

    dest = ROOT / "data/build" / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    manifest = {
        "variant": variant, "rows": len(out), "curated_rows": len(curated), "attacked_rows": len(extra),
        "rows_sha256": digest(out), "curated_rows_sha256": CURATED_SHA256,
        "attacked_by_condition": dict(collections.Counter(r["meta"]["harden"]["condition"] for r in extra)),
        "attacked_by_verdict": dict(collections.Counter(r["target"]["verdict"] for r in extra)),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
