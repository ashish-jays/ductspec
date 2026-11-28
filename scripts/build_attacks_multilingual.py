"""Build the multilingual attack set from the frozen test set.

    PYTHONPATH=src python scripts/build_attacks_multilingual.py

Writes data/stage2/test_attacks_multilingual.jsonl.gz and its manifest. The
English attack set (data/stage2/test_attacks.jsonl.gz) is not touched.
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.attacks_multilingual import build  # noqa: E402

FROZEN_TEST_SHA256 = "8893a128c159d089766e55b5501cb014a1fe66bf181b967976f31ad13f768560"
OUT = ROOT / "data/stage2"


def main() -> None:
    raw = (ROOT / "data/build/curated/test.jsonl").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != FROZEN_TEST_SHA256:
        raise SystemExit("data/build/curated/test.jsonl is not the frozen test set")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    out = build(rows)
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in out).encode("utf-8")
    with open(OUT / "test_attacks_multilingual.jsonl.gz", "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(body)
    manifest = {
        "source_sha256": FROZEN_TEST_SHA256,
        "attack_set_sha256": hashlib.sha256(body).hexdigest(),
        "rows": len(out),
        "conditions": dict(collections.Counter(r["meta"]["attack"]["condition"] for r in out)),
    }
    (OUT / "attacks_multilingual_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
