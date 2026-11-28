"""Build the Stage 2 attack set from the frozen test set.

    PYTHONPATH=src python scripts/build_attacks.py

Reads data/build/curated/test.jsonl (built by scripts/build_arms.py), checks it
is the frozen 333-row set by hash, and writes
  data/stage2/test_attacks.jsonl.gz   every condition of every case
  data/stage2/attacks_manifest.json   counts and hashes
The build is deterministic; running it twice gives byte-identical output.
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.attacks import CONDITIONS, build  # noqa: E402

TEST = ROOT / "data/build/curated/test.jsonl"
# sha256 of the frozen test set the Stage 1 results were scored on, with line
# endings normalised to \n (a Windows build writes \r\n; the rows are the same).
FROZEN_TEST_SHA256 = "8893a128c159d089766e55b5501cb014a1fe66bf181b967976f31ad13f768560"
OUT = ROOT / "data/stage2"


def main() -> None:
    raw = TEST.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_TEST_SHA256:
        raise SystemExit(f"{TEST} is not the frozen test set (sha256 {digest})")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    attacked = build(rows)

    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in attacked).encode("utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    # mtime=0 and no file name inside the gzip header, so the archive bytes are reproducible.
    with open(OUT / "test_attacks.jsonl.gz", "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(body)

    counts = collections.Counter(r["meta"]["attack"]["condition"] for r in attacked)
    should_not = collections.Counter(r["meta"]["attack"]["condition"] for r in attacked
                                     if r["target"]["verdict"] != "COMPLIANT")
    manifest = {
        "source": "data/build/curated/test.jsonl",
        "source_sha256": digest,
        "attack_set_sha256": hashlib.sha256(body).hexdigest(),
        "rows": len(attacked),
        "conditions": {c: {"rows": counts[c], "should_not_approve": should_not[c]} for c in CONDITIONS},
    }
    (OUT / "attacks_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
