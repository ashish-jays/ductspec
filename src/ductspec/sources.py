"""Fetch and freeze the public UFGS corpus.

Every downloaded file is hashed and the hash recorded, so a reviewer can
confirm they are reading the same bytes we trained and evaluated on. UFGS
sections are revised periodically; without the hash, "UFGS 23 30 00" is not a
reproducible reference.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "sources.yaml"
PDF_DIR = ROOT / "data" / "sources"
LOCKFILE = ROOT / "data" / "sources.lock.json"

# wbdg.org rejects the default urllib agent on some paths.
_UA = "ductspec/0.1 (hackathon research; +https://github.com/ashish-jays/ductspec)"


def load_manifest() -> list[dict]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["sources"]


SPLITS = ("train", "train_only", "validation", "holdout")


def registry() -> dict[str, dict]:
    """Every source by id: its issuer, editing convention, licence and split."""
    out = {}
    for s in load_manifest():
        if s["id"] in out:
            raise ValueError(f"{s['id']} is listed twice in {MANIFEST.name}")
        if s.get("split") not in SPLITS:
            raise ValueError(f"{s['id']}: split must be one of {SPLITS}, not {s.get('split')!r}")
        out[s["id"]] = s
    return out


def convention_of(source_id: str) -> str:
    """The editing convention a source uses; UFGS where the manifest says none."""
    return registry()[source_id].get("convention", "ufgs")


def split_of(source_id: str) -> str:
    """train (hashed into train, dev and test by family), train_only (training
    rows only), validation (an authority held out for choosing recipes) or
    holdout (the VA test authority)."""
    return registry()[source_id]["split"]


class CorpusChanged(RuntimeError):
    """A source no longer matches the hash the dataset was built from."""


def verify() -> dict[str, dict]:
    """Check every local PDF against the lockfile. Call before building data.

    UFGS and VA sections are revised in place, so a silent re-download would
    change the evidence under a dataset that is already labelled.
    """
    lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))
    for sid, meta in lock.items():
        path = PDF_DIR / f"{sid}.pdf"
        if not path.exists():
            raise CorpusChanged(f"{sid}: missing — run `python -m ductspec.sources`")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != meta["sha256"]:
            raise CorpusChanged(
                f"{sid}: sha256 {actual[:12]} does not match the locked "
                f"{meta['sha256'][:12]}. The published section was revised; "
                "re-audit the requirements before rebuilding."
            )
    return lock


def fetch_all(force: bool = False) -> dict[str, dict]:
    """Download every source, hash it, and write data/sources.lock.json."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    lock: dict[str, dict] = {}

    for src in load_manifest():
        dest = PDF_DIR / f"{src['id']}.pdf"
        if force or not dest.exists():
            req = urllib.request.Request(src["url"], headers={"User-Agent": _UA})
            with urllib.request.urlopen(req) as resp:
                dest.write_bytes(resp.read())

        data = dest.read_bytes()
        lock[src["id"]] = {
            "title": src["title"],
            "url": src["url"],
            "split": src["split"],
            "convention": src.get("convention", "ufgs"),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    LOCKFILE.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


if __name__ == "__main__":
    for sid, meta in fetch_all().items():
        print(f"{sid:22} {meta['split']:8} {meta['convention']:5} {meta['bytes']:>8}B  "
              f"{meta['sha256'][:12]}  {meta['title']}")
