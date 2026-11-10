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
    return yaml.safe_load(MANIFEST.read_text())["sources"]


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
