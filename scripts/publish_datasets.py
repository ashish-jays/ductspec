"""Publish the datasets to Hugging Face, public and un-gated.

The hackathon requires every artifact on the Hub — models, adapters, logs,
datasets — in public collections. This handles the dataset half.

What goes up, and why each of them:

  train/dev/test per arm   the data itself
  requirements.yaml        the ten audited requirements, each with the quote,
                           source, clause id and page it was read from. This is
                           the only place a model's judgement enters the
                           dataset, so it is the thing a reviewer should check
  requirements.raw.yaml    the same extractions before audit, which is what the
                           raw arm was built from
  rejections.jsonl         what the curation gates dropped and why
  sources.lock.json        sha256 of every source PDF, because these sections
                           are revised in place and the designation alone is
                           not a reproducible reference
  baselines-test.txt       what the metrics do under an oracle, an
                           always-compliant model and an always-abstain one
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import HfApi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "data" / "build"
OWNER = os.environ.get("HF_OWNER", "ashishr10")
REPO = f"{OWNER}/ductspec-conformance"

CARD = """---
license: apache-2.0
task_categories:
  - question-answering
language:
  - en
tags:
  - construction
  - specifications
  - grounding
  - abstention
  - hvac
pretty_name: DuctSpec evidence-bound conformance review
---

# DuctSpec

Can a model refuse to certify compliance it cannot actually see?

A construction drawing shows some things and is silent about others. When a
review pipeline meets a silence it is convenient to fill the gap from the
specification, or from an engineering default, and carry on. If that filled-in
value is later checked against the specification it was copied from, the check
passes and has proved nothing.

Each record is one property check on one duct against supplied clauses. Every
property carries where its value came from. Only `DRAWING_ANNOTATION` and
`DRAWING_SCHEDULE` are independent evidence of what the drawing shows;
`COPIED_FROM_SPEC`, `ENGINEERING_DEFAULT` and `ESTIMATOR_ENTRY` are not.

## How the labels were made

No verdict here was written by hand or produced by a model. Each is computed
from a structured requirement and a structured object, in the order a reviewer
asks the questions: is there an authority that settles this, does it reach this
object, does the drawing independently show the property, and only then,
compare.

The one place a model's judgement enters is reading a requirement out of a
clause. Those are in `requirements.yaml`, each carrying the exact quote, source,
clause id and page, so they can be checked against the document.

## Corpus

UFGS (USACE / NAVFAC / AFCEC / NASA) for training, VA PG-18-1 held out. Both
are works of the US federal government, in the public domain. The holdout is a
different issuing authority on purpose: UFGS marks an unmade choice `[A] [B]`,
VA marks it `//A// //B//`. A model that learned "do not cite square brackets"
fails the holdout; one that learned "a choice nobody has made is not a
requirement" transfers.

No SMACNA, ASHRAE or other paywalled standard is reproduced. Where one is
relevant it is cited by designation only — which is itself a source of genuine
`MISSING_AUTHORITY` cases, since the spec points at a document that legitimately
cannot be in the packet.

## The two arms

`raw` and `curated` differ by exactly one requirement. Same generator, same
seed, same family-to-split assignment, same test set, same number of records.
The raw arm was built from the extraction before it was audited, and so
includes UFGS 3.2.7.1 — a clause offering the engineer two materials for
grease-laden duct, with nobody having chosen yet.

Counterfactual families never straddle a split.
"""


def main(dry_run: bool = True) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    files: list[tuple[Path, str]] = []

    for arm in ("raw", "curated"):
        for split in ("train", "dev", "test"):
            p = BUILD / arm / f"{split}.jsonl"
            if p.exists():
                files.append((p, f"{arm}/{split}.jsonl"))

    for extra, dest in (
        (ROOT / "data/requirements.yaml", "requirements/audited.yaml"),
        (ROOT / "data/requirements.raw.yaml", "requirements/raw.yaml"),
        (ROOT / "data/sources.lock.json", "provenance/sources.lock.json"),
        (ROOT / "data/controls.yaml", "provenance/controls.yaml"),
        (BUILD / "baselines-test.txt", "baselines/baselines-test.txt"),
    ):
        if extra.exists():
            files.append((extra, dest))

    print(f"repo: {REPO}  ({'DRY RUN' if dry_run else 'PUBLISHING'})")
    for path, dest in files:
        print(f"  {dest:38} {path.stat().st_size:>9,} B")
    if dry_run:
        print("\nre-run with --publish to upload")
        return

    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset")
    for path, dest in files:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                        repo_id=REPO, repo_type="dataset")
    print(f"\nhttps://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main(dry_run="--publish" not in sys.argv)
