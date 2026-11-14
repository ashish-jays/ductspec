"""Checks that nothing scored can be learned from.

Each check returns the offending row ids, so a failure names what leaked. They
are run by tests/test_leakage.py on every build, and by scripts/check_leakage.py
on a built dataset.

What is checked:
  family_in_one_split      a family (one requirement on one run) is never split
                           across train, dev and test: its members differ by one
                           controlled change, so a trained sibling gives the
                           answer away.
  holdout_never_trained    no row of the held-out VA authority is in train or dev.
  validation_never_trained no row of the validation authority is in train.
  probe_only_in_test       the curation probe is only ever scored.
  train_only_never_scored  training-only sources and clauses never reach dev or test.
  no_identical_prompt      no training prompt is word for word a dev or test prompt.

What is reported but allowed: clauses of a "train" authority appear in both
train and test (different families, different ducts). That is the
known-authority panel, deliberately separate from the transfer panel.
"""

from __future__ import annotations

from collections import defaultdict

from .generate import PROBE_PANEL
from .prompt import render
from .requirements import load, load_unadopted, train_only_unadopted, validation_unadopted
from .sources import split_of


def _source_of() -> dict[str, str]:
    out = {r.id: r.source for r in load()}
    out.update({r.id: r.source for r, _ in load_unadopted()})
    from .requirements import ROOT
    out.update({r.id: r.source for r in load(ROOT / "data" / "requirements.raw.yaml")})
    return out


def check(splits: dict[str, list[dict]]) -> dict[str, list[str]]:
    source = _source_of()
    train_only_clauses = train_only_unadopted()
    validation_clauses = validation_unadopted()
    problems: dict[str, list[str]] = defaultdict(list)

    where: dict[str, set] = defaultdict(set)
    for split, rows in splits.items():
        for r in rows:
            where[r["family"]].add(split)
    problems["family_in_one_split"] = sorted(f for f, s in where.items() if len(s) > 1)

    for split, rows in splits.items():
        for r in rows:
            req = r["meta"]["requirement"]
            kind = split_of(source[req]) if req in source else None
            if kind == "holdout" and split != "test":
                problems["holdout_never_trained"].append(r["id"])
            if (kind == "validation" or req in validation_clauses) and split == "train":
                problems["validation_never_trained"].append(r["id"])
            if r["panel"] == PROBE_PANEL and split != "test":
                problems["probe_only_in_test"].append(r["id"])
            # A clause explicitly held out for validation is meant to be scored,
            # even when its source is otherwise training-only.
            if (split != "train" and req not in validation_clauses
                    and (kind == "train_only" or req in train_only_clauses)):
                problems["train_only_never_scored"].append(r["id"])

    scored = {render(r["input"]) for s in ("dev", "test") for r in splits.get(s, [])}
    problems["no_identical_prompt"] = [r["id"] for r in splits.get("train", []) if render(r["input"]) in scored]
    return {k: v for k, v in problems.items() if v}


def shared_clauses(splits: dict[str, list[dict]]) -> dict[str, int]:
    """Clauses seen in both train and test: allowed, and counted for the report."""
    clauses = defaultdict(set)
    for split, rows in splits.items():
        for r in rows:
            for e in r["input"]["evidence"]:
                clauses[(e.get("source"), e["id"])].add(split)
    return {f"{s} {c}": 1 for (s, c), where in clauses.items() if {"train", "test"} <= where}
