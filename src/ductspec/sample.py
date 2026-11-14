"""Choose a training set of a given size from everything the generator can make.

The generator produces far more rows than one training run can use, and the
rows it produces are not evenly spread: a requirement with many failing values
yields many non-compliant rows, a clause offering a choice yields only
"applicability unknown" rows, and so on. Training on whatever comes first
teaches whatever happens to be common. Last time that was refusal: the curated
arm was 29% "applicability unknown" and refused 42% of answerable questions.

So the training set is chosen, by rules stated here and applied identically to
both arms:

1. A fixed answer mix (ANSWER_MIX), decided from what the task needs, not
   measured from the test set. "Applicability unknown" is capped at 20%.
2. Within each answer type, coverage before repetition: every clause family
   gets a row before any gets a second, then every requirement within it, then
   every duct system and drawing sheet. A clause is the unit of diversity,
   not a requirement: four pressure classes listed in one Dartmouth sentence
   are one example family, not four.
3. Deterministic for a given seed.

If an answer type has fewer rows than its share, all of them are taken and the
shortfall is spread over the other types in proportion to their shares, so the
set still reaches the requested size whenever enough rows exist.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

# Share of each answer type in a training set. Definitive answers and
# abstentions roughly balance (46% / 54%); among abstentions the copied value is
# the safety-critical case and gets the largest share; "applicability unknown"
# is capped so the model does not learn refusal as the default.
ANSWER_MIX = {
    "COMPLIANT": 0.24,
    "NON_COMPLIANT": 0.22,
    "INFERRED_VALUE_ONLY": 0.22,
    "APPLICABILITY_UNKNOWN": 0.20,
    "MISSING_OBSERVATION": 0.06,
    "MISSING_AUTHORITY": 0.06,
}


def answer_type(row: dict) -> str:
    target = row["target"]
    if target["verdict"] != "INSUFFICIENT_EVIDENCE":
        return target["verdict"]
    return target["missing_evidence"][0]


def _requirement(row: dict) -> str:
    return row["meta"]["requirement"]


def _authority(row: dict) -> str:
    """The issuer, from the requirement id: R-UFGS-..., U-DART-..., R-OFCC-..."""
    return _requirement(row).split("-")[1]


def _sheet(row: dict) -> str:
    return row["input"]["drawing"][0]["id"].split("-")[0]


def _system(row: dict) -> str:
    return row["input"]["object"]["system"]


def _key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()


def _quotas(available: dict[str, int], n: int) -> dict[str, int]:
    """Rows per answer type: shares of n, capped by what exists, shortfall
    spread over the types that still have rows."""
    quota = {k: 0 for k in ANSWER_MIX}
    remaining, open_types = n, [k for k in ANSWER_MIX if available.get(k, 0)]
    while remaining > 0 and open_types:
        weight = sum(ANSWER_MIX[k] for k in open_types)
        grant = {k: max(1, round(remaining * ANSWER_MIX[k] / weight)) for k in open_types}
        progressed = False
        for k in open_types:
            take = min(grant[k], available[k] - quota[k], remaining)
            if take > 0:
                quota[k] += take
                remaining -= take
                progressed = True
            if remaining == 0:
                break
        open_types = [k for k in open_types if quota[k] < available[k]]
        if not progressed:
            break
    return quota


def _clause(row: dict) -> tuple[str, str]:
    """(issuer, clause id) from the requirement id. Several requirements can
    come from one clause (Dartmouth 3.2.A lists four pressure classes); they
    are one family of examples, and share one turn."""
    parts = _requirement(row).split("-")
    return parts[1], parts[2]


class _Rotation:
    """Take one item at a time from nested groups, moving to the next group at
    every level after each take, and dropping groups as they empty."""

    def __init__(self, groups: dict, seed: int, path: tuple = ()):
        self.leaf = not isinstance(groups, dict)
        if self.leaf:
            self.items = sorted(groups, key=lambda r: _key(seed, r["id"]))
            return
        order = sorted(groups, key=lambda g: _key(seed, *map(str, path), str(g)))
        self.children = [_Rotation(groups[g], seed, path + (g,)) for g in order]
        self.at = 0

    def empty(self) -> bool:
        return not self.items if self.leaf else not self.children

    def take(self) -> dict:
        if self.leaf:
            return self.items.pop(0)
        child = self.children[self.at]
        item = child.take()
        if child.empty():
            self.children.pop(self.at)
        else:
            self.at += 1
        if self.children:
            self.at %= len(self.children)
        return item


def _spread(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Take `count` rows, rotating clause family, then requirement, then duct
    system and drawing sheet, so none of them gets a second row before every
    one at its level has had a first."""
    tree: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        tree[_clause(r)][_requirement(r)][(_system(r), _sheet(r))].append(r)
    plain = {c: {q: dict(leaves) for q, leaves in reqs.items()} for c, reqs in tree.items()}
    rotation = _Rotation(plain, seed)
    picked: list[dict] = []
    while len(picked) < count and not rotation.empty():
        picked.append(rotation.take())
    return picked


def balanced(rows: list[dict], n: int, seed: int = 20261114) -> list[dict]:
    """A training set of n rows chosen by the rules in this module's docstring."""
    if n <= 0:
        raise ValueError("n must be positive")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        kind = answer_type(r)
        if kind not in ANSWER_MIX:
            raise ValueError(f"{r['id']}: answer type {kind!r} has no share in ANSWER_MIX")
        by_type[kind].append(r)
    if n > len(rows):
        raise ValueError(f"asked for {n} rows, only {len(rows)} exist")
    quota = _quotas({k: len(v) for k, v in by_type.items()}, n)
    out: list[dict] = []
    for kind in ANSWER_MIX:
        out += _spread(by_type[kind], quota[kind], seed)
    out.sort(key=lambda r: _key(seed, "order", r["id"]))
    if len(out) != n:
        raise AssertionError(f"chose {len(out)} rows, wanted {n}")
    return out
