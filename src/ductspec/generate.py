"""Build the dataset.

Every record is a packet put in front of a reviewer and the answer the
evidence actually supports. The answer is never written by hand and never
produced by a model: it comes out of derive(), which reads the requirement and
the object. What this module chooses is the *situation* — which duct, which
clauses are in the packet, where each value came from — and the situations are
chosen to cover the ways a reviewer can be wrong.

Scenarios come in families. A family shares a requirement and a duct, and
varies one thing at a time, so its members are counterfactual twins of each
other. Families never straddle a split: if a model saw one member in training,
seeing its twin in the test set would measure recall rather than reasoning.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from .derive import Packet, derive
from .objects import (
    DuctObject,
    Property,
    copied_from_spec,
    engineering_default,
    observed,
    scheduled,
)
from .requirements import Requirement, load

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "build"

SYSTEM_FOR = {
    "kitchen_exhaust": "kitchen_exhaust",
    "lab_exhaust": "lab_exhaust",
}

# Values that satisfy and values that do not, per property. Chosen to sit on
# both sides of each requirement's threshold, including on it.
MATERIALS = ("galvanized_steel", "stainless_steel_304", "stainless_steel_316",
             "aluminum", "black_iron")
THICKNESSES = (0.5, 1.0, 1.5, 2.0, 3.0)
PRESSURES = (1.0, 1.5, 2.0, 3.0, 4.0)

VALUES = {
    "material": MATERIALS,
    "insulation_thickness_in": THICKNESSES,
    "pressure_rating_in_wg": PRESSURES,
}


DISTRACTOR_FOR: dict[str, Requirement] = {}


@dataclass
class Scenario:
    """One situation, plus the variations that make it a counterfactual family."""
    family: str
    requirement: Requirement
    duct: DuctObject
    packet: Packet
    prop: str


# A drawing context: which sheet, which run, what size. One requirement checked
# on four different runs is four families, not one, which gives the split
# enough granularity to balance without ever breaking a family apart.
CONTEXTS = (
    {"sheet": "M101", "w": 20, "h": 12, "system": "supply_air"},
    {"sheet": "M102", "w": 36, "h": 18, "system": "return_air"},
    {"sheet": "M201", "w": 14, "h": 10, "system": "general_exhaust"},
    {"sheet": "M601", "w": 48, "h": 24, "system": "supply_air"},
)


def _duct(n: int, req: Requirement, value, origin: str, facts=None, ctx=None) -> DuctObject:
    ctx = ctx or CONTEXTS[0]
    system = req.systems[0] if req.systems else ctx["system"]
    prop: Property
    if origin == "observed":
        prop = observed(value, f"D-{n}", sheet=ctx["sheet"], note=n % 9 + 1,
                        text=f"D-{n}: {ctx['w']}x{ctx['h']}")
    elif origin == "scheduled":
        prop = scheduled(value, f"D-{n}", sheet=ctx["sheet"])
    elif origin == "copied":
        prop = copied_from_spec(value, req.clause_id)
    elif origin == "default":
        prop = engineering_default(value, "house standard: 2 inch minimum")
    else:
        raise ValueError(origin)
    return DuctObject(
        f"D-{n}", system, ctx["w"], ctx["h"], {req.property: prop},
        element=req.element, facts=dict(facts or {}),
    )


def _record(rid: str, family: str, scenario: Scenario, split: str, panel: str) -> dict:
    d = derive(scenario.duct, scenario.prop, scenario.packet)
    evidence = [
        {"id": r.clause_id, "source": r.source, "page": r.page, "text": r.quote}
        for r in scenario.packet.requirements
    ]
    return {
        "id": rid,
        "family": family,
        "split": split,
        "panel": panel,
        "input": {
            "question": f"Does {scenario.duct.id}'s {scenario.prop.replace('_', ' ')} "
                        "meet the supplied requirement?",
            "object": scenario.duct.as_dict(),
            "evidence": evidence,
            "documents_supplied": list(scenario.packet.sources),
            "policy": {
                "id": scenario.packet.policy_id,
                "text": "Review only the requested property, using only the supplied "
                        "evidence. A value that was not read from the drawing does not "
                        "establish what the drawing shows.",
            },
        },
        "target": {
            "verdict": d.verdict,
            "scope": scenario.prop,
            "governing_ids": list(d.governing_ids),
            "observation_ids": list(d.observation_ids),
            "failed_ids": list(d.failed_ids),
            "missing_evidence": list(d.missing_evidence),
            "action": d.action,
            "reason": d.reason,
        },
        # Never shown to the model.
        "meta": {"answer_available": d.answer_available, "requirement": scenario.requirement.id},
    }


def reference_family(req: Requirement, n: int, target: Requirement | None) -> list[Scenario]:
    """A clause that states no requirement, only a pointer to another document.

    The interesting pair is whether the pointed-at document is in the packet.
    Both members name the same section, so a model cannot shortcut this by
    noticing a document id — it has to notice whether the document is there.
    """
    fam = f"F-{req.id}"
    value = THICKNESSES[-1]
    absent = Packet((req,), (req.source,), "POLICY-1")
    out = [
        # The pointer, and the section it points at is not supplied.
        Scenario(fam, req, _duct(n, req, value, "observed"), absent, req.property),
        Scenario(fam, req, _duct(n + 1, req, value, "copied"), absent, req.property),
    ]
    if target is not None:
        # Same pointer, and now the section it points at is in the packet.
        present = Packet((req, target), (req.source, target.source), "POLICY-1")
        passing = next((v for v in THICKNESSES if target.satisfied_by(v)), THICKNESSES[-1])
        failing = next((v for v in THICKNESSES if not target.satisfied_by(v)), THICKNESSES[0])
        facts = {target.condition: True} if target.condition else {}
        out += [
            Scenario(fam, target, _duct(n + 2, target, passing, "observed", facts), present, req.property),
            Scenario(fam, target, _duct(n + 3, target, failing, "observed", facts), present, req.property),
            Scenario(fam, target, _duct(n + 4, target, passing, "copied", facts), present, req.property),
        ]
    return out


def family_for(req: Requirement, n: int, sources: tuple[str, ...], ctx=None) -> list[Scenario]:
    """One requirement on one run becomes a family that differs by one thing."""
    ctx = ctx or CONTEXTS[0]
    pkt = Packet(requirements=(req,), sources=sources, policy_id="POLICY-1")
    values = VALUES[req.property]
    passing = next((v for v in values if req.satisfied_by(v)), values[-1])
    failing = next((v for v in values if not req.satisfied_by(v)), values[0])
    facts = {req.condition: True} if req.condition else {}
    fam = f"F-{req.id}-{ctx['sheet']}"

    ok = [v for v in values if req.satisfied_by(v)]
    bad = [v for v in values if not req.satisfied_by(v)]
    out: list[Scenario] = []
    k = n

    def add(duct, packet=pkt):
        nonlocal k
        out.append(Scenario(fam, req, duct, packet, req.property))
        k += 1

    # Comparison, on both sides of the threshold and on it.
    for v in ok:
        add(_duct(k, req, v, "observed", facts, ctx))
        add(_duct(k, req, v, "scheduled", facts, ctx))
    for v in bad:
        add(_duct(k, req, v, "observed", facts, ctx))

    # The same values, back-filled. Identical numbers, no longer evidence.
    for v in ok[:2] + bad[:1]:
        add(_duct(k, req, v, "copied", facts, ctx))
        add(_duct(k, req, v, "default", facts, ctx))

    # The drawing says nothing.
    add(DuctObject(f"D-{k}", req.systems[0] if req.systems else ctx["system"],
                   ctx["w"], ctx["h"], {}, element=req.element, facts=facts))

    # Nothing supplied governs this property.
    add(_duct(k, req, ok[0] if ok else values[0], "observed", facts, ctx),
        Packet((), sources, "POLICY-1"))

    # A real, authoritative clause about a different element. Must not move
    # the answer, and must not become the citation.
    other = DISTRACTOR_FOR.get(req.element)
    if other is not None:
        with_distractor = Packet((req, other), sources + (other.source,), "POLICY-1")
        for v in (ok[0] if ok else values[0], bad[0] if bad else values[-1]):
            add(_duct(k, req, v, "observed", facts, ctx), with_distractor)

    # A condition the drawing has not established.
    if req.condition:
        for v in (ok[0] if ok else values[0],):
            add(_duct(k, req, v, "observed", facts={}, ctx=ctx))

    return out


def build(seed: int = 20261111, requirements: list[Requirement] | None = None) -> dict[str, list[dict]]:
    """Build the splits from a requirement set.

    The arms differ only here. Same generator, same seed, same situations per
    requirement — what changes is which requirements were considered sound.
    """
    rng = random.Random(seed)
    requirements = requirements if requirements is not None else load()

    # One authoritative clause per element, used as an inapplicable distractor
    # in other families.
    DISTRACTOR_FOR.clear()
    for r in requirements:
        if not r.is_reference and r.source.startswith("UFGS"):
            DISTRACTOR_FOR.setdefault(r.element, r)
    for element in list(DISTRACTOR_FOR):
        other = next((r for e, r in DISTRACTOR_FOR.items() if e != element), None)
        if other is not None:
            DISTRACTOR_FOR[element] = other

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    families: list = []
    n = 100
    for req in requirements:
        held_out = req.source.startswith("VA-")
        panel = "transfer" if held_out else "known_authority"
        sources = (req.source,) if not req.is_reference else (req.source,)

        for ctx in (CONTEXTS if not req.is_reference else CONTEXTS[:2]):
          if req.is_reference:
            target = next(
                (t for t in requirements
                 if t.source == req.predicate["target"] and t.property == req.property
                 and not t.is_reference),
                None,
            )
            scenarios = reference_family(req, n, target)
          else:
            scenarios = family_for(req, n, sources, ctx)

          # The split is chosen once per family, not per record. Members of a
          # family differ by one controlled change, so a twin in the test set
          # whose sibling was trained on measures recall, not reasoning.
          if held_out:
            split = "test"
          else:
            families.append((scenarios, panel, req, ctx))
            n += 20
            continue

          for i, scenario in enumerate(scenarios):
            rid = f"{req.id}-{ctx['sheet']}-{i:02d}"
            splits[split].append(_record(rid, scenario.family, scenario, split, panel))
          n += 20
    # A family's split is decided by its own name, not by its position in the
    # list. Which requirements are present changes the length of this list, and
    # an ablation where a family lands in train for one arm and dev for the
    # other is measuring the shuffle, not the curation.
    for scenarios, panel, req, ctx in families:
        bucket = int(hashlib.sha256(f"{seed}:{scenarios[0].family}".encode()).hexdigest(), 16) % 7
        split = "train" if bucket < 5 else ("dev" if bucket == 5 else "test")
        for i, scenario in enumerate(scenarios):
            rid = f"{req.id}-{ctx['sheet']}-{i:02d}"
            splits[split].append(_record(rid, scenario.family, scenario, split, panel))

    return splits


def write(splits: dict[str, list[dict]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"{split:6} {len(rows):4} records  sha256:{digest}  -> {path.relative_to(ROOT)}")


def build_arms(seed: int = 20261111) -> dict[str, dict[str, list[dict]]]:
    """The two training arms, and the shared evaluation set.

    Both arms train on data generated from their own requirement set. Both are
    scored on the SAME test set, built from the audited requirements — the
    evaluation cannot depend on which arm is being judged, or the comparison
    measures nothing.
    """
    from .curate import CURATED_GATES, RAW_GATES, curate

    raw_reqs, _ = curate(load(ROOT / "data" / "requirements.raw.yaml"), RAW_GATES)
    cur_reqs, _ = curate(load(ROOT / "data" / "requirements.raw.yaml"), CURATED_GATES)
    audited = load()

    shared = build(requirements=audited)
    raw_train = build(requirements=raw_reqs)["train"]
    cur_train = build(requirements=cur_reqs)["train"]

    # Match the arms on size. The raw set contains more requirements, so it
    # generates more records, and an adapter trained on more tokens would look
    # better or worse for a reason that has nothing to do with curation. Both
    # arms are truncated to the smaller count, deterministically, after a
    # shuffle seeded identically — so the difference between them is which
    # requirements were considered sound, and nothing else.
    n = min(len(raw_train), len(cur_train))
    for rows in (raw_train, cur_train):
        random.Random(seed).shuffle(rows)

    return {
        "raw": {"train": raw_train[:n], "dev": shared["dev"], "test": shared["test"]},
        "curated": {"train": cur_train[:n], "dev": shared["dev"], "test": shared["test"]},
    }


def write_arms(arms: dict[str, dict[str, list[dict]]]) -> None:
    for arm, splits in arms.items():
        out = OUT / arm
        out.mkdir(parents=True, exist_ok=True)
        for split, rows in splits.items():
            path = out / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            print(f"{arm:8} {split:6} {len(rows):4} records  sha256:{digest}")


if __name__ == "__main__":
    splits = build()
    write(splits)
    from collections import Counter
    for split, rows in splits.items():
        c = Counter(r["target"]["verdict"] for r in rows)
        m = Counter(code for r in rows for code in r["target"]["missing_evidence"])
        print(f"  {split}: {dict(c)}  reasons={dict(m)}")
