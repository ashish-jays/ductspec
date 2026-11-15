"""Load the audited requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "data" / "requirements.yaml"
UNADOPTED = ROOT / "data" / "unadopted.yaml"


@dataclass(frozen=True)
class Requirement:
    id: str
    source: str
    clause_id: str
    page: int
    property: str
    predicate: dict
    quote: str
    systems: tuple[str, ...] = ()          # empty means every system
    element: str = "duct"
    condition: str = ""                    # a fact the object must establish
    note: str = ""

    @property
    def is_reference(self) -> bool:
        """States no requirement of its own; points at another document."""
        return self.predicate["op"] == "reference"

    @property
    def is_unadopted(self) -> bool:
        """Offers the designer alternatives and records no choice between them."""
        return self.predicate["op"] == "unadopted"

    def applies_to_system(self, system: str) -> bool:
        return not self.systems or system in self.systems

    def satisfied_by(self, value) -> bool:
        op = self.predicate["op"]
        if op == "in":
            return value in self.predicate["values"]
        if op == "at_least":
            return float(value) >= float(self.predicate["value"])
        raise ValueError(f"cannot compare with predicate {op!r}")


def load(path: Path = REQUIREMENTS) -> list[Requirement]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["requirements"]
    out = []
    for r in raw:
        applies = r.get("applies_to", {})
        out.append(Requirement(
            id=r["id"], source=r["source"], clause_id=r["clause_id"], page=r["page"],
            property=r["property"], predicate=r["predicate"], quote=r["quote"],
            systems=tuple(applies.get("systems") or ()),
            element=applies.get("element", "duct"),
            condition=applies.get("condition", ""),
            note=r.get("note", ""),
        ))
    return out


REVIEW_QUEUE = ROOT / "data" / "review_queue.yaml"
# A reading a second annotator disputes, or a reviewer rejects, is not used for
# anything until a person settles it.
HELD_BACK_STATUSES = ("disputed", "rejected")


def held_back(path: Path = REVIEW_QUEUE) -> set[str]:
    """Ids of extracted items that must produce no rows."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["entries"]
    return {e["id"] for e in entries if e["status"] in HELD_BACK_STATUSES}


def validation_unadopted(path: Path = UNADOPTED) -> set[str]:
    """Ids of unmade choices held out for validation, never trained on."""
    return {r["id"] for r in yaml.safe_load(path.read_text(encoding="utf-8"))["unadopted"] if r.get("validation")}


def train_only_unadopted(path: Path = UNADOPTED) -> set[str]:
    """Ids of unmade choices that are only ever trained on."""
    return {r["id"] for r in yaml.safe_load(path.read_text(encoding="utf-8"))["unadopted"] if r.get("train_only")}


def load_unadopted(path: Path = UNADOPTED) -> list[tuple[Requirement, bool]]:
    """Clauses stating an unmade choice, each with whether it is a probe.

    They load as Requirement so a packet can show them as evidence, with a
    predicate that derive() reads as "nothing has been required yet".
    """
    out = []
    for r in yaml.safe_load(path.read_text(encoding="utf-8"))["unadopted"]:
        applies = r.get("applies_to", {})
        out.append((Requirement(
            id=r["id"], source=r["source"], clause_id=r["clause_id"], page=r["page"],
            property=r["property"],
            predicate={"op": "unadopted", "drawn_values": list(r["drawn_values"])},
            quote=r["quote"],
            systems=tuple(applies.get("systems") or ()),
            element=applies.get("element", "duct"),
        ), bool(r["probe"])))
    return out
