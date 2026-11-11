"""Duct objects, and the provenance machinery that makes them interesting.

A duct object stands in for what a drawing reader produced from one sheet. Each
property carries not just a value but where the value came from, because that is
the whole question: a 25 mm insulation thickness read off a callout and a 25 mm
thickness back-filled from the specification are indistinguishable as numbers
and completely different as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import INDEPENDENT_ORIGINS, ORIGINS, PROPERTIES

SYSTEMS = ("supply_air", "return_air", "general_exhaust", "kitchen_exhaust", "lab_exhaust")


@dataclass
class Property:
    value: object
    origin: str
    source_id: str = ""
    source_text: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin {self.origin!r}")

    @property
    def is_independent(self) -> bool:
        """True when the drawing itself establishes this value."""
        return self.origin in INDEPENDENT_ORIGINS

    def as_dict(self) -> dict:
        d = {"value": self.value, "origin": self.origin}
        if self.source_id:
            d["source_id"] = self.source_id
        if self.source_text:
            d["source_text"] = self.source_text
        return d


@dataclass
class DuctObject:
    id: str
    system: str
    width_in: int
    height_in: int
    properties: dict[str, Property] = field(default_factory=dict)
    # What this object is. Requirements are written about particular things —
    # a casing, a liner, an underground run — and a rule about one is not a
    # rule about another.
    element: str = "duct"
    # Applicability facts the drawing establishes, e.g.
    # {"lining_indicated_on_drawings": True}. A fact that is absent is unknown,
    # which is a different thing from false.
    facts: dict[str, bool] = field(default_factory=dict)
    # The sheet this duct is drawn on. Drawing line ids are built from it.
    sheet: str = "M101"

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}")
        for name in self.properties:
            if name not in PROPERTIES:
                raise ValueError(f"unknown property {name!r}")

    @property
    def largest_dimension_in(self) -> int:
        return max(self.width_in, self.height_in)

    def callout(self) -> str:
        """The annotation a drafter would letter beside the duct."""
        abbrev = {
            "supply_air": "SA",
            "return_air": "RA",
            "general_exhaust": "EA",
            "kitchen_exhaust": "KEF",
            "lab_exhaust": "LEF",
        }[self.system]
        parts = [f"{self.id}: {self.width_in}x{self.height_in}"]
        material = self.properties.get("material")
        if material is not None and material.is_independent:
            parts.append(_material_abbrev(material.value))
        parts.append(abbrev)
        return " ".join(parts)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "system": self.system,
            "dimensions_in": [self.width_in, self.height_in],
            "element": self.element,
            "properties": {k: v.as_dict() for k, v in self.properties.items()},
            **({"facts": self.facts} if self.facts else {}),
        }


def _material_abbrev(material: object) -> str:
    return {
        "galvanized_steel": "GALV",
        "stainless_steel_304": "SST304",
        "stainless_steel_316": "SST316",
        "black_iron": "BI",
        "aluminum": "ALUM",
    }.get(str(material), str(material).upper())


def observed(value: object, duct_id: str, sheet: str = "M101", note: int = 1, text: str = "") -> Property:
    """A value the drawing shows: a callout or note on a sheet."""
    return Property(
        value=value,
        origin="DRAWING_ANNOTATION",
        source_id=f"{sheet}-note{note}",
        source_text=text,
    )


def scheduled(value: object, duct_id: str, sheet: str = "M601") -> Property:
    """A value from a schedule table on a sheet — also independent evidence."""
    return Property(
        value=value,
        origin="DRAWING_SCHEDULE",
        source_id=f"{sheet}-duct-schedule-{duct_id}",
        source_text=f"{duct_id} | {value}",
    )


def copied_from_spec(value: object, clause_id: str) -> Property:
    """A gap filled from the requirement being checked. Not evidence."""
    return Property(value=value, origin="COPIED_FROM_SPEC", source_id=clause_id)


def engineering_default(value: object, rule: str) -> Property:
    """A gap filled from a house or industry default. Not evidence."""
    return Property(value=value, origin="ENGINEERING_DEFAULT", source_id=rule)
