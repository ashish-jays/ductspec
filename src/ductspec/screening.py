"""Shortlist the clauses that could state a checkable requirement.

Of 438 operative clauses, most are scope, submittal procedure, warranty or
execution prose. Only a minority state something a duct property can be
compared against. This screen is deliberately generous — it is a funnel for
the teacher to read, not a judgement about what a requirement is. Precision
comes from the teacher; this step only keeps the reading list honest.
"""

from __future__ import annotations

import re

from .clauses import Clause

# Language that tends to accompany a binding requirement in a US federal spec.
OBLIGATION = re.compile(
    r"\b(shall|must|is required to|are required to|provide|furnish|conform to|"
    r"comply with|in accordance with|not less than|minimum of|at least)\b",
    re.I,
)

PROPERTY_CUES: dict[str, re.Pattern] = {
    "material": re.compile(
        r"\b(galvaniz\w*|stainless|aluminum|aluminium|black iron|carbon steel|sheet metal|"
        r"g90|g60|gauge|gage|\bga\b)\b",
        re.I,
    ),
    "pressure_rating_in_wg": re.compile(
        r"(pressure\s+class|static\s+pressure|pressure\s+rating|in(?:ch|\.)?\s*(?:es)?\s*w\.?\s*g|"
        r"water\s+gauge|water\s+gage|\bpascal|\bpa\b)",
        re.I,
    ),
    "insulation_thickness_in": re.compile(
        r"\b(insulat\w*|duct\s+liner|liner|thermal\s+resistance|r-?value|thickness)\b",
        re.I,
    ),
}


def candidate_properties(clause: Clause) -> list[str]:
    """Which properties this clause might constrain. Empty means skip it."""
    if not clause.operative or not OBLIGATION.search(clause.text):
        return []
    return [prop for prop, cue in PROPERTY_CUES.items() if cue.search(clause.text)]


def shortlist(clauses: list[Clause], max_tokens_hint: int = 2000) -> list[tuple[Clause, list[str]]]:
    out = []
    for c in clauses:
        props = candidate_properties(c)
        # A clause longer than the hint is a whole article, not a requirement;
        # the teacher would be guessing which sentence binds.
        if props and len(c.text) < max_tokens_hint * 4:
            out.append((c, props))
    return out
