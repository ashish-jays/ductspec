"""Compute the verdict. Nothing here is predicted.

Given a duct, a property and the evidence actually supplied, this returns the
answer a careful reviewer owes — the verdict, the clauses that govern it, the
observation relied on, and where it falls short, which kind of short and who
to ask. Every label in the dataset comes from here, so the dataset contains no
model's opinion about what the right answer is. The one judgement upstream is
reading a requirement out of a clause, and each of those carries the quote it
came from so it can be checked against the page.

The order of the questions is the order a reviewer asks them:

    1. Is there an authority here that settles this question?
    2. Does it reach this object?
    3. Does the drawing independently show the property?
    4. Only then: compare.

You cannot ask whether the drawing shows enough before knowing what you are
checking against, and you cannot compare a value to a requirement that does
not apply. Taking the questions out of order is how a reviewer ends up
certifying against the wrong clause.
"""

from __future__ import annotations

from dataclasses import dataclass

from .objects import DuctObject
from .requirements import Requirement


# What a reviewer owes when the specification states the requirement and the
# drawing does not show the property. FAR 52.236-21(a): "Anything mentioned in
# the specifications and not shown on the drawings ... shall be of like effect
# as if shown or mentioned in both." A silent drawing is therefore not a
# discrepancy and needs no written determination: the specification applies,
# and the value is recorded as specification-derived rather than as something
# the drawing confirmed. Only a difference between the two goes to an RFI.
SILENT_DRAWING_ACTION = "FOLLOW_SPEC_RECORD_AS_SPEC_DERIVED"


@dataclass(frozen=True)
class Packet:
    """What was actually put in front of the reviewer."""
    requirements: tuple[Requirement, ...]
    sources: tuple[str, ...]      # documents supplied, for resolving references
    policy_id: str


@dataclass(frozen=True)
class Derivation:
    verdict: str
    governing_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    action: str
    reason: str
    # Which of the governing clauses were actually not met. A non-compliant
    # duct is still governed by every applicable clause — citing only the
    # failed one would teach the model that the others stopped applying.
    failed_ids: tuple[str, ...] = ()
    # Evaluation-only. Never shown to the model; it exists so the scorer can
    # tell a right answer from a right answer that happened to match. Without
    # it, a definitive guess on an unanswerable case looks supported whenever
    # its citations line up.
    answer_available: bool = False


def _abstain(code: str, action: str, reason: str, governing=()) -> Derivation:
    return Derivation(
        "INSUFFICIENT_EVIDENCE", tuple(governing), (), (code,), action, reason,
        answer_available=False,
    )


def derive(duct: DuctObject, prop: str, packet: Packet) -> Derivation:
    # 1. Authority. Which supplied requirements govern this property, for this
    #    kind of object, in this system?
    candidates = [
        r for r in packet.requirements
        if r.property == prop and r.element == duct.element and r.applies_to_system(duct.system)
    ]
    if not candidates:
        return _abstain(
            "MISSING_AUTHORITY", "REQUEST_APPLICABILITY_CLARIFICATION",
            "No supplied clause states a requirement for this property on this object.",
        )

    # A clause that only points elsewhere settles nothing by itself. It becomes
    # MISSING_AUTHORITY only if nothing else supplied settles the question —
    # an unresolved pointer alongside an applicable rule is not a dead end.
    concrete = [r for r in candidates if not r.is_reference]
    unresolved = [
        r for r in candidates
        if r.is_reference and r.predicate["target"] not in packet.sources
    ]
    if not concrete:
        if unresolved:
            r = unresolved[0]
            return _abstain(
                "MISSING_AUTHORITY", "REQUEST_MISSING_DETAIL",
                f"The governing clause defers to {r.predicate['target']}, "
                "which is not among the supplied documents.",
                governing=[r.clause_id],
            )
        return _abstain(
            "MISSING_AUTHORITY", "REQUEST_APPLICABILITY_CLARIFICATION",
            "The supplied clauses state no requirement for this property.",
        )

    # 2. Applicability. Some requirements bind only under a condition the
    #    drawing has to establish — VA 2.3.B binds "where indicated on
    #    drawings". If the object does not say, we cannot know whether the
    #    requirement reaches it, and guessing either way is a fabrication.
    for r in concrete:
        if r.condition and r.condition not in duct.facts:
            return _abstain(
                "APPLICABILITY_UNKNOWN", "REQUEST_APPLICABILITY_CLARIFICATION",
                f"Clause {r.clause_id} binds only where {r.condition.replace('_', ' ')}, "
                "and the drawing does not establish whether it does.",
                governing=[r.clause_id],
            )
    governing = [r for r in concrete if not r.condition or duct.facts.get(r.condition)]
    if not governing:
        return _abstain(
            "MISSING_AUTHORITY", "REQUEST_APPLICABILITY_CLARIFICATION",
            "Every supplied requirement for this property is conditional, and the "
            "conditions do not hold for this object.",
        )

    # 3. Evidence. Only now does what the drawing shows become relevant.
    observed = duct.properties.get(prop)
    cites = [r.clause_id for r in governing]
    if observed is None or observed.value is None:
        return _abstain(
            "MISSING_OBSERVATION", SILENT_DRAWING_ACTION,
            "The requirement is known, but the drawing does not show this property.",
            governing=cites,
        )
    if not observed.is_independent:
        return _abstain(
            "INFERRED_VALUE_ONLY", SILENT_DRAWING_ACTION,
            "The only value available was not read from the drawing, so it cannot "
            "show that the drawing complies.",
            governing=cites,
        )

    # 4. Compare. Where several requirements apply, all must be met; a failure
    #    is cited to the clauses that were not met, not to all of them.
    failed = [r for r in governing if not r.satisfied_by(observed.value)]
    if failed:
        return Derivation(
            "NON_COMPLIANT", tuple(cites), (observed.source_id,), (),
            "RFI_DRAWING_SPEC_DISCREPANCY",
            "The value shown on the drawing does not meet "
            + _name(failed) + ".",
            failed_ids=tuple(r.clause_id for r in failed),
            answer_available=True,
        )
    return Derivation(
        "COMPLIANT", tuple(cites), (observed.source_id,), (),
        "NONE_FOR_THIS_CHECK",
        "The value shown on the drawing satisfies " + _name(governing) + ".",
        answer_available=True,
    )


def _name(reqs) -> str:
    ids = [r.clause_id for r in reqs]
    if len(ids) == 1:
        return f"clause {ids[0]}"
    return "clauses " + ", ".join(ids[:-1]) + f" and {ids[-1]}"
