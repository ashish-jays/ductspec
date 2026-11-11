"""Curate the requirement extractions, through CuratorKIT's gate machinery.

The ablation this feeds is the experiment: two adapters trained on datasets
that differ only in whether the requirements behind them were audited.

The raw arm is not a degraded copy written to create an effect. It is what the
extraction actually produced before anything was checked against the PDF page,
errors and all — a requirement cited to a clause id that does not exist, one
cited to a choice the designer has not made, two scoped wider than the clause
they came from. Two of those came from bugs in our own parser. Both files look
equally confident; only opening the document tells them apart.

CuratorKIT supplies the pipeline: samples carrying provenance, gates that must
account for everything they drop ("silent drops are bugs"), rejection reasons
that survive into the output, and trainer-ready exports. The gates below supply
the contracts, because what makes a construction requirement citable is not a
general property of text.
"""

from __future__ import annotations

from pathlib import Path

from curatorkit.interfaces import BaseGate, DataSample, RejectedSample

from .clauses import parse_clauses
from .conventions import BY_NAME
from .requirements import Requirement, load

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources"


def to_samples(requirements: list[Requirement]) -> list[DataSample]:
    """A requirement becomes a curatable sample carrying its own evidence."""
    return [
        DataSample(
            source_uri=f"{r.source}#{r.clause_id}",
            instruction=f"State the {r.property} requirement that clause {r.clause_id} imposes.",
            input=r.quote,
            output=f"{r.predicate}",
            task_type="instruction_following",
            metadata={
                "requirement_id": r.id,
                "source": r.source,
                "clause_id": r.clause_id,
                "page": r.page,
                "property": r.property,
                "element": r.element,
                "systems": list(r.systems),
                "condition": r.condition,
                "predicate": r.predicate,
                "quote": r.quote,
            },
        )
        for r in requirements
    ]


def _reject(sample: DataSample, reason: str, step: str) -> RejectedSample:
    return RejectedSample(
        **sample.model_dump(exclude={"rejection_reason", "rejecting_step", "diagnosis"}),
        rejection_reason=reason,
        rejecting_step=step,
    )


class ClauseExistsGate(BaseGate):
    """The cited clause must exist in the document it is cited to.

    Caught the VA liner requirement, which was recorded against clause 2.2.B.
    There is no 2.2.B in that section: a bug in the designer-note stripper had
    deleted the heading above it, and its paragraphs inherited the previous
    article's numbering.
    """

    name = "clause_exists"

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, object]] = {}

    def _clauses(self, source: str, convention: str):
        if source not in self._cache:
            pdf = SOURCES / f"{source}.pdf"
            if not pdf.exists():
                self._cache[source] = {}
            else:
                self._cache[source] = {
                    c.clause_id: c for c in parse_clauses(pdf, source, BY_NAME[convention])
                }
        return self._cache[source]

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            source = s.metadata["source"]
            clauses = self._clauses(source, "va" if source.startswith("VA-") else "ufgs")
            if not clauses:                       # corpus not fetched; cannot judge
                passed.append(s)
            elif s.metadata["clause_id"] in clauses:
                passed.append(s)
            else:
                rejected.append(_reject(
                    s, f"clause_not_found:{source}#{s.metadata['clause_id']}", self.name))
        return passed, rejected


class OperativeClauseGate(BaseGate):
    """A requirement resting on an unmade choice is not yet binding.

    Caught the grease-duct material requirement, cited to UFGS 3.2.7.1, which
    offers 18 gauge stainless or 16 gauge carbon steel and asks the engineer to
    pick. Until they do, nothing is binding.

    The test is on the quoted span, not the whole article. UFGS 3.2.11 opens
    with an unconditional cross-reference and then offers the engineer a choice
    about outdoor air intakes; rejecting the article would discard a sentence
    that binds because a later one does not. A clause is a container, and
    adoption is a property of the sentence that states the requirement.
    """

    name = "clause_operative"

    def __init__(self) -> None:
        self._gate = ClauseExistsGate()

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            source = s.metadata["source"]
            convention = BY_NAME["va" if source.startswith("VA-") else "ufgs"]
            clauses = self._gate._clauses(source, convention.name)
            clause = clauses.get(s.metadata["clause_id"]) if clauses else None
            quote = s.metadata["quote"]
            if clause is None:
                passed.append(s)
                continue
            problems = []
            if convention.alternatives.search(quote):
                problems.append("UNRESOLVED_ALTERNATIVES")
            if convention.blank.search(quote):
                problems.append("UNFILLED_BLANK")
            if problems:
                rejected.append(_reject(
                    s, f"requirement_not_adopted:{','.join(problems)}", self.name))
            else:
                passed.append(s)
        return passed, rejected


class QuoteFidelityGate(BaseGate):
    """The quote must actually appear in the clause it is attributed to.

    An extraction that paraphrases has stopped being checkable, and a reviewer
    cannot confirm it against the page.
    """

    name = "quote_fidelity"

    def __init__(self, min_overlap: float = 0.8) -> None:
        self.min_overlap = min_overlap
        self._gate = ClauseExistsGate()

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            source = s.metadata["source"]
            clauses = self._gate._clauses(source, "va" if source.startswith("VA-") else "ufgs")
            clause = clauses.get(s.metadata["clause_id"]) if clauses else None
            if clause is None:
                passed.append(s)
                continue
            body = " ".join(clause.text.split()).lower()
            quote = " ".join(s.metadata["quote"].split()).lower()
            words = quote.split()
            hits = sum(1 for w in words if w in body)
            if words and hits / len(words) >= self.min_overlap:
                passed.append(s)
            else:
                rejected.append(_reject(
                    s, f"quote_not_in_clause:{hits}/{len(words)}_words", self.name))
        return passed, rejected


class InheritedScopeGate(BaseGate):
    """A clause inherits its parent heading's subject.

    UFGS 3.2.7.2 "Exposed Ductwork" sits under "3.2.7 Ducts Conveying Smoke and
    Grease Laden Vapors". Recording it as a rule for every system turns a
    kitchen-exhaust requirement into a universal one.
    """

    name = "inherited_scope"
    KITCHEN_PARENTS = ("grease", "kitchen")

    def __init__(self) -> None:
        self._gate = ClauseExistsGate()

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            source = s.metadata["source"]
            clauses = self._gate._clauses(source, "va" if source.startswith("VA-") else "ufgs")
            parent_id = s.metadata["clause_id"].rsplit(".", 1)[0]
            parent = clauses.get(parent_id) if clauses else None
            if parent is None:
                passed.append(s)
                continue
            title = parent.title.lower()
            narrow = any(word in title for word in self.KITCHEN_PARENTS)
            if narrow and not s.metadata["systems"]:
                rejected.append(_reject(
                    s, f"scope_wider_than_parent:{parent_id}_{parent.title[:40]!r}", self.name))
            else:
                passed.append(s)
        return passed, rejected


RAW_GATES: list[BaseGate] = []
# Order matters for the reason a rejection carries. A requirement cited to the
# wrong clause id may still land on a real clause — the VA liner rule cited to
# 2.2.B resolves to the pressure-class paragraph — and would then be rejected
# for whatever happens to be wrong with that clause. Checking the quote first
# names the actual fault.
CURATED_GATES: list[BaseGate] = [
    ClauseExistsGate(), QuoteFidelityGate(), InheritedScopeGate(), OperativeClauseGate(),
]


def curate(requirements: list[Requirement], gates: list[BaseGate]):
    """Run one arm. Returns (kept requirements, rejections)."""
    samples = to_samples(requirements)
    all_rejected: list[RejectedSample] = []
    for gate in gates:
        samples, rejected = gate.run(samples)
        all_rejected.extend(rejected)

    kept_ids = {s.metadata["requirement_id"] for s in samples}
    return [r for r in requirements if r.id in kept_ids], all_rejected


def arms(raw_path: Path, audited_path: Path):
    """Both arms, from the two requirement files."""
    raw_kept, raw_rejected = curate(load(raw_path), RAW_GATES)
    cur_kept, cur_rejected = curate(load(raw_path), CURATED_GATES)
    return {
        "raw": {"requirements": raw_kept, "rejected": raw_rejected},
        "curated": {"requirements": cur_kept, "rejected": cur_rejected},
        "audited": {"requirements": load(audited_path), "rejected": []},
    }
