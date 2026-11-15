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

import re
from pathlib import Path

from curatorkit.interfaces import BaseGate, DataSample, RejectedSample
from pypdf import PdfReader

from .clauses import Clause, parse_clauses, toc_entries
from .conventions import BY_NAME, Convention
from .requirements import Requirement, load
from .sources import convention_of

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


class CorpusMissing(RuntimeError):
    """A source document the gates need is absent or unreadable.

    Every gate here judges a requirement against the page it cites. Without the
    document there is nothing to judge against, and a gate that passed what it
    could not check would hand back the raw arm labelled as curated.
    """


class Corpus:
    """The parsed source documents, read once and shared by the gates."""

    def __init__(self) -> None:
        self._clauses: dict[str, dict[str, Clause]] = {}
        self._headings: dict[str, dict[str, str]] = {}

    @staticmethod
    def convention(source: str) -> Convention:
        return BY_NAME[convention_of(source)]

    def _pdf(self, source: str) -> Path:
        pdf = SOURCES / f"{source}.pdf"
        if not pdf.exists():
            raise CorpusMissing(
                f"{pdf.name} is not in {SOURCES}. Fetch the corpus with "
                f"`python -m ductspec.sources` before curating.")
        return pdf

    def clauses(self, source: str) -> dict[str, Clause]:
        if source not in self._clauses:
            parsed = parse_clauses(self._pdf(source), source, self.convention(source))
            if not parsed:
                raise CorpusMissing(f"{source}: the document parsed to no clauses.")
            self._clauses[source] = {c.clause_id: c for c in parsed}
        return self._clauses[source]

    def headings(self, source: str) -> dict[str, str]:
        """Heading titles from the document's own table of contents.

        The parser keeps only clauses that have a body. A heading that exists
        to group the clauses under it, "3.2.7 Kitchen Exhaust Ductwork", has
        none, and is listed nowhere else.
        """
        if source not in self._headings:
            reader = PdfReader(self._pdf(source))
            self._headings[source] = toc_entries(
                "\n".join(page.extract_text() or "" for page in reader.pages))
        return self._headings[source]

    def lineage(self, source: str, clause_id: str) -> list[tuple[str, str]] | None:
        """The clause's own heading and every heading above it, nearest first.

        Returns None when a heading above it cannot be found. A document
        without a table of contents numbers its articles two deep and letters
        the paragraphs, and a lettered paragraph already carries its article's
        title, so there is nothing above it to look for.
        """
        clauses, headings = self.clauses(source), self.headings(source)
        out = [(clause_id, clauses[clause_id].title)]
        parts = clause_id.split(".")
        for depth in range(len(parts) - 1, 1, -1):
            hid = ".".join(parts[:depth])
            if hid in headings:
                out.append((hid, headings[hid]))
            elif hid in clauses:
                out.append((hid, clauses[hid].title))
            elif headings or not self.convention(source).lettered_subparagraphs:
                return None
        return out


CORPUS = Corpus()


def _not_found(sample: DataSample, step: str) -> RejectedSample:
    meta = sample.metadata
    return _reject(sample, f"clause_not_found:{meta['source']}#{meta['clause_id']}", step)


class ClauseExistsGate(BaseGate):
    """The cited clause must exist in the document it is cited to.

    Caught the VA liner requirement, which was recorded against clause 2.2.B.
    There is no 2.2.B in that section: a bug in the designer-note stripper had
    deleted the heading above it, and its paragraphs inherited the previous
    article's numbering.
    """

    name = "clause_exists"

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            if s.metadata["clause_id"] in CORPUS.clauses(s.metadata["source"]):
                passed.append(s)
            else:
                rejected.append(_not_found(s, self.name))
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

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            source = s.metadata["source"]
            if s.metadata["clause_id"] not in CORPUS.clauses(source):
                rejected.append(_not_found(s, self.name))
                continue
            convention = CORPUS.convention(source)
            quote = s.metadata["quote"]
            problems = []
            if convention.alternatives.search(quote):
                problems.append("UNRESOLVED_ALTERNATIVES")
            if convention.blank.search(quote):
                problems.append("UNFILLED_BLANK")
            if convention.deferral is not None and convention.deferral.search(quote):
                problems.append("DEFERRED_TO_DESIGNER")
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

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            clause = CORPUS.clauses(s.metadata["source"]).get(s.metadata["clause_id"])
            if clause is None:
                rejected.append(_not_found(s, self.name))
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
    """A requirement reaches no further than the headings it sits under.

    UFGS 3.2.7.2 "Exposed Ductwork" sits under "3.2.7 Kitchen Exhaust
    Ductwork". Recorded for every system, a kitchen-exhaust rule becomes a
    universal one. UFGS 2.12.10.1 sits under "2.12.10 Plenums and Casings for
    Field-Fabricated Units". Recorded against ducts, a casing gauge becomes a
    duct material.

    The gate knows the subjects listed below and compares only those. Each is a
    word a heading uses and the scope a requirement under it must carry.
    """

    name = "inherited_scope"
    # heading word -> the only system a requirement under it may list
    SYSTEM_WORDS = {"kitchen": "kitchen_exhaust", "grease": "kitchen_exhaust"}
    # heading word -> the elements a requirement under it may be recorded against
    COMPONENT_WORDS = {"plenum": {"plenum_or_casing"}, "casing": {"plenum_or_casing"}}

    def _fault(self, sample: DataSample, lineage: list[tuple[str, str]]) -> str | None:
        systems, element = sample.metadata["systems"], sample.metadata["element"]
        for hid, title in lineage:
            words = set(re.findall(r"[a-z]+", title.lower()))
            words |= {w[:-1] for w in words if w.endswith("s")}
            where = f"{hid}_{title[:40]!r}"
            for word, system in self.SYSTEM_WORDS.items():
                # Listing the right system among others is still wider than
                # the heading; an empty list means every system.
                if word in words and set(systems) != {system}:
                    return f"scope_wider_than_heading:{where}:needs_system={system}"
            for word, allowed in self.COMPONENT_WORDS.items():
                if word in words and element not in allowed:
                    return f"scope_wider_than_heading:{where}:element={element}"
        return None

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            source, clause_id = s.metadata["source"], s.metadata["clause_id"]
            if clause_id not in CORPUS.clauses(source):
                rejected.append(_not_found(s, self.name))
                continue
            lineage = CORPUS.lineage(source, clause_id)
            if lineage is None:
                rejected.append(_reject(
                    s, f"heading_above_not_found:{source}#{clause_id}", self.name))
                continue
            fault = self._fault(s, lineage)
            if fault:
                rejected.append(_reject(s, fault, self.name))
            else:
                passed.append(s)
        return passed, rejected


RAW_GATES: list[BaseGate] = []
# Order matters for the reason a rejection carries. A requirement cited to the
# wrong clause id may still land on a real clause — the VA liner rule cited to
# 2.2.B resolves to the pressure-class paragraph — and would then be rejected
# for whatever happens to be wrong with that clause. Checking the quote first
# names the actual fault. Adoption comes before scope for the same reason:
# UFGS 3.2.7.1 is both an unmade choice and recorded too widely, and a choice
# nobody made has no scope to get wrong.
CURATED_GATES: list[BaseGate] = [
    ClauseExistsGate(), QuoteFidelityGate(), OperativeClauseGate(), InheritedScopeGate(),
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
