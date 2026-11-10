"""Turn a UFGS PDF into clause records with provenance.

Two things here are not generic PDF chunking, and both exist because a guide
specification is not the same thing as a project specification:

1. **Designer notes are stripped.** UFGS sections carry instructions to the
   engineer who edits them, bracketed by rules of asterisks and beginning
   "NOTE:". They are advice about how to write the spec, not requirements
   binding on anybody. Training a model to cite one as an authority would
   teach it to quote a comment as law.

2. **Clauses are marked operative or not.** UFGS ships with blanks (`[_____]`)
   and unresolved alternatives (`[ASME A13.1] [MIL-STD-101]`) for the designer
   to choose between. Until somebody chooses, there is no requirement to comply
   with. Only operative clauses may be used as governing evidence; the rest are
   useful precisely as material for APPLICABILITY_UNKNOWN cases.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from pypdf import PdfReader

from .conventions import BY_NAME, UFGS, Convention

LETTERED = re.compile(r"^([A-Z])\.\s+(\S.*)$")
CLAUSE_NUM = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s*$")
# Built per convention in parse_clauses, since the heading gap differs.
CLAUSE_INLINE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s{2,}(\S.*)$")
PART = re.compile(r"^PART\s+(\d)\s*[-\u2013]?\s*(\w.*)$")
BLANK = re.compile(r"\[_+\]")
# Two bracketed options offered side by side. The cap was 60 characters and
# let through UFGS 23 30 00 clause 3.2.7.1, whose alternatives are
# "[minimum 1.3 mm 18 gauge , Type 304L or 316L, stainless steel]" and
# "[minimum 1.6 mm 16 gauge carbon steel]" — a real unresolved material choice
# on grease-laden exhaust duct. A spec writer's option is not a requirement.
ALTERNATIVES = re.compile(r"\[[^\[\]]{2,200}\]\s*\[[^\[\]]{2,200}\]")
# A single bracketed option, e.g. "galvanized steel [or aluminum]". Whether it
# matters depends on where it sits: a bracket around the material destroys the
# requirement, a bracket in "Install ductwork[ as indicated]" does not. Too
# context-dependent for a regex, so it is recorded as a flag and the extraction
# step decides per clause, with the quoted span in hand.
OPTIONAL_TEXT = re.compile(r"\[[^\[\]]{2,200}\]")


@dataclass
class Clause:
    source_id: str
    clause_id: str
    title: str
    part: str
    text: str
    page_start: int
    operative: bool
    non_operative_reasons: list[str]
    has_optional_text: bool
    convention: str = "ufgs"


def strip_designer_notes(text: str, convention: Convention = UFGS) -> str:
    """Remove instructions addressed to the engineer editing the specification.

    UFGS delimits them with rules of asterisks, so the block between two rules
    can be dropped whole. VA indents them instead, with no closing delimiter, so
    a note runs from its marker until the next clause heading or PART line.
    """
    if convention.note_block is not None:
        parts = convention.note_block.split(text)
        return "\n".join(p for p in parts if not convention.note_marker.search(p))

    out, skipping = [], False
    for line in text.split("\n"):
        if convention.note_marker.search(line):
            skipping = True
            continue
        if skipping and (CLAUSE_INLINE.match(line) or CLAUSE_NUM.match(line)
                         or PART.match(line.strip()) or LETTERED.match(line.strip())):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def _body_only(text: str) -> str:
    """Skip the cover page and table of contents.

    The body is whatever follows the LAST 'PART 1   GENERAL' — the first one is
    the table-of-contents entry for it.
    """
    matches = list(re.finditer(r"^PART\s+1\s*[-\u2013]?\s*GENERAL\s*$", text, re.M))
    return text[matches[-1].start():] if matches else text


def _page_index(reader: PdfReader) -> list[tuple[int, str]]:
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _why_not_operative(body: str, convention: Convention) -> list[str]:
    reasons = []
    if convention.blank.search(body):
        reasons.append("UNFILLED_BLANK")
    if convention.alternatives.search(body):
        reasons.append("UNRESOLVED_ALTERNATIVES")
    return reasons


def split_lettered(clause: Clause) -> list[Clause]:
    """Split a VA clause into its lettered subparagraphs, so a citation can name
    2.1.A rather than the whole article."""
    lines = clause.text.split("\n")
    marks = [(i, m.group(1)) for i, line in enumerate(lines) if (m := LETTERED.match(line.strip()))]
    if not marks:
        return [clause]

    out = []
    for n, (start, letter) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        reasons = _why_not_operative(body, BY_NAME[clause.convention])
        out.append(replace(
            clause,
            clause_id=f"{clause.clause_id}.{letter}",
            text=body,
            operative=not reasons,
            non_operative_reasons=reasons,
            has_optional_text=bool(BY_NAME[clause.convention].optional.search(body)),
        ))
    return out or [clause]


def parse_clauses(pdf_path: Path, source_id: str, convention: Convention = UFGS) -> list[Clause]:
    reader = PdfReader(pdf_path)
    pages = _page_index(reader)
    clause_inline = re.compile(rf"^(\d+\.\d+(?:\.\d+)*){convention.heading_gap}(\S.*)$")

    full = "\n".join(t for _, t in pages)
    body = _body_only(strip_designer_notes(full, convention))

    # Map a clause id to the page a reviewer should open. The id appears twice
    # — once in the table of contents, once in the body — so take the LAST page
    # that starts a line with it, which is the body occurrence.
    def body_page(clause_id: str) -> int:
        pattern = re.compile(rf"^{re.escape(clause_id)}(?:\s|$)", re.M)
        found = [num for num, text in pages if pattern.search(text)]
        return found[-1] if found else 0

    lines = body.split("\n")
    clauses: list[Clause] = []
    current_part = ""
    pending: tuple[str, str] | None = None   # (clause_id, title)
    buffer: list[str] = []

    def flush() -> None:
        if pending is None:
            return
        cid, title = pending
        text = "\n".join(buffer).strip()
        if not text:
            return
        reasons = _why_not_operative(text, convention)
        clauses.append(
            Clause(
                source_id=source_id,
                clause_id=cid,
                title=title.strip(),
                part=current_part,
                text=text,
                page_start=body_page(cid),
                operative=not reasons,
                non_operative_reasons=reasons,
                has_optional_text=bool(convention.optional.search(text)),
                convention=convention.name,
            )
        )

    i = 0
    while i < len(lines):
        line = lines[i]

        if m := PART.match(line.strip()):
            flush()
            pending, buffer = None, []
            current_part = f"PART {m.group(1)} {m.group(2).strip()}"
            i += 1
            continue

        heading = None
        if m := clause_inline.match(line):
            heading = (m.group(1), m.group(2))
        elif m := CLAUSE_NUM.match(line):
            # Title sits on the next non-empty line.
            nxt = next((lines[j] for j in range(i + 1, min(i + 4, len(lines))) if lines[j].strip()), "")
            if nxt and not CLAUSE_NUM.match(nxt):
                heading = (m.group(1), nxt)
                i += 1

        if heading:
            flush()
            pending, buffer = heading, []
        elif pending is not None:
            buffer.append(line)
        i += 1

    flush()
    if convention.lettered_subparagraphs:
        return [sub for c in clauses for sub in split_lettered(c)]
    return clauses


def to_dicts(clauses: list[Clause]) -> list[dict]:
    return [asdict(c) for c in clauses]
