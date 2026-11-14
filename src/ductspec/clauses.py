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


def _heading_re(convention: Convention) -> re.Pattern:
    return re.compile(rf"^(\d+\.\d+(?:\.\d+)*){convention.heading_gap}(\S.*)$")


def strip_designer_notes(text: str, convention: Convention = UFGS) -> str:
    """Remove instructions addressed to the engineer editing the specification.

    UFGS delimits them with rules of asterisks, so the block between two rules
    can be dropped whole. VA indents them instead, with no closing delimiter, so
    a note runs from its marker until the next clause heading or PART line.
    """
    if convention.note_block is not None:
        parts = convention.note_block.split(text)
        return "\n".join(p for p in parts if not convention.note_marker.search(p))

    heading = _heading_re(convention)
    out, skipping = [], False
    for line in text.split("\n"):
        if convention.note_marker.search(line):
            skipping = True
            continue
        if skipping and (heading.match(line) or CLAUSE_NUM.match(line)
                         or PART.match(line.strip()) or LETTERED.match(line.strip())):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def toc_entries(text: str) -> dict[str, str]:
    """Clause ids the document lists in its own table of contents.

    This is the most reliable statement of what the clauses are, because the
    document makes it itself. It gives both the id and the title, and a body
    heading has to match both: a table value of "3.5" under a column headed
    "90" shares an id with a real clause but not its name.
    """
    start = text.find("TABLE OF CONTENTS")
    if start == -1:
        return {}
    body = list(re.finditer(r"^PART\s+1\s*[-\u2013]?\s*GENERAL\s*$", text, re.M))
    end = body[-1].start() if len(body) > 1 else len(text)
    found = re.findall(r"^\s*(\d+\.\d+(?:\.\d+)*)\s+(\S.*?)\s*$", text[start:end], re.M)
    return {cid: title.strip() for cid, title in found}


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


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
    if convention.deferral is not None and convention.deferral.search(body):
        reasons.append("DEFERRED_TO_DESIGNER")
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
    toc = toc_entries("\n".join(t for _, t in pages))

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
    # Some owner specifications have no PART lines at all and number their
    # articles straight through ("1.1 SECTION INCLUDES", "1.4 COMPONENTS").
    # Without a PART to anchor the first digit, a heading is real when it
    # runs forward from the previous one.
    has_parts = any(PART.match(line.strip()) for line in lines)
    clauses: list[Clause] = []
    current_part = ""
    current_part_no = 0
    last_number: tuple[int, ...] = ()
    pending: tuple[str, str] | None = None   # (clause_id, title)
    buffer: list[str] = []

    def is_real_heading(clause_id: str, title: str) -> bool:
        """Reject numbers that are table data rather than clause headings.

        These specifications are full of figures that look like clause ids —
        "0.235", "401.5", "61.552" — and a run of them inside an insulation
        thickness table was producing clauses numbered after gauge values, with
        ids colliding with real ones. A citation to a clause that does not
        exist is the worst thing this dataset could contain.

        Where the document lists its own clauses, that list decides. An earlier
        attempt used the shape of the numbering instead — same PART, running
        forward — which removed the fabrications but also suppressed real
        clauses sitting after a table: a thickness value of "3.5" pushed the
        counter past the genuine 3.3 and 3.4. The document's own table of
        contents has no such failure mode.
        """
        if toc:
            raw = toc.get(clause_id)
            if raw is None:
                return False
            expected = _normalise(raw)
            actual = _normalise(title)
            if not actual:
                return False
            # A long body heading wraps across lines, so the body may carry
            # only the first words ("Pre Manufactured" for "Pre Manufactured
            # Double Wall Ductwork"). Either may be the prefix of the other.
            n = min(len(actual), len(expected), 12)
            return actual[:n] == expected[:n]
        try:
            number = tuple(int(x) for x in clause_id.split("."))
        except ValueError:
            return False
        if not has_parts:
            return number > last_number
        return number[0] == current_part_no and number > last_number

    def flush() -> None:
        if pending is None:
            return
        cid, title = pending
        text = "\n".join(buffer).strip()

        # A long heading wraps, and the parser takes only its first line. The
        # rest then sits at the top of the body, so clause 2.2.2 of the NASA
        # metal-ducts section came out titled "Round, High-Pres" with a body
        # beginning "sure, Double-Wall Sheet Metal Duct". A quote taken from
        # that body would carry a fragment of its own heading.
        #
        # Where the document lists the full title, use it, and remove whatever
        # part of it leaked into the body.
        full = toc.get(cid)
        if full and _normalise(full).startswith(_normalise(title)) and _normalise(full) != _normalise(title):
            leaked = len(_normalise(full)) - len(_normalise(title))
            kept, seen = [], 0
            for ch in text:
                if seen >= leaked:
                    kept.append(ch)
                elif ch.isalnum():
                    seen += 1
            text = "".join(kept).lstrip(" ,.;:-\n")
            title = full
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
            current_part_no = int(m.group(1))
            last_number = ()
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

        if heading and not is_real_heading(heading[0], heading[1]):
            heading = None   # a table figure, not a clause

        if heading:
            flush()
            pending, buffer = heading, []
            last_number = tuple(int(x) for x in heading[0].split("."))
        elif pending is not None:
            buffer.append(line)
        i += 1

    flush()
    if convention.lettered_subparagraphs:
        return [sub for c in clauses for sub in split_lettered(c)]
    return clauses


def to_dicts(clauses: list[Clause]) -> list[dict]:
    return [asdict(c) for c in clauses]
