"""How a specification marks the parts that are not yet requirements.

Federal guide specifications ship half-edited: they carry instructions to the
engineer, choices the engineer has not made, and blanks nobody has filled.
Those passages are not binding on anyone, so they cannot be cited as governing
evidence — but each issuing authority marks them differently.

UFGS uses rules of asterisks around `NOTE:` blocks, `[_____]` for blanks and
`[A] [B]` for alternatives.

VA PG-18-1 uses indented `SPEC WRITER NOTE:` blocks and `//...//` delimiters,
and says so in its own first note: "Delete between //   // if not applicable
to project." The rule below is the document's own instruction, not our reading
of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Convention:
    name: str
    # Clause headings: "1.2.2.3" (UFGS) or "2.1" plus a lettered subparagraph (VA).
    lettered_subparagraphs: bool
    note_block: re.Pattern | None          # delimiter around designer notes
    note_marker: re.Pattern                # text identifying a note
    blank: re.Pattern
    alternatives: re.Pattern               # two or more choices offered
    optional: re.Pattern                   # one optional passage
    # UFGS lays headings out in columns ("1.4.1   Title"); VA uses a single
    # space ("1.1 DESCRIPTION"). Accepting one space in UFGS would swallow
    # ordinary numbered prose, so the gap is part of the convention.
    heading_gap: str = r"\s{2,}"


UFGS = Convention(
    name="ufgs",
    lettered_subparagraphs=False,
    note_block=re.compile(r"^\*{10,}\s*$", re.M),
    note_marker=re.compile(r"NOTE:"),
    blank=re.compile(r"\[_+\]"),
    alternatives=re.compile(r"\[[^\[\]]{2,200}\]\s*\[[^\[\]]{2,200}\]"),
    optional=re.compile(r"\[[^\[\]]{2,200}\]"),
    heading_gap=r"\s{2,}",
)

VA = Convention(
    name="va",
    lettered_subparagraphs=True,
    note_block=None,  # notes are indented blocks, not delimited by a rule
    note_marker=re.compile(r"SPEC\s+WRITER\s+NOTE|DESIGNER\s*.S\s+NOTE", re.I),
    blank=re.compile(r"_{4,}"),
    alternatives=re.compile(r"//[^/]{2,200}//\s*//[^/]{2,200}//"),
    optional=re.compile(r"//[^/]{2,200}//"),
    heading_gap=r"\s+",
)

BY_NAME = {c.name: c for c in (UFGS, VA)}
