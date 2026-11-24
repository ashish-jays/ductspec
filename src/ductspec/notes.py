"""Split multi-sentence drawing notes into one line per sentence (Stage 2, post-hoc).

Real bid drawings state several requirements in one long general note ("3.
DUCTWORK INSULATION TO MEET ... RECTANGULAR SUPPLY AIR DUCTWORK IS TO BE
EXTERNALLY INSULATED WITH ... RECTANGULAR RETURN AND TRANSFER AIR DUCTS SHALL
BE LINED WITH ..."). Training notes make one claim per line. This step splits a
note at sentence ends and keeps every word, so each line makes one claim and
can be cited on its own.

It is generic on purpose: it knows nothing about ducts, properties or any
document, and uses no labels. A line with one sentence is returned unchanged,
id included. Split lines get ids <id>.s1, <id>.s2, ...; parent() maps them back,
so a citation of any sentence of a note counts as a citation of that note.

Added after the first real-drawing check showed the trained models did not use
long prose notes; reported as post-hoc.
"""

from __future__ import annotations

import copy
import re

# A sentence ends at ". " (or "? " / "! ") followed by a capital, digit or "(".
# "U.N.O." and "1/2\"" have no space after their periods, so they never split.
_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_LIST_MARK = re.compile(r"^\(?[0-9A-Za-z]{1,3}[.)]$")  # "3." or "A." or "(b)" on its own


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _END.split(text.strip()) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and _LIST_MARK.match(out[-1]):
            out[-1] = f"{out[-1]} {p}"  # keep a list number with its sentence
        else:
            out.append(p)
    return out or [text]


def split_lines(lines: list[dict]) -> list[dict]:
    out = []
    for line in lines:
        parts = sentences(line["text"])
        if len(parts) == 1:
            out.append(dict(line))
        else:
            out.extend({"id": f"{line['id']}.s{k}", "text": t} for k, t in enumerate(parts, 1))
    return out


def split_packet(row: dict) -> dict:
    """A copy of a record with its drawing notes split; target and meta untouched."""
    out = copy.deepcopy(row)
    out["input"]["drawing"] = split_lines(row["input"]["drawing"])
    return out


def parent(line_id: str) -> str:
    return re.sub(r"\.s\d+$", "", line_id)
