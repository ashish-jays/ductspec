"""Read a prediction out of whatever the model actually emitted.

A small model asked for JSON will mostly produce JSON, and will sometimes wrap
it in a code fence, introduce it with a sentence, or follow it with a summary.
Those are formatting slips and recovering from them is fair.

What is NOT recovered: a missing verdict, a verdict outside the enum, or text
with no JSON object in it. Those count as failures. Repairing them would hide
exactly the unreliability the evaluation exists to measure — and a reviewer who
cannot parse the reviewer's output has no verdict, whatever the model meant.
"""

from __future__ import annotations

import json
import re

from .schema import ACTIONS, REASON_CODES, VERDICTS

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _candidates(text: str):
    """JSON-looking spans, most likely first."""
    for m in _FENCE.finditer(text):
        yield m.group(1)
    # Outermost brace span: the whole object, including nested ones.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        yield text[start : end + 1]


def parse_prediction(text: str) -> dict | None:
    """Return a normalised prediction, or None if nothing usable was emitted."""
    if not text:
        return None

    for blob in _candidates(text):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return _normalise(data)
    return None


def _normalise(data: dict) -> dict | None:
    # The model may nest its answer under "target", mirroring the training format.
    if "verdict" not in data and isinstance(data.get("target"), dict):
        data = data["target"]

    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in VERDICTS:
        return None

    action = data.get("action")
    action = action.strip().upper() if isinstance(action, str) else ""

    return {
        "verdict": verdict.strip().upper(),
        "governing_ids": _str_list(data.get("governing_ids")),
        "observation_ids": _str_list(data.get("observation_ids")),
        # Unknown reason codes are dropped rather than invented: a made-up code
        # would score as a wrong reason, which is the right outcome.
        "missing_evidence": [c for c in _str_list(data.get("missing_evidence")) if c in REASON_CODES],
        "action": action if action in ACTIONS else "",
        "reason": data.get("reason") if isinstance(data.get("reason"), str) else "",
    }


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []
