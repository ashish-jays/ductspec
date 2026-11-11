"""Scoring a DuctSpec run.

Two things are easy to do by accident and both look like success:

- **Always answer.** Grounded-answer benchmarks reward confident verdicts, and
  real specifications are repetitive enough that a guess is often right.
- **Always abstain.** A model that refuses everything has a perfect unsupported
  -verdict rate and is worthless.

So no single number is the headline. `strict_accuracy` requires the verdict,
the citation and the observation to all be right together; `coverage` and
`false_approval` sit beside it to catch each degenerate strategy; and
`pair_accuracy` scores counterfactual twins jointly, which is the measure that
is genuinely hard to game.
"""

from __future__ import annotations

from collections import Counter

from .schema import INDEPENDENT_ORIGINS, VERDICTS

DEFINITIVE = ("COMPLIANT", "NON_COMPLIANT")


def _scope_origin(rec: dict) -> str | None:
    scope = rec["target"]["scope"]
    prop = rec["input"]["object"]["properties"].get(scope) or {}
    return prop.get("origin")


def _has_independent_evidence(rec: dict) -> bool:
    scope = rec["target"]["scope"]
    prop = rec["input"]["object"]["properties"].get(scope) or {}
    return prop.get("origin") in INDEPENDENT_ORIGINS and prop.get("value") is not None


def is_supported(gold: dict, pred: dict) -> bool:
    """A verdict is supported when the model earned the right to give it.

    For a definitive verdict: the drawing must independently establish the
    property, and the cited governing evidence must be the evidence that
    actually governs. For an abstention: the reason must be the right reason —
    "the drawing is silent" and "the value was copied from the spec" are
    different findings and the follow-up differs.
    """
    verdict = pred.get("verdict")
    if verdict not in VERDICTS:
        return False

    if verdict in DEFINITIVE:
        # Was a definitive answer even available on this case? Checking the
        # citations first lets a lucky guess on an unanswerable packet pass.
        if not gold.get("meta", {}).get("answer_available", _has_independent_evidence(gold)):
            return False
        if not _has_independent_evidence(gold):
            return False
        if set(pred.get("governing_ids") or []) != set(gold["target"]["governing_ids"]):
            return False
        # Which observation the model relied on is part of the answer. A verdict
        # that cites the right clause but points at nothing — or at an id that
        # is not in the object — has not shown its working.
        return set(pred.get("observation_ids") or []) == set(gold["target"]["observation_ids"])

    return set(pred.get("missing_evidence") or []) == set(gold["target"]["missing_evidence"])


def score_one(gold: dict, pred: dict | None) -> dict:
    # An unparseable generation is a failure, not an exception.
    pred = pred or {}
    g, verdict = gold["target"], pred.get("verdict")
    correct_verdict = verdict == g["verdict"]
    supported = is_supported(gold, pred)

    return {
        "schema_valid": verdict in VERDICTS,
        "correct_verdict": correct_verdict,
        # The whole answer, not just its headline: the verdict, the evidence it
        # rests on, and what the reviewer is told to do next.
        "strict": correct_verdict and supported and pred.get("action") == g["action"],
        "definitive": verdict in DEFINITIVE,
        # An unsupported verdict is a confident answer the evidence does not carry.
        "unsupported": verdict in DEFINITIVE and not supported,
        # The expensive error: approving something that is not compliant, or not known to be.
        "false_approval": verdict == "COMPLIANT" and g["verdict"] != "COMPLIANT",
        # Treating a back-filled value as if the drawing had shown it.
        "provenance_error": (
            verdict in DEFINITIVE and _scope_origin(gold) not in INDEPENDENT_ORIGINS
        ),
        # Refusing a question the evidence does in fact answer.
        "over_refusal": verdict == "INSUFFICIENT_EVIDENCE" and g["verdict"] in DEFINITIVE,
        "citation_exact": set(pred.get("governing_ids") or []) == set(g["governing_ids"]),
        "action_correct": pred.get("action") == g["action"],
    }


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def score_run(golds: list[dict], preds: list[dict]) -> dict:
    """Aggregate over a whole test set. `preds` must align with `golds`."""
    if len(golds) != len(preds):
        raise ValueError(f"{len(golds)} gold records but {len(preds)} predictions")

    rows = [score_one(g, p) for g, p in zip(golds, preds)]
    n = len(rows)
    definitive = sum(r["definitive"] for r in rows)
    answerable = sum(1 for g in golds if g["target"]["verdict"] in DEFINITIVE)
    inferred_slice = [
        (g, r) for g, r in zip(golds, rows) if _scope_origin(g) not in INDEPENDENT_ORIGINS
    ]

    return {
        "n": n,
        "strict_accuracy": _rate(sum(r["strict"] for r in rows), n),
        "verdict_accuracy": _rate(sum(r["correct_verdict"] for r in rows), n),
        "verdict_macro_f1": _macro_f1(golds, preds),
        "unsupported_rate": _rate(sum(r["unsupported"] for r in rows), n),
        "unsupported_rate_given_definitive": _rate(sum(r["unsupported"] for r in rows), definitive),
        "false_approval_rate": _rate(
            sum(r["false_approval"] for r in rows), sum(1 for g in golds if g["target"]["verdict"] != "COMPLIANT")
        ),
        "provenance_admission_error": _rate(
            sum(r["provenance_error"] for _, r in inferred_slice), len(inferred_slice)
        ),
        "coverage": _rate(definitive, n),
        "over_refusal_rate": _rate(sum(r["over_refusal"] for r in rows), answerable),
        "citation_precision": _rate(
            sum(r["citation_exact"] for r in rows if r["definitive"]), definitive
        ),
        "action_accuracy": _rate(sum(r["action_correct"] for r in rows), n),
        "schema_validity": _rate(sum(r["schema_valid"] for r in rows), n),
    }


def score_pairs(pairs: list[tuple[dict, dict]], preds: dict[str, dict]) -> dict:
    """Score counterfactual twins jointly: both members right, or the pair fails.

    `pairs` are (base_gold, twin_gold); `preds` maps record id to prediction.
    """
    joint = 0
    for base, twin in pairs:
        both = all(
            score_one(g, preds.get(g["id"], {}))["strict"] for g in (base, twin)
        )
        joint += both
    return {"pairs": len(pairs), "pair_accuracy": _rate(joint, len(pairs))}


def _macro_f1(golds: list[dict], preds: list[dict]) -> float:
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(golds, preds):
        gold_v, pred_v = g["target"]["verdict"], p.get("verdict")
        if gold_v == pred_v:
            tp[gold_v] += 1
        else:
            fn[gold_v] += 1
            if pred_v in VERDICTS:
                fp[pred_v] += 1

    scores = []
    for v in VERDICTS:
        denom_p, denom_r = tp[v] + fp[v], tp[v] + fn[v]
        if not denom_r:
            continue
        precision = tp[v] / denom_p if denom_p else 0.0
        recall = tp[v] / denom_r
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0
