"""DuctSpec as an AuditKIT evaluation.

AuditKIT does the parts that are the same for every evaluation and tedious to
get right: running one dataset against several model states, fingerprinting
each run so results are comparable and cacheable, holding experiments, and
grading a comparison in the correct sense. What it cannot know is what a good
answer looks like here — that a verdict resting on a value nobody read off the
drawing is unsupported however well it is cited, or that an abstention giving
the wrong reason sends the question to the wrong person.

So the division of labour is: AuditKIT orchestrates and compares, and the
domain metrics below say what counts. They are real `ak.Metric` subclasses, so
`compare_models()` grades them, significance testing applies to them, and they
appear in the run's fingerprint.

The `direction` on each is load-bearing. Half of these are error rates, where
lower is better, and a metric declared the wrong way round would report a
regression as an improvement.
"""

from __future__ import annotations

import json
from typing import Any

import auditkit as ak
from auditkit.metric import Metric
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind

from .parsing import parse_prediction
from .prompt import render
from .scoring import score_one

VERSION = "1.0.0"


def to_samples(rows: list[dict]) -> list[ak.Sample]:
    """One record becomes one Sample.

    `input` is the rendered prompt, exactly what the model is shown. Anything
    else and a replayed generation cannot be matched back to the sample that
    produced it — which is how an oracle ends up scoring zero.

    The gold target travels in `metadata`, not in `target`: our answer is a
    structure, and the scorers need the whole of it — the citations, the
    observation, the reason code and the action — not a string to match.
    """
    return [
        ak.Sample(
            input=render(row["input"]),
            id=row["id"],
            task="ductspec-conformance",
            metadata={
                "gold": row,
                "family": row["family"],
                "panel": row["panel"],
                "scope": row["target"]["scope"],
                "verdict": row["target"]["verdict"],
                "answer_available": row["meta"]["answer_available"],
            },
            tags=[row["panel"], row["target"]["verdict"].lower()],
        )
        for row in rows
    ]


class _DuctSpecMetric(Metric):
    """Shared plumbing: parse the generation once, score it against gold."""

    kind = ScoreKind.BENCHMARK
    is_deterministic = True
    version = VERSION
    field = ""
    # AuditKIT refuses to create a Metric subclass without a direction, base
    # classes included. Every concrete metric below overrides it; this value is
    # never the one that grades anything.
    direction = Direction.MAXIMIZE

    def score(self, sample: ak.Sample, output: str, context: Any = None) -> Score:
        gold = sample.metadata["gold"]
        row = score_one(gold, parse_prediction(output))
        value = float(bool(row[self.field]))
        return Score(
            name=self.name,
            value=value,
            direction=self.direction,
            kind=self.kind,
            metadata={"panel": sample.metadata["panel"], "family": sample.metadata["family"]},
        )


class StrictAccuracy(_DuctSpecMetric):
    """The whole answer: verdict, the evidence it rests on, and what to do next."""
    name = "strict_accuracy"
    direction = Direction.MAXIMIZE
    field = "strict"


class UnsupportedVerdict(_DuctSpecMetric):
    """A definitive answer the supplied evidence does not carry."""
    name = "unsupported_verdict_rate"
    direction = Direction.MINIMIZE
    field = "unsupported"


class FalseApproval(_DuctSpecMetric):
    """Approving what is not compliant, or not known to be. The expensive error."""
    name = "false_approval_rate"
    direction = Direction.MINIMIZE
    field = "false_approval"


class ProvenanceAdmission(_DuctSpecMetric):
    """Treating a back-filled value as if the drawing had shown it."""
    name = "provenance_admission_error"
    direction = Direction.MINIMIZE
    field = "provenance_error"


class OverRefusal(_DuctSpecMetric):
    """Refusing a question the supplied evidence does answer."""
    name = "over_refusal_rate"
    direction = Direction.MINIMIZE
    field = "over_refusal"


class Coverage(_DuctSpecMetric):
    """How often it answers at all. Without this, abstaining always wins."""
    name = "coverage"
    direction = Direction.MAXIMIZE
    field = "definitive"


class CitationExact(_DuctSpecMetric):
    name = "citation_exact"
    direction = Direction.MAXIMIZE
    field = "citation_exact"


class ActionAccuracy(_DuctSpecMetric):
    """Whether the right person gets asked the right question."""
    name = "action_accuracy"
    direction = Direction.MAXIMIZE
    field = "action_correct"


class SchemaValid(_DuctSpecMetric):
    """A reviewer whose output cannot be parsed has issued no verdict."""
    name = "schema_validity"
    direction = Direction.MAXIMIZE
    field = "schema_valid"


class AbstentionReason(Metric):
    """Among abstentions, was it the right reason?

    Separate from strict accuracy because it is the one that decides what
    happens next: "the drawing is silent" sends someone to the draughtsman,
    "the section is missing" sends them to the specifier.
    """
    name = "abstention_reason_accuracy"
    direction = Direction.MAXIMIZE
    kind = ScoreKind.BENCHMARK
    is_deterministic = True
    version = VERSION

    def applicable(self, sample: ak.Sample) -> bool:
        return sample.metadata.get("verdict") == "INSUFFICIENT_EVIDENCE"

    def score(self, sample: ak.Sample, output: str, context: Any = None) -> Score:
        gold = sample.metadata["gold"]
        pred = parse_prediction(output) or {}
        ok = set(pred.get("missing_evidence") or []) == set(gold["target"]["missing_evidence"])
        return Score(name=self.name, value=float(ok), direction=self.direction, kind=self.kind)


def scorers() -> list[Metric]:
    """Every metric reported, in the order they should be read."""
    return [
        StrictAccuracy(), Coverage(),
        UnsupportedVerdict(), FalseApproval(), ProvenanceAdmission(), OverRefusal(),
        CitationExact(), AbstentionReason(), ActionAccuracy(), SchemaValid(),
    ]


def with_outputs(rows: list[dict], saved: dict[str, str]) -> list[ak.Sample]:
    """Samples carrying generations that were produced elsewhere.

    AuditKIT identifies a callable model by its code, not by what it closes
    over. Two closures built by the same factory — one replaying the base
    model's answers, one replaying an adapter's — are therefore the same model
    as far as the run cache is concerned, and the second silently receives the
    first's results. Verified: both runs came back with the same run_id, and a
    model answering COMPLIANT to everything reported 1.0000 strict accuracy.

    Attaching the generations to the samples avoids the question entirely.
    """
    from dataclasses import replace

    out = []
    for sample in to_samples(rows):
        if sample.id not in saved:
            raise KeyError(
                f"no generation for {sample.id}. Scoring a partial run would "
                "compare different row sets between models."
            )
        out.append(replace(sample, actual_output=saved[sample.id]))
    return out


def compare(models: list, rows: list[dict], names: list[str] | None = None):
    """Run the model states over one frozen test set and grade the difference.

    This is the before-and-after: the base model, the adapter trained on
    uncurated extractions, and the adapter trained on curated ones, scored on
    identical packets with identical decoding.
    """
    return ak.compare_models(
        models,
        to_samples(rows),
        scorers=scorers(),
        model_names=names,
        experiment_name="ductspec-conformance",
        tags=["ductspec", "track1"],
    )
