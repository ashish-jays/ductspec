"""The metrics have to survive being run by AuditKIT, not just by us."""

import json

import auditkit as ak
import pytest

from ductspec.audit import scorers, to_samples
from ductspec.generate import build

ALWAYS_COMPLIANT = '{"verdict":"COMPLIANT","governing_ids":[],"observation_ids":[],"action":"NONE_FOR_THIS_CHECK"}'
ALWAYS_ABSTAIN = ('{"verdict":"INSUFFICIENT_EVIDENCE","governing_ids":[],"observation_ids":[],'
                  '"missing_evidence":["MISSING_OBSERVATION"],"action":"REQUEST_MISSING_DETAIL"}')


@pytest.fixture(scope="module")
def rows():
    return build()["test"]


def _scores(rows, model):
    return ak.evaluate(to_samples(rows), model=model, scorers=scorers()).headline


def test_every_metric_declares_which_way_is_better():
    """A rate where lower is better, graded as MAXIMIZE, reports a regression
    as an improvement. AuditKIT refuses to build a metric without this, and
    half of ours are error rates."""
    from auditkit.types import Direction
    by_name = {m.name: m.direction for m in scorers()}
    for name in ("unsupported_verdict_rate", "false_approval_rate",
                 "provenance_admission_error", "over_refusal_rate"):
        assert by_name[name] is Direction.MINIMIZE
    for name in ("strict_accuracy", "coverage", "citation_exact"):
        assert by_name[name] is Direction.MAXIMIZE


def test_the_oracle_scores_perfectly(rows):
    by_prompt = {json.dumps(r["input"], indent=1): r for r in rows}
    out = _scores(rows, lambda ps: [json.dumps(by_prompt[p]["target"]) for p in ps])

    assert out["strict_accuracy"] == 1.0
    assert out["unsupported_verdict_rate"] == 0.0
    assert out["provenance_admission_error"] == 0.0
    assert out["abstention_reason_accuracy"] == 1.0


def test_certifying_everything_is_caught(rows):
    out = _scores(rows, lambda ps: [ALWAYS_COMPLIANT] * len(ps))
    assert out["coverage"] == 1.0                    # looks maximally useful
    assert out["unsupported_verdict_rate"] == 1.0    # and is caught here
    assert out["false_approval_rate"] > 0.5
    assert out["provenance_admission_error"] > 0.3
    assert out["strict_accuracy"] == 0.0


def test_certifying_nothing_is_caught(rows):
    out = _scores(rows, lambda ps: [ALWAYS_ABSTAIN] * len(ps))
    # A perfect score on all three headline safety numbers.
    assert out["unsupported_verdict_rate"] == 0.0
    assert out["false_approval_rate"] == 0.0
    assert out["provenance_admission_error"] == 0.0
    # And useless.
    assert out["coverage"] == 0.0
    assert out["over_refusal_rate"] > 0.5
    assert out["strict_accuracy"] < 0.1


def test_the_abstention_reason_metric_only_scores_abstentions(rows):
    result = ak.evaluate(to_samples(rows), model=lambda ps: [ALWAYS_ABSTAIN] * len(ps),
                         scorers=scorers())
    assert "abstention_reason_accuracy" in result.headline
    abstentions = sum(1 for r in rows if r["target"]["verdict"] == "INSUFFICIENT_EVIDENCE")
    assert 0 < abstentions < len(rows)


def test_the_sample_input_is_the_prompt_the_model_sees(rows):
    """A Sample whose input is not the rendered prompt cannot be matched back
    to a saved generation, and an oracle replaying its own answers scores
    zero — which is what happened before this test existed."""
    from ductspec.prompt import render

    samples = to_samples(rows)
    assert samples[0].input == render(rows[0]["input"])

    by_prompt = {render(r["input"]): r for r in rows}
    replayed = ak.evaluate(
        samples,
        model=lambda ps: [json.dumps(by_prompt[p]["target"]) for p in ps],
        scorers=scorers(),
    )
    assert replayed.headline["strict_accuracy"] == 1.0
