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
    from ductspec.prompt import render
    by_prompt = {render(r["input"]): r for r in rows}
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
    # It refuses every question the evidence does answer. The rate is taken
    # over all rows, so it equals the share of the test set that is answerable.
    answerable = sum(r["meta"]["answer_available"] for r in rows) / len(rows)
    assert answerable > 0.4
    assert out["over_refusal_rate"] == pytest.approx(answerable)
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


# --- replaying saved generations --------------------------------------------

def test_two_model_states_do_not_share_one_run(rows):
    """AuditKIT identifies a callable model by its code, not by what it closes
    over. Two closures from the same replay factory were therefore the same
    model to the run cache, and the second silently received the first's
    results — a model answering COMPLIANT to everything reported 1.0000 strict
    accuracy, with the same run_id as the oracle.

    This is the most dangerous shape of bug this project can have: a confident
    number resting on something nobody checked.
    """
    from ductspec.audit import with_outputs

    always_compliant = ('{"verdict":"COMPLIANT","governing_ids":[],'
                        '"observation_ids":[],"action":"NONE_FOR_THIS_CHECK"}')
    perfect = {r["id"]: json.dumps(r["target"]) for r in rows}
    useless = {r["id"]: always_compliant for r in rows}

    first = ak.evaluate(with_outputs(rows, perfect), model="precomputed", scorers=scorers())
    second = ak.evaluate(with_outputs(rows, useless), model="precomputed", scorers=scorers())

    assert first.run_id != second.run_id
    assert first.headline["strict_accuracy"] == 1.0
    assert second.headline["strict_accuracy"] == 0.0
    assert second.headline["unsupported_verdict_rate"] == 1.0


def test_a_missing_generation_stops_the_run(rows):
    """Scoring whatever happens to be present would compare different row
    sets between models and report the difference as a result."""
    from ductspec.audit import with_outputs

    partial = {r["id"]: "{}" for r in rows[:-1]}
    with pytest.raises(KeyError, match="no generation for"):
        with_outputs(rows, partial)
