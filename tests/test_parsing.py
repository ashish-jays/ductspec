from ductspec.parsing import parse_prediction

GOOD = """{"verdict": "COMPLIANT", "governing_ids": ["C-1"], "action": "NONE_FOR_THIS_CHECK"}"""


def test_plain_json():
    assert parse_prediction(GOOD)["verdict"] == "COMPLIANT"


def test_fenced_and_prefaced():
    text = 'Here is my review:\n```json\n' + GOOD + '\n```\nHope that helps.'
    out = parse_prediction(text)
    assert out["verdict"] == "COMPLIANT"
    assert out["governing_ids"] == ["C-1"]


def test_nested_under_target():
    assert parse_prediction('{"target": ' + GOOD + '}')["verdict"] == "COMPLIANT"


def test_lowercase_verdict_is_accepted():
    assert parse_prediction('{"verdict": "compliant"}')["verdict"] == "COMPLIANT"


def test_single_citation_as_string_becomes_a_list():
    assert parse_prediction('{"verdict":"COMPLIANT","governing_ids":"C-1"}')["governing_ids"] == ["C-1"]


def test_invented_reason_code_is_dropped_not_kept():
    out = parse_prediction('{"verdict":"INSUFFICIENT_EVIDENCE","missing_evidence":["VIBES"]}')
    assert out["missing_evidence"] == []


def test_unparseable_output_is_a_failure_not_a_guess():
    for text in ("", "I think it is probably fine.", "{not json", '{"verdict": "MAYBE"}'):
        assert parse_prediction(text) is None
