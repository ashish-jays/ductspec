"""The answer mask: training must supervise exactly the answer.

A tiny stand-in tokenizer renders the chat template's assistant turn the way a
real one does: tokens inside {% generation %} are masked, nothing else is.
Checked separately on the real Aya Expanse tokenizer on Kaggle: 0 of 21 tokens
masked before mark(), the answer plus end-of-turn after it.
"""

import pytest

from ductspec import answer_mask
from ductspec.answer_mask import COHERE_ASSISTANT, COHERE_MARKED

END = "<|END_OF_TURN_TOKEN|>"


class FakeTokenizer:
    """One token per word or special marker; masks what the template marks."""

    all_special_tokens = [END, "<BOS_TOKEN>"]

    def __init__(self, template, mask_prompt=False):
        self.chat_template = template
        self.mask_prompt = mask_prompt
        self.vocab = []

    def get_chat_template(self):
        return self.chat_template["default"]

    def _id(self, piece):
        self.vocab.append(piece)
        return len(self.vocab) - 1

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids).replace(f" {END}", END)

    def apply_chat_template(self, msgs, tokenize, return_dict, return_assistant_tokens_mask):
        marked = "generation %}" in self.chat_template
        ids, mask = [], []
        for word in msgs[0]["content"].split():
            ids.append(self._id(word))
            mask.append(1 if self.mask_prompt else 0)
        for word in msgs[1]["content"].split() + [END]:
            ids.append(self._id(word))
            mask.append(1 if marked else 0)
        return {"input_ids": ids, "assistant_masks": mask}


def _template():
    return "{{ bos_token }}" + COHERE_ASSISTANT + "{% endfor %}"


def test_an_unmarked_template_is_rejected():
    tok = FakeTokenizer(_template())
    with pytest.raises(SystemExit, match="not the answer"):
        answer_mask.verify(tok, "is it galvanized?", '{"verdict": "COMPLIANT"}')


def test_marking_makes_the_mask_exactly_the_answer():
    tok = FakeTokenizer(_template())
    assert answer_mask.mark(tok) == "added generation markers to the assistant turn"
    assert COHERE_MARKED in tok.chat_template and COHERE_ASSISTANT not in tok.chat_template
    assert answer_mask.verify(tok, "is it galvanized?", '{"verdict": "COMPLIANT"}').startswith("answer mask OK")


def test_a_named_template_mapping_is_resolved_before_marking():
    tok = FakeTokenizer({"default": _template(), "rag": "other"})
    answer_mask.mark(tok)
    assert isinstance(tok.chat_template, str) and COHERE_MARKED in tok.chat_template


def test_an_already_marked_template_is_left_alone():
    marked = "{% generation %}{{ content }}{% endgeneration %}"
    tok = FakeTokenizer(marked)
    assert answer_mask.mark(tok) == "template already marks the answer"
    assert tok.chat_template == marked


def test_an_unknown_template_stops_rather_than_guessing():
    tok = FakeTokenizer("{{ content }}")
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        answer_mask.mark(tok)


def test_a_mask_that_also_covers_the_question_is_rejected():
    tok = FakeTokenizer(_template(), mask_prompt=True)
    answer_mask.mark(tok)
    with pytest.raises(SystemExit, match="not the answer"):
        answer_mask.verify(tok, "is it galvanized?", '{"verdict": "COMPLIANT"}')


def test_train_py_checks_the_mask_before_training():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text(encoding="utf-8")
    assert text.index("answer_mask.verify(") < text.index("trainer.train()")
