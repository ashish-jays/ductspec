"""Make training supervise the answer, and only the answer.

Training sets `assistant_only_loss=True`: the loss covers the tokens the chat
template wraps in `{% generation %}`. Aya Expanse's template has no such
marker, so the answer mask is empty (measured on the real tokenizer: 0 of 21
tokens). The trainer does not stop on an empty mask, so a run can complete
without ever supervising the answer the way the configuration says it does.

mark(tok)                 adds the marker around the assistant turn of a
                          Cohere-style template when it is missing; a template
                          that already marks the answer is left alone.
verify(tok, user, answer) raises unless the masked tokens decode to exactly the
                          answer followed by one special (end-of-turn) token.

train.py calls both before training, so an unmarked template stops the run
instead of training on the wrong tokens.
"""

COHERE_ASSISTANT = ("{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() "
                    "+ '<|END_OF_TURN_TOKEN|>' }}")
COHERE_MARKED = ("{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% generation %}"
                 "{{ content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endgeneration %}")


def mark(tok) -> str:
    template = tok.chat_template
    if not isinstance(template, str):
        template = tok.get_chat_template()
    if "generation %}" in template:
        tok.chat_template = template
        return "template already marks the answer"
    found = template.count(COHERE_ASSISTANT)
    if found != 1:
        raise SystemExit(f"answer_mask: assistant anchor found {found} times; cannot mark the answer")
    tok.chat_template = template.replace(COHERE_ASSISTANT, COHERE_MARKED)
    return "added generation markers to the assistant turn"


def verify(tok, user: str, answer: str) -> str:
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    out = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    ids, mask = out["input_ids"], out["assistant_masks"]
    masked = tok.decode([i for i, m in zip(ids, mask) if m])
    tail = masked[len(answer.strip()):].strip()
    if not masked.startswith(answer.strip()) or tail not in set(tok.all_special_tokens):
        raise SystemExit(f"answer_mask: masked text is not the answer plus end of turn: {masked[:200]!r}")
    return f"answer mask OK: {sum(mask)} of {len(ids)} tokens supervised"
