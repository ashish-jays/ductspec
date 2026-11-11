"""The DuctSpec demo.

One idea, shown rather than argued: change where a value came from, change
nothing else, and watch the reviewer stop certifying.

The model runs live on ZeroGPU. Where a GPU is not available the Space falls
back to replaying saved generations from the evaluation run, and says so on
screen — a demo that silently shows canned output is the same species of
problem as a verdict that silently rests on a value nobody read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

BASE = "CohereLabs/tiny-aya-global"
ADAPTER = os.environ.get("DUCTSPEC_ADAPTER", "")
HERE = Path(__file__).parent

INSTRUCTION = (HERE / "instruction.txt").read_text() if (HERE / "instruction.txt").exists() else ""
EXAMPLES = json.loads((HERE / "examples.json").read_text()) if (HERE / "examples.json").exists() else []
REPLAY = json.loads((HERE / "replay.json").read_text()) if (HERE / "replay.json").exists() else {}

_model = None
_tok = None


def _load():
    global _model, _tok
    if _model is not None:
        return _tok, _model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
    _model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _tok, _model


def review(packet_json: str) -> tuple[str, str]:
    """Return (verdict JSON, how it was produced)."""
    try:
        packet = json.loads(packet_json)
    except json.JSONDecodeError as exc:
        return "", f"That packet is not valid JSON: {exc}"

    key = json.dumps(packet, sort_keys=True)
    try:
        import spaces  # noqa: F401  (present only on ZeroGPU)
        import torch

        tok, model = _load()
        prompt = INSTRUCTION + json.dumps(packet, indent=1)
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        return text.strip(), f"Generated live on this Space — {BASE}" + (
            f" with adapter {ADAPTER}" if ADAPTER else " (base model, no adapter)")
    except Exception as exc:
        saved = REPLAY.get(key)
        if saved:
            return saved, ("Replayed from the evaluation run — no GPU available "
                           f"in this environment ({type(exc).__name__}).")
        return "", (f"No GPU here and no saved generation for this packet "
                    f"({type(exc).__name__}: {exc}). Edit one of the examples instead.")


with gr.Blocks(title="DuctSpec") as demo:
    gr.Markdown(
        "# DuctSpec\n"
        "### Can a 3.35B model refuse to certify compliance it cannot see?\n\n"
        "Each packet is one duct, one property, the takeoff value carried for it, the "
        "lines lettered on the drawing, and the specification clauses supplied to the "
        "reviewer. Nothing says where the takeoff value came from. The drawing lines "
        "are the only evidence of what the drawing shows.\n\n"
        "**Try this.** Run the first example. Then delete the value from the drawing "
        "line that carries it, leaving the takeoff value alone, and run it again. The "
        "number being checked has not changed. The answer should."
    )
    with gr.Row():
        with gr.Column():
            packet = gr.Code(label="Packet", language="json", lines=24,
                             value=json.dumps(EXAMPLES[0]["input"], indent=1) if EXAMPLES else "")
            run = gr.Button("Review", variant="primary")
        with gr.Column():
            verdict = gr.Code(label="Verdict", language="json", lines=12)
            provenance = gr.Markdown()
            expected = gr.Markdown()

    if EXAMPLES:
        gr.Examples(
            examples=[[json.dumps(e["input"], indent=1)] for e in EXAMPLES],
            inputs=[packet],
            label="Examples — the pairs differ only in where a value came from",
        )

    def _run(p):
        text, how = review(p)
        return text, f"*{how}*", ""

    run.click(_run, inputs=[packet], outputs=[verdict, provenance, expected])

if __name__ == "__main__":
    demo.launch()
