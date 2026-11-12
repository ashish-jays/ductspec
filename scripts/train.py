"""Train one arm's adapter with AlignTune.

Run twice, once per arm. The arms differ only in their training data, so
everything here — seed, steps, rank, sequence budget — is identical between
them by construction rather than by care.

Two things in this file are not obvious and both were found the expensive way:

The chat template arrives as a dict. Tiny Aya's tokenizer exposes its named
templates as a mapping, and TRL's assistant-mask check expects a string, so
`assistant_only_loss=True` raises TypeError before training starts. Resolving
the template to its default string fixes it, and the supervised span then
covers the JSON answer and its end-of-turn markers.

The safety preamble stays. Tiny Aya's template prepends 365 tokens of refusal
commitments to every turn. Stripping it would reclaim a third of each example
— and would mean deleting the model's safety instructions and then reporting
that safety held, which is the failure this hackathon exists to catch.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = "final_adapter"
from ductspec.model import BASE_MODEL as BASE, HUB_ID  # noqa: E402

# Identical for both arms. Changing anything here invalidates the comparison.
HYPERPARAMETERS = dict(
    backend="trl",
    num_epochs=3,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler="cosine",
    # The longest training example is about 1,330 tokens once the chat
    # template and its safety preamble are applied. export_arms.py prints
    # the current figure; a longer example is truncated, answer first.
    max_seq_length=1536,
    precision="bf16",
    quantization={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    },
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    assistant_only_loss=True,     # supervise the answer, not the packet
    packing=False,
    # gradient_checkpointing is the trainer argument; use_gradient_checkpointing
    # sets a separate model-config field and does not enable it.
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    attn_implementation="sdpa",
    seed=42,
    data_seed=42,
    eval_strategy="no",
    save_steps=40,
    logging_steps=5,
    report_to="none",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["raw", "curated"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap steps; use a small number for a smoke run")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    data = ROOT / "data" / "build" / "messages" / args.arm
    out = Path(args.output or ROOT / "data" / "adapters" / args.arm)
    out.mkdir(parents=True, exist_ok=True)

    from aligntune import create_sft_trainer

    opts = dict(HYPERPARAMETERS)
    if args.max_steps:
        opts["max_steps"] = args.max_steps
    if args.max_samples:
        opts["max_samples"] = args.max_samples

    print(f"arm={args.arm}  data={data}  out={out}")
    trainer = create_sft_trainer(
        model_name=BASE,
        dataset_name=str(data / "train.jsonl"),
        output_dir=str(out),
        **opts,
    )

    trainer.setup_model()
    # See the module docstring: the named templates arrive as a mapping, and
    # TRL's assistant-mask check needs the string.
    template = trainer.tokenizer.get_chat_template()
    if not isinstance(trainer.tokenizer.chat_template, str):
        trainer.tokenizer.chat_template = template
    trainer.model.config.use_cache = False

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    trainer.tokenizer.save_pretrained(out)
    (out / "run.json").write_text(json.dumps({
        "arm": args.arm,
        "base_model": HUB_ID,
        "base_model_loaded_from": BASE,
        "data": str(data.relative_to(ROOT)),
        "hyperparameters": {k: v for k, v in opts.items() if k != "quantization"},
        "quantization": opts["quantization"],
        "wall_clock_seconds": round(elapsed, 1),
        "gpu": os.environ.get("DUCTSPEC_GPU", "unknown"),
        "gpu_hourly_usd": float(os.environ.get("DUCTSPEC_GPU_USD", "0.80")),
    }, indent=2) + "\n")

    print(f"\ntrained in {elapsed / 60:.1f} min -> {out}")

    # The trainer leaves checkpoints at save_steps and nothing at the end, so
    # the adapter that was actually reported on had no fixed address. Saved
    # after run.json, so the record of the run exists whatever happens here.
    trainer.model.save_pretrained(str(out / ADAPTER_DIR))
    print(f"adapter: {out / ADAPTER_DIR}")


if __name__ == "__main__":
    main()
