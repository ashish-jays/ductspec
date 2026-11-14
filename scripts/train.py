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
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = "final_adapter"
from ductspec import answer_mask  # noqa: E402
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
    # precision and bnb_4bit_compute_dtype are filled in by compute_dtype()
    # below: Kaggle's T4 and P100 are pre-Ampere and have no bfloat16, and
    # asking for it there fails at load rather than falling back.
    quantization={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
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


def compute_dtype() -> str:
    """bf16 where the GPU has it, fp16 otherwise.

    Ampere and later carry bfloat16 in hardware; that is compute capability
    8.0. Kaggle's free tiers are below it — the T4 is Turing (7.5) and the
    P100 is Pascal (6.0). The capability is read directly because
    torch.cuda.is_bf16_supported() answers True on a T4: it counts emulation,
    which is not what we are asking. fp16 is the same 16 bits with less
    exponent range, which matters for pre-training and not for a LoRA adapter
    over frozen 4-bit weights. The choice is recorded in run.json, because it
    is a difference between two runs that are otherwise identical.
    """
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return "bf16"
    except Exception:
        pass
    return "fp16"


def fit_dtypes_to_card(model, precision: str) -> dict:
    """Put every tensor in the dtype this card can actually use.

    AlignTune asks transformers for a dtype using `torch_dtype=`, which
    transformers 5 renamed to `dtype=`. The request is dropped with a
    deprecation warning and nothing else, so the model arrives in fp32
    whatever precision was requested. On most models that is survivable. Aya
    Expanse has a 256,000-token vocabulary, so in fp32 its embedding table is
    4.2GB and a single 1,536-token batch of logits is another 1.6GB — about
    two gigabytes more than a 15.6GB T4 has to spare, which is exactly the
    margin the run was failing by.

    The same indifference leaves the 4-bit compute dtype at bitsandbytes'
    bfloat16 default. That is where the bfloat16 gradient came from, and
    AMP's gradient scaler has no bfloat16 kernel to unscale it with.

    So the dtypes are set here rather than requested:

      trainable (LoRA)      fp32  — master weights, the ordinary QLoRA shape
      frozen, 2-D and up    16-bit — the embedding table, where the memory is
      frozen, 1-D           left alone — norms and biases are small, and they
                            are the tensors that want the extra range
      4-bit compute dtype   16-bit — removes bfloat16 from the graph

    Returns a summary for run.json, because this is a difference between two
    runs that are otherwise identical.
    """
    target = torch.bfloat16 if precision == "bf16" else torch.float16
    moved = {"trainable_to_fp32": 0, "frozen_to_16bit": 0, "linear4bit": 0}

    for _, q in model.named_parameters():
        if q.requires_grad:
            if q.dtype != torch.float32:
                q.data = q.data.to(torch.float32)
                moved["trainable_to_fp32"] += 1
        elif q.is_floating_point() and q.dim() > 1 and q.dtype != target:
            q.data = q.data.to(target)
            moved["frozen_to_16bit"] += 1

    try:
        import bitsandbytes as bnb
        for m in model.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                m.compute_dtype = target
                qs = getattr(m, "quant_state", None)
                if qs is not None and getattr(qs, "dtype", None) != target:
                    qs.dtype = target
                moved["linear4bit"] += 1
    except ImportError:
        pass

    return moved


def dtype_census(model) -> dict:
    """What the model is made of, after the casting above."""
    trainable, frozen, buffers = Counter(), Counter(), Counter()
    for _, q in model.named_parameters():
        (trainable if q.requires_grad else frozen)[str(q.dtype)] += 1
    for b in model.buffers():
        buffers[str(b.dtype)] += 1
    compute = Counter()
    try:
        import bitsandbytes as bnb
        for m in model.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                compute[str(getattr(m, "compute_dtype", None))] += 1
    except ImportError:
        pass
    return {"trainable": dict(trainable), "frozen": dict(frozen),
            "buffers": dict(buffers), "linear4bit_compute": dict(compute)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["raw", "curated"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap steps; use a small number for a smoke run")
    parser.add_argument("--max-samples", type=int, default=None)
    # The three knobs the training analysis actually moved. They default to
    # the values above, so an unflagged run is the committed configuration;
    # whatever is passed lands in run.json with the rest.
    parser.add_argument("--lora-r", type=int, default=None,
                        help="LoRA rank; alpha follows at 2x rank")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"],
                        default="auto",
                        help="auto picks bf16 on Ampere+, fp16 below it")
    parser.add_argument("--optimizer", default=None,
                        help="e.g. paged_adamw_8bit, to keep the optimizer "
                             "state off a small card")
    args = parser.parse_args()

    data = ROOT / "data" / "build" / "messages" / args.arm
    out = Path(args.output or ROOT / "data" / "adapters" / args.arm)
    out.mkdir(parents=True, exist_ok=True)

    from aligntune import create_sft_trainer

    opts = dict(HYPERPARAMETERS)
    precision = compute_dtype() if args.precision == "auto" else args.precision
    opts["precision"] = precision
    opts["quantization"] = dict(opts["quantization"])
    # The frozen base still does its matmuls in 16-bit even when the trainable
    # side is fp32: that is where the T4's tensor cores earn their keep.
    opts["quantization"]["bnb_4bit_compute_dtype"] = (
        "bfloat16" if precision == "bf16" else "float16")
    if args.max_steps:
        opts["max_steps"] = args.max_steps
    if args.max_samples:
        opts["max_samples"] = args.max_samples
    if args.lora_r:
        opts["lora_r"] = args.lora_r
        opts["lora_alpha"] = 2 * args.lora_r
    if args.epochs:
        opts["num_epochs"] = args.epochs
    if args.learning_rate:
        opts["learning_rate"] = args.learning_rate
    if args.optimizer:
        opts["optimizer"] = args.optimizer

    print(f"arm={args.arm}  data={data}  out={out}")
    print(f"base={BASE}  precision={precision}  "
          f"rank={opts['lora_r']}  epochs={opts['num_epochs']}  "
          f"lr={opts['learning_rate']}  "
          f"optim={opts.get('optimizer', 'adamw_torch')}")
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
    # Aya Expanse's template marks no answer tokens; without this the answer
    # mask is empty. verify() stops the run unless exactly the answer is supervised.
    print(answer_mask.mark(trainer.tokenizer), flush=True)
    first = json.loads((data / "train.jsonl").open(encoding="utf-8").readline())["messages"]
    print(answer_mask.verify(trainer.tokenizer, first[0]["content"], first[-1]["content"]), flush=True)

    # The trainable parameters have to be fp32, and on this hardware that is
    # not optional. Aya Expanse declares torch_dtype bfloat16 in its config,
    # so PEFT creates the LoRA weights in bfloat16 to match. fp16 training
    # runs through AMP's GradScaler, and the CUDA kernel it unscales
    # gradients with has no bfloat16 implementation — the run dies at the
    # first gradient clip with NotImplementedError. Upcasting the adapter is
    # the ordinary QLoRA arrangement anyway: frozen 4-bit base, fp32 adapter.
    # On Ampere and later the run picks bf16, no GradScaler is used, and this
    # loop changes nothing.
    print(f"  dtypes as loaded: {dtype_census(trainer.model)}")
    moved = fit_dtypes_to_card(trainer.model, precision)
    census = dtype_census(trainer.model)
    print(f"  recast:           {moved}")
    print(f"  dtypes for training: {census}")

    started = time.time()
    try:
        trainer.train()
    except Exception:
        # The bfloat16 that kills fp16 training on this card is not in the
        # model — the census above shows none. It is made somewhere inside
        # the trainer, so the trainer is asked directly, once, on the way
        # out. Diagnostic only: the exception is re-raised untouched.
        import traceback
        print("\n" + "=" * 70)
        print("TRAIN FAILED - what the inner trainer was actually set to")
        print("=" * 70)
        inner = getattr(trainer, "trainer", None)
        if inner is not None:
            a = getattr(inner, "args", None)
            for flag in ("fp16", "bf16", "fp16_full_eval", "bf16_full_eval",
                         "half_precision_backend", "optim", "max_grad_norm",
                         "gradient_checkpointing"):
                print(f"  args.{flag:24} = {getattr(a, flag, '<unset>')}")
            acc = getattr(inner, "accelerator", None)
            print(f"  accelerator.mixed_precision = "
                  f"{getattr(acc, 'mixed_precision', '<none>')}")
            print(f"  scaler                      = "
                  f"{getattr(acc, 'scaler', None)}")
            opt = getattr(inner, "optimizer", None)
            if opt is not None:
                pd, gd = Counter(), Counter()
                groups = getattr(opt, "param_groups", [])
                for g in groups:
                    for q in g.get("params", []):
                        pd[str(q.dtype)] += 1
                        if q.grad is not None:
                            gd[str(q.grad.dtype)] += 1
                print(f"  optimizer {type(opt).__name__}: "
                      f"params={dict(pd)} grads={dict(gd)}")
        print("=" * 70 + "\n")
        traceback.print_exc()
        raise
    elapsed = time.time() - started

    trainer.tokenizer.save_pretrained(out)
    (out / "run.json").write_text(json.dumps({
        "arm": args.arm,
        "base_model": HUB_ID,
        "base_model_loaded_from": BASE,
        "data": str(data.relative_to(ROOT)),
        "hyperparameters": {k: v for k, v in opts.items() if k != "quantization"},
        "quantization": opts["quantization"],
        "dtype_recast": moved,
        "dtypes": census,
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
