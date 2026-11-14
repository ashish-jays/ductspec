"""Generate answers for one model state, and keep every generation.

Generation and scoring are separate on purpose. Generation needs a GPU and
costs money; scoring is deterministic and free, and we will want to re-score
as the metrics are refined. Keeping the raw text also means the published
artifacts include what the model actually said, not only what we made of it —
a reviewer who disagrees with a metric can recompute it.

Decoding is greedy and identical for every model state. A comparison where one
state sampled and another did not would be measuring temperature.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec.prompt import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from ductspec.model import BASE_MODEL as BASE, HUB_ID  # noqa: E402
MAX_NEW_TOKENS = 160


def dtypes() -> tuple["torch.dtype", str]:
    """The compute dtype this GPU actually has, and its name for the record.

    The same pre-Ampere problem as training: Kaggle's T4 and P100 have no
    bfloat16, and torch.cuda.is_bf16_supported() says otherwise because it
    counts emulation. Scoring a model in a dtype the card has to emulate
    would make the comparison measure the card.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def load(adapter: str | None, quantize: bool = True):
    """Load the base model, in 4-bit by default, with the adapter on top.

    Quantized by default because that is how the adapter was trained, and
    because an 8B model in 16-bit is ~16GB — more than a T4 holds before any
    KV cache. `--no-quantize` is there for a card with room, and the choice
    is written into the meta file either way.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype, dtype_name = dtypes()
    tok = AutoTokenizer.from_pretrained(BASE)

    kwargs = {"dtype": dtype}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        # bitsandbytes places the quantized weights itself; a later .to()
        # on the model would undo it.
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"

    model = AutoModelForCausalLM.from_pretrained(BASE, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)

    if quantize:
        device = str(next(model.parameters()).device)
        return tok, model.eval(), device, dtype_name

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    return tok, model.to(device).eval(), device, dtype_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="base | raw | curated")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="data/build/curated/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-quantize", action="store_true",
                    help="load in 16-bit; the default matches training")
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / args.split).open()]
    if args.limit:
        rows = rows[: args.limit]

    tok, model, device, dtype_name = load(
        args.adapter, quantize=not args.no_quantize)
    loaded_in = "16-bit" if args.no_quantize else "4-bit nf4"
    print(f"{args.name}: {len(rows)} rows on {device} "
          f"({loaded_in}, compute {dtype_name})")

    out_path = ROOT / "data" / "predictions" / f"{args.name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    generated_tokens = 0
    with out_path.open("w") as fh:
        for i, row in enumerate(rows, 1):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": render(row["input"])}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            new = out[0][ids.shape[-1]:]
            generated_tokens += len(new)
            fh.write(json.dumps({
                "id": row["id"],
                "output": tok.decode(new, skip_special_tokens=True),
                "prompt_tokens": int(ids.shape[-1]),
                "generated_tokens": int(len(new)),
            }) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}  {generated_tokens / (time.time() - started):.1f} tok/s")

    elapsed = time.time() - started
    (out_path.with_suffix(".meta.json")).write_text(json.dumps({
        "model": args.name,
        "adapter": args.adapter,
        "base": HUB_ID,
        "base_loaded_from": BASE,
        "rows": len(rows),
        "device": device,
        "loaded_in": loaded_in,
        "compute_dtype": dtype_name,
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "wall_clock_seconds": round(elapsed, 1),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / elapsed, 2),
    }, indent=2) + "\n")
    print(f"{args.name}: {elapsed / 60:.1f} min, "
          f"{generated_tokens / elapsed:.1f} tok/s -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
