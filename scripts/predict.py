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


def load(adapter: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    return tok, model.to(device).eval(), device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="base | raw | curated")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="data/build/curated/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / args.split).open()]
    if args.limit:
        rows = rows[: args.limit]

    tok, model, device = load(args.adapter)
    print(f"{args.name}: {len(rows)} rows on {device}")

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
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "wall_clock_seconds": round(elapsed, 1),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / elapsed, 2),
    }, indent=2) + "\n")
    print(f"{args.name}: {elapsed / 60:.1f} min, "
          f"{generated_tokens / elapsed:.1f} tok/s -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
