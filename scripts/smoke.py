"""Load the base model locally and answer three real packets.

This is the free check before any paid GPU minute. It establishes four things
that a training run would otherwise discover expensively: the model loads, the
chat template renders our packet, the generation comes back in a shape our
parser can read, and roughly how fast it runs.

It runs on Apple silicon via MPS. That is fine for inference — the model is
3.35B, about 6.7GB in bfloat16 — and impossible for training, because
bitsandbytes has no MPS backend. Training goes to HF Jobs.
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec.derive import Packet  # noqa: E402,F401  (kept for context)
from ductspec.parsing import parse_prediction  # noqa: E402
from ductspec.prompt import render  # noqa: E402
from ductspec.scoring import score_one  # noqa: E402

from ductspec.model import BASE_MODEL as MODEL  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]


def main(n: int = 3) -> None:
    rows = [json.loads(line) for line in (ROOT / "data/build/curated/test.jsonl").open()][:n]

    print(f"loading {MODEL} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    print(f"loaded on {device} in {time.time() - t0:.0f}s\n")

    parsed = 0
    for row in rows:
        prompt = render(row["input"])
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        ids = ids.to(device)

        t = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        new = out[0][ids.shape[-1]:]
        text = tok.decode(new, skip_special_tokens=True)
        elapsed = time.time() - t

        pred = parse_prediction(text)
        parsed += pred is not None
        scored = score_one(row, pred)

        print(f"--- {row['id']}  (gold {row['target']['verdict']}) ---")
        print(f"    prompt {ids.shape[-1]} tok | generated {len(new)} tok in "
              f"{elapsed:.1f}s ({len(new) / elapsed:.1f} tok/s)")
        print(f"    raw: {text.strip()[:220]}")
        print(f"    parsed: {pred['verdict'] if pred else 'UNPARSEABLE'} | strict={scored['strict']}\n")

    print(f"parseable: {parsed}/{len(rows)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
