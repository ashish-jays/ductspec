"""Measure the sequence budget before configuring a training run.

Tiny Aya's chat template prepends a system preamble. SafeTune's notes warn that
setting max_len below it makes the data loaders train on zero supervised tokens
without raising — so the budget gets measured, not guessed.
"""

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from ductspec.cases import build_record  # noqa: E402
from ductspec.clauses import parse_clauses  # noqa: E402
from ductspec.objects import DuctObject, observed  # noqa: E402

MODEL = "CohereLabs/tiny-aya-global"
ROOT = Path(__file__).resolve().parents[1]


def n_tokens(tk, messages) -> int:
    out = tk.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # transformers v5 returns a BatchEncoding, which is a UserDict and so fails
    # isinstance(out, dict). Ask for the key instead of the type.
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def main() -> None:
    tk = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))

    overhead = n_tokens(tk, [{"role": "user", "content": ""}])
    print(f"chat-template overhead (empty user turn): {overhead} tokens")

    clauses = [c for c in parse_clauses(ROOT / "data/sources/UFGS-23-30-00.pdf", "UFGS-23-30-00") if c.operative]
    lens = sorted(len(tk.encode(c.text)) for c in clauses)
    print(
        f"operative clause tokens: median={statistics.median(lens):.0f} "
        f"p75={lens[int(.75 * len(lens))]} p90={lens[int(.9 * len(lens))]} max={max(lens)}"
    )
    print(f"clauses <= 400 tokens: {sum(1 for n in lens if n <= 400)}/{len(lens)}")

    duct = DuctObject(
        "D-101", "supply_air", 20, 12,
        {"material": observed("galvanized_steel", "D-101", text="D-101: 20x12 GALV SA")},
    )
    short = [c for c in clauses if len(tk.encode(c.text)) <= 400][:3]
    evidence = [{"id": c.clause_id, "text": c.text} for c in short]
    rec = build_record(
        "BUDGET", "Does D-101's shown material meet the supply-duct requirement?",
        duct, evidence, {"id": "P-1", "text": "Compare the required material with the shown material."},
        scope="material", verdict="COMPLIANT", governing_ids=[evidence[0]["id"]],
        action="NONE_FOR_THIS_CHECK",
    )
    prompt = n_tokens(tk, [{"role": "user", "content": json.dumps(rec["input"], indent=1)}])
    target = len(tk.encode(json.dumps(rec["target"], indent=1)))
    print(f"packet (3 clauses): prompt={prompt} target={target} total={prompt + target}")
    print(f"suggested max_len: {max(1536, (prompt + target) + 256)}")


if __name__ == "__main__":
    main()
