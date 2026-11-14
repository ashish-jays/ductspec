"""Write each arm as conversational JSONL for the trainer.

Also measures what the two arms actually expose the model to. Matching record
counts is not matching token exposure: an adapter that saw more supervised
tokens had more to learn from, and the difference between the arms is supposed
to be which requirements were sound.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec.model import BASE_MODEL  # noqa: E402
from ductspec.prompt import to_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "data" / "build"


def main() -> None:
    missing = [str((BUILD / arm / f"{split}.jsonl").relative_to(ROOT))
               for arm in ("raw", "curated") for split in ("train", "dev", "test")
               if not (BUILD / arm / f"{split}.jsonl").exists()]
    if missing:
        raise SystemExit(
            "The arms have not been built: " + ", ".join(missing)
            + ". Run `python scripts/build_arms.py` first.")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as exc:                      # offline / ungated
        print(f"(no tokenizer: {exc}; skipping token measurement)")
        tok = None

    for arm in ("raw", "curated"):
        src = BUILD / arm
        out = BUILD / "messages" / arm
        out.mkdir(parents=True, exist_ok=True)

        for split in ("train", "dev", "test"):
            rows = [json.loads(line) for line in (src / f"{split}.jsonl").open()]
            examples = [to_messages(r) for r in rows]
            (out / f"{split}.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in examples))

            if tok is None or split != "train":
                print(f"{arm:8} {split:6} {len(rows):4} examples")
                continue

            prompt_tokens = [len(tok.encode(e["messages"][0]["content"])) for e in examples]
            target_tokens = [len(tok.encode(e["messages"][1]["content"])) for e in examples]
            verdicts = Counter(r["target"]["verdict"] for r in rows)
            print(
                f"{arm:8} {split:6} {len(rows):4} examples | "
                f"prompt tok sum={sum(prompt_tokens):,} max={max(prompt_tokens)} | "
                f"supervised tok sum={sum(target_tokens):,} max={max(target_tokens)} | "
                f"{dict(verdicts)}"
            )

    print(f"\n-> {(BUILD / 'messages').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
