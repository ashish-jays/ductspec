"""Self-hosting against closed-model token billing, from measured numbers.

Every figure here comes from a file this project produced: throughput from the
generation run's metadata, training time from the training run's, packet sizes
from the dataset. Nothing is estimated except the closed-model price, which is
published, and the review volume, which is a stated assumption.

The honest trap in this comparison is that it is easy to win by choosing the
volume. A GPU costs the same whether it is busy or idle, so self-hosting looks
terrible at low volume and unbeatable at high volume, and the author picks.
This script therefore reports the break-even volume rather than a single
verdict, and prints cost per *supported* answer — a cheap answer that cannot
be relied on is not cheaper.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Published list prices, USD per million tokens. Stated so they can be checked
# and corrected; they move.
CLOSED_MODELS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-haiku": {"input": 0.80, "output": 4.00},
}


def load_measurements() -> dict:
    meta = ROOT / "data" / "predictions" / "curated.meta.json"
    run = ROOT / "data" / "adapters" / "curated" / "run.json"
    out = {
        "tokens_per_second": None, "gpu_hourly_usd": 0.80,
        "training_hours": None, "prompt_tokens": None, "output_tokens": None,
    }
    if meta.exists():
        m = json.loads(meta.read_text())
        out["tokens_per_second"] = m["tokens_per_second"]
        out["output_tokens"] = m["generated_tokens"] / m["rows"]
    if run.exists():
        r = json.loads(run.read_text())
        out["training_hours"] = r["wall_clock_seconds"] / 3600
        out["gpu_hourly_usd"] = r.get("gpu_hourly_usd", 0.80)

    rows = [json.loads(line) for line in (ROOT / "data/build/curated/test.jsonl").open()]
    out["rows"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=100_000,
                    help="reviews per month")
    ap.add_argument("--prompt-tokens", type=int, default=870)
    ap.add_argument("--output-tokens", type=int, default=70)
    ap.add_argument("--supported-rate", type=float, default=None,
                    help="fraction of answers that are strictly supported")
    args = ap.parse_args()

    m = load_measurements()
    tps = m["tokens_per_second"]
    if tps is None:
        print("No generation metadata yet — run scripts/predict.py on the GPU first.\n"
              "Printing the shape of the table with a placeholder throughput.")
        tps = 0.0

    out_tok = int(m["output_tokens"] or args.output_tokens)
    hourly = m["gpu_hourly_usd"]

    print(f"Measured: {tps:.1f} output tok/s on {m.get('gpu', 'L4')} at ${hourly:.2f}/h")
    print(f"Packet {args.prompt_tokens} prompt + {out_tok} output tokens, "
          f"{args.volume:,} reviews/month\n")

    if tps > 0:
        seconds = args.volume * out_tok / tps
        gpu_hours = seconds / 3600
        serve_cost = gpu_hours * hourly
        train_cost = (m["training_hours"] or 0) * hourly * 2  # both arms
        print(f"Self-hosted: {gpu_hours:,.1f} GPU-hours = ${serve_cost:,.2f}"
              f"  (+ ${train_cost:,.2f} one-off training)")
    else:
        serve_cost = float("nan")

    print()
    for name, price in CLOSED_MODELS.items():
        cost = (args.volume * args.prompt_tokens * price["input"]
                + args.volume * out_tok * price["output"]) / 1_000_000
        print(f"{name:14} ${cost:>10,.2f}/month", end="")
        if tps > 0 and cost > 0:
            breakeven = serve_cost / cost * args.volume
            print(f"   break-even at {breakeven:,.0f} reviews/month")
        else:
            print()

    if args.supported_rate:
        print(f"\nPer *supported* answer, at {args.supported_rate:.0%} strict accuracy:")
        for name, price in CLOSED_MODELS.items():
            cost = (args.volume * args.prompt_tokens * price["input"]
                    + args.volume * out_tok * price["output"]) / 1_000_000
            print(f"  {name:14} ${cost / (args.volume * args.supported_rate):.5f}")

    print("\nStated assumptions: list prices above, greedy decoding, one review per "
          "packet, no batching, GPU billed only while serving. Batching would move "
          "self-hosting in its favour; idle time would move it the other way.")


if __name__ == "__main__":
    main()
