"""Measure a built dataset at every level that matters, and check it for leaks.

    python scripts/dataset_report.py --train-size 1200 > dataset_report.md

Prints Markdown: sizes, then for each split the distribution by authority,
clause family, requirement, answer type, property, duct system, drawing sheet,
how the value is shown on the drawing, and panel; prompt uniqueness; and the
leakage checks from ductspec.leakage. Every number in the dataset card comes
from here.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec import leakage  # noqa: E402
from ductspec.drawing import shown_for  # noqa: E402
from ductspec.generate import build_arms  # noqa: E402
from ductspec.prompt import render, view  # noqa: E402
from ductspec.sample import answer_type  # noqa: E402


def how_shown(row: dict) -> str:
    v = view(row["input"])
    lines = shown_for(v["drawing"], v["duct"]["id"], v["duct"]["system"])
    if not lines:
        return "not shown"
    text = next(item["text"] for item in v["drawing"] if item["id"] == lines[0])
    if " | " in text:
        return "schedule row"
    if " THRU " in text:
        return "range note"
    if text.startswith("ALL "):
        return "general note"
    return "own tag"


LEVELS = {
    "authority": lambda r: r["meta"]["requirement"].split("-")[1],
    "answer type": answer_type,
    "property": lambda r: r["input"]["property"],
    "duct system": lambda r: r["input"]["object"]["system"],
    "drawing sheet": lambda r: r["input"]["drawing"][0]["id"].split("-")[0],
    "how the value is shown": how_shown,
    "panel": lambda r: r["panel"],
}


def table(rows: list[dict], name: str, key) -> str:
    counts = Counter(key(r) for r in rows)
    lines = [f"| {name} | rows | share |", "|---|---|---|"]
    for k, v in counts.most_common():
        lines.append(f"| {k} | {v} | {v / len(rows):.0%} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=None)
    args = ap.parse_args()
    arms = build_arms(train_size=args.train_size)

    print("# Dataset report\n")
    print(f"Training set size per arm: {args.train_size or 'every row'}.\n")
    print("| arm | train | dev | test |\n|---|---|---|---|")
    for arm, splits in arms.items():
        print(f"| {arm} | {len(splits['train'])} | {len(splits['dev'])} | {len(splits['test'])} |")

    for arm, splits in arms.items():
        for split in ("train", "dev", "test") if arm == "curated" else ("train",):
            rows = splits[split]
            print(f"\n## {arm} {split}: {len(rows)} rows\n")
            clauses = Counter(tuple(r["meta"]["requirement"].split("-")[1:3]) for r in rows)
            reqs = Counter(r["meta"]["requirement"] for r in rows)
            prompts = Counter(render(r["input"]) for r in rows)
            print(f"- clause families: {len(clauses)} (rows per family: min {min(clauses.values())}, "
                  f"max {max(clauses.values())})")
            print(f"- requirements: {len(reqs)}")
            print(f"- distinct prompts: {len(prompts)} of {len(rows)} rows")
            for name, key in LEVELS.items():
                print("\n" + table(rows, name, key))

    print("\n## Leakage\n")
    for arm, splits in arms.items():
        found = leakage.check(splits)
        print(f"- {arm}: " + ("no leak found by any check" if not found
                                 else "; ".join(f"{k}: {len(v)}" for k, v in found.items())))
    shared = leakage.shared_clauses(arms["curated"])
    print(f"- clauses in both train and test (known-authority panel, allowed): {len(shared)}")


if __name__ == "__main__":
    main()
