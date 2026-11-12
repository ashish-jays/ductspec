"""Build the two training arms and the shared evaluation set.

    python -m ductspec.sources      # fetch the corpus the gates read
    python scripts/build_arms.py    # data/build/raw, data/build/curated
    python scripts/export_arms.py   # the same, as conversations for the trainer

Every later step reads data/build/<arm>/. Nothing else writes it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ductspec.curate import CURATED_GATES, curate  # noqa: E402
from ductspec.generate import ROOT, build_arms, write_arms  # noqa: E402
from ductspec.requirements import load  # noqa: E402


def main() -> None:
    _, rejected = curate(load(ROOT / "data" / "requirements.raw.yaml"), CURATED_GATES)
    print(f"gates rejected {len(rejected)} raw requirement(s):")
    for r in rejected:
        print(f"  {r.metadata['requirement_id']:28} {r.rejecting_step:18} {r.rejection_reason}")
    print()
    write_arms(build_arms())


if __name__ == "__main__":
    main()
