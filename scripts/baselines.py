"""Score three reference behaviours through AuditKIT, before any training.

An evaluation is worth no more than its resistance to being gamed, and the
cheapest way to game a grounding benchmark is to stop answering. These three
runs establish what the metrics do under the two degenerate strategies and
under a perfect answer, so the numbers a trained model earns later can be read
against something.

    oracle            answers every packet with the derived truth
    always-compliant  certifies everything
    always-abstain    certifies nothing
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import auditkit as ak  # noqa: E402

from ductspec.audit import scorers, to_samples  # noqa: E402
from ductspec.prompt import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ALWAYS_COMPLIANT = json.dumps({
    "verdict": "COMPLIANT", "governing_ids": [], "observation_ids": [],
    "missing_evidence": [], "action": "NONE_FOR_THIS_CHECK",
})
ALWAYS_ABSTAIN = json.dumps({
    "verdict": "INSUFFICIENT_EVIDENCE", "governing_ids": [], "observation_ids": [],
    "missing_evidence": ["MISSING_OBSERVATION"], "action": "REQUEST_MISSING_DETAIL",
})


def main(split: str = "test") -> None:
    rows = [json.loads(line) for line in (ROOT / "data" / "build" / f"{split}.jsonl").open()]
    by_prompt = {render(r["input"]): r for r in rows}

    def oracle(prompts):
        return [json.dumps(by_prompt[p]["target"]) for p in prompts]

    result = ak.compare_models(
        [oracle, lambda p: [ALWAYS_COMPLIANT] * len(p), lambda p: [ALWAYS_ABSTAIN] * len(p)],
        to_samples(rows),
        scorers=scorers(),
        model_names=["oracle", "always-compliant", "always-abstain"],
        experiment_name="ductspec-baselines",
        tags=["baseline"],
    )
    print(result.summary())

    out = ROOT / "data" / "build" / f"baselines-{split}.txt"
    out.write_text(result.summary())
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
