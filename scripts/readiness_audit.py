"""Pre-training readiness measurements for the dataset.

    python scripts/readiness_audit.py --train-size 1200 [--predictions DIR]

Each section prints evidence for one question in docs/READINESS.md:

  A  concentration inside the sampled training set
  B  engineering rules shared across authorities and splits
  C  how much of the training set rests on unverified extraction
  D  how the validation set compares with the test set
  E  answers whose label conflates "not applicable" with "no authority"
  F  (with --predictions) where earlier models lost the held-out VA rows:
     the decision, or the citation and action around it

Nothing here changes the data.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from ductspec.generate import build_arms  # noqa: E402
from ductspec.requirements import load, load_unadopted  # noqa: E402
from ductspec.sample import answer_type  # noqa: E402
from ductspec.sources import split_of  # noqa: E402

QUEUE = yaml.safe_load((ROOT / "data" / "review_queue.yaml").read_text(encoding="utf-8"))
UNVERIFIED = {e["id"] for e in QUEUE["entries"] if e["status"] == "needs_human_check"}


def clause(row):
    return tuple(row["meta"]["requirement"].split("-")[1:3])


def share(counter, total):
    return ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in counter.most_common())


def section_a(train):
    print("\n## A. Concentration in the training set\n")
    n = len(train)
    fams = Counter(clause(r) for r in train)
    top = fams.most_common(5)
    print(f"- {len(fams)} clause families; top 5 hold {sum(v for _, v in top) / n:.0%} of rows: "
          + "; ".join(f"{a} {c} {v}" for (a, c), v in top))
    cells = Counter((clause(r), answer_type(r)) for r in train)
    print(f"- (clause, answer type) cells: {len(cells)}; largest: "
          + "; ".join(f"{c[0][0]} {c[0][1]} {c[1]} {v}" for c, v in cells.most_common(4)))
    by_type = defaultdict(Counter)
    for r in train:
        by_type[answer_type(r)][clause(r)] += 1
    for kind, c in sorted(by_type.items()):
        biggest = c.most_common(1)[0]
        print(f"- {kind}: {sum(c.values())} rows from {len(c)} clause families; "
              f"largest family {biggest[0][0]} {biggest[0][1]} = {biggest[1] / sum(c.values()):.0%}")
    fam_rows = Counter(r["family"] for r in train)
    print(f"- distinct generator families (one requirement on one sheet): {len(fam_rows)}; "
          f"rows per family median {sorted(fam_rows.values())[len(fam_rows) // 2]}, max {max(fam_rows.values())}")


def rule_signature(req):
    pred = req.predicate
    accepted = tuple(sorted(map(str, pred.get("values", [])))) or (pred["op"], pred.get("value"))
    return (req.property, req.element, pred["op"], accepted)


def section_b(arms):
    print("\n## B. Engineering rules shared across authorities and splits\n")
    reqs = load() + [r for r, _ in load_unadopted()]
    used = defaultdict(set)
    for split in ("train", "dev", "test"):
        for r in arms["curated"][split]:
            used[r["meta"]["requirement"]].add(split)
    groups = defaultdict(list)
    for r in reqs:
        if r.id in used:
            groups[rule_signature(r)].append(r)
    shared = 0
    for sig, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        splits = set().union(*(used[m.id] for m in members))
        authorities = {m.source.split("-")[0] for m in members}
        if len(authorities) > 1 and len(splits) > 1:
            shared += 1
            print(f"- {sig[0]} / {sig[1]} / {sig[2]} {sig[3]}: "
                  + "; ".join(f"{m.id} [{','.join(sorted(used[m.id]))}]" for m in members))
    print(f"- {shared} rules appear, in different words, both in training and in dev or test")


def section_c(train, dev):
    print("\n## C. Rows resting on unverified extraction\n")
    for name, rows in (("train", train), ("dev", dev)):
        n = sum(r["meta"]["requirement"] in UNVERIFIED for r in rows)
        print(f"- {name}: {n} of {len(rows)} rows ({n / len(rows):.0%}) come from the {len(UNVERIFIED)} "
              "items still awaiting a human check")


def section_d(dev, test):
    print("\n## D. Validation set against test set\n")
    for name, key in (("panel", lambda r: r["panel"]), ("answer type", answer_type),
                      ("property", lambda r: r["input"]["property"]),
                      ("system", lambda r: r["input"]["object"]["system"])):
        print(f"- {name}\n  - dev:  {share(Counter(key(r) for r in dev), len(dev))}\n"
              f"  - test: {share(Counter(key(r) for r in test), len(test))}")
    print(f"- clause families: dev {len({clause(r) for r in dev})}, test {len({clause(r) for r in test})}")


def section_e(arms):
    print("\n## E. 'Not applicable' labelled as 'no authority'\n")
    for split in ("train", "dev", "test"):
        rows = arms["curated"][split]
        missing = [r for r in rows if r["target"]["missing_evidence"] == ["MISSING_AUTHORITY"]]
        not_applicable = [r for r in missing if r["input"]["evidence"]]
        print(f"- {split}: {len(missing)} MISSING_AUTHORITY rows, of which {len(not_applicable)} "
              "have a clause supplied that is simply about something else")


def section_f(pred_dir, test):
    print("\n## F. Where earlier models lost the held-out VA rows\n")
    from ductspec.parsing import parse_prediction
    from ductspec.scoring import score_one

    fields = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")
    for path in sorted(Path(pred_dir).glob("*_4bit.jsonl")):
        outs = {json.loads(l)["id"]: json.loads(l)["output"] for l in path.open(encoding="utf-8")}
        rows = [r for r in test if r["panel"] == "transfer" and r["id"] in outs]
        if not rows:
            continue
        wrong_field, verdict_right_strict_wrong = Counter(), 0
        for r in rows:
            p = parse_prediction(outs[r["id"]]) or {}
            if score_one(r, p)["strict"]:
                continue
            bad = [f for f in fields
                   if sorted(p.get(f) or []) != sorted(r["target"][f])] if True else []
            bad = [f for f in fields if (sorted(p.get(f) or []) if isinstance(r["target"][f], list)
                                          else p.get(f)) != (sorted(r["target"][f]) if isinstance(r["target"][f], list)
                                                             else r["target"][f])]
            wrong_field.update(bad)
            verdict_right_strict_wrong += p.get("verdict") == r["target"]["verdict"]
        wrong = len(rows) - sum(bool(score_one(r, parse_prediction(outs[r["id"]]))["strict"]) for r in rows)
        print(f"- {path.stem}: {wrong} of {len(rows)} transfer rows wrong; verdict right but another field "
              f"wrong in {verdict_right_strict_wrong}; wrong fields: {dict(wrong_field)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=1200)
    ap.add_argument("--predictions", default=None)
    args = ap.parse_args()
    arms = build_arms(train_size=args.train_size)
    train, dev, test = arms["curated"]["train"], arms["curated"]["dev"], arms["curated"]["test"]
    print(f"# Readiness measurements (curated arm, {len(train)} training rows)")
    section_a(train)
    section_b(arms)
    section_c(train, dev)
    section_d(dev, test)
    section_e(arms)
    if args.predictions:
        section_f(args.predictions, test)


if __name__ == "__main__":
    main()
