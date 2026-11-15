"""Near-duplicate leak check: train against dev and test.

Exact Jaccard similarity over word 5-gram shingles (what MinHash estimates),
for every train x scored pair, on two views of each row:
  full      the whole rendered prompt
  case      what makes a case unique: the question, the duct and its drawing
            notes (clause text and the fixed policy text removed)
Reported per scored panel: how many scored rows have a training row above
0.8 / 0.9 similarity, and the highest similarity found.
Run from the ductspec repo with PYTHONPATH=src. Usage: near_dup.py <arm>
"""
import json
import re
import sys
from collections import defaultdict

from ductspec.prompt import render

arm = sys.argv[1] if len(sys.argv) > 1 else "curated"


def shingles(text, n=5):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def case_view(inp):
    keep = {k: v for k, v in inp.items() if k not in ("evidence", "policy", "documents_supplied")}
    return json.dumps(keep, sort_keys=True)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


train = load(f"data/build/{arm}/train.jsonl")
scored = [("dev", r) for r in load("data/build/curated/dev.jsonl")] + \
         [("test", r) for r in load("data/build/curated/test.jsonl")]

for view_name, view in (("full", lambda r: render(r["input"])), ("case", lambda r: case_view(r["input"]))):
    tr = [shingles(view(r)) for r in train]
    stats = defaultdict(lambda: {"n": 0, "over08": 0, "over09": 0, "identical": 0, "max": 0.0})
    worst = (0.0, None, None)
    for split, r in scored:
        s = shingles(view(r))
        best, best_id = 0.0, None
        for t_row, t in zip(train, tr):
            j = len(s & t) / len(s | t)
            if j > best:
                best, best_id = j, t_row["id"]
        k = stats[f"{split}/{r['panel']}"]
        k["n"] += 1
        k["over08"] += best >= 0.8
        k["over09"] += best >= 0.9
        k["identical"] += best >= 0.999
        k["max"] = max(k["max"], best)
        if best > worst[0]:
            worst = (best, r["id"], best_id)
    print(f"\n[{arm} train vs scored, view={view_name}]")
    print(f"  {'panel':28} rows  >=0.8  >=0.9  identical  max")
    for name, k in sorted(stats.items()):
        print(f"  {name:28} {k['n']:4}  {k['over08']:5}  {k['over09']:5}  {k['identical']:9}  {k['max']:.3f}")
    print(f"  closest pair: {worst[1]} ~ {worst[2]} ({worst[0]:.3f})")
