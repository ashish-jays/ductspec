"""Ask a Gemini model the same questions, with the same prompt, and score it the Stage 1 way.

    PYTHONPATH=src python scripts/frontier_gemini.py --env PATH --list
    PYTHONPATH=src python scripts/frontier_gemini.py --env PATH --model MODEL --ids ID [ID ...]

The key is read from GEMINI_API_KEY (environment, or the --env file) and is
never printed. The prompt is ductspec.prompt.render, exactly what the trained
models see; temperature 0. Outputs go to results/stage2-frontier/<model>.jsonl
with the raw text kept, so they can be re-scored.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.parsing import parse_prediction  # noqa: E402
from ductspec.prompt import render  # noqa: E402

API = "https://generativelanguage.googleapis.com/v1beta"
FIELDS = ("verdict", "governing_ids", "observation_ids", "missing_evidence", "action")


def key_from(env_path):
    key = os.environ.get("GEMINI_API_KEY")
    if not key and env_path:
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("no GEMINI_API_KEY in the environment or the --env file")
    return key


def call(url, key, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def norm(field, value):
    return sorted(value or []) if field.endswith("ids") or field == "missing_evidence" else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--rows", default="data/build/curated/test.jsonl")
    args = ap.parse_args()
    key = key_from(args.env)

    if args.list:
        models = call(f"{API}/models?pageSize=200", key)["models"]
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].split("/")[-1], "|", m.get("displayName", ""))
        return

    rows = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / args.rows, encoding="utf-8")}
    out_dir = ROOT / "results/stage2-frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model}.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        for rid in args.ids:
            row = rows[rid]
            started = time.time()
            resp = call(f"{API}/models/{args.model}:generateContent", key, {
                "contents": [{"role": "user", "parts": [{"text": render(row["input"])}]}],
                "generationConfig": {"temperature": 0},
            })
            parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            usage = resp.get("usageMetadata", {})
            fh.write(json.dumps({"id": rid, "model": args.model, "output": text, "usage": usage,
                                 "seconds": round(time.time() - started, 1)}) + "\n")
            p = parse_prediction(text) or {}
            t = row["target"]
            print(f"\n== {rid}  ({time.time() - started:.0f} s, tokens {usage})")
            print("model said:", text.strip()[:800])
            for f in FIELDS:
                ok = norm(f, p.get(f)) == norm(f, t.get(f))
                print(f"  {'ok  ' if ok else 'MISS'} {f:17} model={p.get(f)!r}  target={t.get(f)!r}")
            print("fully correct:", all(norm(f, p.get(f)) == norm(f, t.get(f)) for f in FIELDS),
                  "| wrong approval:", p.get("verdict") == "COMPLIANT" and t["verdict"] != "COMPLIANT")
    print("\nsaved to", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
