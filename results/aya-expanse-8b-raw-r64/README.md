# Aya Expanse 8B, raw arm, rank 64

The same dataset, the same prompts and the same frozen 333-row test set the
Tiny Aya models were scored on. The base model is the only thing this run set
out to change: 3.35B to 8B.

Run on Kaggle, two Tesla T4s, 61.5 minutes to train and 48.0 minutes to
generate. `run.json` carries the full configuration; `metrics.txt` is
AuditKIT's own output; `predictions.jsonl` is every generation, so a reader
who disagrees with a metric can recompute it. The adapter weights are 336MB
and are not in git.

## Against Tiny Aya

Tiny Aya figures are the raw rank-64 row of `DuctSpec_Training_Analysis.md`.

| | Tiny Aya 3.35B | Aya Expanse 8B | |
|---|---|---|---|
| Fully correct | 70.9% | **72.1%** | +1.2 |
| Wrong approvals | 7.8% | **2.4%** | **-5.4** |
| Unneeded refusals | 10.8% | 11.7% | +0.9 |
| Correct citation | **91.6%** | 82.9% | **-8.7** |
| Correct action | 73.3% | 73.6% | +0.3 |
| Schema validity | — | 100% | |

Read it as one result and one warning.

**The result.** Wrong approvals fall by about two thirds, 7.8% to 2.4%, while
overall correctness barely moves. That is the thing this project exists to
measure: the larger model is markedly less willing to certify something it
cannot see, and it does not buy that by refusing more often — over-refusal is
within a point. Against the always-compliant baseline's 71.5% false-approval
rate, 2.4% is the interesting number here, not the 72.1%.

**The warning.** Citation accuracy went *down*, 91.6% to 82.9%. The bigger
model is more cautious about verdicts and worse at naming the clause it
relied on. Nothing in the size hypothesis predicted that, and it should be
understood before this model is preferred over the small one.

## What this comparison does not control for

Two settings differ from the Tiny Aya runs, so the 8B column is not a clean
single-variable result and should not be quoted as one:

- **Effective batch is 16, not 8.** Kaggle allocates two T4s and the trainer
  used both, so gradient accumulation of 8 at batch 1 ran on two devices.
- **No mixed precision.** The trainer's AMP flags are off (`precision: fp32`
  in `run.json`) while the model itself is fp16. On a T4 this was not a
  preference: TRL casts the LoRA parameters to bfloat16 after they are set to
  fp32, and AMP's gradient scaler has no bfloat16 kernel to unscale them
  with, so any fp16 run dies at the first gradient clip. Turning AMP off is
  what removes the scaler from the path.

Re-running Tiny Aya under these same two settings would make the pair
comparable. Until that is done, the honest claim is that an 8B model trained
*this* way beats a 3.35B model trained *that* way on false approvals.

## Reproducing

`DUCTSPEC_BASE_MODEL` points the loader at the Kaggle model mount and
`DUCTSPEC_BASE_HUB_ID` names what the artifacts should report.

```bash
python scripts/train.py raw --lora-r 64 --epochs 4 --learning-rate 2e-4 \
    --precision fp32 --optimizer adamw_torch --output adapters/raw
python scripts/predict.py raw --adapter adapters/raw/final_adapter \
    --split data/build/curated/test.jsonl
python scripts/evaluate.py --states raw
```
