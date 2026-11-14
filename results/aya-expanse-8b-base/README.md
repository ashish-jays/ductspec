# Aya Expanse 8B, untrained

The "before" half of the comparison. Stock Aya Expanse 8B, no adapter, on the
same frozen 333 rows, greedy, loaded 4-bit nf4 exactly as the trained model
was scored. It differs from `../aya-expanse-8b-raw-r64` by the adapter and
nothing else.

Generated on Kaggle, one Tesla T4, 55.7 minutes, 12.4 tokens/s.

## What the untrained model does

| Metric | Untrained | Trained (raw r64) | always-compliant |
|---|---|---|---|
| Fully correct | 2.7% | **72.1%** | 0.0% |
| Wrong approvals | **57.4%** | **2.4%** | 71.5% |
| Unsupported verdicts | 94.3% | 6.6% | 100% |
| Unneeded refusals | 0.0% | 11.7% | 0.0% |
| Correct citation | 19.2% | 82.9% | 18.6% |
| Correct action | 25.5% | 73.6% | 28.5% |
| Coverage | 99.7% | 40.8% | 100% |
| Schema validity | 100% | 100% | 100% |

Read the first and last columns together. The untrained model answers 99.7% of
the rows, refuses none of them, and gets 2.7% fully right. Its citation and
action scores sit within a point of the always-compliant baseline's. It is,
to a close approximation, **a model that certifies whatever it is shown** —
the precise failure this project was built to measure.

It is not incapable. Schema validity is 100%: Aya Expanse produces a
correctly shaped answer on every row, zero-shot, without ever having seen the
format. What it cannot do is decline. 57.4% of its answers approve something
the drawing does not support.

Training moves fully-correct from 2.7% to 72.1% and wrong approvals from
57.4% to 2.4% — a 96% reduction. Over-refusal rises from 0.0% to 11.7%, which
is the cost: the trained model sometimes asks for evidence it already has.
That is the trade the curated-versus-raw question is about.

## Scale alone makes it worse

Tiny Aya's untrained figures, from `DuctSpec_Training_Analysis.md`:

| | Tiny Aya 3.35B | Aya Expanse 8B |
|---|---|---|
| Fully correct | 0.0% | 2.7% |
| Wrong approvals | 24.3% | **57.4%** |
| Correct citation | 20.7% | 19.2% |
| Correct action | 10.2% | 25.5% |

The larger untrained model approves more than twice as often. It is more
fluent, more willing, and more wrong — it writes a well-formed verdict for
every row and certifies more than half of them without support. Whatever the
8B model's advantage is, it does not arrive as caution, and it is not
something scale supplies on its own. The trained pair shows the reverse
ordering (2.4% against Tiny Aya's 7.8%), so the caution is bought by the
fine-tuning, not by the parameters.

## Reproducing

```bash
python scripts/predict.py base --split data/build/curated/test.jsonl
python scripts/evaluate.py --states base
```

No `--adapter`. `predict.py` loads 4-bit nf4 with an fp16 compute dtype by
default, which is how the trained adapter was scored.
