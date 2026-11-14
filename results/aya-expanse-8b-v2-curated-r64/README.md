# Aya Expanse 8B, dataset v2, curated arm, rank 64

**The selected model.** Chosen on dev (402 rows) by fewest wrong approvals,
then fully correct, before the test set was read for selection.

Same frozen 333-row test set as every earlier model. Same settings as
`../aya-expanse-8b-raw-r64`: rank 64, 4 passes, lr 2e-4, `--precision fp32`,
`adamw_torch`, 4-bit nf4. Two things differ: the data (dataset v2, 1,200
balanced rows) and the loss covering the answer only (`ductspec.answer_mask`).

Kaggle, Tesla T4: 334 minutes to train, 14 minutes to score the test set.
The adapter weights (321 MB) are not in git.

## Test (333 rows, AuditKIT)

| | Untrained 8B | 8B raw, old data | 8B raw, v2 | **8B curated, v2** |
|---|---|---|---|---|
| Fully correct | 2.7% | 72.1% | 89.5% | **91.0%** |
| Wrong approvals | 57.4% | 2.4% | 3.6% | **1.2%** |
| Unsupported verdicts | 94.3% | 6.6% | 6.6% | **2.1%** |
| Unneeded refusals | 0.0% | 11.7% | **0.0%** | 1.8% |
| Correct citation | 19.2% | 82.9% | 93.1% | **95.8%** |
| Correct reason for abstaining | 4.0% | 73.1% | 84.6% | **92.0%** |
| Correct action | 25.5% | 73.6% | 90.4% | **91.9%** |

By panel (fully correct, wrong approvals):

| Panel | raw, old data | raw, v2 | **curated, v2** |
|---|---|---|---|
| Curation probe (20) | 0/20, 7 | 0/20, 8 | **20/20, 0** |
| Transfer, VA never trained (148) | 85, 1 | **134**, 3 | 127, 4 |
| Known authority, UFGS (165) | 155, 0 | **164**, 1 | 156, 0 |

## Dev (402 rows): the selection

| | raw, v2 | **curated, v2** |
|---|---|---|
| Wrong approvals | 4.7% | **2.7%** |
| Fully correct | 89.8% | **92.8%** |
| Correct citation | 93.0% | **97.8%** |

## Reading it

- **Data makes it useful.** Old data to v2 (raw, same settings): fully correct
  72.1% to 89.5%, VA 85 to 134 of 148, unneeded refusals 11.7% to 0%.
- **Curation makes it safe.** Raw v2 to curated v2 (same data and settings):
  probe 0/20 to 20/20, wrong approvals 3.6% to 1.2%.
- **Data alone is not safety.** Raw v2 approves more than raw on old data
  (3.6% against 2.4%); 8 of its 12 wrong approvals are the probe.
- **The cost.** Curated is lower than raw v2 on VA (127 against 134) and on
  UFGS (156 against 164), and refuses 1.8% of answerable rows.

## What this comparison does not control for

- The answer-only loss is on for both v2 arms and off (in effect) for the
  old-data run, so "old data to v2" is data plus loss, not data alone.
- The untrained column is the run in `../aya-expanse-8b-base`.
- The test set was read after dev selection, for this report.
