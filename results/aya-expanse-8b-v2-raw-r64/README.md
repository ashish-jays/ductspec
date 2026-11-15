# Aya Expanse 8B, dataset v2, raw arm, rank 64

The raw half of the v2 pair. Identical to `../aya-expanse-8b-v2-curated-r64`
in everything but the training data's curation; the comparison and the
reading are in that folder's README.

Test (333 rows): fully correct 89.5%, wrong approvals 3.6%, unneeded refusals
0.0%, probe 0/20 with 8 wrong approvals.
Dev (402 rows): fully correct 89.8%, wrong approvals 4.7%.

Kaggle, Tesla T4: 303 minutes to train. The adapter (321 MB) is not in git.
