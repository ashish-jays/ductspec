# Dataset v2: pre-training readiness review

Branch `exp/dataset-v2`. No model has been trained on this dataset. The final
test set (333 rows) is byte-for-byte the one every earlier model was scored on.

Every number below is produced by a script in this repo:
`scripts/readiness_audit.py --train-size 1200 --predictions <last run>` and
`scripts/dataset_report.py --train-size 1200`.

## Verdict

| Question | Answer |
|---|---|
| Ready to train? | **Yes, for one controlled round**, after the two fixes already made here (disputed items held back, probe twin moved to validation). |
| Biggest risk | 58% of training rows rest on 22 machine-extracted requirements no person has checked. An independent second reading agrees on 22 of 24 items; the 2 it disputes are held back. |
| Is 1,200 rows right? | For Tiny Aya (2 passes, about 11 h on a T4): yes. For 8B: probably too many for one 12-hour Kaggle session at one pass; the speed test decides, likely 800 to 1,000. |
| Is the schema sufficient? | For this round, yes. It cannot yet express conflicting evidence, overrides, revisions, tolerances or negations; those wait for the next round. |
| Highest-value remaining improvement | A person checking the 22 items (about an hour of an engineer's time; it covers 58% of training). The highest-value *model* change, teaching lettered clause ids such as "2.1.A", is already in this dataset. |

## Evidence, question by question

### 1. Does the 1,200-row sampler hide concentration?

- 26 clause families (curated), 212 generator families; median 5 rows per family, max 12.
- Every answer type draws from 14 to 15 clause families, and no family holds more than 8% of any type.
- **Hidden concentration found: Dartmouth's pressure rules.** 4 of the 5 largest clause families are Dartmouth (3.2.A, 3.2.B, 1.4.A, 1.4.B); together with the other pressure rules, pressure questions are 24% of training. The test set has **no** pressure questions. This is legitimate variety for real reviews, but it does not serve the current test. Not changed: tuning the property mix to the test would be test-informed selection. Revisit if the 8B budget forces a smaller set.

### 2. Authority and family independence

- Splits are by family (one requirement on one sheet), never by row; `ductspec.leakage` checks it, and each of its 6 checks is mutation-tested.
- Authorities: training = UFGS, Dartmouth, Houston, NYCHA; validation = Ohio; test = VA (plus UFGS known-authority rows and the probe). No authority appears in two of these roles.

### 3. Shared engineering rules (semantic duplicates)

Four rules appear, in different words, both in training and in dev or test:

| Rule | Training | Dev / test |
|---|---|---|
| Duct is galvanized steel | Dartmouth 3.2.B (two lines), NYCHA 2.02.A | Ohio 1.4.A (dev) |
| An unmade material choice | Dartmouth 3.2.B.4, Houston 1.2.O | UFGS 3.2.7.1 probe (test); Houston 1.2.L (dev) |
| Duct liner at least 1 in. | UFGS 2.12.11.3 | VA 2.3.B (test), Ohio 1.4.F (dev) |
| Double-wall inner skin galvanized | UFGS 2.12.17, Dartmouth 2.6.A | UFGS 2.12.17 (test) |

This is intended: the transfer question is "can the model apply a familiar
rule written by an unfamiliar issuer". It means the validation and transfer
scores measure **rule transfer across wording and numbering**, not discovery of
new engineering rules, and the write-up must say so.

**Fixed:** Houston 1.2.L (kitchen exhaust, "[316L stainless] [black] steel")
is the probe clause in another issuer's words. Training on it would have made
the probe a recall test. It is now held out for validation
(`data/unadopted.yaml`, `validation: true`), and the leakage check enforces it.

### 4. The 24 extracted items

- `data/review_queue.yaml` lists each item, the judgement call made, and an
  **independent second reading** (a separate model given only the clause text
  and the sentence, not our reading).
- Agreement: 22 of 24 on property, systems, condition and accepted values.
- **Disputed, held back:** Dartmouth 3.2.A exhaust (second reading: every
  exhaust system, not general exhaust only) and Houston 1.2.J (second reading:
  return boots and elbows are always insulated; only "other return ductwork" is
  conditional). `ductspec.requirements.held_back()` removes them from every split.
- Still unverified by a person: 22 items, which carry **58% of training rows and
  80% of dev rows**. The second reading is a second model, not an engineer.

### 5. Synthetic drawings

All drawings are generated: one duct, one property, lettering from a fixed
grammar (tag, range note, general note, schedule row), one distractor.
Missing compared with real sheets: multiple sheets referring to one duct,
revision clouds, legends and abbreviation tables, details and sections,
equivalent notations ("G90", "GALV.", "GALVANIZED"), metric units, illegible
or cropped text, and notes that override one another.

Authentic drawings: public agencies post bid drawing sets as PDFs (school
districts, housing authorities, airports). Using them needs (a) a licence
check per set and (b) a person to annotate each duct property, about 10 to 20
minutes per row. A 20 to 40 row real-drawing evaluation panel is the right
first use (experiment 6 below), not training data.

### 6. Kitchen exhaust (8% of training, 23% of test)

Not fixable by sampling without duplicating rows. It also must not be fixed
by drawing generic "every system" rules on kitchen ducts: NFPA 96 sets kitchen
exhaust apart, and a "galvanized for every system" rule drawn on a kitchen duct
would teach an engineering error. The right fix is more kitchen-specific
clauses from more authorities (owner specifications commonly state
"kitchen exhaust: 16 gauge black steel or stainless, welded"); each one costs
about 15 minutes to extract and check, plus review. Deferred to the next round.

### 7. Is the 402-row validation set representative?

| | dev (validation) | test |
|---|---|---|
| Panels | Ohio authority 320 (80%), UFGS known 82 | known 165, VA transfer 148, probe 20 |
| Applicability unknown | 7% (30 rows, after adding Houston 1.2.L) | 15% |
| Pressure questions | 9% | 0% |
| Systems | lab, shower and dishwasher exhaust 45% | supply 48%, kitchen 23% |
| Clause families | 11 | 12 |

It is independently sourced (Ohio is never trained on), but it is narrower
than the test: mostly material rules, few unmade choices, different systems.
Use it to **compare recipes**, read separately per panel, and never as an
estimate of the test score.

### 8 and 9. Does the answer format preserve the decision, or conflate steps?

The oracle (`ductspec.derive`) makes the decision in the right order:
authority, then applicability, then drawing evidence, then compare. The answer
keeps the outcome of each step in a separate field: `governing_ids`
(authority), `missing_evidence` (which step failed), `observation_ids`
(evidence), `verdict`. So copied value (INFERRED_VALUE_ONLY), drawing silent
(MISSING_OBSERVATION), unmade choice or unmet condition (APPLICABILITY_UNKNOWN)
and no clause (MISSING_AUTHORITY) are distinct labels.

Two real conflations remain, both small:
- **"Not applicable" is labelled MISSING_AUTHORITY** when a clause is supplied
  but is about another system or element: 4 training rows, 0 in dev and test.
- **APPLICABILITY_UNKNOWN covers two different things:** an unmade choice, and a
  condition the drawing does not establish. The action differs only by context.

Measured on the last run, this matters for transfer: on held-out VA rows whose
verdict was right, the commonest error was **INFERRED_VALUE_ONLY answered as
APPLICABILITY_UNKNOWN** (24 rows): on VA's conditional liner clause the model
reached for "condition unknown" when the real problem was a copied value. The
new conditional rules from Dartmouth and Houston give it that contrast to learn.

### F. Where the last models lost the VA rows

| | raw | curated |
|---|---|---|
| VA transfer rows wrong | 72 of 148 | 101 of 148 |
| of those, verdict right but another field wrong | 43 | 34 |
| VA 2.1.A rows citing **no clause at all** | 43 of 68 | 64 of 68 |

The single largest loss is the citation: the model has never seen a lettered
clause id and leaves `governing_ids` empty. The new authorities (1.4.A, 3.2.B,
2.02.A) teach exactly that.

### 10. Difficult cases still missing

| Case | Status |
|---|---|
| Copied value correct / incorrect | Covered |
| Conditional requirements | Covered (new: pressure class, position, "where shown") |
| Two acceptable alternatives | Covered (new: Houston 1.2.M, Ohio 1.4.B) |
| Unmade choice in prose | Covered (new: Dartmouth 3.2.B.4) |
| Cross-reference to another section | Covered (UFGS 3.2.11) |
| Equivalent notation, unit conversion | **Missing** |
| General note overridden by a specific tag ("U.N.O.") | **Missing** (the notes already say U.N.O.; no row tests it) |
| Conflicting values on two sheets | **Missing**; needs an UNRESOLVED_CONFLICT path in `derive` |
| Superseded revision | **Missing** |
| Tolerances, ranges, "at most" | **Missing**; only "in" and "at least" exist |
| Negation ("shall not be insulated") | **Missing**; needs a zero value, which would change existing test rows |
| Exceptions ("uninsulated, except ...") | Partly: encoded as explicit system lists |

## Enrichment experiments for the next round

| Experiment | What it is | Expected benefit | Cost |
|---|---|---|---|
| 1. Counterfactual pairs | Already structural: each family is one duct changed in one way. Add a pair-consistency score (both members right) to the report | Shows whether the model reads the change or guesses; no new data | About 1 hour of code |
| 2. Cross-authority semantic pairs | The 4 shared rules above, scored together: does the model answer the same rule the same way in each issuer's words? | Direct measure of transfer; exposes wording sensitivity | Low: data exists; each new pair is about 15 minutes to extract plus review |
| 3. Hard negatives | Values next to the threshold (1.5 against 2 in.), near-identical tokens (SST304 against SST316), right value lettered on the wrong duct | Fewer wrong approvals on near misses | About 2 hours of generator code; labels are computed, so no annotation |
| 4. Multi-sheet evidence | The same duct on a plan tag and a schedule row, agreeing; both must be cited | Teaches combining evidence | About 4 hours: generator, and evidence lists in the oracle |
| 5. Conflicting revisions | Two sheets disagree; with a revision mark the latest governs, without one the answer is UNRESOLVED_CONFLICT with an RFI | Teaches the model not to pick one value silently | About 6 hours, plus an engineer confirming the label rule |
| 6. Real against synthetic | 20 to 40 rows from public bid drawing sets, annotated by a person, as an evaluation panel only | The only measure of real-world validity | 5 to 10 hours of engineer time, plus a licence check per drawing set |

## Changes made in this review

| File | Change | Test |
|---|---|---|
| `data/review_queue.yaml` | Second reading per item; 2 items `disputed` | `tests/test_review_queue.py` |
| `src/ductspec/requirements.py` | `held_back()`, `validation_unadopted()` | same, and `tests/test_train_contexts.py` |
| `src/ductspec/generate.py` | Disputed and rejected items produce no rows; validation unmade choices go to dev | `test_a_disputed_or_rejected_item_produces_no_rows`, `test_the_probe_twin_is_held_out_for_validation_and_never_trained` |
| `data/unadopted.yaml` | Houston 1.2.L moved from training to validation | as above |
| `src/ductspec/leakage.py` | Validation unmade choices may never be trained, and may be scored | `test_a_validation_unmade_choice_in_training_is_caught` |
| `scripts/readiness_audit.py` | The measurements in this document | run it |

Local suite: 185 tests pass. The test set is unchanged.

## Before training

1. A person checks the 22 items in `data/review_queue.yaml` (recommended, not blocking: training can go ahead with their status stated in the write-up).
2. Add the 4 new PDFs to the Kaggle sources dataset (a new version).
3. 8B speed and memory test, which fixes the 8B training size.
4. Two runs on this data with the last run's recipe: Tiny Aya (isolates the data effect) and 8B (isolates the model effect).
