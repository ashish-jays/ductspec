# DuctSpec dataset review — 2026-11-11

Review only. No product code changed. Existing edits to `clauses.py` and the
untracked `screening.py` were present before this review.

## Verified state

- Read README, all eight source modules, manifest/lock, budget script, and tests.
- Fresh test run: **25 passed**, using:
  `PYTHONPATH=src uv run --no-project --python 3.12 --with pytest --with pypdf --with pyyaml pytest -q -p no:cacheprovider`
- All six downloaded PDFs match the checked-in SHA-256 values.
- Current shortlist computation returns **200**, whereas
  `data/build/shortlist.json` contains 207. Current per-source counts:
  23 30 00: 72; 23 07 00: 76; 23 05 93: 5; 23 08 00: 1;
  22 00 00: 33; NASA Metal Ducts: 13. This artifact needs regeneration.
- Current 23 30 00 parsing: 240 clauses, 169 marked operative.

## Decisions

### (a) Pressure

Keep it at 10% of rows as an exploratory slice. Do not add a document or invent
project requirements to balance a histogram. Material and insulation are the
main result. Report actual independent requirement-family counts, including
the small number behind pressure. Do not equate static operating pressure,
pressure rating, pressure drop, test pressure, or positive and negative ratings.

### (b) Split

Keep NASA Metal Ducts as the whole-document transfer panel. The shortlist is
partly an extraction artifact; changing documents based on screen counts is
premature. Do not hold out 23 07 00: it supplies much of the actual insulation
content and the natural cross-reference task. Do not hold out two documents.

Use two explicitly separate test panels: 160 new scenario records against
known training authorities, and 80 NASA records for document transfer. The
first measures new combinations of observations, provenance, applicability,
and packet contents; it does **not** measure unseen-rule generalization.
The NASA panel is exploratory transfer on one document, not a broad estimate.
If a panel cannot supply a particular property or verdict, disclose that;
do not fabricate authority to fill the cell.

Keep every scenario family, its twins, wording variants, and packet variants
in one split. Keep NASA content out of every training role, including
distractors, question generation, and curation examples. Detect near-verbatim
cross-document duplication. Use development data alone for model decisions.
Public UFGS may be in base-model pretraining: say unseen during this project's
fine-tuning, not guaranteed unseen to the base model.

### (c) Small rule set

Twelve real requirements can support a controlled behavioral benchmark, not a
claim of broad HVAC expertise. Count requirement atoms, parent clause families,
and synthetic scenarios separately. Keep equipment requirements as explicitly
inapplicable distractors, not governing duct rules.

Build difficulty from applicable subject/conditions, exact thresholds,
allowed-material sets, independently observed versus inferred values,
insufficient contextual attributes, real cross-references, and authority
conflicts genuinely constructed and checked. Use a short explicit scope such
as material grade or Table 4 minimum thickness; never certify the entire duct
from a one-property check.

Add only essential context fields needed by audited rules (for example duct
kind, exposure/location, and pressure direction). If the object lacks an
applicability fact, abstain rather than silently assume it. Do not turn this
deadline into an HVAC ontology project.

### (d) Cross-references

Use them heavily, but test applicability and sufficiency rather than a document
name. Make packets containing:

1. The pointer, with no target authority.
2. The pointer plus the right section's irrelevant paragraph.
3. The pointer plus an applicable, complete numeric minimum and headers.
4. The pointer plus a target whose applicability cannot be established.
5. The pointer plus a thermal-performance requirement without the facts needed
   to infer that performance from thickness.

Only (3) necessarily becomes answerable. Missing the reference does not force
abstention if another supplied, applicable, adopted clause independently
settles this narrowly scoped check. Cite the pointer and terminal requirement
as a reference chain, with observation IDs separately.

Useful actual target: 23 07 00 Table 4 spans PDF pages 49–50 and gives numeric
thickness minima. Table 3 on pages 48–49 instead specifies R-values and has
conditions/footnotes; thickness alone does not prove those requirements.
Question scope must distinguish the Table 4 thickness minimum from full
thermal compliance. The Table 3 text also discusses additional insulation and
the most restrictive heating/cooling condition.

23 30 00 clause 3.2.11 is currently non-operative at whole-clause granularity
because its second sentence contains alternatives; its first sentence is an
unconditional pointer. A manually audited span with retained context can be
used without pretending the whole article has been adopted. Clause 3.6.3 is
the cleaner starting pointer.

### (e) CuratorKIT

Use it to ingest candidate structured requirement extractions, run grounding
and domain gates, record rejections/lineage, and export training messages.
It need not generate final answers. In-session teacher outputs can be imported
with truthful teacher metadata. Do not claim an API generation run occurred
if the extraction was done in a chat session.

Freeze one good-faith candidate-extraction pool from training sources. Raw
selects from schema-valid proposals without semantic curation; curated
selects proposals passing quote, semantic scope, operativity, units,
reference-completeness, and applicability checks. Both run the same subsequent
object generator and deterministic verdict procedure. Neither weakens the
existing independent-origin requirement. Raw targets are programmatically
derived from **unverified** extractions; curated/dev/test targets are derived
from audited requirements. That distinction is the actual intervention.

Match count, source-family allocation where possible, property, verdict,
perturbation mix, and token-length distribution. Match optimizer steps and
measure actual non-padding input and supervised target tokens. Padding is not
equal exposure. Disable repair/additional generation in the curated arm for
this first ablation. Publish candidate IDs, accepted IDs, rejection counts,
and a small blinded audit of raw vs curated training targets. Do not seed bad
raw examples, cherry-pick its rejects, or force a difference if all gates pass.

The schema already enforces the core origin rule in both arms. Therefore a
redundant provenance enum gate cannot be credited with correcting that error.
Describe this as semantic requirement/extraction curation; do not isolate a
provenance-gate causal effect without another ablation. Identical accepted
content means the proposed quality ablation has no treatment contrast.

Generate reasons from fixed templates after derivation. Do not spend the
teacher budget on paraphrases or fabricate distractor clauses. CuratorKIT's
generic grounding judge is not a substitute for checking derived comparisons.
Its embedding diversity filter can erase counterfactual twins; deduplicate
scenario families without deleting the required paired examples.

### (f) Derived-label claim

Use: **"Programmatically derived verdicts conditional on audited requirement
extraction, with synthetic observations and explicit provenance."**

Do not use "teacher-free ground truth." A quote proves where words came from,
not that the extraction preserves their scope, comparison, exceptions, units,
or incorporation conditions. Deterministic code can multiply one bad rule into
hundreds of consistent wrong labels. Presently `build_record` accepts verdicts
and citations from its caller; a deterministic requirement evaluator is not
implemented in this repository.

Cheapest check: human-review every distinct accepted requirement card, once,
against rendered full pages plus parent headings and relevant footnotes.
Record exact span, parent subject, adopted status, property/component,
comparator, allowed values, units, direction, conditions/exceptions, and
reference dependencies. Independently work through one compliant, one
non-compliant, and one abstaining example for each applicable rule; test each
numeric boundary. A fresh reviewer/model should critique the card without
seeing the proposed verdict. Resolve disagreements against source context.

## Minimum dataset to ship

These are recommended **row counts**, not counts of independent real rules.
The raw and curated arms each have 800 training rows. Dev and test are shared.
Aim for roughly 12–20 audited requirement atoms if supported by the corpus;
publish the actual count and do not invent atoms to meet it.

| Dimension | Train per arm | Dev | Test |
|---|---:|---:|---:|
| Total | 800 | 120 | 240 |
| COMPLIANT | 240 | 36 | 72 |
| NON_COMPLIANT | 240 | 36 | 72 |
| INSUFFICIENT_EVIDENCE | 320 | 48 | 96 |
| Material | 360 | 54 | 108 |
| Insulation thickness | 360 | 54 | 108 |
| Pressure rating | 80 | 12 | 24 |
| Direct comparison / simple provenance | 256 | 40 | 80 |
| Applicability, distractors, document injection | 256 | 40 | 80 |
| Cross-reference / genuine authority conflict | 288 | 40 | 80 |

These are marginal quotas, not a requirement to populate every cross-product
cell. NASA can contribute only supported properties; put the 24 test pressure
rows in the known-authority scenario panel if NASA supplies no usable pressure
requirement. Never label equipment pressure drop as duct rating to fill quotas.

Count disjoint pairs, not a shared base counted five times:

| Pair type | Train per arm | Dev | Test |
|---|---:|---:|---:|
| Each existing type (five types) | 64 | 10 | 20 |
| Reference-completeness pair | 80 | 10 | 20 |
| Total pairs | 400 | 60 | 120 |

All rows belong to one disjoint evaluation pair. Invariance pairs may start
from abstentions as well as definitive cases; this permits the verdict mix
above and avoids making every injected/distractor record answerable. Balance
both COMPLIANT and NON_COMPLIANT endpoints. Preserve class/variant balance
across citation order and surface wording.

Within the 96 test abstentions, target 24 inferred-value-only, 20 missing
observation, 24 missing authority, 16 unresolved conflict, and 12 applicability
unknown; give each record one unambiguous dominant blocker. Ambiguous
association can wait. Use independent origin types in answerable and
unanswerable records so origin alone does not settle the full task.

Score base/raw/curated with fixed decoding on the same frozen examples.
Report corrected strict accuracy, unsupported definitive rate, coverage,
false approval, answerable recall, reason/action validity, and pair accuracy
by perturbation and panel. Add provenance-only and simple memorized-rule
heuristics to reveal shortcut ceilings. Report number of actual governing
families. A cluster bootstrap needs enough families; repeated objects do not
provide evidence of population-wide rule transfer.

Suggested write-up language:

> Before training we froze source hashes, requirement cards, scenario-family
> assignments, and evaluation records. Scenario families, including every
> counterfactual and wording variant, were confined to one split. The primary
> panel tests new evidence/provenance combinations under known authorities;
> it does not establish unseen-rule generalization. A separate exploratory
> panel uses NASA Metal Ducts, excluded from fine-tuning in every role,
> to assess transfer to one additional document. We screened cross-document
> near-duplicates and report the number of governing requirement families
> alongside the number of synthetic records. Public-source pretraining
> contamination cannot be ruled out.

Use that wording only after the corresponding checks are actually performed.

## Reproduced implementation weaknesses

1. `scoring.py:38–67`: a COMPLIANT prediction with the right governing ID but
   no observation IDs and no action gets `strict=True`, `schema_valid=True`.
   Fake observation IDs and a wrong action also pass strict scoring.
   `tests/test_scoring.py:_pred` omits observations even for its "perfect"
   prediction. Strict abstention scoring ignores citations and action too.
2. `scoring.py:51–54`: a definitive prediction against an insufficient-evidence
   gold can get `unsupported=False` if an independent observation and matching
   governing IDs exist (for example missing precedence). The implementation
   does not establish that the gold is answerable. A wrong definitive conclusion
   with the right citation also is not counted as unsupported. State separately
   whether unsupported means missing evidence or any non-entailed conclusion.
3. `parsing.py:64–67`: unknown reason codes are deleted. Correct code plus
   `VIBES` becomes exactly the correct set and can receive strict credit.
   `score_one(gold, parse_prediction('not JSON'))` raises AttributeError because
   the parser returns None. Normalize failures to scored failures in the runner.
4. `cases.py:105–123`: `remove_precedence` always assigns UNRESOLVED_CONFLICT.
   The existing test calls it on one unconflicted requirement and still passes.
   Mutation preconditions and recomputing the oracle after mutation are needed.
5. `schema.py:72–104`: the schema accepts a definitive verdict on a null value
   bearing a drawing origin, and accepts a generic policy ID as the only
   governing evidence. It does not validate observation ID resolution or the
   source text establishing the asserted value. Checking an origin enum is not
   complete evidence validation. Restricting all properties is not enforced;
   only the target property is checked.
6. `clauses.py`: plain extraction fragments NASA words; decimal table entries
   become fake clause headings (`0.76`, `1.3`, `1.6`, `2.0`) and create duplicate
   IDs. Parent headings with no text are discarded. Operativity checks only the
   body, miss choices in titles and inherited optional blocks, and ignore single
   optional brackets. Conversely an unrelated option makes the whole clause
   unusable. NASA 2.3.4 is visibly an optional block with alternatives in its
   heading, yet is marked operative. Manually audit selected spans now instead
   of attempting a universal parser repair before the deadline.
7. Context error reproduced against source pages: 23 30 00 3.2.7.2 is under
   3.2.7 Kitchen Exhaust Ductwork, not a universal exposed-duct material rule.
   Current objects do not encode exposure, duct construction, component/layer,
   temperature class, coating, or pressure direction. Those absences limit
   which requirements can yield definitive scoped labels.
8. `sources.py:31–52` records new hashes and overwrites the lock; it does not
   check existing expected hashes. The present bytes match, but this is a
   recorded snapshot rather than enforced hash locking. Validate before use.
9. `measure_budget.py:54–65` measures three arbitrary short clauses, not the
   final packet distribution, assistant turn delimiters, or supervised masks.
   Tokenize full finalized messages including targets/EOS; verify target
   survival and nonzero supervised labels on the actual training loader.
   Match real input and target exposure across arms, not padded lengths.
10. Training/evaluation kit integrations, requirement extraction/evaluation,
    artifact publication, and demo remain future pipeline work in this checkout;
    dependencies in pyproject are not evidence of a functioning kit chain.

## Deadline and execution recommendation

Keep Nov 14 as the internal shipping date. The public hackathon page currently
says Nov 15, 2026, 11:59 PM AoE; that is extra buffer, not reason to expand scope.
Nov 11: audit rule cards, correct metric/label failure modes, freeze evaluation.
Nov 12: run one small end-to-end HF pilot, then matched training jobs.
Nov 13: evaluate, publish adapters/data/logs and a working minimal public artifact.
Nov 14: write-up, reproducibility pass, submission.

At the user-quoted $0.80/hour, $20 buys 25 L4 hours before any other charges.
Plan for at most 20 hours, leaving five for retries. Throughput and sequence
budget must come from the pilot. Preserve Tiny Aya's safety preamble and test
assistant-only supervision; including the template does not require training
loss on its preamble. Prefer a correct small experiment over extra data or runs.

## External primary sources consulted

- https://hackathon.lexsi.ai/ — current deadline, repo chaining, artifact rules.
- https://curatorkit.lexsi.ai/latest/guides/quality-gates/ — gate contracts;
  generic grounding receives sample input/output, and missing context can pass.
- https://curatorkit.lexsi.ai/latest/guides/customisation/ — custom generators
  and BaseGate extension points.
- https://www.wbdg.org/FFC/files/media/SI/InsertingBracketedOptionsFY17.pdf —
  single options, whole optional paragraphs, nested brackets and table options.
- The six local hash-verified source PDFs. Rendered NASA PDF page 17, HVAC PDF
  page 80, and insulation PDF page 49; inspected full relevant extracted page
  context, including HVAC page 79 and insulation pages 47–50.
