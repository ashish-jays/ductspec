# DuctSpec

**Can a 3.35B open-weights model refuse to certify compliance it cannot actually see?**

A construction drawing shows some things and is silent about others. When a
review pipeline meets a silence — a duct with no insulation thickness called
out — the convenient thing to do is fill the gap from the specification, or
from an engineering default, and carry on. If that filled-in value is later
checked *against the specification it was copied from*, the check passes. It
has proved nothing. The model has confirmed its own assumption and returned a
verdict that looks like evidence.

DuctSpec trains [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global)
(3.35B) to tell the difference between a value the drawing **shows**, a value a
requirement **demands**, and a value somebody **inferred** — and to abstain,
with a reason and a follow-up action, when only the last two exist.

> **Thesis.** Provenance-curated fine-tuning reduces unsupported conformance
> verdicts — especially where specification-derived or default-filled values
> masquerade as drawing evidence — without sacrificing answerable coverage.

**Track 1 — Fine-tuning and Data Curation.** Problem statements **#1 Domain
assistants for regulated industries** and **#3 Grounded answers and fewer
hallucinations**, addressed as one task: in construction, the governing
documents are mandatory, so a correct answer must say *which* authority governs,
*where it is written*, and when the documents do not settle the question.

## Why this is a regulated domain, not generic document QA

Three things make a conformance verdict different from a retrieval answer:

1. **Applicability is a question in its own right.** A clause about kitchen
   exhaust does not govern a supply duct. The model must decide whether the
   requirement reaches this object before it compares any values.
2. **Authority is supplied, not assumed.** We do *not* train a fixed
   `code > standard > specification > drawing` ladder: which document governs
   depends on jurisdiction, adopted edition, incorporation by reference, and a
   project specification routinely imposes something stricter than a standard's
   minimum. The governing policy is supplied as evidence and the model must
   apply the one it was given.
3. **Abstention has a name, a format and a recipient.** Where the documents
   conflict or fall silent, the correct output is not "I don't know" — it is a
   scoped request for written clarification. [FAR 52.236-21](https://www.acquisition.gov/far/52.236-21)
   gives specifications precedence over drawings *and* requires discrepancies
   to be submitted for a written determination. That is the RFI, in federal
   contract language.

## The four behaviours being trained

1. Does this requirement apply to this object?
2. Which supplied authority governs this question?
3. Is there **independent** evidence for the property being checked?
4. Produce a scoped verdict, or state precisely what clarification is needed.

## Corpus

[UFGS](https://www.wbdg.org/ffc/dod/unified-facilities-guide-specifications-ufgs)
— Unified Facilities Guide Specifications, prepared by USACE / NAVFAC / AFCEC /
NASA. Works of the US federal government, in the public domain, so the whole
dataset can be published. They are a **public surrogate** for "an organisation's
own documents"; no proprietary specification is used, quoted or published.

| Source | Split |
|---|---|
| UFGS 23 30 00 — HVAC Air Distribution | train |
| UFGS 23 07 00 — Thermal Insulation for Mechanical Systems | train |
| UFGS 23 05 93 — Testing, Adjusting and Balancing for HVAC | train |
| UFGS 23 08 00 — Commissioning of HVAC Systems | train |
| UFGS 22 00 00 — Plumbing, General Purpose | train |
| **UFGS 23 31 13.00 40 — Metal Ducts** | **holdout (whole document)** |

The holdout is a *whole-document* holdout: no clause from it appears in
training, so the test set measures transfer to an unseen specification rather
than recall of seen prose. Every PDF is hashed into `data/sources.lock.json` —
UFGS sections are revised, so "UFGS 23 30 00" alone is not a reproducible
reference.

Two properties of a *guide* specification are handled explicitly, because they
decide whether a clause is citable at all:

- **Designer notes are stripped.** UFGS carries instructions to the engineer
  editing it, bracketed by rules of asterisks and beginning `NOTE:`. They are
  advice on how to write the spec, not requirements binding on anyone. Citing
  one as an authority is quoting a comment as law.
- **Clauses are marked operative or not.** UFGS ships with blanks (`[_____]`)
  and unresolved alternatives (`[ASME A13.1] [MIL-STD-101]`). Until the designer
  chooses, there is no requirement to comply with. Only operative clauses are
  usable as governing evidence; the rest are exactly the right material for
  `APPLICABILITY_UNKNOWN` cases.

## Pipeline

```
data/sources.yaml ──► ductspec.sources   fetch + sha256 freeze
                  ──► ductspec.clauses   clause records, notes stripped, operativity marked
                  ──► CuratorKIT         generate + gate composite evidence packets
                  ──► AlignTune          two matched QLoRA adapters: raw arm vs curated arm
                  ──► AuditKIT           base / raw / curated on one frozen test set
```

The **raw-vs-curated ablation** is the experiment, not decoration: same teacher,
same example count, same token exposure, differing only in whether grounding and
provenance gates were applied. Lexsi's own curation study found generator scale
mattered more than filtration, so "curation helps" is something to demonstrate,
not assume. A negative result gets published as readily as a positive one.

## Reproduce

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
python -m ductspec.sources      # fetch + hash the corpus
pytest -q
```

## Hackathon rules compliance

| Rule | How this submission meets it |
|---|---|
| ≥2 repos from the track, 3+ preferred | **3**: CuratorKIT → AlignTune → AuditKIT, each consuming the previous one's artifact |
| Dense open-weights Cohere model | Tiny Aya Global (3.35B). Any other model appears only as a teacher generator or judge, which the rules permit |
| All artifacts on Hugging Face | adapters, datasets, evaluation logs — public, un-gated collections |
| Public deployment by the deadline | HF Space (ZeroGPU) + this public repo |
| Before-and-after metrics, more than one | base / raw-SFT / curated-SFT across the metric table, on one frozen test set |
| Write-up | HF Article, mirrored here |
| ROI comparison *(optional but preferred)* | measured GPU hours and throughput from the training runs, against closed-model token billing at an explicit volume |
| Originality | own work, built during the hackathon window; public-domain corpus only |

## Licensing

- **This code:** Apache-2.0.
- **Dataset:** derived from UFGS, a US federal government work in the public domain.
- **Adapters:** derived from Tiny Aya, which Cohere Labs releases under
  CC-BY-NC 4.0 with an acceptable-use policy — the adapters are therefore
  **non-commercial** and are published as such.
- **Toolkits:** CuratorKIT, AlignTune and AuditKIT are Lexsi Labs
  source-available (LSAL) libraries, used as dependencies, not vendored.
- No SMACNA, ASHRAE or other paywalled standard is reproduced. Where such a
  standard is relevant it is cited by designation only.
