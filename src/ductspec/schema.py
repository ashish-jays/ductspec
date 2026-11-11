"""The DuctSpec record schema.

One record = one property check on one duct against supplied evidence.

The point of the schema is the `origin` field on every observed property. A
construction drawing shows some things and is silent about others; a pipeline
that back-fills the silences from the specification (or from an engineering
default) and then checks that value *against the specification it came from*
has proved nothing. That is the failure this project trains against, so
provenance is a first-class field rather than metadata.
"""

VERDICTS = ("COMPLIANT", "NON_COMPLIANT", "INSUFFICIENT_EVIDENCE")

# Where a property value came from. Only DRAWING_ANNOTATION and
# DRAWING_SCHEDULE are independent evidence of what the drawing shows.
ORIGINS = (
    "DRAWING_ANNOTATION",     # a callout or note on the sheet
    "DRAWING_SCHEDULE",       # a schedule table on the sheet
    "COPIED_FROM_SPEC",       # back-filled from the requirement being checked
    "ENGINEERING_DEFAULT",    # back-filled from a house/industry default
    "ESTIMATOR_ENTRY",        # typed by a human, not read from the drawing
)
INDEPENDENT_ORIGINS = ("DRAWING_ANNOTATION", "DRAWING_SCHEDULE")

# Why a definitive verdict could not be issued.
REASON_CODES = (
    "MISSING_OBSERVATION",     # the drawing does not show this property
    "INFERRED_VALUE_ONLY",     # the only value available was back-filled
    "MISSING_AUTHORITY",       # no supplied clause governs this question
    "APPLICABILITY_UNKNOWN",   # cannot tell whether the clause covers this object
    "UNRESOLVED_CONFLICT",     # two authorities disagree, no precedence supplied
    "AMBIGUOUS_ASSOCIATION",   # cannot tie the callout to this duct
)

ACTIONS = (
    "NONE_FOR_THIS_CHECK",
    "RFI_DRAWING_SPEC_DISCREPANCY",
    "REQUEST_MISSING_DETAIL",
    "FOLLOW_SPEC_RECORD_AS_SPEC_DERIVED",
    "REQUEST_APPLICABILITY_CLARIFICATION",
)

PROPERTIES = ("material", "pressure_rating_in_wg", "insulation_thickness_in")


class SchemaError(ValueError):
    pass


def _require(cond, msg):
    if not cond:
        raise SchemaError(msg)


def validate_record(rec: dict) -> dict:
    """Validate one record, raising SchemaError on the first problem found.

    Returns the record so it can be used inline.
    """
    for key in ("id", "input", "target"):
        _require(key in rec, f"missing top-level key: {key}")

    inp, tgt = rec["input"], rec["target"]
    for key in ("question", "object", "evidence", "policy"):
        _require(key in inp, f"input missing key: {key}")

    obj = inp["object"]
    _require("id" in obj, "object missing id")
    prop = tgt.get("scope")
    _require(prop in PROPERTIES, f"target.scope must be one of {PROPERTIES}, got {prop!r}")

    observed = obj.get("properties", {}).get(prop)
    _require(observed is not None, f"object has no properties entry for {prop!r}")
    _require(
        observed.get("origin") in ORIGINS,
        f"property {prop!r} has invalid origin {observed.get('origin')!r}",
    )

    _require(isinstance(inp["evidence"], list) and inp["evidence"], "evidence must be a non-empty list")
    evidence_ids = {e["id"] for e in inp["evidence"]}
    _require("id" in inp["policy"], "policy missing id")
    evidence_ids.add(inp["policy"]["id"])

    _require(tgt.get("verdict") in VERDICTS, f"invalid verdict {tgt.get('verdict')!r}")
    _require(tgt.get("action") in ACTIONS, f"invalid action {tgt.get('action')!r}")

    for cid in tgt.get("governing_ids", []):
        _require(cid in evidence_ids, f"governing_id {cid!r} is not in the supplied evidence")

    codes = tgt.get("missing_evidence", [])
    for code in codes:
        _require(code in REASON_CODES, f"invalid reason code {code!r}")

    if tgt["verdict"] == "INSUFFICIENT_EVIDENCE":
        _require(codes, "INSUFFICIENT_EVIDENCE requires at least one missing_evidence code")
        _require(tgt["action"] != "NONE_FOR_THIS_CHECK", "abstention requires a follow-up action")
    else:
        _require(not codes, "a definitive verdict must not carry missing_evidence codes")
        _require(
            observed["origin"] in INDEPENDENT_ORIGINS,
            f"definitive verdict on a {observed['origin']} value: a back-filled value "
            "is not independent evidence of what the drawing shows",
        )
        _require(tgt.get("governing_ids"), "a definitive verdict must cite governing evidence")

    return rec
