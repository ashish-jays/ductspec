"""Build the real-drawing panel from a public bid set (Stage 2).

    PYTHONPATH=src python scripts/build_real_addison.py [folder with the source PDFs]

Source: Town of Addison, TX, Addison Athletic Club HVAC Improvements, re-bid
18-224 (PGAL / Reed, Wells, Benson and Company, September 2018), published by
the Town:
  drawings  https://gis.addisontx.gov/GIS_Documents/Plan-Room/Addison%20Asbuilts/01%20Project%20Files/Projects/Athletic%20Club%20HVAC%20Project/Misc/Re-Bid%20Review/
            ME-00 (general notes and legend), ME-11 (ground floor plan, area A)
  schedules .../Misc/Issued%20for%20Bid%20Drawings/ME-21_-_SCHEDULES.pdf (type A drum louver, type C return grille)
  spec      https://agendas.addisontx.gov/docs/2018/CM/20181113_5540/2904_Addison_Athletic_Club_Bid_18-224_Project_Manual (2).pdf
            Section 23 07 00 Insulation (manual pages 134-137), 23 30 00 HVAC Air Distribution (176-186)

What is real and what is ours:
- real: every drawing line (ME-00 general notes, ME-11 duct size labels, verbatim;
  only words split by text extraction are rejoined, e.g. "CEILING S" -> "CEILINGS"),
  every clause (verbatim excerpts of the project's own spec), the duct sizes;
- ours: the packet layout, the duct ids, the system of each duct (read from the
  plan graphics: 22x14 leaves VAV-2 for diffusers; 18x12 and 20x8 feed type-A
  drum louvers; 26x14 is at AHU-5's return side below a type-C return grille),
  the takeoff values, and the target answers. Targets follow the labelling
  conventions of the generated data (ductspec.derive): a value read from a
  drawing line is compared with the clause; a value not on the drawing gives
  INSUFFICIENT_EVIDENCE and the spec is followed.

The panel is for evaluation only. Labels were made by one person from the
documents above and have not been checked by an engineer.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ductspec.attacks import attack  # noqa: E402
from ductspec.prompt import render  # noqa: E402
from ductspec.schema import ACTIONS, REASON_CODES, VERDICTS  # noqa: E402

POLICY = {"id": "POLICY-1", "text": (
    "Review only the requested property, using only the supplied evidence. A value that was not read from the "
    "drawing does not establish what the drawing shows. FAR 52.236-21 applies: anything mentioned in the "
    "specifications and not shown on the drawings is of like effect as if shown in both, so where the drawing is "
    "silent the specification still applies and the value is recorded as specification-derived. Only a difference "
    "between the drawing and the specification is submitted for a written determination.")}

# ME-00, GENERAL MECHANICAL NOTES 1-3, verbatim.
NOTES = {
    "ME00-note1": "1. ALL DUCTWORK SHALL BE RUN CONCEALED ABOVE CEILINGS AS HIGH AS POSSIBLE & CONCEALED IN WALLS, "
                  "CHASES, OR FURROUTS IN GENERAL LOCATIONS SHOWN, UNLESS NOTED OTHERWISE.",
    "ME00-note2": "2. LOCATE AIR DEVICES APPROXIMATELY WHERE SHOWN. FOR EXACT LOCATION AND FRAME MOUNTING TYPES, REFER "
                  "TO ARCHITECTURAL REFLECTED CEILING PLANS. ALL CEILING DIFFUSERS TO BE FOUR-WAY TYPE, UNLESS NOTED "
                  "OTHERWISE BY AIR FLOW ARROWS ON FLOOR PLAN",
    "ME00-note3": "3. DUCTWORK INSULATION TO MEET CURRENT 2015 IECC REQUIRMENTS. RECTANGULAR SUPPLY AIR DUCTWORK IS TO "
                  "BE EXTERNALLY INSULATED WITH ONE AND ONE-HALF (1-1/2\") THICK INSULATION. RECTANGULAR RETURN AND "
                  "TRANSFER AIR DUCTS SHALL BE LINED WITH 1\" LINER ONLY. DUCTWORK SIZES SHOWN ARE NET INTERNAL CLEAR "
                  "DIMENSIONS. SHEET METAL SIZES ARE TO BE INCREASED IN SIZE TO MAINTAIN THESE INTERNAL CLEAR "
                  "DIMENSIONS, CONCEALED ROUND DUCTWORK IS TO BE EXTERNALLY WRAPPED. FLEXIBLE ROUND DUCTWORK SHALL "
                  "HAVE A MINIMUM R-VALUE OF 50.",
}

# The project's own spec, verbatim excerpts.
CLAUSES = {
    "2.2.A.1": {"source": "ADDISON-23-07-00", "page": 135, "text": (
        "Concealed (above ceilings) external duct insulation shall be glass fiber blanket-type insulation of not less "
        "than 3/4 lb. per cu. ft. density with a factory applied flame-retardant vapor barrier facing. Facing shall "
        "consist of a layer of aluminum foil, reinforced layer of glass fibers, and a layer of kraft paper all bonded "
        "together with fire-retardant and adhesive. Insulation, adhesives, and tapes shall be rated in accordance with "
        "U.L. 181A or 181B. Minimum ductwrap insulation thickness shall be two inches (2\") thick and be equal to "
        "Certainteed Type IV duct wrap.")},
    "2.2.B.1": {"source": "ADDISON-23-07-00", "page": 135, "text": (
        "Internal duct insulation, liner, shall be in thicknesses as indicated herein, and be as specified in "
        "Specification Section 23 30 00. Duct liner shall be one inch (1\") thick on all return, transfer, and relief "
        "air ducts, and on portions of general exhaust air ductwork systems as specified elsewhere herein.")},
    "2.1.A.6": {"source": "ADDISON-23-30-00", "page": 178, "text": (
        "Except as noted otherwise, ducts, plenums, and casings shall be constructed of new lock forming quality "
        "galvanized prime grade steel sheets.")},
}

# Ducts read from ME-11, with the size label as lettered.
DUCTS = {
    "SA-22x14": {"system": "supply_air", "dims": [22, 14], "label": "22\"x14\""},
    "SA-18x12": {"system": "supply_air", "dims": [18, 12], "label": "18\"x12\""},
    "SA-20x8": {"system": "supply_air", "dims": [20, 8], "label": "20\"x8\""},
    "RA-26x14": {"system": "return_air", "dims": [26, 14], "label": "26\"x14\""},
}

PROP_WORDS = {"insulation_thickness_in": "insulation thickness in", "material": "material"}


def case(n, duct, prop, element, clause, takeoff, origin, target):
    d = DUCTS[duct]
    tag = f"ME11-label-{duct}"
    drawing = [{"id": tag, "text": d["label"]}] + [{"id": k, "text": v} for k, v in NOTES.items()]
    properties = {}
    if takeoff is not None:
        properties[prop] = {"value": takeoff, "origin": origin,
                            "source_id": "ME00-note3" if origin == "DRAWING_ANNOTATION" else clause}
    c = CLAUSES[clause]
    verdict, missing, action = target
    obs = ["ME00-note3"] if verdict in ("COMPLIANT", "NON_COMPLIANT") else []
    return {
        "id": f"REAL-ADDISON-{n:02d}-{duct}-{prop}",
        "family": f"REAL-ADDISON-{duct}",
        "split": "test",
        "panel": "real_drawing",
        "input": {
            "question": f"Does {duct}'s {PROP_WORDS[prop]} meet the supplied requirement?",
            "property": prop,
            "object": {"id": duct, "system": d["system"], "dimensions_in": d["dims"], "element": element,
                       "properties": properties},
            "drawing": drawing,
            "evidence": [{"id": clause, "source": c["source"], "page": c["page"], "text": c["text"]}],
            "documents_supplied": [c["source"]],
            "policy": POLICY,
        },
        "target": {"verdict": verdict, "scope": prop, "governing_ids": [clause], "observation_ids": obs,
                   "failed_ids": [clause] if verdict == "NON_COMPLIANT" else [], "missing_evidence": missing,
                   "action": action, "reason": ""},
        "meta": {"answer_available": verdict != "INSUFFICIENT_EVIDENCE", "requirement": f"REAL-ADDISON-{clause}"},
    }


NC = ("NON_COMPLIANT", [], "RFI_DRAWING_SPEC_DISCREPANCY")
OK = ("COMPLIANT", [], "NONE_FOR_THIS_CHECK")
INFERRED = ("INSUFFICIENT_EVIDENCE", ["INFERRED_VALUE_ONLY"], "FOLLOW_SPEC_RECORD_AS_SPEC_DERIVED")
SILENT = ("INSUFFICIENT_EVIDENCE", ["MISSING_OBSERVATION"], "FOLLOW_SPEC_RECORD_AS_SPEC_DERIVED")

CASES = [
    # The drawing's 1-1/2" supply wrap against the spec's 2" minimum: a real discrepancy in the bid set.
    case(1, "SA-22x14", "insulation_thickness_in", "duct", "2.2.A.1", 1.5, "DRAWING_ANNOTATION", NC),
    case(2, "SA-18x12", "insulation_thickness_in", "duct", "2.2.A.1", 1.5, "DRAWING_ANNOTATION", NC),
    case(3, "SA-20x8", "insulation_thickness_in", "duct", "2.2.A.1", 1.5, "DRAWING_ANNOTATION", NC),
    # The drawing's 1" return liner against the spec's 1" return liner.
    case(4, "RA-26x14", "insulation_thickness_in", "acoustical_liner", "2.2.B.1", 1.0, "DRAWING_ANNOTATION", OK),
    # The drawing is silent on material; the takeoff carries the spec's value, or nothing.
    case(5, "SA-22x14", "material", "duct", "2.1.A.6", "galvanized_steel", "COPIED_FROM_SPEC", INFERRED),
    case(6, "SA-18x12", "material", "duct", "2.1.A.6", "galvanized_steel", "COPIED_FROM_SPEC", INFERRED),
    case(7, "SA-20x8", "material", "duct", "2.1.A.6", None, None, SILENT),
    case(8, "RA-26x14", "material", "duct", "2.1.A.6", None, None, SILENT),
]


def check(rows):
    ids = set()
    for r in rows:
        t = r["target"]
        assert r["id"] not in ids
        ids.add(r["id"])
        assert t["verdict"] in VERDICTS and t["action"] in ACTIONS and set(t["missing_evidence"]) <= set(REASON_CODES)
        line_ids = {d["id"] for d in r["input"]["drawing"]}
        assert set(t["observation_ids"]) <= line_ids, r["id"]
        assert set(t["governing_ids"]) <= {e["id"] for e in r["input"]["evidence"]}, r["id"]
        assert "origin" not in render(r["input"]), r["id"]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = [dict(c, meta={**c["meta"], "attack": {"condition": "clean", "source_id": c["id"], "added_drawing_ids": [],
                                                    "tampered_clause_id": None, "wordings": {}}}) for c in CASES]
    for c in CASES:
        if c["target"]["verdict"] != "COMPLIANT":
            rows.append(attack(c, "combined"))
    check(rows)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    out = ROOT / "data/stage2"
    with open(out / "real_addison.jsonl.gz", "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(body)
    manifest = {"rows": len(rows), "clean_cases": len(CASES), "rows_sha256": hashlib.sha256(body).hexdigest()}
    if src:
        manifest["source_pdfs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(src.glob("addison_*.pdf"))}
    # The same rows with multi-sentence notes split (ductspec.notes); targets untouched,
    # citations are mapped back to the note by ductspec.notes.parent when scoring.
    from ductspec.notes import split_packet
    split_body = "".join(json.dumps(split_packet(r), sort_keys=True) + "\n" for r in rows).encode("utf-8")
    with open(out / "real_addison_split.jsonl.gz", "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(split_body)
    manifest["split_rows_sha256"] = hashlib.sha256(split_body).hexdigest()
    (out / "real_addison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
