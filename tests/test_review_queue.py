"""Nothing extracted from a non-federal source gets in without a review entry."""

from pathlib import Path

import yaml

from ductspec.requirements import load, load_unadopted

ROOT = Path(__file__).resolve().parents[1]
QUEUE = yaml.safe_load((ROOT / "data" / "review_queue.yaml").read_text(encoding="utf-8"))
STATUSES = {"needs_human_check", "disputed", "confirmed", "corrected", "rejected"}
FEDERAL = ("UFGS-", "VA-")


def _extracted_from_other_authorities() -> set[str]:
    ids = {r.id for r in load() if not r.source.startswith(FEDERAL)}
    ids |= {r.id for r, _ in load_unadopted() if not r.source.startswith(FEDERAL)}
    return ids


def test_every_requirement_from_another_authority_is_queued_for_review():
    queued = {e["id"] for e in QUEUE["entries"]}
    assert _extracted_from_other_authorities() <= queued


def test_the_queue_names_nothing_that_does_not_exist():
    queued = {e["id"] for e in QUEUE["entries"]}
    assert queued <= _extracted_from_other_authorities()


def test_each_entry_has_a_known_status_and_states_its_judgement_call():
    for entry in QUEUE["entries"]:
        assert entry["status"] in STATUSES, entry["id"]
        assert len(entry["judgement"].strip()) > 20, entry["id"]


def test_a_rejected_entry_is_not_in_the_data():
    rejected = {e["id"] for e in QUEUE["entries"] if e["status"] == "rejected"}
    assert not rejected & _extracted_from_other_authorities()


def test_every_entry_records_the_independent_second_reading():
    for entry in QUEUE["entries"]:
        assert entry.get("second_reading"), entry["id"]


def test_a_disputed_or_rejected_item_produces_no_rows():
    from ductspec.generate import build
    from ductspec.requirements import held_back

    withheld = held_back()
    assert withheld, "the queue currently disputes two items"
    rows = [r for split in build().values() for r in split]
    assert not [r["id"] for r in rows if r["meta"]["requirement"] in withheld]
