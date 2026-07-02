"""Day-5 tests: automated leakage/overlap check (rules.md §3.3)."""
import json

from src.leakage_check import identifiers_from_jsonl, run_check


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_identifiers_exclude_age90_and_capture_values(tmp_path):
    p = tmp_path / "s.jsonl"
    _write(p, [
        {"text": "Patient Jane Doe is 92 years old", "contains_phi": 1,
         "spans": [{"start": 8, "end": 16, "type": "NAME"},
                   {"start": 20, "end": 22, "type": "AGE90"}]},
    ])
    ids = identifiers_from_jsonl(p)
    assert "Jane Doe" in ids
    assert "92" not in ids, "AGE90 must be excluded from the identifier pool"


def test_overlap_detection_between_two_splits(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"text": "MRN A55213 here", "contains_phi": 1,
                "spans": [{"start": 4, "end": 10, "type": "MRN"}]}])
    _write(b, [{"text": "record A55213 again", "contains_phi": 1,
                "spans": [{"start": 7, "end": 13, "type": "MRN"}]}])
    ida, idb = identifiers_from_jsonl(a), identifiers_from_jsonl(b)
    assert ida & idb == {"A55213"}, "shared identifier must be detected as overlap"


def test_real_data_is_clean():
    # Integration: the committed generator output must pass the overlap check.
    result = run_check()
    assert result["clean"], f"LEAKAGE in generated data: {result['overlaps']}"
    assert set(result["present"]) >= {"train", "val", "test"}
