"""Day-7 tests: error-analysis per-category recall + FN/FP bookkeeping (no model needed)."""
from src.error_analysis import analyze, build_report


class _Stub:
    name = "stub"

    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, text):
        return [dict(s) for s in self.mapping[text]]


def _records():
    return [
        {"text": "a Jane Doe", "contains_phi": 1,
         "spans": [{"start": 2, "end": 10, "type": "NAME"}]},
        {"text": "b 123", "contains_phi": 1,
         "spans": [{"start": 2, "end": 5, "type": "MRN"}]},
        {"text": "c foo", "contains_phi": 0, "spans": []},
    ]


def test_analyze_recall_and_fn_fp():
    preds = {
        "a Jane Doe": [{"start": 2, "end": 10, "type": "NAME"}],   # TP
        "b 123": [],                                               # FN (MRN missed)
        "c foo": [{"start": 2, "end": 5, "type": "PHONE"}],        # FP on a negative record
    }
    r = analyze(_records(), _Stub(preds))
    assert r["per_cat"]["NAME"] == 1.0
    assert r["per_cat"]["MRN"] == 0.0
    assert r["per_cat"]["SSN"] is None            # absent from gold
    assert r["fp_total"] == 1 and r["fp_on_neg"] == 1
    assert r["fn_ex"]["MRN"][0][1] == "123"       # missed substring captured
    assert r["fp_ex"]["PHONE"][0][2] is True      # flagged on a negative record


def test_build_report_has_sections():
    preds = {"a Jane Doe": [{"start": 2, "end": 10, "type": "NAME"}],
             "b 123": [], "c foo": []}
    res = {"lora": analyze(_records(), _Stub(preds))}
    md = build_report(res, "hard_test", 3)
    assert "Per-category recall" in md
    assert "Hard-test confusion" in md
    assert "false negatives" in md
