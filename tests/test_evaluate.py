"""Day-3 tests: span-matching metrics (overlap vs exact, one-to-one, PRF, per-category, binary)."""
from src.evaluate import _exact, _overlap, match_spans, prf, score_system


def test_overlap_vs_exact_predicate():
    g = {"start": 10, "end": 20, "type": "NAME"}
    partial = {"start": 12, "end": 18, "type": "NAME"}
    wrong_type = {"start": 10, "end": 20, "type": "DATE"}
    assert _overlap(g, partial) and not _exact(g, partial)
    assert _exact(g, g)
    assert not _overlap(g, wrong_type)


def test_match_is_one_to_one():
    # two gold spans, but only one prediction overlapping both regions -> 1 TP, 1 FN
    gold = [{"start": 0, "end": 5, "type": "MRN"}, {"start": 3, "end": 9, "type": "MRN"}]
    pred = [{"start": 2, "end": 6, "type": "MRN"}]
    tp, fp, fn = match_spans(gold, pred, _overlap)
    assert (tp, fp, fn) == (1, 0, 1)


def test_false_positive_counts():
    gold = [{"start": 0, "end": 4, "type": "SSN"}]
    pred = [{"start": 0, "end": 4, "type": "SSN"}, {"start": 10, "end": 14, "type": "PHONE"}]
    tp, fp, fn = match_spans(gold, pred, _overlap)
    assert (tp, fp, fn) == (1, 1, 0)


def test_prf_math():
    r, p, f = prf(tp=8, fp=2, fn=2)
    assert round(r, 3) == 0.8 and round(p, 3) == 0.8 and round(f, 3) == 0.8
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


class _StubPredictor:
    name = "stub"

    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, text):
        return self.mapping[text]


def test_score_system_end_to_end():
    records = [
        {"text": "a", "contains_phi": 1,
         "spans": [{"start": 0, "end": 1, "type": "NAME"}]},
        {"text": "b", "contains_phi": 1,
         "spans": [{"start": 0, "end": 1, "type": "SSN"}]},
        {"text": "c", "contains_phi": 0, "spans": []},
    ]
    preds = {
        "a": [{"start": 0, "end": 1, "type": "NAME"}],   # TP
        "b": [],                                          # FN (missed), not flagged
        "c": [{"start": 0, "end": 1, "type": "PHONE"}],   # FP on a negative record
    }
    res = score_system(records, _StubPredictor(preds), measure_latency=False)
    assert res["overlap"]["tp"] == 1 and res["overlap"]["fn"] == 1 and res["overlap"]["fp"] == 1
    assert round(res["overlap"]["recall"], 3) == 0.5
    assert res["binary_recall"] == 0.5          # 1 of 2 PHI records flagged
    assert res["fp_on_negatives"] == 1
    assert res["per_category_recall"]["NAME"] == 1.0
    assert res["per_category_recall"]["SSN"] == 0.0
    assert res["per_category_recall"]["MRN"] is None   # absent from gold
