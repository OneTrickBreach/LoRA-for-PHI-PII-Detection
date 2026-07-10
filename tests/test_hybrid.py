"""Day-9 tests: hybrid predictor unions LoRA + rule pre-filter (stub-based; no model load)."""
from src.predict import HybridPredictor


class _Stub:
    def __init__(self, spans):
        self._spans = spans

    def predict(self, text):
        return [dict(s) for s in self._spans]


def test_hybrid_unions_lora_and_rule_spans():
    # bypass __init__ so we don't load a real adapter / spaCy model
    h = HybridPredictor.__new__(HybridPredictor)
    lora_span = {"start": 0, "end": 6, "type": "MRN"}      # domain id LoRA catches
    regex_span = {"start": 10, "end": 21, "type": "SSN"}   # format id regex catches
    h.lora = _Stub([lora_span])
    h.rules = [_Stub([regex_span])]
    out = h.predict("A55213 on 402-11-9837")
    assert lora_span in out and regex_span in out
    assert len(out) == 2


def test_hybrid_recall_union_never_below_components():
    # union must contain every span either component produces (recall-maximizing)
    h = HybridPredictor.__new__(HybridPredictor)
    lora_spans = [{"start": 0, "end": 3, "type": "MRN"}, {"start": 5, "end": 8, "type": "DEVICE_ID"}]
    rule_spans = [{"start": 5, "end": 8, "type": "IP"}, {"start": 20, "end": 31, "type": "SSN"}]
    h.lora = _Stub(lora_spans)
    h.rules = [_Stub(rule_spans)]
    out = h.predict("x")
    for s in lora_spans + rule_spans:
        assert s in out
