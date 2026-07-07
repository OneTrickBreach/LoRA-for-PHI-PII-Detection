"""Day-4 tests: LoRA decode-with-threshold logic and recall-first threshold selection.

These test the recall-first machinery WITHOUT loading a model, by driving the pure functions with
synthetic cached token scores / stub predictors.
"""
from src.config import bio_label_list
from src.predict import decode_tokens, select_threshold_for_recall, trim_spans


def test_trim_spans_strips_leading_space_and_drops_empty():
    text = " Angela Martinez  "
    spans = [{"start": 0, "end": 16, "type": "NAME"},   # leading space
             {"start": 16, "end": 18, "type": "NAME"}]  # all whitespace -> dropped
    out = trim_spans(spans, text)
    assert out == [{"start": 1, "end": 16, "type": "NAME"}]
    assert text[out[0]["start"]:out[0]["end"]] == "Angela Martinez"


def test_trim_spans_drops_pure_punctuation():
    text = "SN-0549 . -"
    spans = [{"start": 7, "end": 9, "type": "DEVICE_ID"},    # " ." -> punct only, dropped
             {"start": 10, "end": 11, "type": "MRN"},        # "-" -> dropped
             {"start": 0, "end": 7, "type": "DEVICE_ID"}]    # "SN-0549" -> kept (has alnum)
    out = trim_spans(spans, text)
    assert out == [{"start": 0, "end": 7, "type": "DEVICE_ID"}]

LABELS = bio_label_list()
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
B_SSN = LABELS.index("B-SSN")
I_SSN = LABELS.index("I-SSN")
O_ID = LABELS.index("O")


def _tok(cs, ce, argmax, nonO, prob, special=False):
    return (cs, ce, argmax, nonO, prob, special)


def test_decode_argmax_ignores_special_tokens():
    raw = [
        _tok(0, 0, O_ID, B_SSN, 0.9, special=True),    # [CLS] -> ignored
        _tok(0, 3, B_SSN, B_SSN, 0.9),
        _tok(3, 6, I_SSN, I_SSN, 0.8),
        _tok(6, 6, O_ID, B_SSN, 0.9, special=True),    # [SEP] -> ignored
    ]
    spans = decode_tokens(raw, ID2LABEL, threshold=None)
    assert spans == [{"start": 0, "end": 6, "type": "SSN"}]


def test_threshold_controls_recall_precision_tradeoff():
    # a token whose best non-O prob is 0.4: emitted only when threshold <= 0.4
    raw = [_tok(0, 3, O_ID, B_SSN, 0.4)]
    assert decode_tokens(raw, ID2LABEL, threshold=0.5) == []           # O (below thr)
    assert decode_tokens(raw, ID2LABEL, threshold=0.3) == [
        {"start": 0, "end": 3, "type": "SSN"}]                          # entity (>= thr)
    # argmax here is O, so argmax mode also yields nothing
    assert decode_tokens(raw, ID2LABEL, threshold=None) == []


class _StubLora:
    """Emits an SSN span for a record iff the threshold <= the record's stored confidence."""
    id2label = ID2LABEL

    def __init__(self, confidences):
        self._conf = confidences

    def raw_token_scores(self, text):
        c = self._conf[text]
        return [_tok(0, 3, B_SSN if c >= 0.5 else O_ID, B_SSN, c)]


def test_select_threshold_meets_recall_target():
    # 4 PHI records with confidences 0.9/0.7/0.3/0.2; recall floor 0.97 forces a low threshold.
    records = [{"text": t, "spans": [{"start": 0, "end": 3, "type": "SSN"}]}
               for t in ("a", "b", "c", "d")]
    pred = _StubLora({"a": 0.9, "b": 0.7, "c": 0.3, "d": 0.2})
    sel = select_threshold_for_recall(records, pred, target_recall=0.97,
                                      grid=[0.8, 0.5, 0.25, 0.15])
    assert sel["met"] is True
    assert sel["threshold"] <= 0.2          # must go low enough to catch the 0.2 record
    assert sel["recall"] == 1.0


def test_select_threshold_reports_shortfall_when_unreachable():
    records = [{"text": t, "spans": [{"start": 0, "end": 3, "type": "SSN"}]}
               for t in ("a", "b")]
    pred = _StubLora({"a": 0.9, "b": 0.1})
    # grid never goes below 0.5 -> can't catch the 0.1 record -> recall capped at 0.5
    sel = select_threshold_for_recall(records, pred, target_recall=0.97, grid=[0.9, 0.5])
    assert sel["met"] is False
    assert sel["recall"] == 0.5
