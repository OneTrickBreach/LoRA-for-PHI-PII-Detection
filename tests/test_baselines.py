"""Day-3 tests: regex baseline, few-shot output parsing, and Presidio label mapping."""
from src.baselines.fewshot_baseline import extract_spans_from_output
from src.baselines.presidio_baseline import ENTITY_MAP
from src.baselines.regex_baseline import RegexBaseline
from src.config import label_list


def _find(text, sub, typ):
    i = text.find(sub)
    return {"start": i, "end": i + len(sub), "type": typ}


def test_regex_catches_formatted_identifiers():
    rb = RegexBaseline()
    text = ("SSN 402-11-9837, email jane.doe@gmail.com, call 216-555-0148, "
            "from 73.118.42.9 on 03/14/1981, MRN A55213x.")
    spans = rb.predict(text)
    got = {(s["type"], text[s["start"]:s["end"]]) for s in spans}
    assert ("SSN", "402-11-9837") in got
    assert ("EMAIL", "jane.doe@gmail.com") in got
    assert ("PHONE", "216-555-0148") in got
    assert ("IP", "73.118.42.9") in got
    assert ("DATE", "03/14/1981") in got
    assert any(t == "MRN" for t, _ in got)


def test_regex_offsets_are_exact():
    rb = RegexBaseline()
    text = "The SSN is 123-45-6789 today."
    spans = rb.predict(text)
    for s in spans:
        assert text[s["start"]:s["end"]]
        assert 0 <= s["start"] < s["end"] <= len(text)


def test_regex_false_positives_on_lookalikes():
    # An SSN-shaped ticket and a public support line SHOULD trip the naive regex — this is the
    # weakness the comparison is designed to expose (rules.md §7).
    rb = RegexBaseline()
    text = "Case reference 402-11-9837 opened; support line 800-555-0000."
    spans = rb.predict(text)
    assert any(s["type"] == "SSN" for s in spans)
    assert any(s["type"] == "PHONE" for s in spans)


def test_fewshot_parsing_locates_offsets():
    text = 'Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith.'
    raw = ('Output: [{"text":"John Reyes","type":"NAME"},'
           '{"text":"03/14/1981","type":"DATE"},{"text":"A55213","type":"MRN"}]')
    spans = extract_spans_from_output(text, raw)
    assert _find(text, "John Reyes", "NAME") in spans
    assert _find(text, "03/14/1981", "DATE") in spans
    assert _find(text, "A55213", "MRN") in spans


def test_fewshot_parsing_empty_and_garbage():
    assert extract_spans_from_output("anything", "Output: []") == []
    assert extract_spans_from_output("anything", "no json here") == []
    # unknown type is dropped
    assert extract_spans_from_output("foo bar", '[{"text":"foo","type":"WEIRD"}]') == []


def test_fewshot_parsing_ignores_hallucinated_text():
    # text the model returns that is NOT in the record must be dropped (no false span)
    text = "Order 99812 shipped."
    raw = '[{"text":"Jane Smith","type":"NAME"}]'
    assert extract_spans_from_output(text, raw) == []


def test_fewshot_handles_repeated_substrings():
    text = "Call 111-111-1111 or 111-111-1111 for the two lines."
    raw = ('[{"text":"111-111-1111","type":"PHONE"},'
           '{"text":"111-111-1111","type":"PHONE"}]')
    spans = extract_spans_from_output(text, raw)
    assert len(spans) == 2
    assert spans[0]["start"] != spans[1]["start"], "repeats must map to distinct offsets"


def test_presidio_map_targets_are_valid_types():
    valid = set(label_list())
    for our_type in ENTITY_MAP.values():
        assert our_type in valid, f"{our_type} not in label schema"
