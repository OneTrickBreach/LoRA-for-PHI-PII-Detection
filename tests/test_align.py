"""Day-2 tests: char-span -> BIO alignment, the most safety-critical piece (rules.md §4.3).

Includes the hand-checked 5-example gate and a one-character-offset sensitivity check that proves
the alignment is not trivially passing.
"""
import random

from src.align import (HAND_EXAMPLES, align_spans_to_bio, bio_to_char_spans,
                       label_maps, spans_overlap, verify_examples)
from src.config import load_config
from src.generate import SplitGen, build_A, build_B, build_C, partition_template_ids
from src.id_generators import IdGen


def test_hand_checked_examples_pass(tokenizer):
    ok, _ = verify_examples(tokenizer)
    assert ok, "the 5 hand-checked alignment examples must round-trip exactly"


def test_special_tokens_get_minus_100(tokenizer):
    ex = HAND_EXAMPLES[0]
    enc = align_spans_to_bio(ex["text"], ex["spans"], tokenizer)
    # first and last tokens are [CLS]/[SEP] specials -> ignored by loss
    assert enc["labels"][0] == -100
    assert enc["labels"][-1] == -100
    assert len(enc["labels"]) == len(enc["input_ids"])


def test_alignment_recovers_each_gold_span(tokenizer):
    _, _, id2label = label_maps()
    for ex in HAND_EXAMPLES:
        enc = align_spans_to_bio(ex["text"], ex["spans"], tokenizer)
        rec = bio_to_char_spans(enc["offset_mapping"], enc["labels"], id2label)
        for g in ex["spans"]:
            assert any(spans_overlap(g, r) for r in rec), f"lost span {g} in {ex['text']!r}"


def test_one_char_offset_corrupts_recovery(tokenizer):
    """A deliberately shifted span should change the labels — guards against all-O degeneracy."""
    _, _, id2label = label_maps()
    ex = HAND_EXAMPLES[0]
    good = align_spans_to_bio(ex["text"], ex["spans"], tokenizer)["labels"]
    broken_spans = [{"start": s["start"] - 1, "end": s["start"], "type": s["type"]}
                    for s in ex["spans"]]  # collapse onto the preceding char
    broken = align_spans_to_bio(ex["text"], broken_spans, tokenizer)["labels"]
    assert good != broken, "alignment is insensitive to offsets — that is the corruption bug"


def test_generated_records_round_trip(tokenizer):
    """Integration: real generated records must align and round-trip without losing spans."""
    cfg = load_config()
    _, _, id2label = label_maps()
    gens = IdGen(99)
    sel = random.Random(100)
    parts = partition_template_ids(sel, cfg["data"]["split_fractions"])
    sg = SplitGen(gens, sel, parts["train"]["pos"], parts["train"]["neg"], set())
    builders = [build_A, build_B, build_C]
    checked = 0
    for i in range(30):
        text, spans = builders[i % 3](sg, positive=True)
        # every gold span must be an exact substring at its offsets (offset correctness)
        for s in spans:
            assert text[s["start"]:s["end"]] and 0 <= s["start"] < s["end"] <= len(text)
        enc = align_spans_to_bio(text, spans, tokenizer)
        rec = bio_to_char_spans(enc["offset_mapping"], enc["labels"], id2label)
        for g in spans:
            assert any(spans_overlap(g, r) for r in rec), f"lost {g} in generated record"
        checked += 1
    assert checked == 30
