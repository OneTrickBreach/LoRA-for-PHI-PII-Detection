"""Char-span -> BIO token label alignment via offset_mapping (plan.md §9 Gotcha B, rules.md §4.3).

THE one-character-offset bug corrupts the entire label set. Must be verified on 5 hand-checked
examples BEFORE training on the full set. Special tokens get label -100 so loss ignores them.
Implemented Day 2, verified by tests/test_alignment.py before any training.
"""
# TODO(Day 2): implement align_spans_to_bio(text, spans, tokenizer, max_length) using
# return_offsets_mapping=True; assign B-/I-<TYPE> on overlap, O otherwise, -100 on specials.
