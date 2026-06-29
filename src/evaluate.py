"""Evaluation harness -> comparison table (plan.md §10, rules.md §5). Implemented Day 4, extended Day 7-9.

One command scores any system on the test set. Span correctness = OVERLAP with a gold span of
the SAME type (lead metric); strict-exact also reported (rules.md §5.3). Recall-first: pick the
operating point at recall >= 0.97, then read precision; compare every system at MATCHED recall
(rules.md §5.2). Per-category recall + hard-test confusion + measured latency (rules.md §5.4-5.6).
Emits reports/comparison_table.md.

Usage (Day 4+):  python -m src.evaluate --systems regex presidio fewshot lora --split hard_test
"""
# TODO(Day 4): span match (overlap/exact), recall-first threshold, per-category, latency, table.
