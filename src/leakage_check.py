"""Automated entity/template overlap check (rules.md §3.3 — MANDATORY before any number is trusted).

Asserts ZERO overlap of identifier strings and template_ids across train/val/test/hard_test.
If it fails: regenerate the data — never hand-patch leaky data (rules.md §3.3, §8 last bullet).
Implemented Day 5. Exit non-zero on any overlap.

Usage (Day 5+):  python -m src.leakage_check
"""
# TODO(Day 5): load pools + records, assert disjoint identifier/template sets across splits.
