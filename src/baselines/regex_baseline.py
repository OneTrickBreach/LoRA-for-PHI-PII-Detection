"""Regex baseline (plan.md §8.1). Implemented Day 3.

Uses the EXACT patterns in config.yaml `regex_patterns` — not a deliberately weak strawman
(rules.md §7 last bullet). Expected to miss names / free-text addresses / context cases; that
gap is the point. Returns spans in the unified predict format.
"""
# TODO(Day 3): compile config regex_patterns, scan text, emit typed spans.
