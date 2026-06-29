"""Presidio baseline — THE BAR TO BEAT (plan.md §8.2, rules.md §1.5). Implemented Day 3.

presidio-analyzer out of the box with default recognizers (spaCy en_core_web_lg backend).
Maps Presidio entity labels -> our category set (plan.md §5) for per-category comparison.
Returns spans in the unified predict format.
"""
# TODO(Day 3): AnalyzerEngine().analyze(text, language="en"); map entities -> our types.
