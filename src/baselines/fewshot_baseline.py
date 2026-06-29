"""Few-shot decoder baseline, zero training (plan.md §8.3). Implemented Day 3.

Uses the EXACT prompt in plan.md §8.3. Returns JSON spans {text,type}; resolved to char
offsets by locating the returned text in the record. Decoder is alt/stretch only — never the
primary (rules.md §4.4). Returns spans in the unified predict format.
"""
# TODO(Day 3): run small instruct decoder with the §8.3 prompt; parse JSON; locate offsets.
