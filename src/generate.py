"""Insertion-based synthetic data generator (plan.md §7). Implemented Day 2 (v1), hardened Day 5/6.

Pipeline: pick clean carrier template -> insert Faker/custom identifiers (positives) or
look-alikes only (negatives) at known char offsets -> emit JSONL with exact spans.
`contains_phi` derived as 1 iff spans non-empty (rules.md §3.7). Splits are entity/template
level with disjoint pools (rules.md §3.1-3.2). One-command run (rules.md §6.1).

Usage (Day 2+):  python -m src.generate --version v1
"""
# TODO(Day 2): implement generation loop + per-split pools + JSONL writer.
