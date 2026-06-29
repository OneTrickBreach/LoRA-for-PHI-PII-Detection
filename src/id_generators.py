"""Custom domain identifier generators — Faker does NOT cover these (rules.md §7 anti-pattern).

Implemented Day 2. Each generator returns (text, TYPE) and draws from a per-split pool so
train/val/test entities never overlap (rules.md §3). Covers:
  MRN, NPI, PLAN_ID, DEVICE_ID, ACCOUNT, LICENSE  (+ look-alike variants for hard negatives).
"""
# TODO(Day 2): implement seeded generators with per-split disjoint pools.
