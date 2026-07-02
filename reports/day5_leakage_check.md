# Day 5 — Automated Leakage / Overlap Check (rules.md §3.3)

**Verdict: PASS — zero overlap ✓**

- splits present: ['train', 'val', 'test']
- unique PHI identifiers/split (excl. AGE90): {'train': 1440, 'val': 197, 'test': 166}
- carrier template IDs/split: {'train': 54, 'val': 6, 'test': 8}

## Cross-split overlaps (must all be empty)

| pair | identifier overlap | template overlap |
|---|---|---|
| train-val | 0  | 0  |
| train-test | 0  | 0  |
| val-test | 0  | 0  |

_AGE90 values are intentionally shared across splits (an age must be allowed everywhere) and are excluded from the identifier pool by design (rules.md §8)._
