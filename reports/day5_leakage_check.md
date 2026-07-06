# Day 5 — Automated Leakage / Overlap Check (rules.md §3.3)

**Verdict: PASS — zero overlap ✓**

- splits present: ['train', 'val', 'test', 'hard_test']
- unique PHI identifiers/split (excl. AGE90): {'train': 12491, 'val': 1605, 'test': 1572, 'hard_test': 706}
- carrier template IDs/split: {'train': 68, 'val': 34, 'test': 34, 'hard_test': 68}

## Cross-split overlaps (must all be empty)

| pair | identifier overlap | template overlap |
|---|---|---|
| train-val | 0  | 0  |
| train-test | 0  | 0  |
| train-hard_test | 0  | 0  |
| val-test | 0  | 0  |
| val-hard_test | 0  | 0  |
| test-hard_test | 0  | 0  |

_AGE90 values are intentionally shared across splits (an age must be allowed everywhere) and are excluded from the identifier pool by design (rules.md §8)._
