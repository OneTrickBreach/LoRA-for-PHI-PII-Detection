# Baseline Comparison — split=`test` (n=200)

Span correctness is OVERLAP with a gold span of the same type (lead metric; strict-exact shown alongside). Latency measured on this hardware (RTX 5070 Ti). Recall-first matched-recall comparison and the LoRA system arrive Day 4.

| system | span-R (overlap) | span-P (overlap) | span-F1 (overlap) | span-F1 (exact) | binary-R | FP on negatives | latency ms/rec |
|---|---|---|---|---|---|---|---|
| regex | 0.584 | 0.503 | 0.540 | 0.490 | 0.840 | 75 | 0.03 |
| presidio | 0.825 | 0.213 | 0.338 | 0.230 | 0.980 | 308 | 11.01 |
| fewshot | 0.482 | 0.860 | 0.618 | 0.618 | 0.260 | 9 | 445.16 |

## Per-category recall (overlap)

| system | NAME | ADDRESS | DATE | SSN | MRN | NPI | PLAN_ID | ACCOUNT | LICENSE | DEVICE_ID | VEHICLE_ID | PHONE | EMAIL | URL | IP | AGE90 | OTHER_ID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex | 0.00 | — | 0.65 | — | 0.58 | — | — | — | — | — | — | 1.00 | 1.00 | 0.00 | 1.00 | — | — |
| presidio | 1.00 | — | 1.00 | — | 0.00 | — | — | — | — | — | — | 0.86 | 1.00 | 1.00 | 1.00 | — | — |
| fewshot | 1.00 | — | 1.00 | — | 1.00 | — | — | — | — | — | — | 0.00 | 0.00 | 0.09 | 0.00 | — | — |

**Note:** `—` = category not present in this split's gold (v1 val/test coverage is thin by design; Day 6 hard test set fixes this). FP on negatives counts predicted spans on look-alike-only records (hard-negative false positives).
