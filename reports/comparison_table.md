# Baseline Comparison — split=`test` (n=200)

Span correctness is OVERLAP with a gold span of the same type (lead metric; strict-exact shown alongside). Latency measured on this hardware (RTX 5070 Ti). For LoRA, `argmax` is the default operating point and `R>=<target>@t=<thr>` is the recall-first point whose threshold was selected on val (never on this split).

| system | span-R (overlap) | span-P (overlap) | span-F1 (overlap) | span-F1 (exact) | binary-R | FP on negatives | latency ms/rec |
|---|---|---|---|---|---|---|---|
| regex | 0.584 | 0.503 | 0.540 | 0.490 | 0.840 | 75 | 0.03 |
| presidio | 0.825 | 0.213 | 0.339 | 0.235 | 0.980 | 308 | 9.00 |
| fewshot | 0.476 | 0.849 | 0.610 | 0.610 | 0.260 | 9 | 408.90 |
| lora(argmax) | 0.771 | 0.780 | 0.776 | 0.624 | 0.700 | 4 | 25.98 |
| lora(R>=0.97@t=0.01 (target NOT met on val)) | 0.849 | 0.613 | 0.712 | 0.576 | 0.880 | 9 | 24.84 |

## Per-category recall (overlap)

| system | NAME | ADDRESS | DATE | SSN | MRN | NPI | PLAN_ID | ACCOUNT | LICENSE | DEVICE_ID | VEHICLE_ID | PHONE | EMAIL | URL | IP | AGE90 | OTHER_ID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex | 0.00 | — | 0.65 | — | 0.58 | — | — | — | — | — | — | 1.00 | 1.00 | 0.00 | 1.00 | — | — |
| presidio | 1.00 | — | 1.00 | — | 0.00 | — | — | — | — | — | — | 0.86 | 1.00 | 1.00 | 1.00 | — | — |
| fewshot | 1.00 | — | 1.00 | — | 1.00 | — | — | — | — | — | — | 0.00 | 0.00 | 0.04 | 0.00 | — | — |
| lora(argmax) | 1.00 | — | 1.00 | — | 1.00 | — | — | — | — | — | — | 0.52 | 0.88 | 1.00 | 0.07 | — | — |
| lora(R>=0.97@t=0.01 (target NOT met on val)) | 1.00 | — | 1.00 | — | 1.00 | — | — | — | — | — | — | 1.00 | 0.88 | 1.00 | 0.18 | — | — |

**Note:** `—` = category not present in this split's gold (v1 val/test coverage is thin by design; Day 6 hard test set fixes this). FP on negatives counts predicted spans on look-alike-only records (hard-negative false positives).
