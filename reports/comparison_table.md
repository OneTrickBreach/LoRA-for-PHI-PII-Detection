# Baseline Comparison — split=`hard_test` (n=1500)

Span correctness is OVERLAP with a gold span of the same type (lead metric; strict-exact shown alongside). Latency measured on this hardware (RTX 5070 Ti). For LoRA, `argmax` is the default operating point and `R>=<target>@t=<thr>` is the recall-first point whose threshold was selected on val (never on this split).

| system | span-R (overlap) | span-P (overlap) | span-F1 (overlap) | span-F1 (exact) | binary-R | FP on negatives | latency ms/rec |
|---|---|---|---|---|---|---|---|
| regex | 0.292 | 0.237 | 0.262 | 0.230 | 0.533 | 432 | 0.03 |
| presidio | 0.491 | 0.101 | 0.168 | 0.113 | 0.916 | 1739 | 12.25 |
| fewshot | 0.003 | 0.038 | 0.005 | 0.000 | 0.035 | 9 | 260.27 |
| lora(argmax) | 0.557 | 0.568 | 0.563 | 0.361 | 0.747 | 40 | 43.66 |
| lora(R>=0.97@t=0.7) | 0.527 | 0.664 | 0.587 | 0.376 | 0.676 | 38 | 36.47 |

## Per-category recall (overlap)

| system | NAME | ADDRESS | DATE | SSN | MRN | NPI | PLAN_ID | ACCOUNT | LICENSE | DEVICE_ID | VEHICLE_ID | PHONE | EMAIL | URL | IP | AGE90 | OTHER_ID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex | 0.00 | 0.00 | 0.52 | 1.00 | 0.45 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| presidio | 0.93 | 0.67 | 0.93 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.82 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| fewshot | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lora(argmax) | 0.95 | 0.42 | 0.27 | 0.00 | 0.77 | 0.00 | 0.43 | 0.87 | 0.05 | 0.98 | 1.00 | 0.89 | 1.00 | 1.00 | 0.41 | 0.02 | 0.41 |
| lora(R>=0.97@t=0.7) | 0.93 | 0.42 | 0.27 | 0.00 | 0.77 | 0.00 | 0.07 | 0.84 | 0.00 | 0.98 | 1.00 | 0.82 | 1.00 | 1.00 | 0.41 | 0.02 | 0.41 |

**Note:** `—` = category not present in this split's gold. FP on negatives counts predicted spans on look-alike-only records (hard-negative false positives).
