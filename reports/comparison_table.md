# Baseline Comparison — split=`hard_test` (n=1500)

Span correctness is OVERLAP with a gold span of the same type (lead metric; strict-exact shown alongside). Latency measured on this hardware (RTX 5070 Ti). For LoRA, `argmax` is the default operating point and `R>=<target>@t=<thr>` is the recall-first point whose threshold was selected on val (never on this split).

| system | span-R (overlap) | span-P (overlap) | span-F1 (overlap) | span-F1 (exact) | binary-R | FP on negatives | latency ms/rec |
|---|---|---|---|---|---|---|---|
| regex | 0.292 | 0.237 | 0.262 | 0.230 | 0.533 | 432 | 0.01 |
| presidio | 0.491 | 0.101 | 0.168 | 0.115 | 0.916 | 1739 | 7.20 |
| fewshot | 0.003 | 0.038 | 0.005 | 0.000 | 0.035 | 9 | 168.11 |
| lora(argmax) | 0.569 | 0.607 | 0.587 | 0.398 | 0.751 | 38 | 25.16 |
| lora(R>=0.97@t=0.7) | 0.549 | 0.681 | 0.608 | 0.396 | 0.704 | 34 | 23.17 |
| hybrid | 0.688 | 0.345 | 0.459 | 0.354 | 0.908 | 470 | 24.00 |

## Per-category recall (overlap)

| system | NAME | ADDRESS | DATE | SSN | MRN | NPI | PLAN_ID | ACCOUNT | LICENSE | DEVICE_ID | VEHICLE_ID | PHONE | EMAIL | URL | IP | AGE90 | OTHER_ID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex | 0.00 | 0.00 | 0.52 | 1.00 | 0.45 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| presidio | 0.93 | 0.67 | 0.93 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.82 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| fewshot | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lora(argmax) | 0.98 | 0.47 | 0.20 | 0.00 | 0.80 | 0.00 | 0.57 | 1.00 | 0.00 | 1.00 | 0.89 | 0.98 | 1.00 | 0.98 | 0.41 | 0.02 | 0.39 |
| lora(R>=0.97@t=0.7) | 0.98 | 0.44 | 0.20 | 0.00 | 0.77 | 0.00 | 0.57 | 0.91 | 0.00 | 1.00 | 0.77 | 0.98 | 1.00 | 0.98 | 0.41 | 0.00 | 0.32 |
| hybrid | 0.98 | 0.47 | 0.61 | 1.00 | 0.80 | 0.00 | 0.57 | 1.00 | 0.00 | 1.00 | 0.89 | 1.00 | 1.00 | 0.98 | 1.00 | 0.02 | 0.39 |

**Note:** `—` = category not present in this split's gold. FP on negatives counts predicted spans on look-alike-only records (hard-negative false positives).
