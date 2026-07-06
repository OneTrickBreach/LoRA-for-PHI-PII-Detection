# Day 6 — Synthetic Data Summary (v2)

- seed: `20260629`  | label types: `17`  | per-category target (train): ≥300
- shapes: ['A', 'B', 'C']  | positive fraction: 0.5

## train
- records: 12800 (6400 pos / 6400 neg)
- unique poolable identifiers: 12491  | carrier template IDs: 68
- by shape: {'C': 4218, 'A': 4244, 'B': 4338}
- positive spans by category: {'ACCOUNT': 377, 'ADDRESS': 377, 'AGE90': 377, 'DATE': 2533, 'DEVICE_ID': 377, 'EMAIL': 377, 'IP': 377, 'LICENSE': 377, 'MRN': 2532, 'NAME': 2532, 'NPI': 376, 'OTHER_ID': 376, 'PHONE': 376, 'PLAN_ID': 376, 'SSN': 376, 'URL': 376, 'VEHICLE_ID': 376}

## val
- records: 1600 (800 pos / 800 neg)
- unique poolable identifiers: 1605  | carrier template IDs: 34
- by shape: {'A': 524, 'B': 541, 'C': 535}
- positive spans by category: {'ACCOUNT': 48, 'ADDRESS': 47, 'AGE90': 47, 'DATE': 331, 'DEVICE_ID': 47, 'EMAIL': 47, 'IP': 47, 'LICENSE': 47, 'MRN': 331, 'NAME': 331, 'NPI': 47, 'OTHER_ID': 47, 'PHONE': 47, 'PLAN_ID': 47, 'SSN': 47, 'URL': 47, 'VEHICLE_ID': 47}

## test
- records: 1600 (800 pos / 800 neg)
- unique poolable identifiers: 1572  | carrier template IDs: 34
- by shape: {'B': 531, 'A': 532, 'C': 537}
- positive spans by category: {'ACCOUNT': 48, 'ADDRESS': 47, 'AGE90': 47, 'DATE': 320, 'DEVICE_ID': 47, 'EMAIL': 47, 'IP': 47, 'LICENSE': 47, 'MRN': 320, 'NAME': 320, 'NPI': 47, 'OTHER_ID': 47, 'PHONE': 47, 'PLAN_ID': 47, 'SSN': 47, 'URL': 47, 'VEHICLE_ID': 47}

## hard_test
- records: 1500 (750 pos / 750 neg)
- unique poolable identifiers: 706  | carrier template IDs: 68
- by shape: {'A': 755, 'C': 745}
- positive spans by category: {'ACCOUNT': 45, 'ADDRESS': 45, 'AGE90': 44, 'DATE': 44, 'DEVICE_ID': 44, 'EMAIL': 44, 'IP': 44, 'LICENSE': 44, 'MRN': 44, 'NAME': 44, 'NPI': 44, 'OTHER_ID': 44, 'PHONE': 44, 'PLAN_ID': 44, 'SSN': 44, 'URL': 44, 'VEHICLE_ID': 44}

## Per-category positive coverage by split

| category | train | val | test | hard_test |
|---|---|---|---|---|
| NAME | 2532 | 331 | 320 | 44 |
| ADDRESS | 377 | 47 | 47 | 45 |
| DATE | 2533 | 331 | 320 | 44 |
| SSN | 376 | 47 | 47 | 44 |
| MRN | 2532 | 331 | 320 | 44 |
| NPI | 376 | 47 | 47 | 44 |
| PLAN_ID | 376 | 47 | 47 | 44 |
| ACCOUNT | 377 | 48 | 48 | 45 |
| LICENSE | 377 | 47 | 47 | 44 |
| DEVICE_ID | 377 | 47 | 47 | 44 |
| VEHICLE_ID | 376 | 47 | 47 | 44 |
| PHONE | 376 | 47 | 47 | 44 |
| EMAIL | 377 | 47 | 47 | 44 |
| URL | 376 | 47 | 47 | 44 |
| IP | 377 | 47 | 47 | 44 |
| AGE90 | 377 | 47 | 47 | 44 |
| OTHER_ID | 376 | 47 | 47 | 44 |

- categories with zero positives anywhere: none
- train categories below the ≥300 target: none
