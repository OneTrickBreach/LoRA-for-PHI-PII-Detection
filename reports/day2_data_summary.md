# Day 2 — Synthetic Data Summary (v1)

- seed: `20260629`  | label types: `17`
- shapes: ['A', 'B', 'C']  | positive fraction: 0.5

## train
- records: 1600 (800 pos / 800 neg)
- unique poolable identifiers: 1440  | carrier template IDs: 54
- by shape: {'B': 523, 'C': 545, 'A': 532}
- positive spans by category: {'ACCOUNT': 42, 'ADDRESS': 53, 'AGE90': 47, 'DATE': 325, 'DEVICE_ID': 57, 'EMAIL': 28, 'IP': 35, 'LICENSE': 38, 'MRN': 294, 'NAME': 301, 'NPI': 60, 'OTHER_ID': 51, 'PHONE': 30, 'PLAN_ID': 25, 'SSN': 42, 'URL': 22, 'VEHICLE_ID': 37}

## val
- records: 200 (100 pos / 100 neg)
- unique poolable identifiers: 197  | carrier template IDs: 6
- by shape: {'C': 56, 'A': 72, 'B': 72}
- positive spans by category: {'DATE': 39, 'MRN': 68, 'NAME': 62, 'PLAN_ID': 28}

## test
- records: 200 (100 pos / 100 neg)
- unique poolable identifiers: 166  | carrier template IDs: 8
- by shape: {'C': 69, 'A': 65, 'B': 66}
- positive spans by category: {'DATE': 26, 'EMAIL': 16, 'IP': 28, 'MRN': 26, 'NAME': 26, 'PHONE': 21, 'URL': 23}

## All-splits positive spans by category
{'ACCOUNT': 42, 'ADDRESS': 53, 'AGE90': 47, 'DATE': 390, 'DEVICE_ID': 57, 'EMAIL': 44, 'IP': 63, 'LICENSE': 38, 'MRN': 388, 'NAME': 389, 'NPI': 60, 'OTHER_ID': 51, 'PHONE': 51, 'PLAN_ID': 53, 'SSN': 42, 'URL': 45, 'VEHICLE_ID': 37}

- categories with zero positives (v1; coverage hardened Day 6): none

