# One-Page Memo — PHI/PII Detection via LoRA (POC verdict)

**Date:** 2026-07-10 · **Scope:** 2-week proof-of-value · **Data:** 100% synthetic, English-only ·
**Hardware:** single RTX 5070 Ti (12 GB). All numbers measured, not estimated.

## Question 1 — Does LoRA beat Microsoft Presidio?

**Yes, decisively, on the metric that matters (precision at comparable-or-higher recall).** On the
held-out **hard test set** (n=1,500; span overlap):

| system | recall | precision | FP on look-alike records | latency |
|---|---|---|---|---|
| Presidio (the bar) | 0.491 | 0.101 | 1,739 | 7 ms |
| **LoRA (r16/α16)** | **0.569** | **0.607** | **38** | 25 ms |
| Hybrid (LoRA ∪ regex) | **0.688** | 0.345 | 470 | 24 ms |

LoRA **Pareto-dominates Presidio** — higher recall (0.569 vs 0.491) *and* **~6× the precision**
(0.607 vs 0.101), with **38 false positives vs 1,739** on look-alike-only records. Presidio has no
recognizer for domain identifiers (MRN/NPI/PLAN_ID/DEVICE_ID/ACCOUNT/VEHICLE) and over-flags
look-alikes; LoRA learned the rubric's context and the domain IDs.

**Caveat on the formal bar (recall ≥ 0.97):** on **in-distribution** validation data LoRA reaches
recall **0.972** (threshold 0.7). But **no system reaches 0.97 on the hard test set** — the set
deliberately weighted toward terse/unseen phrasings and dense look-alikes. Best hard-set recall is the
**hybrid at 0.688**. This is a reported finding, not a hidden failure.

## Question 2 — Best config, cost, latency

**Recommended: DeBERTa-v3-base + LoRA, r=16, α=16, lr=2e-4, 3 epochs, `target_modules=all-linear`,
`modules_to_save=["classifier"]`, bf16.** Chosen by a sweep (see `sweep_results.md`; α=r beat α=2r,
bigger rank didn't help and broke latency, lower LR underfit).

- **Train cost:** ~**7 min** on one 12 GB GPU (12.8k examples, deterministic), peak **~3.0 GB**.
- **Adapter size:** **21.9 MB** (vs 184 M-param base — 1.4% trainable).
- **Latency:** **25 ms/record** (LoRA) / **24 ms** (hybrid) — **meets the < 50 ms target.**
  Few-shot decoder (Qwen-1.5B) is **not viable** (≥168 ms/record and ~0 recall on hard inputs).

## Question 3 — Recommendation: pure-rules / pure-LoRA / hybrid

**Recommend the HYBRID (regex pre-filter ∪ LoRA)** for a recall-first (breach-averse) deployment:
- **Best recall (0.688)** and highest binary recall (0.908) of any system, at **24 ms** — it combines
  regex's format strength (SSN 1.00, IP 1.00, PHONE 1.00, DATE 0.61) with LoRA's domain IDs (MRN 0.80,
  DEVICE 1.00, ACCOUNT 1.00, VEHICLE 0.89), and still beats Presidio's precision by 3.4×.
- **Trade-off:** hybrid precision (0.345) is lower than pure-LoRA (0.607) because the regex pre-filter
  re-introduces look-alike false positives; pair it with a human review queue on flagged spans.
- **Pure-LoRA** is the pick if precision matters more than the last points of recall (0.607 P / 0.569 R,
  38 FP). **Pure-rules is insufficient** (regex recall 0.29; Presidio precision 0.10).
- **Next lever to close the recall gap:** a *targeted* hybrid (rules only for the format-strong
  categories LoRA misses — SSN/IP/DATE) would recover much of the hybrid's lost precision; and more/
  more-varied training coverage of terse free-text positives should lift LoRA's own recall.

## Caveat + real-data validation plan

**This is 100% synthetic data, including the "hard" test set — we built both the examples and the
labels.** Synthetic realism is the single biggest threat to these numbers; real clinical/PII text has
messier formats, OCR noise, multilingual fragments, and distribution we did not model. **Before any
production trust:**
1. Assemble a small **real, expert-annotated** eval set (a few hundred records) under IRB/DUA; keep it
   held-out. 2. Re-run this exact harness on it (regex / Presidio / LoRA / hybrid) at matched recall.
3. Compare per-category recall to the synthetic numbers to size the sim-to-real gap.
4. If LoRA transfers, fine-tune/continue-train on a small real slice; re-measure. 5. Add a
   human-in-the-loop review of flagged spans and track breach-relevant **false-negative** rate in
   shadow mode before enforcing redaction.

---
*Full results: [comparison_table.md](comparison_table.md) · [error_analysis.md](error_analysis.md) ·
[sweep_results.md](sweep_results.md) · reproduce with `bash scripts/run_all.sh`.*
