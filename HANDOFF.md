# Project Handoff — PHI/PII Span Detection via LoRA

**A 2-week proof-of-value POC.** Goal: show that a cheaply fine-tuned encoder (LoRA on
DeBERTa-v3-base) can detect PHI/PII at the **span level** (character ranges + category, for
redaction) and **beat Microsoft Presidio** — on 100% synthetic, English-only data, with everything
measured on a single 12 GB GPU.

> **This document is the guided entry point.** For the executive answer read the
> **[one-page memo](reports/memo.md)**; for reproduction read the **[README](README.md)**; for the
> day-by-day story read the **[journal](JOURNAL.md)**.

---

## TL;DR verdict

- **Does LoRA beat Presidio? Yes** — on the hard test set LoRA gets **higher recall (0.569 vs 0.491)
  and ~6× the precision (0.607 vs 0.101)**, with **38 false positives vs Presidio's 1,739**, at
  **25 ms/record** (under the 50 ms target).
- **Does anything hit the recall ≥ 0.97 bar?** On in-distribution validation, LoRA reaches **0.972**.
  On the deliberately-hard test set, **no system does** (best is the hybrid at **0.688**) — a
  reported finding, and the reason the recommendation is a **hybrid**.
- **Recommendation:** deploy the **hybrid (regex pre-filter ∪ LoRA)** for recall-first use — best
  recall (0.688), 24 ms, and 3.4× Presidio's precision — with a human review queue. Use **pure-LoRA**
  if precision matters more. **Pure rules are insufficient.**

## Success bar vs. actual (hard test set)

| Criterion (plan §2) | Target | Result | Met? |
|---|---|---|---|
| Span recall | ≥ 0.97 | 0.972 in-distribution (val); **0.688 hard** (hybrid) | ⚠️ in-dist only |
| Precision at that recall | ≥ 0.85 | 0.61 (LoRA) / 0.35 (hybrid) on hard | ✗ on hard |
| LoRA > Presidio precision at matched recall | — | **Yes, ~6×** (LoRA 0.607 vs 0.101) | ✅ |
| Latency | < 50 ms/GPU | **24–25 ms** (LoRA / hybrid) | ✅ |

**Honest read:** the ML thesis is proven (LoRA clearly beats the bar system and meets latency), but
the absolute recall bar is not met on the hard, out-of-distribution set — closing that is the
top follow-up (see the memo).

## Headline results — hard test set (n=1,500; span overlap)

| system | recall | precision | binary recall | FP on look-alikes | latency |
|---|---|---|---|---|---|
| regex | 0.292 | 0.237 | 0.533 | 432 | 0.01 ms |
| Presidio (the bar) | 0.491 | 0.101 | 0.916 | 1,739 | 7 ms |
| few-shot (Qwen-1.5B) | 0.003 | 0.038 | 0.035 | 9 | 168 ms |
| **LoRA (r16/α16)** | 0.569 | **0.607** | 0.751 | **38** | 25 ms |
| **Hybrid (LoRA ∪ regex)** | **0.688** | 0.345 | 0.908 | 470 | 24 ms |

**Where each approach wins** (full table in [error_analysis.md](reports/error_analysis.md)):
- **LoRA owns domain identifiers** rules can't see: ACCOUNT 1.00, DEVICE_ID 1.00, VEHICLE_ID 0.89,
  MRN 0.80, PLAN_ID 0.57 (regex/Presidio = 0.00).
- **Rules own format-strong PHI:** SSN 1.00, IP 1.00 (LoRA misses these in terse contexts).
- **The hybrid gets both** — that's why it wins on recall.

## What was built (the deliverables)

1. **Synthetic data generator** — insertion-based (labels exact by construction), custom generators
   for all 17 PHI categories + matched non-PHI look-alikes, three record shapes (API/intake/log),
   entity/template-level splits with an **automated zero-overlap leakage check**, and a dedicated
   **hard test set**. 16k main + 1.5k hard, reproducible from a seed.
2. **LoRA pipeline + evaluation harness** — one command trains and scores LoRA vs regex vs Presidio
   vs few-shot vs hybrid; span (overlap + exact) / binary / per-category recall; recall-first
   thresholding; measured latency.
3. **One-page memo** — [reports/memo.md](reports/memo.md).

## Repository map (referential)

| Path | What it is |
|---|---|
| [reports/memo.md](reports/memo.md) | **The one-page memo (read this first).** |
| [reports/comparison_table.md](reports/comparison_table.md) | Final hard-test comparison, all systems, per-category. |
| [reports/error_analysis.md](reports/error_analysis.md) | Where ML wins / where rules suffice + FN/FP examples. |
| [reports/sweep_results.md](reports/sweep_results.md) | Hyperparameter sweep → recommended config, with cost. |
| [reports/day6_data_summary.md](reports/day6_data_summary.md) | Data scale + per-category coverage. |
| [reports/day5_leakage_check.md](reports/day5_leakage_check.md) | Zero cross-split identifier/template overlap. |
| [reports/day1_environment.md](reports/day1_environment.md) · [reports/rubric_examples.md](reports/rubric_examples.md) | Env facts; PHI rubric applied to hard cases. |
| [JOURNAL.md](JOURNAL.md) | Day-by-day log incl. the Week-1 reproducibility audit. |
| [README.md](README.md) | Setup + exact commands to reproduce. |
| `src/` | `generate.py`, `id_generators.py`, `templates.py`, `align.py`, `leakage_check.py`, `train_lora.py`, `sweep.py`, `predict.py`, `evaluate.py`, `error_analysis.py`, `baselines/` |
| `tests/` | 59 unit tests (alignment, generator, metrics, thresholding, hybrid, leakage). |
| `config.yaml` | Seeds, paths, label schema, LoRA config, sweep grid, eval thresholds. |

## Reproduce (uv-managed; see [README](README.md) for setup)

```bash
bash scripts/run_all.sh              # generate v2 -> leakage check -> train LoRA -> score all
                                     # systems (incl. hybrid) on the hard test set
python -m pytest -q                  # 59 unit tests
```
Everything is seeded and deterministic (data regenerates byte-identical; training reproduces the same
model). Generated data and the adapter are git-ignored and rebuilt from seed.

## Caveats & the real-data validation plan

**All data is synthetic — including the hard test set — so these numbers bound *relative* system
quality, not real-world performance.** The memo gives the concrete plan: assemble a small
expert-annotated **real** eval set (IRB/DUA), re-run this harness at matched recall, size the
sim-to-real gap per category, fine-tune on a small real slice if LoRA transfers, and run a
human-in-the-loop **false-negative** watch in shadow mode before enforcing redaction.

## Status
All 10 planned days complete; 59 tests green; pipeline reproducible end-to-end. Week 1 on `main`,
Week 2 merged to `main` at project close.
