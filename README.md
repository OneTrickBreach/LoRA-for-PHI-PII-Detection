# PHI/PII Detection via LoRA — 2-Week POC

Proof-of-value: a cheaply fine-tuned encoder (**LoRA on DeBERTa-v3-base**) that detects PHI/PII at
the **span level** (char ranges + category) and **beats Microsoft Presidio at matched recall**
(recall ≥ 0.97). 100% synthetic data, English-only, recall-first. See [plan.md](plan.md) for the
build plan and [rules.md](rules.md) for the hard constraints (PHI rubric, data integrity, modeling,
eval) — they are the ground truth.

> **Project complete.** Start with the **[HANDOFF](HANDOFF.md)** (guided entry point) and the
> **[one-page memo](reports/memo.md)** (executive verdict). Bottom line: on the hard test set LoRA
> beats Presidio (recall 0.569 vs 0.491, precision 0.607 vs 0.101, 25 ms/rec); the recommended
> deployment is the **hybrid** (LoRA ∪ regex, recall 0.688). No system hits the 0.97 recall bar on
> the hard set — see the memo.

## Status
- **Day 1 — DONE.** Environment up (uv, Python 3.12, torch cu128 on RTX 5070 Ti), `deberta-v3-base`
  loads with a trainable LoRA classifier head, fast-tokenizer `offset_mapping` confirmed, rubric
  applied to 10 tricky examples. See [reports/day1_environment.md](reports/day1_environment.md) and
  [reports/rubric_examples.md](reports/rubric_examples.md).
- **Day 2 — DONE.** Insertion-based synthetic generator (Faker + custom ID generators) across record
  shapes A/B/C; ~2k labeled rows; char-span→BIO alignment verified on 5 hand-checked examples;
  entity/template-level split pools confirmed disjoint; 22 unit tests pass. See
  [reports/day2_alignment.md](reports/day2_alignment.md) and
  [reports/day2_data_summary.md](reports/day2_data_summary.md).
- **Day 3 — DONE.** All three baselines (regex, Presidio, few-shot Qwen2.5-1.5B) implemented and
  scored on the v1 test set via a shared eval harness (overlap/exact span P/R/F1, per-category
  recall, binary recall, FP-on-negatives, measured latency). **Presidio (the bar): overlap recall
  0.825, precision 0.213.** See [reports/comparison_table.md](reports/comparison_table.md).
- **Day 4 — DONE.** First LoRA fine-tune on DeBERTa-v3 (classifier head learns; trains on GPU,
  deterministic, 21.9 MB adapter) + LoRA wired into the harness with recall-first thresholding.
  **At matched recall (0.85 vs 0.83), LoRA precision 0.61 vs Presidio 0.21 (~3×)**, with 4–9 false
  positives vs 308 and MRN recall 1.00 vs 0.00, latency 26 ms (< 50 ms target); recall does **not**
  yet reach the 0.97 bar on v1 (a finding — Day 6 scales the data). See
  [reports/day4_training.md](reports/day4_training.md).
- **Day 5 — DONE.** Automated leakage/overlap check passes (0 identifier & template overlap across
  splits), data regenerates reproducibly, README week-1 reproduction, mid-project self-review. See
  [reports/day5_leakage_check.md](reports/day5_leakage_check.md) and
  [reports/day5_selfreview.md](reports/day5_selfreview.md).
- **Day 6 — DONE.** Scaled to **16k main (train/val/test) + 1.5k dedicated hard test set** via a
  balanced, per-category-partitioned template bank. Every category ≥300 in train; val/test/hard_test
  cover all 17 categories; **0 leakage across all 4 splits**; regenerates reproducibly. See
  [reports/day6_data_summary.md](reports/day6_data_summary.md). (Retrain on v2 is Day 7.)
- **Day 7 — DONE.** Retrained on v2 (val token-recall 0.60→0.96) and ran the full comparison on the
  **hard test set**. On the hard set LoRA Pareto-dominates Presidio (recall 0.557 vs 0.491, precision
  0.568 vs 0.101, 40 FP vs 1,739) — but **no system meets the 0.97 recall bar**. Error analysis
  ([reports/error_analysis.md](reports/error_analysis.md)): LoRA owns domain IDs (MRN/DEVICE/VEHICLE/
  ACCOUNT), rules own format-strong PHI (SSN/IP/DATE) → motivates a hybrid (Day 9).
- **Day 8 — DONE.** Hyperparameter sweep (r/alpha/LR) + recall-first thresholding →
  [reports/sweep_results.md](reports/sweep_results.md). **Recommended: r=16, alpha=16, lr=2e-4**
  (hard-test recall 0.584, precision 0.607, 37 ms/rec) — adopted in `config.yaml`. Bigger rank
  didn't help and broke the latency target; no config hits 0.97 recall on the hard set (→ hybrid).
- **Days 9 & 10 — DONE (project close).** Clean final run with the recommended config; hybrid
  (LoRA ∪ regex) evaluated → **recall 0.688** (best); pure-rules/LoRA/hybrid recommendation with the
  latency finding. Memo + handoff written. See [HANDOFF.md](HANDOFF.md) and
  [reports/memo.md](reports/memo.md). 59 unit tests green; pipeline reproducible end-to-end.

## Common commands
```bash
python -m src.sanity_check            # Day 1: environment + LoRA-gotcha checks
python -m src.generate --version v2   # Day 6: full-scale data (16k) + hard test set -> data/
python -m src.align                   # Day 2: verify char-span->BIO alignment (5 hand-checked)
python -m src.leakage_check           # Day 5: assert zero entity/template overlap across splits
python -m src.train_lora              # Day 4: LoRA fine-tune -> artifacts/lora_adapter
python -m src.evaluate --systems regex presidio fewshot lora hybrid --split hard_test  # comparison
python -m src.error_analysis --split hard_test   # where ML wins / rules suffice + examples
python -m src.sweep                   # hyperparameter sweep -> reports/sweep_results.md
python -m pytest -q                   # run all unit tests (59)
```

## Reproduce end-to-end (one command)
```bash
bash scripts/run_all.sh               # generate v2 (+ hard test set) -> leakage check -> train LoRA
                                      # -> score all systems incl. hybrid -> reports/comparison_table.md
```
Everything is seeded (`config.yaml: seed`) and deterministic, so a repeat run reproduces byte-identical
data and the same model. Generated data and the trained adapter are git-ignored and regenerated from
seed.

## Environment setup (uv — required; see rules.md §6.7)

This project is **uv-managed**. Do not use system `pip`/`python`. Activate the venv before any work.

```bash
# 1. Create the venv (Python 3.12)
uv venv --python 3.12 .venv

# 2. Activate it
source .venv/Scripts/activate     # Windows (Git Bash);  .venv/bin/activate on Linux/macOS

# 3. Install pinned deps. --torch-backend=auto selects the cu128 wheel on this Blackwell GPU
#    (and the CPU wheel on CPU-only machines).
uv pip install -r requirements.txt --torch-backend=auto

# 4. spaCy model for the Presidio baseline (install as a wheel via uv — `spacy download`
#    relies on pip, which uv venvs don't ship).
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
```

### Verify the environment (Day 1)
```bash
python -m src.sanity_check     # writes reports/day1_environment.md; asserts the LoRA gotchas
```

## Hardware used for reported numbers
- GPU: **NVIDIA RTX 5070 Ti Laptop (12 GB, Blackwell sm_120)**, CUDA build 12.8
- torch `2.7.1+cu128`, transformers `4.49.0`, peft `0.14.0`, Python `3.12.10`, Windows 11
- All latency/memory/metrics are **measured on this hardware** (rules.md §1.6), never estimated.

## Repository layout
```
config.yaml         # seeds, paths, composition targets, hyperparameters (config out of code)
requirements.txt    # pinned deps
src/
  config.py         # config + seed loader, BIO label space
  sanity_check.py   # Day 1 environment check
  generate.py       # insertion-based synthetic generator        (Day 2)
  id_generators.py  # MRN/NPI/PLAN_ID/DEVICE_ID/... generators    (Day 2)
  templates.py      # carrier templates, shapes A/B/C             (Day 2)
  align.py          # char-span -> BIO via offset_mapping         (Day 2)
  leakage_check.py  # zero entity/template overlap across splits  (Day 5)
  baselines/        # regex / presidio / fewshot                  (Day 3)
  train_lora.py     # LoRA fine-tune                              (Day 4)
  predict.py        # unified predict interface                   (Day 4)
  evaluate.py       # eval harness -> comparison table            (Day 4)
scripts/run_all.sh  # one command: train + score all systems
reports/            # day1_environment, rubric_examples, comparison_table, error_analysis, memo
data/raw, data/pools
```

## Reproducibility
Fixed global seed in `config.yaml` (`20260629`), logged in every stochastic step; pinned versions
above; one-command runs via `scripts/run_all.sh` (wired Day 4). Generated data and adapters are
git-ignored and regenerated from seed.
