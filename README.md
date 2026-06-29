# PHI/PII Detection via LoRA — 2-Week POC

Proof-of-value: a cheaply fine-tuned encoder (**LoRA on DeBERTa-v3-base**) that detects PHI/PII at
the **span level** (char ranges + category) and **beats Microsoft Presidio at matched recall**
(recall ≥ 0.97). 100% synthetic data, English-only, recall-first. See [plan.md](plan.md) for the
build plan and [rules.md](rules.md) for the hard constraints (PHI rubric, data integrity, modeling,
eval) — they are the ground truth.

## Status
- **Day 1 — DONE.** Environment up (uv, Python 3.12, torch cu128 on RTX 5070 Ti), `deberta-v3-base`
  loads with a trainable LoRA classifier head, fast-tokenizer `offset_mapping` confirmed, rubric
  applied to 10 tricky examples. See [reports/day1_environment.md](reports/day1_environment.md) and
  [reports/rubric_examples.md](reports/rubric_examples.md).
- Day 2–10: see the phase breakdown in [plan.md](plan.md) §11.

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
