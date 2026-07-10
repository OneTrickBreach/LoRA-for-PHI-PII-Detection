#!/usr/bin/env bash
# One command: generate v2 (+ hard test set) -> leakage check -> train LoRA -> score ALL systems
# (incl. hybrid) on the hard test set -> emit comparison table.
# rules.md §6.1 (one-command runs), §5.7 (one command runs the full comparison).
set -euo pipefail

# Activate the uv-managed venv (rules.md §6.7). Works on Windows (Scripts) and POSIX (bin).
if [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "No .venv found. Create it: uv venv --python 3.12 .venv && uv pip install -r requirements.txt --torch-backend=auto" >&2
  exit 1
fi

SPLIT="${1:-hard_test}"   # eval split (default: hard_test, the verdict set)

echo "==> [1/4] Generate synthetic data v2 (16k + 1.5k hard test set)"
python -m src.generate --version v2

echo "==> [2/4] Leakage check (must pass before any number is trusted)"
python -m src.leakage_check     # exits non-zero on any cross-split overlap -> stops the run

echo "==> [3/4] Train LoRA on DeBERTa-v3 (recommended config from config.yaml)"
python -m src.train_lora

echo "==> [4/4] Score all systems (incl. hybrid) on '${SPLIT}' -> reports/comparison_table.md"
python -m src.evaluate --systems regex presidio fewshot lora hybrid --split "${SPLIT}"

echo "Done. See reports/comparison_table.md"
