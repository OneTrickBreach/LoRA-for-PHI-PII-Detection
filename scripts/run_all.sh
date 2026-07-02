#!/usr/bin/env bash
# One command: generate (if needed) -> train LoRA -> score ALL systems -> emit comparison table.
# rules.md §6.1 (one-command runs), §5.7 (one command runs the full comparison).
# Wired up Day 4; full version Day 9.
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

SPLIT="${1:-test}"   # eval split (default: test; Day 6+ uses hard_test)

echo "==> [1/4] Generate synthetic data (v1)"
python -m src.generate --version v1

echo "==> [2/4] Leakage check (must pass before any number is trusted)"
python -m src.leakage_check     # exits non-zero on any cross-split overlap -> stops the run

echo "==> [3/4] Train LoRA on DeBERTa-v3"
python -m src.train_lora

echo "==> [4/4] Score all systems on '${SPLIT}' -> reports/comparison_table.md"
python -m src.evaluate --systems regex presidio fewshot lora --split "${SPLIT}"

echo "Done. See reports/comparison_table.md"
