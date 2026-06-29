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

# TODO(Day 4): python -m src.generate --version v1
# TODO(Day 4): python -m src.train_lora
# TODO(Day 4): python -m src.evaluate --systems regex presidio fewshot lora --split hard_test
echo "run_all.sh is a stub until Day 4."
