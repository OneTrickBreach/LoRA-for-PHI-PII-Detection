"""LoRA fine-tune on DeBERTa-v3 for token classification (plan.md §9, rules.md §4). Implemented Day 4.

CRITICAL: modules_to_save=["classifier"] (config.yaml lora) or the head never learns —
accuracy stuck at chance (plan.md §9 Gotcha A, rules.md §4.1). target_modules="all-linear".
Labels built via src.align (verified before full training). Logs seed, versions, adapter size.

Usage (Day 4+):  python -m src.train_lora
"""
# TODO(Day 4): build datasets via align, LoraConfig per config.yaml, Trainer, save adapter.
