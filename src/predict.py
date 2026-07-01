"""Unified predict interface for every system (plan.md §4).

Every system exposes `predict(text) -> [{"start","end","type"}]` so the eval harness scores them
identically. Day 3 wires the three baselines; Day 4 adds the LoRA system (with a score threshold for
recall-first operating-point selection).
"""
from __future__ import annotations


def load_predictor(system: str, cfg: dict | None = None):
    if system == "regex":
        from src.baselines.regex_baseline import RegexBaseline
        return RegexBaseline(cfg)
    if system == "presidio":
        from src.baselines.presidio_baseline import PresidioBaseline
        return PresidioBaseline()
    if system == "fewshot":
        from src.baselines.fewshot_baseline import FewShotBaseline
        return FewShotBaseline(cfg)
    if system == "lora":
        raise NotImplementedError("LoRA predictor is wired on Day 4 (train_lora.py first).")
    raise ValueError(f"unknown system: {system}")
