"""Unified predict interface for every system (plan.md §4). Implemented Day 4.

All systems (regex / presidio / fewshot / lora) expose predict(text) -> list[{start,end,type}]
so the eval harness scores them identically. Also exposes score-threshold control for the
LoRA system to support recall-first operating-point selection (plan.md §10).
"""
# TODO(Day 4): dispatch to each system; LoRA token-logits -> BIO -> merged char spans + scores.
