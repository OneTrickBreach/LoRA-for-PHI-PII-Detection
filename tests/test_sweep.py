"""Day-8 tests: sweep config validity, best-config selection, and train() signature."""
import inspect

from src.sweep import CONFIGS, _pick_best


def test_sweep_configs_are_valid_and_unique():
    tags = [c["tag"] for c in CONFIGS]
    assert len(set(tags)) == len(tags), "sweep tags must be unique"
    for c in CONFIGS:
        assert c["r"] in (8, 16, 32)
        assert c["alpha"] in (c["r"], 2 * c["r"]), "alpha should be r or 2r (plan §9)"
        assert 1.0e-4 <= c["lr"] <= 2.0e-4
        assert c["epochs"] in (2, 3)


def test_sweep_touches_each_axis():
    ranks = {c["r"] for c in CONFIGS}
    assert {8, 16, 32} <= ranks, "sweep must cover r in {8,16,32}"
    r16 = [c for c in CONFIGS if c["r"] == 16]
    assert {c["alpha"] for c in r16} >= {16, 32}, "must compare alpha=r vs 2r at r=16"
    assert len({c["lr"] for c in r16}) >= 2, "must compare two learning rates at r=16"


def test_pick_best_prefers_recall_then_precision():
    rows = [
        {"tag": "a", "hard_argmax_recall": 0.50, "hard_argmax_precision": 0.90},
        {"tag": "b", "hard_argmax_recall": 0.60, "hard_argmax_precision": 0.40},
        {"tag": "c", "hard_argmax_recall": 0.60, "hard_argmax_precision": 0.70},
    ]
    assert _pick_best(rows)["tag"] == "c"   # recall ties b vs c -> higher precision wins


def test_train_function_accepts_hyperparameter_overrides():
    from src.train_lora import train
    params = set(inspect.signature(train).parameters)
    assert {"r", "lora_alpha", "learning_rate", "epochs", "adapter_out"} <= params
