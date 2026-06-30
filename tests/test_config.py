"""Day-1 tests: config loading, label space, seed determinism."""
import random

from src.config import bio_label_list, label_list, load_config, set_global_seed


def test_config_has_required_keys():
    cfg = load_config()
    for key in ("seed", "paths", "label_types", "model", "lora", "train", "eval", "data"):
        assert key in cfg, f"missing config key: {key}"


def test_label_types_count_and_unique():
    types = label_list()
    assert len(types) == 17
    assert len(set(types)) == 17, "label types must be unique"


def test_bio_label_space():
    labels = bio_label_list()
    # O + B-/I- per type
    assert labels[0] == "O"
    assert len(labels) == 35  # O + B-/I- for each of the 17 types
    for t in label_list():
        assert f"B-{t}" in labels and f"I-{t}" in labels


def test_lora_modules_to_save_present():
    # rules.md §4.1 — the silent killer must be configured.
    cfg = load_config()
    assert cfg["lora"]["modules_to_save"] == ["classifier"]
    assert cfg["lora"]["target_modules"] == "all-linear"


def test_training_uses_bf16_not_fp16():
    # DeBERTa-v3 overflows in fp16; bf16 is required (Day-1 review fix).
    cfg = load_config()
    assert cfg["train"]["bf16"] is True
    assert cfg["train"]["fp16"] is False


def test_seed_is_deterministic():
    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b
