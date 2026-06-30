"""Day-2 tests: generator invariants + entity/template-level split disjointness (rules.md §3)."""
import random

from src.config import load_config
from src.generate import (SplitGen, build_A, build_B, build_C,
                          partition_template_ids)
from src.id_generators import IdGen

VALID_RECORD_TYPES = {"api_request", "intake_form", "facility_form", "log_line"}


def _fresh(seed=42):
    cfg = load_config()
    gens = IdGen(seed)
    sel = random.Random(seed + 1)
    parts = partition_template_ids(sel, cfg["data"]["split_fractions"])
    return gens, sel, parts


def test_contains_phi_is_derived_and_offsets_valid():
    gens, sel, parts = _fresh()
    used: set[str] = set()
    sg = SplitGen(gens, sel, parts["train"]["pos"], parts["train"]["neg"], used)
    builders = [build_A, build_B, build_C]
    for i in range(120):
        positive = (i % 2 == 0)
        text, spans = builders[i % 3](sg, positive)
        contains = 1 if spans else 0
        assert contains == (1 if positive else 0), "contains_phi must equal span presence"
        # offsets exact + non-overlapping + within bounds
        last_end = -1
        for s in sorted(spans, key=lambda x: x["start"]):
            assert 0 <= s["start"] < s["end"] <= len(text)
            assert text[s["start"]:s["end"]], "span must select a non-empty substring"
            assert s["start"] >= last_end, "spans must not overlap"
            last_end = s["end"]


def test_negatives_have_no_spans():
    gens, sel, parts = _fresh(7)
    used: set[str] = set()
    sg = SplitGen(gens, sel, parts["train"]["pos"], parts["train"]["neg"], used)
    for builder in (build_A, build_B, build_C):
        for _ in range(20):
            _, spans = builder(sg, positive=False)
            assert spans == [], "negative records must contain only look-alikes (no PHI spans)"


def test_template_partition_is_disjoint():
    _, sel, parts = _fresh(5)
    ids = {sp: {t["id"] for t in parts[sp]["pos"] + parts[sp]["neg"]}
           for sp in ("train", "val", "test")}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


def test_identifier_pools_are_disjoint_across_splits():
    """Generate train then val sharing the global used set; their identifier pools must not overlap."""
    cfg = load_config()
    gens = IdGen(11)
    sel = random.Random(12)
    parts = partition_template_ids(sel, cfg["data"]["split_fractions"])
    used: set[str] = set()

    sg_train = SplitGen(gens, sel, parts["train"]["pos"], parts["train"]["neg"], used)
    for i in range(150):
        (build_A, build_B, build_C)[i % 3](sg_train, positive=(i % 2 == 0))

    sg_val = SplitGen(gens, sel, parts["val"]["pos"], parts["val"]["neg"], used)
    for i in range(150):
        (build_A, build_B, build_C)[i % 3](sg_val, positive=(i % 2 == 0))

    assert sg_train.identifiers.isdisjoint(sg_val.identifiers), \
        "poolable PHI identifiers must not cross splits (rules.md §3)"


def test_generated_jsonl_on_disk_is_valid():
    """If v1 data has been generated, validate the committed-format invariants on disk."""
    import json
    from pathlib import Path

    from src.config import REPO_ROOT
    train = Path(REPO_ROOT) / "data" / "raw" / "train.jsonl"
    if not train.exists():
        import pytest
        pytest.skip("run `python -m src.generate --version v1` first")
    n = 0
    with open(train, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            assert set(rec) == {"text", "spans", "contains_phi", "record_type"}
            assert rec["record_type"] in VALID_RECORD_TYPES
            assert rec["contains_phi"] == (1 if rec["spans"] else 0)
            for s in rec["spans"]:
                assert rec["text"][s["start"]:s["end"]]
            n += 1
    assert n > 0
