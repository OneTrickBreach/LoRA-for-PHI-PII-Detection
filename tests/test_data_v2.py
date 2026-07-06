"""Day-6 tests: balanced per-category coverage, hard bank validity, and hard-test disjointness."""
import random
from collections import Counter

from src.config import load_config
from src.generate import SplitGen, build_A, partition_template_ids
from src.id_generators import KINDS
from src.templates import (CARRIER_TEMPLATES, HARD_NEG_TEMPLATES,
                           HARD_POS_TEMPLATES)


def test_every_split_covers_every_kind():
    """Per-kind partition must give train/val/test ≥1 pos AND ≥1 neg template for EVERY category."""
    cfg = load_config()
    parts = partition_template_ids(random.Random(1), cfg["data"]["split_fractions"])
    kinds = set(KINDS)
    for sp in ("train", "val", "test"):
        pos_kinds = {t["kind"] for t in parts[sp]["pos"]}
        neg_kinds = {t["kind"] for t in parts[sp]["neg"]}
        assert pos_kinds == kinds, f"{sp} missing pos kinds: {kinds - pos_kinds}"
        assert neg_kinds == kinds, f"{sp} missing neg kinds: {kinds - neg_kinds}"


def test_partition_still_disjoint_across_splits():
    cfg = load_config()
    parts = partition_template_ids(random.Random(2), cfg["data"]["split_fractions"])
    ids = {sp: {t["id"] for t in parts[sp]["pos"] + parts[sp]["neg"]}
           for sp in ("train", "val", "test")}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


def test_round_robin_balances_positive_categories():
    cfg = load_config()
    parts = partition_template_ids(random.Random(3), cfg["data"]["split_fractions"])
    sg = SplitGen(IdGen_stub(), random.Random(4), parts["train"]["pos"],
                  parts["train"]["neg"], set())
    kinds = Counter(sg.next_pos_template()["kind"] for _ in range(len(sg.pos_kinds) * 5))
    # each kind chosen exactly 5 times over 5 full cycles -> perfectly balanced
    assert set(kinds) == set(sg.pos_kinds)
    assert all(v == 5 for v in kinds.values())


def test_hard_bank_templates_have_one_slot_and_disjoint_ids():
    main_ids = {t["id"] for t in CARRIER_TEMPLATES}
    hard = HARD_POS_TEMPLATES + HARD_NEG_TEMPLATES
    hard_ids = {t["id"] for t in hard}
    assert main_ids.isdisjoint(hard_ids), "hard bank IDs must never appear in train/val/test"
    for t in hard:
        assert t["text"].count("{x}") == 1, f"{t['id']} must have exactly one slot"
        assert t["kind"] in KINDS


def test_hard_positives_produce_spans_and_negatives_do_not():
    parts_pos = HARD_POS_TEMPLATES
    sg = SplitGen(_real_idgen(), random.Random(5), HARD_POS_TEMPLATES,
                  HARD_NEG_TEMPLATES, set())
    for _ in range(20):
        _, spans_pos = build_A(sg, positive=True)
        assert spans_pos, "hard positive must contain PHI"
    for _ in range(20):
        _, spans_neg = build_A(sg, positive=False)
        assert spans_neg == [], "hard negative must be look-alikes only"


# --- helpers ---
def _real_idgen():
    from src.id_generators import IdGen
    return IdGen(123)


class IdGen_stub:
    """Minimal stub so next_pos_template can be exercised without generating identifiers."""
    def value_for(self, kind, positive):
        return ("x", None, False)
