"""Day-2 tests: identifier generators — formats, determinism, and look-alike hard-negative traps."""
import re

from src.config import label_list
from src.id_generators import KINDS, IdGen

SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
PHONE_RE = re.compile(r"^(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def test_generation_is_deterministic():
    a = IdGen(7)
    b = IdGen(7)
    for _ in range(50):
        assert a.name() == b.name()
        assert a.ssn() == b.ssn()
        assert a.mrn() == b.mrn()


def test_kinds_cover_every_label_type():
    phi_types = {meta[1] for meta in KINDS.values()}
    assert phi_types == set(label_list()), "every PHI category must have a generator"


def test_phi_formats():
    g = IdGen(1)
    for _ in range(100):
        assert SSN_RE.match(g.ssn())
        assert re.match(r"^\d{10}$", g.npi())
        assert re.match(r"^[A-Z]{2,3}\d{9,11}$", g.plan_id())
        assert 90 <= int(g.age90()) <= 106
        assert 1 <= int(g.age_small()) <= 89


def test_lookalikes_trip_the_naive_regexes():
    # These NON-PHI look-alikes deliberately match the regex baseline patterns — that is the
    # whole point of hard negatives (rules.md §2, §7).
    g = IdGen(2)
    for _ in range(100):
        assert SSN_RE.match(g.ssn_ticket()), "SSN-shaped ticket should match SSN regex"
        assert PHONE_RE.match(g.support_phone()), "support line matches phone regex"
        assert IP_RE.match(g.infra_ip()), "infra IP matches IP regex"


def test_value_for_polarity():
    g = IdGen(3)
    val, phi_type, poolable = g.value_for("mrn", positive=True)
    assert phi_type == "MRN" and poolable is True and val
    val, phi_type, poolable = g.value_for("mrn", positive=False)
    assert phi_type is None, "negative slot must not be labeled PHI"


def test_age_is_not_poolable():
    # age 92 must be allowed to recur across splits (documented exception, rules.md §8).
    assert KINDS["age"][3] is False
