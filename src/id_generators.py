"""Custom domain identifier generators — Faker does NOT cover these (rules.md §7 anti-pattern).

Each PHI "kind" is paired with a deliberately-confusable NON-PHI look-alike so the model must learn
context, not "numbers = PHI" (rules.md §2 look-alikes, §7 flag-everything degeneracy). Examples:
  MRN (PHI) vs order number (look-alike) · SSN (PHI) vs SSN-shaped case ticket ·
  personal phone (PHI) vs public support line · session IP (PHI) vs infra IP ·
  patient name (PHI) vs provider "Dr. X" (NOT PHI) · home address (PHI) vs public business address.

All generators draw from a single seeded RNG/Faker so generation is reproducible (rules.md §6.2).
`KINDS` maps a slot kind -> (phi_generator, PHI type, lookalike_generator, poolable). `poolable`
flags high-cardinality identifiers whose values must be DISJOINT across splits (rules.md §3). AGE90
is intentionally NOT poolable: an age like 92 must be allowed to appear in every split.
"""
from __future__ import annotations

import random
import string

from faker import Faker

_DIGITS = "0123456789"
_ALNUM = string.ascii_uppercase + _DIGITS
_UPPER = string.ascii_uppercase


class IdGen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.fake = Faker("en_US")
        self.fake.seed_instance(seed)

    # ---- helpers ----
    def _d(self, n: int) -> str:
        return "".join(self.rng.choice(_DIGITS) for _ in range(n))

    def _a(self, n: int) -> str:
        return "".join(self.rng.choice(_ALNUM) for _ in range(n))

    def _u(self, n: int) -> str:
        return "".join(self.rng.choice(_UPPER) for _ in range(n))

    def _octet(self) -> str:
        return str(self.rng.randint(0, 255))

    # ---- PHI generators ----
    def name(self) -> str:
        return self.fake.name()

    def address(self) -> str:
        return (f"{self.fake.street_address()}, {self.fake.city()}, "
                f"{self.fake.state_abbr()} {self.fake.postcode()}").replace("\n", ", ")

    def dob_date(self) -> str:
        d = self.fake.date_of_birth(minimum_age=1, maximum_age=89)
        return d.strftime(self.rng.choice(["%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y"]))

    def ssn(self) -> str:
        return self.fake.ssn()

    def mrn(self) -> str:
        style = self.rng.randint(0, 3)
        if style == 0:
            return "MRN-" + self._d(self.rng.randint(6, 10))
        if style == 1:
            return "A" + self._d(self.rng.randint(5, 8))
        if style == 2:
            return self._d(self.rng.randint(7, 10))
        return "MRN:" + self._d(self.rng.randint(6, 9))

    def npi(self) -> str:
        return self._d(10)

    def plan_id(self) -> str:
        return self._u(self.rng.randint(2, 3)) + self._d(self.rng.randint(9, 11))

    def account(self) -> str:
        return "ACCT-" + self._d(4) + "-" + self._d(4)

    def license(self) -> str:
        return self._u(1) + self._d(3) + "-" + self._d(4) + "-" + self._d(2)

    def device_id(self) -> str:
        return "SN-" + self._d(4) + "-" + self._a(3) + "-" + self._d(3)

    def vehicle_id(self) -> str:
        if self.rng.random() < 0.5:
            return self._d(1) + self._u(3) + self._d(3)      # plate-like
        return self._a(17)                                    # VIN-like

    def phone(self) -> str:
        a, b, c = self._d(3), self._d(3), self._d(4)
        return self.rng.choice([f"({a}) {b}-{c}", f"{a}-{b}-{c}",
                                f"+1 {a}-{b}-{c}", f"{a}.{b}.{c}"])

    def email(self) -> str:
        return self.fake.free_email()

    def personal_url(self) -> str:
        return f"https://{self.fake.user_name()}.{self.fake.tld()}/profile"

    def session_ip(self) -> str:
        return self.fake.ipv4_public()

    def age90(self) -> str:
        return str(self.rng.randint(90, 106))

    def other_id(self) -> str:
        return self._a(self.rng.randint(6, 10))

    # ---- look-alike (NON-PHI) generators ----
    def provider_last(self) -> str:
        return self.fake.last_name()

    def org_address(self) -> str:
        return (f"{self.fake.street_address()}, {self.fake.city()}, "
                f"{self.fake.state_abbr()} {self.fake.postcode()}").replace("\n", ", ")

    def build_or_year(self) -> str:
        if self.rng.random() < 0.5:
            return self.fake.date(pattern="%Y-%m-%d")        # software build date
        return str(self.rng.randint(2015, 2026))             # bare year

    def ssn_ticket(self) -> str:
        return f"{self._d(3)}-{self._d(2)}-{self._d(4)}"      # SSN-shaped case reference

    def order_no(self) -> str:
        return self._d(self.rng.randint(4, 6))

    def provider_npi(self) -> str:
        return self._d(10)

    def catalog_code(self) -> str:
        return "CAT-" + self._d(5)

    def invoice_no(self) -> str:
        return "INV-" + self._d(5)

    def cert_version(self) -> str:
        return "CERT-" + str(self.rng.randint(2018, 2026)) + "-" + self._u(3)

    def version_or_sku(self) -> str:
        if self.rng.random() < 0.5:
            return f"v{self.rng.randint(1, 9)}.{self.rng.randint(0, 20)}.{self.rng.randint(0, 99)}"
        return "MED-" + self._d(5)

    def asset_tag(self) -> str:
        return "ASSET-" + self._d(5)

    def support_phone(self) -> str:
        return self.rng.choice([f"800-555-{self._d(4)}", f"888-555-{self._d(4)}",
                                f"(800) 555-{self._d(4)}"])

    def support_email(self) -> str:
        return self.rng.choice(["support@", "info@", "help@"]) + self.fake.domain_name()

    def org_url(self) -> str:
        return f"https://{self.fake.domain_name()}"

    def infra_ip(self) -> str:
        r = self.rng.randint(0, 2)
        if r == 0:
            return f"10.{self._octet()}.{self._octet()}.{self._octet()}"
        if r == 1:
            return f"192.168.{self._octet()}.{self._octet()}"
        return f"172.16.{self._octet()}.{self._octet()}"

    def age_small(self) -> str:
        return str(self.rng.randint(1, 89))

    def request_id(self) -> str:
        return "req_" + self._a(4).lower()

    def build_timestamp(self) -> str:
        # ISO-ish log timestamp; a build/log date, NOT PHI (rules.md §2 Dates).
        return (f"{self.rng.randint(2024, 2026)}-{self.rng.randint(1, 12):02d}-"
                f"{self.rng.randint(1, 28):02d}T{self.rng.randint(0, 23):02d}:"
                f"{self.rng.randint(0, 59):02d}Z")

    def company(self) -> str:
        return self.fake.company()

    def state_abbr(self) -> str:
        return self.fake.state_abbr()

    # ---- dispatch ----
    def value_for(self, kind: str, positive: bool) -> tuple[str, str | None, bool]:
        """Return (value, phi_type_or_None, poolable) for a slot of `kind`.

        positive -> PHI value (+type); negative -> the matching look-alike (no type).
        """
        phi_method, phi_type, look_method, poolable = KINDS[kind]
        if positive:
            return getattr(self, phi_method)(), phi_type, poolable
        return getattr(self, look_method)(), None, False


# kind -> (phi_generator_method, PHI type, lookalike_generator_method, poolable)
KINDS: dict[str, tuple[str, str, str, bool]] = {
    "name":    ("name",         "NAME",       "provider_last",  True),
    "address": ("address",      "ADDRESS",    "org_address",    True),
    "dob":     ("dob_date",     "DATE",       "build_or_year",  True),
    "ssn":     ("ssn",          "SSN",        "ssn_ticket",     True),
    "mrn":     ("mrn",          "MRN",        "order_no",       True),
    "npi":     ("npi",          "NPI",        "provider_npi",   True),
    "plan":    ("plan_id",      "PLAN_ID",    "catalog_code",   True),
    "account": ("account",      "ACCOUNT",    "invoice_no",     True),
    "license": ("license",      "LICENSE",    "cert_version",   True),
    "device":  ("device_id",    "DEVICE_ID",  "version_or_sku", True),
    "vehicle": ("vehicle_id",   "VEHICLE_ID", "asset_tag",      True),
    "phone":   ("phone",        "PHONE",      "support_phone",  True),
    "email":   ("email",        "EMAIL",      "support_email",  True),
    "url":     ("personal_url", "URL",        "org_url",        True),
    "ip":      ("session_ip",   "IP",         "infra_ip",       True),
    "age":     ("age90",        "AGE90",      "age_small",      False),
    "other":   ("other_id",     "OTHER_ID",   "request_id",     True),
}
