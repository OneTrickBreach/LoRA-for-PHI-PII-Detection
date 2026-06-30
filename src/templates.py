"""Carrier templates for record shapes A/B/C (plan.md §6). Clean prose — NO PHI baked in.

The generator inserts identifiers / look-alikes at the single `{x}` slot afterward (insertion
strategy, rules.md §1.2-1.3), so positive labels are exact by construction. Each template has a
stable `id`; the generator partitions these IDs DISJOINTLY across train/val/test so the same carrier
phrasing never crosses splits (rules.md §3.1). `polarity` pos -> the slot holds PHI; neg -> the slot
holds the matching look-alike, phrased so the look-alike reads naturally (e.g. provider "Dr. {x}").

Record-shape scaffolding (JSON keys, intake field labels, log prefix) is fixed FORMAT, not a carrier
template, and is excluded from template-ID leakage accounting — the leakage check covers the
free-text carrier template IDs + identifier values (documented in README/leakage_check).
"""
from __future__ import annotations

# kind -> {"pos": [sentence, ...], "neg": [sentence, ...]}; each sentence has exactly one "{x}".
_RAW: dict[str, dict[str, list[str]]] = {
    "name": {
        "pos": ["Patient {x} checked in at the front desk.",
                "We notified {x} about the lab results."],
        "neg": ["Dr. {x} signed off on the chart.",
                "Reviewed by Dr. {x} during morning rounds."],
    },
    "address": {
        "pos": ["Home address on file: {x}.",
                "The member currently resides at {x}."],
        "neg": ["Ship lab samples to our facility at {x}.",
                "Our clinic is located at {x}."],
    },
    "dob": {
        "pos": ["DOB {x} per the intake form.",
                "Date of birth recorded as {x}."],
        "neg": ["Firmware build dated {x} was deployed.",
                "Coverage became effective in {x}."],
    },
    "ssn": {
        "pos": ["SSN {x} was verified at registration.",
                "Social security number {x} is on record."],
        "neg": ["Case reference {x} was opened by the team.",
                "Internal ticket {x} was logged for follow-up."],
    },
    "mrn": {
        "pos": ["MRN {x} was assigned to the patient.",
                "The chart was pulled for record {x}."],
        "neg": ["Order {x} shipped from the warehouse today.",
                "Confirmation {x} was emailed to the vendor."],
    },
    "npi": {
        "pos": ["The subject's individual NPI {x} was flagged for review.",
                "Member-linked NPI {x} appears in the file."],
        "neg": ["Billing provider NPI {x} is listed on the claim.",
                "The rendering provider NPI {x} matched the directory."],
    },
    "plan": {
        "pos": ["Plan ID {x} is shown on the policy.",
                "Beneficiary number {x} was confirmed."],
        "neg": ["Catalog code {x} is active in the system.",
                "Product code {x} was added to the order."],
    },
    "account": {
        "pos": ["Patient account number {x} was billed.",
                "Account {x} was updated after the visit."],
        "neg": ["Invoice {x} was generated for the supplier.",
                "Purchase order {x} was approved by finance."],
    },
    "license": {
        "pos": ["Driver's license {x} is on file.",
                "License number {x} was recorded at intake."],
        "neg": ["Software license region {x} was provisioned.",
                "Certificate {x} was issued to the organization."],
    },
    "device": {
        "pos": ["Implanted device serial {x} was logged.",
                "Infusion pump serial {x} was scanned."],
        "neg": ["The unit is running firmware {x}.",
                "Catalog SKU {x} is in stock."],
    },
    "vehicle": {
        "pos": ["Vehicle plate {x} was noted in the incident report.",
                "VIN {x} appears on the transport record."],
        "neg": ["Fleet asset tag {x} was inventoried.",
                "Equipment ID {x} was checked out."],
    },
    "phone": {
        "pos": ["Call the patient back at {x}.",
                "Personal cell {x} is the best contact."],
        "neg": ["For help, call our support line at {x}.",
                "The main office number is {x}."],
    },
    "email": {
        "pos": ["Send results to {x} when ready.",
                "Patient email {x} is on file."],
        "neg": ["Contact our team at {x} with questions.",
                "Route general inquiries to {x}."],
    },
    "url": {
        "pos": ["The patient's personal portal is {x}.",
                "Their personal page is {x}."],
        "neg": ["More information is on our site {x}.",
                "See the organization page at {x}."],
    },
    "ip": {
        "pos": ["The user's session originated from {x}.",
                "Login IP {x} was tied to the account."],
        "neg": ["Gateway {x} was briefly unreachable.",
                "Server {x} was restarted overnight."],
    },
    "age": {
        "pos": ["The patient is {x} years old.",
                "Age {x}, per the chart."],
        "neg": ["The patient is {x} years old.",
                "Age {x}, per the chart."],
    },
    "other": {
        "pos": ["Unique identifier {x} is linked to the subject.",
                "Patient-specific ID {x} was generated."],
        "neg": ["Request {x} was received by the API.",
                "Trace {x} was recorded in the log."],
    },
}


def _build() -> list[dict]:
    out: list[dict] = []
    for kind, d in _RAW.items():
        for i, text in enumerate(d["pos"]):
            out.append({"id": f"{kind}_pos_{i}", "kind": kind, "polarity": "pos", "text": text})
        for i, text in enumerate(d["neg"]):
            out.append({"id": f"{kind}_neg_{i}", "kind": kind, "polarity": "neg", "text": text})
    return out


CARRIER_TEMPLATES: list[dict] = _build()
