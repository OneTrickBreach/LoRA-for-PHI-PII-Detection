"""Carrier templates for record shapes A/B/C (plan.md §6). Clean prose — NO PHI baked in.

The generator inserts identifiers / look-alikes at the single `{x}` slot afterward (insertion
strategy, rules.md §1.2-1.3), so positive labels are exact by construction. Each template has a
stable `id`; the generator partitions these IDs DISJOINTLY across splits (rules.md §3.1).

Day-6 change (fix Week-1 thin coverage): the main bank now has **4 pos + 4 neg templates per kind**,
partitioned PER KIND so **every split gets ≥1 pos and ≥1 neg template for every category** — no
category is absent from val/test anymore. A SEPARATE hard bank (`HARD_*`) supplies the dedicated hard
test set with unusual-position/terse positives and look-alike-dense negatives; its IDs never appear in
train/val/test.

Record-shape scaffolding (JSON keys, intake field labels, log prefix) is fixed FORMAT, not a carrier
template, and is excluded from template-ID leakage accounting.
"""
from __future__ import annotations

# kind -> {"pos": [4 sentences], "neg": [4 sentences]}; each sentence has exactly one "{x}".
_RAW: dict[str, dict[str, list[str]]] = {
    "name": {
        "pos": ["Patient {x} checked in at the front desk.",
                "We notified {x} about the lab results.",
                "The care team met with {x} this morning.",
                "Discharge summary was sent to {x}."],
        "neg": ["Dr. {x} signed off on the chart.",
                "Reviewed by Dr. {x} during morning rounds.",
                "The consult was handled by Dr. {x}.",
                "Attending physician Dr. {x} approved the plan."],
    },
    "address": {
        "pos": ["Home address on file: {x}.",
                "The member currently resides at {x}.",
                "Mail the results to the patient at {x}.",
                "Patient relocated to {x} last month."],
        "neg": ["Ship lab samples to our facility at {x}.",
                "Our clinic is located at {x}.",
                "The billing office is at {x}.",
                "Visit our branch at {x} for support."],
    },
    "dob": {
        "pos": ["DOB {x} per the intake form.",
                "Date of birth recorded as {x}.",
                "The patient was born on {x}.",
                "Birth date {x} was confirmed at registration."],
        "neg": ["Firmware build dated {x} was deployed.",
                "Coverage became effective in {x}.",
                "The policy version is dated {x}.",
                "System snapshot taken on {x}."],
    },
    "ssn": {
        "pos": ["SSN {x} was verified at registration.",
                "Social security number {x} is on record.",
                "The patient's SSN is {x}.",
                "Taxpayer SSN {x} was collected for billing."],
        "neg": ["Case reference {x} was opened by the team.",
                "Internal ticket {x} was logged for follow-up.",
                "Claim reference {x} is pending review.",
                "Dispute number {x} was filed."],
    },
    "mrn": {
        "pos": ["MRN {x} was assigned to the patient.",
                "The chart was pulled for record {x}.",
                "Medical record number {x} is active.",
                "Encounter linked to MRN {x}."],
        "neg": ["Order {x} shipped from the warehouse today.",
                "Confirmation {x} was emailed to the vendor.",
                "Tracking number {x} is in transit.",
                "Batch {x} was processed overnight."],
    },
    "npi": {
        "pos": ["The subject's individual NPI {x} was flagged for review.",
                "Member-linked NPI {x} appears in the file.",
                "The patient is also a provider; their NPI {x} is noted.",
                "Self-referred subject NPI {x} recorded."],
        "neg": ["Billing provider NPI {x} is listed on the claim.",
                "The rendering provider NPI {x} matched the directory.",
                "Referring physician NPI {x} was verified.",
                "Facility NPI {x} is on the remittance."],
    },
    "plan": {
        "pos": ["Plan ID {x} is shown on the policy.",
                "Beneficiary number {x} was confirmed.",
                "The member's health-plan ID is {x}.",
                "Enrollment shows plan member ID {x}."],
        "neg": ["Catalog code {x} is active in the system.",
                "Product code {x} was added to the order.",
                "Pricing tier {x} applies to this SKU.",
                "Bundle code {x} was selected."],
    },
    "account": {
        "pos": ["Patient account number {x} was billed.",
                "Account {x} was updated after the visit.",
                "The patient's account on file is {x}.",
                "Payment posted to account {x}."],
        "neg": ["Invoice {x} was generated for the supplier.",
                "Purchase order {x} was approved by finance.",
                "Vendor invoice {x} is overdue.",
                "Receipt {x} was archived."],
    },
    "license": {
        "pos": ["Driver's license {x} is on file.",
                "License number {x} was recorded at intake.",
                "The patient's driver license is {x}.",
                "ID document license {x} was scanned."],
        "neg": ["Software license region {x} was provisioned.",
                "Certificate {x} was issued to the organization.",
                "The product license key is {x}.",
                "Compliance certificate {x} was renewed."],
    },
    "device": {
        "pos": ["Implanted device serial {x} was logged.",
                "Infusion pump serial {x} was scanned.",
                "The patient's monitor serial is {x}.",
                "Pacemaker serial {x} recorded in the note."],
        "neg": ["The unit is running firmware {x}.",
                "Catalog SKU {x} is in stock.",
                "Model revision {x} shipped this quarter.",
                "Part number {x} was reordered."],
    },
    "vehicle": {
        "pos": ["Vehicle plate {x} was noted in the incident report.",
                "VIN {x} appears on the transport record.",
                "The patient's car plate is {x}.",
                "Ambulance intake listed plate {x} for the patient vehicle."],
        "neg": ["Fleet asset tag {x} was inventoried.",
                "Equipment ID {x} was checked out.",
                "Company vehicle unit {x} is in the depot.",
                "Rental stock number {x} was returned."],
    },
    "phone": {
        "pos": ["Call the patient back at {x}.",
                "Personal cell {x} is the best contact.",
                "The patient can be reached at {x}.",
                "Left a voicemail for the patient at {x}."],
        "neg": ["For help, call our support line at {x}.",
                "The main office number is {x}.",
                "Reach our hotline at {x} during business hours.",
                "Dial our appointment desk at {x}."],
    },
    "email": {
        "pos": ["Send results to {x} when ready.",
                "Patient email {x} is on file.",
                "The patient prefers contact at {x}.",
                "Portal invite was sent to {x}."],
        "neg": ["Contact our team at {x} with questions.",
                "Route general inquiries to {x}.",
                "Our support inbox is {x}.",
                "Email billing questions to {x}."],
    },
    "url": {
        "pos": ["The patient's personal portal is {x}.",
                "Their personal page is {x}.",
                "The patient shared their blog {x}.",
                "Personal health tracker at {x} was referenced."],
        "neg": ["More information is on our site {x}.",
                "See the organization page at {x}.",
                "Our clinic website is {x}.",
                "Documentation is hosted at {x}."],
    },
    "ip": {
        "pos": ["The user's session originated from {x}.",
                "Login IP {x} was tied to the account.",
                "The patient portal was accessed from {x}.",
                "Device IP {x} was recorded for the session."],
        "neg": ["Gateway {x} was briefly unreachable.",
                "Server {x} was restarted overnight.",
                "The load balancer at {x} is healthy.",
                "Internal node {x} passed its health check."],
    },
    "age": {
        "pos": ["The patient is {x} years old.",
                "Age {x}, per the chart.",
                "At {x} years of age, the patient was admitted.",
                "The record lists the patient's age as {x}."],
        "neg": ["The patient is {x} years old.",
                "Age {x}, per the chart.",
                "At {x} years of age, the patient was admitted.",
                "The record lists the patient's age as {x}."],
    },
    "other": {
        "pos": ["Unique identifier {x} is linked to the subject.",
                "Patient-specific ID {x} was generated.",
                "The subject's unique token is {x}.",
                "Case-linked identifier {x} was assigned to the patient."],
        "neg": ["Request {x} was received by the API.",
                "Trace {x} was recorded in the log.",
                "Correlation ID {x} was attached.",
                "Session token {x} expired."],
    },
}


def _build(raw: dict) -> list[dict]:
    out: list[dict] = []
    for kind, d in raw.items():
        for i, text in enumerate(d["pos"]):
            out.append({"id": f"{kind}_pos_{i}", "kind": kind, "polarity": "pos", "text": text})
        for i, text in enumerate(d["neg"]):
            out.append({"id": f"{kind}_neg_{i}", "kind": kind, "polarity": "neg", "text": text})
    return out


CARRIER_TEMPLATES: list[dict] = _build(_RAW)


# ---- Hard bank (dedicated hard test set only; IDs never appear in train/val/test) --------------
# Hard POSITIVES: PHI in unusual positions / terse context / minimal labeling cue -> harder to catch.
# Hard NEGATIVES: look-alikes that trip naive detectors (SSN-shaped tickets, infra IPs, build dates,
# support lines, provider names, org addresses, version strings) in terse contexts.
_HARD_POS: dict[str, list[str]] = {
    "name": ["{x} called the desk twice today.", "Left a message — {x} — please follow up."],
    "address": ["{x} is where to send the summary.", "Reship to {x}, patient's home."],
    "dob": ["{x} — verify against the chart.", "b. {x}; flag if mismatch."],
    "ssn": ["on file: {x} (do not share).", "{x} — collected at desk."],
    "mrn": ["chart {x}, room 4.", "{x} pulled for the encounter."],
    "npi": ["subject npi {x} — self-referral.", "flag {x}, member-linked."],
    "plan": ["member {x}, effective now.", "{x} on the card, verify."],
    "account": ["acct {x}, balance due.", "post to {x}."],
    "license": ["dl {x} scanned.", "{x} — id on file."],
    "device": ["serial {x} implant.", "{x} scanned bedside."],
    "vehicle": ["plate {x}, patient car.", "{x} on the transport slip."],
    "phone": ["reach out — {x} — anytime.", "cell: {x}."],
    "email": ["ping {x} when ready.", "{x} is the patient."],
    "url": ["their page: {x}.", "{x} — personal tracker."],
    "ip": ["{x} hit the portal pre-auth.", "from {x}, patient session."],
    "age": ["{x} yo, admitted.", "at {x}, high risk."],
    "other": ["token {x} for the subject.", "{x} — patient-linked."],
}
_HARD_NEG: dict[str, list[str]] = {
    "ssn": ["ref {x} logged.", "dispute {x} pending."],
    "phone": ["hotline {x}.", "desk line {x}."],
    "email": ["inbox {x}.", "helpdesk {x}."],
    "ip": ["node {x} rebooted.", "gateway {x} down."],
    "dob": ["shipped {x}.", "build {x} live."],
    "mrn": ["pkg {x} delivered.", "order {x} out."],
    "name": ["Dr. {x} on call.", "signed: Dr. {x}."],
    "address": ["depot at {x}.", "office {x}."],
    "device": ["fw {x}.", "sku {x} restocked."],
    "plan": ["code {x} active.", "tier {x}."],
    "account": ["inv {x} sent.", "po {x} ok."],
    "license": ["cert {x} valid.", "key {x} region."],
    "npi": ["provider {x} billed.", "rendering npi {x}."],
    "url": ["site {x}.", "docs {x}."],
    "vehicle": ["asset {x} tagged.", "fleet {x}."],
    "other": ["req {x}.", "trace {x}."],
    "age": ["{x} yo, routine.", "age {x}, stable."],
}


def _build_hard(raw: dict, polarity: str, prefix: str) -> list[dict]:
    out: list[dict] = []
    for kind, texts in raw.items():
        for i, text in enumerate(texts):
            out.append({"id": f"{prefix}_{kind}_{i}", "kind": kind,
                        "polarity": polarity, "text": text})
    return out


HARD_POS_TEMPLATES: list[dict] = _build_hard(_HARD_POS, "pos", "hpos")
HARD_NEG_TEMPLATES: list[dict] = _build_hard(_HARD_NEG, "neg", "hneg")
