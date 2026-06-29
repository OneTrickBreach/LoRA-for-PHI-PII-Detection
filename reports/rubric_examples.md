# Rubric Internalization — 10 Tricky Examples (Day 1 DoD)

Applying the PHI rubric (`rules.md` §2) deterministically to hard cases. Each example states the
gold spans and the rubric clause that decides each call. These exercise the look-alikes and
context-dependent distinctions the model must learn — not "numbers = PHI". `contains_phi` is
derived (`1` iff any span).

Span type set: `NAME, ADDRESS, DATE, SSN, MRN, NPI, PLAN_ID, ACCOUNT, LICENSE, DEVICE_ID,
VEHICLE_ID, PHONE, EMAIL, URL, IP, AGE90, OTHER_ID`.

---

### 1. Patient name + DOB + MRN vs. provider name
> `Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith at Mercy General.`

- **PHI:** `John Reyes` (NAME), `03/14/1981` (DATE — DOB tied to individual), `A55213` (MRN).
- **NOT PHI:** `Dr. Smith` (provider in professional capacity), `Mercy General` (facility/org).
- Clauses: Names (patient→PHI, provider→not), Dates (DOB more specific than year→PHI).

### 2. City + ZIP vs. bare state
> `Member resides in Akron, OH 44312; coverage valid in OH.`

- **PHI:** `Akron, OH 44312` (ADDRESS — city + full ZIP tied to the individual, more specific than state).
- **NOT PHI:** the second `OH` (a standalone state).
- Clause: Geography — anything more specific than state tied to an individual → PHI; standalone state → not.

### 3. Org/public business address (hard negative)
> `Send records to our billing office at 500 Corporate Way, Suite 200, Austin TX.`

- **PHI:** none.
- **NOT PHI:** the full street address — it identifies the **entity/facility**, not an individual.
- Clause: Geography — public business/facility address → NOT PHI. `contains_phi = 0`.

### 4. Age 89 vs. age 90+ (the HIPAA AGE90 rule)
> `Patient A is 89 years old; Patient B is 92 years old.`

- **PHI:** `92` (AGE90 — age > 89).
- **NOT PHI:** `89` (age ≤ 89 alone).
- Clause: Ages — ≤89 not PHI; >89 → PHI, tagged `AGE90`. **Conservative note:** the literal age token
  is tagged; surrounding words are not.

### 5. Date look-alikes — DOB vs. software build date vs. bare year
> `Built on 2026-04-12. Patient born 07/22/1990. Policy effective 2025.`

- **PHI:** `07/22/1990` (DATE — birth date tied to individual).
- **NOT PHI:** `2026-04-12` (software build date, not about a person); `2025` (bare year).
- Clause: Dates — date>year about a person → PHI; build date / bare year → not.

### 6. Order/ticket number look-alike + support line (hard negatives)
> `Order 99812 shipped; for help call our support line 800-555-0000.`

- **PHI:** none.
- **NOT PHI:** `99812` (order number look-alike); `800-555-0000` (generic public support line).
- Clause: Look-alikes — order numbers and generic public support lines are deliberately NOT PHI.
  `contains_phi = 0`. (Note: `800-555-0000` *matches the regex phone pattern* — this is exactly the
  false positive the context-aware model must avoid.)

### 7. Personal phone + personal email vs. org URL
> `Reach the patient at (216) 555-0148 or jane.doe@gmail.com; clinic info at https://mercygeneral.org.`

- **PHI:** `(216) 555-0148` (PHONE — personal), `jane.doe@gmail.com` (EMAIL — personal).
- **NOT PHI:** `https://mercygeneral.org` (org URL).
- Clause: Contact — personal phone/email → PHI; generic org URL → not.

### 8. Session/device IP vs. server IP
> `Patient session originated from 73.118.42.9; routed via gateway 10.0.0.1.`

- **PHI:** `73.118.42.9` (IP — tied to an individual's session/device).
- **NOT PHI:** `10.0.0.1` (infrastructure/server IP).
- Clause: Network — individual-session IP → PHI; server/infra IP → not. **Conservative note:** when
  it is genuinely ambiguous whether an IP is personal or infra, tag it PHI (rules.md §8).

### 9. Model/version string + SKU look-alikes vs. real device serial
> `Running firmware v3.14.2 on unit; device serial SN-7741-AAQ-002; catalog SKU MED-00231.`

- **PHI:** `SN-7741-AAQ-002` (DEVICE_ID — device serial/UDI identifies the individual's device).
- **NOT PHI:** `v3.14.2` (model/version string); `MED-00231` (SKU look-alike).
- Clause: Direct identifiers (device serial → PHI) vs. look-alikes (version strings, SKUs → not).

### 10. SSN + Plan ID buried in free text, provider NPI context
> `Free text: "...spoke w/ member, SSN 402-11-9837, plan ID UHC9921047733. Billing provider NPI 1538291046..."`

- **PHI:** `402-11-9837` (SSN), `UHC9921047733` (PLAN_ID — health-plan beneficiary number).
- **NOT PHI (default):** `1538291046` (NPI here is the **billing provider's** professional identifier,
  not the patient's — provider in professional capacity).
- Clauses: Direct identifiers (SSN, plan/beneficiary ID → PHI); Names/provider rule extends to
  provider professional identifiers. **Conservative note:** an NPI presented as the *patient's* or in
  an ambiguous subject role would be tagged PHI. When unsure who an ID belongs to, treat as PHI.

---

**Summary of the distinctions exercised:** patient vs provider (1, 10), geography granularity
(2, 3), date semantics (1, 5), the AGE90 rule (4), and the four look-alike traps that force context
over pattern-matching — order/ticket numbers (6), public support lines (6), version strings/SKUs (9),
and infra IPs (8). These same hard cases seed the Day-6 **hard test set** that decides the verdict.
