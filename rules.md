# RULES.md — Hard Constraints for the PHI/PII LoRA Project

> **For the executing agent:** These rules apply to *every* task in `plan.md`. They override
> convenience, speed, and any local optimization. When a rule and an instinct conflict, the rule wins.
> Re-read §1 and §2 before any data-generation or labeling work. If a situation is genuinely not
> covered here, make the most conservative choice (treat ambiguous identifiers as PHI), note the
> decision, and continue — do not block.

---

## 1. Non-negotiables (read first)

1. **Synthetic data only.** No real records, no real personal data, no scraped PII — ever, at any
   stage. The entire project exists because real data is off-limits.
2. **Never hand-label positives.** PHI is inserted at known offsets, so every positive label is exact
   by construction. If you find yourself manually annotating PHI in generated text, stop — you've
   broken the generation strategy.
3. **The LLM never emits PHI you then have to find.** LLMs generate *clean carrier text and negatives
   only*. Identifiers are inserted afterward by code at known positions.
4. **Recall leads.** A missed PHI span is a breach — the costly error. Optimize for recall, then
   precision. Never tune for F1 at recall's expense.
5. **Presidio is the bar.** "Beating a regex" proves nothing. The success claim is: LoRA beats
   Presidio's precision at matched recall.
6. **Measure, don't estimate.** Latency, memory, adapter size, and all metrics are measured on the
   target hardware and reported as numbers. "Should be fast" is not a result.
7. **Reproducibility is mandatory.** Fixed seeds, pinned versions, one-command runs, a README a
   stranger can follow. If it can't be regenerated, it didn't happen.

## 2. PHI annotation rubric — THIS IS THE GROUND TRUTH

You generate the labels, so consistency here determines whether every downstream number means
anything. Apply these rules **deterministically**. Insertion makes positive labels automatic; this
rubric governs hard cases, look-alikes, and any hand-checking.

**Label as PHI (positive span)** when an identifier could — alone or combined — identify an individual
data subject (patient/member). Anchor on the **HIPAA 18 identifiers.**

**Decision rules:**

- **Names** — Patient/member/subject names → **PHI**. Provider/clinician/staff names in their
  professional capacity → **NOT PHI** (default). Org/facility names → **NOT PHI**.
- **Geography** — Anything more specific than state tied to an individual (street, city, county, full
  ZIP) → **PHI**. A standalone state → **NOT PHI**. A public business/facility address → **NOT PHI**
  (identifies the entity, not the individual).
- **Dates** — Any date more specific than a year that relates to an individual (DOB, admission,
  discharge, death, appointment) → **PHI**. A bare year → **NOT PHI**. Dates unrelated to a person
  (software build date, policy effective year) → **NOT PHI**.
- **Ages** — Age ≤ 89 alone → **NOT PHI**. Age > 89 (or "90+") → **PHI** (HIPAA rule). Tag as `AGE90`.
- **Direct identifiers** — SSN, MRN, health-plan beneficiary number, NPI, account number,
  certificate/license number, device serial/UDI, vehicle ID/plate, biometric IDs → **PHI**.
- **Contact** — Personal phone, fax, email, personal URL → **PHI**. A generic public support line or
  org URL → **NOT PHI**.
- **Network** — IP tied to an individual's session/device → **PHI**. A server/infrastructure IP →
  **NOT PHI**.
- **Look-alikes (hard negatives — deliberately NOT PHI)** — order/ticket/invoice numbers, model
  version strings, generic phone-shaped support lines, public addresses, bare years, SKUs, request
  IDs. **These must appear in the data** so the model learns *context*, not "numbers = PHI."

**Span typing:** every PHI span gets a category from:
`NAME, ADDRESS, DATE, SSN, MRN, NPI, PLAN_ID, ACCOUNT, LICENSE, DEVICE_ID, VEHICLE_ID, PHONE, EMAIL,
URL, IP, AGE90, OTHER_ID`.

## 3. Data integrity rules

1. **Leakage discipline.** Split train/val/test at the **entity & template level**, not the row
   level. The same fake name, MRN, or carrier template must **never** appear in two splits.
2. **Distinct entity pools per split.** Generate separate identifier pools and template-ID sets for
   train, val, and test. Do not share them.
3. **Automated overlap check is required.** Add a check that asserts **zero overlap** of identifiers
   and template IDs across splits. It must pass before any number is trusted. If it fails, regenerate
   — do not "fix" by deleting rows.
4. **Generate across all three record shapes** (A/B/C in `plan.md` §6), with PHI both in obvious
   fields and buried in free text.
5. **Hard test set decides the verdict.** It is weighted toward hard negatives (look-alikes) and hard
   positives (PHI in unusual formats/positions/free-text). Never tune on it; treat it as held-out.
6. **Negatives are guaranteed by construction.** A "no PHI" record contains only inserted look-alikes
   — never trust an LLM's word that text is clean.
7. **`contains_phi` is derived**, never set by hand: `1` iff `spans` is non-empty.

## 4. Modeling rules

1. **`modules_to_save=["classifier"]` is mandatory.** Without it the head never learns and accuracy
   stays at chance. If training looks broken, check this first.
2. **Use `target_modules="all-linear"`** rather than hand-naming DeBERTa's projection layers. If you
   do name them, print `model.named_modules()` first to confirm.
3. **Verify char-span → BIO token alignment on 5 hand-checked examples before training on the full
   set.** A one-character offset bug silently corrupts the whole label set. Use `offset_mapping`; set
   label `-100` on special tokens so loss ignores them.
4. **DeBERTa-v3 encoder is the baseline model.** The decoder is a stretch experiment only — it never
   becomes the primary.
5. **Sweep, don't guess.** Hyperparameters get swept (`r ∈ {8,16,32}`, `alpha ≈ r or 2r`, LR
   `1e-4–2e-4`, 2–3 epochs); report the chosen config with evidence.
6. **Watch val for overfit.** Stop or adjust when val recall/precision diverges from train.

## 5. Evaluation rules

1. **Run all three baselines on the same test set before trusting any LoRA number.** Regex, Presidio,
   few-shot.
2. **Compare every system at matched recall.** Pick the operating point at recall ≥ 0.97, then read
   off precision. Comparing at different recalls is invalid.
3. **Span correctness = overlap with a gold span of the same type.** Report both strict-exact and
   overlap; lead with overlap (redaction only needs to cover the span).
4. **Report per-category recall**, not just an aggregate — it surfaces blind spots (e.g. catches
   SSNs, misses MRNs).
5. **Break out hard-test confusion:** false positives on look-alikes, false negatives on hard
   positives.
6. **Latency is a real requirement, measured on target hardware.** Missing the < 50 ms/GPU target is
   a finding (recommend pre-filter / async scan), not a failure — report it either way.
7. **One command runs the full comparison.** No manual stitching of numbers.

## 6. Code & repo conventions

1. **One-command runs** for data generation, training, and full evaluation.
2. **Fix and log seeds** in every stochastic step.
3. **Pin dependencies** in `requirements.txt`; record library versions used for the reported numbers.
4. **README first-class:** a stranger must be able to reproduce each week from it.
5. **Keep config out of code:** seeds, paths, composition targets, hyperparameters live in
   `config.yaml`.
6. **Findings live in files**, never only in chat — comparison table, error analysis, and memo are
   committed artifacts.
7. **Environment is `uv`-managed, always.** Use `uv` for every environment and package operation —
   never plain `pip`/`python` against system interpreters. The project venv lives at `.venv`
   (Python 3.12, created with `uv venv`). Install with `uv pip install`; run any script through the
   venv (`source .venv/Scripts/activate` then `python ...`, or `uv run python ...`). A stranger
   reproducing this repo activates the uv venv before doing any work.

## 7. Anti-patterns — pre-solved pitfalls, do not reintroduce

- **No real baseline** → Presidio is the bar.
- **Template/entity leakage** → entity-level splits + automated overlap check.
- **Synthetic ≠ real** → memo caveat + concrete real-data validation plan.
- **Optimizing F1 not recall** → recall-first thresholding; a missed span is the costly error.
- **Flag-everything degeneracy** → hard negatives force context.
- **Classifier head won't learn** → `modules_to_save=["classifier"]`.
- **Span/token misalignment** → `offset_mapping` + the 5-example check.
- **Faker lacks domain IDs** → custom generators for MRN/NPI/PLAN_ID/DEVICE_ID/ACCOUNT/LICENSE.
- **Unscalable labeling** → insertion-only PHI; LLM for carrier text only.
- **Latency ignored** → measured against the target; inability to meet it is a finding.
- **Strawman regex** → use the exact patterns specified, not a deliberately weak set.

## 8. When to flag vs. proceed

- **Proceed (use the documented default)** for anything covered in `plan.md` or this file. Never wait
  for sign-off on a pre-made decision.
- **Make the conservative choice and note it** when a labeling edge case isn't explicitly covered:
  treat an ambiguous identifier as PHI rather than risk a miss.
- **Surface as a finding (don't silently absorb)** when a hard requirement can't be met — e.g.
  latency target unreachable, or LoRA fails to beat Presidio. Report it honestly with numbers; that
  is a legitimate, valuable outcome, not a failure to hide.
- **Stop and regenerate** if the overlap check fails — never patch leaky data by hand.
