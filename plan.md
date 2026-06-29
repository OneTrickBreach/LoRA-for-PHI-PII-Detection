# PLAN.md — PHI/PII Detection via LoRA (2-Week POC)

> **For the executing agent:** This is the build plan. Read `rules.md` first and keep it open — it
> contains the hard constraints (PHI rubric, data integrity, modeling, evaluation) that apply to
> *every* task below. This file tells you **what to build and in what order**; `rules.md` tells you
> **what you must never get wrong**. Proceed end-to-end using the defaults here. Do not wait for
> sign-off. If something seems missing, it is almost certainly answered in this file or `rules.md`.

---

## 1. Mission

Build a **proof-of-value POC** that detects PHI/PII (Protected Health Information / Personally
Identifiable Information) in text records, **at the span level** (exact character ranges + category),
so detected spans can be **redacted**. Prove that a cheaply fine-tuned encoder (LoRA on
DeBERTa-v3) **beats Microsoft Presidio** at matched recall.

This is a POC. It produces a reusable **model + data generator + evaluation harness**. It is **not**
wired into any live service in these two weeks.

## 2. Definition of success (the bar)

A run is successful when, on the **hard test set**:

- **Span-level recall ≥ 0.97** (a false negative is a breach — recall leads).
- **Span-level precision ≥ 0.85** at that recall.
- **LoRA beats Presidio's precision at matched recall.** This is what proves ML earns its place.
- **Latency:** target **< 50 ms/record on GPU**. Measure and report regardless. Inability to hit it
  is a *finding* (recommend pre-filter or async scan), **not** a failure.

Scope locks: **English-only** for v1. **100% synthetic data** — no real records ever.

## 3. Tech stack & environment

- **Language:** Python 3.10+
- **Core libs:** `transformers`, `peft`, `datasets`, `accelerate`, `torch`, `seqeval`,
  `scikit-learn`, `faker`, `presidio-analyzer`, `presidio-anonymizer`, `numpy`, `pandas`
- **Base model (primary):** `microsoft/deberta-v3-base` for token classification.
- **Base model (alt, stretch only):** a small decoder instruct model (1–3B) — used **only** for the
  stretch comparison, never as the baseline.
- **Hardware default:** single GPU with ≥12 GB (T4 / 3060 / Colab). If no GPU: DeBERTa-base + LoRA
  still trains on CPU slowly — use a smaller data subset. If more GPU is available, scale batch size.
- **Reproducibility:** set and log a global seed everywhere (data gen, splits, training). Pin
  versions in `requirements.txt`.

## 4. Repository structure (create this)

```
phi-lora-poc/
├── README.md                  # how a stranger reproduces everything
├── requirements.txt
├── config.yaml                # seeds, paths, hyperparams, composition targets
├── data/
│   ├── raw/                   # generated JSONL (train/val/test, hard test)
│   └── pools/                 # per-split entity pools + template IDs
├── src/
│   ├── generate.py            # insertion-based synthetic data generator
│   ├── id_generators.py       # custom MRN/NPI/PLAN_ID/DEVICE_ID/etc. generators
│   ├── templates.py           # carrier templates for record shapes A/B/C
│   ├── leakage_check.py       # asserts zero entity/template overlap across splits
│   ├── align.py               # char-span -> BIO token labels via offset_mapping
│   ├── baselines/
│   │   ├── regex_baseline.py
│   │   ├── presidio_baseline.py
│   │   └── fewshot_baseline.py
│   ├── train_lora.py          # LoRA fine-tune on DeBERTa-v3
│   ├── predict.py             # unified predict interface for every system
│   └── evaluate.py            # the eval harness -> comparison table
├── scripts/
│   └── run_all.sh             # one command: train + score all systems
└── reports/
    ├── comparison_table.md
    ├── error_analysis.md
    └── memo.md                # the one-page deliverable
```

## 5. Label schema (the category set)

Tag every PHI span with one of these types so per-category recall is computable:

```
NAME, ADDRESS, DATE, SSN, MRN, NPI, PLAN_ID, ACCOUNT, LICENSE,
DEVICE_ID, VEHICLE_ID, PHONE, EMAIL, URL, IP, AGE90, OTHER_ID
```

JSONL record format (one record per line):

```json
{"text":"...full record as a string...",
 "spans":[{"start":41,"end":52,"type":"SSN"}],
 "contains_phi":1,"record_type":"intake_form"}
```

`contains_phi` is **derived**: `1` iff `spans` is non-empty. Keep it for the binary view.

> The full rules for *what counts as PHI* live in `rules.md` §2. Do not improvise them.

## 6. Record shapes (generate across all three)

**A. Generic API request with a free-text field** (free text is the risk surface)
```json
{"request_id":"req_8f2","source":"intake",
 "payload":{"text":"<free text, may contain PHI>","notes":["..."]}}
```

**B. Structured intake form** (named fields, some PHI by design)
```json
{"form_type":"intake",
 "fields":{"name":"<PHI?>","dob":"<PHI?>","mrn":"<PHI?>","state":"<not PHI>",
 "complaint":"<free text, may contain PHI>","provider":"<not PHI>"}}
```

**C. Log / record line** (semi-structured text)
```
2026-04-12T09:14Z INFO ingest source=intake msg="<free text, may contain PHI>"
```

PHI must appear both in obvious fields **and** buried in free text. Negatives include records with
only look-alikes and benign org/provider data.

## 7. Synthetic data generation (insertion strategy)

**The core trick — never hand-label positives:**

1. Use **Faker** to mint identifiers and **insert them at known character offsets** into clean
   carrier templates. Because *you* place them, the span label (offset + type) is free and exact.
2. Use an **LLM only for carrier/clean text** (surrounding prose, and negative records). **Never let
   the LLM emit PHI you then have to find.** Generate clean text first, then insert known identifiers.
3. For negatives, generate text and insert **only look-alikes** (also at known offsets), so "no PHI
   span" is guaranteed by construction.

**Custom ID generators to write** (Faker covers names/addresses/SSN/email/phone; it does NOT cover
these):
- **MRN:** 6–10 digit, sometimes alpha-prefixed (`MRN-` / `A` + digits).
- **NPI:** 10 digits (v1 simplification — note in the memo that real NPIs are Luhn-checked).
- **Plan/beneficiary ID:** alpha + 9–11 digits, varied formats.
- **Device serial / UDI:** vendor-prefixed alphanumerics.
- **Account / certificate / license numbers:** mixed alphanumerics with separators.

**Composition targets:**
- ~10k–20k examples total, ~50/50 positive/negative.
- Per PHI category: aim ≥ 300 positives each for common types.
- A dedicated **hard test set** (~1–2k) weighted toward look-alikes (hard negatives) and PHI in
  unusual formats/positions/free-text (hard positives). **This set decides the verdict.**

**Leakage discipline (mandatory — see `rules.md` §3):** split at the **entity & template level**, not
the row level. Generate distinct entity pools per split. Add an automated check asserting zero overlap
of identifiers and template IDs across splits.

## 8. Baselines (run all three before trusting any LoRA number)

**(1) Regex baseline** — implement exactly these, report recall/precision:
```
SSN:    \b\d{3}-\d{2}-\d{4}\b
Email:  \b[\w.+-]+@[\w-]+\.[\w.-]+\b
Phone:  \b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b
IP:     \b(?:\d{1,3}\.){3}\d{1,3}\b
Dates:  \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b
MRN:    \bMRN[-:\s]?\w{5,10}\b
```
(Expected to miss names, free-text addresses, context-dependent cases — that gap is the point.)

**(2) Presidio** — `presidio-analyzer` out of the box, default recognizers. **This is the bar to
beat.** Map its entity labels to the category set in §5 for per-category comparison.

**(3) Base-model few-shot** (decoder, zero training) — use this exact prompt:
```
You are a PHI/PII detector. Return JSON: a list of spans, each {text, type}.
Types: NAME, ADDRESS, DATE, SSN, MRN, NPI, PLAN_ID, ACCOUNT, LICENSE,
DEVICE_ID, VEHICLE_ID, PHONE, EMAIL, URL, IP, AGE90, OTHER_ID.
Patient/member identifiers are PHI. Provider names, org addresses, bare years,
order/ticket numbers are NOT PHI. Return [] if none.

Example 1:
Input: "Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith."
Output: [{"text":"John Reyes","type":"NAME"},{"text":"03/14/1981","type":"DATE"},{"text":"A55213","type":"MRN"}]

Example 2:
Input: "Order 99812 shipped to our Austin office, support 800-555-0000."
Output: []

Input: "<record>"
Output:
```

## 9. LoRA setup (exact config)

Token classification with LoRA on DeBERTa-v3. **Two gotchas are pre-solved — do not skip them.**

**Gotcha A — the classifier head must train.** With PEFT on a classification model the new head is
frozen unless listed:
```python
from peft import LoraConfig, TaskType, get_peft_model
cfg = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",      # robust: don't hand-name DeBERTa's proj layers
    modules_to_save=["classifier"],   # <-- without this the head never learns
)
```
If accuracy is stuck at chance, you almost certainly forgot `modules_to_save`.

**Gotcha B — char spans → token (BIO) labels.** Tokenizers split words; align with `offset_mapping`:
```python
enc = tok(text, return_offsets_mapping=True, truncation=True)
# for each token (char_start, char_end): label B-/I-<TYPE> if it overlaps a PHI span,
# else O; set label = -100 for special tokens so loss ignores them.
```
**Test alignment on 5 hand-checked examples before training on 15k.** A one-character offset bug
corrupts the entire label set.

**Hyperparameters** (start, then sweep): `r ∈ {8,16,32}`, `lora_alpha ≈ r or 2r`, LR `1e-4–2e-4`,
2–3 epochs, watch val for overfit.

## 10. Evaluation harness (metrics, defined precisely)

One command scores any system (regex / Presidio / few-shot / LoRA) on the test set and emits a
comparison table.

- **Span-level recall / precision / F1** — a predicted span is correct on **overlap** with a gold
  span of the same type. Report both strict-exact and overlap; **lead with overlap** (redaction only
  needs to cover the span).
- **Binary recall** (derived) — did the system flag *any* PHI in a record that contained PHI. The
  go/no-go signal.
- **Per-category recall** — surfaces "catches SSNs, misses MRNs."
- **Recall-first thresholding** — pick the operating point that hits **recall ≥ 0.97**, then report
  precision there. **Compare every system at matched recall.**
- **Hard-test confusion** — false positives on look-alikes, false negatives on hard positives,
  broken out.
- **Latency** — ms/record on target hardware, **measured, not estimated.**

**Output table columns:** `system → span-F1 → binary-recall → precision@recall → per-category recall → latency`

## 11. Execution phases (10 working days)

### Week 1 — rubric, data, baselines, working pipeline

**Day 1 — Setup + rubric internalization**
- Install stack; confirm GPU (or accept CPU subset). Read the LoRA paper core + `rules.md` §2 rubric;
  skim Presidio.
- **DoD:** `deberta-v3-base` loads; you can apply the rubric to 10 tricky examples correctly.

**Day 2 — Synthetic data v1 + alignment test**
- Build the insertion-based generator (Faker + custom ID generators). Produce ~2k labeled rows across
  shapes A/B/C. Verify char-span→token alignment on 5 examples.
- **DoD:** Labeled JSONL loads; BIO alignment verified by hand.

**Day 3 — All three baselines**
- Implement regex, run Presidio, run few-shot. Score on v1.
- **DoD:** Baseline table exists; you can state Presidio's recall/precision.

**Day 4 — First LoRA run + eval harness**
- Fine-tune with §9 config. Build the §10 harness. LoRA vs all baselines.
- **DoD:** One command → full comparison table. (Stuck at chance? Check `modules_to_save`.)

**Day 5 — Harden generator + checkpoint**
- Add hard negatives/positives; enforce entity/template-level splits with the automated overlap
  check; write README. Mid-project self-review against this spec + `rules.md`.
- **DoD:** Data regenerates reproducibly; overlap check passes; README lets a stranger reproduce week 1.

### Week 2 — scale, rigor, recommendation

**Day 6 — Data v2 (scale + hard test set)**
- Scale to 10k–20k; build the dedicated hard test set; confirm per-category coverage and clean splits.
- **DoD:** Final train/val/test + documented recipe.

**Day 7 — Retrain + error analysis**
- Retrain on v2; full eval; break down false negatives by category — which does LoRA catch that
  regex/Presidio miss, and vice versa.
- **DoD:** Written error analysis: where ML wins, where rules already suffice.

**Day 8 — Sweep + recall-first thresholding**
- Sweep `r`/`alpha`/target modules; set threshold to hit recall ≥ 0.97; log
  time/memory/adapter-size/latency.
- **DoD:** `setting → recall/precision/latency/cost` table.

**Day 9 — Final eval + recommendation**
- Clean run of best config on the hard test set. LoRA vs regex vs Presidio vs few-shot at matched
  recall. Verdict vs the success bar. Evaluate a regex/Presidio **pre-filter + LoRA hybrid**.
- **DoD:** Recommended config + pure-rules / pure-LoRA / hybrid verdict, with the latency finding.

**Day 10 — Memo + handoff**
- Write the memo; clean code; prepare a 20-min walkthrough; caveat the synthetic-real gap and give a
  real-data validation plan.
- **DoD:** All §12 deliverables done.

## 12. Deliverables

1. **Synthetic data generator** — insertion-based, custom ID generators, leakage-safe splits +
   overlap check, reproducible recipe.
2. **LoRA pipeline + eval harness** — one command: train, then score LoRA vs regex vs Presidio vs
   few-shot; span + binary + per-category; recall-first; with latency.
3. **One-page memo** answering:
   - Does LoRA beat Presidio at matched recall, by how much?
   - Best config + cost + latency?
   - Pure-rules / LoRA / hybrid recommendation?
   - The synthetic-real caveat + a concrete real-data validation plan.

## 13. Stretch goals (only after the core is done)

- Redaction output end-to-end (replace spans), not just detection.
- QLoRA vs plain LoRA (memory/quality trade-off).
- Decoder (alt model) vs encoder head-to-head.
- Hybrid prototype: regex/Presidio pre-filter → LoRA on ambiguous cases only, measured vs either alone.

## 14. Grading rubric (optimize for this)

| Area | Weight | Strong looks like |
|---|---|---|
| Synthetic data quality | 25% | Hard cases, shape-matched, leakage-safe (overlap check passes) |
| Evaluation rigor | 30% | Recall-first, per-category, beats Presidio at matched recall, latency measured |
| Result + trade-offs | 25% | Sweep + thresholding + hybrid analysis + cost/latency |
| Communication | 20% | Memo with synthetic-real caveat; reusable code |
