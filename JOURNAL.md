# Project Journal — PHI/PII Detection via LoRA (2-Week POC)

**Owner:** OneTrickBreach · **Audience:** daily manager update · **Started:** 2026-06-29

**Goal in one line:** Prove that a cheaply fine-tuned encoder (LoRA on DeBERTa-v3-base) detects
PHI/PII at the span level and **beats Microsoft Presidio at matched recall**, on 100% synthetic data.

**Success bar (judged on a held-out hard test set):** span-level recall ≥ 0.97, precision ≥ 0.85 at
that recall, LoRA precision > Presidio precision at matched recall, latency target < 50 ms/record on
GPU (missing it is a documented *finding*, not a failure).

**Plan:** 10 working days. Week 1 = rubric + data + baselines + working pipeline. Week 2 = scale +
rigor + recommendation. Full plan in `plan.md`; hard constraints in `rules.md`.

**Legend:** ✅ done · 🟡 in progress · ⬜ not started

| Day | Focus | Status |
|----|-------|--------|
| 1 | Setup + rubric internalization | ✅ |
| 2 | Synthetic data v1 + alignment test | ✅ |
| 3 | All three baselines (regex, Presidio, few-shot) | ⬜ |
| 4 | First LoRA run + eval harness | ⬜ |
| 5 | Harden generator + leakage check + README | ⬜ |
| 6 | Data v2 (scale + hard test set) | ⬜ |
| 7 | Retrain + error analysis | ⬜ |
| 8 | Hyperparameter sweep + recall-first thresholding | ⬜ |
| 9 | Final eval + recommendation (incl. hybrid) | ⬜ |
| 10 | Memo + handoff | ⬜ |

---

## Day 1 — 2026-06-29 — Setup + rubric internalization ✅

**Objective (from plan §11):** stand up the environment, confirm the base model loads, and prove the
PHI labeling rubric can be applied correctly to hard cases.

### What was done
- Built the full repository skeleton (per plan §4): `src/`, `src/baselines/`, `data/`, `scripts/`,
  `reports/`, `tests/`. Every source module exists as a documented stub tagged with the day it gets
  implemented, so the structure is navigable now and filled in on schedule.
- Wrote `config.yaml` (single source of truth: seed, paths, 17-type label schema, data composition
  targets, LoRA config, sweep grid, eval thresholds, exact regex patterns) and a pinned
  `requirements.txt`.
- Set up the environment with **uv** (per request) on **Python 3.12**, GPU-accelerated PyTorch.
- Wrote and ran an environment sanity check that verifies the two known "silent killer" gotchas
  before they can cost us days later.
- Applied the PHI rubric to **10 deliberately tricky examples** and committed the worked answers.

### Definition of Done — MET ✅
| DoD item | Result (measured, not estimated) |
|---|---|
| `deberta-v3-base` loads as a token classifier | ✅ `deberta-v2`, 183.9M params, 35-label BIO space |
| GPU usable | ✅ **RTX 5070 Ti Laptop (12 GB, Blackwell)**, PyTorch `2.7.1+cu128`, CUDA 12.8 |
| Fast tokenizer + char offsets work (needed for labeling) | ✅ `offset_mapping` returns per-token char ranges |
| LoRA trains the classifier head (the #1 silent failure) | ✅ classifier head trainable; 2.68M / 186.5M params trainable (1.44%) |
| Apply rubric to 10 tricky cases | ✅ `reports/rubric_examples.md` |

Artifacts: `reports/day1_environment.md` (measured environment), `reports/rubric_examples.md`
(rubric applied to patient-vs-provider names, geography granularity, date types, the age-90 rule, and
four "look-alike" traps like order numbers and version strings).

### End-of-day self-review (brutal-truth pass) and fixes applied
I reviewed the day's work against the rules and fixed three issues before committing:
1. **Training precision footgun (fixed):** config had `fp16: true`. DeBERTa-v3 is well known to
   overflow in fp16 (NaN loss) — switched to **bf16**, which this GPU supports natively. This would
   have silently broken Day-4 training.
2. **Reproducibility bug (fixed):** the `data/raw` and `data/pools` directories would have disappeared
   on a fresh clone (empty + contents gitignored), breaking the Day-2 generator for anyone reproducing
   the repo. Added tracked `.gitkeep` placeholders while still ignoring generated data.
3. **Portability (fixed):** the one-command runner hard-coded the Windows venv path; made it work on
   Linux/macOS too, so a stranger can reproduce on any OS.

### Honest status notes
- This day is environment + scaffolding only. **No data generated and no model trained yet** — that
  is correct for Day 1 per the plan. The first real model number arrives Day 4.
- One benign warning observed: DeBERTa-v3's fast tokenizer logs a "byte-fallback" notice. It does not
  affect correctness, but it is exactly why Day 2 includes a mandatory char-span→token alignment check
  on 5 hand-verified examples before any training.

### Risks / watch-items
- **Newer toolchain than the plan assumed** (Python 3.12, Blackwell GPU, latest libraries). Verified
  working today; will keep an eye out for library edge cases as the stack gets exercised.
- The decisive metric depends on synthetic-data realism; the plan already mandates a synthetic-vs-real
  caveat and a real-data validation plan in the final memo.

### Blockers
None.

### Next (Day 2)
Build the insertion-based synthetic data generator (Faker + custom MRN/NPI/Plan-ID/etc. generators)
across the three record shapes, produce ~2k labeled rows, and **verify char-span→BIO alignment on 5
hand-checked examples before any training** — a one-character offset bug would corrupt the whole label
set.

---

## Day 2 — 2026-06-30 — Synthetic data v1 + alignment verification ✅

**Branch note:** all of week 1's work now happens on the `week1` branch (off `main`), per request.

**Objective (from plan §11):** build the insertion-based synthetic data generator, produce ~2k
labeled rows across record shapes A/B/C, and prove the char-span→token alignment is exact before any
model ever trains on it.

### What was done
- **Identifier generators** (`src/id_generators.py`): custom generators for every one of the 17 PHI
  categories (MRN, NPI, Plan-ID, Device serial, Account, License, Vehicle, SSN, phone, email, URL,
  IP, address, DOB, name, AGE90, other), each **paired with a deliberately-confusable non-PHI
  look-alike** — order numbers, SSN-shaped case tickets, public support lines, infra IPs, build
  dates, provider names, SKUs/version strings. The look-alikes are built to *trip the naive
  regex/Presidio baselines on purpose*, which is how the model earns its keep on context.
- **Carrier templates** (`src/templates.py`): clean prose templates (no PHI baked in) with a single
  insertion slot, in positive and negative phrasings, with stable IDs that get partitioned across
  splits.
- **Generator** (`src/generate.py`): assembles the three record shapes — generic API request (A),
  structured intake form (B), and log line (C) — inserting identifiers at known offsets so every
  label is exact by construction. Produced **2,000 rows (50/50 positive/negative)**; all 17
  categories represented.
- **Alignment** (`src/align.py`): char-span→BIO via `offset_mapping`, special tokens set to -100, and
  the inverse decoder for round-trip checks. The 5 hand-checked examples **all pass exact recovery**.
- **Unit tests** (`tests/`, 22 tests): cover Day 1 (config/label-space/seed/LoRA-gotcha config) and
  Day 2 (generator invariants, look-alike traps, alignment round-trip, split disjointness). All pass.

### Definition of Done — MET ✅
| DoD item | Result (verified) |
|---|---|
| Labeled JSONL loads | ✅ 2,000 records load; schema `{text, spans, contains_phi, record_type}` |
| `contains_phi` derived, not hand-set | ✅ 0 invariant violations across all 2,000 records |
| BIO alignment verified by hand | ✅ 5/5 hand-checked examples round-trip exactly (`reports/day2_alignment.md`) |
| Offsets exact | ✅ every span's `text[start:end]` is the inserted identifier; 0 out-of-bounds |
| Data integrity (leakage) | ✅ identifier **and** template pools are **0-overlap** across train/val/test |
| Reproducible | ✅ same seed → byte-identical output files (checked) |

Artifacts: `reports/day2_alignment.md` (token-by-token alignment tables) and
`reports/day2_data_summary.md` (per-split, per-category span counts).

### End-of-day self-review (brutal-truth pass) and outcome
- **Ran a real leakage check, not a trusted assumption.** My first overlap script reported ~90
  colliding identifiers between train and val — alarming. On inspection the *check* was wrong (it
  split multi-word identifiers like names and addresses on spaces), not the data. Re-checked properly
  (line-based): **identifier and template overlap is exactly 0 across all split pairs.** Logged the
  gotcha so the Day-5 automated leakage check splits on newlines, not whitespace.
- **Verified offsets the hard way:** every one of 2,000 records passes the "span text is exactly the
  inserted value" invariant, plus a 30-record generate→align→decode integration test.
- Minor cleanups to test code (removed a no-op string op, simplified an assert). No functional issues
  found in the generator or alignment.

### Honest status notes / limitations (not bugs — scheduled work)
- **Thin per-category coverage in val/test.** Because carrier-template IDs are partitioned disjointly
  across splits (the leakage requirement) and v1 has only a small template bank, the val/test splits
  cover fewer categories than train (train: all 17; val: 4; test: 7, plus NAME/DATE/MRN guaranteed by
  the intake-form shape). This is fine for the Day-4 pipeline smoke test; **Day 6 scales the template
  bank and builds the dedicated hard test set that actually decides the verdict.**
- **AGE90 is genuinely hard by construction.** Positive (age > 89) and negative (age ≤ 89) use
  identical phrasing and differ only by the number, so the model must learn the HIPAA threshold, not
  a keyword. If AGE90 recall lags later, this is why — a real finding to report, not a defect.
- **NPI labeling is the documented conservative call.** A subject-linked NPI is labeled PHI; a
  provider's NPI in professional context is a look-alike (not PHI), matching the rubric. Flagged so
  the choice is visible.
- Still no model trained — that is correct for Day 2. First LoRA numbers arrive Day 4.

### Risks / watch-items
- The disjoint-template constraint vs. per-category coverage tension will need a bigger, well-balanced
  template bank in Day 6 to guarantee ≥300 positives per common category in *every* split.

### Blockers
None.

### Next (Day 3)
Implement and run all three baselines on the v1 set — regex (exact patterns from the plan), Presidio
(the bar to beat), and the few-shot decoder — and produce the first baseline comparison so we can
state Presidio's recall/precision before any LoRA number exists.
