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
| 2 | Synthetic data v1 + alignment test | ⬜ |
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
