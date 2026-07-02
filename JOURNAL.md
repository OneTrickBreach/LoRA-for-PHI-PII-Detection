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
| 3 | All three baselines (regex, Presidio, few-shot) | ✅ |
| 4 | First LoRA run + eval harness | ✅ |
| 5 | Harden generator + leakage check + README | ✅ |
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

---

## Day 3 — 2026-07-01 — All three baselines + eval harness ✅

**Objective (from plan §11):** implement the regex, Presidio, and few-shot baselines, score them on
the v1 test set through one shared evaluation harness, and be able to state Presidio's
recall/precision — the bar LoRA must beat — before any LoRA number exists.

### What was done
- **Regex baseline** (`src/baselines/regex_baseline.py`): the exact six patterns from the plan (SSN,
  email, phone, IP, date, MRN) — not a strawman.
- **Presidio baseline** (`src/baselines/presidio_baseline.py`): `presidio-analyzer` out of the box
  (spaCy `en_core_web_lg`), with its entity labels mapped to our 17-category schema.
- **Few-shot baseline** (`src/baselines/fewshot_baseline.py`): the exact §8.3 prompt on a small
  instruct decoder (Qwen2.5-1.5B-Instruct), greedy decoding; returned JSON spans located back to
  char offsets. JSON parsing factored into a pure, unit-tested function.
- **Shared eval harness** (`src/evaluate.py` + `src/predict.py`): one command scores any system with
  span P/R/F1 on **overlap** (lead) and strict-exact, **per-category recall**, **binary recall**,
  **false positives on negative (look-alike-only) records**, and **measured latency**. Emits
  `reports/comparison_table.md` (+ a JSON dump).
- **13 new unit tests** (8 baseline behavior + few-shot parsing edge cases, 5 metric-math); full
  suite now **35 tests, all green**.

### Results on the v1 test set (n=200) — the headline
| system | span-R (overlap) | span-P (overlap) | binary-R | FP on negatives | latency ms/rec |
|---|---|---|---|---|---|
| regex | 0.584 | 0.503 | 0.840 | 75 | 0.03 |
| **presidio (the bar)** | **0.825** | **0.213** | 0.980 | 308 | 11.0 |
| fewshot (Qwen-1.5B) | 0.482 | 0.860 | 0.260 | 9 | 445 |

**Presidio's recall/precision: 0.825 / 0.213 (overlap).** This is the number to beat — LoRA must
match Presidio's recall (target ≥0.97) while dramatically improving precision.

### What the per-category breakdown reveals (this is the story)
- **Presidio catches names/dates/URLs at 1.00 but misses MRN entirely (0.00)** — it has no
  domain-identifier recognizer (MRN/NPI/PLAN_ID/DEVICE_ID/etc.). Its precision is only 0.213 because
  it over-flags by our rubric: provider names, standalone states, org addresses, and software build
  dates all get tagged (308 false positives on look-alike-only records). That gap — miss domain IDs,
  over-flag look-alikes — is precisely the opening for a rubric-trained model.
- **Regex** is fast and precise-ish on formatted IDs but blind to names and URLs (0.00) and only
  catches MRN when it carries the literal "MRN" prefix (0.58).
- **Few-shot** is the opposite of Presidio: high precision (0.86), poor recall (0.48) and very low
  binary recall (0.26), at 445 ms/record.

### End-of-day self-review (brutal-truth pass) and outcome
- **The one suspicious number — few-shot binary recall of 0.26 — was investigated before trusting
  it.** I ran a 6-record diagnostic printing the model's raw output vs. parsed spans vs. gold. Result:
  it is **genuine model behavior, not a parsing bug.** The decoder extracts PHI well from the
  intake-form named fields (shape B: NAME/DATE/MRN at 1.00) but **misses PHI buried in free text** —
  the `complaint` field, the API-request JSON payload (shape A), and log lines (shape C), where it
  mostly returns `[]`. Parsing correctly handled the model's ```json fenced output and every item it
  returned. So few-shot's per-category zeros on PHONE/EMAIL/URL/IP are real: those categories live in
  free text here. This is a real finding for the memo: zero-training decoders are unreliable for
  recall on realistic noisy records.
- **Latency is measured, not estimated** (rules §5.6): regex 0.03 ms, Presidio 11 ms (under the
  50 ms target), few-shot 445 ms (≈9× over — a finding; a decoder is not viable for low-latency
  inline scanning).
- No functional bugs found in the harness; the metric math is covered by direct unit tests
  (one-to-one greedy matching, overlap vs exact, per-category, binary, FP counting).

### Honest status notes / caveats
- **Scores are on the thin-coverage v1 test split** (7 of 17 categories present). Absolute numbers
  will move once the Day-6 hard test set exists; today's purpose is a directional baseline and the
  Presidio bar, both of which are established.
- **Matched-recall comparison (recall ≥ 0.97) is not applied yet** — that needs a tunable score
  threshold, which only the LoRA system has. Day 4 introduces it; today reports each system at its
  natural operating point.
- Presidio's low precision is partly a rubric-definition mismatch (it isn't wrong about "a person
  name is a name" — it just doesn't know our provider/patient distinction). That is a fair and
  informative comparison, and the reason a task-specific model is expected to win.

### Risks / watch-items
- Few-shot latency and recall make it a non-contender; the real race is **LoRA vs. Presidio**, which
  starts Day 4.

### Blockers
None.

### Next (Day 4)
First LoRA fine-tune on DeBERTa-v3 with the §9 config (**`modules_to_save=["classifier"]`**), wire
the LoRA system into `predict.py`/`evaluate.py`, and produce the first one-command comparison of LoRA
vs. all three baselines. (If training sits at chance accuracy, check `modules_to_save` first.)

---

## Day 4 — 2026-07-03 — First LoRA fine-tune + full comparison harness ✅

> Two days of work today (Day 4 + Day 5) since tomorrow is the Independence Day holiday.

**Objective (from plan §11):** fine-tune DeBERTa-v3 with LoRA using the §9 config, wire it into the
harness, and produce one command that compares LoRA against all three baselines.

### What was done
- **`src/train_lora.py`**: LoRA fine-tune with the exact §9 config — `target_modules="all-linear"`,
  **`modules_to_save=["classifier"]`**, bf16 on GPU (with a hard `assert torch.cuda.is_available()`
  so it can never silently fall back to CPU, as promised). Best checkpoint selected on **validation
  recall** (recall leads). Logs seed, device, measured train time / GPU memory / adapter size.
- **`src/predict.py`**: LoRA predictor with a decision **threshold** for recall-first operating
  points; per-token scores cached so a threshold sweep is cheap. `select_threshold_for_recall`
  picks the operating point on **val** (never on the eval split).
- **`src/evaluate.py`**: LoRA integrated; reports it at argmax and at the recall-first point.
- **`scripts/run_all.sh`**: one command — generate → leakage check → train → score all systems.
- **5 new unit tests** for the threshold/decoding logic (no model load required).

### Result (v1 test, n=200; overlap unless noted)
| system | span-R | span-P | binary-R | FP(neg) | latency |
|---|---|---|---|---|---|
| presidio (the bar) | 0.825 | 0.213 | 0.980 | 308 | 16 ms |
| **lora (argmax)** | 0.741 | **0.976** | 0.640 | **0** | 37 ms |
| lora (recall-first, t=0.01) | 0.813 | 0.726 | 0.840 | 2 | 33 ms |

- **The win:** LoRA precision **0.976 vs Presidio 0.213**, with **0 false positives** on look-alike
  records (Presidio had 308). At ~matched recall (0.81 vs 0.83) LoRA's precision is **3.4× Presidio's**.
  Per-category, **LoRA catches MRN at 1.00 where Presidio scores 0.00** — the domain-identifier gap.
  > **Audit correction (see the Week-1 Audit entry below):** this 0.976 was a *non-reproducible*
  > training run. After fixing training determinism, the reproducible LoRA precision is lower (~0.78);
  > the qualitative win over Presidio (much higher precision, 0 FP on look-alikes, catches MRN) holds,
  > but use the audited numbers as the record.
- **Latency:** 33–37 ms/record on GPU — **meets** the < 50 ms target.
- **The gap (reported, not hidden):** LoRA recall is **0.74 (argmax) / 0.81 (lowest threshold)** —
  it does **NOT** reach the 0.97 success bar on v1, and thresholding alone can't get there (val
  recall caps ~0.72).

### DoD — MET ✅
One command (`run_all.sh` / `python -m src.evaluate --systems ... lora`) produces the full comparison
table. Training is confirmed on GPU. The classifier head **learned** (loss 0.32→0.03, acc 0.977 — not
chance), so `modules_to_save` is doing its job.

### Brutal-truth review (Day 4)
- **A 0.976 precision demanded verification.** I inspected LoRA predictions vs. gold on real records.
  It is genuine: LoRA correctly extracts named-field PHI and returns **nothing** on negative records
  (support emails, org addresses, support phones) — it truly learned the look-alikes. **The recall
  gap is a data-coverage issue, not a bug:** LoRA (like few-shot) misses PHI **buried in free text** —
  email/IP/phone in the `complaint` field, IP inside the shape-A JSON payload and shape-C log lines.
  Per-category confirms it: PHONE 0.29 and **IP 0.00** at argmax.
- **Fixed a real defect found in review:** decoded LoRA spans began one character early (DeBERTa's
  sentencepiece offset includes the leading space → `" Angela Martinez"`). Added `trim_spans`; strict-
  exact F1 improved (LoRA exact-F1 0.795) and redaction boundaries are now clean. Overlap unaffected.
- Fixed a stale auto-generated table caption.

### Next (Day 5) — done same day, below.

---

## Day 5 — 2026-07-03 — Automated leakage check + reproducibility + mid-project review ✅

**Objective (from plan §11):** enforce entity/template-level splits with an automated overlap check,
make the data regenerate reproducibly, get the README to a stranger-can-reproduce state, and do a
mid-project self-review against the spec.

### What was done
- **`src/leakage_check.py`** (rules §3.3, mandatory): asserts **zero overlap** of PHI identifiers and
  carrier template IDs across splits, exits non-zero on any leak. Identifiers are derived from the
  **JSONL ground truth** (the actual span substrings), not just the pools bookkeeping, so a generator
  bug can't hide. AGE90 is excluded by design (an age must be allowed in every split). Compares whole
  lines — deliberately **not** splitting multi-word values on spaces (the Day-2 self-review lesson).
- **Result: PASS** — 0 identifier and 0 template overlap across train/val/test
  (`reports/day5_leakage_check.md`).
- **`run_all.sh`** now runs the leakage check between generation and training, so the pipeline
  **refuses to train on leaky data**.
- **README**: added the full week-1 reproduction path (one-command `run_all.sh`), all per-day
  commands, and the current status.
- **Mid-project self-review** (`reports/day5_selfreview.md`): a rules.md compliance checklist (all
  green), plan progress, the honest position on the success bar, the diagnosed recall gap, and the
  documented conservative decisions (AGE90 sharing, NPI, tokenizer trim, thin v1 coverage).
- **3 new unit tests** for the leakage check (AGE90 exclusion, overlap detection, real-data-clean).

### DoD — MET ✅
- Data regenerates reproducibly (same seed → byte-identical, verified Day 2 and unchanged).
- **Overlap check passes.**
- README lets a stranger reproduce Week 1 end-to-end.
- Full suite: **43 unit tests green.**

### Brutal-truth review (Day 5, sequential over all work)
- Re-ran the whole suite (43 green) and the leakage check (clean) after the Day-4 changes — no
  regressions. Reproducibility, `contains_phi` derivation, offset correctness, alignment round-trip,
  and split disjointness all still hold.
- The self-review checklist surfaced no rule violations. The one substantive open item is honest and
  known: **recall is below the 0.97 bar on v1** — carried into Week 2 as the central risk, with the
  hybrid (Presidio/regex pre-filter → LoRA) already scheduled for Day 9 as the fallback.

### Honest status notes / caveats
- All numbers are on the **thin v1 test split** and are directional. The **hard test set that decides
  the verdict does not exist yet** (Day 6).
- IP recall of 0.00 for LoRA is the most striking single gap — the model isn't recognizing session
  IPs at all on v1 (too few/again free-text). First thing to watch after the Day-6 scale-up.

### Blockers
None.

### Next (Day 6)
Scale the generator to 10–20k with a **larger, balanced template bank** (≥300 positives per common
category **in every split**, especially free-text PHONE/EMAIL/IP), build the dedicated **hard test
set** (look-alike-heavy negatives + unusual-format positives) that decides the verdict, and re-run the
leakage check on the new splits.

---

## Week 1 Audit & Summary — 2026-07-03

End-of-week brutal-truth audit of all Day 1–5 work: I ran the full one-command pipeline as an
integration test, launched **two independent review agents** (data-generation path; modeling/eval
path), and did my own targeted verification. Findings were triaged and the real ones fixed.

### The most important finding: training was not reproducible
Running `run_all.sh` retrained the model and LoRA's precision came out **0.780**, not the **0.976**
my Day-4 entry reported — a large swing from the *same seed*. Root cause: **CUDA nondeterminism**,
amplified by the tiny v1 dataset. This violates the reproducibility rule (§1.7). **Fix:** cuDNN
deterministic + `torch.use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG` + `data_seed`.
**Verified:** two retrains now produce byte-identical val metrics (recall 0.604, precision 0.498).
Training time rose 27 s → 52 s (measured, acceptable). The Day-4 "0.976" is corrected to the
reproducible number below.

### All fixes applied this audit
| Severity | Area | Fix |
|---|---|---|
| bug | reproducibility | Deterministic training; verified identical across reruns |
| bug | eval fairness | Merge overlapping same-type predictions uniformly (was inflating baselines' FP) |
| bug | eval | Latency now honors `latency_warmup` (cold-CUDA record no longer skews the mean) |
| bug | alignment test | Strengthened check caught a real EMAIL off-by-one (44-61 → 44-62) in the hand examples |
| correctness | eval | Order-independent best-overlap matching; per-category derived from the same matched pairs |
| correctness | data gen | Identifier-uniqueness exhaustion now **raises** instead of silently accepting a duplicate |
| correctness | training | Warns if a gold span is lost to `max_length` truncation (matters at Day-6 scale) |
| correctness | predict | `LoraPredictor` asserts the saved `label_list.json` matches config (id-mismap guard) |

Both review agents confirmed the generator/alignment core is empirically sound (0 offset/BIO/leakage
errors over 1,850 spans; determinism holds). Documented-not-fixed (by design / Week-2): Presidio's
OOTB mapping surfaces some non-PHI as PHI (the comparison point), NAME/DATE/MRN over-representation in
v1 (Day-6 balancing), and best-checkpoint selection on token-recall vs span-overlap (Day-8).

### Audited, reproducible results (v1 test, n=200; overlap)
| system | span-R | span-P | binary-R | FP(neg) | latency |
|---|---|---|---|---|---|
| regex | 0.584 | 0.503 | 0.840 | 75 | 0.03 ms |
| **presidio (the bar)** | 0.825 | 0.213 | 0.980 | 308 | 9 ms |
| fewshot (Qwen-1.5B) | 0.476 | 0.849 | 0.260 | 9 | 409 ms |
| **lora (argmax)** | 0.771 | **0.780** | 0.700 | **4** | 26 ms |
| lora (recall-first, t=0.01) | 0.849 | 0.613 | 0.880 | 9 | 25 ms |

### Week 1 verdict (honest)
- **LoRA clearly beats Presidio on precision** at comparable recall: at matched recall (0.849 vs
  0.825) LoRA precision is **0.613 vs 0.213 (~2.9×)**, with **4–9 false positives vs 308**, and it
  **catches MRN (1.00) where Presidio scores 0.00**. It also **meets the < 50 ms latency target**
  (26 ms) where few-shot (409 ms) cannot. The core thesis — a rubric-trained model earns its place
  over Presidio — is supported directionally.
- **The 0.97 recall bar is NOT met on v1** (LoRA recall 0.77 argmax / 0.85 at the lowest threshold).
  Diagnosed cause: LoRA misses PHI **buried in free text** (PHONE/EMAIL/IP in the complaint field and
  JSON/log payloads), a v1 data-coverage limitation, not a modeling bug. This is Day-6's target.
- All numbers are on the **thin v1 test split** and are **directional**; the hard test set that
  decides the verdict is built Day 6.

### Week 1 deliverables — all DoDs met
Day 1 env + rubric ✓ · Day 2 generator + verified alignment ✓ · Day 3 three baselines + harness ✓ ·
Day 4 first LoRA + one-command comparison ✓ · Day 5 leakage check + reproducibility + README ✓.
**45 unit tests green; leakage check clean; pipeline reproducible end-to-end.**

### Risks carried into Week 2
1. **Recall is the whole game.** If Day-6 scale + balanced free-text coverage doesn't lift recall to
   0.97, the honest recommendation may be the **hybrid** (Presidio/regex pre-filter → LoRA), already
   on the Day-9 agenda.
2. **Synthetic ≠ real** — memo must caveat + give a real-data validation plan (Day 10).
3. Small-data seed sensitivity is now controlled (deterministic) but the Day-8 sweep should still
   check a couple of seeds so the reported config isn't a fragile point estimate.
