# Day 5 — Mid-Project Self-Review (against rules.md + plan.md)

Checkpoint at the end of Week 1. Purpose: verify we haven't drifted from the hard constraints before
scaling up in Week 2, and state the honest position on the success bar.

## Rules.md compliance checklist

| Rule (rules.md) | Status | Evidence |
|---|---|---|
| §1.1 Synthetic data only | ✅ | 100% generated; no real/scraped data anywhere |
| §1.2 Never hand-label positives | ✅ | PHI inserted at known offsets; labels exact by construction |
| §1.3 LLM emits carrier/negatives only | ✅ | Carrier text is authored templates; identifiers inserted by code |
| §1.4 Recall leads | ✅ | Best checkpoint selected on **recall**; recall-first threshold path built |
| §1.5 Presidio is the bar | ✅ | Compared vs Presidio (not just regex); Presidio = 0.825R/0.213P |
| §1.6 Measure, don't estimate | ✅ | Train time 31 s, peak GPU 2971 MB, adapter 21.9 MB, latency all measured |
| §1.7 Reproducibility | ✅ | Seeded; byte-identical data; **deterministic training verified identical across reruns**; pinned deps; one-command run |
| §3 Leakage discipline + overlap check | ✅ | `leakage_check.py` passes: 0 identifier & template overlap across splits |
| §3.7 `contains_phi` derived | ✅ | Asserted `1 iff spans`; 0 violations across all records |
| §4.1 `modules_to_save=["classifier"]` | ✅ | Set; head learned (loss 0.32→0.03, acc 0.977 — not chance) |
| §4.2 `target_modules="all-linear"` | ✅ | Used |
| §4.3 Alignment verified on 5 examples | ✅ | `reports/day2_alignment.md`, all pass; round-trip unit-tested |
| §4.4 DeBERTa encoder is primary | ✅ | Decoder used only as the few-shot baseline |
| §5 Eval rigor (overlap-lead, per-cat, binary, latency) | ✅ | All in `evaluate.py`; matched-recall path built |
| §6 One-command runs, config out of code, findings in files | ✅ | `run_all.sh`; `config.yaml`; reports committed |
| §6.7 uv-managed environment | ✅ | `.venv` (uv, Python 3.12) |

## Plan.md progress (Week 1)
Days 1–5 complete, each DoD met. 43 unit tests green. Baselines + first LoRA + full comparison
harness + automated leakage check all in place.

## Honest position on the success bar (plan §2) — audited/reproducible numbers
> These are the reproducible numbers after fixing training determinism (see the Week-1 Audit in the
> journal). An earlier non-deterministic run had reported precision 0.976; that was not reproducible.
- **Precision:** at matched recall (LoRA 0.849 vs Presidio 0.825) LoRA precision **0.613 vs Presidio
  0.213 (~2.9×)**; at argmax LoRA precision **0.780** with only **4 false positives** on look-alike
  records (Presidio: 308). The core thesis — a rubric-trained model beats Presidio on precision — is
  supported on v1.
- **Recall:** LoRA **0.771** (argmax) / **0.849** (lowest threshold) — **does NOT meet the 0.97
  bar** yet, and can't be reached by thresholding alone on v1. A real, reported finding, not hidden.
- **Latency:** LoRA **26 ms/record** on GPU — **meets** the < 50 ms target (few-shot 409 ms does not).

## Root cause of the recall gap (diagnosed, not guessed)
Manual inspection of predictions shows LoRA reliably captures PHI in **structured/named fields**
(intake-form name/dob/mrn) but **misses PHI buried in free text** — email/IP/phone in the free-text
`complaint` field, and identifiers inside the shape-A JSON payload and shape-C log lines. Cause: v1
is small (1,600 train rows) with **thin, disjoint per-split category coverage**, so free-text
positives for several categories are under-represented in training. This is exactly what **Day 6**
addresses (scale to 10–20k, balance per-category coverage, build the dedicated hard test set).

## Documented decisions / deviations (conservative, per rules §8)
- **AGE90 shared across splits** — an age like 92 must appear everywhere; excluded from the
  identifier-disjointness pool by design (noted in `leakage_check.py`).
- **NPI** — subject-linked NPI labeled PHI; provider NPI treated as a non-PHI look-alike (matches the
  rubric's provider-in-professional-capacity carve-out).
- **DeBERTa tokenizer leading-space** — decoded spans previously began one char early; now trimmed in
  `predict.trim_spans` so strict-exact scoring and redaction boundaries are clean.
- **Thin v1 val/test category coverage** — a consequence of disjoint-template splitting on a small
  bank; accepted for the Week-1 pipeline, fixed by the Day-6 template-bank scale-up.

## Risks going into Week 2
1. **Recall is the whole ballgame.** If scaling + balanced free-text coverage doesn't lift recall to
   0.97, the honest outcome may be a **hybrid** (Presidio/regex pre-filter → LoRA) — already on the
   Day-9 agenda.
2. **Synthetic ≠ real.** The memo must caveat this and give a real-data validation plan (Day 10).
3. **Hard test set is the verdict** and doesn't exist yet — all current numbers are directional.

## Next (Week 2)
Day 6 scale + hard test set → Day 7 retrain + error analysis → Day 8 sweep + recall-first threshold →
Day 9 final eval + hybrid verdict → Day 10 memo.
