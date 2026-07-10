"""Error analysis (plan.md §11 Day 7, rules.md §5.4-5.5). Breaks down, per category, where the LoRA
model wins over regex/Presidio and where rules already suffice, plus the hard-test confusion:
false positives on look-alikes (hard negatives) and false negatives on hard positives — with concrete
examples pulled from the data.

Usage:  python -m src.error_analysis --split hard_test --systems regex presidio lora
Writes: reports/error_analysis.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import REPO_ROOT, label_list, load_config, set_global_seed
from src.evaluate import load_records, match_overlap, merge_overlapping_same_type
from src.predict import load_predictor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_MAX_EX = 3


def analyze(records: list[dict], predictor) -> dict:
    """Per-category recall + FN/FP counts and example spans for one system."""
    tp, total = Counter(), Counter()
    fn_ex: dict[str, list] = defaultdict(list)   # type -> [(text, missed)]
    fp_ex: dict[str, list] = defaultdict(list)   # type -> [(text, fp_text, on_negative)]
    fp_total = fp_on_neg = 0

    for rec in records:
        gold = rec["spans"]
        pred = merge_overlapping_same_type(predictor.predict(rec["text"]))
        pairs, used = match_overlap(gold, pred)
        matched = {gi for gi, _ in pairs}
        for gi, g in enumerate(gold):
            total[g["type"]] += 1
            if gi in matched:
                tp[g["type"]] += 1
            elif len(fn_ex[g["type"]]) < _MAX_EX:
                fn_ex[g["type"]].append((rec["text"], rec["text"][g["start"]:g["end"]]))
        for pi, p in enumerate(pred):
            if not used[pi]:
                fp_total += 1
                on_neg = rec["contains_phi"] == 0
                fp_on_neg += int(on_neg)
                if len(fp_ex[p["type"]]) < _MAX_EX:
                    fp_ex[p["type"]].append((rec["text"], rec["text"][p["start"]:p["end"]], on_neg))
    per_cat = {t: (tp[t] / total[t] if total[t] else None) for t in label_list()}
    return {"per_cat": per_cat, "fn_ex": dict(fn_ex), "fp_ex": dict(fp_ex),
            "fp_total": fp_total, "fp_on_neg": fp_on_neg}


def _clip(s: str, n: int = 90) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def build_report(results: dict[str, dict], split: str, n: int) -> str:
    types = label_list()
    L = [f"# Error Analysis — split=`{split}` (n={n})\n",
         "Per-category recall (overlap) across systems, then where ML wins, where rules already "
         "suffice, and the hard-test confusion (FP on look-alikes, FN on hard positives). "
         "LoRA uses its argmax operating point here.\n",
         "## Per-category recall\n",
         "| category | " + " | ".join(results) + " |",
         "|" + "---|" * (len(results) + 1)]
    for t in types:
        cells = []
        for sysname in results:
            v = results[sysname]["per_cat"][t]
            cells.append("—" if v is None else f"{v:.2f}")
        L.append(f"| {t} | " + " | ".join(cells) + " |")

    # Where ML wins / where rules suffice (compare lora vs best rule-based system)
    if "lora" in results:
        rule_systems = [s for s in results if s in ("regex", "presidio")]
        wins, suffice = [], []
        for t in types:
            lo = results["lora"]["per_cat"][t]
            if lo is None:
                continue
            rule_best = max([results[s]["per_cat"][t] or 0.0 for s in rule_systems], default=0.0)
            if lo - rule_best >= 0.30:
                wins.append((t, rule_best, lo))
            elif rule_best >= 0.90 and lo >= 0.90:
                suffice.append((t, rule_best, lo))
        L.append("\n## Where ML earns its place (LoRA ≫ best rule-based, Δrecall ≥ 0.30)\n")
        if wins:
            L.append("| category | best rule recall | LoRA recall |")
            L.append("|---|---|---|")
            for t, rb, lo in sorted(wins, key=lambda x: x[2] - x[1], reverse=True):
                L.append(f"| {t} | {rb:.2f} | {lo:.2f} |")
        else:
            L.append("_None at this threshold._")
        L.append("\n## Where rules already suffice (both ≥ 0.90)\n")
        L.append(", ".join(t for t, _, _ in suffice) if suffice else "_None._")

    # Hard-test confusion
    L.append("\n## Hard-test confusion (false positives)\n")
    L.append("| system | total FP | FP on look-alike-only (negative) records |")
    L.append("|---|---|---|")
    for s in results:
        L.append(f"| {s} | {results[s]['fp_total']} | {results[s]['fp_on_neg']} |")

    # Concrete examples for LoRA (the system of interest)
    focus = "lora" if "lora" in results else list(results)[0]
    r = results[focus]
    L.append(f"\n## `{focus}` false negatives on hard positives (missed PHI), by category\n")
    any_fn = False
    for t in types:
        exs = r["fn_ex"].get(t)
        if exs:
            any_fn = True
            L.append(f"**{t}**")
            for text, missed in exs:
                L.append(f"- missed `{missed}` in: {_clip(text)}")
    if not any_fn:
        L.append("_No false negatives._")
    L.append(f"\n## `{focus}` false positives on look-alikes, by category\n")
    any_fp = False
    for t in types:
        exs = r["fp_ex"].get(t)
        if exs:
            any_fp = True
            L.append(f"**{t}**")
            for text, fp_text, on_neg in exs:
                tag = "neg-record" if on_neg else "wrong-span"
                L.append(f"- flagged `{fp_text}` ({tag}) in: {_clip(text)}")
    if not any_fp:
        L.append("_No false positives._")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-category error analysis for PHI/PII systems.")
    ap.add_argument("--split", default="hard_test",
                    choices=["train", "val", "test", "hard_test"])
    ap.add_argument("--systems", nargs="+", default=["regex", "presidio", "lora", "hybrid"],
                    choices=["regex", "presidio", "lora", "hybrid"])
    ap.add_argument("--out", default="reports/error_analysis.md")
    args = ap.parse_args()

    cfg = load_config()
    set_global_seed()
    records = load_records(Path(REPO_ROOT) / cfg["paths"]["data_raw"] / f"{args.split}.jsonl")
    print(f"Loaded {len(records)} records from {args.split}.jsonl")

    results = {}
    for s in args.systems:
        print(f"Analyzing '{s}' ...")
        results[s] = analyze(records, load_predictor(s, cfg))

    report = build_report(results, args.split, len(records))
    out = Path(REPO_ROOT) / args.out
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
