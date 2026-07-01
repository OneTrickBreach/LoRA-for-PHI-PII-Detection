"""Evaluation harness (plan.md §10, rules.md §5). Day-3 scope: score baselines and emit a
comparison table. Day 4 adds the LoRA system, recall-first thresholding, and one-command run_all.

Span correctness = OVERLAP with a gold span of the SAME type (lead metric); strict-exact also
reported (rules.md §5.3). Matching is one-to-one greedy so a predicted span can satisfy at most one
gold. Reports span P/R/F1, binary recall (did we flag any PHI in a PHI record), per-category recall,
false positives on negative (look-alike-only) records, and MEASURED latency (rules.md §5.6).

Usage:  python -m src.evaluate --systems regex presidio fewshot --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.config import REPO_ROOT, label_list, load_config, set_global_seed
from src.predict import load_predictor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def _overlap(a: dict, b: dict) -> bool:
    return a["type"] == b["type"] and a["start"] < b["end"] and a["end"] > b["start"]


def _exact(a: dict, b: dict) -> bool:
    return a["type"] == b["type"] and a["start"] == b["start"] and a["end"] == b["end"]


def match_spans(gold: list[dict], pred: list[dict], predicate) -> tuple[int, int, int]:
    """One-to-one greedy match. Returns (tp, fp, fn) at the span level."""
    used = [False] * len(pred)
    tp = 0
    for g in gold:
        for i, p in enumerate(pred):
            if not used[i] and predicate(g, p):
                used[i] = True
                tp += 1
                break
    fn = len(gold) - tp
    fp = used.count(False)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return recall, precision, f1


def load_records(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs


def score_system(records: list[dict], predictor, measure_latency: bool = True) -> dict:
    preds: list[list[dict]] = []
    latencies: list[float] = []
    for rec in records:
        t0 = time.perf_counter()
        p = predictor.predict(rec["text"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(p)

    agg = {"overlap": [0, 0, 0], "exact": [0, 0, 0]}
    per_cat = defaultdict(lambda: [0, 0])  # type -> [tp, total_gold] on overlap
    bin_tp = bin_pos = 0
    fp_on_neg = 0

    for rec, pred in zip(records, preds):
        gold = rec["spans"]
        for mode, predicate in (("overlap", _overlap), ("exact", _exact)):
            tp, fp, fn = match_spans(gold, pred, predicate)
            agg[mode][0] += tp
            agg[mode][1] += fp
            agg[mode][2] += fn
        # per-category recall on overlap
        for g in gold:
            per_cat[g["type"]][1] += 1
        matched_used = [False] * len(pred)
        for g in gold:
            for i, p in enumerate(pred):
                if not matched_used[i] and _overlap(g, p):
                    matched_used[i] = True
                    per_cat[g["type"]][0] += 1
                    break
        # binary recall + FP on negatives
        if rec["contains_phi"] == 1:
            bin_pos += 1
            if pred:
                bin_tp += 1
        else:
            fp_on_neg += len(pred)

    result = {"name": predictor.name}
    for mode in ("overlap", "exact"):
        r, p, f = prf(*agg[mode])
        result[mode] = {"recall": r, "precision": p, "f1": f,
                        "tp": agg[mode][0], "fp": agg[mode][1], "fn": agg[mode][2]}
    result["binary_recall"] = (bin_tp / bin_pos) if bin_pos else 0.0
    result["fp_on_negatives"] = fp_on_neg
    result["per_category_recall"] = {
        t: (per_cat[t][0] / per_cat[t][1] if per_cat[t][1] else None)
        for t in label_list()
    }
    if measure_latency and latencies:
        result["latency_ms_mean"] = sum(latencies) / len(latencies)
        result["latency_ms_p50"] = sorted(latencies)[len(latencies) // 2]
    return result


def render_table(results: list[dict], split: str, n_records: int) -> str:
    cfg = load_config()
    types = label_list()
    lines = [f"# Baseline Comparison — split=`{split}` (n={n_records})\n",
             "Span correctness is OVERLAP with a gold span of the same type (lead metric; "
             "strict-exact shown alongside). Latency measured on this hardware (RTX 5070 Ti). "
             "Recall-first matched-recall comparison and the LoRA system arrive Day 4.\n",
             "| system | span-R (overlap) | span-P (overlap) | span-F1 (overlap) | "
             "span-F1 (exact) | binary-R | FP on negatives | latency ms/rec |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['overlap']['recall']:.3f} | {r['overlap']['precision']:.3f} | "
            f"{r['overlap']['f1']:.3f} | {r['exact']['f1']:.3f} | {r['binary_recall']:.3f} | "
            f"{r['fp_on_negatives']} | {r.get('latency_ms_mean', float('nan')):.2f} |")

    lines.append("\n## Per-category recall (overlap)\n")
    header = "| system | " + " | ".join(types) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(types) + 1))
    for r in results:
        cells = []
        for t in types:
            v = r["per_category_recall"][t]
            cells.append("—" if v is None else f"{v:.2f}")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")

    lines.append("\n**Note:** `—` = category not present in this split's gold "
                 "(v1 val/test coverage is thin by design; Day 6 hard test set fixes this). "
                 "FP on negatives counts predicted spans on look-alike-only records "
                 "(hard-negative false positives).")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score PHI/PII detection systems on a split.")
    ap.add_argument("--systems", nargs="+", default=["regex", "presidio"],
                    choices=["regex", "presidio", "fewshot", "lora"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "hard_test"])
    ap.add_argument("--out", default="reports/comparison_table.md")
    args = ap.parse_args()

    cfg = load_config()
    set_global_seed()
    path = Path(REPO_ROOT) / cfg["paths"]["data_raw"] / f"{args.split}.jsonl"
    records = load_records(path)
    print(f"Loaded {len(records)} records from {path.name}")

    results = []
    for system in args.systems:
        print(f"Scoring '{system}' ...")
        predictor = load_predictor(system, cfg)
        res = score_system(records, predictor)
        results.append(res)
        print(f"  overlap R/P/F1 = {res['overlap']['recall']:.3f}/"
              f"{res['overlap']['precision']:.3f}/{res['overlap']['f1']:.3f} | "
              f"binary-R = {res['binary_recall']:.3f} | "
              f"FP(neg) = {res['fp_on_negatives']} | "
              f"latency = {res.get('latency_ms_mean', float('nan')):.2f} ms/rec")

    table = render_table(results, args.split, len(records))
    out_path = Path(REPO_ROOT) / args.out
    out_path.write_text(table, encoding="utf-8")
    # also dump raw metrics as JSON for later reuse
    (out_path.with_suffix(".json")).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
