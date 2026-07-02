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
    """One-to-one greedy match (used for the exact predicate). Returns (tp, fp, fn)."""
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


def match_overlap(gold: list[dict], pred: list[dict]) -> tuple[list[tuple[int, int]], list[bool]]:
    """One-to-one overlap match, each gold taking the SAME-TYPE pred with the LARGEST overlap
    (order-independent, unlike first-match greedy). Returns (matched (gi,pi) pairs, pred-used mask).
    """
    used = [False] * len(pred)
    pairs: list[tuple[int, int]] = []
    for gi, g in enumerate(gold):
        best_pi, best_ov = None, 0
        for pi, p in enumerate(pred):
            if used[pi] or p["type"] != g["type"]:
                continue
            ov = min(g["end"], p["end"]) - max(g["start"], p["start"])
            if ov > 0 and ov > best_ov:
                best_ov, best_pi = ov, pi
        if best_pi is not None:
            used[best_pi] = True
            pairs.append((gi, best_pi))
    return pairs, used


def merge_overlapping_same_type(spans: list[dict]) -> list[dict]:
    """Merge overlapping spans of the SAME type into one, so a system is not double-counted (extra
    TP or FP) for redundant overlapping detections of a single entity. Applied uniformly to every
    system before scoring, to keep the LoRA-vs-Presidio comparison fair (rules.md §7)."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_type[s["type"]].append(s)
    out: list[dict] = []
    for t, items in by_type.items():
        cur = None
        for s in sorted(items, key=lambda x: (x["start"], x["end"])):
            if cur and s["start"] < cur["end"]:          # strict overlap -> merge
                cur["end"] = max(cur["end"], s["end"])
            else:
                if cur:
                    out.append(cur)
                cur = dict(s)
        if cur:
            out.append(cur)
    return sorted(out, key=lambda s: (s["start"], s["end"]))


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


def score_system(records: list[dict], predictor, measure_latency: bool = True,
                 latency_warmup: int = 0) -> dict:
    preds: list[list[dict]] = []
    latencies: list[float] = []
    for rec in records:
        t0 = time.perf_counter()
        raw = predictor.predict(rec["text"])          # time the model call only
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(merge_overlapping_same_type(raw))  # normalize before scoring (post-hoc)

    agg = {"overlap": [0, 0, 0], "exact": [0, 0, 0]}
    per_cat = defaultdict(lambda: [0, 0])  # type -> [tp, total_gold] on overlap
    bin_tp = bin_pos = 0
    fp_on_neg = 0

    for rec, pred in zip(records, preds):
        gold = rec["spans"]
        # overlap: single one-to-one match; per-category derived from the SAME matched pairs
        pairs, used = match_overlap(gold, pred)
        agg["overlap"][0] += len(pairs)
        agg["overlap"][1] += used.count(False)
        agg["overlap"][2] += len(gold) - len(pairs)
        etp, efp, efn = match_spans(gold, pred, _exact)
        agg["exact"][0] += etp
        agg["exact"][1] += efp
        agg["exact"][2] += efn
        for g in gold:
            per_cat[g["type"]][1] += 1
        for gi, _pi in pairs:
            per_cat[gold[gi]["type"]][0] += 1
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
    # latency: discard the first `latency_warmup` records (cold CUDA init) per config (rules.md §5.6)
    stat_lat = latencies[latency_warmup:] if len(latencies) > latency_warmup else latencies
    if measure_latency and stat_lat:
        result["latency_ms_mean"] = sum(stat_lat) / len(stat_lat)
        result["latency_ms_p50"] = sorted(stat_lat)[len(stat_lat) // 2]
    return result


def render_table(results: list[dict], split: str, n_records: int) -> str:
    cfg = load_config()
    types = label_list()
    lines = [f"# Baseline Comparison — split=`{split}` (n={n_records})\n",
             "Span correctness is OVERLAP with a gold span of the same type (lead metric; "
             "strict-exact shown alongside). Latency measured on this hardware (RTX 5070 Ti). "
             "For LoRA, `argmax` is the default operating point and `R>=<target>@t=<thr>` is the "
             "recall-first point whose threshold was selected on val (never on this split).\n",
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


def _score_lora(cfg, records, split, target_recall, report) -> list[dict]:
    """Score LoRA at argmax and at the recall-first operating point (threshold picked on val)."""
    from src.predict import LoraPredictor, select_threshold_for_recall

    results = []
    warmup = cfg["eval"]["latency_warmup"]
    predictor = LoraPredictor(cfg, threshold=None)

    # (a) argmax operating point
    predictor.threshold = None
    res = score_system(records, predictor, latency_warmup=warmup)
    res["name"] = "lora(argmax)"
    results.append(res)
    report(res)

    # (b) recall-first: pick the threshold on VAL (never on the eval split), apply here
    val_path = Path(REPO_ROOT) / cfg["paths"]["val"]
    if split != "val" and val_path.exists():
        val_records = load_records(val_path)
        sel = select_threshold_for_recall(val_records, predictor, target_recall)
        predictor.threshold = sel["threshold"]
        res = score_system(records, predictor, latency_warmup=warmup)
        tag = "" if sel["met"] else " (target NOT met on val)"
        res["name"] = f"lora(R>={target_recall}@t={sel['threshold']}{tag})"
        res["threshold_selection"] = {"on": "val", **sel}
        results.append(res)
        report(res)
        print(f"    threshold {sel['threshold']} selected on val: "
              f"val recall={sel['recall']:.3f} precision={sel['precision']:.3f} "
              f"(target {target_recall} {'met' if sel['met'] else 'NOT met'})")
    return results


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
    target_recall = cfg["eval"]["target_recall"]

    def report(res):
        print(f"  {res['name']}: overlap R/P/F1 = {res['overlap']['recall']:.3f}/"
              f"{res['overlap']['precision']:.3f}/{res['overlap']['f1']:.3f} | "
              f"binary-R = {res['binary_recall']:.3f} | FP(neg) = {res['fp_on_negatives']} | "
              f"latency = {res.get('latency_ms_mean', float('nan')):.2f} ms/rec")

    results = []
    for system in args.systems:
        print(f"Scoring '{system}' ...")
        if system == "lora":
            results.extend(_score_lora(cfg, records, args.split, target_recall, report))
            continue
        predictor = load_predictor(system, cfg)
        res = score_system(records, predictor, latency_warmup=cfg["eval"]["latency_warmup"])
        res["name"] = predictor.name
        results.append(res)
        report(res)

    table = render_table(results, args.split, len(records))
    out_path = Path(REPO_ROOT) / args.out
    out_path.write_text(table, encoding="utf-8")
    # also dump raw metrics as JSON for later reuse
    (out_path.with_suffix(".json")).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
