"""Hyperparameter sweep + recall-first thresholding (plan.md §11 Day 8, §9; rules.md §4.5).

Sweeps LoRA rank/alpha/LR (a curated subset touching each axis — a full 24-cell grid is unnecessary
and expensive), trains each config deterministically, selects the recall-first threshold on VAL
(never on the verdict set), scores on the hard test set, and logs measured
time / GPU-memory / adapter-size / latency. Emits `setting → recall/precision/latency/cost`.

Usage:  python -m src.sweep
Writes: reports/sweep_results.md (+ .json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import REPO_ROOT, load_config, set_global_seed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Curated sweep: r∈{8,16,32} at alpha=2r; plus alpha=r and a lower-LR point at r=16.
# epochs held at 3 (best in Day-4/7); target_modules held at all-linear (rules.md §4.2).
CONFIGS = [
    {"tag": "r8_a16_lr2e4",  "r": 8,  "alpha": 16, "lr": 2.0e-4, "epochs": 3},
    {"tag": "r16_a32_lr2e4", "r": 16, "alpha": 32, "lr": 2.0e-4, "epochs": 3},   # default
    {"tag": "r32_a64_lr2e4", "r": 32, "alpha": 64, "lr": 2.0e-4, "epochs": 3},
    {"tag": "r16_a16_lr2e4", "r": 16, "alpha": 16, "lr": 2.0e-4, "epochs": 3},   # alpha=r
    {"tag": "r16_a32_lr1e4", "r": 16, "alpha": 32, "lr": 1.0e-4, "epochs": 3},   # lower LR
]


def run() -> None:
    from src.evaluate import load_records, score_system
    from src.predict import LoraPredictor, select_threshold_for_recall
    from src.train_lora import train

    cfg = load_config()
    set_global_seed()
    target = cfg["eval"]["target_recall"]
    warmup = cfg["eval"]["latency_warmup"]
    val_records = load_records(Path(REPO_ROOT) / cfg["paths"]["val"])
    hard_records = load_records(Path(REPO_ROOT) / cfg["paths"]["hard_test"])
    sweep_root = Path(REPO_ROOT) / "artifacts" / "sweep"

    import gc

    import torch

    rows = []
    for c in CONFIGS:
        print(f"\n===== sweep config {c['tag']} =====")
        tr = train(cfg, r=c["r"], lora_alpha=c["alpha"], learning_rate=c["lr"],
                   epochs=c["epochs"], adapter_out=sweep_root / c["tag"],
                   ckpt_dir=sweep_root / c["tag"] / "_ckpt")   # per-config ckpt: no cross-config bleed
        predictor = LoraPredictor(cfg, adapter_dir=tr["adapter_dir"], threshold=None)

        # recall-first threshold picked on VAL, applied to hard_test
        sel = select_threshold_for_recall(val_records, predictor, target)

        predictor.threshold = None
        argmax = score_system(hard_records, predictor, latency_warmup=warmup)
        predictor.threshold = sel["threshold"]
        thr = score_system(hard_records, predictor, latency_warmup=warmup)

        row = {
            **{k: c[k] for k in ("tag", "r", "alpha", "lr", "epochs")},
            "trainable_M": round(tr["trainable_params"] / 1e6, 3),
            "train_time_s": round(tr["train_time_s"], 1),
            "peak_mem_mb": round(tr["peak_mem_mb"]),
            "adapter_mb": round(tr["adapter_mb"], 2),
            "val_token_recall": round(tr["val_recall"], 3),
            "hard_argmax_recall": round(argmax["overlap"]["recall"], 3),
            "hard_argmax_precision": round(argmax["overlap"]["precision"], 3),
            "hard_argmax_f1": round(argmax["overlap"]["f1"], 3),
            "threshold": sel["threshold"], "val_thr_recall": round(sel["recall"], 3),
            "target_met_on_val": sel["met"],
            "hard_thr_recall": round(thr["overlap"]["recall"], 3),
            "hard_thr_precision": round(thr["overlap"]["precision"], 3),
            "hard_thr_f1": round(thr["overlap"]["f1"], 3),
            "latency_ms": round(argmax.get("latency_ms_mean", float("nan")), 1),
        }
        rows.append(row)
        print(f"[sweep] {c['tag']}: hard argmax R/P/F1="
              f"{row['hard_argmax_recall']}/{row['hard_argmax_precision']}/{row['hard_argmax_f1']} "
              f"| {row['train_time_s']}s | {row['adapter_mb']}MB")
        # free GPU memory between configs so 5 train+eval cycles don't accumulate
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    _write_report(cfg, rows, target)


def _pick_best(rows: list[dict]) -> dict:
    # Recall leads (rules.md §1.4): best hard-test overlap recall (argmax), tie-break on precision.
    return max(rows, key=lambda r: (r["hard_argmax_recall"], r["hard_argmax_precision"]))


def _write_report(cfg, rows: list[dict], target: float) -> None:
    best = _pick_best(rows)
    L = ["# Day 8 — LoRA Hyperparameter Sweep + Recall-First Thresholding\n",
         f"Sweep on **v2**; hard-test = verdict set; threshold selected on **val** at recall ≥ "
         f"{target} (never on hard_test). target_modules=`all-linear`, epochs=3 (except as noted), "
         "bf16, deterministic. All numbers measured on RTX 5070 Ti.\n",
         "## setting → recall / precision / latency / cost\n",
         "| tag | r | alpha | lr | trainable | hard R (argmax) | hard P (argmax) | hard F1 | "
         "thr | hard R@thr | hard P@thr | latency ms | train s | adapter MB | peak MB |",
         "|" + "---|" * 15]
    for r in rows:
        L.append(
            f"| {r['tag']} | {r['r']} | {r['alpha']} | {r['lr']:.0e} | {r['trainable_M']}M | "
            f"{r['hard_argmax_recall']} | {r['hard_argmax_precision']} | {r['hard_argmax_f1']} | "
            f"{r['threshold']} | {r['hard_thr_recall']} | {r['hard_thr_precision']} | "
            f"{r['latency_ms']} | {r['train_time_s']} | {r['adapter_mb']} | {r['peak_mem_mb']} |")

    L.append(f"\n**Recommended config: `{best['tag']}`** "
             f"(r={best['r']}, alpha={best['alpha']}, lr={best['lr']:.0e}) — highest hard-test "
             f"overlap recall ({best['hard_argmax_recall']}) at precision "
             f"{best['hard_argmax_precision']}, adapter {best['adapter_mb']} MB, "
             f"{best['train_time_s']} s to train, {best['latency_ms']} ms/record.\n")
    met = [r["tag"] for r in rows if r["target_met_on_val"]]
    L.append(f"- configs whose val recall reached ≥{target} (threshold selectable): {met or 'none'}")
    L.append(f"- **No config reaches recall ≥{target} on the hard test set** — consistent with the "
             "Day-7 finding; the hybrid (Day 9) is the path to close the recall gap.\n")
    L.append("_Cost = train time (s) + adapter size (MB) + peak GPU memory (MB). Latency is "
             "argmax ms/record over the hard test set after warmup._")
    out = Path(REPO_ROOT) / "reports" / "sweep_results.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[sweep] best={best['tag']} | wrote {out}")


if __name__ == "__main__":
    run()
