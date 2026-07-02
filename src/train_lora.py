"""LoRA fine-tune on DeBERTa-v3 for token classification (plan.md §9, rules.md §4). Day 4.

CRITICAL gotchas, pre-solved (do not remove):
  * modules_to_save=["classifier"] — or the head never learns; accuracy stuck at chance (§9 A).
  * target_modules="all-linear" — robust vs hand-naming DeBERTa proj layers (§9).
  * labels built via src.align (offset_mapping, -100 on specials), verified before this ran (§9 B).

Recall leads (rules.md §1.4), so we select the best checkpoint on validation RECALL, not F1.
Trains on GPU (hard-asserted). Logs seed, versions, device, val metrics, adapter size, train time.

Usage:  python -m src.train_lora
Writes: artifacts/lora_adapter/, reports/day4_training.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.align import align_spans_to_bio, label_maps
from src.config import REPO_ROOT, load_config, set_global_seed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def build_dataset(path: Path, tokenizer, cfg):
    """Build a token-classification dataset. Also counts gold spans lost to truncation: a span whose
    tokens fall beyond max_length gets no B- tag and would be trained as O (data-review finding)."""
    from datasets import Dataset

    labels, _, id2label = label_maps(cfg)
    b_ids = {i for i, l in id2label.items() if l.startswith("B-")}
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows, truncated = [], 0
    for r in records:
        enc = align_spans_to_bio(r["text"], r["spans"], tokenizer, cfg=cfg)
        n_b = sum(1 for lid in enc["labels"] if lid in b_ids)
        if n_b < len(r["spans"]):
            truncated += len(r["spans"]) - n_b
        rows.append({"input_ids": enc["input_ids"],
                     "attention_mask": enc["attention_mask"],
                     "labels": enc["labels"]})
    if truncated:
        print(f"[train] WARNING: {truncated} gold span(s) lost to max_length truncation in "
              f"{path.name} — increase model.max_length if this is nonzero on v2/hard_test.")
    return Dataset.from_list(rows)


def main() -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from seqeval.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments)

    cfg = load_config()
    seed = set_global_seed()

    # rules.md §1.7 — reproducibility. Make CUDA as deterministic as practical so retraining with the
    # same seed reproduces the same model (the audit caught large run-to-run precision variance).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn_only: don't crash on rare ops

    # rules.md §1.6 — measure on GPU; fail fast and loud if the GPU is not used.
    assert torch.cuda.is_available(), "GPU not available — refusing to train on CPU."
    device = torch.cuda.get_device_name(0)
    print(f"[train] seed={seed} | device={device} | torch={torch.__version__}")

    labels, label2id, id2label = label_maps(cfg)
    base = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)

    train_ds = build_dataset(Path(REPO_ROOT) / cfg["paths"]["train"], tokenizer, cfg)
    val_ds = build_dataset(Path(REPO_ROOT) / cfg["paths"]["val"], tokenizer, cfg)
    print(f"[train] train={len(train_ds)} val={len(val_ds)} | labels={len(labels)}")

    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(labels), id2label=id2label, label2id=label2id)
    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type=TaskType.TOKEN_CLS, r=lcfg["r"], lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"], target_modules=lcfg["target_modules"],
        modules_to_save=lcfg["modules_to_save"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        gold = p.label_ids
        true_pred, true_gold = [], []
        for pr, gd in zip(preds, gold):
            tp = [id2label[int(a)] for a, b in zip(pr, gd) if b != -100]
            tg = [id2label[int(b)] for a, b in zip(pr, gd) if b != -100]
            true_pred.append(tp)
            true_gold.append(tg)
        return {"precision": precision_score(true_gold, true_pred),
                "recall": recall_score(true_gold, true_pred),
                "f1": f1_score(true_gold, true_pred),
                "accuracy": accuracy_score(true_gold, true_pred)}

    tcfg = cfg["train"]
    ckpt_dir = Path(REPO_ROOT) / "artifacts" / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        save_strategy=tcfg["save_strategy"],
        bf16=tcfg["bf16"], fp16=tcfg["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="recall",   # recall leads (rules.md §1.4)
        greater_is_better=True,
        seed=seed, data_seed=seed, report_to=[], logging_dir=str(ckpt_dir / "logs"),
        save_total_limit=1,
    )
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, processing_class=tokenizer,
                      compute_metrics=compute_metrics)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    final = trainer.evaluate()

    adapter_out = Path(REPO_ROOT) / cfg["paths"]["adapter_out"]
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    (adapter_out / "label_list.json").write_text(json.dumps(labels), encoding="utf-8")
    adapter_bytes = sum(f.stat().st_size for f in adapter_out.glob("**/*") if f.is_file())

    _write_report(cfg, seed, device, final, train_time, peak_mem_mb, adapter_bytes,
                  len(train_ds), len(val_ds))
    print(f"[train] done in {train_time:.1f}s | peak GPU {peak_mem_mb:.0f} MB | "
          f"adapter {adapter_bytes/1e6:.1f} MB")
    print(f"[train] val recall={final.get('eval_recall'):.3f} "
          f"precision={final.get('eval_precision'):.3f} f1={final.get('eval_f1'):.3f}")


def _write_report(cfg, seed, device, final, train_time, peak_mem_mb, adapter_bytes,
                  n_train, n_val) -> None:
    lcfg, tcfg = cfg["lora"], cfg["train"]
    lines = [
        "# Day 4 — LoRA Training Summary\n",
        f"- seed: `{seed}` | device: `{device}` | train/val: {n_train}/{n_val}",
        f"- base: `{cfg['model']['base']}` | max_length: {cfg['model']['max_length']}",
        f"- LoRA: r={lcfg['r']} alpha={lcfg['lora_alpha']} dropout={lcfg['lora_dropout']} "
        f"target_modules=`{lcfg['target_modules']}` modules_to_save=`{lcfg['modules_to_save']}`",
        f"- optim: lr={tcfg['learning_rate']} epochs={tcfg['epochs']} "
        f"batch={tcfg['per_device_train_batch_size']} bf16={tcfg['bf16']}",
        f"- best-checkpoint selection metric: **recall** (recall leads)\n",
        "## Measured (this hardware)",
        f"- train time: **{train_time:.1f} s**",
        f"- peak GPU memory: **{peak_mem_mb:.0f} MB**",
        f"- adapter size: **{adapter_bytes/1e6:.2f} MB**",
        f"- val token-level (seqeval): recall **{final.get('eval_recall'):.3f}**, "
        f"precision **{final.get('eval_precision'):.3f}**, f1 **{final.get('eval_f1'):.3f}**, "
        f"accuracy {final.get('eval_accuracy'):.3f}\n",
        "> Token-level seqeval here is the training monitor. The decisive **span-level** comparison "
        "vs. baselines is produced by `src.evaluate` (overlap, per-category, recall-first).",
    ]
    out = Path(REPO_ROOT) / "reports" / "day4_training.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[train] wrote {out}")


if __name__ == "__main__":
    main()
