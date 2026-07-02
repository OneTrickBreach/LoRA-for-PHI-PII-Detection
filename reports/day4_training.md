# Day 4 — LoRA Training Summary

- seed: `20260629` | device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU` | train/val: 1600/200
- base: `microsoft/deberta-v3-base` | max_length: 256
- LoRA: r=16 alpha=32 dropout=0.05 target_modules=`all-linear` modules_to_save=`['classifier', 'classifier', 'score']`
- optim: lr=0.0002 epochs=3 batch=16 bf16=True
- best-checkpoint selection metric: **recall** (recall leads)

## Measured (this hardware)
- train time: **52.2 s**
- peak GPU memory: **3045 MB**
- adapter size: **21.88 MB**
- val token-level (seqeval): recall **0.604**, precision **0.498**, f1 **0.546**, accuracy 0.979

> Token-level seqeval here is the training monitor. The decisive **span-level** comparison vs. baselines is produced by `src.evaluate` (overlap, per-category, recall-first).
