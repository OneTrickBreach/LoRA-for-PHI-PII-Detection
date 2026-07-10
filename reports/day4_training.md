# LoRA Training Summary (canonical run)

- seed: `20260629` | device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU` | train/val: 12800/1600
- base: `microsoft/deberta-v3-base` | max_length: 256
- LoRA: r=16 alpha=16 dropout=0.05 target_modules=`all-linear` modules_to_save=`['classifier', 'classifier', 'score']` | trainable params: 2.681M
- optim: lr=0.0002 epochs=3 batch=16 bf16=True
- best-checkpoint selection metric: **recall** (recall leads)

## Measured (this hardware)
- train time: **411.0 s**
- peak GPU memory: **3049 MB**
- adapter size: **21.88 MB**
- val token-level (seqeval): recall **0.946**, precision **0.907**, f1 **0.926**, accuracy 0.995

> Token-level seqeval here is the training monitor. The decisive **span-level** comparison vs. baselines is produced by `src.evaluate` (overlap, per-category, recall-first).
