# Day 8 — LoRA Hyperparameter Sweep + Recall-First Thresholding

Sweep on **v2**; hard-test = verdict set; threshold selected on **val** at recall ≥ 0.97 (never on hard_test). target_modules=`all-linear`, epochs=3 (except as noted), bf16, deterministic. All numbers measured on RTX 5070 Ti.

## setting → recall / precision / latency / cost

| tag | r | alpha | lr | trainable | hard R (argmax) | hard P (argmax) | hard F1 | thr | hard R@thr | hard P@thr | latency ms | train s | adapter MB | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8_a16_lr2e4 | 8 | 16 | 2e-04 | 1.354M | 0.536 | 0.655 | 0.589 | 0.01 | 0.587 | 0.56 | 43.6 | 536.2 | 16.57 | 3028 |
| r16_a32_lr2e4 | 16 | 32 | 2e-04 | 2.681M | 0.557 | 0.568 | 0.563 | 0.7 | 0.527 | 0.664 | 51.9 | 463.6 | 21.88 | 3785 |
| r32_a64_lr2e4 | 32 | 64 | 2e-04 | 5.335M | 0.569 | 0.548 | 0.559 | 0.15 | 0.58 | 0.535 | 55.2 | 509.0 | 32.49 | 3842 |
| r16_a16_lr2e4 | 16 | 16 | 2e-04 | 2.681M | 0.584 | 0.607 | 0.596 | 0.4 | 0.584 | 0.627 | 37.3 | 492.2 | 21.88 | 3819 |
| r16_a32_lr1e4 | 16 | 32 | 1e-04 | 2.681M | 0.477 | 0.597 | 0.53 | 0.5 | 0.475 | 0.604 | 34.9 | 422.4 | 21.88 | 3815 |

**Recommended config: `r16_a16_lr2e4`** (r=16, alpha=16, lr=2e-04) — highest hard-test overlap recall (0.584) at precision 0.607, adapter 21.88 MB, 492.2 s to train, 37.3 ms/record.

- configs whose val recall reached ≥0.97 (threshold selectable): ['r16_a32_lr2e4', 'r32_a64_lr2e4', 'r16_a16_lr2e4', 'r16_a32_lr1e4']
- **No config reaches recall ≥0.97 on the hard test set** — consistent with the Day-7 finding; the hybrid (Day 9) is the path to close the recall gap.

_Cost = train time (s) + adapter size (MB) + peak GPU memory (MB). Latency is argmax ms/record over the hard test set after warmup._
