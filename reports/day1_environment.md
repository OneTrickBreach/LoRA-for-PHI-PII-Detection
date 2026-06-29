# Day 1 — Environment Sanity Check

- seed: `20260629`
- python: `3.12.10`  platform: `Windows-11-10.0.26200-SP0`
- torch: `2.7.1+cu128`
- transformers: `4.49.0`
- peft: `0.14.0`
- torch.cuda.is_available(): `True`
- cuda device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- cuda capability (sm): `(12, 0)`
- torch CUDA build: `12.8`

## Model load
- base model: `microsoft/deberta-v3-base`  | BIO label space size: `35`
- tokenizer is_fast: `True`
- offset_mapping returned: `True` (25 tokens)
- model loaded: `deberta-v2`  | params: `183.9M`

## LoRA wrap (modules_to_save check)
- trainable params: `2.681M` / `186.5M` (`1.44%`)
- classifier head trainable: `True`  (MUST be True — rules.md §4.1)
- forward pass logits shape: `(1, 25, 35)` on `cuda`

**DoD status:** deberta-v3-base loads ✓, fast tokenizer + offset_mapping ✓, LoRA classifier head trainable ✓, forward pass on cuda ✓.
