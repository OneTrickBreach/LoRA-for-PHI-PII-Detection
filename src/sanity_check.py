"""Day-1 environment sanity check (plan.md §11 Day 1 DoD).

Proves, with MEASURED facts (rules.md §1.6), that the stack is ready:
  1. Library versions + CUDA/GPU actually visible to torch.
  2. microsoft/deberta-v3-base loads as a TokenClassification model with our BIO label space.
  3. The FAST tokenizer returns offset_mapping (prerequisite for the Day-2 alignment gotcha).
  4. LoRA can wrap the model with modules_to_save=["classifier"] and the classifier stays trainable.

Run:  python -m src.sanity_check
Writes: reports/day1_environment.md
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so status glyphs don't crash printing.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src.config import REPO_ROOT, bio_label_list, load_config, set_global_seed


def main() -> None:
    cfg = load_config()
    seed = set_global_seed()
    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg)
        lines.append(msg)

    emit(f"# Day 1 — Environment Sanity Check\n")
    emit(f"- seed: `{seed}`")
    emit(f"- python: `{platform.python_version()}`  platform: `{platform.platform()}`")

    # 1. Versions + CUDA
    import torch
    import transformers
    import peft

    emit(f"- torch: `{torch.__version__}`")
    emit(f"- transformers: `{transformers.__version__}`")
    emit(f"- peft: `{peft.__version__}`")
    cuda = torch.cuda.is_available()
    emit(f"- torch.cuda.is_available(): `{cuda}`")
    if cuda:
        emit(f"- cuda device: `{torch.cuda.get_device_name(0)}`")
        emit(f"- cuda capability (sm): `{torch.cuda.get_device_capability(0)}`")
        emit(f"- torch CUDA build: `{torch.version.cuda}`")
    device = "cuda" if cuda else "cpu"

    # 2. + 3. Load model + fast tokenizer with offset_mapping
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    labels = bio_label_list(cfg)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    base = cfg["model"]["base"]

    emit(f"\n## Model load")
    emit(f"- base model: `{base}`  | BIO label space size: `{len(labels)}`")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    emit(f"- tokenizer is_fast: `{tok.is_fast}`")
    assert tok.is_fast, "Fast tokenizer required for offset_mapping (alignment gotcha)."

    sample = "Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith."
    enc = tok(sample, return_offsets_mapping=True, truncation=True,
              max_length=cfg["model"]["max_length"])
    offsets = enc["offset_mapping"]
    emit(f"- offset_mapping returned: `{offsets is not None}` ({len(offsets)} tokens)")
    assert offsets and any(e > s for s, e in offsets), "offset_mapping looks empty."

    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )
    n_params = sum(p.numel() for p in model.parameters())
    emit(f"- model loaded: `{model.config.model_type}`  | params: `{n_params/1e6:.1f}M`")

    # 4. LoRA wrap with the mandatory classifier head (rules.md §4.1)
    from peft import LoraConfig, TaskType, get_peft_model

    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lcfg["r"], lora_alpha=lcfg["lora_alpha"], lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        modules_to_save=lcfg["modules_to_save"],
    )
    peft_model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    # Confirm the classifier head is actually trainable (the silent killer).
    clf_trainable = any(
        p.requires_grad for n, p in peft_model.named_parameters() if "classifier" in n
    )
    emit(f"\n## LoRA wrap (modules_to_save check)")
    emit(f"- trainable params: `{trainable/1e6:.3f}M` / `{total/1e6:.1f}M` "
         f"(`{100*trainable/total:.2f}%`)")
    emit(f"- classifier head trainable: `{clf_trainable}`  "
         f"(MUST be True — rules.md §4.1)")
    assert clf_trainable, "classifier head is frozen! Check modules_to_save."

    # Tiny forward pass on-device to confirm the wired model runs end to end.
    peft_model.to(device)
    enc2 = tok(sample, return_tensors="pt", truncation=True,
               max_length=cfg["model"]["max_length"]).to(device)
    peft_model.eval()
    import torch as _t
    with _t.no_grad():
        out = peft_model(**enc2)
    emit(f"- forward pass logits shape: `{tuple(out.logits.shape)}` on `{device}`")

    emit(f"\n**DoD status:** deberta-v3-base loads ✓, fast tokenizer + offset_mapping ✓, "
         f"LoRA classifier head trainable ✓, forward pass on {device} ✓.")

    out_path = Path(REPO_ROOT) / "reports" / "day1_environment.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
