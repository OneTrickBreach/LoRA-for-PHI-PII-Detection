"""Unified predict interface for every system (plan.md §4).

Every system exposes `predict(text) -> [{"start","end","type"}]` so the eval harness scores them
identically. The LoRA system additionally supports a decision threshold for recall-first
operating-point selection (plan.md §10, rules.md §5.2): tokens are labeled with their best non-O
entity when that entity's probability clears the threshold, else O. Lowering the threshold raises
recall. `raw_token_scores` caches the per-token distribution once so a threshold sweep is cheap.
"""
from __future__ import annotations

from src.align import bio_to_char_spans, label_maps


def load_predictor(system: str, cfg: dict | None = None, **kwargs):
    if system == "regex":
        from src.baselines.regex_baseline import RegexBaseline
        return RegexBaseline(cfg)
    if system == "presidio":
        from src.baselines.presidio_baseline import PresidioBaseline
        return PresidioBaseline()
    if system == "fewshot":
        from src.baselines.fewshot_baseline import FewShotBaseline
        return FewShotBaseline(cfg)
    if system == "lora":
        return LoraPredictor(cfg, **kwargs)
    raise ValueError(f"unknown system: {system}")


def decode_tokens(raw: list[tuple], id2label: dict[int, str],
                  threshold: float | None) -> list[dict]:
    """Turn cached per-token scores into char spans.

    raw item = (cs, ce, argmax_id, best_nonO_id, best_nonO_prob, is_special).
    threshold None -> plain argmax; else assign best non-O entity iff its prob >= threshold.
    """
    o_id = next(i for i, l in id2label.items() if l == "O")
    offsets, label_ids = [], []
    for cs, ce, argmax_id, nonO_id, nonO_prob, is_special in raw:
        offsets.append((cs, ce))
        if is_special:
            label_ids.append(-100)
        elif threshold is None:
            label_ids.append(argmax_id)
        else:
            label_ids.append(nonO_id if nonO_prob >= threshold else o_id)
    return bio_to_char_spans(offsets, label_ids, id2label)


class LoraPredictor:
    name = "lora"

    def __init__(self, cfg: dict | None = None, adapter_dir: str | None = None,
                 threshold: float | None = None):
        import torch
        from peft import PeftModel
        from transformers import (AutoModelForTokenClassification, AutoTokenizer)

        from src.config import REPO_ROOT, load_config
        cfg = cfg or load_config()
        self.cfg = cfg
        self.threshold = threshold
        self.max_length = cfg["model"]["max_length"]
        labels, self.label2id, self.id2label = label_maps(cfg)
        self.o_id = self.label2id["O"]
        self._entity_ids = [i for i, l in self.id2label.items() if l != "O"]

        import json
        from pathlib import Path
        adir = adapter_dir or str(Path(REPO_ROOT) / cfg["paths"]["adapter_out"])
        # Guard against a config label-schema change after training silently mismapping ids.
        label_file = Path(adir) / "label_list.json"
        if label_file.exists():
            saved = json.loads(label_file.read_text(encoding="utf-8"))
            assert saved == labels, (
                "label schema changed since training — adapter ids would mismap. "
                "Retrain or restore config.label_types.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(adir, use_fast=True)
        base = AutoModelForTokenClassification.from_pretrained(
            cfg["model"]["base"], num_labels=len(labels),
            id2label=self.id2label, label2id=self.label2id)
        self.model = PeftModel.from_pretrained(base, adir).to(self.device)
        self.model.eval()
        self._torch = torch

    def raw_token_scores(self, text: str) -> list[tuple]:
        torch = self._torch
        enc = self.tok(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                       truncation=True, max_length=self.max_length, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()
        special = enc["special_tokens_mask"][0].tolist()
        model_in = {k: v.to(self.device) for k, v in enc.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.no_grad():
            logits = self.model(**model_in).logits[0]
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        raw = []
        for i, ((cs, ce), sp) in enumerate(zip(offsets, special)):
            row = probs[i]
            argmax_id = int(row.argmax())
            # best non-O entity for threshold mode
            ent = max(self._entity_ids, key=lambda j: float(row[j]))
            raw.append((cs, ce, argmax_id, ent, float(row[ent]), bool(sp) or ce <= cs))
        return raw

    def predict(self, text: str) -> list[dict]:
        raw = self.raw_token_scores(text)
        return trim_spans(decode_tokens(raw, self.id2label, self.threshold), text)


def trim_spans(spans: list[dict], text: str) -> list[dict]:
    """Strip leading/trailing whitespace from span char ranges.

    DeBERTa's sentencepiece offsets include the leading space of a word (the ▁ marker), so decoded
    spans start one char early (" Angela" vs "Angela"). Overlap matching tolerates it, but trimming
    improves strict-exact scoring and yields clean redaction boundaries. Drops empty spans.
    """
    out = []
    for s in spans:
        start, end = s["start"], s["end"]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        # Drop empty and pure-punctuation fragments (e.g. "." / "-") — never a real identifier;
        # the hard-test error analysis showed these as spurious false positives.
        if end > start and any(c.isalnum() for c in text[start:end]):
            out.append({"start": start, "end": end, "type": s["type"]})
    return out


def select_threshold_for_recall(records, predictor, target_recall: float,
                                grid=None) -> dict:
    """Recall-first operating point (rules.md §5.2): cache per-record token scores once, then sweep
    thresholds descending and return the HIGHEST threshold whose span-level (overlap) recall >=
    target (maximizing precision subject to the recall floor). Reports shortfall if unreachable.
    """
    from src.evaluate import _overlap, match_spans, prf

    if grid is None:
        grid = [round(x, 3) for x in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                                      0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]]
    cached = [(r["spans"], predictor.raw_token_scores(r["text"])) for r in records]
    id2label = predictor.id2label

    best = None
    for t in grid:
        tp = fp = fn = 0
        for gold, raw in cached:
            pred = decode_tokens(raw, id2label, t)
            a, b, c = match_spans(gold, pred, _overlap)
            tp += a; fp += b; fn += c
        recall, precision, f1 = prf(tp, fp, fn)
        rec = {"threshold": t, "recall": recall, "precision": precision, "f1": f1}
        if recall >= target_recall:
            return {**rec, "met": True}
        if best is None or recall > best["recall"]:
            best = rec
    return {**best, "met": False}
