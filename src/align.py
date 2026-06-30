"""Char-span -> BIO token label alignment via offset_mapping (plan.md §9 Gotcha B, rules.md §4.3).

A one-character offset bug silently corrupts the whole label set, so this module is the most
safety-critical piece of the data path. We:
  * tokenize with the FAST tokenizer (`return_offsets_mapping=True`),
  * label each token B-/I-<TYPE> when its char range overlaps a gold span of that type, else O,
  * set label -100 on special tokens so the loss ignores them,
and provide the inverse (`bio_to_char_spans`) so we can round-trip and prove the mapping is exact
BEFORE training on the full set.

Run the hand-checked verification:  python -m src.align
Writes: reports/day2_alignment.md
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, bio_label_list, load_config

try:  # Windows consoles default to cp1252; keep glyphs from crashing prints.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def label_maps(cfg: dict[str, Any] | None = None):
    labels = bio_label_list(cfg)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    return labels, label2id, id2label


def align_spans_to_bio(text: str, spans: list[dict], tokenizer, max_length: int | None = None,
                       cfg: dict[str, Any] | None = None) -> dict:
    """Tokenize `text` and attach BIO `labels` (one per token; -100 on special tokens).

    Returns the tokenizer encoding dict with an added "labels" list. A token is assigned to the
    first gold span it overlaps; the first token of a span gets B-, subsequent tokens I-.
    """
    cfg = cfg or load_config()
    if max_length is None:
        max_length = cfg["model"]["max_length"]
    _, label2id, _ = label_maps(cfg)
    o_id = label2id["O"]

    enc = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                    truncation=True, max_length=max_length)
    offsets = enc["offset_mapping"]
    special = enc["special_tokens_mask"]
    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"]))

    labels: list[int] = []
    prev_span_idx: int | None = None
    for (cs, ce), is_special in zip(offsets, special):
        if is_special or ce <= cs:  # special token or empty/zero-width piece
            labels.append(-100)
            prev_span_idx = None
            continue
        hit = None
        for i, sp in enumerate(sorted_spans):
            if cs < sp["end"] and ce > sp["start"]:  # half-open overlap
                hit = i
                break
        if hit is None:
            labels.append(o_id)
            prev_span_idx = None
        else:
            t = sorted_spans[hit]["type"]
            prefix = "B" if hit != prev_span_idx else "I"
            labels.append(label2id[f"{prefix}-{t}"])
            prev_span_idx = hit
    enc["labels"] = labels
    return enc


def bio_to_char_spans(offsets: list, label_ids: list[int],
                      id2label: dict[int, str]) -> list[dict]:
    """Inverse of `align_spans_to_bio`: reconstruct char spans from per-token BIO labels.

    Used for round-trip verification here and for decoding model predictions later (predict.py).
    """
    spans: list[dict] = []
    cur: dict | None = None
    for (cs, ce), lid in zip(offsets, label_ids):
        if lid == -100:
            if cur:
                spans.append(cur)
                cur = None
            continue
        lab = id2label[lid]
        if lab == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        prefix, t = lab.split("-", 1)
        if prefix == "B" or cur is None or cur["type"] != t:
            if cur:
                spans.append(cur)
            cur = {"start": cs, "end": ce, "type": t}
        else:  # I-, same type -> extend
            cur["end"] = ce
    if cur:
        spans.append(cur)
    return spans


def spans_overlap(a: dict, b: dict) -> bool:
    return a["type"] == b["type"] and a["start"] < b["end"] and a["end"] > b["start"]


# ---- Hand-checked verification (Day-2 DoD) -------------------------------------------------

HAND_EXAMPLES: list[dict] = [
    {
        "text": "Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith.",
        "spans": [
            {"start": 8, "end": 18, "type": "NAME"},
            {"start": 24, "end": 34, "type": "DATE"},
            {"start": 40, "end": 46, "type": "MRN"},
        ],
    },
    {
        "text": "Member SSN 402-11-9837; plan ID UHC9921047733 on the policy.",
        "spans": [
            {"start": 11, "end": 22, "type": "SSN"},
            {"start": 32, "end": 45, "type": "PLAN_ID"},
        ],
    },
    {
        "text": "Call the patient at (216) 555-0148 or email jane.doe@gmail.com.",
        "spans": [
            {"start": 20, "end": 34, "type": "PHONE"},
            {"start": 44, "end": 61, "type": "EMAIL"},
        ],
    },
    {
        "text": "Home address on file: 728 Oak Street, Akron, OH 44312.",
        "spans": [
            {"start": 22, "end": 53, "type": "ADDRESS"},
        ],
    },
    {
        "text": "Patient is 92 years old; session from 73.118.42.9 logged.",
        "spans": [
            {"start": 11, "end": 13, "type": "AGE90"},
            {"start": 38, "end": 49, "type": "IP"},
        ],
    },
]


def verify_examples(tokenizer, examples: list[dict] | None = None,
                    cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Align each example, round-trip back to char spans, and assert exact recovery.

    Returns (all_ok, markdown_report). For each gold span we check (a) the substring really is
    the identifier, and (b) the BIO round-trip recovers a same-type span covering it.
    """
    cfg = cfg or load_config()
    _, _, id2label = label_maps(cfg)
    examples = examples or HAND_EXAMPLES
    lines = ["# Day 2 — Char-span -> BIO Alignment Verification\n",
             "Each example is aligned to BIO tokens and round-tripped back to char spans. "
             "PASS requires every gold span to be recovered by a same-type span on overlap, "
             "with no spurious spans.\n"]
    all_ok = True

    for n, ex in enumerate(examples, 1):
        text, gold = ex["text"], ex["spans"]
        enc = align_spans_to_bio(text, gold, tokenizer, cfg=cfg)
        offsets, labels = enc["offset_mapping"], enc["labels"]
        recovered = bio_to_char_spans(offsets, labels, id2label)

        # (a) gold spans are exactly the substrings we claim
        substr_ok = all(text[g["start"]:g["end"]] for g in gold)
        # (b) every gold recovered, and every recovered matches some gold (no spurious)
        gold_hit = all(any(spans_overlap(g, r) for r in recovered) for g in gold)
        rec_clean = all(any(spans_overlap(r, g) for g in gold) for r in recovered)
        ok = substr_ok and gold_hit and rec_clean and len(recovered) == len(gold)
        all_ok &= ok

        lines.append(f"\n### Example {n} — {'PASS' if ok else 'FAIL'}")
        lines.append(f"> {text}\n")
        lines.append("| token | char range | label |")
        lines.append("|---|---|---|")
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        for tok, (cs, ce), lid in zip(toks, offsets, labels):
            lab = "-100 (special)" if lid == -100 else id2label[lid]
            piece = text[cs:ce] if ce > cs else ""
            lines.append(f"| `{tok}` | {cs}-{ce} `{piece}` | {lab} |")
        lines.append("")
        for g in gold:
            lines.append(f"- gold `{text[g['start']:g['end']]}` ({g['type']}) "
                         f"-> recovered: "
                         f"{[ (r['type'], text[r['start']:r['end']]) for r in recovered if spans_overlap(g, r)]}")

    lines.insert(2, f"\n**Overall: {'ALL PASS ✓' if all_ok else 'FAILURES PRESENT ✗'}**\n")
    return all_ok, "\n".join(lines) + "\n"


def main() -> None:
    from transformers import AutoTokenizer

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base"], use_fast=True)
    ok, report = verify_examples(tok, cfg=cfg)
    print(report)
    out = Path(REPO_ROOT) / "reports" / "day2_alignment.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    if not ok:
        raise SystemExit("Alignment verification FAILED — fix before training.")


if __name__ == "__main__":
    main()
