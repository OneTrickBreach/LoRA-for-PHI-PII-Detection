"""Few-shot decoder baseline, zero training (plan.md §8.3).

Uses the EXACT prompt in plan.md §8.3 with a small instruct decoder (config `model.fewshot`,
default Qwen2.5-1.5B-Instruct). The model returns a JSON list of {text, type}; we resolve each to a
char offset by locating the returned text in the record. The decoder is alt/stretch only — never the
primary (rules.md §4.4). Greedy decoding for reproducibility. Predict returns unified spans.
"""
from __future__ import annotations

import json
import re

from src.config import label_list, load_config

PROMPT = """You are a PHI/PII detector. Return JSON: a list of spans, each {text, type}.
Types: NAME, ADDRESS, DATE, SSN, MRN, NPI, PLAN_ID, ACCOUNT, LICENSE,
DEVICE_ID, VEHICLE_ID, PHONE, EMAIL, URL, IP, AGE90, OTHER_ID.
Patient/member identifiers are PHI. Provider names, org addresses, bare years,
order/ticket numbers are NOT PHI. Return [] if none.

Example 1:
Input: "Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith."
Output: [{"text":"John Reyes","type":"NAME"},{"text":"03/14/1981","type":"DATE"},{"text":"A55213","type":"MRN"}]

Example 2:
Input: "Order 99812 shipped to our Austin office, support 800-555-0000."
Output: []

Input: "%s"
Output:"""


def extract_spans_from_output(record_text: str, raw_output: str,
                              label_set: set[str] | None = None) -> list[dict]:
    """Parse the model's JSON output and locate each returned string in `record_text`.

    Pure function (no model needed) so it is unit-testable. Locates each returned text at its first
    unused occurrence to disambiguate repeats; drops items whose type is unknown or text not found.
    """
    label_set = label_set or set(label_list())
    arr = _first_json_array(raw_output)
    if arr is None:
        return []
    spans: list[dict] = []
    cursor: dict[str, int] = {}  # per-substring search offset for repeated values
    for item in arr:
        if not isinstance(item, dict):
            continue
        frag = str(item.get("text", "")).strip()
        typ = str(item.get("type", "")).strip().upper()
        if not frag or typ not in label_set:
            continue
        start_from = cursor.get(frag, 0)
        idx = record_text.find(frag, start_from)
        if idx < 0:
            idx = record_text.find(frag)  # fall back to first occurrence
        if idx < 0:
            continue
        cursor[frag] = idx + len(frag)
        spans.append({"start": idx, "end": idx + len(frag), "type": typ})
    return sorted(spans, key=lambda s: (s["start"], s["end"]))


def _first_json_array(text: str):
    """Extract and parse the first balanced [...] JSON array in `text`."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class FewShotBaseline:
    name = "fewshot"

    def __init__(self, cfg: dict | None = None, model_id: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = cfg or load_config()
        self.model_id = model_id or cfg["model"]["fewshot"]
        self.label_set = set(label_list(cfg))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> list[dict]:
        import torch

        # Sanitize quotes so they don't break the single-line prompt template.
        safe = text.replace('"', "'").replace("\n", " ")
        user = PROMPT % safe
        messages = [{"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True,
                          max_length=2048).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_spans_from_output(text, gen, self.label_set)
