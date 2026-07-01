"""Regex baseline (plan.md §8.1). Uses the EXACT patterns in config.yaml `regex_patterns` — not a
deliberately weak strawman (rules.md §7 last bullet).

Expected to miss names / free-text addresses / context-dependent cases, and to FALSE-POSITIVE on
look-alikes that share a shape (SSN-formatted tickets, public support lines, infra IPs, build
dates). That gap is the point of the comparison. Predict returns spans in the unified format
`[{"start", "end", "type"}]`.
"""
from __future__ import annotations

import re

from src.config import load_config

# config key -> our category type (keys already match our type names)
_KEY_TO_TYPE = {"SSN": "SSN", "EMAIL": "EMAIL", "PHONE": "PHONE",
                "IP": "IP", "DATE": "DATE", "MRN": "MRN"}


class RegexBaseline:
    name = "regex"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.patterns = [(_KEY_TO_TYPE[k], re.compile(p))
                         for k, p in cfg["regex_patterns"].items() if k in _KEY_TO_TYPE]

    def predict(self, text: str) -> list[dict]:
        spans: list[dict] = []
        for phi_type, pat in self.patterns:
            for m in pat.finditer(text):
                spans.append({"start": m.start(), "end": m.end(), "type": phi_type})
        return _dedupe(spans)


def _dedupe(spans: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for s in spans:
        key = (s["start"], s["end"], s["type"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return sorted(out, key=lambda s: (s["start"], s["end"]))
