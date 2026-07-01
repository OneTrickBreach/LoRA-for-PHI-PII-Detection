"""Presidio baseline — THE BAR TO BEAT (plan.md §8.2, rules.md §1.5).

presidio-analyzer out of the box with default recognizers (spaCy en_core_web_lg backend). Maps
Presidio entity labels -> our category set (plan.md §5) for per-category comparison. Presidio has no
recognizer for MRN / NPI / PLAN_ID / DEVICE_ID / VEHICLE_ID / AGE90, which is precisely where a
trained model should win. Predict returns spans in the unified format `[{"start","end","type"}]`.
"""
from __future__ import annotations

# Presidio default entity type -> our category type. Unmapped entities are dropped.
ENTITY_MAP = {
    "PERSON": "NAME",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "IP_ADDRESS": "IP",
    "DATE_TIME": "DATE",
    "LOCATION": "ADDRESS",
    "URL": "URL",
    "US_DRIVER_LICENSE": "LICENSE",
    "US_BANK_NUMBER": "ACCOUNT",
    "IBAN_CODE": "ACCOUNT",
    "CREDIT_CARD": "ACCOUNT",
    "MEDICAL_LICENSE": "LICENSE",
    "US_ITIN": "OTHER_ID",
    "US_PASSPORT": "OTHER_ID",
}


class PresidioBaseline:
    name = "presidio"

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine

        self.engine = AnalyzerEngine()

    def predict(self, text: str) -> list[dict]:
        results = self.engine.analyze(text=text, language="en")
        spans: list[dict] = []
        for r in results:
            t = ENTITY_MAP.get(r.entity_type)
            if t is not None:
                spans.append({"start": r.start, "end": r.end, "type": t})
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
