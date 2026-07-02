"""Automated entity/template overlap check (rules.md §3.3 — MANDATORY before any number is trusted).

Asserts ZERO overlap of PHI identifiers and carrier template IDs across splits. If it fails:
REGENERATE the data — never hand-patch leaky data (rules.md §3.3, §8). Exits non-zero on any overlap.

Two authoritative checks:
  1. Identifiers are derived from the JSONL GROUND TRUTH (the actual PHI span substrings), not just
     the pools bookkeeping, so a generator bug can't hide. AGE90 is excluded by design — an age like
     92 must be allowed in every split (documented exception, rules.md §8).
  2. Template IDs come from data/pools/<split>_templates.txt (carrier template partition).
Note: identifiers/templates are compared as whole lines — multi-word values (names, addresses)
must NOT be split on spaces (a lesson from the Day-2 self-review).

Usage:  python -m src.leakage_check
Writes: reports/day5_leakage_check.md ; exit code 0 = clean, 1 = leakage.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

from src.config import REPO_ROOT, load_config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

EXCLUDE_FROM_IDENTIFIER_POOL = {"AGE90"}  # intentionally shared across splits


def identifiers_from_jsonl(path: Path) -> set[str]:
    """The ground-truth PHI identifier strings in a split (excluding AGE90)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for s in rec["spans"]:
                if s["type"] in EXCLUDE_FROM_IDENTIFIER_POOL:
                    continue
                ids.add(rec["text"][s["start"]:s["end"]])
    return ids


def _lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {l for l in path.read_text(encoding="utf-8").split("\n") if l}


def run_check(cfg: dict | None = None, splits=("train", "val", "test", "hard_test")) -> dict:
    cfg = cfg or load_config()
    raw = Path(REPO_ROOT) / cfg["paths"]["data_raw"]
    pool = Path(REPO_ROOT) / cfg["paths"]["data_pools"]

    present = [sp for sp in splits if (raw / f"{sp}.jsonl").exists()]
    ident = {sp: identifiers_from_jsonl(raw / f"{sp}.jsonl") for sp in present}
    tmpl = {sp: _lines(pool / f"{sp}_templates.txt") for sp in present}

    overlaps = {"identifiers": {}, "templates": {}}
    for a, b in combinations(present, 2):
        io = ident[a] & ident[b]
        to = tmpl[a] & tmpl[b]
        overlaps["identifiers"][f"{a}-{b}"] = sorted(io)
        overlaps["templates"][f"{a}-{b}"] = sorted(to)

    clean = all(not v for v in overlaps["identifiers"].values()) and \
        all(not v for v in overlaps["templates"].values())
    return {"present": present, "counts": {sp: len(ident[sp]) for sp in present},
            "template_counts": {sp: len(tmpl[sp]) for sp in present},
            "overlaps": overlaps, "clean": clean}


def _write_report(result: dict) -> None:
    lines = ["# Day 5 — Automated Leakage / Overlap Check (rules.md §3.3)\n",
             f"**Verdict: {'PASS — zero overlap ✓' if result['clean'] else 'FAIL — LEAKAGE ✗'}**\n",
             f"- splits present: {result['present']}",
             f"- unique PHI identifiers/split (excl. AGE90): {result['counts']}",
             f"- carrier template IDs/split: {result['template_counts']}\n",
             "## Cross-split overlaps (must all be empty)\n",
             "| pair | identifier overlap | template overlap |",
             "|---|---|---|"]
    for pair in result["overlaps"]["identifiers"]:
        io = result["overlaps"]["identifiers"][pair]
        to = result["overlaps"]["templates"][pair]
        lines.append(f"| {pair} | {len(io)} {io[:3] if io else ''} | "
                     f"{len(to)} {to[:3] if to else ''} |")
    lines.append("\n_AGE90 values are intentionally shared across splits (an age must be allowed "
                 "everywhere) and are excluded from the identifier pool by design (rules.md §8)._")
    out = Path(REPO_ROOT) / "reports" / "day5_leakage_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    result = run_check()
    for pair, io in result["overlaps"]["identifiers"].items():
        to = result["overlaps"]["templates"][pair]
        print(f"{pair}: identifier overlap={len(io)}, template overlap={len(to)}")
    _write_report(result)
    if not result["clean"]:
        raise SystemExit("LEAKAGE DETECTED — regenerate the data (do not hand-patch).")
    print("PASS: zero identifier/template overlap across splits.")


if __name__ == "__main__":
    main()
