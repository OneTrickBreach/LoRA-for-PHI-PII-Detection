"""Insertion-based synthetic data generator (plan.md §7, rules.md §1-3).

Pipeline per record: pick a record shape (A/B/C) -> assemble CLEAN scaffolding -> insert
identifiers (positives) or look-alikes only (negatives) at KNOWN char offsets via `Emitter`, which
records each PHI span exactly as it places the value. `contains_phi` is derived (1 iff spans).
Splits are entity/template-level: carrier template IDs are partitioned disjointly, and poolable PHI
identifier VALUES are globally unique so no identifier crosses splits (rules.md §3). Pools are
written to data/pools for the Day-5 overlap check.

One-command run:  python -m src.generate --version v1
Writes: data/raw/{train,val,test}.jsonl, data/pools/*, reports/day2_data_summary.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from src.config import REPO_ROOT, load_config, set_global_seed
from src.id_generators import IdGen
from src.templates import (CARRIER_TEMPLATES, HARD_NEG_TEMPLATES,
                           HARD_POS_TEMPLATES)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


class Emitter:
    """Builds a record string left-to-right, recording exact char spans for inserted PHI."""

    def __init__(self) -> None:
        self.buf: list[str] = []
        self.pos = 0
        self.spans: list[dict] = []

    def lit(self, s: str) -> None:
        self.buf.append(s)
        self.pos += len(s)

    def slot(self, value: str, phi_type: str | None) -> None:
        start = self.pos
        self.buf.append(value)
        self.pos += len(value)
        if phi_type is not None:
            self.spans.append({"start": start, "end": self.pos, "type": phi_type})

    def text(self) -> str:
        return "".join(self.buf)


class SplitGen:
    """Holds per-split disjointness state and emits identifiers without cross-split collisions."""

    def __init__(self, gens: IdGen, sel: random.Random,
                 pos_ids: list[dict], neg_ids: list[dict], used_phi: set[str]):
        self.gens = gens
        self.sel = sel
        self.pos_templates = pos_ids
        self.neg_templates = neg_ids
        self.used_phi = used_phi          # GLOBAL across splits -> guarantees disjoint identifiers
        self.identifiers: set[str] = set()  # poolable PHI values used in THIS split
        self.template_ids: set[str] = set()
        # Round-robin over kinds so positives are BALANCED per category (fix Week-1 skew).
        self.pos_by_kind: dict[str, list[dict]] = {}
        for t in pos_ids:
            self.pos_by_kind.setdefault(t["kind"], []).append(t)
        self.pos_kinds = sorted(self.pos_by_kind)
        self._rr = 0

    def next_pos_template(self) -> dict:
        """Next positive template, cycling kinds so every category gets roughly equal positives."""
        kind = self.pos_kinds[self._rr % len(self.pos_kinds)]
        self._rr += 1
        return self.sel.choice(self.pos_by_kind[kind])

    def _emit_value(self, em: Emitter, value: str, phi_type: str | None, poolable: bool,
                    regen=None) -> None:
        if phi_type is not None and poolable:
            tries = 0
            while value in self.used_phi and regen is not None and tries < 100:
                value = regen()
                tries += 1
            if value in self.used_phi:
                # rules.md §3.3: never silently accept a duplicate identifier (silent leakage).
                raise RuntimeError(
                    f"identifier generator exhausted uniqueness for a {phi_type} value; "
                    "increase generator cardinality or reduce dataset size — do not accept dupes.")
            self.used_phi.add(value)
            self.identifiers.add(value)
        em.slot(value, phi_type)

    def carrier(self, em: Emitter, tmpl: dict) -> None:
        """Render one carrier sentence: literal prefix + slot value + literal suffix."""
        kind, positive = tmpl["kind"], tmpl["polarity"] == "pos"
        parts = tmpl["text"].split("{x}")
        assert len(parts) == 2, f"template {tmpl['id']} must have exactly one {{x}} slot"
        pre, post = parts
        em.lit(pre)
        value, phi_type, poolable = self.gens.value_for(kind, positive)
        self._emit_value(em, value, phi_type, poolable,
                         regen=lambda: self.gens.value_for(kind, positive)[0])
        em.lit(post)
        self.template_ids.add(tmpl["id"])

    def field_phi(self, em: Emitter, gen_method: str, phi_type: str) -> None:
        self._emit_value(em, getattr(self.gens, gen_method)(), phi_type, True,
                         regen=lambda: getattr(self.gens, gen_method)())

    def pick_sentences(self, positive: bool) -> list[dict]:
        if positive:
            # round-robin primary positive (balanced categories) + some look-alikes (hard positives)
            sents = [self.next_pos_template()]
            for _ in range(self.sel.randint(0, 2)):
                sents.append(self.sel.choice(self.neg_templates))
        else:
            sents = [self.sel.choice(self.neg_templates)
                     for _ in range(self.sel.randint(1, 3))]
        self.sel.shuffle(sents)
        return sents


def _render_sentence_list(sg: SplitGen, em: Emitter, sentences: list[dict]) -> None:
    for i, t in enumerate(sentences):
        if i:
            em.lit(" ")
        sg.carrier(em, t)


def build_A(sg: SplitGen, positive: bool) -> tuple[str, list[dict]]:
    """Shape A — generic API request with a free-text payload (plan.md §6 A)."""
    em = Emitter()
    em.lit('{"request_id": "')
    em.lit(sg.gens.request_id())                      # look-alike, not PHI
    em.lit('", "source": "intake", "payload": {"text": "')
    _render_sentence_list(sg, em, sg.pick_sentences(positive))
    em.lit('", "notes": ["')
    em.lit(sg.gens.request_id())                       # benign note id
    em.lit('"]}}')
    return em.text(), em.spans


def build_B(sg: SplitGen, positive: bool) -> tuple[str, list[dict]]:
    """Shape B — structured intake form with named fields (plan.md §6 B)."""
    em = Emitter()
    if positive:
        em.lit("form_type: intake | name: ")
        sg.field_phi(em, "name", "NAME")
        em.lit(" | dob: ")
        sg.field_phi(em, "dob_date", "DATE")
        em.lit(" | mrn: ")
        sg.field_phi(em, "mrn", "MRN")
        em.lit(" | state: ")
        em.lit(sg.gens.state_abbr())                   # standalone state, NOT PHI
        em.lit(" | complaint: ")
        sg.carrier(em, sg.next_pos_template())   # balanced extra PHI category in free text
        em.lit(" | provider: Dr. ")
        em.lit(sg.gens.provider_last())                # provider name, NOT PHI
    else:
        em.lit("form_type: facility | org: ")
        em.lit(sg.gens.company())                      # org name, NOT PHI
        em.lit(" | address: ")
        em.lit(sg.gens.org_address())                  # public business address, NOT PHI
        em.lit(" | support: ")
        em.lit(sg.gens.support_phone())                # public support line, NOT PHI
        em.lit(" | note: ")
        sg.carrier(em, sg.sel.choice(sg.neg_templates))
        em.lit(" | contact: Dr. ")
        em.lit(sg.gens.provider_last())
    return em.text(), em.spans


def build_C(sg: SplitGen, positive: bool) -> tuple[str, list[dict]]:
    """Shape C — semi-structured log line (plan.md §6 C)."""
    em = Emitter()
    em.lit(sg.gens.build_timestamp())                  # log timestamp, NOT PHI
    em.lit(' INFO ingest source=intake msg="')
    _render_sentence_list(sg, em, sg.pick_sentences(positive))
    em.lit('"')
    return em.text(), em.spans


SHAPES = {
    "A": (build_A, "api_request"),
    "B": (build_B, None),   # record_type set by polarity below
    "C": (build_C, "log_line"),
}


def partition_template_ids(sel: random.Random, fracs: dict) -> dict[str, dict]:
    """Partition carrier template IDs PER KIND across train/val/test so every split has ≥1 pos and
    ≥1 neg template for EVERY category (fixes Week-1 thin val/test coverage), while keeping template
    IDs disjoint across splits (rules.md §3.1). With 4 templates/kind: val=1, test=1, train=rest.
    """
    out = {sp: {"pos": [], "neg": []} for sp in ("train", "val", "test")}
    by_kind: dict[tuple[str, str], list[dict]] = {}
    for t in CARRIER_TEMPLATES:
        by_kind.setdefault((t["kind"], t["polarity"]), []).append(t)
    for (kind, pol), items in by_kind.items():
        items = list(items)
        sel.shuffle(items)
        assert len(items) >= 3, f"need ≥3 {pol} templates for kind {kind} to cover all splits"
        out["val"][pol].append(items[0])
        out["test"][pol].append(items[1])
        out["train"][pol].extend(items[2:])
    for sp in ("train", "val", "test"):
        assert out[sp]["pos"] and out[sp]["neg"], f"split {sp} missing pos/neg templates"
    return out


def _generate_split(split, n, sg, shapes, pos_frac, sel, raw_dir, pool_dir) -> dict:
    n_pos = int(round(n * pos_frac))
    flags = [True] * n_pos + [False] * (n - n_pos)
    sel.shuffle(flags)

    records, cat_counter, shape_counter = [], Counter(), Counter()
    for positive in flags:
        shape = sel.choice(shapes)
        builder, rtype = SHAPES[shape]
        text, spans = builder(sg, positive)
        if rtype is None:  # shape B
            rtype = "intake_form" if positive else "facility_form"
        rec = {"text": text, "spans": spans,
               "contains_phi": 1 if spans else 0, "record_type": rtype}
        assert rec["contains_phi"] == (1 if positive else 0), "contains_phi mismatch"
        for s in spans:
            assert 0 <= s["start"] < s["end"] <= len(text), "span out of bounds"
            cat_counter[s["type"]] += 1
        shape_counter[shape] += 1
        records.append(rec)

    with open(raw_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (pool_dir / f"{split}_identifiers.txt").write_text(
        "\n".join(sorted(sg.identifiers)) + "\n", encoding="utf-8")
    (pool_dir / f"{split}_templates.txt").write_text(
        "\n".join(sorted(sg.template_ids)) + "\n", encoding="utf-8")

    print(f"[{split}] {n} records ({sum(flags)} pos / {n - sum(flags)} neg), "
          f"{len(sg.identifiers)} identifiers, {len(sg.template_ids)} templates")
    return {"n": n, "positives": sum(flags), "negatives": n - sum(flags),
            "identifiers": len(sg.identifiers), "templates": len(sg.template_ids),
            "by_category": dict(cat_counter), "by_shape": dict(shape_counter)}


def generate(version: str) -> None:
    cfg = load_config()
    seed = set_global_seed()
    total = cfg["data"][f"{version}_total"]
    pos_frac = cfg["data"]["positive_fraction"]
    fracs = cfg["data"]["split_fractions"]
    shapes = cfg["data"]["record_shapes"]

    sel = random.Random(seed + 1)
    gens = IdGen(seed)
    parts = partition_template_ids(sel, fracs)
    used_phi: set[str] = set()                          # global -> cross-split identifier disjoint

    raw_dir = Path(REPO_ROOT) / cfg["paths"]["data_raw"]
    pool_dir = Path(REPO_ROOT) / cfg["paths"]["data_pools"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        n = int(round(total * fracs[split]))
        sg = SplitGen(gens, sel, parts[split]["pos"], parts[split]["neg"], used_phi)
        summary[split] = _generate_split(split, n, sg, shapes, pos_frac, sel, raw_dir, pool_dir)

    # Dedicated HARD TEST SET (rules.md §3.5) — the split that decides the verdict. Uses the separate
    # hard bank (disjoint template IDs) + free-text shapes (A/C), weighted to hard positives (PHI in
    # terse/unusual positions, mixed with look-alikes) and hard negatives (dense look-alikes).
    if version == "v2":
        n_hard = cfg["data"]["hard_test_size"]
        sg = SplitGen(gens, sel, HARD_POS_TEMPLATES, HARD_NEG_TEMPLATES, used_phi)
        summary["hard_test"] = _generate_split("hard_test", n_hard, sg, ["A", "C"],
                                               pos_frac, sel, raw_dir, pool_dir)

    _write_summary(cfg, version, seed, summary)


def _write_summary(cfg, version, seed, summary) -> None:
    day = "Day 6" if version == "v2" else "Day 2"
    target = cfg["data"]["min_positives_per_category"]
    types = cfg["label_types"]
    lines = [f"# {day} — Synthetic Data Summary ({version})\n",
             f"- seed: `{seed}`  | label types: `{len(types)}`  | "
             f"per-category target (train): ≥{target}",
             f"- shapes: {cfg['data']['record_shapes']}  | positive fraction: "
             f"{cfg['data']['positive_fraction']}\n"]
    agg: Counter = Counter()
    for sp, s in summary.items():
        lines.append(f"## {sp}")
        lines.append(f"- records: {s['n']} ({s['positives']} pos / {s['negatives']} neg)")
        lines.append(f"- unique poolable identifiers: {s['identifiers']}  | "
                     f"carrier template IDs: {s['templates']}")
        lines.append(f"- by shape: {s['by_shape']}")
        lines.append(f"- positive spans by category: "
                     f"{dict(sorted(s['by_category'].items()))}\n")
        agg.update(s["by_category"])

    # Per-category coverage table across splits + target check on train.
    lines.append("## Per-category positive coverage by split\n")
    lines.append("| category | " + " | ".join(summary.keys()) + " |")
    lines.append("|" + "---|" * (len(summary) + 1))
    for t in types:
        cells = [str(summary[sp]["by_category"].get(t, 0)) for sp in summary]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines.append("")
    missing = [t for t in types if t not in agg]
    lines.append(f"- categories with zero positives anywhere: {missing or 'none'}")
    if "train" in summary:
        short = [t for t in types if summary["train"]["by_category"].get(t, 0) < target]
        lines.append(f"- train categories below the ≥{target} target: {short or 'none'}")
    out = Path(REPO_ROOT) / "reports" / f"{'day6' if version == 'v2' else 'day2'}_data_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Insertion-based synthetic PHI/PII data generator.")
    ap.add_argument("--version", default="v1", choices=["v1", "v2"],
                    help="composition target in config.yaml (v1 quick set, v2 full scale).")
    args = ap.parse_args()
    generate(args.version)


if __name__ == "__main__":
    main()
