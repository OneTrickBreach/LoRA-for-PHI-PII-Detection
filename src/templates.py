"""Carrier templates for record shapes A/B/C (plan.md §6). Implemented Day 2.

Templates are CLEAN (no PHI). The generator inserts identifiers / look-alikes at known
offsets afterward (insertion strategy, rules.md §1.2-1.3). Each template has a stable
template_id used for entity/template-level split disjointness (rules.md §3).
  A: generic API request with free-text payload
  B: structured intake form (named fields)
  C: semi-structured log line
"""
# TODO(Day 2): define template banks per shape with stable template_ids.
