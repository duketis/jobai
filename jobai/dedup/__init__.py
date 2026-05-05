"""Deduplication pipeline.

Two passes (per ARCHITECTURE §4.9):

* **Deterministic** — a SHA-256 dedup key over normalised
  ``(company, title, location_country)``. Cheap, runs on every upsert.
* **Fuzzy** — rapidfuzz token-set ratio between titles within
  ``(company_norm, location_country)`` groups. Catches "Senior
  Software Engineer" vs "Sr. Software Engineer". Runs as a separate
  reconciliation pass.

The promotion module is what the runner calls to write a
:class:`~jobai.sources.base.NormalizedJob` into the canonical ``jobs``
table and the ``job_sources`` join. The reconcile module is the
maintenance pass that merges fuzzy duplicates after the fact.
"""

from __future__ import annotations
