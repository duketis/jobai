"""Per-source schema-change detection.

When a source's payload shape drifts (a renamed JSON key, a field that
used to be populated and now arrives null, an entire column dropped),
the parser quietly produces :class:`NormalizedJob` instances with
``None`` where there used to be data. Without detection, this looks
healthy from a row-count perspective — fresh listings keep flowing —
but the agent's ability to answer questions ("which jobs are remote?")
silently degrades.

Approach: each scrape run computes a per-field presence count
(``how many jobs had a non-null value for this field``). The run
writes that map to ``scrape_runs.field_stats_json``; the next run
compares against the previous successful run and flags fields whose
null-rate jumped beyond a threshold.

Threshold rather than absolute count so a small run (5 listings on a
quiet weekend) doesn't drown out signal — we look at the *fraction*,
not the raw delta.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from jobai.sources.base import NormalizedJob

#: Subset of NormalizedJob fields whose null-rate matters for schema
#: tracking. ``raw_data`` and ``extra_tags`` are excluded — they're
#: unstructured payload, not part of the parser's surface contract.
TRACKED_FIELDS: tuple[str, ...] = (
    "title",
    "company",
    "apply_url",
    "location_raw",
    "location_country",
    "location_city",
    "remote_type",
    "employment_type",
    "posted_at",
    "salary_min",
    "salary_max",
    "salary_currency",
    "description_text",
    "description_html",
)


@dataclass(frozen=True, slots=True)
class FieldStats:
    """Per-field presence counts for one scrape run.

    ``total`` is the number of jobs in the run; ``present`` maps each
    tracked field to how many jobs had a non-null value for it. The
    null-rate is computed on demand to keep the dataclass JSON-clean.
    """

    total: int
    present: dict[str, int] = field(default_factory=dict)

    def null_rate(self, field_name: str) -> float:
        if self.total == 0:
            return 0.0
        return 1.0 - (self.present.get(field_name, 0) / self.total)

    def to_json(self) -> str:
        return json.dumps({"total": self.total, "present": self.present}, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str | None) -> FieldStats | None:
        """Re-hydrate from a ``scrape_runs.field_stats_json`` value.

        Returns ``None`` for missing or malformed input — comparison
        callers should treat that as "no baseline, no detection".
        """
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        total = data.get("total")
        present = data.get("present")
        if not isinstance(total, int) or not isinstance(present, dict):
            return None
        coerced = {str(k): int(v) for k, v in present.items() if isinstance(v, int)}
        return cls(total=total, present=coerced)


@dataclass(frozen=True, slots=True)
class FieldChange:
    """A single field whose null-rate changed beyond the threshold."""

    field: str
    prev_null_rate: float
    curr_null_rate: float
    delta: float
    prev_total: int
    curr_total: int


def compute_field_stats(jobs: Iterable[NormalizedJob]) -> FieldStats:
    """Walk a NormalizedJob iterable and return its presence counts."""
    total = 0
    present: dict[str, int] = dict.fromkeys(TRACKED_FIELDS, 0)
    for job in jobs:
        total += 1
        for field_name in TRACKED_FIELDS:
            value = getattr(job, field_name, None)
            if value not in (None, ""):
                present[field_name] += 1
    return FieldStats(total=total, present=present)


def update_stats(stats: FieldStats, job: NormalizedJob) -> FieldStats:
    """Return a new :class:`FieldStats` incrementing for one job.

    Lets the runner accumulate stats during async iteration without
    holding the entire job list in memory.
    """
    new_present = dict(stats.present) if stats.present else dict.fromkeys(TRACKED_FIELDS, 0)
    for field_name in TRACKED_FIELDS:
        value = getattr(job, field_name, None)
        if value not in (None, ""):
            new_present[field_name] = new_present.get(field_name, 0) + 1
    return dataclasses.replace(stats, total=stats.total + 1, present=new_present)


def empty_stats() -> FieldStats:
    """Return a zero-count FieldStats with all tracked fields registered."""
    return FieldStats(total=0, present=dict.fromkeys(TRACKED_FIELDS, 0))


#: Minimum jobs in either run before we trust the comparison. Two
#: jobs going from "100 % present" to "0 % present" is just noise on
#: a thin run; require a baseline volume before alerting.
DEFAULT_MIN_VOLUME = 5

#: Default null-rate jump that constitutes a schema change (30%).
DEFAULT_THRESHOLD = 0.30


def detect_changes(
    prev: FieldStats | None,
    curr: FieldStats,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_volume: int = DEFAULT_MIN_VOLUME,
) -> list[FieldChange]:
    """Compare two runs and return fields whose null-rate jumped.

    Returns an empty list when ``prev`` is ``None`` (first run for the
    source) or when either run has fewer than ``min_volume`` jobs.
    """
    if prev is None:
        return []
    if prev.total < min_volume or curr.total < min_volume:
        return []

    changes: list[FieldChange] = []
    for field_name in TRACKED_FIELDS:
        prev_rate = prev.null_rate(field_name)
        curr_rate = curr.null_rate(field_name)
        delta = curr_rate - prev_rate
        if delta >= threshold:
            changes.append(
                FieldChange(
                    field=field_name,
                    prev_null_rate=prev_rate,
                    curr_null_rate=curr_rate,
                    delta=delta,
                    prev_total=prev.total,
                    curr_total=curr.total,
                ),
            )
    return changes


__all__: list[str] = [
    "DEFAULT_MIN_VOLUME",
    "DEFAULT_THRESHOLD",
    "TRACKED_FIELDS",
    "FieldChange",
    "FieldStats",
    "compute_field_stats",
    "detect_changes",
    "empty_stats",
    "update_stats",
]
