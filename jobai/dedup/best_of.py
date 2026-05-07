"""Cross-source field merger.

When the same job is scraped from multiple platforms (e.g. Atlassian's
"Senior Python Engineer" surfacing on both Greenhouse and Indeed), the
canonical row should reflect the **best** information across all
sources, not whichever happened to scrape last.

The pre-merger pipeline used SQL ``COALESCE(?, existing)`` which is
``new wins if non-null, else keep old``. That had two failure modes:

1. A source emitting ``None`` for a field it doesn't track wouldn't
   erase data — good — but a source emitting a *different* non-null
   value would silently override the first reading. So a 200-char
   Indeed teaser could overwrite Greenhouse's full multi-paragraph
   description.
2. The fuzzy reconcile pass (``jobai.dedup.reconcile``) merged near-
   duplicates by transferring ``job_sources`` links and deleting the
   loser, but it kept only the survivor's field values — so any data
   the duplicate uniquely had (a salary the survivor lacked, a richer
   description) was lost on every reconcile pass.

This module fixes both: the per-field rules below pick the best value
across two field-sets, and both ``promote.py`` and ``reconcile.py``
delegate to ``merge_canonical_fields``.

The rules are conservative: they prefer **stability** for categorical
fields (don't flip the canonical value back and forth across scrape
cycles) and **richness** for free-text fields (longer description
beats shorter). Salary is treated as stable: the first non-null value
wins, because we don't have signal to judge which of two non-null
salaries is more correct.
"""

from __future__ import annotations

from collections.abc import Iterable


def first_non_null[T](values: Iterable[T | None]) -> T | None:
    """Return the first ``value is not None``; ``None`` if all are None.

    Distinct from "first truthy": ``0``, ``False``, and ``""`` count as
    present. The merger relies on this — ``salary_min == 0`` is a real
    (if unusual) signal, not an absence.
    """
    for v in values:
        if v is not None:
            return v
    return None


def first_non_empty(values: Iterable[str | None]) -> str | None:
    """Return the first ``value`` that is neither ``None`` nor ``""``.

    Some sources emit ``""`` rather than ``None`` for missing string
    fields (e.g. an empty ``apply_url`` element in a JSON payload),
    so treating empty-string as missing keeps the canonical row clean.
    """
    for v in values:
        if v is not None and v != "":
            return v
    return None


def longest(values: Iterable[str | None]) -> str | None:
    """Return the longest non-empty string; ``None`` if no candidates qualify.

    Ties resolve to the first occurrence — stable so repeated scrapes
    don't flip the canonical value between equally-long alternatives.
    """
    best: str | None = None
    for v in values:
        if v is None or v == "":
            continue
        if best is None or len(v) > len(best):
            best = v
    return best


def earliest(values: Iterable[str | None]) -> str | None:
    """Return the lexicographically smallest non-empty string.

    Used for ``posted_at``: ISO-8601 timestamps with consistent
    timezone formatting sort lexically the same as chronologically,
    so we don't need to parse them. Boards occasionally backdate or
    re-list a role with a fresher timestamp; we want the original.
    """
    valid = [v for v in values if v is not None and v != ""]
    return min(valid) if valid else None


# Fields the merger owns. The caller is responsible for ``id``,
# ``dedup_key``, ``first_seen_at``, ``last_seen_at``, ``closed_at``,
# ``fingerprint_json``, ``title``, ``company``, ``company_norm`` —
# those have their own rules (immutable identity / scrape-bookkeeping).
_MERGEABLE_FIELDS: tuple[str, ...] = (
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
    "apply_url",
)


def mergeable_fields() -> tuple[str, ...]:
    """The set of column names ``merge_canonical_fields`` returns.

    Exposed so callers (promote, reconcile) can build the UPDATE
    statement from a single source of truth without hard-coding the
    list in two places.
    """
    return _MERGEABLE_FIELDS


def merge_canonical_fields(
    old: dict[str, object | None],
    new: dict[str, object | None],
) -> dict[str, object | None]:
    """Merge two field-sets representing the same canonical job.

    ``old`` is the existing canonical state (from the DB on update,
    or the survivor row in fuzzy reconcile). ``new`` is the incoming
    state (from a fresh scrape, or the duplicate in fuzzy reconcile).

    The ordering matters for ``first_non_null`` and ``first_non_empty``
    rules — they encode "preserve the original value" — so callers
    must pass the older / preferred snapshot as ``old``.
    """
    return {
        # Free-text location: pick the richer string.
        "location_raw": longest([_s(old, "location_raw"), _s(new, "location_raw")]),
        # Categorical / structured location: stable, first non-null wins.
        "location_country": first_non_null(
            [old.get("location_country"), new.get("location_country")]
        ),
        "location_city": first_non_null([old.get("location_city"), new.get("location_city")]),
        # Categorical with heuristic inference behind it — don't flip.
        "remote_type": first_non_null([old.get("remote_type"), new.get("remote_type")]),
        "employment_type": first_non_null([old.get("employment_type"), new.get("employment_type")]),
        # Earliest posting date wins (boards re-list with fresh timestamps).
        "posted_at": earliest([_s(old, "posted_at"), _s(new, "posted_at")]),
        # Salary: stable, first non-null wins. Don't second-guess.
        "salary_min": first_non_null([old.get("salary_min"), new.get("salary_min")]),
        "salary_max": first_non_null([old.get("salary_max"), new.get("salary_max")]),
        "salary_currency": first_non_null([old.get("salary_currency"), new.get("salary_currency")]),
        # Description: pick the richest available, independently for
        # text and html so a source emitting only one doesn't blank
        # the other.
        "description_text": longest([_s(old, "description_text"), _s(new, "description_text")]),
        "description_html": longest([_s(old, "description_html"), _s(new, "description_html")]),
        # apply_url: keep the existing one if set; only fill it in.
        # Per-source URLs live in ``job_sources`` so the canonical row
        # only needs *some* working URL.
        "apply_url": first_non_empty([_s(old, "apply_url"), _s(new, "apply_url")]),
    }


def _s(row: dict[str, object | None], key: str) -> str | None:
    """Return ``row[key]`` cast to ``str | None`` for the string-typed
    helpers (``longest``, ``earliest``, ``first_non_empty``).

    The mergeable fields are typed at the schema level, but ``row`` is
    a generic ``dict[str, object | None]`` because it round-trips
    through sqlite. This narrows safely without ``cast`` noise at every
    call site.
    """
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
