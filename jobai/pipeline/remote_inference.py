"""Infer ``remote_type`` for jobs whose source didn't tell us.

Every canonical job needs to land in exactly one of
``remote`` / ``hybrid`` / ``onsite`` so the agent and the UI's
remote-mode filter aren't full of nulls. Most ATS APIs surface a
remote flag (Greenhouse/Lever/Ashby), the AU state-gov sources
sometimes do, and the federal APS feed always does — but plenty of
listings come through without one, especially when the description
is on a detail page we haven't backfilled yet.

This module is the safety net. It runs a deterministic keyword
scan over title + description + location and resolves to a single
value, defaulting to ``onsite`` when nothing in the text is
explicit (the conservative bet — assume in-office unless the
listing says otherwise).

The heuristic is intentionally cheap and offline. We could ask an
LLM to classify each row but at 8k+ jobs per cycle the latency
and token cost wouldn't justify the small accuracy lift over a
well-tuned regex pass on the modal AU phrasing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

_log = logging.getLogger(__name__)

RemoteType = Literal["remote", "hybrid", "onsite"]

#: Default when none of the patterns match. Most listings without an
#: explicit work-mode signal are in-office; assuming onsite minimises
#: false positives in the ``remote=true`` filter (the highest-stakes
#: query for users who actually need remote).
_DEFAULT: Final[RemoteType] = "onsite"

#: Patterns evaluated in priority order: a ``remote`` hit always wins
#: over a ``hybrid`` hit (an explicit "fully remote" listing that also
#: mentions "hybrid working as a backup" is still a remote role), and
#: ``hybrid`` wins over ``onsite``. Each pattern is a compiled regex
#: so the per-job cost is one walk through the text per category.
#:
#: Word boundaries ``\b`` keep us from false-matching inside other
#: words (``remoter`` shouldn't trigger; ``onsite`` and ``on site``
#: are spelled both ways in the wild so we match either).
_REMOTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfully\s+remote\b", re.IGNORECASE),
    re.compile(r"\b100\s*%\s*remote\b", re.IGNORECASE),
    re.compile(r"\bremote[\s\-]first\b", re.IGNORECASE),
    # ``remote\s+work`` is intentionally NOT in the noun list — it
    # appears in negated form ("no remote work available") often
    # enough to be a false-positive trap. ``role/position/...`` are
    # safer because nobody writes "no remote position".
    re.compile(r"\bremote\s+(?:role|position|opportunity|job)\b", re.IGNORECASE),
    re.compile(r"\b(?:work|working)\s+from\s+(?:home|anywhere)\b", re.IGNORECASE),
    re.compile(r"\bwfh\b", re.IGNORECASE),
    re.compile(r"\banywhere\s+in\s+(?:australia|the\s+country|aus)\b", re.IGNORECASE),
    re.compile(r"\b(?:work(?:ing)?|based)\s+remotely\b", re.IGNORECASE),
)

_HYBRID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhybrid\b", re.IGNORECASE),
    re.compile(r"\bflexible\s+(?:work|working|arrangement|location)", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:days?|x)\s*(?:per\s+week\s+)?in\s+(?:the\s+)?office\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:days?|x)\s*from\s+home\b", re.IGNORECASE),
    re.compile(r"\bblended\s+(?:work|working|arrangement)", re.IGNORECASE),
    re.compile(r"\bsplit\s+(?:between|across)\b.*\boffice\b", re.IGNORECASE),
)

_ONSITE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bon[\s\-]?site\b", re.IGNORECASE),
    re.compile(r"\bin[\s\-]?office\b", re.IGNORECASE),
    re.compile(r"\boffice[\s\-]?based\b", re.IGNORECASE),
    re.compile(r"\bin[\s\-]?person\b", re.IGNORECASE),
    re.compile(r"\bmust\s+be\s+(?:located|based)\s+in\b", re.IGNORECASE),
    re.compile(r"\bbased\s+(?:in|at)\s+our\s+\w+\s+office\b", re.IGNORECASE),
    re.compile(r"\bfull[\s\-]time\s+in\s+(?:the\s+)?office\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One row's text payload for the heuristic to chew on."""

    title: str | None
    description: str | None
    location: str | None

    def haystack(self) -> str:
        """Concatenate the available text fields with newline gaps.

        Newlines stop "remote" in a title from running into a "based
        in our Sydney office" phrase in the description and tripping
        a multi-token regex that wouldn't actually fire on either
        field alone.
        """
        return "\n".join(s for s in (self.title, self.description, self.location) if s)


def infer_remote_type(
    *,
    title: str | None,
    description: str | None = None,
    location: str | None = None,
    default: RemoteType = _DEFAULT,
) -> RemoteType:
    """Best-guess work-mode classification from free-text fields.

    Args:
        title: job title.
        description: free-text description (HTML-stripped or raw —
            the regex passes work fine on either).
        location: ``location_raw`` from the canonical row. A
            comma-separated multi-city string nudges hybrid only if
            no explicit pattern fired earlier.
        default: returned when nothing matches. Defaults to
            ``onsite`` so a missing signal doesn't pollute the
            ``remote=true`` filter.
    """
    candidate = _Candidate(title=title, description=description, location=location)
    haystack = candidate.haystack()
    if not haystack:
        return default

    if _any_match(_REMOTE_PATTERNS, haystack):
        return "remote"
    if _any_match(_HYBRID_PATTERNS, haystack):
        return "hybrid"
    if _any_match(_ONSITE_PATTERNS, haystack):
        return "onsite"

    # Fallback hint: a multi-city ``location_raw`` (e.g. APS Jobs'
    # "Adelaide SA, Brisbane QLD, Canberra ACT, …") almost always
    # means a distributed / hybrid role. Single-city locations don't
    # signal anything either way, so they fall through to ``default``.
    if location and location.count(",") >= 2:
        return "hybrid"

    return default


def _any_match(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) is not None for p in patterns)


@dataclass(frozen=True, slots=True)
class RemoteBackfillResult:
    """Summary returned by :func:`backfill_remote_types`."""

    inspected: int
    updated: int
    by_value: dict[RemoteType, int]


def backfill_remote_types(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> RemoteBackfillResult:
    """Walk jobs whose ``remote_type`` is null/empty and infer one.

    Updates each row in place; returns a small summary so callers
    (CLI, scheduler) can log how many landed in each bucket. Pass
    ``limit`` to bound a single pass — handy for the scheduler tick
    that doesn't want to monopolise the connection.
    """
    sql = (
        "SELECT id, title, description_text, location_raw "
        "FROM jobs "
        "WHERE remote_type IS NULL OR remote_type = '' "
        "ORDER BY last_seen_at DESC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()

    by_value: dict[RemoteType, int] = {"remote": 0, "hybrid": 0, "onsite": 0}
    updated = 0
    for row in rows:
        job_id = int(row[0])
        inferred = infer_remote_type(
            title=row[1],
            description=row[2],
            location=row[3],
        )
        conn.execute(
            "UPDATE jobs SET remote_type = ? WHERE id = ?",
            (inferred, job_id),
        )
        by_value[inferred] += 1
        updated += 1

    if updated:
        conn.commit()
    _log.info(
        "remote_type_backfill",
        extra={
            "inspected": len(rows),
            "updated": updated,
            **{f"to_{k}": v for k, v in by_value.items()},
        },
    )
    return RemoteBackfillResult(inspected=len(rows), updated=updated, by_value=by_value)


__all__: list[str] = [
    "RemoteBackfillResult",
    "RemoteType",
    "backfill_remote_types",
    "infer_remote_type",
]
