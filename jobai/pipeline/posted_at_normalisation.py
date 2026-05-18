"""Normalise ``posted_at`` into ISO-8601 UTC for every canonical job.

The boards are wildly inconsistent about what they put in a job's
"posted" field. The ATS APIs (Greenhouse / Lever / Ashby /
SmartRecruiters / Workable) hand us real ISO-8601 timestamps. Seek
hands us its UI label — ``"8d ago"``. Indeed hands us
``formattedRelativeTime`` — ``"4 days ago"`` / ``"Just posted"`` /
``"30+ days ago"``. The AU state-gov scrapers hand us whatever text
sat in a "Listed On" table cell — often ``"14/05/2026"``.

Before this module that raw text flowed straight into
``jobs.posted_at``, which broke two things:

* **Sorting.** ``ORDER BY posted_at DESC`` is a *lexical* string
  sort. Over ``"Just posted"`` / ``"9d ago"`` / ``"4 days ago"`` it
  produces nonsense (``'J' > '9' > '4'``), so "Newest posted"
  returned an arbitrary order.
* **Display.** The frontend's relative-time formatter does
  ``new Date(posted_at)``; ``new Date("8d ago")`` is *Invalid Date*,
  so it fell back to printing the raw string verbatim — Seek rows
  showed ``"8d ago"`` while Indeed rows showed ``"4 days ago"``.

The contract is now: a canonical job's ``posted_at`` is **either an
ISO-8601 UTC timestamp or NULL — never free text**. NULL sorts
``NULLS LAST`` (honest "we don't know") instead of polluting the
order. This mirrors the ``remote_type`` safety net in
:mod:`jobai.pipeline.remote_inference`: applied at canonicalisation
inside the runner, with a backfill pass for rows already in the DB.

The parser is deterministic and offline — relative strings are
resolved against an injected reference instant (scrape time for live
canonicalisation, the row's ``first_seen_at`` for backfill, because
that's when "8d ago" was actually true).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

_log = logging.getLogger(__name__)

#: Phrases every board uses for "this just went up". All map to the
#: reference instant.
_FRESH_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "just now",
        "just posted",
        "just listed",
        "just added",
        "posted today",
        "today",
        "new",
        "new!",
    }
)

#: Leading verbs the boards prepend ("Posted 3 days ago",
#: "Listed 14/05/2026"). Stripped before any other parsing.
_LEADING_VERB: Final[re.Pattern[str]] = re.compile(
    r"^(?:posted|listed|added|updated|created)\s+",
    re.IGNORECASE,
)

#: Seconds per relative unit. Months/years are deliberate
#: approximations — at month granularity the exact day doesn't move
#: a "newest first" ordering.
_UNIT_SECONDS: Final[dict[str, int]] = {
    "minute": 60,
    "hour": 3_600,
    "day": 86_400,
    "week": 7 * 86_400,
    "month": 30 * 86_400,
    "year": 365 * 86_400,
}

#: Maps every shorthand/longhand the boards use onto a canonical unit
#: key in :data:`_UNIT_SECONDS`. Longest spellings first so the regex
#: never settles on a prefix ("mo" must beat "m" for "1mo ago").
_UNIT_ALIASES: Final[tuple[tuple[str, str], ...]] = (
    ("minutes", "minute"),
    ("minute", "minute"),
    ("mins", "minute"),
    ("min", "minute"),
    ("months", "month"),
    ("month", "month"),
    ("mon", "month"),
    ("mo", "month"),
    ("hours", "hour"),
    ("hour", "hour"),
    ("hrs", "hour"),
    ("hr", "hour"),
    ("weeks", "week"),
    ("week", "week"),
    ("wks", "week"),
    ("wk", "week"),
    ("days", "day"),
    ("day", "day"),
    ("years", "year"),
    ("year", "year"),
    ("yrs", "year"),
    ("yr", "year"),
    ("m", "minute"),
    ("h", "hour"),
    ("d", "day"),
    ("w", "week"),
    ("y", "year"),
)

_RELATIVE: Final[re.Pattern[str]] = re.compile(
    r"^(\d+)\s*\+?\s*(" + "|".join(alias for alias, _ in _UNIT_ALIASES) + r")\s*(?:ago)?$",
    re.IGNORECASE,
)

#: AU-convention absolute date formats from the state-gov table cells.
#: Day-first — never month-first — because that's the AU norm and the
#: scrapers all read government sites.
_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d %B %Y",
)


def _to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as an aware UTC datetime (naive → assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_absolute(value: str) -> datetime | None:
    """Parse an epoch / ISO-8601 / AU-date string, else ``None``."""
    if value.isdigit():
        # Indeed's ``createDate`` can be epoch. 13 digits = millis,
        # 10 = seconds; other lengths aren't a plausible epoch (a
        # bare year etc.) so we don't guess.
        if len(value) == 13:
            return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
        if len(value) == 10:
            return datetime.fromtimestamp(int(value), tz=UTC)
        return None
    try:
        return _to_utc(datetime.fromisoformat(value))
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return _to_utc(datetime.strptime(value, fmt))
        except ValueError:
            continue
    return None


def _parse_relative(lowered: str, reference: datetime) -> datetime | None:
    """Resolve a relative label ("8d ago") to an absolute instant."""
    if lowered in _FRESH_PHRASES:
        return reference
    if lowered == "yesterday":
        return reference - timedelta(days=1)
    match = _RELATIVE.match(lowered)
    if match is None:
        return None
    quantity = int(match.group(1))
    unit_key = next(canon for alias, canon in _UNIT_ALIASES if alias == match.group(2))
    return reference - timedelta(seconds=quantity * _UNIT_SECONDS[unit_key])


def normalise_posted_at(raw: str | None, *, now: datetime) -> str | None:
    """Normalise a source's ``posted_at`` string to ISO-8601 UTC.

    Returns an ISO-8601 UTC string, or ``None`` when the input is
    empty or unparseable. ``now`` is the reference instant relative
    strings ("8d ago") are measured back from; it must be timezone
    aware. Idempotent: feeding a value this function produced back in
    returns the same value.
    """
    text = raw.strip() if raw is not None else ""
    if not text:
        return None

    absolute = _parse_absolute(text)
    if absolute is not None:
        return absolute.isoformat()

    lowered = _LEADING_VERB.sub("", text).strip().lower()
    relative = _parse_relative(lowered, _to_utc(now))
    return relative.isoformat() if relative is not None else None


@dataclass(frozen=True, slots=True)
class PostedAtBackfillResult:
    """Summary returned by :func:`backfill_posted_at`."""

    inspected: int
    updated: int
    parsed: int
    nulled: int


def backfill_posted_at(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    limit: int | None = None,
) -> PostedAtBackfillResult:
    """Re-normalise ``posted_at`` for every row that has a value.

    Relative strings are resolved against the row's ``first_seen_at``
    (when "8d ago" was actually true), falling back to ``now`` (or
    the current time) when ``first_seen_at`` itself won't parse.
    Already-canonical ISO values are left byte-for-byte unchanged, so
    repeated runs are no-ops. Unparseable text is nulled — NULL sorts
    last, raw garbage sorts randomly.
    """
    fallback = _to_utc(now) if now is not None else datetime.now(UTC)
    sql = (
        "SELECT id, posted_at, first_seen_at FROM jobs "
        "WHERE posted_at IS NOT NULL AND posted_at != '' "
        "ORDER BY last_seen_at DESC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()

    updated = parsed = nulled = 0
    for row in rows:
        job_id = int(row[0])
        current = row[1]
        ref = _parse_absolute(str(row[2])) if row[2] is not None else None
        normalised = normalise_posted_at(current, now=ref or fallback)
        if normalised is not None:
            parsed += 1
        if normalised == current:
            continue
        conn.execute(
            "UPDATE jobs SET posted_at = ? WHERE id = ?",
            (normalised, job_id),
        )
        updated += 1
        if normalised is None:
            nulled += 1

    if updated:
        conn.commit()
    _log.info(
        "posted_at_backfill",
        extra={
            "inspected": len(rows),
            "updated": updated,
            "parsed": parsed,
            "nulled": nulled,
        },
    )
    return PostedAtBackfillResult(
        inspected=len(rows),
        updated=updated,
        parsed=parsed,
        nulled=nulled,
    )


__all__: list[str] = [
    "PostedAtBackfillResult",
    "backfill_posted_at",
    "normalise_posted_at",
]
