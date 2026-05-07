"""Infer ``salary_min`` / ``salary_max`` / ``salary_currency`` from free text.

Most AU listings carry the salary in the description body rather than a
structured field — councils write ``Band 8 - $123,558 to $138,752 + super``,
recruiters drop ``$120k - $160k`` into the title, scale-ups bury the
range mid-paragraph. The ATS sources (Greenhouse, Lever, Ashby) rarely
populate the structured ``compensation`` block. The result before this
module: ~22% of canonical jobs had any salary signal at all.

This pass is regex-only inference. Three guards keep it conservative:

1. **Salary-keyword adjacency.** Each ``$X`` candidate must be next to a
   salary marker (``Salary:``, ``Compensation:``, ``per annum``,
   ``+ super``, ``Band N -``). A bare dollar amount with no nearby
   marker is rejected — that's how we filter out funding raises, AUM
   numbers, and product-pricing mentions.
2. **Negative-context veto.** A small set of specific patterns
   (``raised $X``, ``backed by $X``, ``AUM`` adjacent to ``$``,
   ``Series A round of $X``) reject the match outright even if a
   salary keyword is also nearby.
3. **Bounds + period.** Annual-only — hourly / daily contractor rates
   are rejected because extrapolating ``$700/day`` to ``$154k/year``
   conflates contract vs perm. Numbers outside ``[$10k, $5M]`` are
   noise.

Pattern after :mod:`jobai.pipeline.remote_inference`: a pure
``infer_salary`` function for the runner's per-job hot path, and a
``backfill_salaries`` batch pass for the CLI / scheduler tick.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# Plausible annual salary band. Reject anything outside; cheaper than
# trying to enumerate every false-positive context.
_MIN_PLAUSIBLE_ANNUAL = 10_000
_MAX_PLAUSIBLE_ANNUAL = 5_000_000

# A ``$`` optionally preceded by a US-currency marker. Used as the
# leading anchor for both range and single-value patterns so
# ``USD $120,000`` and ``US$120,000`` parse like the bare ``$120,000``.
_DOLLAR = r"(?:US\s*\$|USD\s*\$|U\.S\.\s*\$|\$)"

_NUM_PATTERN = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"

# ``$X``, ``$100,000``, ``$100K``, ``$100k``, ``USD $100``, ``US$100``.
_AMOUNT_PATTERN = re.compile(
    rf"{_DOLLAR}\s*({_NUM_PATTERN})\s*([Kk])?",
)

# ``$X - $Y`` / ``$X to $Y`` / ``$X<U+2013>$Y``. Either side may be K-suffixed
# independently; a trailing K on the max with no K on the min is
# common shorthand (``$100-$150K``) - applied at parse time.
# We accept hyphen-minus, U+2013 EN DASH and U+2014 EM DASH as range
# separators because real listings use all three.
_RANGE_SEPARATOR = r"(?:-|–|—|to)"
_RANGE_PATTERN = re.compile(
    rf"""
    {_DOLLAR}\s*(?P<min_num>{_NUM_PATTERN})\s*(?P<min_k>[Kk])?
    \s*{_RANGE_SEPARATOR}\s*
    {_DOLLAR}?\s*(?P<max_num>{_NUM_PATTERN})\s*(?P<max_k>[Kk])?
    """,
    re.VERBOSE,
)

# Salary-keyword + dollar in the same clause. The keyword is REQUIRED
# to be followed by a colon - that's the structural marker that
# separates a label ("Compensation: $X") from prose ("attractive
# compensation package"). Many sources emit description_text with HTML
# stripped down to glued strings like "timeCompensation:$180K", so we
# don't anchor on ``\b`` before the keyword - the trailing colon does
# the work.
#
# Between the colon and the ``$`` we allow up to 80 chars of arbitrary
# text but NOT a sentence terminator (``.`` or newline). That covers
# real-world phrasings like "Compensation: Our midpoint TTR for this
# role is $236,500" without crossing into the next sentence.
_KEYWORD_TO_DOLLAR_GAP = r"\s*:[^.\n]{0,80}?(?:US|USD|U\.S\.)?\s*"
_SALARY_ADJACENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"salar(?:y|ies){_KEYWORD_TO_DOLLAR_GAP}\$", re.IGNORECASE),
    re.compile(rf"compensation{_KEYWORD_TO_DOLLAR_GAP}\$", re.IGNORECASE),
    re.compile(rf"remuneration{_KEYWORD_TO_DOLLAR_GAP}\$", re.IGNORECASE),
    re.compile(rf"(?:base\s+)?package{_KEYWORD_TO_DOLLAR_GAP}\$", re.IGNORECASE),
    re.compile(rf"OTE{_KEYWORD_TO_DOLLAR_GAP}\$"),  # case-sensitive
    re.compile(rf"pay{_KEYWORD_TO_DOLLAR_GAP}\$", re.IGNORECASE),
    re.compile(rf"\bband\s+\d+\s*[-:]?\s*{_DOLLAR}", re.IGNORECASE),
    # Suffix patterns: ``$X per annum`` / ``$X + super`` / ``$X /yr``.
    # These don't need a leading colon because the SUFFIX is the label.
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,60}\bper\s+annum\b", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,60}\bper\s+year\b", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,60}\bp\.?\s*a\.?\b", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,60}\bannually\b", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,60}\+\s*super", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,60}\bplus\s+super", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,30}/\s*yr\b", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKk\-–to$US\s]{0,30}/\s*year\b", re.IGNORECASE),
)

# Negative context — specific anchored patterns. Rejecting on a bare
# keyword ("investors") is too aggressive (it kills "Salary: $X +
# investors equity"). Anchoring to ``$`` adjacency makes the rejection
# precise.
_NEGATIVE_ADJACENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\braising?\s+\$", re.IGNORECASE),
    re.compile(r"\braised\s+\$", re.IGNORECASE),
    re.compile(r"\braise\s+of\s+\$", re.IGNORECASE),
    re.compile(r"\bbacked\s+by\s+(?:nearly\s+|over\s+)?\$", re.IGNORECASE),
    re.compile(r"\bvalu(?:ed|ation)\s+(?:at\s+)?\$", re.IGNORECASE),
    re.compile(r"\bAUM(?:\s+of)?\s+\$"),  # ``AUM $X`` / ``AUM of $X``
    re.compile(r"\$[\d,\.\sKkMmBb]{0,12}\s*AUM\b"),  # ``$X AUM``
    re.compile(r"\bARR\s+\$"),
    re.compile(r"\$[\d,\.\sKkMmBb]{0,12}\s*ARR\b"),
    re.compile(r"\$[\d,\.\sKkMmBb]{0,12}\s+(?:in\s+)?revenue\b", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKkMmBb]{0,12}\s+in\s+(?:the\s+)?transaction\s+flow", re.IGNORECASE),
    re.compile(r"\b(?:Series\s+[A-G]\s+(?:round\s+of\s+)?)\$", re.IGNORECASE),
    re.compile(r"\$[\d,\.\sKkMmBb]{0,12}\s+(?:in\s+)?Series\s+[A-G]\b"),
    re.compile(r"\$[\d,\.\sKkMmBb]{0,12}\s+(?:in\s+)?funding\b", re.IGNORECASE),
    re.compile(r"\bfunding\s+of\s+\$", re.IGNORECASE),
)

# Period markers — if any appears within a tight suffix window, the
# rate is hourly/daily/weekly and we reject (annual extrapolation
# conflates contract vs perm).
_NON_ANNUAL_PERIOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bper\s+hour\b", re.IGNORECASE),
    re.compile(r"\bper\s+day\b", re.IGNORECASE),
    re.compile(r"\bper\s+week\b", re.IGNORECASE),
    re.compile(r"\bhourly\b", re.IGNORECASE),
    re.compile(r"\bdaily\b", re.IGNORECASE),
    re.compile(r"\bweekly\b", re.IGNORECASE),
    re.compile(r"/\s*hr\b", re.IGNORECASE),
    re.compile(r"/\s*hour\b", re.IGNORECASE),
    re.compile(r"/\s*day\b", re.IGNORECASE),
)

# Currency override. Default is AUD; USD applies when a clear US-dollar
# marker is in close proximity.
_USD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bUSD\b"),
    re.compile(r"\bUS\$"),
    re.compile(r"\bU\.S\.\s*\$"),
)

_PERIOD_WINDOW = 30  # chars after the match for hourly/daily markers
_CONTEXT_RADIUS = 60  # chars on either side for the salary keyword scan
# Negative scan is intentionally tighter than salary scan: ``raised $X``
# 60 chars before a *different* ``$Y`` is information about a previous
# funding round, not a veto on a clearly-marked salary later in the
# sentence. Keep this small enough that the negative keyword has to be
# almost-adjacent to the matched ``$`` to fire.
_NEGATIVE_RADIUS = 20


def infer_salary(
    *,
    title: str | None,
    description: str | None,
) -> tuple[int | None, int | None, str | None]:
    """Best-guess annual salary band from free-text fields.

    Returns ``(min, max, currency)`` if the parser is confident,
    ``(None, None, None)`` otherwise. ``min == max`` when the listing
    quotes a single value rather than a range. Decimal cents are
    floored — the schema stores ints.
    """
    haystack = "\n".join(s for s in (title, description) if s)
    if not haystack:
        return (None, None, None)

    # Try ranges first — they're more specific and a range token in
    # the text is itself evidence the listing is quoting a salary band.
    saw_range_match = False
    for match in _RANGE_PATTERN.finditer(haystack):
        saw_range_match = True
        parsed = _qualify_range(haystack, match)
        if parsed is not None:
            return parsed

    # If a range pattern matched but didn't qualify (inverted, OOB,
    # negative context), refuse to fall through to single-value
    # parsing — the salary text is malformed and any single $ value
    # we'd extract would be misleading.
    if saw_range_match:
        return (None, None, None)

    for match in _AMOUNT_PATTERN.finditer(haystack):
        parsed = _qualify_single(haystack, match)
        if parsed is not None:
            return parsed

    return (None, None, None)


def _qualify_range(
    haystack: str,
    match: re.Match[str],
) -> tuple[int, int, str] | None:
    """Return the parsed range or ``None`` if any guard fails."""
    if _has_negative_context(haystack, match):
        return None
    if _has_non_annual_period(haystack, match):
        return None
    if not _has_salary_context(haystack, match):
        return None

    min_value = _parse_amount(
        match.group("min_num"),
        match.group("min_k") or match.group("max_k"),
    )
    max_value = _parse_amount(match.group("max_num"), match.group("max_k"))
    if min_value > max_value:
        return None
    if not _within_bounds(min_value) or not _within_bounds(max_value):
        return None

    currency = _detect_currency(haystack, match)
    return (min_value, max_value, currency)


def _qualify_single(
    haystack: str,
    match: re.Match[str],
) -> tuple[int, int, str] | None:
    """Single-value variant of :func:`_qualify_range`."""
    if _has_negative_context(haystack, match):
        return None
    if _has_non_annual_period(haystack, match):
        return None
    if not _has_salary_context(haystack, match):
        return None

    value = _parse_amount(match.group(1), match.group(2))
    if not _within_bounds(value):
        return None

    currency = _detect_currency(haystack, match)
    return (value, value, currency)


def _parse_amount(num_text: str, k_marker: str | None) -> int:
    """Convert a regex-matched numeral + optional ``K`` to an int.

    The regex shape (``\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?``) guarantees
    ``num_text`` is always a valid numeral string; we don't guard the
    ``float()`` call.
    """
    cleaned = num_text.replace(",", "")
    value = float(cleaned)
    if k_marker:
        value *= 1_000
    return int(value)


def _within_bounds(value: int) -> bool:
    return _MIN_PLAUSIBLE_ANNUAL <= value <= _MAX_PLAUSIBLE_ANNUAL


def _has_negative_context(haystack: str, match: re.Match[str]) -> bool:
    """True if a specific funding/AUM/revenue pattern is adjacent to
    ``match``. The patterns are anchored to ``$`` proximity, so generic
    keywords like ``investors`` don't fire on their own."""
    window = _slice(haystack, match, _NEGATIVE_RADIUS)
    return any(p.search(window) is not None for p in _NEGATIVE_ADJACENT_PATTERNS)


def _has_non_annual_period(haystack: str, match: re.Match[str]) -> bool:
    suffix = haystack[match.end() : match.end() + _PERIOD_WINDOW]
    return any(p.search(suffix) is not None for p in _NON_ANNUAL_PERIOD_PATTERNS)


def _has_salary_context(haystack: str, match: re.Match[str]) -> bool:
    window = _slice(haystack, match, _CONTEXT_RADIUS)
    return any(p.search(window) is not None for p in _SALARY_ADJACENT_PATTERNS)


def _detect_currency(haystack: str, match: re.Match[str]) -> str:
    window = _slice(haystack, match, _CONTEXT_RADIUS)
    if any(p.search(window) is not None for p in _USD_PATTERNS):
        return "USD"
    return "AUD"


def _slice(haystack: str, match: re.Match[str], radius: int) -> str:
    start = max(0, match.start() - radius)
    end = min(len(haystack), match.end() + radius)
    return haystack[start:end]


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SalaryBackfillResult:
    """Summary returned by :func:`backfill_salaries`."""

    inspected: int
    updated: int


def backfill_salaries(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> SalaryBackfillResult:
    """Walk jobs whose salary fields are null and infer from text.

    Updates each row in place when the parser produces a confident
    answer; leaves rows untouched when the description has no signal.
    Pass ``limit`` to bound a single pass.
    """
    sql = (
        "SELECT id, title, description_text "
        "FROM jobs "
        "WHERE salary_min IS NULL AND salary_max IS NULL "
        "ORDER BY last_seen_at DESC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()

    updated = 0
    for row in rows:
        job_id = int(row[0])
        salary_min, salary_max, currency = infer_salary(
            title=row[1],
            description=row[2],
        )
        if salary_min is None and salary_max is None:
            continue
        conn.execute(
            "UPDATE jobs SET salary_min = ?, salary_max = ?, salary_currency = ? WHERE id = ?",
            (salary_min, salary_max, currency, job_id),
        )
        updated += 1

    if updated:
        conn.commit()
    _log.info(
        "salary_backfill",
        extra={"inspected": len(rows), "updated": updated},
    )
    return SalaryBackfillResult(inspected=len(rows), updated=updated)


__all__: list[str] = [
    "SalaryBackfillResult",
    "backfill_salaries",
    "infer_salary",
]
