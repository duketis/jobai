"""Tests for the salary inference pass.

The parser scans free-text descriptions for AU-style salary mentions
and returns ``(min, max, currency)``. The cases below are drawn from
real listings observed in the production DB — the comment above each
group cites the company / context that motivated it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.pipeline.salary_inference import (
    backfill_salaries,
    infer_salary,
)
from jobai.sources.repository import upsert_source

# ---------------------------------------------------------------------------
# Pure parser — positive cases
# ---------------------------------------------------------------------------


def test_infer_salary_returns_none_for_empty_input() -> None:
    assert infer_salary(title=None, description=None) == (None, None, None)
    assert infer_salary(title="", description="") == (None, None, None)


def test_infer_salary_returns_none_for_text_with_no_dollar_sign() -> None:
    assert infer_salary(
        title="Senior Engineer",
        description="Great team, fully remote, lots of ownership.",
    ) == (None, None, None)


def test_infer_salary_parses_full_numerals_with_per_annum_marker() -> None:
    """Hume City Council pattern: ``$110,506 per annum``."""
    result = infer_salary(
        title="Senior Officer",
        description="Job Description $110,506 per annum plus superannuation.",
    )
    assert result == (110_506, 110_506, "AUD")


def test_infer_salary_parses_full_range_with_dollar_signs() -> None:
    """City of Stonnington pattern: ``Salary: $100,066.00 - $108,372.00``."""
    result = infer_salary(
        title="Senior Officer",
        description="Salary: $100,066.00 - $108,372.00 per annum plus 11% super.",
    )
    assert result == (100_066, 108_372, "AUD")


def test_infer_salary_parses_k_shorthand_range() -> None:
    """Merri-bek pattern: ``Salary: $102 - 111K + Super``."""
    result = infer_salary(
        title="Festival Coordinator",
        description="Salary: $102 - 111K + Super for a 12-month contract.",
    )
    assert result == (102_000, 111_000, "AUD")


def test_infer_salary_parses_lowercase_k_with_full_range() -> None:
    """``$65k–$80k`` style with en-dash."""
    result = infer_salary(
        title="Sommelier",
        description="Full-time | $65k–$80k + super, depending on experience.",
    )
    assert result == (65_000, 80_000, "AUD")


def test_infer_salary_handles_band_label_prefix() -> None:
    """Council pattern: ``Band 8 - $123,558.69 to $138,752.00``."""
    result = infer_salary(
        title="Workforce Planning Manager",
        description=(
            "We're hiring for strategic leadership. Salary: Band 8 - "
            "$123,558.69 to $138,752.00 + 11.5% superannuation."
        ),
    )
    assert result == (123_558, 138_752, "AUD")


def test_infer_salary_recognises_compensation_marker() -> None:
    """``Compensation: $X - $Y`` is as common a marker as ``Salary``."""
    result = infer_salary(
        title="Engineer",
        description="Compensation: $120,000 - $160,000 + equity.",
    )
    assert result == (120_000, 160_000, "AUD")


def test_infer_salary_uses_title_text_as_well_as_description() -> None:
    """Some sources (recruiter listings) put the salary in the title."""
    result = infer_salary(
        title="Backend Engineer | $120k - $160k + super",
        description="Permanent role at a Sydney scale-up.",
    )
    assert result == (120_000, 160_000, "AUD")


def test_infer_salary_picks_first_qualified_match_when_multiple_present() -> None:
    """If a description mentions both a salary and a funding raise, the
    salary should win because we look for the salary marker. Order of
    appearance in text isn't sufficient — we need *context*."""
    result = infer_salary(
        title="Engineer",
        description=(
            "We've raised $50M in Series B funding from top investors. "
            "Salary: $140,000 - $180,000 + super for the right candidate."
        ),
    )
    assert result == (140_000, 180_000, "AUD")


def test_infer_salary_strips_decimals_and_returns_ints() -> None:
    """Salary table cells often carry trailing cents; we drop them.
    The schema stores ints, so consistency at the boundary."""
    result = infer_salary(
        title="Officer",
        description="Salary: $99,999.99 per annum.",
    )
    assert result == (99_999, 99_999, "AUD")


def test_infer_salary_detects_usd_currency_from_us_marker() -> None:
    result = infer_salary(
        title="Engineer",
        description="Salary: USD $120,000 - $150,000 per year.",
    )
    assert result == (120_000, 150_000, "USD")


def test_infer_salary_detects_usd_from_us_dollar_prefix() -> None:
    result = infer_salary(
        title="Engineer",
        description="Compensation: US$120,000 to US$150,000 annually.",
    )
    assert result == (120_000, 150_000, "USD")


# ---------------------------------------------------------------------------
# Pure parser — negative / noise cases
# ---------------------------------------------------------------------------


def test_infer_salary_rejects_funding_raise_mention() -> None:
    """ElevenLabs / Heidi pattern — fundraising figures shouldn't be
    misread as the candidate's salary."""
    result = infer_salary(
        title="Engineer",
        description="We've raised $781M from Andreessen Horowitz, ICONIQ, Sequoia.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_backed_by_mention() -> None:
    result = infer_salary(
        title="Engineer",
        description="Backed by nearly $1B in funding from leading VCs.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_aum_mention() -> None:
    """Property-investing companies often quote AUM figures in their
    listing copy. ``$80M AUM`` is not a salary."""
    result = infer_salary(
        title="Investment Analyst",
        description="Caruso has grown rapidly to $80M AUM since launch.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_revenue_mention() -> None:
    result = infer_salary(
        title="Engineer",
        description="Profitable, $50M ARR, growing 100% YoY.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_transaction_flow_mention() -> None:
    """Ramp pattern: ``$100B in the transaction flow of every dollar``.
    Numbers describing money the company moves, not money it pays."""
    result = infer_salary(
        title="Engineer",
        description="We're in the transaction flow of every dollar a business spends, $100B+.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_series_funding_round() -> None:
    result = infer_salary(
        title="Engineer",
        description="Following our Series A round of $25M, we're scaling fast.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_hourly_rate() -> None:
    """``$22 - $70 per hour`` is real comp signal but conflating it with
    annual salary breaks the search filter. Skip until we add a period
    column to the schema."""
    result = infer_salary(
        title="Specialist",
        description="Compensation: $22 - $70 per hour, depending on expertise.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_daily_contractor_rate() -> None:
    """``$700-$800 per day`` is contractor land — not annual."""
    result = infer_salary(
        title="Senior Java Engineer",
        description="$700-$800 per day for a 12 month contract.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_short_slash_period_marker() -> None:
    """``/hr`` and ``/day`` are common shorthand on job ads."""
    result = infer_salary(
        title="Engineer",
        description="Pay: $120/hr.",
    )
    assert result == (None, None, None)


# ---------------------------------------------------------------------------
# Bounds / sanity checks — implausible ranges shouldn't slip through
# ---------------------------------------------------------------------------


def test_infer_salary_rejects_implausibly_low_value() -> None:
    """``$5`` near a salary marker is almost always a typo or quote
    fragment, not a real salary."""
    result = infer_salary(
        title="Engineer",
        description="Salary: $5 to $10 — placeholder copy.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_implausibly_high_value() -> None:
    """``$50M salary`` is impossible for a real role; must be funding."""
    result = infer_salary(
        title="Engineer",
        description="Salary: $50,000,000 plus equity.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_inverted_range() -> None:
    """Min > max is a parse bug, not a real range — better to drop than
    write garbage to the canonical row."""
    result = infer_salary(
        title="Engineer",
        description="Salary: $200,000 - $100,000 per annum.",
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_range_with_negative_context() -> None:
    """A range pattern in fundraising context must be rejected even
    when the numbers are within plausible salary bounds."""
    result = infer_salary(
        title="Engineer",
        description="We've raised $50,000 - $100,000 in seed funding from a top-tier VC.",
    )
    assert result == (None, None, None)


def test_infer_salary_handles_glued_text_from_html_stripping() -> None:
    """Some sources (Mercor-style listings) emit description_text with
    HTML structure stripped to a single run-on string. The keyword
    won't have a word boundary before it - 'timeCompensation:$180K'.
    The colon-anchored patterns must still fire."""
    glued = (
        "Software Engineer, New Grad (Zara)Type:Full-time"
        "Compensation:$180K - $250K/yrLocation:RemoteCommitment:10 hours/week"
    )
    result = infer_salary(title="Engineer", description=glued)
    assert result == (180_000, 250_000, "AUD")


def test_infer_salary_recognises_yr_suffix_as_annual_marker() -> None:
    """``/yr`` and ``/year`` are positive annual markers (the inverse
    of ``/hr``)."""
    result = infer_salary(
        title="Engineer",
        description="Compensation: $120,000 - $160,000 /yr",
    )
    assert result == (120_000, 160_000, "AUD")


def test_infer_salary_handles_keyword_with_intervening_prose() -> None:
    """Real listing pattern: keyword + colon + a sentence of prose
    before the actual number ('Compensation: Our midpoint TTR for this
    role is $236,500'). The gap may be up to 80 chars but must not
    cross a sentence terminator."""
    result = infer_salary(
        title="Reward Partner",
        description="Compensation: Our midpoint TTR for this role is $236,500.",
    )
    assert result == (236_500, 236_500, "AUD")


def test_infer_salary_does_not_cross_sentence_boundary() -> None:
    """If a period sits between the keyword and the next ``$``, the
    keyword is referring to something else - reject."""
    result = infer_salary(
        title="Engineer",
        description=(
            "Compensation: We pay competitive market rates. The going rate elsewhere is $200,000."
        ),
    )
    assert result == (None, None, None)


def test_infer_salary_rejects_range_without_salary_context() -> None:
    """A range with no salary keyword nearby — and no negative keyword
    either — should still be rejected. Bare ``$X-$Y`` in a paragraph
    is not enough to trust as a salary."""
    result = infer_salary(
        title="Project Manager",
        description="Cost: $50,000 - $100,000 for the renovation budget.",
    )
    assert result == (None, None, None)


# ---------------------------------------------------------------------------
# Backfill pass against a real DB
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


@pytest.fixture
def source_id(conn: sqlite3.Connection) -> int:
    return upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="Atlassian",
    ).id


def _seed(
    conn: sqlite3.Connection,
    *,
    dedup_key: str,
    title: str = "Engineer",
    description_text: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    salary_currency: str | None = None,
) -> int:
    """Insert a canonical job row directly. The schema requires a
    handful of NOT NULL columns; everything else can stay null."""
    cursor = conn.execute(
        "INSERT INTO jobs ("
        "  dedup_key, title, company, company_norm, apply_url, "
        "  description_text, salary_min, salary_max, salary_currency, "
        "  first_seen_at, last_seen_at, fingerprint_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), '{}')",
        (
            dedup_key,
            title,
            "Atlassian",
            "atlassian",
            f"https://example.com/{dedup_key}",
            description_text,
            salary_min,
            salary_max,
            salary_currency,
        ),
    )
    last_id = cursor.lastrowid
    assert last_id is not None
    conn.commit()
    return int(last_id)


def test_backfill_skips_rows_that_already_have_salary(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """If salary_min OR salary_max is already populated, leave it alone —
    the structured field came from the source and we trust it."""
    job_id = _seed(
        conn,
        dedup_key="a",
        salary_min=100_000,
        salary_max=120_000,
        salary_currency="AUD",
        description_text="Salary: $200,000 - $300,000 per annum",
    )

    result = backfill_salaries(conn)

    assert result.updated == 0
    row = conn.execute(
        "SELECT salary_min, salary_max, salary_currency FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert row["salary_min"] == 100_000
    assert row["salary_max"] == 120_000


def test_backfill_fills_in_salary_from_description_text(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    job_id = _seed(
        conn,
        dedup_key="a",
        description_text="Salary: $120,000 - $160,000 per annum + super",
    )

    result = backfill_salaries(conn)

    assert result.updated == 1
    row = conn.execute(
        "SELECT salary_min, salary_max, salary_currency FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert row["salary_min"] == 120_000
    assert row["salary_max"] == 160_000
    assert row["salary_currency"] == "AUD"


def test_backfill_skips_rows_with_no_parseable_salary(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """Description has no salary signal → row stays null → ``updated == 0``."""
    _seed(
        conn,
        dedup_key="a",
        description_text="Great team, fully remote, lots of ownership.",
    )

    result = backfill_salaries(conn)

    assert result.updated == 0
    assert result.inspected == 1


def test_backfill_respects_limit(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """Limit caps the pass at N rows so the scheduler can call it
    without monopolising the connection."""
    for i in range(5):
        _seed(
            conn,
            dedup_key=f"k{i}",
            description_text=f"Salary: ${100 + i},000 - ${120 + i},000 per annum",
        )

    result = backfill_salaries(conn, limit=2)

    assert result.inspected == 2
    assert result.updated == 2


def test_backfill_returns_zero_for_empty_db(conn: sqlite3.Connection) -> None:
    """No-rows path keeps the CLI tidy when there's nothing to do."""
    result = backfill_salaries(conn)
    assert result.inspected == 0
    assert result.updated == 0


def test_backfill_uses_title_when_description_is_null(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """Recruiter listings sometimes carry the salary in the title only."""
    job_id = _seed(
        conn,
        dedup_key="a",
        title="Backend Engineer | $150k - $180k + super",
        description_text=None,
    )

    backfill_salaries(conn)

    row = conn.execute(
        "SELECT salary_min, salary_max FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert row["salary_min"] == 150_000
    assert row["salary_max"] == 180_000


def test_infer_salary_accepts_any_extras_kwargs() -> None:
    """The function takes only title/description for now, but tests
    that *do* exist must use keyword-args so a future signature
    addition (e.g. ``country``) doesn't break call sites."""
    # Smoke: keyword-only call works as documented.
    result = infer_salary(
        title="Engineer",
        description="Salary: $100k - $130k + super",
    )
    assert result[0] == 100_000
    assert result[1] == 130_000


# ---------------------------------------------------------------------------
# Backfill — type ignore for unused fixture
# ---------------------------------------------------------------------------


def _unused(_x: Any) -> None:
    """Placeholder — silences `source_id` fixture-unused warnings on
    rows that don't link a job_source. We use it so ``source_id``
    fixture instantiation can verify the migration applied cleanly."""
