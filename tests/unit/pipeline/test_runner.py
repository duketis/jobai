"""Tests for the single-source scrape runner."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.fetcher.base import Fetcher, Response
from jobai.pipeline.runner import RunResult, run_source
from jobai.sources.base import BaseSource, NormalizedJob
from jobai.sources.repository import SourceRow, upsert_source


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


@pytest.fixture
def source_row(conn: sqlite3.Connection) -> SourceRow:
    return upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="Atlassian",
    )


class _StubFetcher:
    """Minimal Fetcher that returns canned responses keyed by URL."""

    def __init__(self, responses: dict[str, Response] | None = None) -> None:
        self._responses = responses or {}

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
        wait_for_selector: str | None = None,
    ) -> Response:
        del method, headers, json, data, timeout
        if url in self._responses:
            return self._responses[url]
        return Response(url=url, status_code=200, headers={}, body=b"{}")

    async def aclose(self) -> None:
        return None


class _FixedSource(BaseSource):
    """A source that yields a pre-built list of jobs without touching network."""

    kind = "stub"

    def __init__(self, account: str, jobs: list[NormalizedJob]) -> None:
        self.account = account
        self._jobs = jobs

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        del fetcher
        for job in self._jobs:
            yield job


class _FailingSource(BaseSource):
    """A source whose discover() always raises after yielding one job."""

    kind = "stub"

    def __init__(self, account: str) -> None:
        self.account = account

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        del fetcher
        yield NormalizedJob(
            source_external_id="1",
            title="OK",
            company="X",
            apply_url="https://example.com/1",
            raw_data={},
        )
        raise RuntimeError("simulated parser explosion")


def _make_job(external_id: str, *, title: str = "Engineer") -> NormalizedJob:
    return NormalizedJob(
        source_external_id=external_id,
        title=title,
        company="Atlassian",
        apply_url=f"https://example.com/{external_id}",
        raw_data={"id": external_id},
    )


async def test_run_source_writes_scrape_run_row_with_success_status(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    source = _FixedSource("atlassian", [_make_job("1"), _make_job("2")])

    result = await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    assert isinstance(result, RunResult)
    assert result.status == "success"
    assert result.items_seen == 2
    assert result.items_new == 2
    assert result.items_updated == 0

    row = conn.execute(
        "SELECT * FROM scrape_runs WHERE id = ?",
        (result.run_id,),
    ).fetchone()
    assert row["status"] == "success"
    assert row["items_seen"] == 2
    assert row["items_new"] == 2
    assert row["finished_at"] is not None


async def test_run_source_inserts_jobs_raw_rows(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    source = _FixedSource("atlassian", [_make_job("1"), _make_job("2")])

    await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    rows = conn.execute(
        "SELECT source_external_id, raw_json FROM jobs_raw "
        "WHERE source_id = ? ORDER BY source_external_id",
        (source_row.id,),
    ).fetchall()
    assert [r["source_external_id"] for r in rows] == ["1", "2"]
    payload = json.loads(rows[0]["raw_json"])
    assert payload["title"] == "Engineer"
    assert payload["company"] == "Atlassian"


async def test_run_source_second_run_updates_last_seen_only_when_unchanged(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    job = _make_job("1")
    source = _FixedSource("atlassian", [job])

    first = await run_source(
        conn=conn, source=source, source_row=source_row, fetcher=_StubFetcher()
    )
    second = await run_source(
        conn=conn, source=source, source_row=source_row, fetcher=_StubFetcher()
    )

    assert first.items_new == 1
    assert second.items_new == 0
    assert second.items_updated == 1

    rows = conn.execute("SELECT first_seen_at, last_seen_at FROM jobs_raw").fetchall()
    assert len(rows) == 1
    # last_seen_at moves forward; first_seen_at is fixed.
    assert rows[0]["last_seen_at"] >= rows[0]["first_seen_at"]


async def test_run_source_updates_raw_when_payload_changes(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    initial = _make_job("1", title="Engineer")
    updated = _make_job("1", title="Senior Engineer")

    await run_source(
        conn=conn,
        source=_FixedSource("atlassian", [initial]),
        source_row=source_row,
        fetcher=_StubFetcher(),
    )
    await run_source(
        conn=conn,
        source=_FixedSource("atlassian", [updated]),
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    raw_json = conn.execute("SELECT raw_json FROM jobs_raw").fetchone()["raw_json"]
    assert json.loads(raw_json)["title"] == "Senior Engineer"


async def test_run_source_marks_run_failed_when_source_raises(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    source = _FailingSource("atlassian")

    result = await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    assert result.status == "failed"
    assert "simulated parser explosion" in (result.error_summary or "")
    # Items yielded before the raise are kept.
    assert result.items_seen == 1
    assert result.items_new == 1

    row = conn.execute(
        "SELECT status, error_summary FROM scrape_runs WHERE id = ?",
        (result.run_id,),
    ).fetchone()
    assert row["status"] == "failed"
    assert "RuntimeError" in row["error_summary"]


async def test_run_source_promotes_jobs_into_canonical_table(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    """The runner must populate the canonical jobs table and job_sources
    link alongside jobs_raw."""
    source = _FixedSource(
        "atlassian",
        [_make_job("1", title="Backend Engineer"), _make_job("2", title="Frontend Engineer")],
    )

    await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    canonical_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM job_sources").fetchone()[0]
    assert canonical_count == 2
    assert link_count == 2


async def test_run_source_records_raw_responses_via_recording_fetcher(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    """A source that uses the fetcher must produce raw_responses rows."""

    class _NetworkSource(BaseSource):
        kind = "stub"

        def __init__(self, account: str) -> None:
            self.account = account

        async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
            await fetcher.fetch("https://api.example.com/jobs")
            yield _make_job("1")

    fetcher = _StubFetcher(
        {
            "https://api.example.com/jobs": Response(
                url="https://api.example.com/jobs",
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=b'{"jobs":[]}',
            )
        }
    )

    result = await run_source(
        conn=conn,
        source=_NetworkSource("atlassian"),
        source_row=source_row,
        fetcher=fetcher,
    )

    raw_count = conn.execute(
        "SELECT COUNT(*) FROM raw_responses WHERE run_id = ?",
        (result.run_id,),
    ).fetchone()[0]
    assert raw_count == 1


# ---------------------------------------------------------------------------
# Inference: ``_ensure_salary`` integration in the run loop
# ---------------------------------------------------------------------------


async def test_run_source_fills_in_salary_from_description_when_source_omits_it(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    """If the source emits a job with no structured salary but the
    description carries one, the runner must persist the parsed salary
    on the canonical row."""
    job = NormalizedJob(
        source_external_id="1",
        title="Senior Engineer",
        company="Atlassian",
        apply_url="https://example.com/1",
        raw_data={"id": "1"},
        description_text="Salary: $140,000 - $180,000 per annum + super.",
    )
    source = _FixedSource("atlassian", [job])

    await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    row = conn.execute("SELECT salary_min, salary_max, salary_currency FROM jobs").fetchone()
    assert row["salary_min"] == 140_000
    assert row["salary_max"] == 180_000
    assert row["salary_currency"] == "AUD"


async def test_run_source_preserves_structured_salary_from_source(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    """Sources that DO surface a structured salary (Ashby, APS Jobs)
    must pass through untouched — the inference is a fallback, not an
    override."""
    job = NormalizedJob(
        source_external_id="1",
        title="Senior Engineer",
        company="Atlassian",
        apply_url="https://example.com/1",
        raw_data={"id": "1"},
        salary_min=200_000,
        salary_max=250_000,
        salary_currency="USD",
        description_text="Salary: $140,000 - $180,000 per annum + super.",
    )
    source = _FixedSource("atlassian", [job])

    await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    row = conn.execute("SELECT salary_min, salary_max, salary_currency FROM jobs").fetchone()
    assert row["salary_min"] == 200_000
    assert row["salary_max"] == 250_000
    assert row["salary_currency"] == "USD"


async def test_run_source_leaves_salary_null_when_description_has_no_signal(
    conn: sqlite3.Connection,
    source_row: SourceRow,
) -> None:
    """No salary on source + no parseable signal in description → null
    stays null. The pass must not invent numbers to fill the schema."""
    job = NormalizedJob(
        source_external_id="1",
        title="Senior Engineer",
        company="Atlassian",
        apply_url="https://example.com/1",
        raw_data={"id": "1"},
        description_text="Great team, fully remote, lots of ownership.",
    )
    source = _FixedSource("atlassian", [job])

    await run_source(
        conn=conn,
        source=source,
        source_row=source_row,
        fetcher=_StubFetcher(),
    )

    row = conn.execute("SELECT salary_min, salary_max FROM jobs").fetchone()
    assert row["salary_min"] is None
    assert row["salary_max"] is None
