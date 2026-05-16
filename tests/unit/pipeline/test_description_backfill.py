"""Tests for the per-job description backfill."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.fetcher.base import Response
from jobai.pipeline.description_backfill import (
    RECIPES,
    BackfillResult,
    DescriptionRecipe,
    _indeed_side_panel_url,
    _parse_indeed_description,
    _parse_linkedin_description,
    backfill_descriptions,
    select_pending_jobs,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "backfill.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


_seed_counter = 0


def _seed_job(
    conn: sqlite3.Connection,
    *,
    kind: str,
    title: str = "Senior Engineer",
    apply_url: str = "https://example.com/job/1",
    description: str | None = None,
) -> int:
    """Insert one source + raw + canonical job row, return job_id."""
    global _seed_counter  # noqa: PLW0603 - test-only fixture state
    _seed_counter += 1
    account = f"test{_seed_counter}"
    cur = conn.execute(
        "INSERT INTO sources "
        "(kind, account, display_name, default_tier, enabled, cadence_seconds) "
        "VALUES (?, ?, ?, ?, 1, 3600)",
        (kind, account, f"{kind} {account}", 3),
    )
    source_id = cur.lastrowid
    raw = conn.execute(
        "INSERT INTO jobs_raw (source_id, source_external_id, raw_json, raw_sha256, "
        "first_seen_at, last_seen_at) VALUES (?, ?, '{}', 'x', datetime('now'), datetime('now'))",
        (source_id, apply_url),
    )
    raw_id = raw.lastrowid
    job = conn.execute(
        "INSERT INTO jobs "
        "(dedup_key, title, company, company_norm, apply_url, description_text, "
        " first_seen_at, last_seen_at, fingerprint_json) "
        "VALUES (?, ?, 'Co', 'co', ?, ?, datetime('now'), datetime('now'), '{}')",
        (apply_url, title, apply_url, description),
    )
    job_id = job.lastrowid
    conn.execute(
        "INSERT INTO job_sources (job_id, source_id, jobs_raw_id, apply_url) VALUES (?, ?, ?, ?)",
        (job_id, source_id, raw_id, apply_url),
    )
    conn.commit()
    assert job_id is not None
    return int(job_id)


class _ScriptedFetcher:
    """Per-URL canned-response fetcher."""

    def __init__(self, responses: dict[str, Response | BaseException]) -> None:
        self._responses = dict(responses)
        self.calls: list[str] = []

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
        wait_until: str = "networkidle",
    ) -> Response:
        self.calls.append(url)
        item = self._responses.get(url)
        if item is None:
            return Response(
                url=url,
                status_code=404,
                headers={},
                body=b"",
                fetched_at=datetime.now(tz=UTC),
            )
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        return None


def _resp(status: int, body: bytes = b"") -> Response:
    return Response(
        url="https://example.com",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# select_pending_jobs
# ---------------------------------------------------------------------------


def test_select_pending_jobs_returns_only_null_descriptions(
    conn: sqlite3.Connection,
) -> None:
    needs = _seed_job(conn, kind="linkedin", apply_url="https://l.in/1")
    _seed_job(
        conn,
        kind="linkedin",
        apply_url="https://l.in/2",
        description="already filled",
    )
    rows = select_pending_jobs(conn, kinds=("linkedin",))
    assert [r[0] for r in rows] == [needs]


def test_select_pending_jobs_filters_by_kind(conn: sqlite3.Connection) -> None:
    _seed_job(conn, kind="greenhouse", apply_url="https://g.io/1")
    needs = _seed_job(conn, kind="linkedin", apply_url="https://l.in/1")
    rows = select_pending_jobs(conn, kinds=("linkedin",))
    assert [r[0] for r in rows] == [needs]


def test_select_pending_jobs_handles_empty_kind_tuple(conn: sqlite3.Connection) -> None:
    _seed_job(conn, kind="linkedin", apply_url="https://l.in/1")
    rows = select_pending_jobs(conn, kinds=())
    assert rows == []


def test_select_pending_jobs_treats_empty_string_as_pending(
    conn: sqlite3.Connection,
) -> None:
    """Some sources insert ``''`` instead of NULL — both should backfill."""
    needs = _seed_job(conn, kind="linkedin", apply_url="https://l.in/1", description="")
    rows = select_pending_jobs(conn, kinds=("linkedin",))
    assert [r[0] for r in rows] == [needs]


# ---------------------------------------------------------------------------
# _parse_linkedin_description
# ---------------------------------------------------------------------------


def test_parse_linkedin_description_extracts_text() -> None:
    html = (
        "<html><body>"
        '<div class="description__text"><p>Build cool things.</p>'
        "<p>Remote-friendly.</p></div></body></html>"
    )
    assert _parse_linkedin_description(html) == "Build cool things.Remote-friendly."


def test_parse_linkedin_description_falls_back_to_inner_wrapper() -> None:
    html = '<html><body><div class="show-more-less-html__markup">Markup body</div></body></html>'
    assert _parse_linkedin_description(html) == "Markup body"


def test_parse_linkedin_description_returns_none_on_missing() -> None:
    assert _parse_linkedin_description("<html><body></body></html>") is None


def test_seek_recipe_uses_domcontentloaded_and_jd_selector() -> None:
    """Seek's detail SPA never goes network-idle; the recipe must opt
    into domcontentloaded + the JD-container selector."""
    recipe = RECIPES["seek"]
    assert recipe.wait_until == "domcontentloaded"
    assert recipe.wait_selector == '[data-automation="jobAdDetails"]'


# ---------------------------------------------------------------------------
# backfill_descriptions (end-to-end with fakes)
# ---------------------------------------------------------------------------


async def test_backfill_fills_pending_linkedin_jobs(conn: sqlite3.Connection) -> None:
    job_id = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/42")
    fetcher = _ScriptedFetcher(
        {
            "https://l.in/job/42": _resp(
                200,
                body=(
                    b'<html><body><div class="description__text">'
                    b"Real description body.</div></body></html>"
                ),
            ),
        },
    )

    result = await backfill_descriptions(conn, fetcher)

    assert result == BackfillResult(attempted=1, filled=1, skipped=0)
    row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row[0] == "Real description body."


async def test_backfill_fills_seek_job_with_domcontentloaded(
    conn: sqlite3.Connection,
) -> None:
    """A pending Seek row is filled from the JD container, and the
    fetcher is driven with the verified domcontentloaded + selector
    strategy rather than the default networkidle wait."""
    job_id = _seed_job(conn, kind="seek", apply_url="https://www.seek.com.au/job/77")

    seen: dict[str, object] = {}

    class _RecordingFetcher:
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
            wait_until: str = "networkidle",
        ) -> Response:
            seen["url"] = url
            seen["wait_for_selector"] = wait_for_selector
            seen["wait_until"] = wait_until
            return _resp(
                200,
                body=(
                    b'<html><body><div data-automation="jobAdDetails">'
                    b"Full Seek JD body.</div></body></html>"
                ),
            )

        async def aclose(self) -> None:
            return None

    result = await backfill_descriptions(conn, _RecordingFetcher())

    assert result == BackfillResult(attempted=1, filled=1, skipped=0)
    row = conn.execute(
        "SELECT description_text FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert row[0] == "Full Seek JD body."
    assert seen["wait_until"] == "domcontentloaded"
    assert seen["wait_for_selector"] == '[data-automation="jobAdDetails"]'


async def test_backfill_skips_non_2xx(conn: sqlite3.Connection) -> None:
    job_id = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/blocked")
    fetcher = _ScriptedFetcher({"https://l.in/job/blocked": _resp(403)})

    result = await backfill_descriptions(conn, fetcher)

    assert result.attempted == 1
    assert result.filled == 0
    assert result.skipped == 1
    row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row[0] is None


async def test_backfill_continues_after_fetch_exception(conn: sqlite3.Connection) -> None:
    """One job's network failure shouldn't tank the whole batch."""
    bad = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/bad")
    good = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/good")
    fetcher = _ScriptedFetcher(
        {
            "https://l.in/job/bad": RuntimeError("network down"),
            "https://l.in/job/good": _resp(
                200,
                body=(b'<html><body><div class="description__text">OK</div></body></html>'),
            ),
        },
    )

    result = await backfill_descriptions(conn, fetcher)

    assert result.attempted == 2
    assert result.filled == 1
    assert result.skipped == 1
    bad_row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (bad,)).fetchone()
    good_row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (good,)).fetchone()
    assert bad_row[0] is None
    assert good_row[0] == "OK"


async def test_backfill_respects_limit(conn: sqlite3.Connection) -> None:
    for i in range(3):
        _seed_job(conn, kind="linkedin", apply_url=f"https://l.in/job/{i}")
    fetcher = _ScriptedFetcher(
        {
            f"https://l.in/job/{i}": _resp(
                200,
                body=(b'<html><body><div class="description__text">D</div></body></html>'),
            )
            for i in range(3)
        },
    )

    result = await backfill_descriptions(conn, fetcher, limit=2)

    assert result.attempted == 2
    assert result.filled == 2
    assert len(fetcher.calls) == 2


async def test_backfill_returns_zero_when_nothing_pending(
    conn: sqlite3.Connection,
) -> None:
    _seed_job(
        conn,
        kind="linkedin",
        apply_url="https://l.in/job/done",
        description="already there",
    )
    fetcher = _ScriptedFetcher({})

    result = await backfill_descriptions(conn, fetcher)

    assert result == BackfillResult(attempted=0, filled=0, skipped=0)


async def test_backfill_skips_when_parser_returns_empty_description(
    conn: sqlite3.Connection,
) -> None:
    """A 200 response whose body the recipe can't parse out a non-empty
    description from gets skipped (no UPDATE). Exercises the
    ``not description: continue`` branch."""
    _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/blank")
    # LinkedIn parser hunts for div.description__text; an empty body
    # produces no match -> parse returns None.
    fetcher = _ScriptedFetcher({"https://l.in/job/blank": _resp(200, body=b"<html></html>")})

    result = await backfill_descriptions(conn, fetcher)

    assert result.filled == 0
    assert result.skipped == 1


def test_recipes_registry_exports_linkedin() -> None:
    assert "linkedin" in RECIPES
    # LinkedIn fetches the apply URL as-is; no wait selector needed.
    assert RECIPES["linkedin"].fetch_url("https://l.in/x") == "https://l.in/x"
    assert RECIPES["linkedin"].wait_selector is None


def test_recipes_registry_exports_indeed() -> None:
    assert "indeed" in RECIPES
    indeed = RECIPES["indeed"]
    assert indeed.wait_selector is not None
    # Indeed's fetch URL must rewrite ``/viewjob`` to a search-page hit
    # (Cloudflare gates the apply URL itself).
    rewritten = indeed.fetch_url("https://au.indeed.com/viewjob?jk=abc123")
    assert "vjk=abc123" in rewritten
    assert "/jobs?" in rewritten


# ---------------------------------------------------------------------------
# _indeed_side_panel_url
# ---------------------------------------------------------------------------


def test_indeed_side_panel_url_extracts_jk() -> None:
    """The transform pulls the ``jk`` param out of any-position query
    string and emits a side-panel URL anchored at /jobs."""
    url = _indeed_side_panel_url("https://au.indeed.com/viewjob?jk=abc123")
    assert url == "https://au.indeed.com/jobs?q=&l=Australia&vjk=abc123"


def test_indeed_side_panel_url_handles_extra_params() -> None:
    url = _indeed_side_panel_url(
        "https://au.indeed.com/viewjob?from=serp&jk=def456&advn=12345",
    )
    assert url.endswith("vjk=def456")


def test_indeed_side_panel_url_passes_through_when_no_jk() -> None:
    """Without a ``jk`` param the rewrite would mislead, so leave the
    URL untouched and let the fetch fail in a known way."""
    untouched = "https://au.indeed.com/jobs?q=python"
    assert _indeed_side_panel_url(untouched) == untouched


# ---------------------------------------------------------------------------
# _parse_indeed_description
# ---------------------------------------------------------------------------


def test_parse_indeed_description_extracts_text() -> None:
    html = (
        "<html><body>"
        '<div id="jobDescriptionText"><p>About Acme.</p>'
        "<p>You will build cool things.</p></div></body></html>"
    )
    assert _parse_indeed_description(html) == "About Acme.You will build cool things."


def test_parse_indeed_description_falls_back_to_data_testid() -> None:
    html = (
        '<html><body><div data-testid="jobsearch-JobComponent-description">Body</div></body></html>'
    )
    assert _parse_indeed_description(html) == "Body"


def test_parse_indeed_description_returns_none_on_missing() -> None:
    assert _parse_indeed_description("<html><body></body></html>") is None


# ---------------------------------------------------------------------------
# Indeed integration: backfill applies the URL transform
# ---------------------------------------------------------------------------


async def test_backfill_uses_indeed_side_panel_url(conn: sqlite3.Connection) -> None:
    """An Indeed pending job triggers a fetch against the rewritten URL,
    not the raw ``/viewjob`` apply URL."""
    job_id = _seed_job(
        conn,
        kind="indeed",
        apply_url="https://au.indeed.com/viewjob?jk=feedface",
    )
    side_panel_url = "https://au.indeed.com/jobs?q=&l=Australia&vjk=feedface"
    fetcher = _ScriptedFetcher(
        {
            side_panel_url: _resp(
                200,
                body=(
                    b'<html><body><div id="jobDescriptionText">'
                    b"Real Indeed body.</div></body></html>"
                ),
            ),
        },
    )

    result = await backfill_descriptions(conn, fetcher)

    assert result == BackfillResult(attempted=1, filled=1, skipped=0)
    # The fetch went to the rewritten URL, not the raw apply URL.
    assert fetcher.calls == [side_panel_url]
    row = conn.execute(
        "SELECT description_text FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert row[0] == "Real Indeed body."


async def test_backfill_overrides_recipes_via_kwarg(conn: sqlite3.Connection) -> None:
    """Tests can inject a custom recipe map; production map untouched."""
    _seed_job(
        conn,
        kind="custom",
        apply_url="https://example.com/job/77",
    )
    fetcher = _ScriptedFetcher(
        {
            "https://example.com/job/77": _resp(
                200,
                body=b"<html><body><pre>Custom body</pre></body></html>",
            ),
        },
    )
    custom = {
        "custom": DescriptionRecipe(
            parse=lambda html: "Custom body" if "<pre>Custom body</pre>" in html else None,
        ),
    }
    result = await backfill_descriptions(conn, fetcher, recipes=custom)
    assert result.filled == 1
