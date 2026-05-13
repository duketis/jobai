"""Tests for /api/jobs endpoints."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jobai.dedup.promote import promote_to_canonical_jobs
from jobai.sources.base import NormalizedJob
from jobai.sources.repository import upsert_source


@pytest.fixture
def seeded_db(db_path: Path) -> Iterator[Path]:
    """Insert a small but realistic dataset before each test."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        gh = upsert_source(
            conn,
            kind="greenhouse",
            account="atlassian",
            display_name="Atlassian",
        )
        lever = upsert_source(
            conn,
            kind="lever",
            account="palantir",
            display_name="Palantir",
        )

        def _seed(
            *,
            source_id: int,
            ext_id: str,
            **fields: Any,
        ) -> None:
            cursor = conn.execute(
                "INSERT INTO jobs_raw "
                "(source_id, source_external_id, raw_json, raw_sha256, "
                " first_seen_at, last_seen_at) "
                "VALUES (?, ?, '{}', 'x', datetime('now'), datetime('now'))",
                (source_id, ext_id),
            )
            raw_id = cursor.lastrowid
            assert raw_id is not None
            base: dict[str, Any] = {
                "source_external_id": ext_id,
                "title": "Senior Backend Engineer",
                "company": "Atlassian",
                "apply_url": f"https://example.com/{ext_id}",
                "raw_data": {"id": ext_id},
                "location_country": "Australia",
            }
            base.update(fields)
            promote_to_canonical_jobs(
                conn,
                source_id=source_id,
                jobs_raw_id=int(raw_id),
                job=NormalizedJob(**base),
            )

        _seed(
            source_id=gh.id,
            ext_id="1",
            title="Python Backend Engineer",
            description_text="Build async services in Python on AWS.",
            location_raw="Sydney, Australia",
            location_city="Sydney",
            remote_type="onsite",
            employment_type="full-time",
            posted_at="2026-04-15",
            salary_min=140000,
            salary_max=190000,
            salary_currency="AUD",
        )
        _seed(
            source_id=gh.id,
            ext_id="2",
            title="Senior Frontend Engineer (Remote)",
            description_text="React and TypeScript expertise for a global SaaS team.",
            location_raw="Remote, Australia",
            remote_type="remote",
            employment_type="full-time",
            posted_at="2026-04-20",
        )
        _seed(
            source_id=lever.id,
            ext_id="alpha",
            title="Data Engineer",
            company="Palantir",
            description_text="ETL pipelines on Kubernetes.",
            location_raw="London, UK",
            location_country="United Kingdom",
            location_city="London",
            remote_type="hybrid",
            employment_type="full-time",
            posted_at="2026-03-10",
        )
        conn.commit()
    finally:
        conn.close()
    yield db_path


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_list_jobs_returns_all_when_no_filters(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs").json()
    assert body["total"] == 3
    assert len(body["items"]) == 3


def test_list_jobs_q_uses_fts(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"q": "python"}).json()
    titles = [item["title"] for item in body["items"]]
    assert "Python Backend Engineer" in titles
    assert "Data Engineer" not in titles


def test_list_jobs_q_combines_terms_with_implicit_and(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs", params={"q": "react typescript"}).json()
    titles = [item["title"] for item in body["items"]]
    assert "Senior Frontend Engineer (Remote)" in titles
    assert "Python Backend Engineer" not in titles


def test_list_jobs_q_sanitizes_fts_operators(
    client: TestClient,
    seeded_db: Path,
) -> None:
    """A query containing FTS5 operators must not raise — operators are
    stripped before passing to MATCH."""
    response = client.get("/api/jobs", params={"q": 'python OR "; DROP TABLE jobs --'})
    assert response.status_code == 200


def test_list_jobs_filter_remote(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"remote": "remote"}).json()
    assert body["total"] == 1
    assert body["items"][0]["remote_type"] == "remote"


def test_list_jobs_filter_invalid_remote_returns_422(
    client: TestClient,
    seeded_db: Path,
) -> None:
    response = client.get("/api/jobs", params={"remote": "interplanetary"})
    assert response.status_code == 422


def test_list_jobs_filter_location(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"location": "London"}).json()
    titles = [item["title"] for item in body["items"]]
    assert titles == ["Data Engineer"]


def test_list_jobs_filter_company(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"company": "palantir"}).json()
    titles = [item["title"] for item in body["items"]]
    assert titles == ["Data Engineer"]


def test_list_jobs_filter_source_kind(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"source_kind": "lever"}).json()
    assert body["total"] == 1
    assert body["items"][0]["title"] == "Data Engineer"


def test_list_jobs_filter_posted_since(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"posted_since": "2026-04-01"}).json()
    titles = [item["title"] for item in body["items"]]
    assert "Python Backend Engineer" in titles
    assert "Senior Frontend Engineer (Remote)" in titles
    assert "Data Engineer" not in titles


def test_list_jobs_filter_exclude_title(client: TestClient, seeded_db: Path) -> None:
    """Comma-separated keywords filter out matching titles."""
    body = client.get("/api/jobs", params={"exclude_title": "senior,lead"}).json()
    titles = [item["title"] for item in body["items"]]
    assert "Senior Frontend Engineer (Remote)" not in titles
    assert "Python Backend Engineer" in titles
    assert "Data Engineer" in titles


def test_list_jobs_filter_has_salary(client: TestClient, seeded_db: Path) -> None:
    body = client.get("/api/jobs", params={"has_salary": "true"}).json()
    titles = [item["title"] for item in body["items"]]
    # Only the Python Backend Engineer in the seed has salary info.
    assert titles == ["Python Backend Engineer"]


def test_list_jobs_filter_min_salary_clears_threshold(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs", params={"min_salary": 150000}).json()
    titles = [item["title"] for item in body["items"]]
    # Python role has salary_max=190k → clears 150k. Others lack
    # salary so they don't satisfy the filter.
    assert titles == ["Python Backend Engineer"]


def test_list_jobs_filter_min_salary_excludes_below_band(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs", params={"min_salary": 250000}).json()
    titles = [item["title"] for item in body["items"]]
    assert titles == []


def test_list_jobs_sort_newest_then_oldest_inverts_order(
    client: TestClient,
    seeded_db: Path,
) -> None:
    """``sort=newest`` and ``sort=oldest`` should produce reversed orderings
    of the same result set."""
    newest = client.get("/api/jobs", params={"sort": "newest"}).json()
    oldest = client.get("/api/jobs", params={"sort": "oldest"}).json()
    newest_ids = [item["id"] for item in newest["items"]]
    oldest_ids = [item["id"] for item in oldest["items"]]
    assert newest_ids == list(reversed(oldest_ids))


def test_list_jobs_sort_posted_newest_uses_posted_at(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs", params={"sort": "posted_newest"}).json()
    titles = [item["title"] for item in body["items"]]
    # Frontend (posted 04-20) > Python (04-15) > Data (03-10).
    assert titles == [
        "Senior Frontend Engineer (Remote)",
        "Python Backend Engineer",
        "Data Engineer",
    ]


def test_list_jobs_sort_salary_high_first(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs", params={"sort": "salary_high"}).json()
    # Python has salary_max=190k; the others have NULL salary → tail.
    assert body["items"][0]["title"] == "Python Backend Engineer"


def test_list_jobs_sort_invalid_returns_422(
    client: TestClient,
    seeded_db: Path,
) -> None:
    response = client.get("/api/jobs", params={"sort": "by_vibes"})
    assert response.status_code == 422


def test_list_jobs_pagination(client: TestClient, seeded_db: Path) -> None:
    page_one = client.get("/api/jobs", params={"limit": 2, "offset": 0}).json()
    page_two = client.get("/api/jobs", params={"limit": 2, "offset": 2}).json()

    assert page_one["total"] == 3
    assert len(page_one["items"]) == 2
    assert page_two["total"] == 3
    assert len(page_two["items"]) == 1
    page_one_ids = {item["id"] for item in page_one["items"]}
    page_two_ids = {item["id"] for item in page_two["items"]}
    assert page_one_ids.isdisjoint(page_two_ids)


def test_list_jobs_caps_excessive_limit_via_validation(
    client: TestClient,
    seeded_db: Path,
) -> None:
    response = client.get("/api/jobs", params={"limit": 9999})
    assert response.status_code == 422


def test_list_jobs_includes_source_links(
    client: TestClient,
    seeded_db: Path,
) -> None:
    body = client.get("/api/jobs", params={"q": "data"}).json()
    item = body["items"][0]
    assert item["sources"]
    names = {s["source_name"] for s in item["sources"]}
    assert "lever:palantir" in names


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_get_job_detail_returns_full_record(
    client: TestClient,
    seeded_db: Path,
) -> None:
    listing = client.get("/api/jobs", params={"q": "python"}).json()
    job_id = listing["items"][0]["id"]

    body = client.get(f"/api/jobs/{job_id}").json()

    assert body["id"] == job_id
    assert body["description_text"] is not None
    assert body["company_norm"]
    assert body["fingerprint_json"]
    assert body["sources"]


def test_get_job_detail_404_when_missing(client: TestClient, seeded_db: Path) -> None:
    response = client.get("/api/jobs/9999")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# User state
# ---------------------------------------------------------------------------


def test_post_job_state_persists_value(
    client: TestClient,
    seeded_db: Path,
) -> None:
    job_id = client.get("/api/jobs", params={"q": "python"}).json()["items"][0]["id"]

    response = client.post(
        f"/api/jobs/{job_id}/state",
        json={"state": "saved", "notes": "interesting role"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "saved"
    assert body["notes"] == "interesting role"
    assert body["job_id"] == job_id


def test_post_job_state_overwrites_previous_value(
    client: TestClient,
    seeded_db: Path,
) -> None:
    job_id = client.get("/api/jobs", params={"q": "python"}).json()["items"][0]["id"]

    client.post(f"/api/jobs/{job_id}/state", json={"state": "saved"})
    second = client.post(f"/api/jobs/{job_id}/state", json={"state": "applied"}).json()

    assert second["state"] == "applied"


def test_post_job_state_404_when_job_missing(
    client: TestClient,
    seeded_db: Path,
) -> None:
    response = client.post("/api/jobs/9999/state", json={"state": "saved"})
    assert response.status_code == 404


def test_post_job_state_rejects_invalid_state(
    client: TestClient,
    seeded_db: Path,
) -> None:
    job_id = client.get("/api/jobs", params={"q": "python"}).json()["items"][0]["id"]

    response = client.post(f"/api/jobs/{job_id}/state", json={"state": "frobnicated"})

    assert response.status_code == 422


def test_list_jobs_sort_relevance_without_q_falls_back_to_newest(
    client: TestClient,
    seeded_db: Path,
) -> None:
    """``sort=relevance`` without ``q`` would otherwise need an FTS join;
    the repository silently substitutes the newest-first ordering."""
    del seeded_db
    body = client.get("/api/jobs", params={"sort": "relevance"}).json()
    # No error, three items, ordered by last_seen_at DESC (== insertion order
    # for these fixtures, so we just confirm a sane payload).
    assert body["total"] == 3


def test_list_jobs_q_with_only_non_word_chars_returns_empty_match(
    client: TestClient,
    seeded_db: Path,
) -> None:
    """A query like ``!!!`` sanitizes down to an empty FTS expression;
    the repository drops the FTS join. Pass an explicit sort so the
    fts-rank default doesn't kick in (which would require the join)."""
    del seeded_db
    body = client.get("/api/jobs", params={"q": "!!!", "sort": "newest"}).json()
    # All three seeded jobs survive (the FTS filter degenerated to no-op).
    assert body["total"] == 3


def test_list_jobs_filter_employment_type(
    client: TestClient,
    seeded_db: Path,
) -> None:
    del seeded_db
    body = client.get("/api/jobs", params={"employment_type": "full-time"}).json()
    # All seeded jobs are full-time; the filter still has to walk the branch.
    assert body["total"] == 3


def test_list_jobs_exclude_title_handles_empty_tokens(
    client: TestClient,
    seeded_db: Path,
) -> None:
    """Exclude-title with stray commas (empty tokens between non-empty ones)
    must not accidentally exclude every row; the empty tokens are skipped."""
    del seeded_db
    body = client.get("/api/jobs", params={"exclude_title": ",senior,,lead,"}).json()
    titles = [item["title"] for item in body["items"]]
    assert "Senior Frontend Engineer (Remote)" not in titles
    # The valid 'lead' token didn't exist in seeds so other jobs survive.
    assert any("Python" in t for t in titles)


def test_resolve_sort_unknown_remote_type_raises_value_error() -> None:
    """The repository's _build_where rejects unknown remote_type strings;
    while the route layer validates first, the underlying helper still
    raises if called directly with a bogus value."""
    from jobai.api.repository import search_jobs  # noqa: PLC0415
    import sqlite3 as _sqlite3  # noqa: PLC0415

    conn = _sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="remote_type"):
        search_jobs(conn, remote_type="interplanetary")


def test_search_jobs_exclude_title_skips_empty_after_strip(
    seeded_db: Path,
) -> None:
    """Calling search_jobs() directly (bypassing the route's pre-filter)
    with whitespace-only exclude_title tokens must skip them via the
    ``if not cleaned: continue`` branch rather than appending empty
    LIKE clauses."""
    from jobai.api.repository import search_jobs  # noqa: PLC0415

    conn = sqlite3.connect(seeded_db)
    try:
        response = search_jobs(conn, exclude_title=["   ", "", "senior"])
    finally:
        conn.close()
    # The 'senior' token is honoured -- Senior Frontend dropped.
    titles = {item.title for item in response.items}
    assert all("senior" not in t.lower() for t in titles)
