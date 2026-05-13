"""Tests for the Lever source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.lever import LeverFetchError, LeverSource

_FIXTURE = (Path(__file__).parent / "fixtures" / "lever_palantir.json").read_text(encoding="utf-8")


def test_lever_source_name_includes_account() -> None:
    source = LeverSource(account="palantir")
    assert source.name == "lever:palantir"
    assert source.kind == "lever"


async def test_discover_yields_one_job_per_array_entry() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/palantir").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = LeverSource(account="palantir")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert len(jobs) == 5
    assert all(j.source_external_id for j in jobs)
    assert all(j.title for j in jobs)


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/palantir").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = LeverSource(account="palantir")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    first = jobs[0]
    assert first.company == "palantir"
    assert first.apply_url.startswith("http")
    assert first.description_html is not None
    assert first.description_text is not None
    assert first.location_raw is not None


async def test_discover_normalises_workplace_type() -> None:
    payload = (
        "["
        '{"id":"a","text":"A","applyUrl":"https://example.com/a",'
        '"workplaceType":"remote","categories":{"location":"X"},'
        '"createdAt":1700000000000},'
        '{"id":"b","text":"B","applyUrl":"https://example.com/b",'
        '"workplaceType":"hybrid","categories":{"location":"X"},'
        '"createdAt":1700000000000},'
        '{"id":"c","text":"C","applyUrl":"https://example.com/c",'
        '"workplaceType":"on-site","categories":{"location":"X"},'
        '"createdAt":1700000000000}'
        "]"
    )
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/x").mock(
            return_value=httpx.Response(200, text=payload),
        )

        source = LeverSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["a"].remote_type == "remote"
    assert by_id["b"].remote_type == "hybrid"
    assert by_id["c"].remote_type == "onsite"


async def test_discover_converts_created_at_to_iso() -> None:
    """Lever's createdAt is a millisecond Unix timestamp; we store ISO 8601."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/palantir").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = LeverSource(account="palantir")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    for job in jobs:
        if job.posted_at is not None:
            # Sanity: ISO 8601 form starts with YYYY-MM-DD
            assert job.posted_at[4] == "-"
            assert job.posted_at[7] == "-"


async def test_discover_preserves_raw_payload() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/palantir").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = LeverSource(account="palantir")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert "categories" in jobs[0].raw_data


async def test_discover_raises_on_non_2xx_status() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/missing").mock(
            return_value=httpx.Response(404, text="Not Found"),
        )

        source = LeverSource(account="missing")
        async with HttpFetcher() as fetcher:
            with pytest.raises(LeverFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.status_code == 404
    assert excinfo.value.account == "missing"


async def test_discover_handles_empty_array() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/empty").mock(
            return_value=httpx.Response(200, text="[]"),
        )

        source = LeverSource(account="empty")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs == []


async def test_discover_skips_non_dict_array_entries() -> None:
    """Defensive: if the API ever returns mixed garbage, we skip non-dicts."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/x").mock(
            return_value=httpx.Response(
                200,
                text=(
                    '[{"id":"1","text":"OK","applyUrl":"https://e.com/1","createdAt":0},'
                    '"unexpected string",null]'
                ),
            ),
        )

        source = LeverSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert len(jobs) == 1


async def test_discover_raises_when_payload_is_not_a_list() -> None:
    """A 200 with a non-list (eg dict envelope) at the top is a wire-
    format mismatch; raise rather than silently yield nothing."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.lever.co/v0/postings/wrong").mock(
            return_value=httpx.Response(200, json={"oops": True}),
        )
        source = LeverSource(account="wrong")
        async with HttpFetcher() as fetcher:
            with pytest.raises(LeverFetchError):
                async for _ in source.discover(fetcher):
                    pass


def test_normalise_workplace_type_returns_none_when_no_keyword_matches() -> None:
    """When the workplace string mentions none of remote / hybrid /
    onsite / office, the helper returns None."""
    from jobai.sources.lever import _normalise_workplace_type  # noqa: PLC0415

    assert _normalise_workplace_type("on the moon") is None
    assert _normalise_workplace_type(None) is None  # non-string


def test_extract_commitment_handles_non_dict_and_non_string_value() -> None:
    """``categories`` may be missing entirely or contain a non-string
    'commitment' value; both branches return None."""
    from jobai.sources.lever import _extract_commitment  # noqa: PLC0415

    assert _extract_commitment(None) is None
    assert _extract_commitment("not-a-dict") is None
    assert _extract_commitment({"commitment": 42}) is None
    assert _extract_commitment({"commitment": "Full-time"}) == "Full-time"


def test_extract_created_at_handles_non_numeric_and_overflow() -> None:
    """createdAt that isn't an int/float -> None; an absurdly large value
    that overflows fromtimestamp -> None (the OverflowError except branch)."""
    from jobai.sources.lever import _extract_created_at  # noqa: PLC0415

    assert _extract_created_at({}) is None
    assert _extract_created_at({"createdAt": "not-a-number"}) is None
    # Year ~3000+ ms timestamp overflows on some platforms; use a value
    # well beyond what fromtimestamp can handle.
    assert _extract_created_at({"createdAt": 10**20}) is None
    # Sanity: a normal value still works.
    assert _extract_created_at({"createdAt": 0}) is not None
