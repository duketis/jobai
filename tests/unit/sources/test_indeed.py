"""Tests for the Indeed (au.indeed.com) source.

Driven against a real fixture cut from a live au.indeed.com search
result page (mosaic-provider-jobcards data island, slimmed to 3
representative entries).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.indeed import IndeedFetchError, IndeedSource, build_query

_FIXTURE = (Path(__file__).parent / "fixtures" / "indeed_python_au.html").read_text(
    encoding="utf-8"
)
_QUERY = "q=python&l=Australia&fromage=7"
_URL = f"https://au.indeed.com/jobs?{_QUERY}"


def _only_first_page(request: httpx.Request) -> httpx.Response:
    """Serve fixture for page 0 (no ``start=`` param), empty thereafter."""
    if "start=" in str(request.url):
        return httpx.Response(200, text="<html><body></body></html>")
    return httpx.Response(200, text=_FIXTURE)


def test_indeed_source_name_includes_query() -> None:
    source = IndeedSource(account=_QUERY)
    assert source.name == f"indeed:{_QUERY}"


def test_build_query_url_encodes_inputs() -> None:
    assert build_query(keywords="python", location="Australia", recency_days=7) == _QUERY


async def test_discover_yields_one_job_per_result() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=_only_first_page)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    assert len(jobs) == 3
    # Real ids from the captured fixture
    assert {j.source_external_id for j in jobs} == {
        "e3eba48ce44f8a99",
        "2903c36943e71c58",
        "31d32114b8222d25",
    }


async def test_discover_canonicalises_apply_url() -> None:
    """``viewjob?jk={id}`` only — no tracking params."""
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=_only_first_page)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    for j in jobs:
        assert j.apply_url == f"https://au.indeed.com/viewjob?jk={j.source_external_id}"
        assert "advn=" not in j.apply_url
        assert "tk=" not in j.apply_url


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=_only_first_page)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    first = by_id["e3eba48ce44f8a99"]
    assert "Python" in first.title
    assert first.company  # whatever Indeed surfaced
    assert first.location_country == "Australia"


async def test_discover_returns_empty_when_data_island_missing() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(
            return_value=httpx.Response(200, text="<html><body><p>nope</p></body></html>")
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]
    assert jobs == []


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(403))
        async with HttpFetcher() as fetcher:
            with pytest.raises(IndeedFetchError) as excinfo:
                async for _ in IndeedSource(account=_QUERY).discover(fetcher):
                    pass
    assert excinfo.value.status_code == 403


async def test_discover_walks_multiple_pages_and_dedups() -> None:
    """Indeed paginates via ``&start=N`` (offset, 10 per page)."""
    new_entry = {
        "jobkey": "newjob123",
        "displayTitle": "Brand New Role",
        "company": "Other Co",
        "formattedLocation": "Sydney NSW",
    }

    def page_for(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        if start == 0:
            return httpx.Response(200, text=_FIXTURE)
        if start == 10:
            payload = (
                '<html><body><script>window.mosaic.providerData["mosaic-provider-jobcards"]='
                '{"metaData":{"mosaicProviderJobCardsModel":{"results":['
                '{"jobkey":"newjob123","displayTitle":"Brand New Role",'
                '"company":"Other Co","formattedLocation":"Sydney NSW"}'
                "]}}};window.next=true;</script></body></html>"
            )
            assert new_entry  # silence unused-var; helps debugging
            return httpx.Response(200, text=payload)
        return httpx.Response(200, text="<html><body></body></html>")

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY, max_pages=5).discover(fetcher)]

    ids = {j.source_external_id for j in jobs}
    assert "newjob123" in ids
    # Page-0 fixture ids stay in
    assert "e3eba48ce44f8a99" in ids


async def test_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        IndeedSource(account=_QUERY, max_pages=0)


async def test_discover_stops_silently_on_mid_walk_failure() -> None:
    """A non-2xx after page 0 ends the walk; everything yielded so far survives."""

    calls = {"n": 0}

    def page_for(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text=_FIXTURE)
        return httpx.Response(500, text="server-side")

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY, max_pages=5).discover(fetcher)]
    # Page 0's fixture jobs are preserved despite page 1 erroring out.
    assert len(jobs) > 0


async def test_discover_walks_to_max_pages_with_only_new_cards() -> None:
    """Exhausting all max_pages requires every page to add a new id."""

    def page_for(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        payload = (
            "<html><body><script>"
            'window.mosaic.providerData["mosaic-provider-jobcards"]='
            f'{{"metaData":{{"mosaicProviderJobCardsModel":{{"results":[{{'
            f'"jobkey":"unique-{start}","title":"R {start}","company":"C","formattedLocation":"L"'
            f"}}]}}}}}};window.next=true;</script></body></html>"
        )
        return httpx.Response(200, text=payload)

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY, max_pages=2).discover(fetcher)]
    assert {j.source_external_id for j in jobs} == {"unique-0", "unique-10"}


async def test_discover_skips_duplicate_job_ids_across_pages() -> None:
    """When page 1 yields an id we already saw on page 0, ``continue`` skips
    it (no yield, no duplicate canonical job)."""

    def page_for(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        # Both pages return the SAME job id; page 1's dupe must be skipped.
        payload = (
            "<html><body><script>"
            'window.mosaic.providerData["mosaic-provider-jobcards"]='
            '{"metaData":{"mosaicProviderJobCardsModel":{"results":[{'
            '"jobkey":"shared-id","title":"R","company":"C","formattedLocation":"L"'
            "}]}}};window.next=true;</script></body></html>"
        )
        del start
        return httpx.Response(200, text=payload)

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="au.indeed.com", path="/jobs").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY, max_pages=3).discover(fetcher)]
    # Only one canonical job emitted despite the same id appearing twice.
    assert len(jobs) == 1


def test_extract_results_returns_empty_when_no_path_matches() -> None:
    """When neither the mosaic nor the initial-data payload yields a
    results list at any of the known paths, the extractor returns []."""
    from jobai.sources.indeed import _extract_results  # noqa: PLC0415

    # No script island anywhere -> both _parse_* return None -> returns [].
    assert _extract_results("<html></html>") == []


def test_extract_results_falls_through_paths_when_payload_has_no_match() -> None:
    """When the payload is a dict but doesn't carry a results list at any
    of the searched paths, the inner for-loop completes without
    returning and the outer falls back to the next payload."""
    from jobai.sources.indeed import _extract_results  # noqa: PLC0415

    html = (
        "<html><body><script>window._initialData={"
        '"some": "other", "data": {"no": "results"}'
        "};</script></body></html>"
    )
    assert _extract_results(html) == []


def test_extract_payload_returns_none_for_no_match() -> None:
    """When the anchor regex doesn't match the HTML, _extract_payload returns None."""
    from jobai.sources.indeed import (  # noqa: PLC0415
        _INITIAL_DATA_START_RE,
        _extract_payload,
    )

    assert _extract_payload("<html><body>nothing</body></html>", _INITIAL_DATA_START_RE) is None


def test_extract_payload_returns_none_for_malformed_json() -> None:
    from jobai.sources.indeed import (  # noqa: PLC0415
        _INITIAL_DATA_START_RE,
        _extract_payload,
    )

    # JSON-like brace-balanced shape with invalid contents -> JSONDecodeError -> None.
    html = '<html><body><script>window._initialData={"a": not-json}</script></body></html>'
    assert _extract_payload(html, _INITIAL_DATA_START_RE) is None


def test_extract_payload_returns_none_when_braces_unbalanced() -> None:
    """When the regex matches and ``{`` follows but the literal never
    closes (truncated HTML mid-script), _extract_balanced_object
    returns None and so does the extractor (line 149)."""
    from jobai.sources.indeed import (  # noqa: PLC0415
        _INITIAL_DATA_START_RE,
        _extract_payload,
    )

    # Note: no closing brace anywhere. The regex matches the prefix and
    # finds `{`, but _extract_balanced_object walks to the end without
    # finding the matching close.
    html = '<html><body><script>window._initialData={"a": 1, "b": 2'
    assert _extract_payload(html, _INITIAL_DATA_START_RE) is None


def test_extract_balanced_object_handles_no_open_brace() -> None:
    from jobai.sources.indeed import _extract_balanced_object  # noqa: PLC0415

    assert _extract_balanced_object("nothing-here", 0) is None
    # start position out of bounds.
    assert _extract_balanced_object("abc", 100) is None


def test_extract_balanced_object_returns_none_when_unbalanced() -> None:
    from jobai.sources.indeed import _extract_balanced_object  # noqa: PLC0415

    # An opening brace with no closing brace ever -> None.
    assert _extract_balanced_object("{unclosed", 0) is None


def test_parse_job_returns_none_when_required_pieces_missing() -> None:
    from jobai.sources.indeed import _parse_job  # noqa: PLC0415

    assert _parse_job({}) is None
    assert _parse_job({"jobkey": "k1"}) is None  # no title
    assert _parse_job({"title": "t"}) is None  # no key
    assert _parse_job({"jobkey": "k1", "title": "   "}) is None  # blank title


def test_resolve_apply_url_falls_back_to_explicit_link_or_base() -> None:
    from jobai.sources.indeed import _resolve_apply_url  # noqa: PLC0415

    # No jobkey, explicit link -> joined onto base URL.
    out = _resolve_apply_url({"viewJobLink": "/viewjob?jk=ABC"}, None)
    assert "ABC" in out
    # No jobkey AND no explicit link -> bare base URL.
    out = _resolve_apply_url({}, None)
    assert out.startswith("http")


def test_city_from_handles_empty_and_non_string_inputs() -> None:
    from jobai.sources.indeed import _city_from  # noqa: PLC0415

    assert _city_from(None) is None
    assert _city_from("") is None
    assert _city_from(",") is None
    assert _city_from("Sydney, NSW") == "Sydney"


def test_remote_from_pulls_value_from_remote_work_model_or_location() -> None:
    from jobai.sources.indeed import _remote_from  # noqa: PLC0415

    # remoteWorkModel attr matches.
    assert _remote_from({"remoteWorkModel": "REMOTE_FRIENDLY"}, "Sydney") == "remote"
    assert _remote_from({"workArrangementType": "HYBRID"}, "Sydney") == "hybrid"
    # No attr -> falls back to location string.
    assert _remote_from({}, "Hybrid - Sydney") == "hybrid"
    assert _remote_from({}, "Remote, AU") == "remote"
    # Neither matches -> None.
    assert _remote_from({}, "Sydney NSW") is None
    # Non-string remoteWorkModel + non-string location -> None.
    assert _remote_from({"remoteWorkModel": 42}, None) is None
    # remoteWorkModel is a string but mentions neither -> falls through.
    assert _remote_from({"remoteWorkModel": "IN_PERSON"}, "Sydney NSW") is None


def test_normalise_str_strips_strings_and_joins_lists() -> None:
    from jobai.sources.indeed import _normalise_str  # noqa: PLC0415

    assert _normalise_str("  hi  ") == "hi"
    assert _normalise_str("   ") is None
    assert _normalise_str(["a", "b", "c"]) == "a, b, c"
    # List with all non-strings collapses to empty -> None.
    assert _normalise_str([1, 2]) is None
    # Non-list non-string -> None.
    assert _normalise_str(42) is None


def test_parse_salary_helpers_return_none_when_text_has_no_numbers() -> None:
    from jobai.sources.indeed import (  # noqa: PLC0415
        _parse_salary_max,
        _parse_salary_min,
    )

    assert _parse_salary_min(None) is None
    assert _parse_salary_min("salary negotiable") is None
    assert _parse_salary_max("only one number $100k") is None
    # Min works on a single-number salary; max needs two.
    assert _parse_salary_min("$80,000 to $120,000") == 80_000
    assert _parse_salary_max("$80,000 to $120,000") == 120_000


def test_to_int_returns_none_for_non_digit_token() -> None:
    from jobai.sources.indeed import _to_int  # noqa: PLC0415

    assert _to_int("not-a-number") is None
    assert _to_int("85") == 85_000  # thousands shorthand
    assert _to_int("85,000") == 85_000
