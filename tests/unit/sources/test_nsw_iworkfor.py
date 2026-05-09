"""Tests for the NSW iworkfor.nsw.gov.au source.

Driven against a real fixture cut from a live iworkfor.nsw.gov.au
search-results page (3 representative ``article.search-job-card``
entries). The discover loop runs through ``run_in_page`` against a
:class:`FakeBrowserFetcher` since the live SPA paginates client-side
via XHR + an Ant Design pagination control we can't replay through
plain HTTP.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.nsw_iworkfor import (
    NSWIWorkForBlockedError,
    NSWIWorkForFetchError,
    NSWIWorkForSource,
    _extract_job_id,
    _first_segment,
    _is_cloudflare_challenge,
    _page_url,
    _parse_card,
    _parse_info_dl,
    _parse_salary,
)
from tests.unit.sources._browser_fakes import FakeBrowserFetcher, html_response

_FIXTURE = (Path(__file__).parent / "fixtures" / "nsw_iworkfor.html").read_text(encoding="utf-8")
_SLUG = (
    "jobs/all-keywords/all-agencies/all-organisations-entities/"
    "all-categories/all-locations/all-worktypes"
)
_URL = f"https://iworkfor.nsw.gov.au/{_SLUG}"


def test_source_name_includes_slug() -> None:
    source = NSWIWorkForSource(account=_SLUG)
    assert source.name == f"nsw_iworkfor:{_SLUG}"


async def test_discover_rejects_non_browser_fetcher() -> None:
    """The walker drives a Playwright Page via run_in_page; an HTTP-only
    fetcher must fail fast with a clear TypeError."""
    async with HttpFetcher() as fetcher:
        with pytest.raises(TypeError, match="run_in_page"):
            async for _ in NSWIWorkForSource(account=_SLUG).discover(fetcher):
                pass


async def test_discover_yields_one_job_per_card() -> None:
    fetcher = FakeBrowserFetcher(html_response(_FIXTURE))
    jobs = [j async for j in NSWIWorkForSource(account=_SLUG).discover(fetcher)]
    assert {j.source_external_id for j in jobs} == {"576439", "575973", "576521"}


async def test_discover_maps_core_fields() -> None:
    fetcher = FakeBrowserFetcher(html_response(_FIXTURE))
    jobs = [j async for j in NSWIWorkForSource(account=_SLUG).discover(fetcher)]
    by_id = {j.source_external_id: j for j in jobs}
    deputy = by_id["576439"]
    assert deputy.title == "Deputy Commissioner"
    assert deputy.company == "NSW Police Force"
    assert deputy.apply_url.startswith("https://iworkfor.nsw.gov.au/job/")
    assert deputy.location_country == "Australia"
    # Deputy Commissioner has a salary range in the fixture
    assert deputy.salary_min == 373_951
    assert deputy.salary_max == 527_050
    assert deputy.salary_currency == "AUD"


async def test_discover_raises_on_non_2xx() -> None:
    fetcher = FakeBrowserFetcher(html_response("<html/>", status_code=503))
    with pytest.raises(NSWIWorkForFetchError) as excinfo:
        async for _ in NSWIWorkForSource(account=_SLUG).discover(fetcher):
            pass
    assert excinfo.value.status_code == 503
    assert excinfo.value.slug == _SLUG


async def test_discover_returns_empty_on_page_with_no_cards() -> None:
    fetcher = FakeBrowserFetcher(html_response("<html><body><main></main></body></html>"))
    jobs = [j async for j in NSWIWorkForSource(account=_SLUG).discover(fetcher)]
    assert jobs == []


async def test_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        NSWIWorkForSource(account=_SLUG, max_pages=0)


# ---------------------------------------------------------------------------
# Cloudflare challenge detection
# ---------------------------------------------------------------------------

_CF_CHALLENGE_HTML = (
    "<!DOCTYPE html><html><head>"
    "<title>Just a moment...</title>"
    '<meta http-equiv="content-security-policy" content="...challenge-platform...">'
    "</head><body>"
    '<script src="https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1"></script>'
    "<noscript>cf-mitigated</noscript>"
    "</body></html>"
)


def test_is_cloudflare_challenge_recognises_real_interstitial() -> None:
    assert _is_cloudflare_challenge(_CF_CHALLENGE_HTML) is True


def test_is_cloudflare_challenge_does_not_false_positive_on_normal_page() -> None:
    """Body copy that happens to contain 'Just a moment' must not trip
    detection - the title alone isn't enough; we require a CF asset
    marker too."""
    benign = (
        "<!DOCTYPE html><html><head><title>Junior Engineer</title></head>"
        "<body><p>Please wait, just a moment as we load the form.</p></body></html>"
    )
    assert _is_cloudflare_challenge(benign) is False


def test_is_cloudflare_challenge_returns_false_for_empty_input() -> None:
    assert _is_cloudflare_challenge("") is False


async def test_discover_raises_blocked_error_on_cloudflare_challenge() -> None:
    """A 200 OK with the Cloudflare interstitial body must surface as a
    failure, not a silent zero-card success."""
    fetcher = FakeBrowserFetcher(html_response(_CF_CHALLENGE_HTML))
    with pytest.raises(NSWIWorkForBlockedError) as excinfo:
        async for _ in NSWIWorkForSource(account=_SLUG).discover(fetcher):
            pass
    assert excinfo.value.slug == _SLUG


# ---------------------------------------------------------------------------
# Helper-function coverage
# ---------------------------------------------------------------------------


def test_page_url_handles_query_string_in_slug() -> None:
    """If the slug already carries ``?key=val``, the page param uses ``&``."""
    url = _page_url("jobs?cat=tech", page=2)
    assert url == "https://iworkfor.nsw.gov.au/jobs?cat=tech&page=2"


def test_page_url_handles_blank_slug() -> None:
    """No slug -> the bare base URL on page 1."""
    assert _page_url("", page=1) == "https://iworkfor.nsw.gov.au"


def test_extract_job_id_falls_back_to_aria_labelledby() -> None:
    """When the apply path has no trailing id, ``aria-labelledby``
    on the card carries ``job-title-{id}``."""
    html = (
        '<article class="search-job-card" aria-labelledby="job-title-987654">'
        '<a class="search-job-card__title-link" href="/job/no-id">x</a>'
        "</article>"
    )
    card = HTMLParser(html).css_first("article")
    assert card is not None
    assert _extract_job_id("/job/no-id", card) == "987654"


def test_extract_job_id_returns_none_when_neither_source_has_id() -> None:
    html = '<article class="search-job-card"></article>'
    card = HTMLParser(html).css_first("article")
    assert card is not None
    assert _extract_job_id("/job/no-id", card) is None


def test_parse_card_returns_none_when_required_fields_missing() -> None:
    """A card with no title node, no apply link, or no id collapses to
    ``None`` so the discover loop skips it rather than yielding garbage."""
    no_title = HTMLParser('<article class="search-job-card"></article>').css_first("article")
    assert no_title is not None
    assert _parse_card(no_title) is None

    title_only = HTMLParser(
        '<article class="search-job-card">'
        '<h3 class="search-job-card__title">Engineer</h3></article>',
    ).css_first("article")
    assert title_only is not None
    # No apply path -> still None.
    assert _parse_card(title_only) is None


def test_parse_info_dl_returns_empty_when_no_dl() -> None:
    card = HTMLParser('<article class="search-job-card"><div></div></article>').css_first(
        "article",
    )
    assert card is not None
    assert _parse_info_dl(card) == {}


def test_first_segment_handles_empty_or_none() -> None:
    assert _first_segment(None) is None
    assert _first_segment("") is None
    assert _first_segment("Sydney, NSW") == "Sydney"


def test_parse_salary_parses_single_value() -> None:
    """Some NSW listings carry a flat figure rather than a range."""
    low, high, currency = _parse_salary("$120,000")
    assert (low, high, currency) == (120_000, None, "AUD")


def test_parse_salary_handles_unit_dollars() -> None:
    """A bare-int salary like ``250`` (thousands of dollars) is upscaled
    so downstream filters compare like-with-like."""
    low, _, _ = _parse_salary("$250")
    assert low == 250_000


def test_parse_salary_returns_none_when_no_digits() -> None:
    assert _parse_salary("Negotiable") == (None, None, None)
    assert _parse_salary(None) == (None, None, None)


def test_nsw_iworkfor_source_declares_needs_persistent_session() -> None:
    """NSW iworkfor.nsw.gov.au is fronted by Cloudflare strict-challenge
    mode. The cleared ``cf_clearance`` cookie is bound to the TLS
    handshake of the issuing context, so the runner MUST build a
    long-lived shared context. Pin this so a refactor that drops the
    flag gets caught immediately."""
    assert NSWIWorkForSource.needs_persistent_session is True
    assert NSWIWorkForSource(account="jobs").needs_persistent_session is True
