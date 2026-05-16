"""Tests for the tier-2 :class:`BrowserFetcher`.

These exercise the fetcher's contract — argument validation, Response
translation, lifecycle — against a fake driver. The Playwright-backed
:class:`PlaywrightDriver` is a thin wrapper and is covered separately
by an integration test that runs only when Chromium is installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from jobai.fetcher.base import Fetcher, Response
from jobai.fetcher.browser import BrowserFetcher


class _FakeDriver:
    """Minimal driver implementation for unit tests."""

    def __init__(self, response: Response | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._response = response or Response(
            url="https://example.com/jobs",
            status_code=200,
            headers={"content-type": "text/html"},
            body=b"<html><body>OK</body></html>",
            fetched_at=datetime.now(tz=UTC),
        )

    async def fetch_page(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        timeout_ms: float,
        wait_for_selector: str | None = None,
        wait_until: str = "networkidle",
    ) -> Response:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "timeout_ms": timeout_ms,
                "wait_for_selector": wait_for_selector,
                "wait_until": wait_until,
            },
        )
        return self._response

    async def run_in_page(
        self,
        url: str,
        *,
        timeout_ms: float,
        page_script: Any,
    ) -> Response:
        del page_script  # fake driver doesn't run the script
        self.calls.append(
            {"url": url, "timeout_ms": timeout_ms, "kind": "run_in_page"},
        )
        return self._response

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_browser_fetcher_satisfies_fetcher_protocol() -> None:
    fetcher = BrowserFetcher(driver=_FakeDriver())
    assert isinstance(fetcher, Fetcher)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_fetch_returns_driver_response() -> None:
    expected = Response(
        url="https://example.com/jobs",
        status_code=200,
        headers={"x-test": "1"},
        body=b"hello",
        fetched_at=datetime.now(tz=UTC),
    )
    driver = _FakeDriver(response=expected)
    async with BrowserFetcher(driver=driver) as fetcher:
        response = await fetcher.fetch("https://example.com/jobs")
    assert response is expected
    assert response.is_ok
    assert response.text == "hello"


async def test_fetch_passes_headers_and_timeout_to_driver() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver, timeout=15.0) as fetcher:
        await fetcher.fetch(
            "https://example.com/x",
            headers={"X-Run-Id": "abc"},
            timeout=5.0,
        )
    assert driver.calls == [
        {
            "url": "https://example.com/x",
            "headers": {"X-Run-Id": "abc"},
            "timeout_ms": 5_000.0,
            "wait_for_selector": None,
            "wait_until": "networkidle",
        }
    ]


async def test_fetch_uses_default_timeout_when_none_given() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver, timeout=20.0) as fetcher:
        await fetcher.fetch("https://example.com/y")
    assert driver.calls[0]["timeout_ms"] == 20_000.0


async def test_fetch_passes_wait_for_selector_to_driver() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver) as fetcher:
        await fetcher.fetch(
            "https://example.com/spa",
            wait_for_selector="article[data-jobcard]",
        )
    assert driver.calls[0]["wait_for_selector"] == "article[data-jobcard]"


async def test_fetch_defaults_wait_until_to_networkidle() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver) as fetcher:
        await fetcher.fetch("https://example.com/spa")
    assert driver.calls[0]["wait_until"] == "networkidle"


async def test_fetch_passes_wait_until_override_to_driver() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver) as fetcher:
        await fetcher.fetch(
            "https://www.seek.com.au/job/1",
            wait_for_selector='[data-automation="jobAdDetails"]',
            wait_until="domcontentloaded",
        )
    assert driver.calls[0]["wait_until"] == "domcontentloaded"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
async def test_fetch_rejects_non_get_methods(method: str) -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver) as fetcher:
        with pytest.raises(ValueError, match="only supports GET"):
            await fetcher.fetch("https://example.com", method=method)
    assert driver.calls == []


async def test_fetch_rejects_json_payload() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver) as fetcher:
        with pytest.raises(ValueError, match="json"):
            await fetcher.fetch("https://example.com", json={"k": 1})
    assert driver.calls == []


async def test_fetch_rejects_form_data_payload() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver) as fetcher:
        with pytest.raises(ValueError, match="data"):
            await fetcher.fetch("https://example.com", data={"k": "v"})
    assert driver.calls == []


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_aclose_closes_driver() -> None:
    driver = _FakeDriver()
    fetcher = BrowserFetcher(driver=driver)
    await fetcher.aclose()
    assert driver.closed is True


async def test_async_context_manager_closes_driver_on_exit() -> None:
    driver = _FakeDriver()
    async with BrowserFetcher(driver=driver):
        assert driver.closed is False
    assert driver.closed is True


async def test_run_in_page_delegates_to_driver() -> None:
    """``BrowserFetcher.run_in_page`` is the public escape hatch -- the
    body is one delegation call, but covering it makes sure the timeout
    + page_script kwargs reach the driver intact."""

    async def _script(_page: Any) -> None:  # pragma: no cover - fake never runs scripts
        return None

    driver = _FakeDriver()
    fetcher = BrowserFetcher(driver=driver, timeout=7.0)
    response = await fetcher.run_in_page(
        "https://example.test/form",
        page_script=_script,
    )
    assert response.status_code == 200  # the fake driver's default response
    assert driver.calls[-1]["url"] == "https://example.test/form"
    assert driver.calls[-1]["kind"] == "run_in_page"
    assert driver.calls[-1]["timeout_ms"] == 7000.0


async def test_run_in_page_uses_explicit_timeout_override() -> None:
    async def _script(_page: Any) -> None:  # pragma: no cover - fake never runs scripts
        return None

    driver = _FakeDriver()
    fetcher = BrowserFetcher(driver=driver, timeout=7.0)
    await fetcher.run_in_page("https://example.test/", timeout=3.0, page_script=_script)
    assert driver.calls[-1]["timeout_ms"] == 3000.0


def test_playwright_driver_init_stores_config() -> None:
    """``PlaywrightDriver.__init__`` is side-effect-free -- it just stores
    config + creates an asyncio.Lock. Cover it without spinning up real
    Chromium (the methods that do are pragma'd)."""
    from jobai.fetcher.browser import PlaywrightDriver  # noqa: PLC0415

    sentinel_factory = object()
    driver = PlaywrightDriver(
        user_agent="x/1",
        headless=False,
        playwright_factory=sentinel_factory,
        persistent_session=True,
    )
    assert driver._user_agent == "x/1"
    assert driver._headless is False
    assert driver._factory is sentinel_factory
    assert driver._persistent_session is True
    assert driver._browser is None
    assert driver._playwright is None
    assert driver._persistent_context is None
