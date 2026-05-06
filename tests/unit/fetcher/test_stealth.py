"""Tests for the tier-3 stealth fetcher (Patchright shim)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

from jobai.fetcher.browser import BrowserFetcher, PlaywrightDriver
from jobai.fetcher.stealth import build_stealth_fetcher


def _peek(fetcher: BrowserFetcher) -> tuple[float, PlaywrightDriver]:
    """Reach into the fetcher's private state for white-box assertions."""
    return fetcher._timeout, cast(PlaywrightDriver, fetcher._driver)


def _driver_internals(driver: PlaywrightDriver) -> tuple[str, bool, Any]:
    return (
        driver._user_agent,
        driver._headless,
        driver._factory,
    )


def test_build_stealth_fetcher_returns_browser_fetcher() -> None:
    fetcher = build_stealth_fetcher()
    assert isinstance(fetcher, BrowserFetcher)


def test_build_stealth_fetcher_uses_patchright_factory() -> None:
    """The driver must be built with Patchright's async_playwright."""
    with patch("jobai.fetcher.stealth.patchright_playwright") as mock_factory:
        fetcher = build_stealth_fetcher()
        _, driver = _peek(fetcher)
        _, _, factory = _driver_internals(driver)
        assert factory is mock_factory


def test_build_stealth_fetcher_default_user_agent_marks_patchright() -> None:
    """The UA includes 'patchright' so traffic logs can distinguish tiers."""
    fetcher = build_stealth_fetcher()
    _, driver = _peek(fetcher)
    user_agent, _, _ = _driver_internals(driver)
    assert "patchright" in user_agent.lower()


def test_build_stealth_fetcher_accepts_overrides() -> None:
    fetcher = build_stealth_fetcher(
        timeout=45.0,
        headless=False,
        user_agent="custom-ua",
    )
    timeout, driver = _peek(fetcher)
    user_agent, headless, _ = _driver_internals(driver)
    assert timeout == 45.0
    assert user_agent == "custom-ua"
    assert headless is False
