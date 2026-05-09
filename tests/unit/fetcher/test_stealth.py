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


def test_build_stealth_fetcher_default_user_agent_is_clean_browser_string() -> None:
    """The default UA must NOT carry any bot-identifying tokens
    (``jobai/...``, ``patchright``, ``bot``, etc.) - Cloudflare's
    strict-mode detection fires on any non-browser string. Tier
    distinction lives in the ``raw_responses`` table now, not the UA."""
    fetcher = build_stealth_fetcher()
    _, driver = _peek(fetcher)
    user_agent, _, _ = _driver_internals(driver)
    lowered = user_agent.lower()
    for forbidden in ("patchright", "jobai", "bot", "scraper", "crawler", "spider"):
        assert forbidden not in lowered, f"UA must not include {forbidden!r}: {user_agent!r}"
    # Must look like a real Chrome on macOS.
    assert "mozilla" in lowered
    assert "chrome" in lowered
    assert "applewebkit" in lowered


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


def test_build_stealth_fetcher_persistent_session_off_by_default() -> None:
    """Per-fetch contexts are right for the common case - sharing a
    context across unrelated sources risks cookie/auth pollution.
    Persistent mode must be opt-in."""
    fetcher = build_stealth_fetcher()
    _, driver = _peek(fetcher)
    assert driver._persistent_session is False


def test_build_stealth_fetcher_propagates_persistent_session_flag() -> None:
    """``persistent_session=True`` must reach the underlying driver so
    Cloudflare-protected sources (NSW iworkfor) get a long-lived
    context that holds the cleared TLS handshake state."""
    fetcher = build_stealth_fetcher(persistent_session=True)
    _, driver = _peek(fetcher)
    assert driver._persistent_session is True
