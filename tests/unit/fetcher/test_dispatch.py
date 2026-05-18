"""Tests for the fetcher tier-dispatch factory."""

from __future__ import annotations

from typing import cast

import pytest

from jobai.fetcher.browser import BrowserFetcher, PlaywrightDriver
from jobai.fetcher.dispatch import build_fetcher
from jobai.fetcher.escalation import EscalatingFetcher
from jobai.fetcher.http import HttpFetcher
from jobai.fetcher.ratelimit import RateLimitedFetcher, get_global_limiter
from jobai.fetcher.retry import RetryingFetcher


async def test_tier_1_returns_retrying_http_fetcher() -> None:
    fetcher = build_fetcher(tier=1)
    try:
        assert isinstance(fetcher, RetryingFetcher)
        # The wrapped inner is a plain HTTP fetcher.
        assert isinstance(fetcher._inner, HttpFetcher)
    finally:
        await fetcher.aclose()


async def test_tier_2_returns_escalating_with_browser_fallback() -> None:
    fetcher = build_fetcher(tier=2)
    try:
        assert isinstance(fetcher, EscalatingFetcher)
        # Primary is a retrying HTTP fetcher; fallback factory builds
        # a BrowserFetcher lazily, so we don't validate the fallback
        # type here without forcing a Chromium launch.
        assert isinstance(fetcher._primary, RetryingFetcher)
        # Sanity-check that the factory builds the right type when called.
        fb = fetcher._fallback_factory()
        assert isinstance(fb, BrowserFetcher)
        await fb.aclose()
    finally:
        await fetcher.aclose()


async def test_tier_3_returns_retrying_ratelimited_stealth_fetcher() -> None:
    fetcher = build_fetcher(tier=3)
    try:
        assert isinstance(fetcher, RetryingFetcher)
        # Stealth tier: Retrying -> RateLimited -> BrowserFetcher
        # (Patchright is wire-compatible). RateLimited is innermost so
        # every retry attempt is paced too.
        rl = fetcher._inner
        assert isinstance(rl, RateLimitedFetcher)
        assert isinstance(rl._inner, BrowserFetcher)
        # The limiter is the shared process-global, not a per-fetcher
        # one — that's what makes the rate cap hold across all slugs.
        assert rl._limiter is get_global_limiter()
    finally:
        await fetcher.aclose()


def test_unknown_tier_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown fetcher tier"):
        build_fetcher(tier=4)


async def test_tier_3_persistent_session_reaches_underlying_driver() -> None:
    """``persistent_session=True`` must propagate from build_fetcher
    through the RetryingFetcher wrapper and the BrowserFetcher to the
    PlaywrightDriver - otherwise CF-protected sources still get
    per-fetch contexts and the bypass doesn't kick in."""
    fetcher = build_fetcher(tier=3, persistent_session=True)
    try:
        assert isinstance(fetcher, RetryingFetcher)
        rl = cast(RateLimitedFetcher, fetcher._inner)
        browser_fetcher = cast(BrowserFetcher, rl._inner)
        driver = cast(PlaywrightDriver, browser_fetcher._driver)
        assert driver._persistent_session is True
    finally:
        await fetcher.aclose()


async def test_tier_3_persistent_session_default_is_false() -> None:
    """Per-fetch isolation is the right default - persistent mode is
    opt-in to avoid leaking cookies across unrelated CF-clean sources."""
    fetcher = build_fetcher(tier=3)
    try:
        assert isinstance(fetcher, RetryingFetcher)
        rl = cast(RateLimitedFetcher, fetcher._inner)
        browser_fetcher = cast(BrowserFetcher, rl._inner)
        driver = cast(PlaywrightDriver, browser_fetcher._driver)
        assert driver._persistent_session is False
    finally:
        await fetcher.aclose()
