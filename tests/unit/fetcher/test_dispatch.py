"""Tests for the fetcher tier-dispatch factory."""

from __future__ import annotations

import pytest

from jobai.fetcher.browser import BrowserFetcher
from jobai.fetcher.dispatch import build_fetcher
from jobai.fetcher.escalation import EscalatingFetcher
from jobai.fetcher.http import HttpFetcher
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


async def test_tier_3_returns_retrying_stealth_fetcher() -> None:
    fetcher = build_fetcher(tier=3)
    try:
        assert isinstance(fetcher, RetryingFetcher)
        # The stealth tier wraps a BrowserFetcher under the hood
        # (Patchright is wire-compatible).
        assert isinstance(fetcher._inner, BrowserFetcher)
    finally:
        await fetcher.aclose()


def test_unknown_tier_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown fetcher tier"):
        build_fetcher(tier=4)
