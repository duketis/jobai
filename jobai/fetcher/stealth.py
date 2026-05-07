"""Tier-3 stealth fetcher backed by Patchright.

Patchright is a Playwright fork that ships anti-detection patches
(navigator.webdriver hidden, plugins fingerprint normalised, Chrome
runtime quirks restored, etc.). For sites that fingerprint plain
Playwright — LinkedIn, Indeed, Glassdoor — Patchright slips through
where vanilla Chromium gets a 403 or a "verify you are human" wall.

The API is wire-compatible with :mod:`playwright.async_api`, so the
fetcher reuses :class:`jobai.fetcher.browser.PlaywrightDriver` and
just hands it the Patchright entry point. Keeping this as a thin
shim avoids parallel maintenance of two browser pipelines and means
any future browser-tier improvement (cookie persistence, retry
logic) lands in both at once.
"""

from __future__ import annotations

from patchright.async_api import async_playwright as patchright_playwright

from jobai import __version__
from jobai.fetcher.browser import BrowserFetcher, PlaywrightDriver

#: Suffixed with ``patchright`` so traffic logs distinguish stealth
#: traffic from vanilla browser-tier traffic during incident triage.
_STEALTH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    f"Chrome/120.0.0.0 Safari/537.36 jobai/{__version__} (patchright)"
)


def build_stealth_fetcher(
    *,
    timeout: float = 30.0,
    headless: bool = True,
    user_agent: str = _STEALTH_USER_AGENT,
) -> BrowserFetcher:
    """Construct a :class:`BrowserFetcher` driven by Patchright.

    Returns a regular :class:`BrowserFetcher` so every place that
    accepts the :class:`Fetcher` Protocol works unchanged. The only
    difference from the tier-2 ``BrowserFetcher()`` constructor is
    the underlying playwright-compatible runtime.
    """
    driver = PlaywrightDriver(
        user_agent=user_agent,
        headless=headless,
        playwright_factory=patchright_playwright,
    )
    return BrowserFetcher(timeout=timeout, driver=driver)
