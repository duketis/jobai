"""Construct the right fetcher tier for a given source.

Sources declare a ``default_tier`` (1/2/3); the runner shouldn't care
which tier it is, only that it gets a :class:`Fetcher` that does the
right thing. Centralising tier selection here means callers (CLI,
scheduler) stay terse and the rules live in one place.

Tier shape:

* **1 — HTTP**: plain :class:`HttpFetcher` wrapped in
  :class:`RetryingFetcher`. Cheap; works for ATS APIs.
* **2 — Browser-escalating**: HTTP first via
  :class:`EscalatingFetcher`; transparently switches to a Chromium
  :class:`BrowserFetcher` on 403/429/Cloudflare interstitial. Right
  for sources that *might* serve over plain HTTP but block sometimes.
* **3 — Stealth**: :func:`build_stealth_fetcher` (Patchright). For
  sources that fingerprint vanilla Playwright (LinkedIn, etc.).

All tiers go through :class:`RetryingFetcher` so transient network
hiccups don't turn into hard failures regardless of which tier was
selected.
"""

from __future__ import annotations

from jobai.fetcher.base import Fetcher
from jobai.fetcher.browser import BrowserFetcher
from jobai.fetcher.escalation import EscalatingFetcher
from jobai.fetcher.http import HttpFetcher
from jobai.fetcher.retry import RetryingFetcher
from jobai.fetcher.stealth import build_stealth_fetcher


def build_fetcher(*, tier: int, persistent_session: bool = False) -> Fetcher:
    """Return a configured :class:`Fetcher` for ``tier``.

    Caller is responsible for closing the returned fetcher (call
    ``await fetcher.aclose()`` or use it as an async context manager).

    ``persistent_session=True`` is only meaningful for tier 3
    (stealth) and tells the underlying browser to keep one context
    alive across all fetches in this fetcher's lifetime - required
    for Cloudflare-protected sources where the ``cf_clearance``
    cookie is tied to the TLS handshake of the issuing context.
    """
    if tier == 1:
        return RetryingFetcher(HttpFetcher())
    if tier == 2:
        primary = RetryingFetcher(HttpFetcher())
        return EscalatingFetcher(
            primary=primary,
            fallback_factory=BrowserFetcher,
        )
    if tier == 3:
        return RetryingFetcher(build_stealth_fetcher(persistent_session=persistent_session))
    msg = f"unknown fetcher tier: {tier} (expected 1, 2, or 3)"
    raise ValueError(msg)
