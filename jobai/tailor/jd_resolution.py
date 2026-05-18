"""Resolve a job-description URL to plain JD text on jobai's tiers.

Tailor-from-URL hands an arbitrary JD URL to the sibling renderers.
For the *non-gated* boards (Greenhouse, Lever, Ashby, SmartRecruiters,
Workable, the state-gov sites) the sibling's own fetch is fine, so
this resolver returns ``None`` and the chain falls through to the
plain URL path.

For the *anti-bot-gated* boards we scrape — Seek (Cloudflare),
LinkedIn (auth wall), Indeed (Cloudflare on ``/viewjob``) — a plain
sibling fetch gets a 403 / thin / walled body. jobai already knows
how to get past each one: the per-board detail recipes in
:mod:`jobai.pipeline.description_backfill` (URL transform + wait
strategy + parser), driven here on jobai's tier-3 stealth fetcher.

Some boards aren't anti-bot but are JS single-page apps whose raw
HTML carries no JD at all (Eightfold / Microsoft Careers): those get
a dedicated *detail resolver* that rewrites onto the platform's JSON
API. One dispatch, every board that needs jobai to fetch for it —
not a Seek special case.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from jobai.fetcher.base import Fetcher
from jobai.fetcher.dispatch import build_fetcher
from jobai.pipeline.description_backfill import RECIPES
from jobai.sources.eightfold_detail import fetch_eightfold_jd_text

_log = logging.getLogger(__name__)

#: Host substring -> the ``RECIPES`` key that knows how to fetch +
#: parse that board's detail page. Only the gated boards need this;
#: everything else resolves to ``None`` (sibling fetches it directly).
_HOST_TO_KIND: tuple[tuple[str, str], ...] = (
    ("seek.com", "seek"),
    ("linkedin.com", "linkedin"),
    ("indeed.", "indeed"),
)

#: Host substring -> a ``(url, fetcher) -> str | None`` resolver for
#: JS-SPA boards that need a JSON-API rewrite rather than an HTML
#: recipe. Checked before :data:`_HOST_TO_KIND`.
_DETAIL_RESOLVERS: tuple[tuple[str, Callable[[str, Fetcher], Awaitable[str | None]]], ...] = (
    ("careers.microsoft.com", fetch_eightfold_jd_text),
)


def _kind_for(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    for needle, kind in _HOST_TO_KIND:
        if needle in host:
            return kind
    return None


def _detail_resolver_for(
    url: str,
) -> Callable[[str, Fetcher], Awaitable[str | None]] | None:
    host = urlparse(url).netloc.lower()
    for needle, resolver in _DETAIL_RESOLVERS:
        if needle in host:
            return resolver
    return None


async def resolve_jd_text(jd_url: str) -> str | None:
    """Return the full JD text for ``jd_url``, or ``None`` to defer.

    ``None`` means "let the sibling fetch this URL itself" — correct
    for every non-gated, non-SPA board. For Seek / LinkedIn / Indeed
    it runs the board's detail recipe; for Eightfold (Microsoft
    Careers) it runs the JSON-API resolver. Everything goes through
    one fresh tier-3 stealth fetcher; any failure (no handler,
    non-2xx, unparsable body, fetcher error) is swallowed and returns
    ``None`` so the chain degrades to the plain URL path.
    """
    resolver = _detail_resolver_for(jd_url)
    kind = None if resolver is not None else _kind_for(jd_url)
    if resolver is None and kind is None:
        return None

    fetcher = build_fetcher(tier=3)
    try:
        if resolver is not None:
            return await resolver(jd_url, fetcher)
        recipe = RECIPES.get(kind or "")
        if recipe is None:  # pragma: no cover - _HOST_TO_KIND keys are RECIPES keys
            return None
        response = await fetcher.fetch(
            recipe.fetch_url(jd_url),
            wait_for_selector=recipe.wait_selector,
            wait_until=recipe.wait_until,
        )
        if not response.is_ok:
            return None
        return recipe.parse(response.text)
    except Exception as exc:  # noqa: BLE001 - any fetch failure → defer, never raise
        _log.info(
            "jd_resolution_fetch_failed",
            extra={"url": jd_url, "error_class": type(exc).__name__},
        )
        return None
    finally:
        with contextlib.suppress(Exception):
            await fetcher.aclose()
