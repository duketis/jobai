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
One dispatch, every gated board — not a Seek special case.
"""

from __future__ import annotations

import contextlib
import logging
from urllib.parse import urlparse

from jobai.fetcher.dispatch import build_fetcher
from jobai.pipeline.description_backfill import RECIPES

_log = logging.getLogger(__name__)

#: Host substring -> the ``RECIPES`` key that knows how to fetch +
#: parse that board's detail page. Only the gated boards need this;
#: everything else resolves to ``None`` (sibling fetches it directly).
_HOST_TO_KIND: tuple[tuple[str, str], ...] = (
    ("seek.com", "seek"),
    ("linkedin.com", "linkedin"),
    ("indeed.", "indeed"),
)


def _kind_for(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    for needle, kind in _HOST_TO_KIND:
        if needle in host:
            return kind
    return None


async def resolve_jd_text(jd_url: str) -> str | None:
    """Return the full JD text for ``jd_url``, or ``None`` to defer.

    ``None`` means "let the sibling fetch this URL itself" — correct
    for every non-gated board. For Seek / LinkedIn / Indeed it runs
    the board's detail recipe on a fresh tier-3 stealth fetcher and
    returns the parsed JD; any failure (no recipe, non-2xx, unparsable
    body, fetcher error) is swallowed and returns ``None`` so the
    chain degrades to the plain URL path rather than failing.
    """
    kind = _kind_for(jd_url)
    if kind is None:
        return None
    recipe = RECIPES.get(kind)
    if recipe is None:  # pragma: no cover - _HOST_TO_KIND keys are RECIPES keys
        return None

    fetcher = build_fetcher(tier=3)
    try:
        response = await fetcher.fetch(
            recipe.fetch_url(jd_url),
            wait_for_selector=recipe.wait_selector,
            wait_until=recipe.wait_until,
        )
    except Exception as exc:  # noqa: BLE001 - any fetch failure → defer, never raise
        _log.info(
            "jd_resolution_fetch_failed",
            extra={"kind": kind, "url": jd_url, "error_class": type(exc).__name__},
        )
        return None
    finally:
        with contextlib.suppress(Exception):
            await fetcher.aclose()

    if not response.is_ok:
        return None
    return recipe.parse(response.text)
