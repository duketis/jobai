"""Fetch + parse a Seek job-detail page into plain JD text.

Seek listing scrapes only capture the ~120-char teaser from the
search-results card; the full job description lives on the
``/job/<id>`` detail page. That page is Cloudflare-gated and its SPA
**never reaches network idle** (it long-polls analytics forever), so
the default ``wait_until='networkidle'`` navigation times out. The
working strategy — verified against live Seek — is the tier-3 stealth
fetcher with ``wait_until='domcontentloaded'`` plus an explicit wait
on the JD container selector.

Two consumers share this module so the selector + wait strategy live
in exactly one place:

* the scheduled description backfill (steady state — every Seek row
  eventually gets a real ``description_text``), and
* the tailor chain's on-demand path (immediate — the JD is fetched
  right before a resume/cover-letter run when the catalogue only has
  the teaser).
"""

from __future__ import annotations

from selectolax.parser import HTMLParser

from jobai.fetcher.base import Fetcher

#: The container Seek renders the full job ad into. ``data-automation``
#: is Seek's stable test hook and survives the class-name churn from
#: their frequent Next.js redeploys.
SEEK_JD_SELECTOR = '[data-automation="jobAdDetails"]'


def parse_seek_description(html: str) -> str | None:
    """Extract the JD text from a Seek detail-page DOM snapshot.

    Returns ``None`` when the container is absent (challenge page,
    expired ad, markup change) or empty so callers treat it the same
    as a failed fetch rather than persisting an empty description.
    """
    node = HTMLParser(html).css_first(SEEK_JD_SELECTOR)
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


async def fetch_seek_jd_text(url: str, fetcher: Fetcher) -> str | None:
    """Fetch a Seek detail URL and return its JD text, or ``None``.

    ``fetcher`` must be the tier-3 stealth fetcher — vanilla httpx and
    plain Playwright both get a hard 403 from Seek's Cloudflare. The
    caller owns the fetcher lifecycle. Any non-2xx response or unparsable
    body yields ``None`` (soft failure: the chain falls back to the URL
    path / the backfill skips the row) rather than raising.
    """
    response = await fetcher.fetch(
        url,
        wait_for_selector=SEEK_JD_SELECTOR,
        wait_until="domcontentloaded",
    )
    if not response.is_ok:
        return None
    return parse_seek_description(response.text)
