"""Fetch + parse a LinkedIn job-detail page into plain JD text.

The public ``/jobs/view/<slug>-<id>`` page carries the full JD but
also auth-wall markup, so handing that URL straight to the sibling
renderers gets a thin/blocked result. The guest *fragment* endpoint
``/jobs-guest/jobs/api/jobPosting/<id>`` returns the same description
with **no auth wall** (verified live 2026-05-18: HTTP 200, full
body). :func:`guest_jd_url` rewrites any LinkedIn job URL onto that
fragment so the tailor JD-resolver can pull a clean JD on jobai's
stealth tier — exactly the role :mod:`jobai.sources.seek_detail`
plays for Seek.

The parser is shared with the scheduled description backfill
(``RECIPES["linkedin"]``) so the LinkedIn description selectors live
in one place.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from selectolax.parser import HTMLParser

from jobai.fetcher.base import Fetcher

#: LinkedIn embeds the numeric job id in the detail-path slug
#: (``/jobs/view/<slug>-<id>`` or bare ``/jobs/view/<id>``) and in the
#: search page's ``currentJobId`` param. 8+ digits avoids matching
#: short numbers that appear in slugs ("web3", "2024").
_VIEW_ID_RE = re.compile(r"/jobs/view/(?:[^/?#]*-)?(\d{8,})")
_CURRENT_ID_RE = re.compile(r"[?&]currentJobId=(\d{8,})")

_GUEST_JD_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"


def guest_jd_url(url: str) -> str | None:
    """Rewrite a LinkedIn job URL onto the auth-free guest fragment.

    Returns ``None`` for non-LinkedIn hosts or LinkedIn URLs with no
    extractable numeric job id (caller falls back to the plain path).
    """
    host = urlparse(url).netloc.lower()
    if "linkedin.com" not in host:
        return None
    match = _VIEW_ID_RE.search(url) or _CURRENT_ID_RE.search(url)
    if match is None:
        return None
    return _GUEST_JD_URL.format(job_id=match.group(1))


def parse_linkedin_description(html: str) -> str | None:
    """Extract the JD text from a LinkedIn detail / guest-fragment DOM.

    LinkedIn renders the description into ``div.description__text``;
    ``show-more-less-html__markup`` is its inner wrapper. Either
    selector reaches the same content. Returns ``None`` when neither
    is present (auth wall, expired post, markup change) or empty.
    """
    tree = HTMLParser(html)
    node = tree.css_first("div.description__text") or tree.css_first(
        "div.show-more-less-html__markup",
    )
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


async def fetch_linkedin_jd_text(url: str, fetcher: Fetcher) -> str | None:
    """Fetch a LinkedIn job's JD text via the guest fragment, or ``None``.

    Non-LinkedIn URLs and id-less LinkedIn URLs short-circuit to
    ``None`` without a fetch. Any non-2xx response or unparsable body
    yields ``None`` (soft failure: the tailor chain falls back to the
    plain URL path) rather than raising. The caller owns the fetcher.
    """
    target = guest_jd_url(url)
    if target is None:
        return None
    response = await fetcher.fetch(target)
    if not response.is_ok:
        return None
    return parse_linkedin_description(response.text)
