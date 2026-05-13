"""ATS slug auto-discovery from existing apply URLs.

Every Seek / Indeed / LinkedIn job we scrape carries an ``apply_url``
that very often points straight at one of the well-known ATS providers
we already have parsers for (Greenhouse, Lever, Ashby, SmartRecruiters,
Workable). This module mines the canonical ``jobs`` table for those
URLs and extracts the company slug embedded in the path so callers can
diff against ``companies.yaml`` and seed any missing ATS sources.

Why this exists: the per-ATS APIs require knowing the slug per-call
(there's no public 'list every company on Greenhouse' endpoint).
Curating ``companies.yaml`` by hand is the obvious-but-wrong approach
-- the obvious-and-right approach is to read what our aggregator
sources have already discovered and turn that into direct-ATS feeds.
A direct ATS scrape produces structured data (salary, departments,
posted_at, etc.) that the aggregator scrape can't.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Final

#: Mapping of ATS provider -> compiled regex that captures the company slug.
#: Each pattern anchors on the ATS provider's hostname so a stray match
#: in a description field can't fool us.
_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "smartrecruiters": re.compile(
        r"(?:jobs|careers|api)\.smartrecruiters\.com/([A-Za-z0-9_-]+)/",
    ),
    "greenhouse": re.compile(
        r"(?:boards|job-boards)\.greenhouse\.io/([A-Za-z0-9_-]+)/?",
    ),
    "lever": re.compile(r"(?:jobs|api)\.lever\.co/([A-Za-z0-9_-]+)/"),
    "ashby": re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_-]+)"),
    "workable": re.compile(r"apply\.workable\.com/([A-Za-z0-9_-]+)/"),
}


@dataclass(frozen=True)
class SlugCount:
    """One ``(kind, account, count)`` triple from a discovery pass."""

    kind: str
    account: str
    count: int


def discover_slugs(conn: sqlite3.Connection) -> list[SlugCount]:
    """Return every ATS slug found in ``jobs.apply_url``, newest-first by count.

    The returned list contains rows for every (kind, account) pair found
    in the apply URLs, ordered by descending observation count. The
    caller decides what to do with them (eg diff against companies.yaml
    + emit a seed patch).

    The function runs against a SELECT DISTINCT so the cost scales with
    the number of unique URLs, not the total row count.
    """
    rows = conn.execute("SELECT DISTINCT apply_url FROM jobs").fetchall()
    counters: dict[str, Counter[str]] = {kind: Counter() for kind in _PATTERNS}
    for (url,) in rows:
        # ``jobs.apply_url`` is NOT NULL at the schema layer, so this guard
        # exists as belt-and-braces against a future migration that relaxes
        # the constraint.
        if not url:  # pragma: no cover
            continue
        for kind, rx in _PATTERNS.items():
            match = rx.search(url)
            if match is not None:
                counters[kind][match.group(1)] += 1
                # First match wins -- a URL hits at most one ATS host.
                break
    out: list[SlugCount] = []
    for kind, counter in counters.items():
        for account, count in counter.most_common():
            out.append(SlugCount(kind=kind, account=account, count=count))
    return out


def diff_against_seeded(
    discovered: list[SlugCount],
    seeded: dict[str, set[str]],
) -> list[SlugCount]:
    """Return the slugs in ``discovered`` that are NOT in ``seeded``.

    ``seeded`` maps ``kind -> set(account)``. Comparison is
    case-insensitive because ATS APIs treat slugs that way (eg
    SmartRecruiters resolves both ``Canva`` and ``canva`` to the same
    employer); matching case-sensitively would let a user re-seed the
    same company twice and double-scrape it.
    """
    seeded_lower: dict[str, set[str]] = {
        kind: {acct.lower() for acct in accounts} for kind, accounts in seeded.items()
    }
    return [s for s in discovered if s.account.lower() not in seeded_lower.get(s.kind, set())]


def load_seeded_accounts(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Return ``{kind: {account, ...}}`` for every row in ``sources``.

    Used by the CLI's diff command so we don't double-add slugs that
    are already registered (even if they happen to be disabled --
    re-enabling a disabled slug is the user's call).
    """
    by_kind: dict[str, set[str]] = {kind: set() for kind in _PATTERNS}
    rows = conn.execute("SELECT kind, account FROM sources").fetchall()
    for kind, account in rows:
        if kind in by_kind:
            by_kind[kind].add(str(account))
    return by_kind
