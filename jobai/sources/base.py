"""Source base class and the canonical per-source job shape.

A source pulls jobs from one provider (an ATS like Greenhouse, or a
single hostile site like Seek) and yields :class:`NormalizedJob`
instances. Sources implement either single-stage discovery (fetch one
listing, parse all jobs) or two-stage (fetch listing, then per-job
detail pages); the runner does not care which.

:class:`NormalizedJob` is *per-source canonical*, not the cross-source
deduplicated shape. The dedup pipeline takes NormalizedJob objects and
produces rows in the ``jobs`` table; this module only deals with what a
single source surfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from jobai.fetcher.base import Fetcher


@dataclass(frozen=True, slots=True)
class NormalizedJob:
    """A job in per-source canonical shape.

    Maps roughly to a row in the ``jobs_raw`` table (after JSON
    serialisation of the fields). Cross-source dedup happens later;
    this object knows nothing about other sources.

    Optional fields are common because not every provider exposes every
    field — Greenhouse omits salary, Lever omits structured location,
    LinkedIn guest mode omits employment type, etc. The runner
    preserves the original payload in :attr:`raw_data` so we can
    re-extract fields later without re-scraping.
    """

    source_external_id: str
    title: str
    company: str
    apply_url: str
    raw_data: dict[str, Any]
    location_raw: str | None = None
    location_country: str | None = None
    location_city: str | None = None
    remote_type: str | None = None
    employment_type: str | None = None
    posted_at: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    description_text: str | None = None
    description_html: str | None = None
    extra_tags: tuple[str, ...] = field(default_factory=tuple)


class BaseSource(ABC):
    """Abstract base class for every source.

    A concrete source declares its identity via the ``kind`` and
    ``account`` attributes (set in ``__init__`` or as class-level
    constants where appropriate) and implements :meth:`discover`. The
    fully-qualified ``name`` defaults to ``"{kind}:{account}"``.
    """

    kind: str
    account: str

    @property
    def name(self) -> str:
        """Human-and-machine-readable identifier, e.g. ``greenhouse:atlassian``."""
        return f"{self.kind}:{self.account}" if self.account else self.kind

    @abstractmethod
    def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        """Yield jobs from this source.

        Implementations are async generators; they may perform any
        number of fetch calls (single listing, listing + N per-job
        details, paginated, etc.) as long as the Fetcher Protocol is
        the only side door.

        The runner injects the fetcher tier appropriate for the source
        (HTTP for ATS APIs, browser for Cloudflare-fronted sites,
        stealth for hostile sites). The source itself never picks a
        tier.
        """
