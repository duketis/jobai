"""Greenhouse ATS source.

Greenhouse exposes a public boards API at::

    https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true

Adding ``content=true`` returns the full HTML description, so this
source is API-only — no per-job HTML fetch is needed.

One :class:`GreenhouseSource` instance covers one board (typically one
company); the runner iterates over the configured boards in
``companies.yaml`` and runs the source per-board.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BOARD_URL_TEMPLATE = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"


class GreenhouseSource(BaseSource):
    """Pulls jobs from one Greenhouse-hosted board.

    The ``account`` constructor argument is the board slug used in the
    public boards-api URL (e.g. ``"atlassian"`` for
    ``boards-api.greenhouse.io/v1/boards/atlassian/jobs``).
    """

    kind = "greenhouse"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = _BOARD_URL_TEMPLATE.format(board=self.account)
        response = await fetcher.fetch(url)

        if not response.is_ok:
            raise GreenhouseFetchError(self.account, response.status_code)

        payload = json.loads(response.body)
        for job_data in payload.get("jobs", []):
            yield _parse_job(job_data, board=self.account)


class GreenhouseFetchError(RuntimeError):
    """Raised when a Greenhouse board returns a non-2xx status."""

    def __init__(self, board: str, status_code: int) -> None:
        super().__init__(f"greenhouse:{board} returned HTTP {status_code}")
        self.board = board
        self.status_code = status_code


def _parse_job(job: dict[str, Any], *, board: str) -> NormalizedJob:
    """Map one Greenhouse job dict to :class:`NormalizedJob`."""
    location = job.get("location") or {}
    location_raw = location.get("name") if isinstance(location, dict) else None

    return NormalizedJob(
        source_external_id=str(job["id"]),
        title=str(job["title"]),
        company=str(job.get("company") or board),
        apply_url=str(job["absolute_url"]),
        raw_data=job,
        location_raw=location_raw,
        remote_type=_infer_remote_type(location_raw),
        posted_at=_extract_posted_at(job),
        description_html=job.get("content"),
    )


def _infer_remote_type(location_raw: str | None) -> str | None:
    """Guess remote / hybrid / onsite from a free-text location string.

    Greenhouse does not expose a structured remote-type field; we infer
    a coarse value from the location text. ``None`` means we declined
    to guess (caller can fall back to job description analysis later).
    """
    if not location_raw:
        return None
    lower = location_raw.lower()
    if "remote" in lower:
        return "remote"
    if "hybrid" in lower:
        return "hybrid"
    return None


def _extract_posted_at(job: dict[str, Any]) -> str | None:
    """Pull a posted-at timestamp from any of the date-ish fields a Greenhouse
    payload exposes. Returns the first non-empty string-like value, or None.
    """
    for key in ("first_published", "updated_at", "created_at"):
        value = job.get(key)
        if isinstance(value, str) and value:
            return value
    return None
