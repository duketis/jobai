"""Recording fetcher: persists every successful fetch as a raw_responses row.

The runner wraps the underlying tier-1/2/3 fetcher in a
:class:`RecordingFetcher` before passing it to the source. The source
treats it as any other Fetcher; the recording happens transparently.

The decorator pattern keeps the underlying fetchers focused on HTTP
behaviour; persistence is a cross-cutting concern that lives here. If
recording becomes more complex (compression choice, deduplication of
identical bodies across runs, async writes) it stays in this module.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from jobai.fetcher.base import Fetcher, Response


class RecordingFetcher:
    """Wraps a Fetcher; writes each successful Response to ``raw_responses``."""

    def __init__(
        self,
        inner: Fetcher,
        *,
        conn: sqlite3.Connection,
        run_id: int,
        source_id: int,
        retention_days: int = 30,
    ) -> None:
        self._inner = inner
        self._conn = conn
        self._run_id = run_id
        self._source_id = source_id
        self._retention = timedelta(days=retention_days)

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109
        wait_for_selector: str | None = None,
    ) -> Response:
        response = await self._inner.fetch(
            url,
            method=method,
            headers=headers,
            json=json,
            timeout=timeout,
            wait_for_selector=wait_for_selector,
        )
        if response.is_ok:
            self._record(response)
        return response

    async def run_in_page(self, *args: Any, **kwargs: Any) -> Response:
        """Forward to the inner fetcher's ``run_in_page`` (browser-only).

        Sources that drive Playwright workflows reach for this method;
        we still record the resulting Response if the inner fetcher
        implements it. Raises :class:`AttributeError` if the inner
        fetcher is HTTP-only — same behaviour as calling
        ``run_in_page`` on plain HttpFetcher.
        """
        method = self._inner.run_in_page  # type: ignore[attr-defined]
        response: Response = await method(*args, **kwargs)
        if response.is_ok:
            self._record(response)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()

    def _record(self, response: Response) -> None:
        body_gz = gzip.compress(response.body)
        body_sha256 = hashlib.sha256(response.body).hexdigest()
        expires_at = (datetime.now(tz=UTC) + self._retention).isoformat()

        self._conn.execute(
            "INSERT INTO raw_responses "
            "(run_id, source_id, url, fetched_at, status_code, "
            " headers_json, body_gz, body_sha256, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._run_id,
                self._source_id,
                response.url,
                response.fetched_at.isoformat(),
                response.status_code,
                _serialize_headers(response.headers),
                body_gz,
                body_sha256,
                expires_at,
            ),
        )
        self._conn.commit()


def _serialize_headers(headers: Mapping[str, str]) -> str:
    """Serialize headers for storage. Lower-cased keys for consistency."""
    return json.dumps({k.lower(): v for k, v in headers.items()}, sort_keys=True)
