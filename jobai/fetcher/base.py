"""Fetcher Protocol and Response dataclass.

The Fetcher protocol is structural: any object exposing the right async
methods satisfies it, no inheritance required. This keeps the three
tiers (HTTP / browser / stealth) cleanly decoupled — none of them needs
to know about the others.

Sources receive a Fetcher and call :meth:`Fetcher.fetch`; the runner
chooses which concrete tier to inject based on the source's declared
default tier and runtime escalation state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Response:
    """A fetched HTTP response.

    Immutable so it can safely be passed across boundaries (parsed,
    archived, replayed in tests). The body is stored as raw bytes so
    parsers can decode with the appropriate encoding; :attr:`text` is
    a convenience for the common UTF-8 case.
    """

    url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def is_ok(self) -> bool:
        """True for 2xx responses."""
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        """The body decoded as UTF-8, replacing invalid bytes."""
        return self.body.decode("utf-8", errors="replace")


@runtime_checkable
class Fetcher(Protocol):
    """An async HTTP fetcher.

    Implementations:
      * :class:`jobai.fetcher.http.HttpFetcher` — tier 1, plain HTTP.
      * (browser tier — added in a later phase)
      * (stealth tier — added in a later phase)

    Concrete implementations should also work as async context managers
    so callers can use ``async with HttpFetcher() as fetcher:``.
    """

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109  - delegates to httpx, not asyncio.timeout
    ) -> Response:
        """Issue a request and return the :class:`Response`.

        Implementations must return a Response even for non-2xx
        statuses; raising is reserved for genuine network or protocol
        failures (timeouts, connection resets, malformed responses).
        Status-based decisions belong to the caller.
        """
        ...

    async def aclose(self) -> None:
        """Release any pooled connections held by this fetcher."""
        ...
