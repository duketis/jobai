"""Tests for :class:`EscalatingFetcher`."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from jobai.fetcher.base import Fetcher, Response
from jobai.fetcher.escalation import EscalatingFetcher


class _ScriptedFetcher:
    """A fake fetcher returning canned responses in order.

    Records every call so tests can assert what the wrapper passed
    through.
    """

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109  - matches Fetcher Protocol
    ) -> Response:
        self.calls.append(
            {
                "url": url,
                "method": method,
                "headers": dict(headers or {}),
                "json": json,
                "timeout": timeout,
            },
        )
        return self._responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _resp(status: int, body: bytes = b"<html>ok</html>") -> Response:
    return Response(
        url="https://example.com",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_escalating_fetcher_satisfies_fetcher_protocol() -> None:
    primary = _ScriptedFetcher([])
    fetcher = EscalatingFetcher(primary=primary, fallback_factory=lambda: _ScriptedFetcher([]))
    assert isinstance(fetcher, Fetcher)


# ---------------------------------------------------------------------------
# Happy path: 200 → no escalation
# ---------------------------------------------------------------------------


async def test_200_response_passes_through_without_escalation() -> None:
    primary = _ScriptedFetcher([_resp(200, b"<html>jobs</html>")])
    fallback_factory_calls = 0

    def factory() -> _ScriptedFetcher:
        nonlocal fallback_factory_calls
        fallback_factory_calls += 1
        return _ScriptedFetcher([])

    async with EscalatingFetcher(primary=primary, fallback_factory=factory) as fetcher:
        response = await fetcher.fetch("https://example.com/jobs")

    assert response.status_code == 200
    assert response.text == "<html>jobs</html>"
    assert fallback_factory_calls == 0
    assert fetcher.escalated is False


# ---------------------------------------------------------------------------
# Status-code escalation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [403, 429])
async def test_block_status_codes_trigger_escalation(status: int) -> None:
    primary = _ScriptedFetcher([_resp(status)])
    fallback = _ScriptedFetcher([_resp(200, b"<html>via-browser</html>")])

    async with EscalatingFetcher(primary=primary, fallback_factory=lambda: fallback) as fetcher:
        response = await fetcher.fetch("https://example.com/jobs")

    assert response.status_code == 200
    assert response.text == "<html>via-browser</html>"
    assert fetcher.escalated is True
    assert len(fallback.calls) == 1


# ---------------------------------------------------------------------------
# Body-signal escalation (Cloudflare interstitial)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"<html><title>Just a moment...</title></html>",
        b"<p>Checking your browser before accessing</p>",
        b"<!-- Cloudflare Ray ID: abc123 -->",
        b"Attention Required! | Cloudflare",
    ],
)
async def test_cloudflare_interstitial_triggers_escalation(body: bytes) -> None:
    primary = _ScriptedFetcher([_resp(200, body)])
    fallback = _ScriptedFetcher([_resp(200, b"<html>real-content</html>")])

    async with EscalatingFetcher(primary=primary, fallback_factory=lambda: fallback) as fetcher:
        response = await fetcher.fetch("https://example.com")

    assert response.text == "<html>real-content</html>"
    assert fetcher.escalated is True


# ---------------------------------------------------------------------------
# Sticky escalation: subsequent calls bypass the primary
# ---------------------------------------------------------------------------


async def test_subsequent_fetches_skip_primary_after_escalation() -> None:
    primary = _ScriptedFetcher([_resp(403)])
    fallback = _ScriptedFetcher([_resp(200, b"first"), _resp(200, b"second"), _resp(200, b"third")])

    async with EscalatingFetcher(primary=primary, fallback_factory=lambda: fallback) as fetcher:
        await fetcher.fetch("https://example.com/a")
        await fetcher.fetch("https://example.com/b")
        third = await fetcher.fetch("https://example.com/c")

    assert third.text == "third"
    # Primary called once (the trigger); fallback called for all three.
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 3


# ---------------------------------------------------------------------------
# Lazy fallback construction
# ---------------------------------------------------------------------------


async def test_fallback_is_not_constructed_unless_needed() -> None:
    primary = _ScriptedFetcher([_resp(200), _resp(200), _resp(200)])
    factory_calls = 0

    def factory() -> _ScriptedFetcher:
        nonlocal factory_calls
        factory_calls += 1
        return _ScriptedFetcher([])

    async with EscalatingFetcher(primary=primary, fallback_factory=factory) as fetcher:
        for _ in range(3):
            await fetcher.fetch("https://example.com/x")

    assert factory_calls == 0
    assert fetcher.escalated is False


async def test_fallback_constructed_only_once_across_many_fetches() -> None:
    primary = _ScriptedFetcher([_resp(403)])
    factory_calls = 0

    def factory() -> _ScriptedFetcher:
        nonlocal factory_calls
        factory_calls += 1
        return _ScriptedFetcher([_resp(200), _resp(200), _resp(200)])

    async with EscalatingFetcher(primary=primary, fallback_factory=factory) as fetcher:
        await fetcher.fetch("https://example.com/a")
        await fetcher.fetch("https://example.com/b")
        await fetcher.fetch("https://example.com/c")

    assert factory_calls == 1


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------


async def test_arguments_forwarded_to_primary() -> None:
    primary = _ScriptedFetcher([_resp(200)])

    async with EscalatingFetcher(
        primary=primary,
        fallback_factory=lambda: _ScriptedFetcher([]),
    ) as fetcher:
        await fetcher.fetch(
            "https://example.com/post",
            method="POST",
            headers={"X-Run-Id": "abc"},
            json={"q": "python"},
            timeout=12.5,
        )

    assert primary.calls[0] == {
        "url": "https://example.com/post",
        "method": "POST",
        "headers": {"X-Run-Id": "abc"},
        "json": {"q": "python"},
        "timeout": 12.5,
    }


async def test_arguments_forwarded_to_fallback_after_escalation() -> None:
    primary = _ScriptedFetcher([_resp(403)])
    fallback = _ScriptedFetcher([_resp(200)])

    async with EscalatingFetcher(
        primary=primary,
        fallback_factory=lambda: fallback,
    ) as fetcher:
        await fetcher.fetch(
            "https://example.com/jobs",
            headers={"Accept": "text/html"},
            timeout=20.0,
        )

    assert fallback.calls[0]["url"] == "https://example.com/jobs"
    assert fallback.calls[0]["headers"] == {"Accept": "text/html"}
    assert fallback.calls[0]["timeout"] == 20.0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_aclose_closes_primary_only_when_no_fallback_built() -> None:
    primary = _ScriptedFetcher([_resp(200)])
    fallback = _ScriptedFetcher([])

    async with EscalatingFetcher(
        primary=primary,
        fallback_factory=lambda: fallback,
    ) as fetcher:
        await fetcher.fetch("https://example.com")

    assert primary.closed is True
    assert fallback.closed is False


async def test_aclose_closes_both_after_escalation() -> None:
    primary = _ScriptedFetcher([_resp(403)])
    fallback = _ScriptedFetcher([_resp(200)])

    async with EscalatingFetcher(
        primary=primary,
        fallback_factory=lambda: fallback,
    ) as fetcher:
        await fetcher.fetch("https://example.com")

    assert primary.closed is True
    assert fallback.closed is True


# ---------------------------------------------------------------------------
# Non-block 4xx / 5xx pass through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 500, 502, 503])
async def test_non_block_errors_pass_through_without_escalation(status: int) -> None:
    primary = _ScriptedFetcher([_resp(status)])

    async with EscalatingFetcher(
        primary=primary,
        fallback_factory=lambda: _ScriptedFetcher([]),
    ) as fetcher:
        response = await fetcher.fetch("https://example.com/missing")

    assert response.status_code == status
    assert fetcher.escalated is False
