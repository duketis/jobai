"""Tests for :class:`RetryingFetcher`."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from jobai.fetcher.base import Fetcher, Response
from jobai.fetcher.retry import RetryingFetcher


class _ScriptedFetcher:
    """Plays back a script of canned responses or exceptions in order."""

    def __init__(self, script: list[Response | BaseException]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109  - matches Fetcher Protocol
        wait_for_selector: str | None = None,
    ) -> Response:
        self.calls.append(
            {
                "url": url,
                "method": method,
                "headers": dict(headers or {}),
                "json": json,
                "data": dict(data) if data is not None else None,
                "timeout": timeout,
            },
        )
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


def _resp(status: int, *, headers: dict[str, str] | None = None) -> Response:
    return Response(
        url="https://example.com",
        status_code=status,
        headers=headers or {},
        body=b"ok",
        fetched_at=datetime.now(tz=UTC),
    )


@pytest.fixture
def sleeps() -> list[float]:
    """Capture each ``await sleep(...)`` so tests can assert backoff timing."""
    return []


@pytest.fixture
def fake_sleep(sleeps: list[float]) -> Any:
    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    return _sleep


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_retrying_fetcher_satisfies_fetcher_protocol(fake_sleep: Any) -> None:
    fetcher = RetryingFetcher(_ScriptedFetcher([]), sleep=fake_sleep)
    assert isinstance(fetcher, Fetcher)


def test_invalid_max_attempts_raises() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryingFetcher(_ScriptedFetcher([]), max_attempts=0)


# ---------------------------------------------------------------------------
# Happy path: 200 short-circuits, no sleep
# ---------------------------------------------------------------------------


async def test_200_passes_through_without_retries(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([_resp(200)])
    async with RetryingFetcher(inner, sleep=fake_sleep) as fetcher:
        response = await fetcher.fetch("https://example.com")
    assert response.status_code == 200
    assert len(inner.calls) == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# Status-based retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retryable_statuses_trigger_retry(
    status: int, sleeps: list[float], fake_sleep: Any
) -> None:
    inner = _ScriptedFetcher([_resp(status), _resp(200)])
    async with RetryingFetcher(
        inner, max_attempts=3, backoff_base=0.1, jitter=0.0, sleep=fake_sleep
    ) as fetcher:
        response = await fetcher.fetch("https://example.com")
    assert response.status_code == 200
    assert len(inner.calls) == 2
    assert len(sleeps) == 1


async def test_non_retryable_4xx_passes_through(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([_resp(404)])
    async with RetryingFetcher(inner, sleep=fake_sleep) as fetcher:
        response = await fetcher.fetch("https://example.com")
    assert response.status_code == 404
    assert sleeps == []


async def test_max_attempts_returns_last_response(sleeps: list[float], fake_sleep: Any) -> None:
    """If every attempt fails with a retryable status, the final
    response is returned so callers can decide what to do."""
    inner = _ScriptedFetcher([_resp(503), _resp(503), _resp(503)])
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.1,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        response = await fetcher.fetch("https://example.com")
    assert response.status_code == 503
    assert len(inner.calls) == 3
    assert len(sleeps) == 2  # one sleep between each pair of attempts


# ---------------------------------------------------------------------------
# Network exception retry
# ---------------------------------------------------------------------------


async def test_transient_exception_is_retried(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher(
        [
            httpx.ConnectError("network down"),
            httpx.ReadTimeout("slow"),
            _resp(200),
        ]
    )
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.1,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        response = await fetcher.fetch("https://example.com")
    assert response.status_code == 200
    assert len(inner.calls) == 3
    assert len(sleeps) == 2


async def test_transient_exception_after_max_attempts_re_raises(
    sleeps: list[float], fake_sleep: Any
) -> None:
    inner = _ScriptedFetcher(
        [
            httpx.ConnectError("a"),
            httpx.ConnectError("b"),
            httpx.ConnectError("c"),
        ]
    )
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.1,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        with pytest.raises(httpx.ConnectError):
            await fetcher.fetch("https://example.com")
    assert len(inner.calls) == 3
    assert len(sleeps) == 2


async def test_non_transient_exception_is_not_retried(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([ValueError("not retryable")])
    async with RetryingFetcher(
        inner, max_attempts=3, backoff_base=0.1, sleep=fake_sleep
    ) as fetcher:
        with pytest.raises(ValueError, match="not retryable"):
            await fetcher.fetch("https://example.com")
    assert sleeps == []


# ---------------------------------------------------------------------------
# Retry-After honouring
# ---------------------------------------------------------------------------


async def test_retry_after_seconds_is_honoured(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([_resp(429, headers={"retry-after": "7"}), _resp(200)])
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.1,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        await fetcher.fetch("https://example.com")
    assert sleeps == [7.0]


async def test_retry_after_http_date_is_honoured(sleeps: list[float], fake_sleep: Any) -> None:
    future = datetime.now(tz=UTC) + timedelta(seconds=12)
    http_date = format_datetime(future, usegmt=True)
    inner = _ScriptedFetcher([_resp(429, headers={"retry-after": http_date}), _resp(200)])
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.1,
        jitter=0.0,
        backoff_max=60.0,
        sleep=fake_sleep,
    ) as fetcher:
        await fetcher.fetch("https://example.com")
    # Allow a small drift since the call computes "now" again.
    assert 9.0 <= sleeps[0] <= 13.0


async def test_retry_after_capped_by_backoff_max(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([_resp(429, headers={"retry-after": "9999"}), _resp(200)])
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.1,
        backoff_max=10.0,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        await fetcher.fetch("https://example.com")
    assert sleeps == [10.0]


async def test_invalid_retry_after_falls_back_to_backoff(
    sleeps: list[float], fake_sleep: Any
) -> None:
    inner = _ScriptedFetcher([_resp(429, headers={"retry-after": "not-a-date"}), _resp(200)])
    async with RetryingFetcher(
        inner,
        max_attempts=3,
        backoff_base=0.5,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        await fetcher.fetch("https://example.com")
    assert sleeps == [0.5]  # base * 2^0 with no jitter


# ---------------------------------------------------------------------------
# Exponential backoff
# ---------------------------------------------------------------------------


async def test_exponential_backoff_doubles_per_attempt(
    sleeps: list[float], fake_sleep: Any
) -> None:
    inner = _ScriptedFetcher([_resp(503), _resp(503), _resp(503), _resp(200)])
    async with RetryingFetcher(
        inner,
        max_attempts=4,
        backoff_base=1.0,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        await fetcher.fetch("https://example.com")
    # attempts 1->2: 1s, 2->3: 2s, 3->4: 4s
    assert sleeps == [1.0, 2.0, 4.0]


async def test_backoff_capped_at_backoff_max(sleeps: list[float], fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([_resp(503), _resp(503), _resp(503), _resp(200)])
    async with RetryingFetcher(
        inner,
        max_attempts=4,
        backoff_base=10.0,
        backoff_max=15.0,
        jitter=0.0,
        sleep=fake_sleep,
    ) as fetcher:
        await fetcher.fetch("https://example.com")
    # 10, 20 -> capped to 15, 40 -> capped to 15
    assert sleeps == [10.0, 15.0, 15.0]


# ---------------------------------------------------------------------------
# Argument forwarding + lifecycle
# ---------------------------------------------------------------------------


async def test_arguments_forwarded_to_inner(fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([_resp(200)])
    async with RetryingFetcher(inner, sleep=fake_sleep) as fetcher:
        await fetcher.fetch(
            "https://example.com/x",
            method="POST",
            headers={"X-Run-Id": "abc"},
            json={"q": "python"},
            timeout=5.0,
        )
    assert inner.calls[0] == {
        "url": "https://example.com/x",
        "method": "POST",
        "headers": {"X-Run-Id": "abc"},
        "json": {"q": "python"},
        "data": None,
        "timeout": 5.0,
    }


async def test_aclose_closes_inner(fake_sleep: Any) -> None:
    inner = _ScriptedFetcher([])
    async with RetryingFetcher(inner, sleep=fake_sleep):
        pass
    assert inner.closed is True
