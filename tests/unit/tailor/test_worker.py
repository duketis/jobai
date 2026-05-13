"""Pool coverage for jobai.tailor.worker."""

from __future__ import annotations

import asyncio

import pytest

from jobai.tailor.worker import TailorPool


async def test_active_tasks_reports_outstanding_count() -> None:
    """The property surfaces in-flight task count for the /health-ish UI."""
    pool = TailorPool(max_concurrent=1)
    block = asyncio.Event()

    async def _holds() -> None:
        await block.wait()

    pool.submit(_holds)
    assert pool.active_tasks == 1
    block.set()
    await pool.drain()
    assert pool.active_tasks == 0


def test_constructor_rejects_zero_or_negative_concurrency() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        TailorPool(max_concurrent=0)


async def test_submit_runs_factory_under_semaphore() -> None:
    """Submitted factory eventually runs to completion; pool returns task handle."""
    ran = asyncio.Event()

    async def _factory() -> None:
        ran.set()

    pool = TailorPool(max_concurrent=2)
    task = pool.submit(_factory)
    await task
    assert ran.is_set()


async def test_semaphore_caps_concurrent_factories() -> None:
    """With cap=1 the second task must wait for the first to release."""
    pool = TailorPool(max_concurrent=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    async def _first() -> None:
        first_started.set()
        await release_first.wait()

    async def _second() -> None:
        second_started.set()

    t1 = pool.submit(_first)
    t2 = pool.submit(_second)
    await first_started.wait()
    # cap=1 means second should NOT have entered the body yet.
    await asyncio.sleep(0)
    assert not second_started.is_set()
    release_first.set()
    await t1
    await t2
    assert second_started.is_set()


async def test_drain_returns_immediately_when_no_tasks() -> None:
    pool = TailorPool(max_concurrent=2)
    await pool.drain()  # no exception; just returns


async def test_drain_logs_and_swallows_task_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A factory that raises is logged but doesn't crash drain()."""

    async def _boom() -> None:
        msg = "kaboom"
        raise RuntimeError(msg)

    pool = TailorPool(max_concurrent=1)
    pool.submit(_boom)
    with caplog.at_level("WARNING"):
        await pool.drain()
    # The structured ``error`` field carries the exception string.
    assert any(getattr(r, "error", None) == "kaboom" for r in caplog.records)
