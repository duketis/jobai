"""Concurrency-capped pool for submitting tailor chains.

Hidden behind a tiny class so the routes never see ``asyncio.Semaphore``
directly and tests can swap the pool for a recorder.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

_log = logging.getLogger(__name__)

# A submitted job is a 0-arg coroutine factory; the pool calls it under the
# semaphore. Using a factory (not a coroutine instance) means the pool can
# defer creation until a slot is free, which matters when many chains are
# queued at once.
JobFactory = Callable[[], Awaitable[None]]


class TailorPool:
    """Schedule chain coroutines under a configurable concurrency cap.

    ``submit`` is non-blocking: it spawns a wrapper task that waits on
    the semaphore before invoking the factory. ``drain`` awaits every
    outstanding task, intended for shutdown.
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        if max_concurrent < 1:
            msg = f"max_concurrent must be >= 1, got {max_concurrent}"
            raise ValueError(msg)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active_tasks(self) -> int:
        """Number of submitted-but-not-finished tasks. Useful for tests."""
        return len(self._tasks)

    def submit(self, factory: JobFactory) -> asyncio.Task[None]:
        """Schedule ``factory()`` under the semaphore.

        Returns the wrapper task so callers (mostly tests) can await it
        directly. Production callers fire-and-forget.
        """
        task = asyncio.create_task(self._guarded(factory))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Await every outstanding task. Logs but never re-raises failures."""
        if not self._tasks:
            return
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        for outcome in results:
            if isinstance(outcome, BaseException):
                _log.warning(
                    "tailor_pool_task_failed",
                    extra={"error": str(outcome)},
                )

    async def _guarded(self, factory: JobFactory) -> None:
        async with self._semaphore:
            await factory()
