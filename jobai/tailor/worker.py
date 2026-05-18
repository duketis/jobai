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
        # Maps a caller-supplied key (the tailor_run id) to its live
        # task so a cancel request can actually interrupt the work
        # instead of just flipping the DB row while the chain keeps
        # burning sibling calls in the background.
        self._by_key: dict[int, asyncio.Task[None]] = {}

    @property
    def active_tasks(self) -> int:
        """Number of submitted-but-not-finished tasks. Useful for tests."""
        return len(self._tasks)

    def submit(self, factory: JobFactory, *, key: int | None = None) -> asyncio.Task[None]:
        """Schedule ``factory()`` under the semaphore.

        Returns the wrapper task so callers (mostly tests) can await it
        directly. Production callers fire-and-forget. ``key`` (the
        tailor_run id) registers the task for :meth:`cancel`; it's
        cleared automatically when the task finishes.
        """
        task = asyncio.create_task(self._guarded(factory))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if key is not None:
            self._by_key[key] = task

            def _drop_key(_t: asyncio.Task[None], k: int = key) -> None:
                self._by_key.pop(k, None)

            task.add_done_callback(_drop_key)
        return task

    def cancel(self, key: int) -> bool:
        """Request cancellation of the task submitted under ``key``.

        Returns ``True`` if a live task was found and cancellation was
        requested, ``False`` if no task is tracked for ``key`` (already
        finished, orphaned by a restart, or never submitted with a
        key). Best-effort: the caller still authoritatively marks the
        DB row failed so a keyless / dead task can't leave a zombie.
        """
        task = self._by_key.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        return True

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
