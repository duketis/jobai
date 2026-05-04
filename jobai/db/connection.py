"""SQLite connection helper.

Centralises the PRAGMAs and connection settings every caller needs:

* WAL journal mode for concurrent reads alongside a writer.
* Foreign keys ON (off by default in SQLite).
* :class:`sqlite3.Row` row factory for dict-style column access.
* A reasonable busy timeout so transient lock contention does not raise.

Callers acquire a connection through :func:`connect` (a context manager
that closes on exit) rather than calling :func:`sqlite3.connect`
directly. This guarantees the PRAGMAs are always applied; constructing
:class:`sqlite3.Connection` instances elsewhere in the codebase is a
bug.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_BUSY_TIMEOUT_SECONDS = 10.0


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a configured SQLite connection, closed on context exit.

    Args:
        db_path: filesystem path to the database file. The file is
            created if it does not exist (standard sqlite3 behaviour).
            ``:memory:`` is not supported here because WAL mode requires
            a file-backed database; tests that need an in-memory DB
            should construct :class:`sqlite3.Connection` directly.

    Yields:
        A :class:`sqlite3.Connection` with WAL, foreign keys, and the
        :class:`sqlite3.Row` factory configured.
    """
    connection = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        yield connection
    finally:
        connection.close()
