"""FastAPI dependencies — database connection per request.

Every route that touches SQLite acquires its connection through
``Depends(get_conn)``. This keeps connection lifecycle in one place
(open per request, close at exit) and makes routes trivially mockable
in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from jobai.config import get_settings
from jobai.db.connection import connect


def get_db_path() -> Path:
    """Return the configured SQLite path. Indirected so tests can override."""
    return get_settings().db_path


def get_conn(
    db_path: Annotated[Path, Depends(get_db_path)],
) -> Iterator[sqlite3.Connection]:
    """Yield a configured (WAL, foreign keys, row factory) connection."""
    with connect(db_path) as conn:
        yield conn


#: Convenience type alias for routes: ``conn: ConnDep``.
ConnDep = Annotated[sqlite3.Connection, Depends(get_conn)]
