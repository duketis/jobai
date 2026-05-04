"""Database layer.

Owns the SQLite schema, migration runner, and connection helper. Every
caller in :mod:`jobai` that touches the database goes through this module
rather than constructing :class:`sqlite3.Connection` instances directly,
so WAL mode, foreign keys, and the row factory are guaranteed to be set
consistently.
"""

from __future__ import annotations
