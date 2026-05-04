"""Structured JSON logging configuration.

Configures :mod:`structlog` so every log call across jobai emits a single
ISO-8601-timestamped JSON line. By default logs go to stderr; pass a
``log_dir`` to also append to a rotating ``jobai.log`` file.

Call :func:`configure_logging` exactly once at process startup. After
that, any module obtains a logger with::

    log = jobai.observability.logging.get_logger(__name__)

and emits structured events::

    log.info("scrape_complete", source="greenhouse", items_new=12)

The output for the example above is one line of JSON like::

    {"event": "scrape_complete", "source": "greenhouse", "items_new": 12,
     "timestamp": "2026-05-04T08:42:11.123456Z", "log_level": "info"}
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, cast

import structlog


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    log_max_bytes: int = 10_000_000,
    log_backup_count: int = 5,
) -> None:
    """Configure structlog and stdlib logging to emit JSON.

    Args:
        level: Minimum log level. One of ``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, ``CRITICAL`` (case-insensitive).
        log_dir: Optional directory for a rotating ``jobai.log`` file.
            ``None`` (default) means stdout/stderr only. The directory is
            created if it does not exist.
        log_max_bytes: Maximum bytes per rotated log file.
        log_backup_count: Number of rotated files to retain.
    """
    numeric_level = _resolve_level(level)

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_dir / "jobai.log",
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
        )

    plain_formatter = logging.Formatter("%(message)s")
    for handler in handlers:
        handler.setFormatter(plain_formatter)

    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
        existing.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(numeric_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for ``name``.

    Convenience wrapper so callers do not need to import :mod:`structlog`
    directly. The cast satisfies mypy because :func:`structlog.get_logger`
    is typed loosely; in practice every logger in this codebase is a
    :class:`structlog.stdlib.BoundLogger`.
    """
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))


def _resolve_level(level: str) -> int:
    """Translate a level name to its numeric value, defaulting to INFO."""
    mapping: dict[str, int] = logging.getLevelNamesMapping()
    return mapping.get(level.upper(), logging.INFO)


# Re-export for convenience: callers can do
#     from jobai.observability.logging import bind_contextvars
# instead of also importing structlog directly.
bind_contextvars: Any = structlog.contextvars.bind_contextvars
clear_contextvars: Any = structlog.contextvars.clear_contextvars
