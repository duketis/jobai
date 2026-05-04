"""Runtime configuration loaded from the environment and optional ``.env`` file.

A single :class:`Settings` instance holds every value the runtime needs:
where the database lives, how verbose logs are, where the rotating log
file goes, default scrape cadences, the HTTP API bind address, the
default User-Agent header for tier-1 fetches.

Settings are loaded from environment variables prefixed ``JOBAI_`` with a
``.env`` file in the working directory as fallback. Validation runs at
construction time, so an invalid configuration fails fast with a clear
:class:`pydantic.ValidationError` rather than a confusing runtime error
deeper in the stack.

Use :func:`get_settings` to obtain the process-wide instance; it is
cached so repeated calls are free.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level runtime configuration for jobai."""

    model_config = SettingsConfigDict(
        env_prefix="JOBAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    db_path: Path = Field(
        default=Path("jobai.db"),
        description="Filesystem path to the SQLite database file.",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG / INFO / WARNING / ERROR / CRITICAL).",
    )
    log_dir: Path | None = Field(
        default=Path("logs"),
        description=(
            "Directory for the rotating ``jobai.log`` file. "
            "Set to null/empty in the environment to log only to stdout."
        ),
    )
    log_max_bytes: int = Field(
        default=10_000_000,
        ge=1024,
        description="Maximum bytes per rotated log file.",
    )
    log_backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of rotated log files to keep.",
    )
    api_host: str = Field(
        default="127.0.0.1",
        description="Bind host for the FastAPI HTTP API.",
    )
    api_port: int = Field(
        default=8421,
        ge=1,
        le=65535,
        description="Bind port for the FastAPI HTTP API.",
    )
    default_cadence_seconds: int = Field(
        default=1800,
        ge=10,
        description="Default cadence for sources that do not declare one.",
    )
    user_agent: str = Field(
        default="jobai/0.0.1 (+https://github.com/duketis/jobai)",
        description="Default HTTP User-Agent header for the tier-1 fetcher.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    Cached so the first call does the env / .env parsing and subsequent
    calls are O(1) attribute lookups. Tests that mutate the environment
    must call ``get_settings.cache_clear()`` between cases.
    """
    return Settings()
