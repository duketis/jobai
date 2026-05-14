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

from jobai import __version__

#: Single source of truth for the User-Agent string. Built from
#: ``jobai.__version__`` so a version bump in one place propagates
#: everywhere — Settings.user_agent, the browser-tier UA, and the
#: stealth-tier UA all interpolate this constant.
DEFAULT_USER_AGENT = f"jobai/{__version__} (+https://github.com/duketis/jobai)"


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
        default=DEFAULT_USER_AGENT,
        description="Default HTTP User-Agent header for the tier-1 fetcher.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description=(
            "Anthropic API key for the AI agent. If unset, the SDK falls back "
            "to the ANTHROPIC_API_KEY environment variable."
        ),
    )
    anthropic_model: str = Field(
        default="claude-opus-4-7",
        description="Anthropic model id to use for the agent.",
    )
    agent_backend: str = Field(
        default="api",
        description=(
            "Which auth path the agent uses. ``api`` (default) drives the "
            "Anthropic SDK with an API key — pay-per-token billing. "
            "``subscription`` drives the Claude Agent SDK which runs the "
            "logged-in ``claude`` CLI in a subprocess, so calls bill against "
            "your Claude Pro/Max subscription quota instead. Subscription "
            "mode requires the ``claude`` CLI installed and authenticated "
            "(typically via mounting your ``~/.claude/`` into the container)."
        ),
    )
    resumeai_url: str = Field(
        default="http://resumeai:8765",
        description=(
            "Base URL for the resumeai sibling service. Default targets the "
            "service name on the shared ``ai-tailor-network`` docker network. "
            "Override to ``http://localhost:8765`` for host-mode development."
        ),
    )
    coverletterai_url: str = Field(
        default="http://coverletterai:8766",
        description=(
            "Base URL for the coverletterai sibling service. Default targets "
            "the service name on the shared ``ai-tailor-network`` docker "
            "network. Override to ``http://localhost:8766`` for host-mode dev."
        ),
    )
    tailor_max_concurrent: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Cap on concurrent tailor chains in flight at once. The siblings "
            "serialise their internal renderers, so a high cap mostly stacks "
            "polls; 3-5 is the sweet spot for a batch."
        ),
    )
    tailor_output_dir: str = Field(
        default="/data/tailored",
        description=(
            "Directory where every successful tailor run drops a per-job "
            "folder containing the resume PDF, cover-letter PDF, JD markdown, "
            "QA verdict JSON, application checklist, and metadata. The same "
            "folder layout is what the sibling ``interviewai`` reads from to "
            "prepare interview answers against the artefacts you actually "
            "submitted. Default is ``/data/tailored`` inside the Docker "
            "container; on the host this maps to the ``jobai-data`` named "
            "volume. Set ``JOBAI_TAILOR_OUTPUT_DIR=/Users/you/jobai-tailored`` "
            "for host-mode dev so the folders land somewhere your file "
            "manager can navigate to directly."
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    Cached so the first call does the env / .env parsing and subsequent
    calls are O(1) attribute lookups. Tests that mutate the environment
    must call ``get_settings.cache_clear()`` between cases.
    """
    return Settings()
