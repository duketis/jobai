"""Tool definitions and executor for the AI agent.

Tools wrap the data-layer operations (search, detail, state, source
health, aggregate health) as Anthropic-tool-compatible dicts. The
executor dispatches a tool call (name + input) to the matching helper
and returns a JSON-serialisable result.

Why a custom executor instead of the SDK's beta ``tool_runner``: the
streaming chat loop needs to surface intermediate events (text deltas,
tool-call announcements, tool results) as Server-Sent Events. The
tool_runner abstracts the loop and returns whole messages, which loses
that visibility. Manual dispatch + streaming is the right shape here.

Programmer errors raise; "expected" failures (job not found) come back
as ``{"error": "..."}`` results so the model can read them and adapt
without a full tool_use exception path. The runner converts genuine
exceptions into ``is_error: true`` tool_result blocks so the model still
sees them and can recover.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from jobai.api.repository import get_job_detail, search_jobs

# ---------------------------------------------------------------------------
# Tool definitions (the JSON schemas Anthropic's API accepts in `tools=`)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_jobs",
        "description": (
            "Search and filter the canonical jobs table. Use whenever the user "
            "asks about finding, listing, filtering, or counting jobs. The free-"
            "text 'q' parameter goes through SQLite FTS5 across title, company, "
            "description, and location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Free-text search query.",
                },
                "location": {
                    "type": "string",
                    "description": "Substring match on city, country, or raw location.",
                },
                "remote": {
                    "type": "string",
                    "enum": ["remote", "hybrid", "onsite"],
                    "description": "Filter by work mode.",
                },
                "company": {
                    "type": "string",
                    "description": "Substring match on normalised company name.",
                },
                "source_kind": {
                    "type": "string",
                    "description": (
                        "Restrict to jobs surfaced by a specific source kind "
                        "(greenhouse, lever, ashby, workable, smartrecruiters)."
                    ),
                },
                "posted_since": {
                    "type": "string",
                    "description": "ISO 8601 date; only jobs posted on or after this date.",
                },
                "exclude_title": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Title keywords to exclude (case-insensitive substring "
                        "match). Use this for 'no seniors', 'no managers', "
                        "'no leads' style filters — not the free-text 'q' "
                        "(which is for ranking, not exclusion). Example: "
                        "['senior', 'staff', 'principal', 'lead', 'manager']."
                    ),
                },
                "min_salary": {
                    "type": "integer",
                    "description": (
                        "Minimum salary (the upper end of the band must clear "
                        "this). Use when the user specifies a salary floor."
                    ),
                },
                "has_salary": {
                    "type": "boolean",
                    "description": (
                        "When true, restrict to jobs that publish a salary. "
                        "Use when the user asks for 'roles with salary listed'."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "relevance",
                        "newest",
                        "oldest",
                        "posted_newest",
                        "posted_oldest",
                        "salary_high",
                        "salary_low",
                    ],
                    "description": (
                        "Sort order. Default: 'relevance' when q is set, "
                        "otherwise 'newest'. Pick 'posted_newest' when the "
                        "user wants 'most recently posted', 'salary_high' "
                        "for 'highest paid first', etc."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (1-100, default 20).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default 0).",
                },
            },
        },
    },
    {
        "name": "get_job_detail",
        "description": (
            "Fetch one canonical job by id, including the full description "
            "and the list of sources that surfaced it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "mark_job_state",
        "description": (
            "Set the user's triage state for a job. Use when the user says they "
            "want to save / apply to / dismiss / reject a job, or wants to "
            "attach notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "state": {
                    "type": "string",
                    "enum": ["new", "saved", "applied", "dismissed", "rejected"],
                },
                "notes": {"type": "string"},
            },
            "required": ["job_id", "state"],
        },
    },
    {
        "name": "list_sources",
        "description": (
            "Return every configured job source with its current runtime health "
            "(last success, last error, consecutive failures). Use to answer "
            "'where do my jobs come from?' or 'which sources are failing?'"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_health",
        "description": (
            "Aggregate snapshot of the data layer: total jobs, jobs added in "
            "the last 24h, source counts, source failures, and the timestamp "
            "of the last successful scrape run. Use to answer 'how is the "
            "system doing?' or 'how fresh is the data?'"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

_VALID_USER_STATES = frozenset({"new", "saved", "applied", "dismissed", "rejected"})


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class UnknownToolError(KeyError):
    """Raised when the agent calls a tool we have not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class ToolExecutor:
    """Dispatches Anthropic tool calls to local Python handlers.

    Construct once per request with the SQLite connection; call
    :meth:`execute` for each ``tool_use`` block the agent emits.
    Results are JSON-serialisable dicts ready to drop into the next
    user-turn ``tool_result`` block via :func:`serialise_result`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._handlers: dict[str, Callable[[Mapping[str, Any]], Any]] = {
            "search_jobs": self._search_jobs,
            "get_job_detail": self._get_job_detail,
            "mark_job_state": self._mark_job_state,
            "list_sources": self._list_sources,
            "get_health": self._get_health,
        }

    def execute(self, name: str, tool_input: Mapping[str, Any]) -> Any:
        """Run the handler for ``name`` and return its result."""
        handler = self._handlers.get(name)
        if handler is None:
            raise UnknownToolError(name)
        return handler(tool_input)

    # ---- handlers ----

    def _search_jobs(self, args: Mapping[str, Any]) -> dict[str, Any]:
        result = search_jobs(
            self._conn,
            q=_opt_str(args.get("q")),
            location=_opt_str(args.get("location")),
            remote_type=_opt_str(args.get("remote")),
            employment_type=_opt_str(args.get("employment_type")),
            posted_since=_opt_str(args.get("posted_since")),
            company=_opt_str(args.get("company")),
            source_kind=_opt_str(args.get("source_kind")),
            exclude_title=_opt_str_list(args.get("exclude_title")),
            min_salary=_opt_int(args.get("min_salary")),
            has_salary=bool(args.get("has_salary", False)),
            sort=_opt_str(args.get("sort")),
            limit=int(args.get("limit", 20)),
            offset=int(args.get("offset", 0)),
        )
        return result.model_dump()

    def _get_job_detail(self, args: Mapping[str, Any]) -> dict[str, Any]:
        try:
            job_id = int(args["job_id"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = "job_id (integer) is required"
            raise ValueError(msg) from exc
        detail = get_job_detail(self._conn, job_id)
        if detail is None:
            return {"error": f"job {job_id} not found"}
        return detail.model_dump()

    def _mark_job_state(self, args: Mapping[str, Any]) -> dict[str, Any]:
        try:
            job_id = int(args["job_id"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = "job_id (integer) is required"
            raise ValueError(msg) from exc

        state = str(args.get("state", "")).strip()
        if state not in _VALID_USER_STATES:
            msg = f"state must be one of {sorted(_VALID_USER_STATES)}"
            raise ValueError(msg)

        notes = args.get("notes")
        if notes is not None and not isinstance(notes, str):
            msg = "notes must be a string when provided"
            raise TypeError(msg)

        if self._conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
            return {"error": f"job {job_id} not found"}

        now = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            "INSERT INTO jobs_user_state (job_id, state, notes, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET "
            "  state = excluded.state, "
            "  notes = excluded.notes, "
            "  updated_at = excluded.updated_at",
            (job_id, state, notes, now),
        )
        self._conn.commit()

        return {
            "job_id": job_id,
            "state": state,
            "notes": notes,
            "updated_at": now,
        }

    def _list_sources(self, _args: Mapping[str, Any]) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT s.id, s.kind, s.account, s.display_name, s.default_tier, "
            "       s.enabled, s.cadence_seconds, "
            "       rs.current_tier, rs.last_success_at, rs.last_error_at, "
            "       rs.last_error_class, rs.consecutive_failures, rs.cooldown_until "
            "FROM sources s LEFT JOIN source_runtime_state rs ON rs.source_id = s.id "
            "ORDER BY s.kind, s.account"
        ).fetchall()
        items: list[dict[str, Any]] = []
        for r in rows:
            kind = str(r[1])
            account = str(r[2])
            items.append(
                {
                    "id": int(r[0]),
                    "name": f"{kind}:{account}" if account else kind,
                    "kind": kind,
                    "display_name": str(r[3]),
                    "default_tier": int(r[4]),
                    "enabled": bool(r[5]),
                    "cadence_seconds": int(r[6]),
                    "current_tier": _opt_int(r[7]),
                    "last_success_at": _opt_str(r[8]),
                    "last_error_at": _opt_str(r[9]),
                    "last_error_class": _opt_str(r[10]),
                    "consecutive_failures": int(r[11]) if r[11] is not None else 0,
                    "cooldown_until": _opt_str(r[12]),
                }
            )
        return {"items": items}

    def _get_health(self, _args: Mapping[str, Any]) -> dict[str, Any]:
        last_scrape_row = self._conn.execute(
            "SELECT MAX(finished_at) FROM scrape_runs WHERE status = 'success'"
        ).fetchone()
        last_scrape_at = (
            str(last_scrape_row[0])
            if last_scrape_row is not None and last_scrape_row[0] is not None
            else None
        )
        return {
            "jobs_total": _scalar(self._conn, "SELECT COUNT(*) FROM jobs"),
            "jobs_added_24h": _scalar(
                self._conn,
                "SELECT COUNT(*) FROM jobs WHERE first_seen_at >= datetime('now', '-1 day')",
            ),
            "sources_total": _scalar(self._conn, "SELECT COUNT(*) FROM sources"),
            "sources_enabled": _scalar(
                self._conn, "SELECT COUNT(*) FROM sources WHERE enabled = 1"
            ),
            "sources_failing": _scalar(
                self._conn,
                "SELECT COUNT(*) FROM source_runtime_state "
                "WHERE last_error_at IS NOT NULL "
                "AND last_error_at > COALESCE(last_success_at, '1970-01-01') "
                "AND last_error_at >= datetime('now', '-1 day')",
            ),
            "last_scrape_at": last_scrape_at,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def serialise_result(value: Any) -> str:
    """Convert a tool result to the JSON string Anthropic expects in
    ``tool_result.content``."""
    return json.dumps(value, default=str)


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row is not None else 0


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        return str(v)
    return v if v else None


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)


def _opt_str_list(v: Any) -> list[str] | None:
    """Coerce the agent's ``exclude_title`` argument into a list[str].

    Accepts a JSON array (the schema's preferred shape) or — defensively
    — a comma-separated string when the model abbreviates. Empty
    tokens are dropped so a stray comma can't accidentally exclude
    every row.
    """
    if v is None:
        return None
    if isinstance(v, str):
        tokens = [t.strip() for t in v.split(",")]
    elif isinstance(v, list):
        tokens = [str(t).strip() for t in v]
    else:
        return None
    cleaned = [t for t in tokens if t]
    return cleaned or None
