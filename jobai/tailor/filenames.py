"""Compose the descriptive per-run PDF filenames.

Used both by the PDF-stream routes (Content-Disposition) and by the
orchestrator (which caches the result on the tailor_runs row at
terminal success so the frontend can render it as a link label +
``<a download=...>`` attribute without a per-row sibling fetch).

Filename source-of-truth:

* Applicant name: pulled from the resumeai sibling's tailored.name
  payload -- resumeai owns the candidate identity.
* Title + company: from the ``jobs`` row when the run is catalogue-
  matched; from the sibling's parsed ``requirements`` block when the
  chain came in via ``POST /api/tailor/url`` against an off-catalogue
  JD.
* Suffix: ``Resume`` or ``CoverLetter`` (matches the artefact kind).

Every field has a fallback so a partially-populated row still
produces a usable filename, and a sibling 5xx during the lookup
degrades the filename without breaking the actual PDF stream.
"""

from __future__ import annotations

import re
import sqlite3
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from jobai.tailor.client import ResumeaiClient

#: Characters illegal on Windows / macOS filesystems. We replace each
#: occurrence with a space so the sanitiser's whitespace-collapse step
#: leaves a tidy result.
_FILENAME_BAD_CHARS: Final[re.Pattern[str]] = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Any run of whitespace (incl. NBSP-ish chars that survive ASCII-fold)
#: collapses to a single space.
_FILENAME_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def sanitize_filename_part(text: str | None, *, fallback: str) -> str:
    """Strip path-illegal characters, collapse whitespace, ASCII-fold.

    Returns ``fallback`` when the input is empty / None or sanitises
    down to nothing -- keeps the final filename non-empty even for
    bare-URL runs the sibling hasn't tagged with a title yet.
    """
    if not text:
        return fallback
    cleaned = _FILENAME_BAD_CHARS.sub(" ", text)
    cleaned = _FILENAME_WHITESPACE.sub(" ", cleaned).strip(" .")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or fallback


def _compose(name: str, title: str, company: str, kind: str) -> str:
    """Glue sanitised parts into ``<Name>-<Title>-<Company>-<Kind>.pdf``."""
    suffix = "Resume" if kind == "resume" else "CoverLetter"
    parts = [
        p.replace(" ", "_")
        for p in (
            sanitize_filename_part(name, fallback="Applicant"),
            sanitize_filename_part(title, fallback="Job"),
            sanitize_filename_part(company, fallback="Company"),
            suffix,
        )
    ]
    return "-".join(parts) + ".pdf"


async def build_pdf_filenames(
    *,
    conn: sqlite3.Connection,
    resume_client: ResumeaiClient,
    tailor_run_id: int,
) -> tuple[str, str]:
    """Compose both ``(resume, letter)`` filenames in one pass.

    The orchestrator caches both at terminal success and we want a
    single resumeai sibling fetch -- batching here avoids the
    duplicate ``get_run`` call we'd get from two ``build_pdf_filename``
    invocations.
    """
    name, title, company = await _fetch_filename_parts(
        conn=conn,
        resume_client=resume_client,
        tailor_run_id=tailor_run_id,
    )
    return _compose(name, title, company, "resume"), _compose(name, title, company, "letter")


async def build_pdf_filename(
    *,
    conn: sqlite3.Connection,
    resume_client: ResumeaiClient,
    tailor_run_id: int,
    kind: str,
) -> str:
    """Compose the per-run filename for ``kind`` (``"resume"`` /
    ``"letter"``).

    Reads identity off the resumeai sibling and title/company off
    either the ``jobs`` row (catalogue path) or the sibling's parsed
    requirements (bare-URL path). Every fetch is defensive: a sibling
    outage here degrades only the filename, not the PDF stream.
    """
    name, title, company = await _fetch_filename_parts(
        conn=conn,
        resume_client=resume_client,
        tailor_run_id=tailor_run_id,
    )
    return _compose(name, title, company, kind)


async def _fetch_filename_parts(
    *,
    conn: sqlite3.Connection,
    resume_client: ResumeaiClient,
    tailor_run_id: int,
) -> tuple[str, str, str]:
    """Return ``(name, title, company)`` for the filename composer.

    Performs at most ONE resumeai ``get_run`` call so the caller can
    compose both resume and letter filenames without doubling the
    sibling fetch.
    """
    # Lazy-import to avoid a circular dependency with the orchestrator
    # module (which imports from this module). The repository import is
    # cheap; this just keeps the static graph clean.
    from jobai.tailor.repository import get_tailor_run  # noqa: PLC0415

    record = get_tailor_run(conn, tailor_run_id)
    if record is None:  # pragma: no cover - the route guard runs first
        return "Applicant", "Job", "Company"

    title = "Job"
    company = "Company"
    if record.job_id is not None:
        row = conn.execute(
            "SELECT title, company FROM jobs WHERE id = ?",
            (record.job_id,),
        ).fetchone()
        if row is not None:
            title = row[0] or title
            company = row[1] or company

    name = "Applicant"
    if record.resume_run_id:
        try:
            resume_rec = await resume_client.get_run(record.resume_run_id)
        except Exception:  # noqa: BLE001 - filename is best-effort
            resume_rec = {}
        tailored = resume_rec.get("tailored") if isinstance(resume_rec, dict) else None
        if isinstance(tailored, dict):
            name_val = tailored.get("name")
            if isinstance(name_val, str):
                name = name_val
        if record.job_id is None:
            reqs = resume_rec.get("requirements") if isinstance(resume_rec, dict) else None
            if isinstance(reqs, dict):
                t = reqs.get("title")
                c = reqs.get("company")
                if isinstance(t, str):
                    title = t
                if isinstance(c, str):
                    company = c

    return name, title, company
