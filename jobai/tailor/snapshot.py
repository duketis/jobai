"""On-disk snapshot of every successful tailor run.

Why this module exists:

* Without it, the only way to grab the tailored PDFs is through the
  HTTP routes (``GET /api/tailor/runs/{id}/resume.pdf`` etc.). That's
  fine for browser preview but painful when the user wants to apply
  to twenty jobs by drag-and-dropping PDFs into application forms.
* The sibling project ``interviewai`` needs structured access to
  every applied job's artefacts (JD, resume, letter, QA verdict) when
  the user later sits down to prep for interviews. Without an
  on-disk layout the only integration path would be the HTTP API,
  which couples interviewai's runtime to jobai's container being up.

The folder layout is the contract. One folder per tailor run, named
``<Company>-<Title>-<JobID>``. Inside:

* ``<Name>-<Title>-<Company>-Resume.pdf`` — the tailored resume.
* ``<Name>-<Title>-<Company>-CoverLetter.pdf`` — the tailored letter.
* ``jd.md`` — the job description text used for tailoring.
* ``qa.json`` — full QA assessment (scores, must-fix, nice-to-fix).
* ``metadata.json`` — apply URL, sibling run IDs, timestamps.
* ``CHECKLIST.md`` — the per-job application checklist the user opens
  and ticks off as they fill the form.

A master ``INDEX.md`` at the output root is regenerated on every
snapshot so the user can ``cat`` it (or open it in any editor) and
see every job + its applied / not-applied state at a glance.

Failure to snapshot must not fail the chain. A disk-full error or a
sibling 5xx while we're fetching the PDFs leaves the row at
SUCCEEDED with a logged warning -- the user can still grab the PDFs
through the HTTP route, they just don't get the folder convenience.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobai.tailor.client import CoverletteraiClient, ResumeaiClient
    from jobai.tailor.models import TailorRunRecord

from jobai.tailor.filenames import sanitize_filename_part
from jobai.tailor.repository import get_tailor_run, list_tailor_runs

_log = logging.getLogger(__name__)

#: Characters illegal as path components on Windows / macOS.
_PATH_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: Whitespace runs that should collapse before becoming filesystem names.
_PATH_WHITESPACE = re.compile(r"\s+")


def _folder_name(record: TailorRunRecord, *, title: str, company: str) -> str:
    """Compose the per-job folder name as ``<Company>-<Title>-<JobID>``.

    Sanitised the same way ``filenames.sanitize_filename_part`` cleans
    the PDF parts so the folder + the PDFs inside it use a consistent
    convention. Job id (or ``url<run_id>`` for bare-URL runs) is
    appended so two jobs with identical company/title still get
    distinct folders.
    """
    company_s = sanitize_filename_part(company, fallback="Company").replace(" ", "_")
    title_s = sanitize_filename_part(title, fallback="Job").replace(" ", "_")
    suffix = f"job{record.job_id}" if record.job_id is not None else f"run{record.id}"
    raw = f"{company_s}-{title_s}-{suffix}"
    # Belt-and-braces sweep over the joined string for any path-illegal
    # chars that survived per-part sanitising (the hyphen separators
    # we just added shouldn't, but a future edit could regress).
    cleaned = _PATH_BAD_CHARS.sub("_", raw)
    return _PATH_WHITESPACE.sub("_", cleaned).strip("_")


async def write_snapshot(
    *,
    output_dir: Path,
    db_path: Path,
    tailor_run_id: int,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    apply_profile: dict[str, str] | None = None,
) -> Path | None:
    """Write the per-job snapshot folder for a completed tailor run.

    Returns the path of the created folder, or ``None`` if the
    snapshot couldn't be produced (run not found, no PDFs yet, disk
    error). All exceptions are caught and logged -- the caller (the
    orchestrator's terminal-success hook) treats failure as a missing
    nice-to-have, not a chain-blocker.
    """
    try:
        return await _write_snapshot_inner(
            output_dir=output_dir,
            db_path=db_path,
            tailor_run_id=tailor_run_id,
            resume_client=resume_client,
            letter_client=letter_client,
            apply_profile=apply_profile or {},
        )
    except Exception:  # noqa: BLE001 - snapshot is best-effort
        _log.warning(
            "tailor_snapshot_failed",
            extra={"tailor_run_id": tailor_run_id, "output_dir": str(output_dir)},
            exc_info=True,
        )
        return None


async def _write_snapshot_inner(
    *,
    output_dir: Path,
    db_path: Path,
    tailor_run_id: int,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    apply_profile: dict[str, str],
) -> Path | None:
    # Pull the row + the job's title/company under one connection.
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        record = get_tailor_run(conn, tailor_run_id)
        if record is None:
            _log.warning(
                "tailor_snapshot_row_missing",
                extra={"tailor_run_id": tailor_run_id},
            )
            return None
        title, company, apply_url = _resolve_job_metadata(conn, record)

    # Pull the parsed JD + tailored payloads from resumeai once so we
    # can write the JD markdown without an extra round-trip per file.
    resume_record: dict[str, object] = {}
    if record.resume_run_id:
        try:
            resume_record = await resume_client.get_run(record.resume_run_id)
        except Exception:  # noqa: BLE001 - degrade silently to placeholders
            _log.warning(
                "tailor_snapshot_resume_record_unreachable",
                extra={
                    "tailor_run_id": tailor_run_id,
                    "resume_run_id": record.resume_run_id,
                },
                exc_info=True,
            )

    # Bare-URL runs may have title/company on the sibling's parsed
    # requirements (the catalogue path already populated them above).
    if record.job_id is None and isinstance(resume_record, dict):
        reqs = resume_record.get("requirements") if resume_record else None
        if isinstance(reqs, dict):
            t = reqs.get("title")
            c = reqs.get("company")
            if isinstance(t, str) and t:
                title = t
            if isinstance(c, str) and c:
                company = c

    folder_name = _folder_name(record, title=title, company=company)
    folder = output_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    resume_name = (
        record.resume_filename
        or f"{sanitize_filename_part('Applicant', fallback='Applicant')}-Resume.pdf"
    )
    letter_name = (
        record.letter_filename
        or f"{sanitize_filename_part('Applicant', fallback='Applicant')}-CoverLetter.pdf"
    )

    # PDFs: stream the bytes once and write to disk.
    if record.resume_run_id:
        await _stream_pdf_to_disk(
            client=resume_client,
            run_id=record.resume_run_id,
            target=folder / resume_name,
        )
    if record.letter_run_id:
        await _stream_pdf_to_disk(
            client=letter_client,
            run_id=record.letter_run_id,
            target=folder / letter_name,
        )

    # JD markdown -- prefer the sibling's parsed requirements when
    # available (richer, structured), fall back to a stub pointing at
    # the apply URL otherwise.
    (folder / "jd.md").write_text(
        _build_jd_markdown(record, resume_record, title=title, company=company),
        encoding="utf-8",
    )

    # QA verdict as JSON for interviewai to consume.
    qa_path = folder / "qa.json"
    if record.qa_assessment is not None:
        qa_path.write_text(
            record.qa_assessment.model_dump_json(indent=2),
            encoding="utf-8",
        )
    else:
        qa_path.write_text("{}", encoding="utf-8")

    # Metadata: stable schema for downstream readers (interviewai).
    metadata = {
        "schema_version": 1,
        "tailor_run_id": record.id,
        "job_id": record.job_id,
        "apply_url": apply_url,
        "jd_url": record.jd_url,
        "resume_run_id": record.resume_run_id,
        "letter_run_id": record.letter_run_id,
        "resume_filename": resume_name,
        "letter_filename": letter_name,
        "qa_status": record.qa_status.value if record.qa_status else None,
        "qa_attempts": record.qa_attempts,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "snapshotted_at": datetime.now(tz=UTC).isoformat(),
        "title": title,
        "company": company,
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Per-job checklist. The applied-on line stays blank; the user
    # fills it in when they submit. The regenerate_index pass below
    # parses that line to flip the master INDEX from ☐ to ✅.
    (folder / "CHECKLIST.md").write_text(
        _build_checklist(
            title=title,
            company=company,
            apply_url=apply_url,
            resume_filename=resume_name,
            letter_filename=letter_name,
            profile=apply_profile,
        ),
        encoding="utf-8",
    )

    regenerate_index(output_dir)
    return folder


async def _stream_pdf_to_disk(
    *,
    client: ResumeaiClient | CoverletteraiClient,
    run_id: str,
    target: Path,
) -> None:
    """Stream a sibling PDF response and persist to ``target``.

    Errors get raised so the outer ``write_snapshot`` swallows them
    along with the rest -- one bad PDF shouldn't half-write a folder
    and look "done" to the user.
    """
    response = await client.stream_pdf(run_id)
    try:
        # Read the whole body at once -- PDFs are small (<200 KB) and
        # the streaming response would otherwise leak a connection if
        # we forgot to close it. The siblings already buffer the
        # rendered PDF in memory before returning anyway.
        body = await response.aread()
    finally:
        await response.aclose()
    # Sync write inside an async function. PDFs are small (<200 KB)
    # and the snapshot path is fire-and-forget at the tail of a chain
    # that's about to release its loop slot anyway; the trio.Path /
    # anyio.Path async-write rewrite isn't worth the dependency.
    target.write_bytes(body)  # noqa: ASYNC240 - small write, no contention


def _resolve_job_metadata(
    conn: sqlite3.Connection,
    record: TailorRunRecord,
) -> tuple[str, str, str | None]:
    """Return ``(title, company, apply_url)`` for the record.

    Catalogue runs read from the ``jobs`` row; bare-URL runs return
    defaults plus the on-row ``jd_url`` as the apply URL. The caller
    may overwrite ``title`` / ``company`` later from sibling data.
    """
    title = "Job"
    company = "Company"
    apply_url: str | None = record.jd_url
    if record.job_id is None:
        return title, company, apply_url
    row = conn.execute(
        "SELECT title, company, apply_url FROM jobs WHERE id = ?",
        (record.job_id,),
    ).fetchone()
    if row is None:  # pragma: no cover - jobs FK CASCADE means the run is gone too
        return title, company, apply_url
    title = row["title"] or title
    company = row["company"] or company
    apply_url = row["apply_url"] or apply_url
    return title, company, apply_url


def _build_jd_markdown(
    record: TailorRunRecord,
    resume_record: dict[str, object],
    *,
    title: str,
    company: str,
) -> str:
    """Compose the per-job ``jd.md`` body.

    Uses resumeai's parsed ``requirements`` when available -- it's
    pre-cleaned and structured (required_skills, responsibilities,
    nice_to_haves) so interviewai can parse it back if it wants. When
    only the apply URL is known we emit a stub pointing the reader
    at the source.
    """
    reqs = resume_record.get("requirements") if isinstance(resume_record, dict) else None
    lines: list[str] = [
        f"# {title} — {company}",
        "",
    ]
    if isinstance(reqs, dict):
        if record.jd_url:
            lines.append(f"Source: {record.jd_url}")
            lines.append("")
        summary = reqs.get("summary") or reqs.get("description")
        if isinstance(summary, str) and summary.strip():
            lines.append("## Summary")
            lines.append(summary.strip())
            lines.append("")
        for key, header in (
            ("required_skills", "Required skills"),
            ("responsibilities", "Responsibilities"),
            ("nice_to_haves", "Nice-to-haves"),
        ):
            values = reqs.get(key)
            if isinstance(values, list) and values:
                lines.append(f"## {header}")
                for v in values:
                    if isinstance(v, str) and v.strip():
                        lines.append(f"- {v.strip()}")
                lines.append("")
    else:
        lines.append(
            f"Source: {record.jd_url or '(not recorded)'}",
        )
        lines.append("")
        lines.append(
            "(The sibling's parsed JD wasn't available at snapshot "
            "time. Open the URL above for the live description.)",
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_checklist(
    *,
    title: str,
    company: str,
    apply_url: str | None,
    resume_filename: str,
    letter_filename: str,
    profile: dict[str, str],
) -> str:
    """Compose the per-job ``CHECKLIST.md`` body.

    Profile fields appear inline so the user can copy each one
    straight into the application form without bouncing back to a
    Settings page. The bottom "Applied on:" line is what
    ``regenerate_index`` parses to flip the master state.
    """
    lines: list[str] = [
        f"# {company} — {title}",
        "",
    ]
    if apply_url:
        lines.append(f"Apply URL: {apply_url}")
        lines.append("")
    lines.append("## Submit")
    lines.append("- [ ] Open apply URL above")
    lines.append(f"- [ ] Upload `{resume_filename}`")
    lines.append(f"- [ ] Upload `{letter_filename}`")
    for label, key in (
        ("Name", "full_name"),
        ("Email", "email"),
        ("Phone", "phone"),
        ("Location", "location"),
        ("LinkedIn", "linkedin_url"),
        ("GitHub", "github_url"),
        ("Right to work", "right_to_work"),
        ("Notice period", "notice_period"),
        ("Salary expectation", "salary_expectation"),
    ):
        value = profile.get(key)
        if value:
            lines.append(f"- [ ] {label}: {value}")
    lines.append("- [ ] Submit")
    lines.append("- [ ] Applied on: ____________")
    lines.append("")
    return "\n".join(lines)


def regenerate_index(output_dir: Path) -> Path:
    """Rebuild ``INDEX.md`` from every folder under ``output_dir``.

    Walks the output dir once, reads each folder's ``metadata.json``
    plus ``CHECKLIST.md`` (for the "Applied on:" line), and emits a
    flat checklist of every job sorted newest-first by
    ``snapshotted_at``. Cheap enough to run after every snapshot --
    20 folders is ~20 file reads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, bool, str, str, str, str]] = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        snapshotted_at = str(meta.get("snapshotted_at") or "")
        title = str(meta.get("title") or "Job")
        company = str(meta.get("company") or "Company")
        applied = _checklist_is_applied(child / "CHECKLIST.md")
        entries.append(
            (snapshotted_at, applied, company, title, child.name, str(meta.get("apply_url") or "")),
        )

    entries.sort(key=lambda e: e[0], reverse=True)

    lines: list[str] = [
        "# jobai tailored applications",
        "",
        (
            "Tick the ☐ in each folder's `CHECKLIST.md` and fill the "
            "`Applied on:` date to flip a row to ✅ here."
        ),
        "",
    ]
    for _ts, applied, company, title, folder, url in entries:
        mark = "✅" if applied else "☐"
        suffix = f" — {url}" if url else ""
        lines.append(f"- {mark} **{company}** — {title} — `{folder}/`{suffix}")
    lines.append("")

    index_path = output_dir / "INDEX.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def _checklist_is_applied(checklist_path: Path) -> bool:
    """Return True if the ``Applied on:`` line in the checklist has
    been filled in (any non-whitespace content past the underscore
    placeholder counts as applied)."""
    if not checklist_path.is_file():
        return False
    try:
        text = checklist_path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - defensive; file existed at is_file() check
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("- [ ] applied on:"):
            payload = stripped.split(":", 1)[1].strip().strip("_").strip()
            return bool(payload)
        if stripped.lower().startswith("- [x] applied on:"):
            return True
    return False


__all__ = [
    "regenerate_index",
    "write_snapshot",
]


def list_snapshot_folders(output_dir: Path) -> list[Path]:
    """Return every snapshot folder under ``output_dir`` (newest first).

    Used by tests and by the future interviewai integration to
    enumerate available applications without having to query the API.
    """
    if not output_dir.is_dir():
        return []
    folders: list[tuple[str, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        folders.append((str(meta.get("snapshotted_at") or ""), child))
    return [f for _, f in sorted(folders, key=lambda e: e[0], reverse=True)]


# Re-export list_tailor_runs for the (future) interviewai integration
# helper that wants to cross-reference disk folders with live DB rows.
__all__ += ["list_snapshot_folders", "list_tailor_runs"]
