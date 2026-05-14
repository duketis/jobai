"""Coverage for jobai.tailor.snapshot.

v1.18.0 stripped the CHECKLIST.md + INDEX.md writers -- applied
state lives in ``tailor_runs.applied_at`` now, and the per-folder
files are PDFs + jd.md + qa.json + metadata.json only. These tests
cover that reduced contract end-to-end without standing up the full
orchestrator.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from jobai.db.connection import connect
from jobai.tailor.models import QAAssessment, QAIssue, QAStatus, TailorRunStatus
from jobai.tailor.repository import create_tailor_run, update_status
from jobai.tailor.snapshot import (
    _folder_name,
    list_snapshot_folders,
    write_snapshot,
)
from tests.unit.tailor.conftest import (
    ScriptedLetterClient,
    ScriptedResumeClient,
)


def _qa_pass() -> QAAssessment:
    """Canned pass verdict for snapshot tests."""
    return QAAssessment(
        status=QAStatus.PASS,
        coverage_score=90,
        consistency_score=88,
        format_score=92,
        must_fix_issues=[],
        nice_to_fix_issues=[
            QAIssue(severity="nice_to_fix", category="content", summary="polish item"),
        ],
        summary="Solid submission.",
    )


def _mark_succeeded(
    db_path: Path,
    tailor_run_id: int,
    *,
    resume_filename: str = "Jane_Doe-Software_Engineer-Acme-Resume.pdf",
    letter_filename: str = "Jane_Doe-Software_Engineer-Acme-CoverLetter.pdf",
    qa: QAAssessment | None = None,
) -> None:
    """Move a freshly-created run to terminal SUCCESS with cached filenames + IDs."""
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.SUCCEEDED,
            resume_run_id="rs_1",
            resume_status="succeeded",
            letter_run_id="ls_1",
            letter_status="succeeded",
            qa_status=QAStatus.PASS if qa is not None else None,
            qa_assessment=qa,
            qa_attempts=1,
            resume_filename=resume_filename,
            letter_filename=letter_filename,
        )


async def test_write_snapshot_happy_path_writes_full_folder(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A successful tailor run produces a folder containing both
    PDFs, jd.md, qa.json, and metadata.json. No CHECKLIST.md or
    INDEX.md (those were removed in v1.18.0 -- applied state now
    lives in the DB)."""
    pdf_bytes = b"%PDF-1.4 fake"
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
        run_record={
            "id": "rs_1",
            "tailored": {"name": "Jane Doe"},
            "requirements": {
                "title": "Software Engineer",
                "company": "Acme",
                "required_skills": ["Python", "AWS"],
                "responsibilities": ["Ship code"],
                "summary": "Build things.",
            },
        },
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    _mark_succeeded(tailor_db_path, record.id, qa=_qa_pass())

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )

    assert folder is not None
    assert folder.is_dir()
    pdf_names = {p.name for p in folder.iterdir() if p.suffix == ".pdf"}
    assert "Jane_Doe-Software_Engineer-Acme-Resume.pdf" in pdf_names
    assert "Jane_Doe-Software_Engineer-Acme-CoverLetter.pdf" in pdf_names
    assert (folder / "Jane_Doe-Software_Engineer-Acme-Resume.pdf").read_bytes() == pdf_bytes
    assert (folder / "jd.md").is_file()
    assert (folder / "qa.json").is_file()
    assert (folder / "metadata.json").is_file()
    # v1.18.0: the markdown files are gone.
    assert not (folder / "CHECKLIST.md").exists()
    assert not (tmp_path / "INDEX.md").exists()

    jd_md = (folder / "jd.md").read_text(encoding="utf-8")
    assert "# Engineer — Acme" in jd_md
    assert "## Required skills" in jd_md
    assert "- Python" in jd_md
    assert "Build things." in jd_md

    qa_data = json.loads((folder / "qa.json").read_text(encoding="utf-8"))
    assert qa_data["status"] == "pass"
    assert qa_data["coverage_score"] == 90

    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["tailor_run_id"] == record.id
    assert meta["title"] == "Engineer"
    assert meta["company"] == "Acme"
    assert meta["qa_status"] == "pass"


async def test_write_snapshot_returns_none_when_run_missing(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A snapshot for a nonexistent run id logs and returns None;
    no snapshot folder gets created under the output dir."""
    output_dir = tmp_path / "tailored-output"
    folder = await write_snapshot(
        output_dir=output_dir,
        db_path=tailor_db_path,
        tailor_run_id=99_999,
        resume_client=ScriptedResumeClient(),
        letter_client=ScriptedLetterClient(),
    )
    assert folder is None
    assert not output_dir.exists()


async def test_write_snapshot_degrades_when_sibling_get_run_fails(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """If the sibling 5xx's during get_run, we still write what we
    have. PDFs come from stream_pdf which is independent."""
    pdf_bytes = b"%PDF-1.4 fake"

    class _BoomResume(ScriptedResumeClient):
        async def get_run(self, run_id: str) -> dict[str, object]:
            msg = "resumeai 503"
            raise RuntimeError(msg)

    resume_client = _BoomResume(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    _mark_succeeded(tailor_db_path, record.id, qa=_qa_pass())

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    # JD markdown falls back to the stub since requirements unreachable.
    jd_md = (folder / "jd.md").read_text(encoding="utf-8")
    assert "Source:" in jd_md


async def test_write_snapshot_handles_missing_qa_assessment(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A run that ended successfully but never ran QA (qa_client=None
    branch) still snapshots -- qa.json gets an empty object."""
    pdf_bytes = b"%PDF-1.4 fake"
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    _mark_succeeded(tailor_db_path, record.id)  # qa=None

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    assert (folder / "qa.json").read_text(encoding="utf-8") == "{}"


async def test_write_snapshot_falls_back_to_defaults_when_no_filenames(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A run whose terminal SUCCESS didn't cache resume/letter
    filenames (pre-v1.15.0 row, sibling outage during caching) still
    produces a folder using Applicant fallbacks."""
    pdf_bytes = b"%PDF-1.4 fake"
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
        update_status(
            conn,
            record.id,
            status=TailorRunStatus.SUCCEEDED,
            resume_run_id="rs_1",
            letter_run_id="ls_1",
        )

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    files = {p.name for p in folder.iterdir()}
    assert "Applicant-Resume.pdf" in files
    assert "Applicant-CoverLetter.pdf" in files


async def test_write_snapshot_bare_url_uses_requirements_for_folder_name(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A POST /api/tailor/url run with no catalogue match pulls title
    + company from the sibling's parsed requirements so the folder
    still gets a useful name (no 'Job-Company' placeholder)."""
    pdf_bytes = b"%PDF-1.4 fake"
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
        run_record={
            "id": "rs_1",
            "tailored": {"name": "Jane Doe"},
            "requirements": {
                "title": "Senior Backend Engineer",
                "company": "Globex",
            },
        },
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, jd_url="https://example.com/jd-x")
    _mark_succeeded(tailor_db_path, record.id, qa=_qa_pass())

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    assert folder.name.startswith("Globex-Senior_Backend_Engineer-")
    assert folder.name.endswith(f"run{record.id}")


async def test_write_snapshot_swallows_disk_write_failure(
    tailor_db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk-full / permissions error during the folder write logs
    and returns None rather than propagating -- snapshotting is
    best-effort, the row already settled SUCCEEDED."""
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    _mark_succeeded(tailor_db_path, record.id, qa=_qa_pass())

    def _explode(self: Path, *args: Any, **kwargs: Any) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(Path, "mkdir", _explode)

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=ScriptedResumeClient(
            stream_response=httpx.Response(200, content=b"%PDF-1.4"),
        ),
        letter_client=ScriptedLetterClient(
            stream_response=httpx.Response(200, content=b"%PDF-1.4"),
        ),
    )
    assert folder is None


def test_folder_name_sanitises_unsafe_characters() -> None:
    """Names with slashes, colons, control chars produce a filesystem-
    safe folder name with the per-part underscoring intact."""
    from jobai.tailor.models import TailorRunRecord  # noqa: PLC0415

    record = TailorRunRecord(
        id=42,
        job_id=99,
        jd_url=None,
        status=TailorRunStatus.SUCCEEDED,
        created_at="2026-05-14T00:00:00+00:00",
        updated_at="2026-05-14T00:00:00+00:00",
    )
    name = _folder_name(record, title="Senior / Staff Eng", company="Acme: Co.")
    assert "/" not in name
    assert ":" not in name
    assert name.endswith("job99")


def test_list_snapshot_folders_returns_empty_when_dir_missing(
    tmp_path: Path,
) -> None:
    """No output dir yet (cold start) -> empty list, no crash."""
    assert list_snapshot_folders(tmp_path / "does-not-exist") == []


def test_list_snapshot_folders_sorts_newest_first(tmp_path: Path) -> None:
    """The discovery helper interviewai will use sorts by
    snapshotted_at descending so the most recent applications surface
    first. Stray files / folders without metadata are skipped silently."""
    old = tmp_path / "Old-Engineer-job1"
    old.mkdir()
    (old / "metadata.json").write_text(
        json.dumps({"snapshotted_at": "2026-01-01T00:00:00+00:00"}),
    )
    new = tmp_path / "New-Engineer-job2"
    new.mkdir()
    (new / "metadata.json").write_text(
        json.dumps({"snapshotted_at": "2026-05-14T00:00:00+00:00"}),
    )
    bad = tmp_path / "Bad-Meta-job3"
    bad.mkdir()
    (bad / "metadata.json").write_text("{ broken")
    (tmp_path / "no-meta").mkdir()
    (tmp_path / "stray.txt").write_text("not a folder")

    listing = list_snapshot_folders(tmp_path)
    assert [p.name for p in listing] == ["New-Engineer-job2", "Old-Engineer-job1"]


async def test_write_snapshot_skips_sibling_fetch_when_run_ids_missing(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A tailor row that somehow reached SUCCEEDED without sibling
    run ids (test seed, replay, migration weirdness) still snapshots
    -- no PDFs land but jd.md + qa.json + metadata.json do, so the
    user can at least see what JD was queued."""
    resume_client = ScriptedResumeClient()  # never called -- ids absent
    letter_client = ScriptedLetterClient()
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
        update_status(
            conn,
            record.id,
            status=TailorRunStatus.SUCCEEDED,
            qa_status=QAStatus.PASS,
            qa_assessment=_qa_pass(),
        )

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    pdfs = [p for p in folder.iterdir() if p.suffix == ".pdf"]
    assert pdfs == []
    assert (folder / "jd.md").is_file()
    assert (folder / "metadata.json").is_file()


async def test_write_snapshot_bare_url_with_non_dict_requirements(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """Bare-URL run whose sibling returned a payload with no usable
    'requirements' block (e.g. JD wasn't parsed, partial response)
    falls back to the Job/Company defaults rather than crashing on
    .get() against a non-dict."""
    pdf_bytes = b"%PDF-1.4 fake"
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
        run_record={
            "id": "rs_1",
            "tailored": {"name": "Jane Doe"},
        },
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, jd_url="https://example.com/jd-x")
    _mark_succeeded(tailor_db_path, record.id, qa=_qa_pass())

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    assert folder.name.startswith("Company-Job-run")


async def test_write_snapshot_with_blank_title_in_requirements(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """Bare-URL run where the sibling's requirements block has empty
    title/company strings: the helper keeps the Job/Company defaults
    rather than overwriting with empty values."""
    pdf_bytes = b"%PDF-1.4 fake"
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
        run_record={
            "id": "rs_1",
            "tailored": {"name": "Jane Doe"},
            "requirements": {
                "title": "",
                "company": "",
            },
        },
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf_bytes),
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, jd_url="https://example.com/jd-x")
    _mark_succeeded(tailor_db_path, record.id, qa=_qa_pass())

    folder = await write_snapshot(
        output_dir=tmp_path,
        db_path=tailor_db_path,
        tailor_run_id=record.id,
        resume_client=resume_client,
        letter_client=letter_client,
    )
    assert folder is not None
    assert folder.name.startswith("Company-Job-")


def test_build_jd_markdown_skips_non_string_skill_entries() -> None:
    """A sibling that returns garbage entries in required_skills (None
    placeholders, integers from a misparse) gets filtered: each entry
    must be a non-empty str before it's emitted as a bullet."""
    from jobai.tailor.models import TailorRunRecord  # noqa: PLC0415
    from jobai.tailor.snapshot import _build_jd_markdown  # noqa: PLC0415

    record = TailorRunRecord(
        id=1,
        job_id=1,
        jd_url="https://example.com/jd",
        status=TailorRunStatus.SUCCEEDED,
        created_at="2026-05-14T00:00:00+00:00",
        updated_at="2026-05-14T00:00:00+00:00",
    )
    resume_record: dict[str, object] = {
        "requirements": {
            "required_skills": [
                "Python",
                "",
                None,
                42,
                "AWS",
            ],
        },
    }
    md = _build_jd_markdown(record, resume_record, title="Eng", company="Acme")
    assert "- Python" in md
    assert "- AWS" in md
    assert "- \n" not in md
    assert "- None" not in md
    assert "- 42" not in md


@pytest.fixture
def tailor_db_path(tmp_path: Path) -> Path:
    """Migrated SQLite DB with one seeded job (id=1) the snapshot
    tests target for their tailor runs."""
    from jobai.db.migrations import apply_pending  # noqa: PLC0415
    from tests.unit.tailor.conftest import _seed_one_job  # noqa: PLC0415

    db = tmp_path / "snapshot-test.db"
    conn = sqlite3.connect(db)
    try:
        apply_pending(conn)
        _seed_one_job(conn)
    finally:
        conn.close()
    return db
