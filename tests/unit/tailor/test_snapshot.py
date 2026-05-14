"""Coverage for jobai.tailor.snapshot.

Tests the on-disk folder writer end-to-end without standing up the full
orchestrator. Sibling clients are scripted via ``ScriptedResumeClient`` /
``ScriptedLetterClient``; the DB is a fresh migrated SQLite per test.
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
    _checklist_is_applied,
    _folder_name,
    list_snapshot_folders,
    regenerate_index,
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
    """A successful tailor run produces a folder containing resume,
    letter, jd.md, qa.json, metadata.json, and CHECKLIST.md."""
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
        apply_profile={
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+61 400 000 000",
            "linkedin_url": "linkedin.com/in/janedoe",
        },
    )

    assert folder is not None
    assert folder.is_dir()
    # All artefacts present. The catalogue-path run uses the cached
    # filename from the row (`Jane_Doe-Software_Engineer-Acme-...`),
    # set via ``_mark_succeeded`` above.
    pdf_names = {p.name for p in folder.iterdir() if p.suffix == ".pdf"}
    assert "Jane_Doe-Software_Engineer-Acme-Resume.pdf" in pdf_names
    assert "Jane_Doe-Software_Engineer-Acme-CoverLetter.pdf" in pdf_names
    assert (folder / "Jane_Doe-Software_Engineer-Acme-Resume.pdf").read_bytes() == pdf_bytes
    assert (folder / "jd.md").is_file()
    assert (folder / "qa.json").is_file()
    assert (folder / "metadata.json").is_file()
    assert (folder / "CHECKLIST.md").is_file()

    # JD markdown for the catalogue path uses the jobs row's title +
    # company (seed fixture: title='Engineer', company='Acme'). The
    # sibling's parsed requirements still feed the summary + skills
    # sections so QA-relevant content is preserved either way.
    jd_md = (folder / "jd.md").read_text(encoding="utf-8")
    assert "# Engineer — Acme" in jd_md
    assert "## Required skills" in jd_md
    assert "- Python" in jd_md
    assert "Build things." in jd_md

    # QA JSON parses cleanly back to a QAAssessment shape.
    qa_data = json.loads((folder / "qa.json").read_text(encoding="utf-8"))
    assert qa_data["status"] == "pass"
    assert qa_data["coverage_score"] == 90

    # Metadata has the stable schema interviewai will read.
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["tailor_run_id"] == record.id
    assert meta["title"] == "Engineer"
    assert meta["company"] == "Acme"
    assert meta["qa_status"] == "pass"

    # Checklist embeds the profile fields inline so the user can copy-
    # paste straight into apply forms.
    checklist = (folder / "CHECKLIST.md").read_text(encoding="utf-8")
    assert "Acme — Engineer" in checklist
    assert "Jane Doe" in checklist
    assert "jane@example.com" in checklist
    assert "linkedin.com/in/janedoe" in checklist
    assert "- [ ] Submit" in checklist
    assert "Applied on: ____________" in checklist

    # Master INDEX.md regenerated and references the new folder.
    index = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "Acme" in index
    assert "Engineer" in index
    assert folder.name in index
    assert "☐" in index


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
    # Bare-URL run -> folder name uses run<id> as the disambiguator,
    # not job<job_id>, since there's no catalogue job behind it.
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


def test_regenerate_index_marks_applied_when_checklist_filled(
    tmp_path: Path,
) -> None:
    """If a per-job CHECKLIST.md has 'Applied on: 2026-05-14', the
    master INDEX flips that row to ✅."""
    folder_a = tmp_path / "Acme-Engineer-job1"
    folder_a.mkdir()
    (folder_a / "metadata.json").write_text(
        json.dumps(
            {
                "snapshotted_at": "2026-05-14T10:00:00+00:00",
                "title": "Engineer",
                "company": "Acme",
                "apply_url": "https://acme/jobs/1",
            },
        ),
    )
    (folder_a / "CHECKLIST.md").write_text(
        "# Acme — Engineer\n- [ ] Submit\n- [ ] Applied on: 2026-05-14\n",
    )

    folder_b = tmp_path / "Globex-Engineer-job2"
    folder_b.mkdir()
    (folder_b / "metadata.json").write_text(
        json.dumps(
            {
                "snapshotted_at": "2026-05-14T11:00:00+00:00",
                "title": "Engineer",
                "company": "Globex",
                "apply_url": "https://globex/jobs/2",
            },
        ),
    )
    (folder_b / "CHECKLIST.md").write_text(
        "# Globex — Engineer\n- [ ] Submit\n- [ ] Applied on: ____________\n",
    )

    index_path = regenerate_index(tmp_path)
    text = index_path.read_text(encoding="utf-8")
    # Globex is newer -> appears first (sort by snapshotted_at desc).
    globex_pos = text.index("Globex")
    acme_pos = text.index("Acme")
    assert globex_pos < acme_pos
    # Acme is applied -> ✅; Globex is not -> ☐.
    assert "✅ **Acme**" in text
    assert "☐ **Globex**" in text


def test_regenerate_index_skips_folders_without_metadata(
    tmp_path: Path,
) -> None:
    """A folder under output_dir that doesn't have metadata.json
    (e.g. user-created scratch dir, partial snapshot) is ignored.
    Same for a folder whose metadata.json is corrupt JSON."""
    (tmp_path / "no-meta").mkdir()
    bad = tmp_path / "bad-meta"
    bad.mkdir()
    (bad / "metadata.json").write_text("{ not json")

    good = tmp_path / "Acme-Engineer-job1"
    good.mkdir()
    (good / "metadata.json").write_text(
        json.dumps(
            {
                "snapshotted_at": "2026-05-14T10:00:00+00:00",
                "title": "Engineer",
                "company": "Acme",
                "apply_url": "",
            },
        ),
    )

    index = regenerate_index(tmp_path).read_text(encoding="utf-8")
    assert "Acme" in index
    assert "no-meta" not in index
    assert "bad-meta" not in index


def test_regenerate_index_skips_files_at_root(tmp_path: Path) -> None:
    """A regular file at the output_dir root (the INDEX.md itself, a
    .DS_Store, etc) is ignored by the walker."""
    (tmp_path / "stray.txt").write_text("not a folder")
    index = regenerate_index(tmp_path).read_text(encoding="utf-8")
    # No entries -> just the header and the help line.
    assert "stray.txt" not in index


def test_checklist_is_applied_recognises_filled_in_dates() -> None:
    """The applied-on detector accepts any non-whitespace, non-
    underscore-only content as 'applied'. Also catches the
    ticked-box form '- [x] Applied on:'."""
    from tempfile import NamedTemporaryFile  # noqa: PLC0415

    with NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# job\n- [ ] Applied on: 2026-05-14\n")
        path = Path(f.name)
    assert _checklist_is_applied(path) is True

    with NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# job\n- [ ] Applied on: ____________\n")
        path = Path(f.name)
    assert _checklist_is_applied(path) is False

    with NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# job\n- [x] Applied on: 2026-05-14\n")
        path = Path(f.name)
    assert _checklist_is_applied(path) is True

    # Missing file -> defaults to False (defensive guard).
    assert _checklist_is_applied(Path("/nonexistent/CHECKLIST.md")) is False


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
    # A stray file at the root (the INDEX.md we'd write, .DS_Store on
    # macOS, etc) must be skipped without crashing the walker.
    (tmp_path / "INDEX.md").write_text("# index")

    listing = list_snapshot_folders(tmp_path)
    assert [p.name for p in listing] == ["New-Engineer-job2", "Old-Engineer-job1"]


async def test_write_snapshot_skips_sibling_fetch_when_run_ids_missing(
    tailor_db_path: Path,
    tmp_path: Path,
) -> None:
    """A tailor row that somehow reached SUCCEEDED without sibling
    run ids (test seed, replay, migration weirdness) still snapshots
    -- no PDFs land but jd.md + qa.json + metadata.json + CHECKLIST.md
    do, so the user can at least see what JD was queued."""
    resume_client = ScriptedResumeClient()  # never called -- ids absent
    letter_client = ScriptedLetterClient()
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
        # Mark SUCCEEDED but DO NOT set resume_run_id / letter_run_id.
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
    # No PDFs (no run ids), but supporting files exist.
    pdfs = [p for p in folder.iterdir() if p.suffix == ".pdf"]
    assert pdfs == []
    assert (folder / "jd.md").is_file()
    assert (folder / "metadata.json").is_file()
    assert (folder / "CHECKLIST.md").is_file()


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
            # No requirements key at all -- mimics a sibling response
            # where JD parsing failed mid-run.
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


def test_checklist_is_applied_returns_false_when_no_applied_line() -> None:
    """A checklist file that doesn't contain an 'Applied on:' line at
    all (custom template, user edited it heavily) defaults to not-
    applied. Same fail-safe as a missing file."""
    from tempfile import NamedTemporaryFile  # noqa: PLC0415

    with NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# job\nNo applied line here.\n- [ ] something else\n")
        path = Path(f.name)
    assert _checklist_is_applied(path) is False


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
                "",  # blank string -> skipped
                None,  # not a string -> skipped
                42,  # not a string -> skipped
                "AWS",
            ],
        },
    }
    md = _build_jd_markdown(record, resume_record, title="Eng", company="Acme")
    assert "- Python" in md
    assert "- AWS" in md
    # The garbage entries don't surface as empty bullets.
    assert "- \n" not in md
    assert "- None" not in md
    assert "- 42" not in md


def test_build_checklist_omits_apply_url_line_when_none() -> None:
    """A run with no apply URL (sibling didn't return one + no jobs
    row had one cached) skips the 'Apply URL:' line in the checklist
    -- we don't emit a bullet pointing to nothing."""
    from jobai.tailor.snapshot import _build_checklist  # noqa: PLC0415

    text = _build_checklist(
        title="Engineer",
        company="Acme",
        apply_url=None,
        resume_filename="r.pdf",
        letter_filename="l.pdf",
        profile={},
    )
    assert "Apply URL:" not in text
    assert "## Submit" in text
    assert "Open apply URL above" in text  # still present as the step name


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
                "title": "",  # explicit empty
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
    # Empty strings get rejected; Job/Company defaults stick.
    assert folder.name.startswith("Company-Job-")


@pytest.fixture
def tailor_db_path(tmp_path: Path) -> Path:
    """Migrated SQLite DB with one seeded job (id=1) the snapshot
    tests target for their tailor runs.

    Duplicated from ``conftest.py`` rather than importing because
    that fixture lives in the tailor conftest and pytest's fixture
    discovery is per-directory; importing it directly would be a
    surprising cross-module coupling.
    """
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
