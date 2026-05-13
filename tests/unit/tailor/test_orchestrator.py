"""Chain coverage for jobai.tailor.orchestrator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jobai.db.connection import connect
from jobai.tailor.models import QAAssessment, QAStatus, TailorRunStatus
from jobai.tailor.orchestrator import (
    TailorChainError,
    _load_apply_url,
    run_chain,
)
from jobai.tailor.repository import create_tailor_run, get_tailor_run
from tests.unit.tailor.conftest import (
    ScriptedLetterClient,
    ScriptedResumeClient,
    Sleeper,
)


class _ScriptedQAClient:
    """Tiny in-memory QA client: returns whatever JSON the test supplies."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        return self.response


def _good_qa_json() -> str:
    """A canned ``pass`` assessment for the happy-path chain tests."""
    import json  # noqa: PLC0415

    return json.dumps(
        {
            "status": "pass",
            "coverage_score": 90,
            "consistency_score": 85,
            "format_score": 88,
            "must_fix_issues": [],
            "nice_to_fix_issues": [],
            "summary": "Strong application.",
        },
    )


async def test_happy_path_walks_full_state_machine(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Resume kicks, letters kicks, both poll once and succeed."""
    delays, sleeper = recording_sleeper
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.resume_run_id == "rs_1"
    assert final.resume_status == "succeeded"
    assert final.letter_run_id == "ls_1"
    assert final.letter_status == "succeeded"
    assert final.finished_at is not None
    assert final.error is None

    # The kick requests carried the JD URL from the seeded job.
    assert scripted_resume_client.kick_requests[0].jd_url == "https://example.com/jd-1"
    assert scripted_letter_client.kick_requests[0].jd_url == "https://example.com/jd-1"
    assert scripted_letter_client.kick_requests[0].resume_run_id == "rs_1"
    # First poll returns terminal-succeeded so the sleeper should never have been called.
    assert delays == []


async def test_chain_polls_until_resume_terminal(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Resume goes through ``tailoring -> verifying -> succeeded`` before letter kicks."""
    delays, sleeper = recording_sleeper
    resume_client = ScriptedResumeClient(
        poll_statuses=["tailoring", "verifying", "succeeded"],
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        poll_interval_s=0.5,
    )

    assert resume_client.poll_calls == ["rs_1", "rs_1", "rs_1"]
    # Two sleeps fired between the three polls -- the third returned terminal.
    assert delays == [0.5, 0.5]


async def test_chain_fails_when_resume_returns_failed(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A resume run that ends in ``failed`` aborts the chain before letter kicks."""
    _, sleeper = recording_sleeper
    resume_client = ScriptedResumeClient(poll_statuses=["failed"])
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.FAILED
    assert final.error is not None
    assert "resumeai" in final.error
    # Letter client must not have been touched.
    assert scripted_letter_client.kick_requests == []


async def test_chain_fails_when_letter_returns_failed(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A cover-letter run that ends in ``failed`` flips the row to failed."""
    _, sleeper = recording_sleeper
    letter_client = ScriptedLetterClient(poll_statuses=["failed"])
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=letter_client,
        sleeper=sleeper,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.FAILED
    assert final.error is not None
    assert "coverletterai" in final.error
    # The resume artefact survives so the user can still download what worked.
    assert final.resume_run_id == "rs_1"


async def test_chain_fails_on_resume_kick_exception(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A network-level error during the resume kick is recorded as ``failed``."""
    _, sleeper = recording_sleeper
    resume_client = ScriptedResumeClient(kick_error=RuntimeError("dns-timeout"))
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.FAILED
    assert final.error == "dns-timeout"


async def test_chain_fails_when_poll_cap_hit(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Sibling that never terminates within the poll cap fails out."""
    _, sleeper = recording_sleeper
    resume_client = ScriptedResumeClient(poll_statuses=["tailoring"])
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        max_polls=3,
        poll_interval_s=0.1,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.FAILED
    assert "did not terminate" in (final.error or "")


async def test_chain_fails_when_tailor_run_row_missing(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A non-existent tailor_run id fails gracefully without sibling calls."""
    _, sleeper = recording_sleeper
    await run_chain(
        9999,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
    )
    assert scripted_resume_client.kick_requests == []


async def test_happy_path_runs_qa_when_client_supplied(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """With ``qa_client`` supplied, the chain walks an extra qa_running
    stage and persists the structured assessment on the row."""
    _, sleeper = recording_sleeper
    qa = _ScriptedQAClient(_good_qa_json())
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        qa_model="claude-opus-4-7",
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.qa_status is QAStatus.PASS
    assert isinstance(final.qa_assessment, QAAssessment)
    assert final.qa_assessment.coverage_score == 90
    # The QA client saw the JD + tailored payloads pulled from siblings.
    assert qa.calls and "Engineer" in str(qa.calls[0]["user"])
    assert scripted_resume_client.get_run_calls == ["rs_1"]
    assert scripted_letter_client.get_run_calls == ["ls_1"]


async def test_qa_skipped_when_no_client_supplied(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Without a ``qa_client``, the chain skips the QA stage entirely
    -- the run terminates at succeeded with no qa_status persisted."""
    _, sleeper = recording_sleeper
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=None,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.qa_status is None
    assert final.qa_assessment is None
    # Neither sibling client was asked for its full record (we only
    # need that for the QA stage).
    assert scripted_resume_client.get_run_calls == []
    assert scripted_letter_client.get_run_calls == []


async def test_qa_stage_handles_unparseable_response_gracefully(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A malformed QA response becomes a synthetic ``fail`` assessment.
    The chain still completes at succeeded -- the PDFs ship regardless."""
    _, sleeper = recording_sleeper
    qa = _ScriptedQAClient("garbage not-json")
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.qa_status is QAStatus.FAIL
    assert final.qa_assessment is not None
    assert "could not be parsed" in final.qa_assessment.must_fix_issues[0].summary


async def test_qa_stage_handles_non_dict_sibling_payloads(
    tailor_db_path: Path,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """When a sibling's ``get_run`` returns a payload whose ``tailored``
    field isn't a dict (eg a future SDK change), the QA stage forwards
    ``None`` for that input rather than crashing the chain."""
    _, sleeper = recording_sleeper
    # Resume returns a record where ``tailored`` is a string (not a dict).
    resume = ScriptedResumeClient(
        run_record={
            "id": "rs_1",
            "status": "succeeded",
            "requirements": "not-a-dict",
            "tailored": "also-not-a-dict",
        },
    )
    # Letter returns a normal record so the assessment can still grade
    # one of the two artefacts.
    letter = ScriptedLetterClient()
    qa = _ScriptedQAClient(_good_qa_json())
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume,
        letter_client=letter,
        sleeper=sleeper,
        qa_client=qa,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.qa_status is QAStatus.PASS


def test_load_apply_url_raises_when_job_deleted(tailor_db_path: Path) -> None:
    """If the canonical job is gone, the chain refuses to fire off a NULL URL.

    We construct the race condition by inserting an orphan ``tailor_runs``
    row with FK enforcement off (simulating a job deleted out from under
    an in-flight chain), then re-enabling FKs for the read path.
    """
    bare = sqlite3.connect(tailor_db_path)
    try:
        # Use the same connection (FK off) for both insert + delete so the
        # cascade doesn't take the tailor_run with it.
        bare.execute("PRAGMA foreign_keys=OFF")
        cursor = bare.execute(
            "INSERT INTO tailor_runs (job_id, status, created_at, updated_at) "
            "VALUES (?, 'pending', datetime('now'), datetime('now'))",
            (999_999,),  # job_id that doesn't exist
        )
        orphan_id = int(cursor.lastrowid or 0)
        bare.commit()
    finally:
        bare.close()

    with pytest.raises(TailorChainError, match="no longer exists"):
        _load_apply_url(tailor_db_path, orphan_id)


def test_load_apply_url_returns_jd_url_when_present(tailor_db_path: Path) -> None:
    """One-off URL-tailor rows carry the JD URL on tailor_runs.jd_url;
    the loader prefers that over the jobs join so the chain works for
    JDs jobai never scraped."""
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, jd_url="https://example.com/off-network")
    assert _load_apply_url(tailor_db_path, record.id) == "https://example.com/off-network"
