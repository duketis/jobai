"""Chain coverage for jobai.tailor.orchestrator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jobai.db.connection import connect
from jobai.tailor.models import QAAssessment, QAStatus, TailorRunStatus
from jobai.tailor.orchestrator import (
    TailorChainError,
    _load_jd_payload,
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
    # Two get_run calls on resumeai: one for QA, one for filename cache
    # at terminal SUCCESS (batched via build_pdf_filenames so the letter
    # filename comes free).
    assert scripted_resume_client.get_run_calls == ["rs_1", "rs_1"]
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
    # The letter sibling was never asked for its full record (we only
    # need that for the QA stage). The resume sibling is hit once at
    # terminal SUCCESS to build the cached filenames so the frontend
    # can render proper download names.
    assert scripted_resume_client.get_run_calls == ["rs_1"]
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
        _load_jd_payload(tailor_db_path, orphan_id)


def test_load_jd_payload_returns_jd_url_when_present(tailor_db_path: Path) -> None:
    """One-off URL-tailor rows carry the JD URL on tailor_runs.jd_url;
    the loader returns it directly with no jd_text (we have no
    description on hand for off-network URLs)."""
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, jd_url="https://example.com/off-network")
    payload = _load_jd_payload(tailor_db_path, record.id)
    assert payload.jd_url == "https://example.com/off-network"
    assert payload.jd_text is None


def test_load_jd_payload_prefers_description_text_when_present(
    tailor_db_path: Path,
) -> None:
    """Catalogue path with a populated description_text returns it on
    jd_text so the siblings skip the URL fetch entirely. Without this
    the chain depends on resumeai's HTTP tier surviving anti-bot 403s
    on the JD page."""
    bare = sqlite3.connect(tailor_db_path)
    try:
        bare.execute(
            "UPDATE jobs SET description_text = ? WHERE id = 1",
            ("x" * 500,),  # well above the 200-char usefulness floor
        )
        bare.commit()
    finally:
        bare.close()
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    payload = _load_jd_payload(tailor_db_path, record.id)
    assert payload.jd_url == "https://example.com/jd-1"
    assert payload.jd_text is not None
    assert len(payload.jd_text) >= 500


def test_load_jd_payload_strips_html_when_only_html_is_present(
    tailor_db_path: Path,
) -> None:
    """SmartRecruiters et al populate description_html but leave
    description_text null. The loader strips the HTML so we still
    get a text payload without depending on a sibling fetch."""
    bare = sqlite3.connect(tailor_db_path)
    try:
        bare.execute(
            "UPDATE jobs SET description_html = ? WHERE id = 1",
            ("<p>" + ("Senior Backend Engineer with strong Python and AWS. " * 20) + "</p>",),
        )
        bare.commit()
    finally:
        bare.close()
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    payload = _load_jd_payload(tailor_db_path, record.id)
    assert payload.jd_text is not None
    assert "Senior Backend Engineer" in payload.jd_text
    # No HTML tags survived.
    assert "<p>" not in payload.jd_text


def test_load_jd_payload_falls_back_to_url_when_text_too_short(
    tailor_db_path: Path,
) -> None:
    """A description blob below the usefulness threshold (placeholder
    text, 'see full description on the apply page' etc.) is treated
    as missing -- we fall back to the URL rather than send the model
    a fragment."""
    bare = sqlite3.connect(tailor_db_path)
    try:
        bare.execute(
            "UPDATE jobs SET description_text = ?, description_html = NULL WHERE id = 1",
            ("see full description",),  # 20 chars, well below 200
        )
        bare.commit()
    finally:
        bare.close()
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    payload = _load_jd_payload(tailor_db_path, record.id)
    assert payload.jd_text is None
    assert payload.jd_url == "https://example.com/jd-1"


def test_load_jd_payload_short_text_with_html_falls_through_to_html(
    tailor_db_path: Path,
) -> None:
    """When description_text is too short to be useful but description_html
    is populated, the loader strips the HTML rather than falling back
    to the URL."""
    bare = sqlite3.connect(tailor_db_path)
    try:
        bare.execute(
            "UPDATE jobs SET description_text = ?, description_html = ? WHERE id = 1",
            (
                "tagline only",  # too short
                "<p>" + ("Real long description body. " * 20) + "</p>",
            ),
        )
        bare.commit()
    finally:
        bare.close()
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    payload = _load_jd_payload(tailor_db_path, record.id)
    assert payload.jd_text is not None
    assert "Real long description body" in payload.jd_text


def test_load_jd_payload_returns_none_text_when_html_strips_to_nothing(
    tailor_db_path: Path,
) -> None:
    """description_html that's just tags / whitespace shouldn't produce
    a 0-char jd_text -- we treat it as missing and fall back to URL."""
    bare = sqlite3.connect(tailor_db_path)
    try:
        bare.execute(
            "UPDATE jobs SET description_text = NULL, description_html = ? WHERE id = 1",
            ("<div>  <span>  </span>  </div>",),
        )
        bare.commit()
    finally:
        bare.close()
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import create_tailor_run  # noqa: PLC0415

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)
    payload = _load_jd_payload(tailor_db_path, record.id)
    assert payload.jd_text is None
    assert payload.jd_url == "https://example.com/jd-1"


def test_build_resume_request_prefers_jd_text() -> None:
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _build_letter_request,
        _build_resume_request,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://example.com/x", jd_text="body text")
    req = _build_resume_request(payload)
    assert req.jd_text == "body text"
    assert req.jd_url is None

    letter = _build_letter_request(payload, resume_run_id="rs_1")
    assert letter.jd_text == "body text"
    assert letter.jd_url is None
    assert letter.resume_run_id == "rs_1"


def test_strip_html_to_text_returns_none_for_empty_input() -> None:
    """The helper short-circuits on empty / None input before invoking
    the parser, so callers that have description_html=None get None
    back without touching selectolax."""
    from jobai.tailor.orchestrator import _strip_html_to_text  # noqa: PLC0415

    assert _strip_html_to_text(None) is None
    assert _strip_html_to_text("") is None


def test_strip_html_to_text_returns_none_when_only_tags() -> None:
    """``<br/><br/>``-style content yields an empty string after stripping;
    the helper returns None so the loader's 'fall back to URL' branch
    fires cleanly."""
    from jobai.tailor.orchestrator import _strip_html_to_text  # noqa: PLC0415

    assert _strip_html_to_text("<br/><br/><hr/>") is None


def test_build_resume_request_falls_back_to_jd_url() -> None:
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _build_letter_request,
        _build_resume_request,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://example.com/x", jd_text=None)
    req = _build_resume_request(payload)
    assert req.jd_url == "https://example.com/x"
    assert req.jd_text is None

    letter = _build_letter_request(payload, resume_run_id="rs_1")
    assert letter.jd_url == "https://example.com/x"
    assert letter.jd_text is None


# ---------------------------------------------------------------------------
# QA auto-fix loop: re-kick the letter with feedback on must-fix issues
# ---------------------------------------------------------------------------


def _qa_fail_then_pass_responses() -> list[str]:
    """First QA pass returns a fail with a must-fix; second returns pass.

    Drives the orchestrator's retry path: QA must-fix → letter re-kick
    with feedback → re-poll → second QA → pass → succeeded."""
    import json  # noqa: PLC0415

    fail = json.dumps(
        {
            "status": "fail",
            "coverage_score": 80,
            "consistency_score": 50,
            "format_score": 90,
            "must_fix_issues": [
                {
                    "severity": "must_fix",
                    "category": "consistency",
                    "summary": "letter overstates X",
                    "detail": "the letter claims X but the resume puts X at a client engagement",
                },
            ],
            "nice_to_fix_issues": [],
            "summary": "fix the X attribution",
        },
    )
    return [fail, _good_qa_json()]


class _SequencedQAClient:
    """Returns scripted responses in order; raises if asked too many times."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        return self._responses.pop(0)


async def test_qa_must_fix_triggers_letter_rekick_with_feedback(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """QA returns must-fix → orchestrator re-kicks the letter with the
    feedback in the JD payload → re-polls → re-runs QA → passes →
    chain settles succeeded."""
    _, sleeper = recording_sleeper
    qa = _SequencedQAClient(_qa_fail_then_pass_responses())
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
    assert final.qa_status is QAStatus.PASS
    assert final.qa_attempts == 2
    # Letter was kicked twice: initial + retry. Second kick carries
    # the QA feedback appended to the JD text.
    assert len(scripted_letter_client.kick_requests) == 2
    retry_request = scripted_letter_client.kick_requests[1]
    assert retry_request.jd_text is not None
    assert "QA FEEDBACK FROM PREVIOUS ATTEMPT" in retry_request.jd_text
    assert "letter overstates X" in retry_request.jd_text


async def test_qa_must_fix_stops_after_max_attempts(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """When QA still flags must-fix after the retry, the chain stops
    at max_qa_attempts rather than looping forever. Row still settles
    succeeded -- the PDFs ship with the failing QA verdict attached."""
    _, sleeper = recording_sleeper
    # Both QA passes return the same fail; orchestrator should give
    # up after attempts == max_qa_attempts.
    qa = _SequencedQAClient(
        [_qa_fail_then_pass_responses()[0]] * 2,
    )
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_qa_attempts=2,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.qa_status is QAStatus.FAIL
    assert final.qa_attempts == 2
    # Letter kicked twice (initial + 1 retry) -- not 3 times.
    assert len(scripted_letter_client.kick_requests) == 2


async def test_qa_pass_first_try_does_not_retry(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """When QA accepts the first attempt, the orchestrator stops
    immediately -- qa_attempts is 1 and the letter is kicked once."""
    _, sleeper = recording_sleeper
    qa = _SequencedQAClient([_good_qa_json()])
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
    assert final.qa_attempts == 1
    assert len(scripted_letter_client.kick_requests) == 1


async def test_qa_concerns_with_only_nice_to_fix_does_not_retry(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A ``concerns`` verdict with ONLY nice-to-fix issues isn't worth
    re-burning LLM time on -- retry triggers on must-fix only."""
    import json  # noqa: PLC0415

    _, sleeper = recording_sleeper
    concerns_only_nice = json.dumps(
        {
            "status": "concerns",
            "coverage_score": 75,
            "consistency_score": 80,
            "format_score": 85,
            "must_fix_issues": [],
            "nice_to_fix_issues": [
                {
                    "severity": "nice_to_fix",
                    "category": "format",
                    "summary": "date format differs slightly",
                },
            ],
            "summary": "minor polish only",
        },
    )
    qa = _SequencedQAClient([concerns_only_nice])
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
    assert final.qa_attempts == 1
    assert final.qa_status is QAStatus.CONCERNS
    assert len(scripted_letter_client.kick_requests) == 1


async def test_qa_retry_letter_failure_keeps_first_pass_letter(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If the retry-kicked letter fails (sibling returned ``failed``),
    the chain must NOT mark the row failed. The first-pass letter
    rendered successfully and is the right artefact to ship -- the
    retry overstep (LLM producing a schema-violating letter) is a
    quality-improvement attempt that didn't pan out, not a reason to
    throw away the usable original.

    See `_private/mistake_log.md` (2026-05-14, fabricated-stats
    cascade) for why this matters: a false-positive must-fix from QA
    can poison the retry, and falling-over on retry failure means a
    minor QA misjudgement turns into a fully failed run."""
    _, sleeper = recording_sleeper
    # Letter succeeds on first kick, fails on second.
    letter_client = ScriptedLetterClient(
        poll_statuses=["succeeded", "failed"],
    )
    # Two fail verdicts: with the v1.27.0 iterate-to-cap behaviour a
    # failed retry no longer aborts the loop, so QA is asked again on
    # the (capped) second attempt. max_qa_attempts=2 bounds it.
    qa = _SequencedQAClient([_qa_fail_then_pass_responses()[0]] * 2)
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_qa_attempts=2,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    # Chain still succeeds; the row carries the FIRST-pass letter id.
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.letter_run_id == "ls_1"
    # Two kicks went out (first pass + retry) but only one returned ok.
    assert len(letter_client.kick_requests) == 2


def test_augment_payload_with_feedback_appends_must_fix_to_jd_text() -> None:
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _augment_payload_with_feedback,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://x", jd_text="JD body")
    assessment = QAAssessment(
        status=QAStatus.FAIL,
        coverage_score=80,
        consistency_score=50,
        format_score=90,
        must_fix_issues=[
            QAIssue(
                severity="must_fix",
                category="consistency",
                summary="letter overstates X",
                detail="x detail",
            ),
        ],
        nice_to_fix_issues=[],
        summary="fix X",
    )
    augmented = _augment_payload_with_feedback(payload, assessment)
    assert augmented.jd_url == "https://x"
    assert augmented.jd_text is not None
    assert "JD body" in augmented.jd_text
    assert "QA FEEDBACK FROM PREVIOUS ATTEMPT" in augmented.jd_text
    assert "letter overstates X" in augmented.jd_text
    assert "x detail" in augmented.jd_text


def test_merge_layout_into_assessment_appends_must_fix_and_drags_format_score() -> None:
    """A clean LLM assessment with 1 layout issue: must-fix gets the
    issue appended, format_score drops by 15 (per-issue penalty),
    status downgrades to fail because must_fix_count > 0."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import _merge_layout_into_assessment  # noqa: PLC0415

    base = QAAssessment(
        status=QAStatus.PASS,
        coverage_score=90,
        consistency_score=85,
        format_score=95,
        must_fix_issues=[],
        nice_to_fix_issues=[],
        summary="clean",
    )
    layout = [
        QAIssue(
            severity="must_fix",
            category="format",
            summary="resume: page 2 starts with 1 bullet stranded before 'Profile'",
        ),
    ]
    out = _merge_layout_into_assessment(base, layout)
    assert len(out.must_fix_issues) == 1
    assert out.format_score == 80  # 95 - 15
    assert out.status is QAStatus.FAIL  # must_fix forces fail


def test_merge_layout_into_assessment_clamps_format_at_zero() -> None:
    """Many layout issues + low starting format score shouldn't push
    format_score below 0."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import _merge_layout_into_assessment  # noqa: PLC0415

    base = QAAssessment(
        status=QAStatus.CONCERNS,
        coverage_score=80,
        consistency_score=80,
        format_score=10,
        must_fix_issues=[],
        nice_to_fix_issues=[],
        summary="weak format",
    )
    layout = [
        QAIssue(severity="must_fix", category="format", summary=f"issue {i}") for i in range(5)
    ]  # 5 * 15 = 75 penalty, but capped by current format_score=10
    out = _merge_layout_into_assessment(base, layout)
    assert out.format_score == 0
    assert out.status is QAStatus.FAIL


def test_recompute_status_pass_when_all_scores_above_80_and_no_issues() -> None:
    from jobai.tailor.models import QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import _recompute_status  # noqa: PLC0415

    assert _recompute_status(90, 85, 88, must_fix_count=0, nice_to_fix_count=0) is QAStatus.PASS


def test_recompute_status_concerns_on_nice_to_fix_or_mid_score() -> None:
    from jobai.tailor.models import QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import _recompute_status  # noqa: PLC0415

    # Mid score (60-79) without must-fix => concerns.
    assert _recompute_status(70, 85, 88, must_fix_count=0, nice_to_fix_count=0) is QAStatus.CONCERNS
    # All scores >= 80 but a nice-to-fix exists => concerns.
    assert _recompute_status(90, 85, 88, must_fix_count=0, nice_to_fix_count=2) is QAStatus.CONCERNS


def test_recompute_status_fail_when_score_below_60_or_must_fix_present() -> None:
    from jobai.tailor.models import QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import _recompute_status  # noqa: PLC0415

    assert _recompute_status(50, 85, 88, must_fix_count=0, nice_to_fix_count=0) is QAStatus.FAIL
    assert _recompute_status(90, 85, 88, must_fix_count=1, nice_to_fix_count=0) is QAStatus.FAIL


async def test_gather_layout_issues_collects_from_both_clients() -> None:
    """Walks both sibling stream_pdf endpoints; issues from BOTH PDFs
    get folded into the merged list with appropriate document_label."""
    import httpx  # noqa: PLC0415

    from jobai.tailor.orchestrator import _gather_layout_issues  # noqa: PLC0415

    # Bytes that look enough like a PDF for pypdf to parse but with
    # no extractable text -- so layout check returns no issues. The
    # path itself is what matters here; the issue-merge logic is
    # tested separately above.
    pdf = _make_minimal_pdf_bytes()
    resume_client = ScriptedResumeClient(
        stream_response=httpx.Response(200, content=pdf),
    )
    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=pdf),
    )
    issues = await _gather_layout_issues(
        resume_client=resume_client,
        letter_client=letter_client,
        resume_run_id="rs_1",
        letter_run_id="ls_1",
    )
    # Both PDFs fetched (verified via stream_calls) -- no issues for
    # a blank PDF, but the path was exercised.
    assert resume_client.stream_calls == ["rs_1"]
    assert letter_client.stream_calls == ["ls_1"]
    assert issues == []


async def test_qa_stage_merges_layout_issues_into_assessment(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: when ``_gather_layout_issues`` returns a non-empty
    list (simulated via monkeypatch), the QA stage folds those into
    the assessment, status drops to fail, and the auto-fix loop
    re-kicks the artefact the layout problem belongs to. v1.15.0:
    a layout issue whose summary says ``resume: page 2...`` targets
    the resume; the letter is left alone."""
    from jobai.tailor import orchestrator as orch  # noqa: PLC0415
    from jobai.tailor.models import QAIssue  # noqa: PLC0415

    layout_issue = QAIssue(
        severity="must_fix",
        category="format",
        summary="resume: page 2 starts with 1 bullet stranded before 'Profile'",
    )

    call_count = {"n": 0}

    async def fake_gather(**_kwargs: object) -> list[QAIssue]:
        call_count["n"] += 1
        # Return the layout issue on the FIRST pass only; the retry
        # produces a clean PDF (no layout issues) so the chain settles.
        if call_count["n"] == 1:
            return [layout_issue]
        return []

    monkeypatch.setattr(orch, "_gather_layout_issues", fake_gather)

    _, sleeper = recording_sleeper
    qa = _SequencedQAClient([_good_qa_json(), _good_qa_json()])
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
    # Auto-fix loop ran: 2 attempts. The resume was re-kicked because
    # the layout-issue summary mentions "resume" -- the letter was
    # left alone (its kick_requests count stays at 1, the initial).
    assert final.qa_attempts == 2
    assert len(scripted_resume_client.kick_requests) == 2
    assert len(scripted_letter_client.kick_requests) == 1
    # The retry kick's jd_text carries the layout-specific guidance.
    retry_jd = scripted_resume_client.kick_requests[1].jd_text or ""
    assert "LAYOUT:" in retry_jd


async def test_gather_layout_issues_continues_when_one_fetch_fails() -> None:
    """If one sibling's stream_pdf raises, the other still runs and
    its issues are returned -- a transient fetch error doesn't take
    the layout check down."""
    import httpx  # noqa: PLC0415

    from jobai.tailor.orchestrator import _gather_layout_issues  # noqa: PLC0415

    class _ExplodingResume:
        async def stream_pdf(self, _run_id: str) -> httpx.Response:
            msg = "network blew up"
            raise RuntimeError(msg)

    letter_client = ScriptedLetterClient(
        stream_response=httpx.Response(200, content=_make_minimal_pdf_bytes()),
    )
    issues = await _gather_layout_issues(
        resume_client=_ExplodingResume(),  # type: ignore[arg-type]
        letter_client=letter_client,
        resume_run_id="rs_1",
        letter_run_id="ls_1",
    )
    assert letter_client.stream_calls == ["ls_1"]
    assert issues == []  # blank PDF has no layout issues


def _make_minimal_pdf_bytes() -> bytes:
    """Build a tiny blank PDF in memory -- enough for pypdf to parse."""
    import io  # noqa: PLC0415

    from pypdf import PdfWriter  # noqa: PLC0415

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_augment_payload_includes_layout_guidance_when_layout_issue_present() -> None:
    """When a must-fix issue mentions a page (i.e. a layout problem),
    the feedback augmenter appends layout-specific guidance telling
    the LLM to shorten the output, not just fix wording."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _augment_payload_with_feedback,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://x", jd_text="JD body")
    assessment = QAAssessment(
        status=QAStatus.FAIL,
        coverage_score=80,
        consistency_score=80,
        format_score=70,
        must_fix_issues=[
            QAIssue(
                severity="must_fix",
                category="format",
                summary=(
                    "resume: page 2 starts with 1 bullet stranded before 'Professional Experience'"
                ),
                detail="trim 1-2 lines",
            ),
        ],
        nice_to_fix_issues=[],
        summary="layout",
    )
    augmented = _augment_payload_with_feedback(payload, assessment)
    assert augmented.jd_text is not None
    assert "LAYOUT:" in augmented.jd_text


def test_augment_payload_with_feedback_inlines_url_when_no_jd_text() -> None:
    """One-off URL path (no jd_text) -- the feedback block surfaces
    the URL inline so the LLM has SOME context for the retry."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _augment_payload_with_feedback,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://example.com/jd", jd_text=None)
    assessment = QAAssessment(
        status=QAStatus.FAIL,
        coverage_score=70,
        consistency_score=60,
        format_score=80,
        must_fix_issues=[
            QAIssue(
                severity="must_fix",
                category="content",
                summary="missing tone match",
            ),
        ],
        nice_to_fix_issues=[],
        summary="tone",
    )
    augmented = _augment_payload_with_feedback(payload, assessment)
    assert augmented.jd_text is not None
    assert "https://example.com/jd" in augmented.jd_text
    assert "missing tone match" in augmented.jd_text


def test_augment_payload_with_feedback_inlines_verified_context_when_provided() -> None:
    """When qa_context is supplied, the augmenter pastes the verified
    facts block into the JD text so the retry LLM sees the ground
    truth literally in the prompt -- not relying on the LLM to remember
    to re-pull from the context pool. Closes the v1.15.0 hole where
    same-JD retries non-deterministically failed to converge on
    correct stats (run 23: kept the wrong number; run 24: corrected
    it -- pure LLM nondeterminism)."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _augment_payload_with_feedback,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://x", jd_text="JD body")
    assessment = QAAssessment(
        status=QAStatus.FAIL,
        coverage_score=85,
        consistency_score=55,
        format_score=88,
        must_fix_issues=[
            QAIssue(
                severity="must_fix",
                category="consistency",
                summary="Resume cites ~38k LOC but verified context says ~15,700",
            ),
        ],
        nice_to_fix_issues=[],
        summary="LOC mismatch",
    )
    verified = (
        "## VERIFIED jobai project stats\n"
        "- Python LOC: ~15,700 across jobai/.\n"
        "- 1126 backend tests at 100% line + branch coverage."
    )
    augmented = _augment_payload_with_feedback(
        payload,
        assessment,
        qa_context=verified,
    )
    assert augmented.jd_text is not None
    assert "VERIFIED FACTS" in augmented.jd_text
    assert "Python LOC: ~15,700" in augmented.jd_text
    assert "1126 backend tests" in augmented.jd_text
    # Stronger guidance now insists on verbatim numbers + no estimating.
    assert "VERBATIM" in augmented.jd_text


def test_augment_payload_with_feedback_no_verified_block_when_no_context() -> None:
    """The VERIFIED FACTS section is only emitted when qa_context is
    supplied. Without it the augmenter behaves the same as v1.14.0 --
    just the must-fix list + generic guidance."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import (  # noqa: PLC0415
        _augment_payload_with_feedback,
        _JDPayload,
    )

    payload = _JDPayload(jd_url="https://x", jd_text="JD body")
    assessment = QAAssessment(
        status=QAStatus.FAIL,
        coverage_score=80,
        consistency_score=60,
        format_score=85,
        must_fix_issues=[
            QAIssue(severity="must_fix", category="content", summary="weak ending"),
        ],
        nice_to_fix_issues=[],
        summary="...",
    )
    augmented = _augment_payload_with_feedback(payload, assessment)
    assert augmented.jd_text is not None
    assert "VERIFIED FACTS" not in augmented.jd_text


async def test_run_chain_forwards_qa_context_into_retry_payload(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """End-to-end: when fetch_qa_context returns ground truth, the
    retry kick's jd_text must carry the VERIFIED FACTS block. This is
    the wiring that gives convergence its strongest signal."""
    _, sleeper = recording_sleeper

    async def _fetch() -> str | None:
        return "## VERIFIED jobai project stats\n- Python LOC: ~15,700"

    # Sequence: fail on first QA, pass on second; resume retry fires.
    qa = _SequencedQAClient(_qa_resume_fail_then_pass())
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        fetch_qa_context=_fetch,
    )

    # Resume was re-kicked; the retry payload must carry the verified
    # facts block inline so the LLM sees ~15,700 in the prompt itself.
    assert len(scripted_resume_client.kick_requests) == 2
    retry_jd = scripted_resume_client.kick_requests[1].jd_text or ""
    assert "VERIFIED FACTS" in retry_jd
    assert "Python LOC: ~15,700" in retry_jd


# ---------------------------------------------------------------------------
# Pre-tailor context refresh hook
# ---------------------------------------------------------------------------


async def test_refresh_context_hook_fires_before_resume_kick(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """The orchestrator must call the refresh hook BEFORE kicking
    resumeai so the LLM sees today's project-scan numbers and can't
    cite stale stats."""
    _, sleeper = recording_sleeper
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    call_order: list[str] = []

    async def _refresh() -> None:
        call_order.append("refresh")

    # Wrap the scripted resume client's kick so we can record the
    # order without mutating the shared fake.
    from jobai.tailor.models import ResumeaiTailorRequest  # noqa: PLC0415

    original_kick = scripted_resume_client.kick

    async def _kick_recording(req: ResumeaiTailorRequest) -> str:
        call_order.append("resume_kick")
        return await original_kick(req)

    scripted_resume_client.kick = _kick_recording  # type: ignore[assignment,method-assign]

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        refresh_context_scans=_refresh,
    )

    assert call_order == ["refresh", "resume_kick"]


async def test_refresh_context_hook_failure_does_not_block_chain(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If the refresh helper raises (resumeai unreachable, network blip),
    the chain must still complete -- a context-pool hiccup must never
    block the user's tailor."""
    _, sleeper = recording_sleeper
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    async def _refresh() -> None:
        msg = "resumeai unreachable"
        raise RuntimeError(msg)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        refresh_context_scans=_refresh,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    # Resume and letter still kicked and finished -- the refresh failure
    # didn't poison the rest of the chain.
    assert scripted_resume_client.kick_requests
    assert scripted_letter_client.kick_requests


async def test_refresh_context_hook_omitted_skips_cleanly(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Tests and host-mode dev pass ``refresh_context_scans=None``; the
    orchestrator must skip the hook silently and proceed straight to
    resume kick. (Default value should match this behaviour.)"""
    _, sleeper = recording_sleeper
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        # refresh_context_scans deliberately omitted -- exercises the
        # default-None path.
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# QA-context fetch hook + retry-failure resilience
# ---------------------------------------------------------------------------


async def test_run_chain_forwards_qa_context_into_assess_call(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """The fetch_qa_context closure result must flow into the QA
    client's user prompt so the model has ground truth to validate
    numeric claims against."""
    _, sleeper = recording_sleeper

    async def _fetch() -> str | None:
        return "## VERIFIED jobai stats\n- 1126 tests, 100% coverage"

    qa = _SequencedQAClient([_good_qa_json()])
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        fetch_qa_context=_fetch,
    )

    assert len(qa.calls) == 1
    user_prompt = qa.calls[0]["user"]
    assert isinstance(user_prompt, str)
    assert "USER CONTEXT" in user_prompt
    assert "1126 tests, 100% coverage" in user_prompt


async def test_run_chain_qa_context_fetch_failure_is_non_fatal(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If the context-fetch closure raises (resumeai down, network
    glitch), QA still runs against the artefacts alone -- it just
    doesn't get the ground-truth block. Chain must still succeed."""
    _, sleeper = recording_sleeper

    async def _boom() -> str | None:
        msg = "context pool unreachable"
        raise RuntimeError(msg)

    qa = _SequencedQAClient([_good_qa_json()])
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        fetch_qa_context=_boom,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    # QA still ran -- with no USER CONTEXT block in the prompt.
    assert len(qa.calls) == 1
    user_prompt = qa.calls[0]["user"]
    assert isinstance(user_prompt, str)
    assert "USER CONTEXT" not in user_prompt


async def test_run_chain_qa_retry_letter_kick_exception_keeps_first_pass(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If letter_client.kick() raises during the retry (sibling 5xx,
    network blip), the chain falls back to the first-pass letter
    rather than failing the whole run.

    See `_private/mistake_log.md` (2026-05-14): a false-positive
    must-fix from QA must not be able to cascade into a chain failure
    by way of the retry-kick crashing."""
    from jobai.tailor.models import CoverletteraiTailorRequest  # noqa: PLC0415

    _, sleeper = recording_sleeper

    # First kick succeeds and the poll returns succeeded; second kick
    # raises. The orchestrator must catch and fall back.
    letter_client = ScriptedLetterClient()
    original_kick = letter_client.kick
    call_count = {"n": 0}

    async def _kick_then_boom(req: CoverletteraiTailorRequest) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return await original_kick(req)
        msg = "coverletterai 503"
        raise RuntimeError(msg)

    letter_client.kick = _kick_then_boom  # type: ignore[assignment,method-assign]

    # Two fail verdicts: with the v1.27.0 iterate-to-cap behaviour a
    # failed retry no longer aborts the loop, so QA is asked again on
    # the (capped) second attempt. max_qa_attempts=2 bounds it.
    qa = _SequencedQAClient([_qa_fail_then_pass_responses()[0]] * 2)
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_qa_attempts=2,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    # First-pass letter id preserved (the retry kick blew up before
    # producing a new id).
    assert final.letter_run_id == "ls_1"


async def test_run_chain_qa_retry_poll_cap_keeps_first_pass(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If the retry letter's poll exceeds max_polls (sibling stuck
    forever), the chain falls back to the first-pass letter rather
    than marking the row failed."""
    _, sleeper = recording_sleeper

    # First poll returns 'succeeded'; subsequent polls never resolve
    # (return 'tailoring' forever). Run with max_polls=1 so the retry
    # hits the cap on its very first re-poll.
    letter_client = ScriptedLetterClient(poll_statuses=["succeeded", "tailoring"])
    # Two fail verdicts: with the v1.27.0 iterate-to-cap behaviour a
    # failed retry no longer aborts the loop, so QA is asked again on
    # the (capped) second attempt. max_qa_attempts=2 bounds it.
    qa = _SequencedQAClient([_qa_fail_then_pass_responses()[0]] * 2)
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_polls=1,
        max_qa_attempts=2,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    # Chain still settles SUCCEEDED with the first-pass letter.
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.letter_run_id == "ls_1"


# ---------------------------------------------------------------------------
# Resume-side retry (v1.15.0 — run 22 fix)
# ---------------------------------------------------------------------------


def _qa_resume_fail_then_pass() -> list[str]:
    """First QA verdict flags a RESUME issue; second verdict passes.

    Drives the resume-retry path: QA must-fix on the resume -> orchestrator
    re-kicks the resume -> re-poll -> second QA -> pass."""
    import json  # noqa: PLC0415

    fail = json.dumps(
        {
            "status": "fail",
            "coverage_score": 86,
            "consistency_score": 55,
            "format_score": 90,
            "must_fix_issues": [
                {
                    "severity": "must_fix",
                    "category": "consistency",
                    "summary": "Resume cites ~38k LOC but verified context says ~15,700",
                    "detail": "The resume bullet under personal_projects.jobai overstates LOC",
                },
            ],
            "nice_to_fix_issues": [],
            "summary": "fix the LOC mismatch in the resume",
        },
    )
    return [fail, _good_qa_json()]


async def test_qa_must_fix_on_resume_triggers_resume_rekick(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """When a must-fix issue's summary mentions ``resume``, the
    orchestrator re-kicks resumeai (not coverletterai) with the QA
    feedback appended. Fixes the v1.10.0 limitation that auto-fix
    couldn't repair resume-side hallucinations -- see run 22, where
    the resume claimed ~38k LOC but the verified context said
    ~15,700, and the letter-only retry could never close the gap."""
    _, sleeper = recording_sleeper
    qa = _SequencedQAClient(_qa_resume_fail_then_pass())
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
    assert final.qa_status is QAStatus.PASS
    assert final.qa_attempts == 2
    # The resume was kicked twice (initial + retry); the letter only
    # once (the must-fix didn't mention the letter so no retry there).
    assert len(scripted_resume_client.kick_requests) == 2
    assert len(scripted_letter_client.kick_requests) == 1
    # Retry kick's jd_text carries the QA feedback so the LLM has a
    # concrete diff to apply.
    retry_jd = scripted_resume_client.kick_requests[1].jd_text or ""
    assert "Resume cites ~38k LOC" in retry_jd


async def test_qa_resume_retry_kick_exception_keeps_first_pass(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If resumeai.kick() raises during the retry (sibling 5xx,
    network blip), the chain falls back to the first-pass resume
    rather than failing -- same shape as the letter-side resilience."""
    from jobai.tailor.models import ResumeaiTailorRequest  # noqa: PLC0415

    _, sleeper = recording_sleeper

    resume_client = ScriptedResumeClient()
    original_kick = resume_client.kick
    call_count = {"n": 0}

    async def _kick_then_boom(req: ResumeaiTailorRequest) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return await original_kick(req)
        msg = "resumeai 503"
        raise RuntimeError(msg)

    resume_client.kick = _kick_then_boom  # type: ignore[assignment,method-assign]

    qa = _SequencedQAClient([_qa_resume_fail_then_pass()[0]] * 3)
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_qa_attempts=3,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    # First-pass resume id preserved (the retry kick raised before
    # producing a new id).
    assert final.resume_run_id == "rs_1"


async def test_qa_resume_retry_poll_cap_keeps_first_pass(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If the retry resume's poll hits the cap, the chain falls back
    to the first-pass resume rather than failing."""
    _, sleeper = recording_sleeper

    resume_client = ScriptedResumeClient(
        poll_statuses=["succeeded", "tailoring"],
    )
    qa = _SequencedQAClient([_qa_resume_fail_then_pass()[0]] * 3)
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_polls=1,
        max_qa_attempts=3,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    # First-pass resume id preserved -- the cap-hit short-circuit
    # returned None, so the orchestrator kept the original.
    assert final.resume_run_id == "rs_1"


async def test_qa_resume_retry_returns_failed_keeps_first_pass(
    tailor_db_path: Path,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """If resumeai's retry render terminates with ``failed`` status,
    keep the first-pass resume rather than failing the chain."""
    _, sleeper = recording_sleeper

    # First poll succeeds (initial render); second poll returns failed
    # (retry render). Orchestrator falls back to first-pass id.
    resume_client = ScriptedResumeClient(poll_statuses=["succeeded", "failed"])
    qa = _SequencedQAClient([_qa_resume_fail_then_pass()[0]] * 3)
    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        qa_client=qa,
        max_qa_attempts=3,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert final.resume_run_id == "rs_1"


async def test_retry_targets_picks_resume_and_letter_independently() -> None:
    """The keyword scan over must-fix issue text decides which
    artefact(s) to retry. Resume-only, letter-only, both, and
    none-mentioned (default to letter) all need to behave."""
    from jobai.tailor.models import QAAssessment, QAIssue, QAStatus  # noqa: PLC0415
    from jobai.tailor.orchestrator import _retry_targets  # noqa: PLC0415

    def _asses(*summaries: str) -> QAAssessment:
        return QAAssessment(
            status=QAStatus.FAIL,
            coverage_score=80,
            consistency_score=50,
            format_score=90,
            must_fix_issues=[
                QAIssue(severity="must_fix", category="consistency", summary=s) for s in summaries
            ],
            nice_to_fix_issues=[],
            summary="...",
        )

    assert _retry_targets(_asses("Resume cites 38k LOC")) == {"resume"}
    assert _retry_targets(_asses("Cover letter claims X")) == {"letter"}
    assert _retry_targets(
        _asses("Resume cites 38k LOC", "Letter claims something else"),
    ) == {"resume", "letter"}
    # Nothing mentions either artefact -> default to letter (v1.10.0
    # behaviour for must-fix issues that don't name an artefact).
    assert _retry_targets(_asses("Tone too casual")) == {"letter"}


async def test_filename_cache_failure_leaves_row_with_null_filenames(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the filename-build helper raises at terminal SUCCESS
    (sibling outage, bad payload), the chain still settles SUCCEEDED
    with NULL filename columns -- the PDF route falls back to live
    computation in that case."""
    from jobai.tailor import orchestrator as orch  # noqa: PLC0415

    async def _boom(**_kwargs: object) -> tuple[str, str]:
        msg = "build failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(orch, "build_pdf_filenames", _boom)

    _, sleeper = recording_sleeper
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
    assert final.resume_filename is None
    assert final.letter_filename is None


# ---------------------------------------------------------------------------
# on-demand JD resolution (Seek / Cloudflare-SPA boards)
# ---------------------------------------------------------------------------


async def test_resolve_jd_text_fills_thin_catalogue_payload(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Job 1 only has a teaser, so the resolver fires and both siblings
    receive the fetched JD as jd_text instead of the un-fetchable URL."""
    _, sleeper = recording_sleeper
    seen: list[str] = []

    async def _resolver(url: str) -> str | None:
        seen.append(url)
        return "FULL SEEK JOB DESCRIPTION " * 20

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        resolve_jd_text=_resolver,
    )

    assert seen == ["https://example.com/jd-1"]
    resume_req = scripted_resume_client.kick_requests[0]
    letter_req = scripted_letter_client.kick_requests[0]
    assert resume_req.jd_text is not None
    assert resume_req.jd_text.startswith("FULL SEEK JOB DESCRIPTION")
    assert resume_req.jd_url is None
    assert letter_req.jd_text == resume_req.jd_text


async def test_resolve_jd_text_none_keeps_url_payload(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Resolver declines (non-Seek / fetch miss) -> unchanged URL path."""
    _, sleeper = recording_sleeper

    async def _resolver(_url: str) -> str | None:
        return None

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        resolve_jd_text=_resolver,
    )

    assert scripted_resume_client.kick_requests[0].jd_url == "https://example.com/jd-1"
    assert scripted_resume_client.kick_requests[0].jd_text is None


async def test_resolve_jd_text_exception_is_swallowed(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """A resolver blow-up must not fail the chain; fall back to the URL."""
    _, sleeper = recording_sleeper

    async def _resolver(_url: str) -> str | None:
        msg = "stealth fetcher exploded"
        raise RuntimeError(msg)

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        resolve_jd_text=_resolver,
    )

    with connect(tailor_db_path) as conn:
        final = get_tailor_run(conn, record.id)
    assert final is not None
    assert final.status is TailorRunStatus.SUCCEEDED
    assert scripted_resume_client.kick_requests[0].jd_url == "https://example.com/jd-1"
    assert scripted_resume_client.kick_requests[0].jd_text is None


async def test_resolve_jd_text_skipped_when_payload_already_has_text(
    tailor_db_path: Path,
    scripted_resume_client: ScriptedResumeClient,
    scripted_letter_client: ScriptedLetterClient,
    recording_sleeper: tuple[list[float], Sleeper],
) -> None:
    """Catalogue rows with a real description never pay the prefetch."""
    _, sleeper = recording_sleeper
    bare = sqlite3.connect(tailor_db_path)
    try:
        bare.execute(
            "UPDATE jobs SET description_text = ? WHERE id = 1",
            ("Substantial real description body. " * 20,),
        )
        bare.commit()
    finally:
        bare.close()

    called = False

    async def _resolver(_url: str) -> str | None:
        nonlocal called
        called = True
        return "should not be used"

    with connect(tailor_db_path) as conn:
        record = create_tailor_run(conn, job_id=1)

    await run_chain(
        record.id,
        db_path=tailor_db_path,
        resume_client=scripted_resume_client,
        letter_client=scripted_letter_client,
        sleeper=sleeper,
        resolve_jd_text=_resolver,
    )

    assert called is False
    assert scripted_resume_client.kick_requests[0].jd_text is not None
    assert "Substantial real description body." in (
        scripted_resume_client.kick_requests[0].jd_text or ""
    )
