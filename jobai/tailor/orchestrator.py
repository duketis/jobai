"""The tailor chain as a single coroutine.

``run_chain`` kicks resumeai, polls until it terminates, kicks
coverletterai with the resume run id, polls until it terminates, and
records every transition through the repository.

The function takes every collaborator (clients, repo, sleeper) as
arguments so the test suite can drive it deterministically — no
``time.sleep`` actually fires in tests, no httpx call hits the wire.
The production wiring is in :mod:`jobai.tailor.worker`.

Statuses returned by resumeai / coverletterai are stringly typed:
``loading_context``, ``tailoring``, ``verifying``, ``succeeded``,
``failed``. We treat anything other than ``succeeded`` / ``failed`` as
in-flight.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from selectolax.parser import HTMLParser

from jobai.db.connection import connect
from jobai.tailor.client import CoverletteraiClient, ResumeaiClient
from jobai.tailor.filenames import build_pdf_filenames
from jobai.tailor.layout_check import check_pdf_layout
from jobai.tailor.models import (
    CoverletteraiTailorRequest,
    QAAssessment,
    QAIssue,
    QAStatus,
    ResumeaiTailorRequest,
    SiblingRunSnapshot,
    TailorRunStatus,
)
from jobai.tailor.qa import QAClient, assess
from jobai.tailor.repository import get_tailor_run, update_status
from jobai.tailor.snapshot import write_snapshot

_log = logging.getLogger(__name__)

#: Sibling-side statuses that mean "this run is finished, look at the artefact".
_TERMINAL_SUCCESS: frozenset[str] = frozenset({"succeeded"})
#: Sibling-side statuses that mean "this run won't progress further".
_TERMINAL_FAILURE: frozenset[str] = frozenset({"failed"})

#: How long to wait between polls of a sibling run, in seconds. Matched to the
#: handoff guidance (10s) — fast enough to surface terminal states promptly,
#: slow enough that we don't hammer the sibling APIs during a long render.
DEFAULT_POLL_INTERVAL_S: float = 10.0
#: Hard ceiling on poll count per sibling. At 10s/poll that's 30 minutes,
#: well beyond the ~3-minute upper bound for a normal run.
DEFAULT_MAX_POLLS: int = 180

#: Cap on total QA passes per chain. The first pass is the initial
#: grade; each subsequent pass follows a retry of the artefact(s) that
#: had must-fix issues. Three attempts is the sweet spot: the LLM is
#: non-deterministic enough that the same JD can converge on attempt 1
#: OR attempt 3, but past 3 the marginal signal is noise (the model
#: that hasn't fixed a fabricated stat after seeing the verified
#: context twice rarely fixes it on a fourth try).
DEFAULT_MAX_QA_ATTEMPTS: int = 3

# Sleeper signature: ``await sleeper(seconds)``. Defaults to ``asyncio.sleep``
# in production; tests supply a recorder that records the requested delay and
# returns immediately.
Sleeper = Callable[[float], Awaitable[None]]


class TailorChainError(RuntimeError):
    """Raised when the chain cannot complete and the run is marked failed."""


async def run_chain(
    tailor_run_id: int,
    *,
    db_path: Path,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    sleeper: Sleeper,
    qa_client: QAClient | None = None,
    qa_model: str | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    max_polls: int = DEFAULT_MAX_POLLS,
    max_qa_attempts: int = DEFAULT_MAX_QA_ATTEMPTS,
    refresh_context_scans: Callable[[], Awaitable[None]] | None = None,
    fetch_qa_context: Callable[[], Awaitable[str | None]] | None = None,
    snapshot_output_dir: Path | None = None,
) -> None:
    """Drive one tailor chain to terminal state.

    Opens its own sqlite connection per stage so a long-running chain
    doesn't pin a connection. Each repository call commits before
    returning, so the UI sees in-flight transitions in real time.

    ``TailorChainError`` is caught here, recorded as ``failed`` on the
    row, and never re-raised — the worker's task wrapper would otherwise
    log an unhandled exception that's not actionable.

    ``qa_client`` is optional. When supplied, the chain runs an extra
    QA stage (``qa_running``) after both PDFs render; the assessment
    lands on the row's ``qa_status`` + ``qa_assessment_json`` fields.
    When ``None``, the chain terminates at ``succeeded`` immediately
    after the letter -- tests that don't care about QA can omit it.
    """
    try:
        await _run_chain_inner(
            tailor_run_id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            sleeper=sleeper,
            qa_client=qa_client,
            qa_model=qa_model,
            poll_interval_s=poll_interval_s,
            max_polls=max_polls,
            max_qa_attempts=max_qa_attempts,
            refresh_context_scans=refresh_context_scans,
            fetch_qa_context=fetch_qa_context,
            snapshot_output_dir=snapshot_output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - top-level boundary, see docstring
        _log.exception("tailor_chain_failed", extra={"tailor_run_id": tailor_run_id})
        with connect(db_path) as conn:
            update_status(
                conn,
                tailor_run_id,
                status=TailorRunStatus.FAILED,
                error=str(exc),
            )


async def _run_chain_inner(  # noqa: PLR0912 - chain state machine; splitting harms readability more than it helps
    tailor_run_id: int,
    *,
    db_path: Path,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    sleeper: Sleeper,
    qa_client: QAClient | None,
    qa_model: str | None,
    poll_interval_s: float,
    max_polls: int,
    max_qa_attempts: int,
    refresh_context_scans: Callable[[], Awaitable[None]] | None,
    fetch_qa_context: Callable[[], Awaitable[str | None]] | None,
    snapshot_output_dir: Path | None,
) -> None:
    payload = _load_jd_payload(db_path, tailor_run_id)

    # ---- context refresh -----------------------------------------------
    # Re-scan every local-project context entry so the siblings tailor
    # against the user's CURRENT repo state -- fresh test count, fresh
    # coverage %, fresh recent-commit list. Without this, a tailored
    # resume can keep citing yesterday's "705 tests at 89% coverage"
    # numbers long after the repo has moved on. Failures are swallowed;
    # the chain proceeds even when the context pool is unreachable.
    if refresh_context_scans is not None:
        try:
            await refresh_context_scans()
        except Exception:  # noqa: BLE001 - never let context-refresh block tailoring
            _log.warning(
                "tailor_context_refresh_failed",
                extra={"tailor_run_id": tailor_run_id},
                exc_info=True,
            )

    # ---- resume ---------------------------------------------------------
    with connect(db_path) as conn:
        update_status(conn, tailor_run_id, status=TailorRunStatus.RESUME_RUNNING)

    resume_run_id = await resume_client.kick(_build_resume_request(payload))
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.RESUME_RUNNING,
            resume_run_id=resume_run_id,
        )

    resume_snapshot = await _poll_until_terminal(
        kind="resume",
        run_id=resume_run_id,
        poll=resume_client.poll,
        sleeper=sleeper,
        poll_interval_s=poll_interval_s,
        max_polls=max_polls,
        db_path=db_path,
        tailor_run_id=tailor_run_id,
        status_field="resume_status",
    )
    if resume_snapshot.status not in _TERMINAL_SUCCESS:
        msg = f"resumeai run {resume_run_id} ended in status {resume_snapshot.status!r}"
        raise TailorChainError(msg)

    # ---- cover letter ---------------------------------------------------
    with connect(db_path) as conn:
        update_status(conn, tailor_run_id, status=TailorRunStatus.LETTER_RUNNING)

    letter_run_id = await letter_client.kick(
        _build_letter_request(payload, resume_run_id=resume_run_id),
    )
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.LETTER_RUNNING,
            letter_run_id=letter_run_id,
        )

    letter_snapshot = await _poll_until_terminal(
        kind="letter",
        run_id=letter_run_id,
        poll=letter_client.poll,
        sleeper=sleeper,
        poll_interval_s=poll_interval_s,
        max_polls=max_polls,
        db_path=db_path,
        tailor_run_id=tailor_run_id,
        status_field="letter_status",
    )
    if letter_snapshot.status not in _TERMINAL_SUCCESS:
        msg = f"coverletterai run {letter_run_id} ended in status {letter_snapshot.status!r}"
        raise TailorChainError(msg)

    # ---- QA context summary --------------------------------------------
    # Pull the verified + project-scan context once per chain so every
    # QA pass (initial + auto-fix retries) sees the same ground truth.
    # Failures here are not fatal -- QA without ground truth falls back
    # to within-artefact consistency checks.
    qa_context: str | None = None
    if fetch_qa_context is not None:
        try:
            qa_context = await fetch_qa_context()
        except Exception:  # noqa: BLE001 - never let context-fetch break QA
            _log.warning(
                "tailor_qa_context_fetch_failed",
                extra={"tailor_run_id": tailor_run_id},
                exc_info=True,
            )

    # ---- QA + auto-fix loop --------------------------------------------
    #
    # First pass grades the initial resume + letter. If the verdict
    # flags must-fix issues AND we have attempts left, re-kick the
    # cover letter with the QA feedback appended to the JD payload --
    # the LLM sees "here's the JD, and here's what went wrong last
    # time, do better" and produces a corrected letter. Then re-run
    # QA. Loops until QA accepts or the attempt cap is hit. The
    # current ``letter_run_id`` always points at the LATEST attempt
    # so PDF downloads + the UI reflect the final artefact.
    if qa_client is not None:
        attempts = 0
        # The natural-exit branch (condition false on a recheck) is
        # unreachable: every iteration breaks before incrementing past
        # the cap. Pragma'd so the False branch doesn't drag coverage
        # while the True branch is exercised normally.
        while attempts < max_qa_attempts:  # pragma: no branch
            attempts += 1
            assessment = await _run_qa_stage(
                tailor_run_id=tailor_run_id,
                db_path=db_path,
                resume_client=resume_client,
                letter_client=letter_client,
                resume_run_id=resume_run_id,
                letter_run_id=letter_run_id,
                qa_client=qa_client,
                qa_model=qa_model,
                attempts=attempts,
                qa_context=qa_context,
            )
            # Stop if QA passed, or if we have no must-fix work to drive
            # a retry against (concerns + only nice-to-fix is acceptable),
            # or if we've burned our retry budget.
            if not assessment.must_fix_issues or attempts >= max_qa_attempts:
                break

            # Re-kick whichever artefact(s) the QA must-fix issues
            # actually reference. v1.10.0 shipped letter-only retries
            # on the assumption that resume content was "verifiable
            # career data" the LLM wouldn't hallucinate over. v1.15.0
            # learned otherwise: the resume can claim a wrong LOC /
            # test count / coverage stat (run 22 saw "~38k Python
            # LOC" while the verified context said "~15,700"), and
            # the letter-only retry can't fix a resume-side problem.
            targets = _retry_targets(assessment)
            if "resume" in targets:
                new_resume_run_id = await _rekick_resume_with_feedback(
                    tailor_run_id=tailor_run_id,
                    db_path=db_path,
                    resume_client=resume_client,
                    payload=payload,
                    assessment=assessment,
                    sleeper=sleeper,
                    poll_interval_s=poll_interval_s,
                    max_polls=max_polls,
                    qa_context=qa_context,
                )
                if new_resume_run_id is not None:
                    resume_run_id = new_resume_run_id
                # If the resume retry failed we keep the original and
                # still try the letter retry; both artefacts ship.
            if "letter" in targets:
                # Re-kick the cover letter with the QA feedback so the
                # LLM has a concrete list of what to fix. The retry
                # uses the LATEST resume_run_id so a freshly-corrected
                # resume gets cited consistently in the letter body.
                new_letter_run_id = await _rekick_letter_with_feedback(
                    tailor_run_id=tailor_run_id,
                    db_path=db_path,
                    letter_client=letter_client,
                    payload=payload,
                    resume_run_id=resume_run_id,
                    assessment=assessment,
                    sleeper=sleeper,
                    poll_interval_s=poll_interval_s,
                    max_polls=max_polls,
                    qa_context=qa_context,
                )
                if new_letter_run_id is None:
                    _log.warning(
                        "tailor_qa_retry_letter_failed",
                        extra={
                            "tailor_run_id": tailor_run_id,
                            "previous_letter_run_id": letter_run_id,
                        },
                    )
                    break
                letter_run_id = new_letter_run_id

    await _settle_terminal_success(
        tailor_run_id=tailor_run_id,
        db_path=db_path,
        resume_client=resume_client,
        letter_client=letter_client,
        snapshot_output_dir=snapshot_output_dir,
    )


async def _settle_terminal_success(
    *,
    tailor_run_id: int,
    db_path: Path,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    snapshot_output_dir: Path | None,
) -> None:
    """Cache filenames, mark the row succeeded, and snapshot to disk.

    Three things happen at the chain's tail:

    1. Compute the descriptive PDF filenames once (via a single
       batched sibling fetch) so the list-runs endpoint can return
       them without an N+1 call per row.
    2. Persist the filenames + SUCCEEDED status on the row.
    3. Snapshot the per-job folder to disk so the user can grab the
       PDFs from a filesystem location (no UI clicks) and so the
       sibling ``interviewai`` can read JD + resume + letter + QA off
       the same folder layout. Skipped when ``snapshot_output_dir`` is
       ``None`` (tests + host-mode dev that doesn't need the folder).

    Every step is fail-soft: a sibling outage at this point only
    degrades the post-success conveniences, never the row's terminal
    state. The PDFs are still streamable through the HTTP routes.
    """
    resume_filename: str | None = None
    letter_filename: str | None = None
    try:
        with connect(db_path) as conn:
            resume_filename, letter_filename = await build_pdf_filenames(
                conn=conn,
                resume_client=resume_client,
                tailor_run_id=tailor_run_id,
            )
    except Exception:  # noqa: BLE001 - filename caching is best-effort
        _log.warning(
            "tailor_filename_cache_failed",
            extra={"tailor_run_id": tailor_run_id},
            exc_info=True,
        )

    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.SUCCEEDED,
            resume_filename=resume_filename,
            letter_filename=letter_filename,
        )

    if snapshot_output_dir is not None:
        await write_snapshot(
            output_dir=snapshot_output_dir,
            db_path=db_path,
            tailor_run_id=tailor_run_id,
            resume_client=resume_client,
            letter_client=letter_client,
        )


async def _run_qa_stage(
    *,
    tailor_run_id: int,
    db_path: Path,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    resume_run_id: str,
    letter_run_id: str,
    qa_client: QAClient,
    qa_model: str | None,
    attempts: int,
    qa_context: str | None = None,
) -> QAAssessment:
    """Run one QA pass against the current resume + letter artefacts.

    Returns the assessment so the orchestrator can decide whether to
    retry (must-fix issues + attempts left) or settle. The chain-
    level exception handler in :func:`run_chain` catches QA failures
    so the row still ships with whatever PDFs were generated.
    """
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RUNNING,
            qa_status=QAStatus.RUNNING,
            qa_attempts=attempts,
        )

    resume_record = await resume_client.get_run(resume_run_id)
    letter_record = await letter_client.get_run(letter_run_id)
    jd = resume_record.get("requirements")
    resume_tailored = resume_record.get("tailored")
    letter_tailored = letter_record.get("tailored")

    assessment = await assess(
        jd=jd if isinstance(jd, dict) else None,
        resume_tailored=resume_tailored if isinstance(resume_tailored, dict) else None,
        letter_tailored=letter_tailored if isinstance(letter_tailored, dict) else None,
        client=qa_client,
        model=qa_model,
        user_context=qa_context,
    )

    # Layout check: pull the rendered PDFs and run heuristic checks for
    # orphan bullets / split section headers. The LLM grades content
    # consistency but is blind to pagination (it reads JSON, not the
    # PDF). Layout issues become must-fix in the assessment so the
    # auto-fix loop triggers a retry with concrete trim instructions.
    layout_issues = await _gather_layout_issues(
        resume_client=resume_client,
        letter_client=letter_client,
        resume_run_id=resume_run_id,
        letter_run_id=letter_run_id,
    )
    if layout_issues:
        assessment = _merge_layout_into_assessment(assessment, layout_issues)

    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RUNNING,
            qa_status=assessment.status,
            qa_assessment=assessment,
            qa_attempts=attempts,
        )
    return assessment


async def _gather_layout_issues(
    *,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    resume_run_id: str,
    letter_run_id: str,
) -> list[QAIssue]:
    """Fetch both rendered PDFs and run the heuristic layout checks.

    Failures here (sibling 404, malformed PDF) just return an empty
    list -- layout issues are an addition, not a gate, and a hiccup
    in the layout path must not break the QA stage.
    """
    issues: list[QAIssue] = []
    for kind, client, run_id in (
        ("resume", resume_client, resume_run_id),
        ("cover letter", letter_client, letter_run_id),
    ):
        try:
            response = await client.stream_pdf(run_id)
            try:
                pdf_bytes = await response.aread()
            finally:
                await response.aclose()
        except Exception:  # noqa: BLE001 - report-and-continue per artefact
            _log.warning("layout_check_fetch_failed", extra={"kind": kind, "run_id": run_id})
            continue
        issues.extend(check_pdf_layout(pdf_bytes, document_label=kind))
    return issues


def _merge_layout_into_assessment(
    assessment: QAAssessment,
    layout_issues: list[QAIssue],
) -> QAAssessment:
    """Fold deterministic layout issues into an LLM-graded assessment.

    Each layout issue becomes a must-fix entry (severity always
    must_fix; category always format). Format score is knocked down
    by 15 points per layout issue (capped at 0) so the verdict tone
    matches reality -- a passing-content / broken-layout chain still
    flags as concerns or fail.
    """
    new_must_fix = list(assessment.must_fix_issues) + list(layout_issues)
    penalty = min(assessment.format_score, len(layout_issues) * 15)
    new_format = max(0, assessment.format_score - penalty)
    # Recompute status with the layout drag baked in.
    new_status = _recompute_status(
        assessment.coverage_score,
        assessment.consistency_score,
        new_format,
        must_fix_count=len(new_must_fix),
        nice_to_fix_count=len(assessment.nice_to_fix_issues),
    )
    return assessment.model_copy(
        update={
            "must_fix_issues": new_must_fix,
            "format_score": new_format,
            "status": new_status,
        },
    )


def _recompute_status(
    coverage: int,
    consistency: int,
    fmt: int,
    *,
    must_fix_count: int,
    nice_to_fix_count: int,
) -> QAStatus:
    """Apply the same banding the QA prompt encodes: any must-fix or
    score < 60 = fail; any nice-to-fix or score 60-79 = concerns; all
    scores >= 80 and zero must-fix = pass."""
    scores = (coverage, consistency, fmt)
    if must_fix_count > 0 or any(s < 60 for s in scores):
        return QAStatus.FAIL
    if nice_to_fix_count > 0 or any(s < 80 for s in scores):
        return QAStatus.CONCERNS
    return QAStatus.PASS


def _retry_targets(assessment: QAAssessment) -> set[str]:
    """Decide which artefact(s) to re-kick from the QA must-fix list.

    Scans every must-fix issue's summary + detail for keywords that
    point at the resume vs the letter. Returns a set with one or both
    of ``"resume"`` / ``"letter"``.

    Default to ``{"letter"}`` when nothing matches -- the original
    v1.10.0 behaviour, which is correct for the common "letter
    overstates X" / "letter wording inconsistent" case where the
    issue is freer-form letter content and the resume is fine.
    Resume retries are added when the issues explicitly name the
    resume so we don't burn a render cycle re-tailoring an artefact
    the LLM had no complaint about.
    """
    targets: set[str] = set()
    for issue in assessment.must_fix_issues:
        text = f"{issue.summary} {issue.detail or ''}".lower()
        if "resume" in text:
            targets.add("resume")
        if "letter" in text or "cover" in text:
            targets.add("letter")
    if not targets:
        targets = {"letter"}
    return targets


async def _rekick_resume_with_feedback(
    *,
    tailor_run_id: int,
    db_path: Path,
    resume_client: ResumeaiClient,
    payload: _JDPayload,
    assessment: QAAssessment,
    sleeper: Sleeper,
    poll_interval_s: float,
    max_polls: int,
    qa_context: str | None = None,
) -> str | None:
    """Re-tailor the resume with the QA must-fix list appended.

    Returns the new ``resume_run_id`` on success, or ``None`` if the
    retry could not produce a usable artefact (kick exception, sibling
    returned ``failed``, poll cap hit). Like the letter retry, this
    is non-fatal: failure leaves the first-pass resume in place and
    the chain still ships.

    The row transitions through ``qa_retry_running`` so the UI shows
    "auto-fix attempt" the same as it does for the letter retry path.
    """
    feedback_payload = _augment_payload_with_feedback(
        payload,
        assessment,
        qa_context=qa_context,
    )
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RETRY_RUNNING,
        )

    try:
        new_resume_run_id = await resume_client.kick(_build_resume_request(feedback_payload))
    except Exception:  # noqa: BLE001 - retry-kick failure is non-fatal
        _log.warning(
            "tailor_qa_retry_resume_kick_failed",
            extra={"tailor_run_id": tailor_run_id},
            exc_info=True,
        )
        return None
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RETRY_RUNNING,
            resume_run_id=new_resume_run_id,
        )

    try:
        snapshot = await _poll_until_terminal(
            kind="resume",
            run_id=new_resume_run_id,
            poll=resume_client.poll,
            sleeper=sleeper,
            poll_interval_s=poll_interval_s,
            max_polls=max_polls,
            db_path=db_path,
            tailor_run_id=tailor_run_id,
            status_field="resume_status",
            outer_status=TailorRunStatus.QA_RETRY_RUNNING,
        )
    except TailorChainError:
        _log.warning(
            "tailor_qa_retry_resume_poll_cap_hit",
            extra={
                "tailor_run_id": tailor_run_id,
                "retry_resume_run_id": new_resume_run_id,
            },
            exc_info=True,
        )
        return None
    if snapshot.status not in _TERMINAL_SUCCESS:
        return None
    return new_resume_run_id


async def _rekick_letter_with_feedback(
    *,
    tailor_run_id: int,
    db_path: Path,
    letter_client: CoverletteraiClient,
    payload: _JDPayload,
    resume_run_id: str,
    assessment: QAAssessment,
    sleeper: Sleeper,
    poll_interval_s: float,
    max_polls: int,
    qa_context: str | None = None,
) -> str | None:
    """Re-tailor the cover letter with the QA must-fix list appended.

    Returns the new ``letter_run_id`` on success, or ``None`` if the
    retry could not produce a usable artefact (kick exception, sibling
    returned ``failed``, schema validation error on the LLM output).
    The row transitions through ``qa_retry_running`` so the UI can
    show "auto-fix attempt" -- but on failure we keep the previous
    ``letter_run_id`` and surface that as the chain's terminal
    artefact rather than discarding the first-pass letter.

    Why non-fatal: the first-pass letter has already rendered
    successfully. A retry failure means the auto-fix attempt
    overstepped (e.g. produced a schema-violating 3-paragraph letter
    when the schema caps at 2), not that the original artefact is
    bad. The caller logs the retry failure and ships the original.
    """
    feedback_payload = _augment_payload_with_feedback(
        payload,
        assessment,
        qa_context=qa_context,
    )
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RETRY_RUNNING,
        )

    try:
        new_letter_run_id = await letter_client.kick(
            _build_letter_request(feedback_payload, resume_run_id=resume_run_id),
        )
    except Exception:  # noqa: BLE001 - retry-kick failure is non-fatal
        _log.warning(
            "tailor_qa_retry_kick_failed",
            extra={"tailor_run_id": tailor_run_id},
            exc_info=True,
        )
        return None
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RETRY_RUNNING,
            letter_run_id=new_letter_run_id,
        )

    try:
        snapshot = await _poll_until_terminal(
            kind="letter",
            run_id=new_letter_run_id,
            poll=letter_client.poll,
            sleeper=sleeper,
            poll_interval_s=poll_interval_s,
            max_polls=max_polls,
            db_path=db_path,
            tailor_run_id=tailor_run_id,
            status_field="letter_status",
            outer_status=TailorRunStatus.QA_RETRY_RUNNING,
        )
    except TailorChainError:
        # Poll cap hit during retry -- not fatal; keep first-pass letter.
        _log.warning(
            "tailor_qa_retry_poll_cap_hit",
            extra={
                "tailor_run_id": tailor_run_id,
                "retry_letter_run_id": new_letter_run_id,
            },
            exc_info=True,
        )
        return None
    if snapshot.status not in _TERMINAL_SUCCESS:
        return None
    return new_letter_run_id


def _augment_payload_with_feedback(
    payload: _JDPayload,
    assessment: QAAssessment,
    *,
    qa_context: str | None = None,
) -> _JDPayload:
    """Append QA feedback + verified-facts context to the JD text so
    the LLM sees the requirements, the prior attempt's problems, AND
    the ground truth it must conform to.

    Two things matter here for convergence:

    1. **Verified context inline.** The siblings already pull from the
       resumeai context pool, but the LLM has been observed to
       hallucinate stats (e.g. ~38k LOC vs verified ~15,700) on every
       attempt anyway. Pasting the verified block into the JD itself
       puts ground truth literally in the prompt the LLM is reading
       to generate the artefact -- no context-priority guessing.
    2. **Strong, artefact-agnostic guidance.** The wording is generic
       ("this submission") so the same augmenter works for resume and
       letter retries. The hardest rule -- "use verified numbers
       verbatim" -- is called out specifically so the LLM can't
       substitute its own estimate.

    When the original payload had no jd_text (one-off URL path with
    no catalogue match), this also surfaces the URL inline so the
    feedback block isn't orphaned from any context.
    """
    base = payload.jd_text or f"JOB URL: {payload.jd_url}\n(JD content not pre-fetched)"
    must_fix_lines = [
        f"- ({issue.category}) {issue.summary}" + (f": {issue.detail}" if issue.detail else "")
        for issue in assessment.must_fix_issues
    ]
    has_layout_issue = any(
        issue.category == "format" and "page" in issue.summary.lower()
        for issue in assessment.must_fix_issues
    )
    guidance: list[str] = []
    if qa_context:
        guidance.append(
            "When producing this submission, ground every numeric or "
            "factual claim in the VERIFIED FACTS block below. Use those "
            "numbers VERBATIM -- do not round, estimate, or substitute "
            "your own figure. If a JD requirement isn't backed by the "
            "candidate's verified experience, address it honestly "
            "rather than overstating. If the verified facts don't "
            "supply a specific number for something, write a general "
            'phrase ("comprehensive test suite", "production-scale '
            'system") rather than inventing one.',
        )
    else:
        guidance.append(
            "When producing this submission, ground every claim in the "
            "candidate's actual context. Do NOT attribute features to "
            "projects that don't have them, and do NOT invent metrics. "
            "If a JD requirement isn't backed by the candidate's "
            "experience, address it honestly rather than overstating.",
        )
    if has_layout_issue:
        guidance.append(
            "LAYOUT: the previous PDF had orphan bullets or a split "
            "section header. Produce a noticeably shorter artefact this "
            "time (aim for 3-4 fewer sentences, or tighter paragraphs) "
            "so the rendered output fits its page cleanly without a "
            "trailing line spilling over.",
        )

    sections: list[str] = [
        "",
        "QA FEEDBACK FROM PREVIOUS ATTEMPT — must address:",
        *must_fix_lines,
    ]
    if qa_context:
        sections.extend(
            [
                "",
                "VERIFIED FACTS (single source of truth for the candidate's "
                "projects and stats -- use these numbers verbatim):",
                qa_context.strip(),
            ],
        )
    sections.extend(["", *guidance])
    feedback_block = "\n".join(sections)
    return _JDPayload(jd_url=payload.jd_url, jd_text=base + feedback_block)


async def _poll_until_terminal(
    *,
    kind: str,
    run_id: str,
    poll: Callable[[str], Awaitable[SiblingRunSnapshot]],
    sleeper: Sleeper,
    poll_interval_s: float,
    max_polls: int,
    db_path: Path,
    tailor_run_id: int,
    status_field: str,
    outer_status: TailorRunStatus | None = None,
) -> SiblingRunSnapshot:
    """Poll ``poll(run_id)`` until the sibling returns a terminal status.

    Every successful poll persists the freshly observed sibling status
    onto the matching column (``resume_status`` or ``letter_status``)
    so the UI sees the progression. Raises :class:`TailorChainError` if
    the poll cap is hit without a terminal state.

    ``outer_status`` overrides the row-level ``status`` value written
    during each poll. Defaults to deriving from ``status_field``
    (``RESUME_RUNNING`` / ``LETTER_RUNNING``); the QA-retry path passes
    ``QA_RETRY_RUNNING`` so the UI keeps distinguishing the retry from
    the initial letter render.
    """
    derived_status = (
        outer_status
        if outer_status is not None
        else (
            TailorRunStatus.RESUME_RUNNING
            if status_field == "resume_status"
            else TailorRunStatus.LETTER_RUNNING
        )
    )
    for attempt in range(max_polls):
        snapshot = await poll(run_id)
        with connect(db_path) as conn:
            kwargs: dict[str, str] = {status_field: snapshot.status}
            update_status(
                conn,
                tailor_run_id,
                status=derived_status,
                **kwargs,  # type: ignore[arg-type] # narrow string-keyed dict spread
            )
        if snapshot.status in _TERMINAL_SUCCESS or snapshot.status in _TERMINAL_FAILURE:
            return snapshot
        _log.debug(
            "tailor_poll",
            extra={
                "kind": kind,
                "run_id": run_id,
                "attempt": attempt,
                "status": snapshot.status,
            },
        )
        await sleeper(poll_interval_s)
    msg = (
        f"{kind} run {run_id} did not terminate after {max_polls} polls "
        f"({max_polls * poll_interval_s:.0f}s)"
    )
    raise TailorChainError(msg)


@dataclass(frozen=True, slots=True)
class _JDPayload:
    """JD data resolved for one tailor run, ready to forward to the siblings.

    Both fields can be set (catalogue path with a known apply URL plus a
    description we extracted from raw HTML); when ``jd_text`` is set
    we PREFER it over the URL because resumeai's URL fetcher gets 403'd
    by anti-bot on several boards (SmartRecruiters in particular).
    Falling back to the URL only when we have no text avoids the
    re-fetch entirely for the 99% case where jobai already scraped
    the description.
    """

    jd_url: str
    jd_text: str | None


# Below ~200 chars we don't trust the extracted text -- some sources
# fill description fields with a tagline or "see full description on
# the apply page" placeholder. Falling back to the URL is safer than
# sending the model two-line garbage.
_MIN_USEFUL_JD_TEXT_LEN = 200


def _build_resume_request(payload: _JDPayload) -> ResumeaiTailorRequest:
    """Construct the resumeai request, preferring ``jd_text`` over ``jd_url``."""
    if payload.jd_text:
        return ResumeaiTailorRequest(jd_text=payload.jd_text)
    return ResumeaiTailorRequest(jd_url=payload.jd_url)


def _build_letter_request(
    payload: _JDPayload,
    *,
    resume_run_id: str,
) -> CoverletteraiTailorRequest:
    """Construct the coverletterai request, preferring ``jd_text`` over ``jd_url``."""
    if payload.jd_text:
        return CoverletteraiTailorRequest(
            jd_text=payload.jd_text,
            resume_run_id=resume_run_id,
        )
    return CoverletteraiTailorRequest(
        jd_url=payload.jd_url,
        resume_run_id=resume_run_id,
    )


def _strip_html_to_text(html: str | None) -> str | None:
    """Best-effort HTML → plain-text conversion.

    Returns ``None`` if the input is empty or strips to nothing. Joins
    runs of whitespace into single spaces so the resulting blob looks
    like flowing text rather than the original HTML's indentation.
    """
    if not html:
        return None
    tree = HTMLParser(html)
    text = tree.text(separator="\n", strip=True)
    if not text:
        return None
    return text


def _load_jd_payload(db_path: Path, tailor_run_id: int) -> _JDPayload:
    """Resolve the JD payload (url + optional text) for one tailor run.

    Two row shapes exist:

    * **Catalogue path** (``tailor_runs.job_id`` set) -- look up
      ``jobs.apply_url`` AND ``jobs.description_text`` /
      ``jobs.description_html``. When we have a useful description in
      our own DB we forward it as ``jd_text`` so resumeai skips the
      URL fetch entirely (this is the path that kept getting 403'd
      by SmartRecruiters etc).
    * **One-off URL path** (``tailor_runs.jd_url`` set) -- only the
      URL is available; the siblings have to fetch.

    Surfaces a clean error if the row is missing, the joined job
    has been deleted out from under us, or neither column is
    populated.
    """
    with connect(db_path) as conn:
        record = get_tailor_run(conn, tailor_run_id)
        if record is None:
            msg = f"tailor_run {tailor_run_id} not found"
            raise TailorChainError(msg)
        # One-off path: the row carries the URL directly and we have
        # no description on hand. The siblings will have to fetch it.
        if record.jd_url:
            return _JDPayload(jd_url=record.jd_url, jd_text=None)
        # pragma: no cover -- the DB-level CHECK on tailor_runs forbids
        # rows with neither field set. The Python guard is here so a
        # future schema-relaxation can't trigger a NULL URL to a
        # sibling, but exercising it requires bypassing the CHECK in
        # ways that aren't reachable from any production code path.
        if record.job_id is None:  # pragma: no cover
            msg = (
                f"tailor_run {tailor_run_id} carries neither job_id nor jd_url; "
                "cannot resolve a JD URL for the chain"
            )
            raise TailorChainError(msg)
        row: sqlite3.Row | None = conn.execute(
            "SELECT apply_url, description_text, description_html FROM jobs WHERE id = ?",
            (record.job_id,),
        ).fetchone()
        if row is None:
            msg = f"job {record.job_id} for tailor_run {tailor_run_id} no longer exists"
            raise TailorChainError(msg)
        apply_url = str(row["apply_url"])
        # Prefer description_text when it's substantial; otherwise
        # strip description_html into plain text. Either is forwarded
        # to the siblings via jd_text so they skip the URL fetch.
        text = row["description_text"]
        if not text or len(text) < _MIN_USEFUL_JD_TEXT_LEN:
            text = _strip_html_to_text(row["description_html"])
        if text and len(text) < _MIN_USEFUL_JD_TEXT_LEN:
            text = None
        return _JDPayload(jd_url=apply_url, jd_text=text)
