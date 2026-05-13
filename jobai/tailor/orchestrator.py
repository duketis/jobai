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
from pathlib import Path

from jobai.db.connection import connect
from jobai.tailor.client import CoverletteraiClient, ResumeaiClient
from jobai.tailor.models import (
    CoverletteraiTailorRequest,
    QAStatus,
    ResumeaiTailorRequest,
    SiblingRunSnapshot,
    TailorRunStatus,
)
from jobai.tailor.qa import QAClient, assess
from jobai.tailor.repository import get_tailor_run, update_status

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


async def _run_chain_inner(
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
) -> None:
    apply_url = _load_apply_url(db_path, tailor_run_id)

    # ---- resume ---------------------------------------------------------
    with connect(db_path) as conn:
        update_status(conn, tailor_run_id, status=TailorRunStatus.RESUME_RUNNING)

    resume_run_id = await resume_client.kick(ResumeaiTailorRequest(jd_url=apply_url))
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
        CoverletteraiTailorRequest(jd_url=apply_url, resume_run_id=resume_run_id),
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

    # ---- QA (cross-artefact pass) --------------------------------------
    if qa_client is not None:
        await _run_qa_stage(
            tailor_run_id=tailor_run_id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            resume_run_id=resume_run_id,
            letter_run_id=letter_run_id,
            qa_client=qa_client,
            qa_model=qa_model,
        )

    # ---- terminal success ----------------------------------------------
    with connect(db_path) as conn:
        update_status(conn, tailor_run_id, status=TailorRunStatus.SUCCEEDED)


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
) -> None:
    """Pull both sibling run records and run the cross-artefact QA pass.

    The orchestrator catches the run-level exception in :func:`run_chain`
    so a QA-stage failure won't kill the chain -- the PDFs still ship
    with a failed QA assessment attached.
    """
    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RUNNING,
            qa_status=QAStatus.RUNNING,
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
    )

    with connect(db_path) as conn:
        update_status(
            conn,
            tailor_run_id,
            status=TailorRunStatus.QA_RUNNING,
            qa_status=assessment.status,
            qa_assessment=assessment,
        )


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
) -> SiblingRunSnapshot:
    """Poll ``poll(run_id)`` until the sibling returns a terminal status.

    Every successful poll persists the freshly observed sibling status
    onto the matching column (``resume_status`` or ``letter_status``)
    so the UI sees the progression. Raises :class:`TailorChainError` if
    the poll cap is hit without a terminal state.
    """
    for attempt in range(max_polls):
        snapshot = await poll(run_id)
        with connect(db_path) as conn:
            kwargs: dict[str, str] = {status_field: snapshot.status}
            update_status(
                conn,
                tailor_run_id,
                status=(
                    TailorRunStatus.RESUME_RUNNING
                    if status_field == "resume_status"
                    else TailorRunStatus.LETTER_RUNNING
                ),
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


def _load_apply_url(db_path: Path, tailor_run_id: int) -> str:
    """Resolve the JD URL we send to the siblings from the tailor row's job.

    Surfaces a clean error if the row or its job has been deleted out
    from under us so the chain aborts rather than calling the sibling
    with a NULL URL.
    """
    with connect(db_path) as conn:
        record = get_tailor_run(conn, tailor_run_id)
        if record is None:
            msg = f"tailor_run {tailor_run_id} not found"
            raise TailorChainError(msg)
        row: sqlite3.Row | None = conn.execute(
            "SELECT apply_url FROM jobs WHERE id = ?",
            (record.job_id,),
        ).fetchone()
        if row is None:
            msg = f"job {record.job_id} for tailor_run {tailor_run_id} no longer exists"
            raise TailorChainError(msg)
        return str(row["apply_url"])
