-- Make the QA stage an auto-fix loop, not a passive verdict.
--
-- Pre-this-migration the chain was: resume -> letter -> QA -> done.
-- If QA flagged must-fix issues the user saw the verdict and was on
-- their own to manually re-tailor anything. That misses the point of
-- having a QA agent at the end -- if it can detect a problem it
-- should drive the fix.
--
-- Two schema changes enable the new flow:
--
--   * ``qa_attempts`` -- how many QA passes have run for this row.
--     The orchestrator caps retries (default 2 total attempts) so we
--     don't burn unbounded LLM time on a hopeless prompt. When the
--     value is null/0 the chain still ships before any QA fires;
--     >0 means QA has graded at least once.
--   * ``qa_retry_running`` state -- visible to the UI while the
--     orchestrator is re-kicking the cover letter with QA feedback
--     between attempts. Without this new state, the row would flicker
--     back to ``letter_running`` and lose the distinction between
--     "first attempt" and "fix attempt".

ALTER TABLE tailor_runs ADD COLUMN qa_attempts INTEGER NOT NULL DEFAULT 0;

-- Recreate the table to widen the status CHECK with the new
-- ``qa_retry_running`` enum value. Same pattern as 0006/0007.
CREATE TABLE tailor_runs_new (
    id              INTEGER PRIMARY KEY,
    job_id          INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    jd_url          TEXT,
    status          TEXT    NOT NULL CHECK (status IN (
                        'pending', 'resume_running', 'letter_running',
                        'qa_running', 'qa_retry_running',
                        'succeeded', 'failed'
                    )),
    resume_run_id   TEXT,
    resume_status   TEXT,
    letter_run_id   TEXT,
    letter_status   TEXT,
    qa_status       TEXT,
    qa_assessment_json TEXT,
    qa_attempts     INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    CHECK (job_id IS NOT NULL OR jd_url IS NOT NULL)
);

INSERT INTO tailor_runs_new (
    id, job_id, jd_url, status, resume_run_id, resume_status,
    letter_run_id, letter_status, qa_status, qa_assessment_json,
    qa_attempts, error, created_at, updated_at, finished_at
)
SELECT
    id, job_id, jd_url, status, resume_run_id, resume_status,
    letter_run_id, letter_status, qa_status, qa_assessment_json,
    qa_attempts, error, created_at, updated_at, finished_at
FROM tailor_runs;

DROP TABLE tailor_runs;
ALTER TABLE tailor_runs_new RENAME TO tailor_runs;

CREATE INDEX idx_tailor_runs_job_id  ON tailor_runs (job_id);
CREATE INDEX idx_tailor_runs_status  ON tailor_runs (status);
CREATE INDEX idx_tailor_runs_created ON tailor_runs (created_at DESC);
