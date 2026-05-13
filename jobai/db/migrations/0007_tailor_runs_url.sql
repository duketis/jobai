-- Support one-off tailor chains kicked from a bare JD URL.
--
-- The Tailor button on the Jobs page (and the chat agent's
-- ``kick_tailor`` tool) both require the job to already exist in
-- jobai's catalogue -- they resolve the JD URL by joining
-- tailor_runs.job_id -> jobs.apply_url. That's fine for jobs we
-- scraped, but useless when the user has a JD URL from somewhere
-- jobai doesn't crawl (a LinkedIn DM, a recruiter email, a friend's
-- referral, anything random).
--
-- Two schema changes make the URL-only path first-class:
--
--   * ``job_id`` becomes nullable so a row can exist for a JD that
--     was never scraped into the catalogue.
--   * ``jd_url`` carries the URL directly on the row when no
--     ``job_id`` is set; the orchestrator prefers this column over
--     the jobs join when it's populated.
--
-- The FK + ON DELETE CASCADE stays in place so catalogue-driven
-- runs still get cleaned up when their job is deleted; URL-only
-- runs are unaffected by the cascade because their FK is null.

ALTER TABLE tailor_runs ADD COLUMN jd_url TEXT;

-- Recreate the table to drop the NOT NULL on job_id. Same standard
-- SQLite 'create + copy + drop + rename' pattern as migration 0006.
CREATE TABLE tailor_runs_new (
    id              INTEGER PRIMARY KEY,
    job_id          INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    jd_url          TEXT,
    status          TEXT    NOT NULL CHECK (status IN (
                        'pending', 'resume_running', 'letter_running',
                        'qa_running', 'succeeded', 'failed'
                    )),
    resume_run_id   TEXT,
    resume_status   TEXT,
    letter_run_id   TEXT,
    letter_status   TEXT,
    qa_status       TEXT,
    qa_assessment_json TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    -- Every row carries EITHER a job_id (catalogue path) OR a
    -- jd_url (one-off path). Both being null is invalid; both
    -- being set is unusual but allowed (the URL would override
    -- the jobs.apply_url lookup if anything ever populated both).
    CHECK (job_id IS NOT NULL OR jd_url IS NOT NULL)
);

INSERT INTO tailor_runs_new (
    id, job_id, jd_url, status, resume_run_id, resume_status,
    letter_run_id, letter_status, qa_status, qa_assessment_json,
    error, created_at, updated_at, finished_at
)
SELECT
    id, job_id, jd_url, status, resume_run_id, resume_status,
    letter_run_id, letter_status, qa_status, qa_assessment_json,
    error, created_at, updated_at, finished_at
FROM tailor_runs;

DROP TABLE tailor_runs;
ALTER TABLE tailor_runs_new RENAME TO tailor_runs;

CREATE INDEX idx_tailor_runs_job_id  ON tailor_runs (job_id);
CREATE INDEX idx_tailor_runs_status  ON tailor_runs (status);
CREATE INDEX idx_tailor_runs_created ON tailor_runs (created_at DESC);
