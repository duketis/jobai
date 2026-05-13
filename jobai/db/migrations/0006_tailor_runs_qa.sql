-- Extend tailor_runs with a final-QA stage.
--
-- After resumeai + coverletterai both succeed, jobai runs ONE additional
-- LLM call that reads both rendered artefacts alongside the JD and
-- returns a structured assessment (coverage / consistency / format scores
-- plus enumerated issues). This catches inter-document problems neither
-- sibling can see in isolation -- a resume bullet contradicting a cover-
-- letter claim, a JD keyword missing from both, format inconsistency
-- between the two PDFs side-by-side.
--
-- ``qa_status`` walks: null -> 'running' -> 'pass' | 'concerns' | 'fail'.
-- ``qa_assessment_json`` carries the structured details for the UI's
-- drill-in panel and is null until the QA pass completes.

ALTER TABLE tailor_runs ADD COLUMN qa_status TEXT;
ALTER TABLE tailor_runs ADD COLUMN qa_assessment_json TEXT;

-- Recreate the CHECK constraint via the standard 'rename + create + copy
-- + drop' SQLite pattern so the status enum can grow a new value.
CREATE TABLE tailor_runs_new (
    id              INTEGER PRIMARY KEY,
    job_id          INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
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
    finished_at     TEXT
);

INSERT INTO tailor_runs_new (
    id, job_id, status, resume_run_id, resume_status,
    letter_run_id, letter_status, qa_status, qa_assessment_json,
    error, created_at, updated_at, finished_at
)
SELECT
    id, job_id, status, resume_run_id, resume_status,
    letter_run_id, letter_status, qa_status, qa_assessment_json,
    error, created_at, updated_at, finished_at
FROM tailor_runs;

DROP TABLE tailor_runs;
ALTER TABLE tailor_runs_new RENAME TO tailor_runs;

CREATE INDEX idx_tailor_runs_job_id  ON tailor_runs (job_id);
CREATE INDEX idx_tailor_runs_status  ON tailor_runs (status);
CREATE INDEX idx_tailor_runs_created ON tailor_runs (created_at DESC);
