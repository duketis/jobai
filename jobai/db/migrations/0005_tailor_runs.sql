-- Tailor runs: one row per (job, kick-off) chain through resumeai + coverletterai.
--
-- The chain is two HTTP calls. We track its lifecycle here so the UI can show
-- progress, the user can re-kick on failure without re-typing the JD URL, and
-- a soak/scheduler can replay a failed run later.
--
-- ``status`` walks a small state machine driven by the orchestrator:
--   pending -> resume_running -> letter_running -> succeeded
--                              \-> failed (at any stage)
--
-- Sibling-service run ids live on the row so the frontend can build PDF URLs
-- without an extra round-trip; the per-sibling status fields record the last
-- terminal/in-flight status we polled, useful for surfacing partial failures.

CREATE TABLE tailor_runs (
    id              INTEGER PRIMARY KEY,
    job_id          INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    status          TEXT    NOT NULL CHECK (status IN (
                        'pending', 'resume_running', 'letter_running',
                        'succeeded', 'failed'
                    )),
    resume_run_id   TEXT,
    resume_status   TEXT,
    letter_run_id   TEXT,
    letter_status   TEXT,
    error           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT
);

CREATE INDEX idx_tailor_runs_job_id  ON tailor_runs (job_id);
CREATE INDEX idx_tailor_runs_status  ON tailor_runs (status);
CREATE INDEX idx_tailor_runs_created ON tailor_runs (created_at DESC);
