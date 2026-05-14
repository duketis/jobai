-- Track the date the user actually submitted each tailored application.
--
-- The tailor pipeline produces a resume + cover letter pair and that
-- lives on the ``tailor_runs`` row. But "have I applied for this job
-- yet" is a separate dimension -- a user can tailor today and apply
-- tomorrow, or tailor and never apply, or re-tailor a job they've
-- already submitted to. v1.17.0 tried to solve this with a per-folder
-- CHECKLIST.md the user ticked off; v1.18.0 moves the source of truth
-- into the database so the UI can show applied-state directly.
--
-- ``applied_at`` is nullable: NULL means "not yet applied", any value
-- means "applied at this UTC timestamp". The API exposes a PATCH that
-- toggles between the two so a misclick is recoverable.

ALTER TABLE tailor_runs ADD COLUMN applied_at TEXT;
