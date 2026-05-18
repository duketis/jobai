-- Persist the resume-only QA gate's verdict separately from the
-- combined (letter) QA verdict.
--
-- v1.28.0 splits the single post-render QA pass into two sequential
-- gates: the resume is graded against (JD + verified context) FIRST,
-- and the cover letter is only kicked once the resume gate has had
-- its fix-loop. The existing ``qa_status`` / ``qa_assessment_json``
-- columns keep carrying the final (letter-stage) verdict so the
-- existing API + UI badge are unchanged; these two new columns make
-- the resume gate's own verdict auditable rather than transient.
--
-- Both nullable: NULL = the resume gate didn't run (no qa_client, or
-- a pre-v1.28.0 row). A value mirrors the ``qa_status`` /
-- ``qa_assessment_json`` shape (StrEnum string + QAAssessment JSON).

ALTER TABLE tailor_runs ADD COLUMN resume_qa_status TEXT;
ALTER TABLE tailor_runs ADD COLUMN resume_qa_assessment_json TEXT;
