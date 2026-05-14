-- Cache the per-run download filenames on the tailor_runs row.
--
-- v1.13.0 introduced descriptive PDF filenames built from the
-- applicant name + job title + company + kind (Resume / CoverLetter).
-- The PDF route computes the filename at request time and emits it
-- via Content-Disposition. Building the filename requires a sibling
-- HTTP call (to fetch the applicant identity off resumeai) which is
-- fine on the streaming endpoint but too slow to do once per row in
-- the list-runs response.
--
-- This migration adds two nullable text columns. The orchestrator
-- populates them when the chain reaches terminal SUCCESS so the
-- frontend can render the real filename as the link label and pin it
-- to the <a download=...> attribute -- without an N+1 sibling fetch.
--
-- Existing rows (already-finished chains pre-this-migration) carry
-- NULL; the API endpoint falls back to live computation on access so
-- old runs keep working. The PDF route also still computes live if
-- the cached value is missing.

ALTER TABLE tailor_runs ADD COLUMN resume_filename TEXT;
ALTER TABLE tailor_runs ADD COLUMN letter_filename TEXT;
