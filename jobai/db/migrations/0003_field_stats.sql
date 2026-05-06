-- Track per-field presence rates per scrape run for schema-change detection.
--
-- The pipeline runner walks every NormalizedJob and computes a
-- {field_name: present_count, null_count} map at the end of each run.
-- Storing the JSON on scrape_runs lets the next run compare against
-- the previous successful run for this source, surfacing fields that
-- suddenly went null (parser drift, upstream schema change).
--
-- Why JSON not a separate child table: stats are write-once at run
-- finalisation and read as a whole blob; a child table would force a
-- 20-row INSERT per run with no query benefits we need today.

ALTER TABLE scrape_runs ADD COLUMN field_stats_json TEXT;
