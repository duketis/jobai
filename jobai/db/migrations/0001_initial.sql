-- Initial schema for jobai.
--
-- One file = one migration. Filename format: {NNNN}_{snake_case_name}.sql.
-- The migration runner (jobai/db/migrations.py) records each applied
-- migration in `_schema_migrations` so re-applying is a no-op.
--
-- Every TEXT timestamp is ISO 8601 UTC unless explicitly named otherwise.

-- ---------------------------------------------------------------------------
-- Source registry
-- ---------------------------------------------------------------------------
CREATE TABLE sources (
    id              INTEGER PRIMARY KEY,
    kind            TEXT    NOT NULL,
    account         TEXT    NOT NULL DEFAULT '',
    display_name    TEXT    NOT NULL,
    default_tier    INTEGER NOT NULL DEFAULT 1,
    enabled         INTEGER NOT NULL DEFAULT 1,
    cadence_seconds INTEGER NOT NULL,
    config_json     TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (kind, account)
);

CREATE TABLE source_runtime_state (
    source_id            INTEGER PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    current_tier         INTEGER NOT NULL,
    last_success_at      TEXT,
    last_error_at        TEXT,
    last_error_class     TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    cooldown_until       TEXT
);

-- ---------------------------------------------------------------------------
-- Scrape runs and raw responses
-- ---------------------------------------------------------------------------
CREATE TABLE scrape_runs (
    id            INTEGER PRIMARY KEY,
    source_id     INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    started_at    TEXT    NOT NULL,
    finished_at   TEXT,
    status        TEXT    NOT NULL CHECK (status IN ('running', 'success', 'partial', 'failed')),
    items_seen    INTEGER NOT NULL DEFAULT 0,
    items_new     INTEGER NOT NULL DEFAULT 0,
    items_updated INTEGER NOT NULL DEFAULT 0,
    tier_used     INTEGER NOT NULL,
    error_summary TEXT
);
CREATE INDEX idx_scrape_runs_source_started ON scrape_runs (source_id, started_at DESC);

CREATE TABLE raw_responses (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES scrape_runs(id) ON DELETE CASCADE,
    source_id    INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    url          TEXT    NOT NULL,
    fetched_at   TEXT    NOT NULL,
    status_code  INTEGER NOT NULL,
    headers_json TEXT    NOT NULL,
    body_gz      BLOB    NOT NULL,
    body_sha256  TEXT    NOT NULL,
    expires_at   TEXT    NOT NULL
);
CREATE INDEX idx_raw_responses_source_fetched ON raw_responses (source_id, fetched_at DESC);
CREATE INDEX idx_raw_responses_expires        ON raw_responses (expires_at);

-- ---------------------------------------------------------------------------
-- Per-source canonical job records (one row per source's external id)
-- ---------------------------------------------------------------------------
CREATE TABLE jobs_raw (
    id                 INTEGER PRIMARY KEY,
    source_id          INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_external_id TEXT    NOT NULL,
    raw_json           TEXT    NOT NULL,
    raw_sha256         TEXT    NOT NULL,
    first_seen_at      TEXT    NOT NULL,
    last_seen_at       TEXT    NOT NULL,
    closed_at          TEXT,
    UNIQUE (source_id, source_external_id)
);
CREATE INDEX idx_jobs_raw_last_seen ON jobs_raw (last_seen_at DESC);

-- ---------------------------------------------------------------------------
-- Cross-source canonical jobs
-- ---------------------------------------------------------------------------
CREATE TABLE jobs (
    id               INTEGER PRIMARY KEY,
    dedup_key        TEXT    NOT NULL UNIQUE,
    title            TEXT    NOT NULL,
    company          TEXT    NOT NULL,
    company_norm     TEXT    NOT NULL,
    location_raw     TEXT,
    location_country TEXT,
    location_city    TEXT,
    remote_type      TEXT,
    employment_type  TEXT,
    posted_at        TEXT,
    salary_min       INTEGER,
    salary_max       INTEGER,
    salary_currency  TEXT,
    description_text TEXT,
    description_html TEXT,
    apply_url        TEXT    NOT NULL,
    first_seen_at    TEXT    NOT NULL,
    last_seen_at     TEXT    NOT NULL,
    closed_at        TEXT,
    fingerprint_json TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_jobs_company_norm ON jobs (company_norm);
CREATE INDEX idx_jobs_last_seen    ON jobs (last_seen_at DESC);
CREATE INDEX idx_jobs_posted_at    ON jobs (posted_at DESC);

CREATE TABLE job_sources (
    job_id      INTEGER NOT NULL REFERENCES jobs(id)      ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES sources(id)   ON DELETE CASCADE,
    jobs_raw_id INTEGER NOT NULL REFERENCES jobs_raw(id)  ON DELETE CASCADE,
    apply_url   TEXT    NOT NULL,
    PRIMARY KEY (job_id, source_id, jobs_raw_id)
);

-- ---------------------------------------------------------------------------
-- Full-text search index over jobs(title, company, description_text, location_raw)
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE jobs_fts USING fts5 (
    title,
    company,
    description_text,
    location_raw,
    content='jobs',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER jobs_after_insert AFTER INSERT ON jobs BEGIN
    INSERT INTO jobs_fts (rowid, title, company, description_text, location_raw)
    VALUES (new.id, new.title, new.company, new.description_text, new.location_raw);
END;

CREATE TRIGGER jobs_after_delete AFTER DELETE ON jobs BEGIN
    INSERT INTO jobs_fts (jobs_fts, rowid, title, company, description_text, location_raw)
    VALUES ('delete', old.id, old.title, old.company, old.description_text, old.location_raw);
END;

CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs BEGIN
    INSERT INTO jobs_fts (jobs_fts, rowid, title, company, description_text, location_raw)
    VALUES ('delete', old.id, old.title, old.company, old.description_text, old.location_raw);
    INSERT INTO jobs_fts (rowid, title, company, description_text, location_raw)
    VALUES (new.id, new.title, new.company, new.description_text, new.location_raw);
END;

-- ---------------------------------------------------------------------------
-- User state and AI analysis (separate tables to avoid write contention)
-- ---------------------------------------------------------------------------
CREATE TABLE jobs_user_state (
    job_id     INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    state      TEXT    NOT NULL CHECK (state IN ('new', 'saved', 'applied', 'dismissed', 'rejected')),
    notes      TEXT,
    updated_at TEXT    NOT NULL
);

CREATE TABLE jobs_ai_analysis (
    job_id                INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    fit_score             REAL,
    fit_reasoning         TEXT,
    extracted_skills_json TEXT,
    analyzed_at           TEXT NOT NULL,
    model                 TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Errors and notifications
-- ---------------------------------------------------------------------------
CREATE TABLE errors (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER REFERENCES scrape_runs(id) ON DELETE SET NULL,
    source_id   INTEGER REFERENCES sources(id)     ON DELETE SET NULL,
    error_class TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    traceback   TEXT,
    raised_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_errors_raised ON errors (raised_at DESC);

CREATE TABLE notifications (
    id           INTEGER PRIMARY KEY,
    kind         TEXT    NOT NULL,
    severity     TEXT    NOT NULL CHECK (severity IN ('info', 'warn', 'error')),
    title        TEXT    NOT NULL,
    body         TEXT,
    payload_json TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    read_at      TEXT
);
CREATE INDEX idx_notifications_unread ON notifications (read_at) WHERE read_at IS NULL;
