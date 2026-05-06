-- App-level runtime settings.
--
-- A single key/value table that lets the UI persist user-tunable
-- configuration without editing .env. Values that the user sets at
-- runtime (agent backend, API key, OAuth token, model id) override
-- the env-file defaults loaded by jobai.config; absence falls back
-- to the env value.
--
-- Single row per key — settings are global to this jobai instance,
-- not per-user (it's a single-tenant local-first app).

CREATE TABLE IF NOT EXISTS app_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
