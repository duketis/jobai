# jobai

[![CI](https://github.com/duketis/jobai/actions/workflows/ci.yml/badge.svg)](https://github.com/duketis/jobai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type-checked: mypy](https://img.shields.io/badge/type--checked-mypy%20strict-1f5082.svg)](http://mypy-lang.org/)

A local-first AI job-hunting agent for the Australian market. One process scrapes 70+ AU and global job boards on a schedule into a SQLite database, exposes a REST + SSE API, and runs an Anthropic-powered chat agent that uses tools to search and triage roles. The whole thing ships as a single container.

> **Status:** v1.5.x — data layer, agent layer, frontend, Docker deploy, end-to-end resume + cover-letter tailoring with a **final cross-artefact QA pass** (works against both pay-per-token API and Claude Pro/Max subscription billing), **daily auto-discovery of new ATS slugs** from existing apply URLs, a **shared user-context pool** UI proxied through to resumeai (with **snippet + multi-file/folder upload + local git-project scan**), **chat-driven tailor kicks** (the agent has `kick_tailor` / `list_tailor_runs` / `get_tailor_run` tools so you can say "tailor this for me" instead of clicking the button), and a **Select-all** button in batch mode. Generic catalogue (no role bias). **Backend at 100% line + branch coverage; tailor + QA + context UI at 100% line + branch + functions.**

## What it does

- **Ingests ~15,000+ jobs per cycle** across 50 ATS sources (Greenhouse, Lever, Ashby, SmartRecruiters, Workable), 15 Seek slugs (every Australian capital + major regional centre + a national remote-only filter), 17 Indeed and 10 LinkedIn slugs (same coverage strategy), 4 AU state government boards (VIC / QLD / SA / WA — fully paginated, ~3,000 jobs/cycle from these alone), and the federal APS Jobs board. The big-board walkers (Seek, Indeed, LinkedIn) walk each query to its natural ceiling rather than stopping at an artificial page cap; multiplying location/work-mode slugs is how we work around the per-query ceilings the boards impose.
- **Deduplicates and best-of-merges** the same role across boards into one canonical row, with all source links preserved. When sources disagree (one has salary and one doesn't, one has a full description and one a teaser), per-field rules pick the richest value: longest description, earliest `posted_at`, first non-null salary.
- **Infers salary from description text** when the structured field is null. AU listings notoriously bury comp in the body — `Band 8 - $123,558 to $138,752 + super` on council jobs, `$120k - $160k + super` from recruiters, `Salary: $100,066 - $108,372 per annum`. The regex parser anchors on salary keywords (Salary:, Compensation:, per annum, + super, Band N -), rejects fundraising/AUM/revenue context, and refuses hourly/daily contractor rates to avoid mis-extrapolation. Falls back to a stripped version of `description_html` when sources (Greenhouse, SmartRecruiters, APS Jobs) populate only HTML.
- **Backfills full descriptions** on a slower cadence — LinkedIn guest-mode and Indeed both bypass per-page anti-bot via a session-aware fetch path.
- **Tailors a resume + cover-letter PDF per job, one click.** A new "Tailor" button on every job row kicks an async chain through the two sibling services [resumeai](https://github.com/duketis/resumeai) and [coverletterai](https://github.com/duketis/coverletterai): jobai POSTs the JD URL → resumeai produces a tailored resume → jobai feeds that resume id to coverletterai → coverletterai produces a matching cover letter → a **final QA agent** reads both artefacts back against the JD and emits a structured assessment (coverage / consistency / format scores 0–100, plus must-fix and nice-to-fix issue lists). The QA agent honours the same backend selector as the chat agent — it routes through your Claude Pro/Max quota under `JOBAI_AGENT_BACKEND=subscription` or pay-per-token under `api`, picked up live from the Settings UI without a restart. Both PDFs stream straight back through jobai. A batch mode lets you tick N jobs and queue them all (capped at 3 concurrent chains so the LLM-bound renderers don't pile up). A dedicated `/tailor-runs` page shows the lifecycle of every chain; the QA verdict shows up as a clickable badge that opens a drill-in panel with the full assessment.
- **Auto-discovers new ATS slugs once a day.** APScheduler runs the discovery job alongside the hourly scrapes (24h cadence): it mines every job's `apply_url` for Greenhouse / Lever / Ashby / SmartRecruiters / Workable slugs that aren't already seeded in `companies.yaml`, then upserts them as enabled sources so the next scrape picks them up. Zero manual maintenance — if a company shows up in a job we already ingested, we'll discover their full board within 24 hours.
- **Manages the shared user-context pool from one place.** A new `/context` page lists every snippet / file / project audit resumeai + coverletterai use during tailoring, with three inline forms: free-text snippet, multi-file or whole-folder upload (PDF / CSV / markdown / text — folder mode recursively walks the picked directory and skips unsupported types), and local git-project scan (resumeai walks the repo's commit history against the `/host/personal` mount). The source of truth still lives on the resumeai sibling; jobai proxies through so the full job-hunt workflow (browse → tailor → curate context) stays behind one URL.
- **Lets the agent do the whole flow for you.** The chat agent has tools for the full tailor lifecycle: `kick_tailor` queues the chain, `list_tailor_runs` reports what's in flight, `get_tailor_run` pulls the QA verdict + scores. Say "tailor that Atlassian role for me" and the agent finds the job id, kicks the chain, and reports back when it's done — same flow as clicking the Tailor button. Batch mode in the jobs list now has a Select-all toggle so you can queue every visible row in two clicks.
- **Serves a single-page React app** at `/` for browsing, filtering, and chatting with the agent. The Jobs header surfaces a live "updated X mins ago" freshness chip so you can see the data is current at a glance. The agent is an Anthropic SDK client driving 5 tools against the local DB; responses stream over SSE with full per-token visibility.

## Quick start (Docker)

The fastest path: one container, one volume, three commands.

```bash
git clone https://github.com/duketis/jobai.git
cd jobai
cp .env.example .env  # then fill in either an API key or an OAuth token, see below
docker compose up -d
```

The app is at <http://localhost:8421>. The SQLite DB lives in the `jobai-data` named volume, so `docker compose down` is safe — your scraped jobs persist.

### Tailor integration (optional)

The "Tailor" button needs the two sibling services running on the same Docker network. Bring them up first (each is a one-liner against their own checkout); jobai attaches to the shared `ai-tailor-network` automatically:

```bash
# In each sibling repo:
(cd ~/Documents/personal/resumeai      && docker compose up -d)   # :8765
(cd ~/Documents/personal/coverletterai && docker compose up -d)   # :8766
# Then bring up jobai (creates the ai-tailor-network if siblings haven't yet):
docker compose up -d
```

Without the siblings, jobai still runs the data + agent layers normally; only the `POST /api/tailor/*` routes fail at runtime. You can also point jobai at non-default sibling URLs via the `JOBAI_RESUMEAI_URL` / `JOBAI_COVERLETTERAI_URL` env vars (default: the service names on `ai-tailor-network`).

### Agent backends — pay-per-token vs subscription

The chat dock has two auth paths; pick one in `.env`:

* **`JOBAI_AGENT_BACKEND=api`** (default) — uses `ANTHROPIC_API_KEY` against the Anthropic API. Get a key from <https://console.anthropic.com>. Pay-per-token billing.
* **`JOBAI_AGENT_BACKEND=subscription`** — uses your Claude Pro/Max plan via the `claude` CLI bundled into the image. Run `claude setup-token` on your host to generate a long-lived OAuth token (`sk-ant-oat-…`), paste it into `CLAUDE_CODE_OAUTH_TOKEN`, and the in-container CLI authenticates with your Max account. **Calls bill against your Max quota, not API credit.**

Both are wired to the same `/api/agent/chat` endpoint and emit identical SSE events; the chat dock UI doesn't know which backend it's talking to.

## Quick start (local Python)

Requires Python 3.12, Node 22+, and a Chromium that Playwright can drive.

```bash
git clone https://github.com/duketis/jobai.git
cd jobai

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium
patchright install chromium

(cd frontend && npm ci && npm run build)

jobai migrate
jobai source sync
jobai serve  # http://localhost:8421
```

For a one-off scrape without booting the API:

```bash
jobai run --source greenhouse:atlassian
jobai run --enabled    # walks every enabled source sequentially
```

## CLI

```text
jobai migrate                       Apply pending DB migrations.
jobai serve                         Start the scheduler + HTTP API + agent.
jobai run --source <kind>:<acct>    Run one source ad hoc.
jobai run --enabled                 Run every enabled source sequentially.
jobai reconcile                     Re-run cross-source fuzzy reconciliation.
jobai infer-remote                  Backfill remote_type for jobs missing one.
jobai infer-salary                  Backfill salary fields from description text.
jobai source sync                   Upsert source rows from companies.yaml.
jobai source list                   List configured sources.
jobai source enable <name>          Re-enable a disabled source.
jobai source disable <name>         Disable a source (kill switch).
jobai source discover [--register]  Mine apply_url for ATS slugs we haven't
                                    seeded yet (Greenhouse / Lever / Ashby /
                                    SmartRecruiters / Workable). ``--register``
                                    upserts each as an enabled source so the
                                    next ``run --enabled`` picks them up.
```

## HTTP API

Full OpenAPI spec at <http://localhost:8421/docs>. The headline endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/jobs` | Search + filter (q, location, remote, posted_since, source, limit, offset). |
| `GET` | `/api/jobs/{id}` | Full job detail. |
| `POST` | `/api/jobs/{id}/state` | Mark saved / applied / dismissed. |
| `GET` | `/api/sources` | Source registry with per-source health. |
| `GET` | `/api/health` | Aggregate health snapshot. |
| `POST` | `/api/agent/chat` | SSE-streamed agent turn. Body: `{conversation_id?, message}`. |
| `GET` | `/api/conversations` | List recent conversations. |
| `GET` | `/api/conversations/{id}` | Full message history for a conversation. |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation. |
| `POST` | `/api/tailor/jobs/{id}` | Kick off a tailor chain for one job (resumeai + coverletterai). |
| `POST` | `/api/tailor/batch` | Kick off chains for many jobs at once. Body: `{job_ids: [...]}`. |
| `GET` | `/api/tailor/runs` | List tailor runs, newest first; filterable by `job_id` / `status`. |
| `GET` | `/api/tailor/runs/{id}` | Inspect one tailor run (state, sibling run ids, error). |
| `GET` | `/api/tailor/runs/{id}/resume.pdf` | Stream the tailored resume PDF (proxied from resumeai). |
| `GET` | `/api/tailor/runs/{id}/letter.pdf` | Stream the tailored cover-letter PDF (proxied from coverletterai). |

## Architecture

```text
┌─ jobai/sources/         per-board parsers (one Source class per kind)
│  ├─ {greenhouse,lever,ashby,smartrecruiters,workable}.py  ATS APIs
│  ├─ {seek,linkedin,indeed}.py                             big private boards
│  ├─ {nsw,vic,qld,sa,wa}_*.py                              AU state govs
│  └─ apsjobs.py                                            AU federal (Salesforce Lightning)
├─ jobai/fetcher/         3-tier fetcher pattern
│  ├─ http.py             tier 1 (httpx)
│  ├─ browser.py          tier 2 (Playwright + run_in_page escape hatch)
│  ├─ stealth.py          tier 3 (Patchright)
│  └─ escalation.py       transparent tier promotion on 403
├─ jobai/dedup/           deterministic SHA256 + fuzzy rapidfuzz + per-field best-of merger
├─ jobai/pipeline/        scrape runner, schema-change detection, description backfill
├─ jobai/agent/           Anthropic SDK agent — 8 tools (search, detail, state, sources, health, kick_tailor, list/get_tailor_runs), manual loop, SSE streaming
├─ jobai/tailor/          orchestrator + Protocol-based sibling clients for resumeai/coverletterai + QA agent
├─ jobai/context/         user-context pool proxy (lists / snippets / uploads / deletes forwarded to resumeai)
├─ jobai/api/             FastAPI app: /api/* + the React SPA mounted at /
├─ jobai/scheduler.py     APScheduler runs in the FastAPI lifespan
└─ frontend/              React + Vite + TypeScript + Tailwind v4 SPA
```

A few decisions worth pulling out:

- **3-tier fetcher** — sources declare `default_tier=1`; the runtime wraps tier-1 in `EscalatingFetcher` so a single 403 promotes the whole cycle to tier-2 (or tier-3 if the source's quirk demands it). Saves the request budget against walls.
- **`run_in_page()` escape hatch** — the AU state government boards (VIC, SA, WA) only render results after a click on a search-form button. Rather than force every source to learn Playwright, the browser tier exposes a single `run_in_page(url, page_script)` method that hands the source a Playwright `Page` to drive directly.
- **Frozen system prompt + cache_control** — keeps the Anthropic prompt cache warm across agent turns so token cost stays low and latency stays predictable.
- **Modular monolith** — one process, SQLite + WAL, no message broker. Solo user, complexity-budget reasoning.

## Development

```bash
pytest -q                       # 1041 tests pass
pytest --cov=jobai --cov-branch --cov-report=term-missing
mypy jobai tests                # strict
ruff check . && ruff format --check .

(cd frontend && npm ci && npm run build)        # TypeScript strict, Vite production build
(cd frontend && npm run test:coverage)          # Vitest -- 53 tests, 100% on tailor + QA + context UI
```

CI runs ruff, mypy, pytest, and the frontend build on every push to `main`. All commits are GPG-signed. **The Python backend is at 100.0% combined line + branch coverage** (1041 tests, every module). Lines that genuinely cannot be exercised under unit tests (real Chromium / Patchright via Playwright, the `claude` CLI subprocess in subscription mode, defensive guards for SQLite invariants like `cursor.lastrowid is None`) are excluded via `# pragma: no cover` with a one-line reason; everything else lives behind tests. The tailor + QA + context UI (`TailorButton`, `TailorStatusPill`, `QABadge`, `useLatestTailorRunsByJob`, `TailorRunsPage`, `ContextPage`) is at 100% line + branch + functions under Vitest.

## Known limitations

- **NSW Government (`iworkfor.nsw.gov.au`) — full coverage via in-browser pagination.** The site sits behind Cloudflare's strict challenge mode (May 2026 onward). The tier-3 stealth fetcher with `needs_persistent_session=True` keeps one browser context alive across all NSW fetches in a scrape cycle, solves CF once via the clean-UA + `networkidle` navigation pattern, then drives the SPA's Ant Design pagination control (`[aria-label="Go to next page"]`) one click at a time, accumulating each page's `article.search-job-card` HTML in Python and injecting everything into the final DOM snapshot. URL `?page=N` does NOT work (the Angular app ignores it). Discover still raises `NSWIWorkForBlockedError` on the CF interstitial so a real block surfaces as a failure instead of a silent zero-card success. The Cloudflare fingerprint check rate-limits aggressive re-runs from the same IP — production hourly cadence is fine; bench-testing the same source repeatedly will trip the throttle.
- **Anonymous pagination caps.** Seek caps unauthenticated search at ~2,200 results per query, LinkedIn at ~1,000, Indeed varies. The walkers stop on the first empty page so we get whatever the site is willing to serve.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built by [Jonathan Duketis](https://github.com/duketis). Code generation and review assisted by [Claude](https://claude.ai) (Anthropic).
