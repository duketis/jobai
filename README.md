# jobai

[![CI](https://github.com/duketis/jobai/actions/workflows/ci.yml/badge.svg)](https://github.com/duketis/jobai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type-checked: mypy](https://img.shields.io/badge/type--checked-mypy%20strict-1f5082.svg)](http://mypy-lang.org/)

A local-first AI job-hunting agent for the Australian market. One process scrapes 70+ AU and global job boards on a schedule into a SQLite database, exposes a REST + SSE API, and runs an Anthropic-powered chat agent that uses tools to search and triage roles. The whole thing ships as a single container.

> **Status:** v1.0.x — data layer, agent layer, frontend, and Docker deploy all live. Generic catalogue (no role bias).

## What it does

- **Ingests ~9,000+ jobs per cycle** across 50 ATS sources (Greenhouse, Lever, Ashby, SmartRecruiters, Workable), 5 Seek slugs, 3 Indeed and 3 LinkedIn searches, all 5 AU state government boards (NSW / VIC / QLD / SA / WA), and the federal APS Jobs board.
- **Deduplicates and best-of-merges** the same role across boards into one canonical row, with all source links preserved. When sources disagree (one has salary and one doesn't, one has a full description and one a teaser), per-field rules pick the richest value: longest description, earliest `posted_at`, first non-null salary.
- **Backfills full descriptions** on a slower cadence — LinkedIn guest-mode and Indeed both bypass per-page anti-bot via a session-aware fetch path.
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
jobai source sync                   Upsert source rows from companies.yaml.
jobai source list                   List configured sources.
jobai source enable <name>          Re-enable a disabled source.
jobai source disable <name>         Disable a source (kill switch).
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
├─ jobai/agent/           Anthropic SDK agent — 5 tools, manual loop, SSE streaming
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
pytest -q                       # 640+ tests
pytest --cov=jobai --cov-report=term-missing
mypy jobai tests                # strict
ruff check . && ruff format --check .

(cd frontend && npm run build)  # TypeScript strict, Vite production build
```

CI runs ruff, mypy, pytest, and the frontend build on every push to `main`. All commits are GPG-signed. New code lands at 100% coverage on the modules it touches — no exceptions, no `# pragma: no cover`.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built by [Jonathan Duketis](https://github.com/duketis). Code generation and review assisted by [Claude](https://claude.ai) (Anthropic).
