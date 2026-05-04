# jobai

A local-first job-hunting agent. Continuously ingests listings from many sources, deduplicates them across providers, and exposes them through a search API and a conversational AI layer that helps you find, evaluate, and apply to roles.

> **Status:** in active development. The data layer (sprint 1) is being built; the AI/agent layer follows.

## Features

- **Multi-source ingestion** — Greenhouse, Lever, Ashby, Workable, SmartRecruiters, Seek, Jora, EthicalJobs, RemoteOK, Remotive, We Work Remotely, Hacker News "Who is Hiring", LinkedIn (guest), AU government boards, and more.
- **Cross-source deduplication** — the same role posted to Greenhouse and LinkedIn resolves to a single record, with both apply links preserved.
- **Local-first storage** — SQLite with full-text search; no external services required.
- **Tiered fetching** — plain HTTP, headless browser, and stealth browser, picked per source.
- **Resilient orchestration** — per-source schedules, automatic retry/backoff, schema-change detection, in-app notifications when something needs attention.
- **HTTP API** — clean REST surface for programmatic access and consumption by the AI layer.
- **Conversational AI** *(sprint 2)* — natural-language search, fit scoring, resume tailoring, cover-letter drafting.

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/duketis/jobai.git
cd jobai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
playwright install chromium
```

## Quick start

```bash
jobai migrate              # create the SQLite DB
jobai run --source greenhouse:atlassian   # one-shot scrape of one source
jobai serve                # start the scheduler + HTTP API on :8421
```

Then in another terminal:

```bash
curl 'http://localhost:8421/api/jobs?q=python&remote=true&limit=20'
curl 'http://localhost:8421/api/health'
```

The OpenAPI spec lives at `http://localhost:8421/docs`.

## CLI

```
jobai migrate                       Apply pending DB migrations.
jobai serve                         Start the scheduler + HTTP API.
jobai run --source <name>           Run a single source once, ad hoc.
jobai source list                   List configured sources and last-success times.
jobai source enable <name>          Re-enable a disabled source.
jobai source disable <name>         Disable a source (kill switch).
jobai health                        Print a health summary.
jobai notifications                 List unread notifications.
```

## HTTP API

| Method | Path | Description |
|---|---|---|
| GET | `/api/jobs` | Search and filter jobs (q, location, remote, posted_since, source, limit, offset). |
| GET | `/api/jobs/{id}` | Full job detail. |
| POST | `/api/jobs/{id}/state` | Mark saved / applied / dismissed. |
| GET | `/api/sources` | Source registry with health per source. |
| GET | `/api/health` | Aggregate system health. |
| GET | `/api/notifications` | In-app notifications. |
| POST | `/api/notifications/{id}/read` | Mark notification read. |

## Development

```bash
pytest                  # run all tests
pytest --cov            # with coverage
mypy                    # strict type checking
ruff check .            # lint
ruff format .           # format
pre-commit run --all-files
```

CI runs ruff, mypy, and pytest on every push and pull request. Pre-commit hooks enforce the same checks locally before you commit.

## Project layout

```
jobai/                  # source package
├── api/                # FastAPI server and routes
├── cli.py              # Typer CLI
├── db/                 # SQLite schema and migrations
├── dedup/              # deterministic + fuzzy job deduplication
├── fetcher/            # HTTP / browser / stealth fetchers
├── notifications/      # in-app notifications service
├── observability/      # structured logging
├── parsers/            # raw-to-canonical job normalization
├── pipeline/           # orchestration: scheduler, runner, cleanup
└── sources/            # one module per source family

tests/
├── unit/
├── integration/
└── fixtures/           # captured responses used by parser tests
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built by [Jonathan Duketis](https://github.com/duketis). Code generation and review assisted by [Claude](https://claude.ai) (Anthropic).
