"""Pipeline orchestration: scrape runs, raw cleanup, source scheduling.

The runner module is the top-level entry point: given a configured
source row + a fetcher, it executes one scrape cycle (fetch, record
raw, parse, upsert jobs_raw) and finalises the scrape_runs row. The
scheduler and cleanup modules land in later phases.
"""

from __future__ import annotations
