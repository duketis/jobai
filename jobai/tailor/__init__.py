"""Tailor orchestration package.

Chains resumeai + coverletterai for a given canonical job:

  1. POST http://resumeai:8765/api/tailor  -> resume_run_id
  2. poll  http://resumeai:8765/api/runs/{id} until terminal
  3. POST http://coverletterai:8766/api/tailor with resume_run_id -> letter_run_id
  4. poll  http://coverletterai:8766/api/runs/{id} until terminal
  5. persist artefacts so the UI can stream both PDFs

The package is split so the moving parts are independently testable:

* :mod:`jobai.tailor.models`        — Pydantic request / response shapes
* :mod:`jobai.tailor.client`        — Protocol + httpx clients for both siblings
* :mod:`jobai.tailor.repository`    — SQL CRUD for ``tailor_runs``
* :mod:`jobai.tailor.orchestrator`  — the chain itself, expressed as a coroutine
* :mod:`jobai.tailor.worker`        — concurrency-capped pool that submits chains
"""

from __future__ import annotations
