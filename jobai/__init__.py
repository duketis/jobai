"""jobai — local-first AI job-hunting agent.

The data layer ingests job listings from many sources (ATS aggregators, public
job boards, hostile-tier sites), deduplicates them across providers, and
exposes them through an HTTP API consumed by the AI/agent layer.

Public surface is intentionally minimal at the package level; consumers should
import from submodules (``jobai.db``, ``jobai.fetcher``, ``jobai.api``) rather
than re-exporting through here.
"""

from __future__ import annotations

__version__ = "1.13.0"
__all__ = ["__version__"]
