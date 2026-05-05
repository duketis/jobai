"""HTTP API: the read surface every consumer talks to.

The data layer's one and only contract with the rest of the system.
Wraps SQLite in typed Pydantic responses behind FastAPI's auto-
generated OpenAPI docs at ``/docs``.

The AI/agent layer (and any future frontend) consumes this API; the
data layer never imports its consumers. That boundary is what lets
the data layer evolve (schema changes, new sources, dedup tuning)
without coordinating releases with everything that reads from it.
"""

from __future__ import annotations
