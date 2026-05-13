"""Client + types for the shared user-context pool.

The single source of truth for the context pool (snippets, files,
project audits) is the :mod:`resumeai` sibling service. jobai exposes
a UI page that proxies through to it so the user manages their
portfolio data in the same app they use to discover jobs and kick
tailor chains -- they don't need to bounce between two ports.

This package contains:

* :class:`ContextClient` Protocol -- the wire surface the routes
  depend on.
* :class:`HttpxContextClient` -- the production httpx implementation.
* :class:`ContextFile` -- the typed record returned by the list /
  get endpoints (mirrors resumeai's response shape exactly so the
  UI can render the same fields).
"""

from __future__ import annotations
