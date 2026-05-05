"""Job sources.

Each source ingests jobs from one provider (an ATS family like
Greenhouse, or a single hostile site like Seek). All sources share the
:class:`~jobai.sources.base.BaseSource` ABC and the
:class:`~jobai.sources.base.NormalizedJob` shape, so the runner,
scheduler, and dedup logic do not care which source they are running.
"""

from __future__ import annotations
