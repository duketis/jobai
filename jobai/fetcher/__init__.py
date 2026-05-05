"""HTTP fetching abstraction.

Three tiers (HTTP, browser, stealth) all implement the same
:class:`~jobai.fetcher.base.Fetcher` Protocol so callers do not know how a
response was obtained. ``base`` defines the contract; ``http`` provides
the tier-1 implementation. Browser and stealth tiers land in later
phases.
"""

from __future__ import annotations
