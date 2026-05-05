"""Source registry: maps a source ``kind`` string to a concrete class.

The registry exists so the runner and CLI can construct sources from
DB-driven configuration without importing every concrete class
themselves. Adding a new source family means adding one line here.

The runtime instance is a regular dict keyed by ``kind`` (the string
that appears in the ``sources.kind`` column and in the CLI argument
form ``kind:account``). Lookup raises :class:`UnknownSourceKindError`
with a helpful message listing the registered kinds.
"""

from __future__ import annotations

from collections.abc import Mapping

from jobai.sources.ashby import AshbySource
from jobai.sources.base import BaseSource
from jobai.sources.greenhouse import GreenhouseSource
from jobai.sources.lever import LeverSource

_REGISTRY: Mapping[str, type[BaseSource]] = {
    "ashby": AshbySource,
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
}


def get_source_class(kind: str) -> type[BaseSource]:
    """Return the concrete source class registered for ``kind``."""
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise UnknownSourceKindError(kind) from exc


def known_kinds() -> list[str]:
    """List every registered source kind in lexicographic order."""
    return sorted(_REGISTRY.keys())


class UnknownSourceKindError(KeyError):
    """Raised when ``get_source_class`` is called with a kind not in the registry."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind

    def __str__(self) -> str:
        return f"unknown source kind {self.kind!r}; known: {known_kinds()}"
