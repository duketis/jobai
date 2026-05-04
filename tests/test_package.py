"""Smoke tests for the top-level package.

These run on every CI pipeline as the cheapest signal that the install is
sound and the package metadata is intact. They should never need updating
unless the package itself is restructured.
"""

from __future__ import annotations

import re

import jobai


def test_package_imports() -> None:
    """The package must import cleanly with no side effects."""
    assert jobai is not None


def test_version_is_set() -> None:
    """``__version__`` must be present and follow semver-ish formatting."""
    assert hasattr(jobai, "__version__")
    assert isinstance(jobai.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", jobai.__version__), (
        f"unexpected version format: {jobai.__version__!r}"
    )
