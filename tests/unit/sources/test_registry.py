"""Tests for the source registry."""

from __future__ import annotations

import pytest

from jobai.sources.greenhouse import GreenhouseSource
from jobai.sources.registry import (
    UnknownSourceKindError,
    get_source_class,
    known_kinds,
)


def test_get_source_class_returns_greenhouse() -> None:
    assert get_source_class("greenhouse") is GreenhouseSource


def test_get_source_class_raises_for_unknown_kind() -> None:
    with pytest.raises(UnknownSourceKindError) as excinfo:
        get_source_class("not-a-real-ats")
    assert excinfo.value.kind == "not-a-real-ats"
    assert "not-a-real-ats" in str(excinfo.value)


def test_known_kinds_includes_greenhouse_and_is_sorted() -> None:
    kinds = known_kinds()
    assert "greenhouse" in kinds
    assert kinds == sorted(kinds)


def test_unknown_source_kind_error_is_a_key_error() -> None:
    """For ``except KeyError`` clauses to catch this naturally."""
    err = UnknownSourceKindError("x")
    assert isinstance(err, KeyError)
