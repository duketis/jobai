"""Tests for :class:`BaseSource` and :class:`NormalizedJob`."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob


def test_normalized_job_minimum_fields() -> None:
    """The four required fields are sufficient to construct a job; everything
    else is optional."""
    job = NormalizedJob(
        source_external_id="42",
        title="Senior Engineer",
        company="Atlassian",
        apply_url="https://example.com/jobs/42",
        raw_data={"id": 42},
    )
    assert job.source_external_id == "42"
    assert job.location_raw is None
    assert job.salary_min is None
    assert job.extra_tags == ()


def test_normalized_job_is_immutable() -> None:
    job = NormalizedJob(
        source_external_id="1",
        title="Engineer",
        company="X",
        apply_url="https://example.com",
        raw_data={},
    )
    with pytest.raises((AttributeError, Exception)):
        job.title = "different"  # type: ignore[misc]


def test_base_source_name_combines_kind_and_account() -> None:
    class _Greenhouse(BaseSource):
        kind = "greenhouse"

        def __init__(self, board: str) -> None:
            self.account = board

        def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
            del fetcher
            raise NotImplementedError

    source = _Greenhouse(board="atlassian")
    assert source.name == "greenhouse:atlassian"


def test_base_source_name_falls_back_to_kind_when_account_empty() -> None:
    class _HackerNews(BaseSource):
        kind = "hackernews"
        account = ""

        def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
            del fetcher
            raise NotImplementedError

    source = _HackerNews()
    assert source.name == "hackernews"


def test_base_source_cannot_be_instantiated_without_discover() -> None:
    class _Incomplete(BaseSource):
        kind = "broken"
        account = "x"

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()  # type: ignore[abstract]


def test_base_source_needs_persistent_session_defaults_to_false() -> None:
    """Most sources don't need session persistence - the runner should
    build per-fetch contexts (cheap, isolated) by default."""

    class _Boring(BaseSource):
        kind = "boring"
        account = ""

        def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
            del fetcher
            raise NotImplementedError

    assert _Boring.needs_persistent_session is False
    assert _Boring().needs_persistent_session is False


def test_base_source_subclass_can_opt_in_to_persistent_session() -> None:
    """Cloudflare-protected sources opt in via the class attribute so
    the runner picks the long-lived browser context tier when
    constructing the fetcher."""

    class _CloudflareFronted(BaseSource):
        kind = "cf_fronted"
        account = ""
        needs_persistent_session = True

        def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
            del fetcher
            raise NotImplementedError

    assert _CloudflareFronted.needs_persistent_session is True
    assert _CloudflareFronted().needs_persistent_session is True
