"""Tests for the deterministic dedup-key computation."""

from __future__ import annotations

import pytest

from jobai.dedup.hashing import (
    compute_dedup_key,
    normalize_company,
    normalize_country,
    normalize_title,
)

# ---------------------------------------------------------------------------
# normalize_company
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Atlassian", "atlassian"),
        ("ATLASSIAN", "atlassian"),
        ("Atlassian Pty Ltd", "atlassian"),
        ("Atlassian, Inc.", "atlassian"),
        ("Atlassian Inc", "atlassian"),
        ("Atlassian Holdings", "atlassian"),
        ("  Atlassian  ", "atlassian"),
        ("Café Pty Ltd", "cafe"),
        ("Müller GmbH", "muller"),
        ("Smith & Co.", "smith"),
        ("ACME Technologies", "acme"),
        ("Foo Holdings Ltd", "foo"),
        ("", ""),
    ],
)
def test_normalize_company_strips_known_variations(raw: str, expected: str) -> None:
    assert normalize_company(raw) == expected


def test_normalize_company_preserves_distinctive_words() -> None:
    """Words that distinguish two companies must survive normalisation."""
    assert normalize_company("Acme Health") != normalize_company("Acme Foods")


# ---------------------------------------------------------------------------
# normalize_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Software Engineer", "software engineer"),
        ("Senior  Software   Engineer", "senior software engineer"),
        ("Software Engineer (Backend)", "software engineer backend"),
        ("Front-end Developer", "front-end developer"),
        ("Café Manager", "cafe manager"),
        ("  Engineer  ", "engineer"),
        ("", ""),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


def test_normalize_title_preserves_seniority_words() -> None:
    """Senior / Staff / Principal must NOT collapse into the same string."""
    senior = normalize_title("Senior Software Engineer")
    staff = normalize_title("Staff Software Engineer")
    principal = normalize_title("Principal Software Engineer")
    assert senior != staff
    assert staff != principal


# ---------------------------------------------------------------------------
# normalize_country
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Australia", "australia"),
        ("AU", "au"),
        ("  united states ", "united states"),
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_country(raw: str | None, expected: str) -> None:
    assert normalize_country(raw) == expected


# ---------------------------------------------------------------------------
# compute_dedup_key
# ---------------------------------------------------------------------------


def test_dedup_key_is_deterministic() -> None:
    a = compute_dedup_key(company="Atlassian", title="Engineer", country="AU")
    b = compute_dedup_key(company="Atlassian", title="Engineer", country="AU")
    assert a == b


def test_dedup_key_collapses_company_suffix_variants() -> None:
    """Atlassian Pty Ltd / Atlassian Inc / Atlassian — same key."""
    base = compute_dedup_key(company="Atlassian", title="Engineer", country="AU")
    assert compute_dedup_key(company="Atlassian Pty Ltd", title="Engineer", country="AU") == base
    assert compute_dedup_key(company="Atlassian, Inc.", title="Engineer", country="AU") == base
    assert compute_dedup_key(company="ATLASSIAN", title="Engineer", country="AU") == base


def test_dedup_key_collapses_punctuation_and_whitespace_in_title() -> None:
    base = compute_dedup_key(company="X", title="Software Engineer", country="AU")
    assert compute_dedup_key(company="X", title="Software  Engineer", country="AU") == base
    assert (
        compute_dedup_key(company="X", title="Software-Engineer", country="AU")
        != base  # hyphen IS preserved (different role connotation)
    )


def test_dedup_key_collapses_country_case() -> None:
    a = compute_dedup_key(company="X", title="Y", country="Australia")
    b = compute_dedup_key(company="X", title="Y", country="AUSTRALIA")
    assert a == b


def test_dedup_key_distinguishes_different_jobs() -> None:
    a = compute_dedup_key(company="X", title="Engineer", country="AU")
    b = compute_dedup_key(company="X", title="Engineer", country="US")
    c = compute_dedup_key(company="X", title="Designer", country="AU")
    d = compute_dedup_key(company="Y", title="Engineer", country="AU")
    assert len({a, b, c, d}) == 4


def test_dedup_key_handles_none_country() -> None:
    a = compute_dedup_key(company="X", title="Y", country=None)
    b = compute_dedup_key(company="X", title="Y", country="")
    assert a == b


def test_dedup_key_returns_64_char_hex() -> None:
    key = compute_dedup_key(company="X", title="Y", country="AU")
    assert len(key) == 64
    int(key, 16)  # raises if non-hex
