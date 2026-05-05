"""Tests for the rapidfuzz-backed title-similarity helpers."""

from __future__ import annotations

import pytest

from jobai.dedup.fuzzy import (
    DEFAULT_SIMILARITY_THRESHOLD,
    find_similar_match,
    is_similar_title,
    title_similarity,
)

# ---------------------------------------------------------------------------
# title_similarity
# ---------------------------------------------------------------------------


def test_identical_titles_score_100() -> None:
    assert title_similarity("Software Engineer", "Software Engineer") == 100


def test_case_difference_does_not_reduce_score() -> None:
    assert title_similarity("Software Engineer", "SOFTWARE ENGINEER") == 100


def test_punctuation_only_difference_scores_high() -> None:
    score = title_similarity("Engineering Manager, Platform", "Engineering Manager - Platform")
    assert score >= DEFAULT_SIMILARITY_THRESHOLD


def test_abbreviation_only_difference_scores_high_enough() -> None:
    """The 'Sr.' <-> 'Senior' case the dedup pass cannot catch on its own.

    Token-sort scores depend on the surrounding words' lengths;
    "Backend" produces 88 while shorter words can dip below threshold.
    """
    score = title_similarity("Sr. Backend Engineer", "Senior Backend Engineer")
    assert score >= DEFAULT_SIMILARITY_THRESHOLD


def test_seniority_difference_scores_below_threshold() -> None:
    """The threshold must NOT merge 'Senior' with 'Staff' — different roles."""
    score = title_similarity("Senior Software Engineer", "Staff Software Engineer")
    assert score < DEFAULT_SIMILARITY_THRESHOLD


def test_empty_inputs_score_zero() -> None:
    assert title_similarity("", "Engineer") == 0
    assert title_similarity("Engineer", "") == 0
    assert title_similarity("", "") == 0


# ---------------------------------------------------------------------------
# is_similar_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Software Engineer", "Software Engineer", True),
        ("Sr. Backend Engineer", "Senior Backend Engineer", True),
        ("Senior Software Engineer", "Staff Software Engineer", False),
        ("Designer", "Software Engineer", False),
        ("", "Engineer", False),
    ],
)
def test_is_similar_title(a: str, b: str, expected: bool) -> None:
    assert is_similar_title(a, b) is expected


def test_is_similar_title_respects_custom_threshold() -> None:
    """A stricter threshold is allowed to reject a borderline match."""
    a = "Engineering Manager, Platform"
    b = "Engineering Manager - Platform"
    assert is_similar_title(a, b, threshold=80) is True
    # Push threshold above the score; same pair must reject.
    assert is_similar_title(a, b, threshold=99) is False


# ---------------------------------------------------------------------------
# find_similar_match
# ---------------------------------------------------------------------------


def test_find_similar_match_returns_best_above_threshold() -> None:
    """Among unrelated candidates, the abbreviation-equivalent match wins."""
    candidates = [
        (1, "Designer"),
        (2, "Senior Backend Engineer"),
        (3, "Marketing Manager"),
    ]
    result = find_similar_match("Sr. Backend Engineer", candidates)
    assert result is not None
    matched_id, score = result
    assert matched_id == 2
    assert score >= DEFAULT_SIMILARITY_THRESHOLD


def test_find_similar_match_returns_none_when_nothing_qualifies() -> None:
    candidates = [
        (1, "Designer"),
        (2, "Marketing Manager"),
    ]
    assert find_similar_match("Senior Software Engineer", candidates) is None


def test_find_similar_match_handles_empty_candidates() -> None:
    assert find_similar_match("Engineer", []) is None


def test_find_similar_match_handles_empty_target() -> None:
    candidates = [(1, "Engineer")]
    assert find_similar_match("", candidates) is None


def test_find_similar_match_picks_highest_score_when_multiple_qualify() -> None:
    """When two candidates pass the threshold, the higher-scoring one wins."""
    candidates = [
        (1, "Engineering Manager - Platform"),  # punct-only diff, ~95
        (2, "Engineering Manager, Platform"),  # exact match -> 100
    ]
    result = find_similar_match("Engineering Manager, Platform", candidates)
    assert result is not None
    matched_id, score = result
    assert matched_id == 2
    assert score == 100
