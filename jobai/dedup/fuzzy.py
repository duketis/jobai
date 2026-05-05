"""Fuzzy title-similarity helpers backed by :mod:`rapidfuzz`.

The deterministic dedup key (``jobai.dedup.hashing``) catches cross-
source duplicates that share an exact (company, title, country)
tuple. The fuzzy pass catches the residual:

* "Sr. Backend Engineer" vs "Senior Backend Engineer"  -> match
* "Engineering Manager, Platform" vs "Engineering Manager - Platform"  -> match
* "Senior Software Engineer" vs "Staff Software Engineer"  -> no match

**Metric choice.** We use ``token_sort_ratio`` rather than
``token_set_ratio``. ``token_set_ratio`` returns 100 whenever one
string's tokens are a subset of the other's (e.g. "Backend Engineer"
vs "Senior Backend Engineer" scores 100), which over-merges roles
that genuinely differ in seniority. ``token_sort_ratio`` sorts tokens
then runs a single Levenshtein, penalising genuine token-count
differences more honestly.

**Threshold 85** is calibrated against rapidfuzz's empirical scores
for our target cases:

  Sr. Backend Engineer    vs Senior Backend Engineer  -> 88   ✓ match
  Senior Software Eng.    vs Staff Software Eng.      -> 80   ✗ no match
  Senior Backend Eng.     vs Backend Engineer         -> 82   ✗ no match
  Eng. Mgr, Platform      vs Eng. Mgr - Platform      -> 95   ✓ match

**Known asymmetric edge case.** Because "Sr." is short, it scores 89
against bare "Backend Engineer" — a false-positive merge in the
abbreviated form. This is acceptable: in practice, a company doesn't
post the same role both with and without seniority, and when it
does, treating them as the same role is usually correct. A future
abbreviation-expansion pre-pass would close this gap if it ever
matters.
"""

from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz

#: Minimum ``token_sort_ratio`` for two titles to be considered the same role.
DEFAULT_SIMILARITY_THRESHOLD = 85


def title_similarity(title_a: str, title_b: str) -> int:
    """Return the rapidfuzz token-sort ratio (0-100) between two titles.

    Both inputs are lower-cased before comparison so trivial casing
    differences don't reduce the score. Empty strings score 0.
    """
    if not title_a or not title_b:
        return 0
    return int(fuzz.token_sort_ratio(title_a.lower(), title_b.lower()))


def is_similar_title(
    title_a: str,
    title_b: str,
    *,
    threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """Return True if two titles meet ``threshold``."""
    return title_similarity(title_a, title_b) >= threshold


def find_similar_match(
    target_title: str,
    candidates: Iterable[tuple[int, str]],
    *,
    threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[int, int] | None:
    """Find the best match for ``target_title`` among ``(id, title)`` pairs.

    Args:
        target_title: the title we're trying to find a duplicate for.
        candidates: an iterable of ``(id, title)`` pairs (typically
            jobs from the same company/country group).
        threshold: minimum token-set ratio for a candidate to qualify.

    Returns:
        ``(id, score)`` of the best-scoring candidate above
        ``threshold``, or ``None`` if no candidate clears the bar.
    """
    if not target_title:
        return None

    best_id: int | None = None
    best_score = threshold - 1
    for cand_id, cand_title in candidates:
        score = title_similarity(target_title, cand_title)
        if score > best_score:
            best_score = score
            best_id = cand_id

    if best_id is None:
        return None
    return (best_id, best_score)
