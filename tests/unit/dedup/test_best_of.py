"""Tests for the best-of field merger.

The merger is the heart of cross-source data quality: when the same job
turns up on Greenhouse (full description, salary listed) and Indeed
(truncated teaser, no salary), the canonical row should reflect the
**best** information across both — not whichever scraped most recently.

Each rule below corresponds to a real-world failure mode in the
pre-merger ``COALESCE(?, existing)`` behaviour:

* ``first_non_null`` — preserve original salary; later sources can't
  silently override it (a scraper bug emitting None must not erase data)
* ``first_non_empty`` — same, but treats ``""`` as missing (some sources
  emit empty strings rather than null)
* ``longest`` — pick the richer description / location string
* ``earliest`` — preserve the original ``posted_at`` timestamp
"""

from __future__ import annotations

from jobai.dedup.best_of import (
    earliest,
    first_non_empty,
    first_non_null,
    longest,
    merge_canonical_fields,
    mergeable_fields,
)

# ---------------------------------------------------------------------------
# first_non_null
# ---------------------------------------------------------------------------


def test_first_non_null_returns_none_for_empty_iterable() -> None:
    assert first_non_null([]) is None


def test_first_non_null_returns_none_when_all_inputs_are_none() -> None:
    assert first_non_null([None, None, None]) is None


def test_first_non_null_returns_first_non_none_value() -> None:
    assert first_non_null([None, "a", "b"]) == "a"


def test_first_non_null_treats_zero_and_false_as_present() -> None:
    """``0`` and ``False`` are valid values, not absences. The rule is
    'first non-None', not 'first truthy'."""
    assert first_non_null([None, 0, 1]) == 0
    assert first_non_null([None, False, True]) is False


def test_first_non_null_preserves_old_when_old_is_set() -> None:
    """The merge contract: old wins when set; new only fills in nulls."""
    assert first_non_null([100_000, 130_000]) == 100_000


# ---------------------------------------------------------------------------
# first_non_empty
# ---------------------------------------------------------------------------


def test_first_non_empty_treats_empty_string_as_missing() -> None:
    assert first_non_empty(["", "real value"]) == "real value"


def test_first_non_empty_treats_none_as_missing() -> None:
    assert first_non_empty([None, "real value"]) == "real value"


def test_first_non_empty_returns_none_for_all_missing() -> None:
    assert first_non_empty(["", None, ""]) is None


def test_first_non_empty_returns_none_for_empty_iterable() -> None:
    assert first_non_empty([]) is None


# ---------------------------------------------------------------------------
# longest
# ---------------------------------------------------------------------------


def test_longest_returns_the_longest_string() -> None:
    assert longest(["short", "much longer description"]) == "much longer description"


def test_longest_ignores_none_and_empty() -> None:
    assert longest([None, "", "real"]) == "real"


def test_longest_returns_none_when_no_candidates_qualify() -> None:
    assert longest([None, "", None]) is None


def test_longest_returns_none_for_empty_iterable() -> None:
    assert longest([]) is None


def test_longest_breaks_ties_by_keeping_first_occurrence() -> None:
    """Stable tie-break: the *first* equal-length value wins so repeated
    runs don't flip the canonical value back and forth."""
    assert longest(["abc", "xyz"]) == "abc"


# ---------------------------------------------------------------------------
# earliest
# ---------------------------------------------------------------------------


def test_earliest_returns_lexicographically_smallest() -> None:
    """ISO-8601 timestamps sort lexically the same as chronologically — we
    don't need to parse them to compare."""
    assert (
        earliest(["2026-05-07T05:00:00+00:00", "2026-05-01T12:00:00+00:00"])
        == "2026-05-01T12:00:00+00:00"
    )


def test_earliest_ignores_none() -> None:
    assert earliest([None, "2026-05-01T00:00:00+00:00"]) == "2026-05-01T00:00:00+00:00"


def test_earliest_ignores_empty_string() -> None:
    assert earliest(["", "2026-05-01T00:00:00+00:00"]) == "2026-05-01T00:00:00+00:00"


def test_earliest_returns_none_when_no_candidates() -> None:
    assert earliest([None, "", None]) is None


def test_earliest_returns_none_for_empty_iterable() -> None:
    assert earliest([]) is None


# ---------------------------------------------------------------------------
# merge_canonical_fields — the orchestrator
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object | None]:
    """Build a complete canonical-job field dict with sensible defaults
    so each test can override only the fields it cares about."""
    base: dict[str, object | None] = {
        "location_raw": None,
        "location_country": None,
        "location_city": None,
        "remote_type": None,
        "employment_type": None,
        "posted_at": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "description_text": None,
        "description_html": None,
        "apply_url": None,
    }
    base.update(overrides)
    return base


def test_merge_keeps_existing_salary_when_new_scrape_has_none() -> None:
    """Greenhouse first scrape: salary $150k. Indeed scrape later: no salary.
    Result: $150k preserved (the COALESCE behaviour, retained)."""
    old = _row(salary_min=150_000, salary_max=180_000, salary_currency="AUD")
    new = _row()
    merged = merge_canonical_fields(old, new)
    assert merged["salary_min"] == 150_000
    assert merged["salary_max"] == 180_000
    assert merged["salary_currency"] == "AUD"


def test_merge_fills_in_salary_when_existing_is_null() -> None:
    """Indeed first scrape: no salary. Greenhouse later: salary $150k.
    Result: $150k captured."""
    old = _row()
    new = _row(salary_min=150_000, salary_max=180_000, salary_currency="AUD")
    merged = merge_canonical_fields(old, new)
    assert merged["salary_min"] == 150_000
    assert merged["salary_max"] == 180_000
    assert merged["salary_currency"] == "AUD"


def test_merge_does_not_override_existing_salary_with_different_value() -> None:
    """Source A says $150k, source B says $130k two days later.
    Don't second-guess: preserve the original (first non-null) value."""
    old = _row(salary_min=150_000, salary_max=180_000, salary_currency="AUD")
    new = _row(salary_min=130_000, salary_max=160_000, salary_currency="USD")
    merged = merge_canonical_fields(old, new)
    assert merged["salary_min"] == 150_000
    assert merged["salary_max"] == 180_000
    assert merged["salary_currency"] == "AUD"


def test_merge_picks_longer_description_text() -> None:
    """Indeed has a 200-char teaser, Greenhouse has a full multi-paragraph
    description. The full one wins regardless of scrape order."""
    short = "Senior Python role at Atlassian. Apply now."
    long = "Senior Python Engineer at Atlassian. " * 30
    old = _row(description_text=short)
    new = _row(description_text=long)
    merged = merge_canonical_fields(old, new)
    assert merged["description_text"] == long


def test_merge_picks_longer_description_text_when_old_is_longer() -> None:
    """Inverse direction: the existing canonical row already has the rich
    description; a leaner re-scrape must not truncate it."""
    short = "Senior Python role at Atlassian. Apply now."
    long = "Senior Python Engineer at Atlassian. " * 30
    old = _row(description_text=long)
    new = _row(description_text=short)
    merged = merge_canonical_fields(old, new)
    assert merged["description_text"] == long


def test_merge_picks_longer_description_html_independently() -> None:
    """``description_html`` and ``description_text`` are merged independently
    so a source that emits only one of them doesn't blank the other."""
    old = _row(description_text="long " * 50, description_html=None)
    new = _row(description_text="short", description_html="<p>html short</p>")
    merged = merge_canonical_fields(old, new)
    assert merged["description_text"] == "long " * 50
    assert merged["description_html"] == "<p>html short</p>"


def test_merge_keeps_earliest_posted_at() -> None:
    """If Source A says posted 2026-05-01 and Source B says 2026-05-07,
    the original 2026-05-01 wins — boards sometimes backdate or re-list
    a role with a fresher timestamp."""
    old = _row(posted_at="2026-05-01T00:00:00+00:00")
    new = _row(posted_at="2026-05-07T00:00:00+00:00")
    merged = merge_canonical_fields(old, new)
    assert merged["posted_at"] == "2026-05-01T00:00:00+00:00"


def test_merge_uses_new_posted_at_when_old_is_null() -> None:
    old = _row()
    new = _row(posted_at="2026-05-07T00:00:00+00:00")
    merged = merge_canonical_fields(old, new)
    assert merged["posted_at"] == "2026-05-07T00:00:00+00:00"


def test_merge_picks_earlier_posted_at_when_new_is_older() -> None:
    """A late-arriving source with the original posting date should
    correct an earlier-recorded timestamp that was actually a re-list."""
    old = _row(posted_at="2026-05-07T00:00:00+00:00")
    new = _row(posted_at="2026-05-01T00:00:00+00:00")
    merged = merge_canonical_fields(old, new)
    assert merged["posted_at"] == "2026-05-01T00:00:00+00:00"


def test_merge_keeps_richer_location_raw() -> None:
    """``"Sydney"`` vs ``"Sydney NSW 2000, Australia"`` — keep the richer one."""
    old = _row(location_raw="Sydney")
    new = _row(location_raw="Sydney NSW 2000, Australia")
    merged = merge_canonical_fields(old, new)
    assert merged["location_raw"] == "Sydney NSW 2000, Australia"


def test_merge_preserves_first_non_null_for_categorical_fields() -> None:
    """For fields where 'longer is better' doesn't apply (remote_type,
    employment_type, location_country/city), keep the first non-null
    rather than letting subsequent scrapes flip them."""
    old = _row(
        location_country="AU",
        location_city="Sydney",
        remote_type="hybrid",
        employment_type="full_time",
    )
    new = _row(
        location_country="US",
        location_city="Melbourne",
        remote_type="remote",
        employment_type="contract",
    )
    merged = merge_canonical_fields(old, new)
    assert merged["location_country"] == "AU"
    assert merged["location_city"] == "Sydney"
    assert merged["remote_type"] == "hybrid"
    assert merged["employment_type"] == "full_time"


def test_merge_fills_in_categorical_when_old_is_null() -> None:
    old = _row()
    new = _row(
        location_country="AU",
        location_city="Sydney",
        remote_type="hybrid",
        employment_type="full_time",
    )
    merged = merge_canonical_fields(old, new)
    assert merged["location_country"] == "AU"
    assert merged["location_city"] == "Sydney"
    assert merged["remote_type"] == "hybrid"
    assert merged["employment_type"] == "full_time"


def test_merge_keeps_existing_apply_url_when_new_is_empty() -> None:
    """``apply_url`` is required on a NormalizedJob (sources can't post
    without one), but a re-scrape of the same role might emit ``""`` if
    the URL field is gone from the listing — don't blank our copy."""
    old = _row(apply_url="https://boards.greenhouse.io/atlassian/jobs/123")
    new = _row(apply_url="")
    merged = merge_canonical_fields(old, new)
    assert merged["apply_url"] == "https://boards.greenhouse.io/atlassian/jobs/123"


def test_merge_returns_only_the_mergeable_fields() -> None:
    """The output is restricted to the fields the merger owns. Caller
    decides what to do with id / dedup_key / first_seen_at / last_seen_at,
    which have their own update rules outside this helper."""
    old = _row(salary_min=100_000)
    new = _row()
    merged = merge_canonical_fields(old, new)
    expected_keys = {
        "location_raw",
        "location_country",
        "location_city",
        "remote_type",
        "employment_type",
        "posted_at",
        "salary_min",
        "salary_max",
        "salary_currency",
        "description_text",
        "description_html",
        "apply_url",
    }
    assert set(merged.keys()) == expected_keys


def test_mergeable_fields_matches_merge_output_keys() -> None:
    """Single source of truth: the public ``mergeable_fields`` tuple
    must match the keys ``merge_canonical_fields`` returns. promote.py
    and reconcile.py rely on this to build their UPDATE statements."""
    merged = merge_canonical_fields(_row(), _row())
    assert set(mergeable_fields()) == set(merged.keys())


def test_merge_coerces_non_string_values_through_str_helpers() -> None:
    """sqlite round-trips can hand us back a non-string for a string-typed
    column when the schema stores it in a different affinity or a custom
    converter is registered. The merger coerces defensively so the
    string-rule helpers (``longest``, ``earliest``, ``first_non_empty``)
    stay total functions on string input."""
    old: dict[str, object | None] = {"location_raw": 12345, "posted_at": None}
    new: dict[str, object | None] = {
        "location_raw": "Sydney NSW",
        "posted_at": "2026-05-01T00:00:00+00:00",
    }
    merged = merge_canonical_fields(old, new)
    # ``"Sydney NSW"`` is longer than ``"12345"`` so it wins anyway, but the
    # important assertion is that the merge didn't crash on a non-str.
    assert merged["location_raw"] == "Sydney NSW"
