"""Coverage for ``jobai.tailor.layout_check``.

The pure-function path is tested with fake page contents; the PDF
wrapper goes through real bytes via ``pypdf``.
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from jobai.tailor.layout_check import (
    check_layout_from_pages,
    check_pdf_layout,
)


def _make_minimal_pdf(num_pages: int = 1) -> bytes:
    """Mint a tiny in-memory PDF -- enough bytes for pypdf to parse,
    with no text content (so the heuristics see empty pages)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=300, height=300)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pure layout check
# ---------------------------------------------------------------------------


def test_check_layout_returns_empty_for_clean_single_page() -> None:
    """A single page with a section + bullets all on the same page
    raises no issues -- nothing to be orphaned by a page break."""
    pages = [
        [
            "Profile",
            "• alpha",
            "• beta",
            "• gamma",
        ],
    ]
    assert check_layout_from_pages(pages) == []


def test_check_layout_flags_orphan_leading_bullets() -> None:
    """Page 2 opens with a bullet then a section header -- that bullet
    is stranded from the section that closed page 1."""
    pages = [
        ["Profile", "• alpha", "• beta", "Key Achievements"],
        [
            "• orphan bullet from Key Achievements",
            "Professional Experience",
            "DiUS — Premium Valet",
            "• job bullet",
        ],
    ]
    issues = check_layout_from_pages(pages)
    summaries = [i.summary for i in issues]
    assert any("page 2 starts with 1 bullet" in s for s in summaries)
    assert all(i.severity == "must_fix" for i in issues)
    assert all(i.category == "format" for i in issues)


def test_check_layout_flags_split_section_header() -> None:
    """Page 1 ends with 'DiUS — WhichCar' as the last line; page 2
    opens with bullets. The role header is split from its body."""
    pages = [
        [
            "Profile",
            "• summary bullet",
            "Professional Experience",
            "DiUS — Premium Valet",
            "• premium bullet",
            "DiUS — WhichCar (Are Media)",
        ],
        [
            "• Built Node.js ingestion services",
            "• Implemented image streaming",
            "DiUS — Sørgenfrey",
            "• sorgenfrey bullet",
        ],
    ]
    issues = check_layout_from_pages(pages)
    summaries = [i.summary for i in issues]
    assert any("DiUS — WhichCar" in s for s in summaries)
    assert any("bottom" in s.lower() for s in summaries)


def test_check_layout_does_not_flag_continuation_when_no_header_follows() -> None:
    """If a page starts with bullets but no section header follows
    immediately, those bullets are just the legitimate continuation
    of the previous section, NOT orphans."""
    pages = [
        ["Section A", "• one", "• two", "• three"],
        ["• four (continuation)", "• five", "• six", "• seven"],
    ]
    assert check_layout_from_pages(pages) == []


def test_check_layout_ignores_dates_and_locations_as_orphan_headers() -> None:
    """Renderer puts dates ('2024 — 2025') and locations
    ('Melbourne, Australia') on their own lines. Those shouldn't
    trip the split-header heuristic when they happen to be the
    last meaningful line of a page."""
    pages = [
        [
            "Section A",
            "• one",
            "2024 — 2025",
            "Melbourne, Australia",
        ],
        ["• continuation bullet", "• another"],
    ]
    # Neither orphan-bullet nor split-header fires:
    #  - bullets follow but no SECTION HEADER comes after
    #  - last meaningful line on page 1 is a bullet (• one) once
    #    the date/location lines are filtered out
    assert check_layout_from_pages(pages) == []


def test_check_layout_skips_when_too_many_leading_bullets() -> None:
    """A page opening with many bullets is a normal section
    continuation; only 1-3 orphan-leading bullets count as a true
    stranded tail."""
    pages = [
        ["Section A", "• one"],
        [
            "• two",
            "• three",
            "• four",
            "• five",
            "Key Achievements",
            "• new section bullet",
        ],
    ]
    # 4 leading bullets > _MAX_ORPHAN_LEADING_BULLETS, so not flagged.
    assert check_layout_from_pages(pages) == []


def test_check_layout_uses_document_label_in_summaries() -> None:
    """``document_label`` shows up in the issue summary so the
    auto-fix prompt can target the right artefact."""
    pages = [
        ["Section A", "• one", "Key Achievements"],
        ["• orphan", "Professional Experience", "• new"],
    ]
    issues = check_layout_from_pages(pages, document_label="cover letter")
    assert all("cover letter" in i.summary for i in issues)


def test_check_layout_handles_empty_pages_list() -> None:
    assert check_layout_from_pages([]) == []
    assert check_layout_from_pages([["• alpha"]]) == []  # single page


def test_check_layout_skips_blank_lines_between_bullets_and_header() -> None:
    """The bullet-scanning loop must skip whitespace-only lines
    between the trailing bullets and the next section header so a
    visually orphan tail still gets flagged even when the renderer
    inserts vertical breathing room."""
    pages = [
        ["Section A", "• one"],
        [
            "• orphan",
            "   ",  # whitespace-only -- should be skipped, not treated as content
            "Key Achievements",
            "• new section",
        ],
    ]
    issues = check_layout_from_pages(pages)
    assert any("page 2 starts with 1 bullet" in i.summary for i in issues)


def test_is_meaningful_returns_false_for_empty_and_whitespace() -> None:
    """Direct exercise of the empty-line short-circuit (the orchestrator
    path filters most empties earlier, so this branch otherwise
    wouldn't be hit by integration tests)."""
    from jobai.tailor.layout_check import _is_meaningful  # noqa: PLC0415

    assert _is_meaningful("") is False
    assert _is_meaningful("   ") is False


def test_check_layout_skips_completely_blank_prev_page() -> None:
    """A truly blank previous page has no last-meaningful line, so
    neither heuristic triggers."""
    pages = [
        ["", "", ""],
        ["• alpha", "Profile", "• beta"],
    ]
    # Orphan-leading-bullets check runs against page 2, sees a
    # bullet + a section header, and DOES flag -- but only because
    # the start-of-page-2 pattern is the same regardless of page 1.
    issues = check_layout_from_pages(pages)
    # Should still flag the orphan (page-2 view is what matters).
    assert any("page 2 starts with" in i.summary for i in issues)


# ---------------------------------------------------------------------------
# PDF wrapper
# ---------------------------------------------------------------------------


def test_check_pdf_layout_returns_empty_for_empty_bytes() -> None:
    """An empty byte string short-circuits with no issues -- the
    upstream sibling fetch may legitimately return empty when a run
    is mid-render or its PDF artefact is missing."""
    assert check_pdf_layout(b"") == []


def test_check_pdf_layout_returns_empty_for_malformed_pdf() -> None:
    """Garbage bytes (truncated upload, network glitch) don't blow
    up the QA stage -- they degrade to 'no layout issues'."""
    assert check_pdf_layout(b"not a pdf, just some junk bytes") == []


def test_check_pdf_layout_runs_against_a_real_minimal_pdf() -> None:
    """A real (but blank) PDF parses cleanly; with no text content
    there are no layout issues to flag."""
    pdf = _make_minimal_pdf(num_pages=2)
    assert check_pdf_layout(pdf) == []


def test_check_pdf_layout_handles_pypdf_extract_text_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If pypdf raises on a page (corrupt page tree, etc), the
    extractor catches and yields an empty page -- the run-level
    heuristic still produces an empty issues list."""
    import jobai.tailor.layout_check as mod  # noqa: PLC0415

    class _BoomPage:
        def extract_text(self) -> str:
            msg = "broken"
            raise RuntimeError(msg)

    class _BoomReader:
        def __init__(self, _stream: object) -> None:
            self.pages = [_BoomPage(), _BoomPage()]

    monkeypatch.setattr(mod, "PdfReader", _BoomReader)
    # Real bytes don't matter -- the patched reader replaces pypdf.
    assert check_pdf_layout(b"%PDF-1.4 ...") == []


def test_check_pdf_layout_handles_reader_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If pypdf raises while CONSTRUCTING the reader (truly bad
    bytes that get past the empty-check), the outer try/except in
    check_pdf_layout catches and returns an empty issues list."""
    import jobai.tailor.layout_check as mod  # noqa: PLC0415

    def _boom(_stream: object) -> object:
        msg = "header missing"
        raise RuntimeError(msg)

    monkeypatch.setattr(mod, "PdfReader", _boom)
    assert check_pdf_layout(b"%PDF-...some garbage") == []
