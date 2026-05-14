"""Heuristic layout QA for rendered resume / cover-letter PDFs.

The cross-artefact QA agent (see :mod:`jobai.tailor.qa`) reads the
**structured** tailored JSON the siblings emit. Useful for "letter
contradicts resume" / "JD requirement missing" -- but blind to how
the PDF actually paginates. So things like a section header at the
bottom of page N with its bullets on page N+1, or a single trailing
bullet orphaned at the top of the next page, slipped through and the
user had to catch them by eye.

This module reads the rendered PDF and produces deterministic
:class:`QAIssue` objects for the layout problems jobai can detect
without rasterising / vision. The orchestrator merges these into
the LLM-graded assessment so the auto-fix loop sees layout as a
must-fix on the same footing as a content inconsistency.

What we detect (V1):

* **Orphan leading bullets** -- a page begins with one or two bullet
  lines and then a SECTION header. Those bullets clearly belong to
  the section on the previous page; the page break carved them off.
* **Split section header** -- a page ENDS with what looks like a
  section header (employer/role line, "Section Name" line) with no
  following content on the same page, and the next page opens with
  bullets. The header is divorced from its body.

What we deliberately DON'T detect (yet):

* Subtle whitespace drift / leading -- requires real font metrics.
* "Too long, fits but feels cramped" -- subjective; punt.
* Cover-letter pagination -- letters should always be 1 page; jobai
  already enforces that schema-side via coverletterai. If the letter
  is > 1 page that's a different fault, surfaced via page count.
"""

from __future__ import annotations

import io
import re
from typing import Final

from pypdf import PdfReader

from jobai.tailor.models import QAIssue

#: Bullet glyphs the renderer emits. Resumeai uses •; coverletterai
#: doesn't bullet (single-paragraph body). We treat lines starting
#: with any of these as bullets.
_BULLET_PREFIXES: Final[tuple[str, ...]] = ("•", "·", "-", "*", "◦", "‣")

#: Cap on what counts as an "orphan few bullets" at the top of a page.
#: More than this and it's just continuing the previous section
#: legitimately, not a stranded tail.
_MAX_ORPHAN_LEADING_BULLETS: Final[int] = 3

#: Regex for a section-level header (Profile / Education / etc).
#: The renderer puts these on their own line in title case, no
#: trailing period, no leading bullet.
_SECTION_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(Profile|Technical Skills|Education|Key Achievements|"
    r"Professional Experience|Personal Projects|Interests|"
    r"Publications|Certifications|Awards|References)$"
)

#: Regex for a per-role header line emitted by resumeai. The renderer
#: writes ``<Company> — <Role>`` on its own line above the bullets.
_ROLE_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z][\w& .'-]+\s—\s.+$")


def _is_bullet(line: str) -> bool:
    """Return True if ``line`` starts with one of the renderer's bullet glyphs."""
    stripped = line.lstrip()
    return any(
        stripped.startswith(prefix + " ") or stripped == prefix for prefix in _BULLET_PREFIXES
    )


def _is_section_header(line: str) -> bool:
    """Return True if ``line`` looks like a top-level section header."""
    return bool(_SECTION_HEADER_RE.match(line.strip()))


def _is_role_header(line: str) -> bool:
    """Return True if ``line`` looks like a per-role header ('Company — Role')."""
    return bool(_ROLE_HEADER_RE.match(line.strip()))


def _is_meaningful(line: str) -> bool:
    """Filter out empty lines + dates / locations the renderer puts on
    their own lines (those count as 'context' rather than 'a section'
    when deciding whether a header is orphaned)."""
    stripped = line.strip()
    if not stripped:
        return False
    # Single-line dates ("2024 — 2025"), locations ("Melbourne, Australia"),
    # role summaries (italic single-line subtitles) are common renderer
    # output but shouldn't gate the orphan-header detection.
    if re.fullmatch(r"\d{4}(\s*[—-]\s*\d{4})?(\s*\([^)]+\))?", stripped):
        return False
    # "City, Country" subtitle lines (Melbourne, Australia) are
    # context, not a section -- skip them when deciding whether the
    # previous page's last "meaningful" line is an orphan header.
    return not re.fullmatch(r"\w[\w ]+,\s+\w[\w ]+", stripped)


def check_layout_from_pages(
    pages: list[list[str]],
    *,
    document_label: str = "resume",
) -> list[QAIssue]:
    """Pure layout-check function. Returns the issues found in ``pages``.

    Tests drive this directly with fake page contents; the wrapper
    :func:`check_pdf_layout` extracts pages from PDF bytes and forwards.

    ``document_label`` shows up in issue summaries ("resume bullet
    orphaned on page 2" vs "cover letter ...") so the auto-fix prompt
    can target the right artefact.
    """
    issues: list[QAIssue] = []
    for idx in range(1, len(pages)):
        prev_page = pages[idx - 1]
        this_page = pages[idx]

        # ---- orphan leading bullets ------------------------------------
        # Walk the start of this page collecting bullets until we hit
        # a non-bullet line. If the non-bullet line is a section
        # header AND we collected 1..N bullets, those bullets came
        # from the previous page's section.
        leading_bullets: list[str] = []
        first_non_bullet: str | None = None
        for line in this_page:
            if _is_bullet(line):
                leading_bullets.append(line)
            elif line.strip():
                first_non_bullet = line.strip()
                break
        if (
            leading_bullets
            and first_non_bullet is not None
            and _is_section_header(first_non_bullet)
            and len(leading_bullets) <= _MAX_ORPHAN_LEADING_BULLETS
        ):
            issues.append(
                QAIssue(
                    severity="must_fix",
                    category="format",
                    summary=(
                        f"{document_label}: page {idx + 1} starts with "
                        f"{len(leading_bullets)} bullet"
                        f"{'s' if len(leading_bullets) != 1 else ''} stranded "
                        f"before '{first_non_bullet}' — they belong to the "
                        f"previous section on page {idx}."
                    ),
                    detail=(
                        "Shorten an earlier section by 1-2 lines so the "
                        f"trailing bullets fit on page {idx} alongside their "
                        "parent section, or trim the trailing section to "
                        "remove the bullets entirely."
                    ),
                ),
            )

        # ---- split section / role header --------------------------------
        # Find the last meaningful line on the previous page. If it
        # looks like a section or role header (no bullets following
        # it on that page), and this page starts with bullets, the
        # header is split from its body.
        prev_last_meaningful: str | None = None
        prev_had_bullets_after_last_header = False
        for line in reversed(prev_page):
            if not line.strip():
                continue
            if _is_bullet(line):
                prev_had_bullets_after_last_header = True
                break
            if not _is_meaningful(line):
                continue
            prev_last_meaningful = line.strip()
            break
        if (
            prev_last_meaningful is not None
            and not prev_had_bullets_after_last_header
            and (_is_section_header(prev_last_meaningful) or _is_role_header(prev_last_meaningful))
            and any(_is_bullet(line) for line in this_page[:5])
        ):
            issues.append(
                QAIssue(
                    severity="must_fix",
                    category="format",
                    summary=(
                        f"{document_label}: section header "
                        f"'{prev_last_meaningful}' at the bottom of "
                        f"page {idx} but its content starts on page {idx + 1}."
                    ),
                    detail=(
                        "Trim the section above by 2-3 lines so the header "
                        "moves to the next page where its bullets are, OR "
                        "expand the content above so the header lands well "
                        "below the page break."
                    ),
                ),
            )
    return issues


def _extract_pages_from_pdf(pdf_bytes: bytes) -> list[list[str]]:
    """Read a PDF and return each page as a list of text lines.

    Empty pages come back as ``[]``. Pages where pypdf can't extract
    text (corrupted / image-only) also come back as ``[]``; the
    caller's heuristics ignore empty pages so a bad extraction
    silently downgrades to "no layout issues" rather than raising.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[list[str]] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - pypdf raises generic Exceptions on bad pages
            text = ""
        lines = [line for line in text.splitlines() if line]
        pages.append(lines)
    return pages


def check_pdf_layout(pdf_bytes: bytes, *, document_label: str = "resume") -> list[QAIssue]:
    """Extract pages from ``pdf_bytes`` and run the layout heuristics.

    Wrapper around :func:`check_layout_from_pages` -- the split keeps
    the pure-function testable without minting real PDFs for every
    case, while the wrapper covers the I/O path.
    """
    if not pdf_bytes:
        return []
    try:
        pages = _extract_pages_from_pdf(pdf_bytes)
    except Exception:  # noqa: BLE001 - pypdf can throw on malformed input
        # Don't fail the QA stage over a bad PDF; layout issues just
        # won't surface. The structured-content QA from the LLM still
        # runs against the same chain.
        return []
    return check_layout_from_pages(pages, document_label=document_label)
