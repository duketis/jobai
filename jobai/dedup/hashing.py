"""Deterministic dedup-key computation.

The dedup key is a SHA-256 hash over the unit-separator-joined tuple
``(company_norm, title_norm, country_norm)``. Two jobs with the same
key are the same job — guaranteed, not heuristic. The fuzzy pass
(``jobai.dedup.fuzzy`` + ``reconcile``) handles near-misses that share
no exact key.

Normalisation collapses the variation we know about — case, accents,
punctuation, whitespace, common corporate suffixes — so trivial
formatting differences across sources don't produce different keys.
We deliberately do **not** normalise seniority words (Senior, Staff,
Lead, Principal) because those distinguish real roles; the fuzzy pass
catches abbreviation-only variants like "Sr." -> "Senior".
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# ASCII unit separator; cannot collide with any character in normalised input.
_DEDUP_SEPARATOR = "\x1f"

_PUNCT_PATTERN = re.compile(r"[^\w\s-]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Common corporate suffixes stripped from company names so "Atlassian Pty Ltd"
# and "Atlassian" produce the same normalised form. The pattern is anchored to
# the end of the string with optional trailing comma / period to handle "Foo,
# Inc." cleanly.
_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\s*,?\s*"
    r"(?:"
    r"inc(?:orporated)?|"
    r"corp(?:oration)?|"
    r"co(?:mpany)?|"
    r"llc|llp|"
    r"l\.l\.c|"
    r"ltd|limited|"
    r"pty\s*ltd|pty\s*limited|"
    r"pte\s*ltd|"
    r"plc|"
    r"gmbh|"
    r"sas|sa|s\.a|"
    r"sarl|s\.a\.r\.l|"
    r"srl|s\.r\.l|"
    r"nv|n\.v|"
    r"ag|"
    r"kk|k\.k|"
    r"holdings|group|technologies|technology"
    r")"
    r"\.?\s*$",
    re.IGNORECASE,
)


def normalize_company(name: str) -> str:
    """Return the canonical lower-case representation of ``name``.

    Steps: NFKD-decompose to drop combining marks (accents), lower-case,
    strip recognised corporate suffixes, drop punctuation, collapse
    whitespace.
    """
    if not name:
        return ""
    text = _strip_accents(name).lower().strip()
    # Strip suffixes iteratively in case there are multiple ("Foo Holdings Ltd").
    while True:
        new_text = _COMPANY_SUFFIX_PATTERN.sub("", text).strip()
        if new_text == text:
            break
        text = new_text
    text = _PUNCT_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_title(title: str) -> str:
    """Lower-case, drop accents, drop punctuation, collapse whitespace."""
    if not title:
        return ""
    text = _strip_accents(title).lower().strip()
    text = _PUNCT_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_country(country: str | None) -> str:
    """Lower-case, trim. ``None`` and empty become ``""``."""
    if not country:
        return ""
    return country.strip().lower()


def compute_dedup_key(*, company: str, title: str, country: str | None) -> str:
    """Compute the SHA-256 dedup key over the normalised tuple.

    Identical normalised inputs produce identical keys; hash output is
    stable across processes and across machines.
    """
    payload = _DEDUP_SEPARATOR.join(
        [
            normalize_company(company),
            normalize_title(title),
            normalize_country(country),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strip_accents(text: str) -> str:
    """NFKD decomposition + drop combining marks. ``"Café"`` -> ``"Cafe"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
