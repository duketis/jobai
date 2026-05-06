#!/usr/bin/env bash
# Commit and push Phase 6 work that was built overnight.
#
# Run this AFTER unlocking GPG (`echo test | gpg --clearsign > /dev/null`).
# The script first re-signs the three unsigned commits at the base of
# the jobai history so the final tree is 100% signed (no half-signed
# story to explain in an interview), then layers on the Phase 6
# commits, then force-pushes.
#
# !!  WARNING — this rewrites every commit on `main` (SHAs change)   !!
# !!  and force-pushes to the public repo. That's safe in a solo,    !!
# !!  no-PR setup (yours), but it WILL look bad in the GitHub        !!
# !!  network graph if anyone has cloned mid-flight. Skim the steps  !!
# !!  before answering the prompt.                                   !!

set -euo pipefail

cd "$(dirname "$0")/.."

cat <<'BANNER'
This script will:
  1. Rebase --root with --exec to amend+sign every existing commit.
     -> SHAs change for all 9 existing commits (3 unsigned, 6 signed).
     -> Commit messages, dates, and authors are preserved.
  2. Add 7 new signed commits for Phase 6 work.
  3. git push --force-with-lease origin main.

Type yes to proceed, anything else to abort.
BANNER
read -r -p "Proceed? " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborted. No changes made."
    exit 1
fi

# Sanity check: run the local quality gate first so we don't push red.
echo "==> Running quality gate"
/tmp/jobai-tools/bin/ruff check .
/tmp/jobai-tools/bin/ruff format --check .
/tmp/jobai-tools/bin/mypy jobai tests
/tmp/jobai-tools/bin/pytest -q

# ---------------------------------------------------------------------------
# Phase 0 — re-sign the three unsigned early commits.
#
# `git rebase --root --exec 'git commit --amend --no-edit -S'` walks
# every commit from the root and re-amends with a signature. Signed
# commits get re-signed (idempotent); unsigned ones get a sig added.
# Commit dates are preserved by amend; SHAs change because the commit
# object now embeds the GPG sig.
# ---------------------------------------------------------------------------
echo "==> Re-signing all commits to produce a uniform signed history"
git rebase --root --exec 'git commit --amend --no-edit -S'

# 6.1 — browser fetcher (Playwright)
echo "==> Phase 6.1: BrowserFetcher (Playwright)"
git add jobai/fetcher/browser.py tests/unit/fetcher/test_browser.py
git commit -m "feat(fetcher): add tier-2 BrowserFetcher backed by Playwright" \
  -m "Implements the Fetcher Protocol for JS-rendered sources (Workable listings, Seek, modern enterprise SPAs). PlaywrightDriver owns the browser lifecycle — lazy launch, one fresh context per fetch for cookie isolation. BrowserFetcher orchestrates argument validation and Response translation, with the driver injectable so unit tests run without launching Chromium."

# 6.2 — tier escalation
echo "==> Phase 6.2: EscalatingFetcher"
git add jobai/fetcher/escalation.py tests/unit/fetcher/test_escalation.py
git commit -m "feat(fetcher): add EscalatingFetcher for HTTP -> browser tier promotion" \
  -m "Wraps a primary HTTP fetcher and a browser fallback factory. Escalates on 403/429 status codes or Cloudflare interstitial body signatures (Just a moment, Checking your browser, etc). Sticky once-escalated semantics — re-probing the primary after a block would just hit the same wall and burn requests. Lazy fallback construction means the cheap path stays cheap when nothing escalates."

# 6.3 — Seek source
echo "==> Phase 6.3: Seek source"
git add jobai/sources/seek.py tests/unit/sources/test_seek.py \
  tests/unit/sources/fixtures/seek_python_melbourne.html
git commit -m "feat(sources): add Seek source for AU job listings" \
  -m "Parses the Next.js __NEXT_DATA__ JSON island embedded in seek.com.au search-results pages — far more stable than scraping CSS-class-styled DOM. Defensive throughout: missing or malformed islands return zero results rather than raising, entries without id/title are skipped."

# 6.4 — Patchright stealth fetcher
echo "==> Phase 6.4: StealthFetcher (Patchright)"
git add jobai/fetcher/stealth.py tests/unit/fetcher/test_stealth.py
git commit -m "feat(fetcher): add tier-3 stealth fetcher using Patchright" \
  -m "Patchright is a Playwright fork with anti-detection patches. Reuses PlaywrightDriver via a thin shim that swaps in patchright.async_playwright as the runtime; everything else (lifecycle, response translation) is shared with the tier-2 BrowserFetcher. Keeps both pipelines on one maintenance path."

# 6.5 — LinkedIn + Indeed
echo "==> Phase 6.5: LinkedIn + Indeed sources"
git add jobai/sources/linkedin.py jobai/sources/indeed.py \
  tests/unit/sources/test_linkedin.py tests/unit/sources/test_indeed.py \
  tests/unit/sources/fixtures/linkedin_python_au.html \
  tests/unit/sources/fixtures/indeed_python_melbourne.html
git commit -m "feat(sources): add LinkedIn + Indeed sources" \
  -m "LinkedIn parses guest-mode job-card markup with selectolax (extracting job id from data-entity-urn). Indeed parses the SSR window._initialData island. Both ship with build_query() helpers, defensive parsing that drops malformed entries, and per-source error types that include the originating query for triage."

# 6.6 — APS Jobs (federal AU government)
echo "==> Phase 6.6: APS Jobs source + loader fix"
git add jobai/sources/apsjobs.py jobai/sources/loader.py \
  tests/unit/sources/test_apsjobs.py \
  tests/unit/sources/fixtures/apsjobs_software.atom
git commit -m "feat(sources): add APS Jobs (federal AU government) source" \
  -m "Parses the Atom feed at apsjobs.gov.au/s/search.atom — stable structured data, HTTP-tier compatible. Agency name extracted from the 'X is hiring' summary convention; location and salary parsed from the same block. Loader relaxed to allow empty 'account' strings since BaseSource explicitly supports kind-only sources."

# Wire-up: registry + companies.yaml (combined since they touch the same files)
echo "==> Phase 6.x: register sources and seed defaults"
git add jobai/sources/registry.py jobai/sources/companies.yaml
git commit -m "feat(sources): register new sources and seed companies.yaml" \
  -m "Wires apsjobs, indeed, linkedin, seek into the source registry and seeds default search slugs/queries in companies.yaml. AU-tilted: every new source ships with at least one Australia-focused config so the runner can hit them out of the box."

# Morning artifacts (private)
echo "==> Phase 6 wrap-up: morning checklist + commit script"
git add _private/MORNING_CHECKLIST.md scripts/commit_overnight_work.sh
git commit -m "chore: add overnight Phase 6 commit script and morning checklist" \
  -m "Internal tooling — captures the unattended-work flow so the next overnight session has a template."

echo "==> Force-pushing main (history was rewritten by the re-sign rebase)"
git push --force-with-lease

echo "==> Done. Watch CI: gh run watch --exit-status"
