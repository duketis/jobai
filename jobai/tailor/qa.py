"""Cross-artefact QA agent for the tailor chain.

After resumeai + coverletterai both succeed, this module reads the
two ``tailored`` JSON blobs alongside the parsed JD and asks an LLM
to assess them as a single application package. It catches problems
neither sibling can see in isolation:

* A cover-letter claim the resume doesn't substantiate.
* A JD must-have keyword that appears in neither artefact.
* Tonal / format drift between the two documents.
* Bullets the resume foregrounds that the letter fails to echo.

The agent returns a :class:`~jobai.tailor.models.QAAssessment` -- the
orchestrator persists it on the ``tailor_runs`` row and the frontend
surfaces it as a badge + drill-in panel. We do NOT gatekeep on the
verdict: the user sees the assessment, but the artefacts ship either
way. Auto-retry on must-fix issues is intentionally NOT here -- it's
a future iteration that needs careful guardrails (LLM cost,
oscillation between revisions) that aren't worth the complexity for
v1.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol

from jobai.tailor.models import QAAssessment, QAIssue, QAStatus

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from jobai.api.runtime_settings import EffectiveAgentConfig

_log = logging.getLogger(__name__)

#: Cap on how many characters of any one context entry we feed into
#: the QA prompt. Project-scan markdown can be hundreds of KB; the
#: head (README + commit log) is what carries verifiable claims, and
#: keeping per-entry size bounded keeps the user prompt within token
#: budget even with many entries.
_PER_ENTRY_CHAR_CAP: int = 6_000

#: Cap on the assembled QA-context string total. Anthropic models
#: handle 100KB-class user prompts without issue; this is a safety
#: belt against pathologically large pools.
_TOTAL_CHAR_CAP: int = 60_000

_SYSTEM_PROMPT = """\
You are the final quality gate for a tailored job application: a
resume and a cover letter the system has produced for a specific
job description. You receive the structured JSON of each artefact
and the parsed JobRequirements. You DO NOT edit them. You audit
them as a single submission and return a structured assessment.

What you assess:

1. COVERAGE (0-100): How well does the application hit every must-
   have in the JD's `requirements.required_skills` /
   `responsibilities`? A high coverage score means the resume +
   letter together address every requirement at least once.
   Missing must-haves are 'must_fix' coverage issues. Missing
   nice-to-haves are 'nice_to_fix'.

2. CONSISTENCY (0-100): Do the resume and letter agree? Examples
   of inconsistency:
   - Letter claims a metric the resume doesn't carry ("scaled to
     10k QPS" in the letter, no such bullet in the resume).
   - Letter says "led a team of 8" but the resume's strongest
     bullet says "led a team of 5".
   - Letter cites a project / employer the resume doesn't list.
   - Tone clash (resume formal, letter casual).
   Any contradiction is 'must_fix' consistency. Same-but-tonally-
   off is 'nice_to_fix'.

   FABRICATED OPEN-SOURCE STATS:
   The candidate's own open-source / portfolio repos have ground-
   truth numbers that live in the ``# USER CONTEXT (VERIFIED)``
   section of this prompt when one is provided. The LLM that
   produced these artefacts has been known to substitute plausible-
   looking but stale or invented stats for real ones.

   Decision rule:
   - If a USER CONTEXT section is provided AND a numeric claim in
     the resume / letter CONTRADICTS a verified number in it
     (e.g. resume says "705+ tests at 89% coverage" but USER CONTEXT
     says "1126 tests at 100% coverage"), that is a 'must_fix'
     consistency issue. Phrase the summary so the auto-fix prompt
     can replace the wrong number with the verified one.
   - If a USER CONTEXT section is provided AND a numeric claim is
     ABSENT from it (no contradiction, just no source either way),
     treat that as at most 'nice_to_fix' consistency -- the
     candidate may have authoritative knowledge the context doesn't
     surface. Do NOT mark as must_fix.
   - If NO USER CONTEXT section is provided, you have no ground
     truth and must NOT flag numeric claims on this basis alone --
     fall back to the normal consistency rules above
     (contradictions WITHIN the artefacts are still must_fix).

   This rule covers the candidate's OWN projects (anything in their
   USER CONTEXT). Claims about prior client engagements
   (DiUS / InTruth / etc.) follow the normal consistency rules.

3. FORMAT (0-100): Are the two documents stylistically aligned?
   Same name + contact block, same date format, complementary
   header treatment. The PDFs are LaTeX-rendered so we don't
   expect identical fonts, but the candidate's presented identity
   should match across them.

How to score:

- 90-100 = excellent, no edits needed.
- 75-89  = solid; minor polish only.
- 60-74  = workable but flagged concerns.
- < 60   = serious issues; this application would lose to a tidier
           competitor.

How to set ``status``:
- 'pass'     : every score >= 80 AND zero must_fix issues.
- 'concerns' : 60 <= any score < 80, OR any nice_to_fix issue.
- 'fail'     : any score < 60, OR any must_fix issue.

# OUTPUT FORMAT

Return STRICT JSON only. No fences, no preamble, no commentary --
the orchestrator parses your output directly with a Pydantic model
and will mark the run as a parse failure if anything else is present.

EVERY field below is REQUIRED. ``coverage_score``,
``consistency_score`` and ``format_score`` are integers 0-100; emit
them on every response, even when you can't draw strong signal --
use a neutral 60 in that case and call out the missing context in
``summary`` instead of omitting the field. ``must_fix_issues`` and
``nice_to_fix_issues`` are arrays (may be empty); ``summary`` is a
short paragraph.

Exact response shape:

{
  "status": "pass" | "concerns" | "fail",
  "coverage_score": 0..100,
  "consistency_score": 0..100,
  "format_score": 0..100,
  "must_fix_issues": [
    {"severity": "must_fix", "category": "coverage"|"consistency"|"format"|"content",
     "summary": "short headline", "detail": "optional longer reason"}
  ],
  "nice_to_fix_issues": [
    {"severity": "nice_to_fix", "category": "coverage"|"consistency"|"format"|"content",
     "summary": "short headline", "detail": "optional longer reason"}
  ],
  "summary": "one short paragraph"
}
"""

#: Resume-only QA gate. Runs BEFORE the cover letter is ever kicked
#: (v1.28.0), so there is no letter to cross-check — the resume is
#: graded purely against the JD and the verified USER CONTEXT. Same
#: output schema; the cross-document CONSISTENCY/FORMAT rules collapse
#: to within-resume + resume-vs-context checks.
_RESUME_SYSTEM_PROMPT = """\
You are the FIRST quality gate for a tailored job application. Only
the RESUME exists at this point — the cover letter has NOT been
written yet (it is generated only after this resume passes). You
receive the structured JSON of the resume and the parsed
JobRequirements. You DO NOT edit it. You audit the resume ALONE.

What you assess:

1. COVERAGE (0-100): How well does the RESUME hit every must-have in
   the JD's `requirements.required_skills` / `responsibilities`?
   Missing must-haves are 'must_fix' coverage issues; missing
   nice-to-haves are 'nice_to_fix'. (There is no letter to share the
   load — the resume must carry coverage on its own.)

2. CONSISTENCY (0-100): Is the resume internally consistent, and does
   every numeric / factual claim about the candidate's OWN projects
   agree with the `# USER CONTEXT (VERIFIED)` section when provided?

   FABRICATED OPEN-SOURCE STATS — decision rule:
   - USER CONTEXT provided AND a resume number CONTRADICTS a verified
     number in it (e.g. resume "705+ tests at 89% coverage" vs
     context "1126 tests at 100% coverage") -> 'must_fix' consistency.
     Phrase the summary so an auto-fix prompt can substitute the
     verified number.
   - USER CONTEXT provided AND the number is merely ABSENT from it ->
     at most 'nice_to_fix' (candidate may know more than the context
     surfaces). NOT must_fix.
   - NO USER CONTEXT -> no ground truth; do NOT flag numbers on that
     basis. Within-resume contradictions are still must_fix.
   Claims about prior client engagements follow normal consistency
   rules, not the verified-stats rule.

3. FORMAT (0-100): Is the resume well-formed for a 1-page LaTeX
   render — a coherent header (name + contact), consistent date
   formatting, no obviously malformed/empty required sections?

Scoring, status, and the OUTPUT FORMAT are EXACTLY as below.
""" + _SYSTEM_PROMPT[_SYSTEM_PROMPT.index("How to score:") :]


class QAClient(Protocol):
    """Wire surface for the QA LLM call.

    Async so it composes cleanly with the orchestrator's chain
    coroutine. Tests inject a fake that returns canned JSON without
    touching the network.
    """

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        """Send one completion and return the raw text response."""
        ...


async def fetch_qa_context_summary(resumeai_url: str) -> str | None:
    """Pull the resumeai context pool and assemble the QA ground-truth
    summary that gets fed into :func:`assess`.

    Selects every entry tagged ``verified`` (pinned single-source-of-
    truth snippets the user maintains) plus every entry tagged
    ``source:local_project`` (auto-refreshed project scans). Each
    entry is truncated to :data:`_PER_ENTRY_CHAR_CAP`; the assembled
    string is truncated to :data:`_TOTAL_CHAR_CAP`.

    Returns ``None`` if the pool is unreachable, empty, or contains
    no QA-relevant entries -- QA then falls back to within-artefact
    consistency checks only.
    """
    from jobai.context.client import HttpxContextClient  # noqa: PLC0415

    client = HttpxContextClient(base_url=resumeai_url)
    try:
        try:
            entries = await client.list_files()
        except Exception:  # noqa: BLE001 - context pool unreachable
            _log.warning("qa_context_fetch_failed", exc_info=True)
            return None

        relevant = [
            e
            for e in entries
            if ("verified" in e.tags or "source:local_project" in e.tags) and e.extracted_text
        ]
        if not relevant:
            return None

        parts: list[str] = []
        budget = _TOTAL_CHAR_CAP
        for entry in relevant:
            assert entry.extracted_text is not None  # noqa: S101 - filtered above
            body = entry.extracted_text[:_PER_ENTRY_CHAR_CAP]
            tag_label = " ".join(sorted(entry.tags)) if entry.tags else "(no tags)"
            block = f"## {entry.name}\n_tags: {tag_label}_\n\n{body}"
            if len(block) + 2 > budget:
                break
            parts.append(block)
            budget -= len(block) + 2
        if not parts:
            return None
        return "\n\n".join(parts)
    finally:
        await client.aclose()


def build_user_prompt(
    *,
    jd: dict[str, Any] | None,
    resume_tailored: dict[str, Any] | None,
    letter_tailored: dict[str, Any] | None,
    user_context: str | None = None,
) -> str:
    """Compose the user-prompt that pairs every input doc for the agent.

    ``user_context``, when supplied, is the concatenated text of the
    candidate's verified context-pool entries (pinned snippets,
    project scans). The QA prompt cross-references numeric claims
    against it so a resume that says "705 tests at 89% coverage"
    gets flagged when the context confirms the real number is
    "1126 tests at 100% coverage". Without it, QA has no ground
    truth to compare against and the FABRICATED OPEN-SOURCE STATS
    rule does not fire.
    """
    parts: list[str] = []
    if user_context:
        parts.append("# USER CONTEXT (VERIFIED)")
        parts.append(
            "The following are verified facts about the candidate's projects "
            "and experience -- the resumeai / coverletterai siblings consumed "
            "the same context when producing the artefacts. Treat each entry "
            "as ground truth and use it to validate numeric / factual claims "
            "in the resume and letter.",
        )
        parts.append("")
        parts.append(user_context.strip())
        parts.append("")
    parts.append("# JOB DESCRIPTION")
    parts.append(json.dumps(jd, indent=2, default=str) if jd else "(not available)")
    parts.append("")
    parts.append("# TAILORED RESUME")
    parts.append(
        json.dumps(resume_tailored, indent=2, default=str)
        if resume_tailored
        else "(not available)",
    )
    parts.append("")
    parts.append("# TAILORED COVER LETTER")
    parts.append(
        json.dumps(letter_tailored, indent=2, default=str)
        if letter_tailored
        else "(not available)",
    )
    parts.append("")
    parts.append("# OUTPUT")
    parts.append("Return the QAAssessment JSON per the system prompt's schema.")
    return "\n".join(parts)


async def assess(
    *,
    jd: dict[str, Any] | None,
    resume_tailored: dict[str, Any] | None,
    letter_tailored: dict[str, Any] | None,
    client: QAClient,
    model: str | None = None,
    user_context: str | None = None,
    stage: Literal["combined", "resume"] = "combined",
) -> QAAssessment:
    """Run the QA pass and return a parsed :class:`QAAssessment`.

    ``user_context`` is forwarded into the user prompt. When set,
    QA has ground truth to validate numeric / factual claims
    against; when ``None``, QA falls back to within-artefact
    consistency checks only (no fabricated-stats rule).

    Validation failure (malformed JSON, missing fields) is surfaced
    as a ``fail`` assessment so the orchestrator doesn't break the
    chain on a model misbehaviour -- the user still gets their PDFs
    and the failed QA explains why we couldn't grade them.
    """
    raw = await client.complete(
        system=_RESUME_SYSTEM_PROMPT if stage == "resume" else _SYSTEM_PROMPT,
        user=build_user_prompt(
            jd=jd,
            resume_tailored=resume_tailored,
            letter_tailored=letter_tailored,
            user_context=user_context,
        ),
        model=model,
    )
    return _parse_assessment(raw)


class AnthropicQAClient:
    """:class:`QAClient` backed by the shared ``AsyncAnthropic`` SDK.

    Built once per lifespan and reused across tailor runs; the SDK
    internally maintains an httpx connection pool so this is cheap.
    """

    def __init__(self, *, client: AsyncAnthropic, default_model: str) -> None:
        self._client = client
        self._default_model = default_model

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        # ``messages.create`` returns a ``Message`` whose ``content`` is
        # a list of content blocks. The QA prompt asks for strict JSON
        # only, so we expect a single text block.
        response = await self._client.messages.create(
            model=model or self._default_model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        chunks: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)


class SubscriptionQAClient:
    """:class:`QAClient` backed by the ``claude`` CLI via ``claude_agent_sdk``.

    Mirrors :mod:`jobai.agent.subscription_loop` -- one-shot completion,
    no MCP tools, no streaming history. Calls bill against the user's
    Claude Pro/Max subscription quota rather than a paid API key.

    The OAuth token is captured at construction and forwarded to the
    CLI subprocess via ``options.env`` (same pattern the chat agent
    uses) so the secret never leaks onto the FastAPI server's process
    environment.
    """

    def __init__(self, *, default_model: str, oauth_token: str | None = None) -> None:
        self._default_model = default_model
        self._oauth_token = oauth_token

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        # Lazy-imported so the rest of the module stays importable in
        # environments without the SDK installed (the API-mode path
        # only needs ``anthropic``).
        from claude_agent_sdk import (  # noqa: PLC0415
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        cli_env: dict[str, str] = {}
        if self._oauth_token:
            cli_env["CLAUDE_CODE_OAUTH_TOKEN"] = self._oauth_token

        options = ClaudeAgentOptions(
            system_prompt=system,
            # No code-tools, no MCP servers -- the QA pass is a pure
            # text-in / JSON-out completion. ``bypassPermissions`` keeps
            # the CLI from prompting for tool-use confirmation that the
            # background tailor pool can't answer.
            tools=[],
            mcp_servers={},
            allowed_tools=[],
            model=model or self._default_model,
            max_turns=1,
            permission_mode="bypassPermissions",
            env=cli_env,
        )

        chunks: list[str] = []
        async for message in query(prompt=user, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                # SDK guarantees ResultMessage closes every turn -- safe
                # to stop iterating here without draining further.
                break
        return "".join(chunks)


def build_qa_client(cfg: EffectiveAgentConfig) -> QAClient | None:
    """Pick the right QA client for the live agent config.

    Resolution order:
    1. ``subscription`` backend + an OAuth token -> SubscriptionQAClient
       (bills the user's Max quota; same auth path as the chat agent).
    2. ``api`` backend + an API key (any source) -> AnthropicQAClient
       (pay-per-token billing).
    3. Otherwise ``None`` -- the orchestrator skips QA cleanly and both
       PDFs still ship. The UI's QA badge stays empty for runs that
       reach succeeded without a verdict.

    Lazy-imports the Anthropic SDK so a missing install doesn't break
    the subscription path.
    """
    if cfg.agent_backend == "subscription" and cfg.claude_code_oauth_token:
        return SubscriptionQAClient(
            default_model=cfg.anthropic_model,
            oauth_token=cfg.claude_code_oauth_token,
        )
    if cfg.anthropic_api_key:
        from anthropic import AsyncAnthropic  # noqa: PLC0415

        client = AsyncAnthropic(api_key=cfg.anthropic_api_key)
        return AnthropicQAClient(client=client, default_model=cfg.anthropic_model)
    return None


def _parse_assessment(raw: str) -> QAAssessment:
    """Decode the model's JSON output into a :class:`QAAssessment`.

    Strips an optional Markdown fence the model sometimes emits even
    when told not to, then validates against the Pydantic schema.
    Anything that fails parse / validate becomes a synthetic ``fail``
    assessment with a single must-fix issue describing the parse
    error -- the chain still completes and the user sees what
    happened in the UI.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Drop the first fence line + the trailing fence.
        lines = text.splitlines()
        # The outer guard already confirms text starts with ``` so the
        # first-line check below is always true; pragma'd because we
        # keep the defensive ``if lines`` for readability.
        if lines and lines[0].startswith("```"):  # pragma: no branch
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        return QAAssessment.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        return QAAssessment(
            status=QAStatus.FAIL,
            coverage_score=0,
            consistency_score=0,
            format_score=0,
            must_fix_issues=[
                QAIssue(
                    severity="must_fix",
                    category="content",
                    summary="QA agent output could not be parsed.",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            ],
            summary=(
                "The cross-artefact QA pass failed to return a valid assessment; "
                "the PDFs still rendered cleanly."
            ),
        )
