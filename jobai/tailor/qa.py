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
from typing import TYPE_CHECKING, Any, Protocol

from jobai.tailor.models import QAAssessment, QAIssue, QAStatus

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from jobai.api.runtime_settings import EffectiveAgentConfig

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

   FABRICATED OPEN-SOURCE STATS — flag as 'must_fix' consistency:
   The candidate's own open-source / portfolio repos (jobai, any
   GitHub-hosted personal project) have ground-truth numbers that
   live in the user-context pool the siblings pull from at tailor
   time. The LLM that produced these artefacts has been known to
   substitute plausible-looking but stale or invented stats for
   real ones. If a resume bullet or letter sentence makes a SPECIFIC
   numeric claim about the candidate's own portfolio repo --
   - a test count ("705+ tests", "with 240 tests"),
   - a coverage percentage ("89% test coverage", ">89% coverage"),
   - a commit count, star count, LOC figure, or test-runtime number
     about a candidate-controlled repo --
   and there is NO source in the JD or in the structured artefacts
   you've been shown to substantiate that exact number, treat it as
   a 'must_fix' consistency issue. Phrase the summary so the auto-
   fix prompt knows to either delete the unsupported claim or
   replace it with a more general phrasing ("comprehensive test
   suite", "high coverage discipline") that doesn't pin a number.
   This applies to the candidate's OWN projects only -- claims about
   prior client engagements (DiUS / InTruth / etc.) follow the
   normal consistency rules above.

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


class QAClient(Protocol):
    """Wire surface for the QA LLM call.

    Async so it composes cleanly with the orchestrator's chain
    coroutine. Tests inject a fake that returns canned JSON without
    touching the network.
    """

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        """Send one completion and return the raw text response."""
        ...


def build_user_prompt(
    *,
    jd: dict[str, Any] | None,
    resume_tailored: dict[str, Any] | None,
    letter_tailored: dict[str, Any] | None,
) -> str:
    """Compose the user-prompt that pairs every input doc for the agent."""
    parts: list[str] = []
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
) -> QAAssessment:
    """Run the QA pass and return a parsed :class:`QAAssessment`.

    Validation failure (malformed JSON, missing fields) is surfaced
    as a ``fail`` assessment so the orchestrator doesn't break the
    chain on a model misbehaviour -- the user still gets their PDFs
    and the failed QA explains why we couldn't grade them.
    """
    raw = await client.complete(
        system=_SYSTEM_PROMPT,
        user=build_user_prompt(
            jd=jd, resume_tailored=resume_tailored, letter_tailored=letter_tailored
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
