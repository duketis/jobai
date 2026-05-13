"""Coverage for the cross-artefact QA agent."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from jobai.tailor.models import QAAssessment, QAStatus
from jobai.tailor.qa import (
    AnthropicQAClient,
    _parse_assessment,
    assess,
    build_user_prompt,
)


def _good_assessment_json() -> str:
    """A canned QA verdict that round-trips cleanly through the parser."""
    return json.dumps(
        {
            "status": "pass",
            "coverage_score": 92,
            "consistency_score": 88,
            "format_score": 95,
            "must_fix_issues": [],
            "nice_to_fix_issues": [
                {
                    "severity": "nice_to_fix",
                    "category": "content",
                    "summary": "Letter opening could be more concrete.",
                },
            ],
            "summary": "Strong application; minor polish suggested.",
        },
    )


class _ScriptedQAClient:
    """In-memory :class:`QAClient` returning canned responses + capturing calls."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        return self.response


def test_build_user_prompt_includes_every_input_when_present() -> None:
    """All three inputs (JD, resume, letter) appear under their named sections."""
    prompt = build_user_prompt(
        jd={"title": "Engineer", "required_skills": ["python"]},
        resume_tailored={"name": "Jane", "summary": "...resume..."},
        letter_tailored={"opening": "Dear hiring manager..."},
    )
    assert "# JOB DESCRIPTION" in prompt
    assert "Engineer" in prompt
    assert "# TAILORED RESUME" in prompt
    assert "Jane" in prompt
    assert "# TAILORED COVER LETTER" in prompt
    assert "hiring manager" in prompt


def test_build_user_prompt_marks_missing_inputs_explicitly() -> None:
    """Each None input is surfaced as ``(not available)`` rather than
    silently omitted so the model knows what it's working with."""
    prompt = build_user_prompt(jd=None, resume_tailored=None, letter_tailored=None)
    assert prompt.count("(not available)") == 3


async def test_assess_round_trips_canned_response_into_assessment() -> None:
    client = _ScriptedQAClient(_good_assessment_json())
    out = await assess(
        jd={"title": "Engineer"},
        resume_tailored={"name": "Jane"},
        letter_tailored={"opening": "x"},
        client=client,
        model="claude-opus-4-7",
    )
    assert isinstance(out, QAAssessment)
    assert out.status == QAStatus.PASS
    assert out.coverage_score == 92
    assert len(out.must_fix_issues) == 0
    assert len(out.nice_to_fix_issues) == 1
    assert client.calls[0]["model"] == "claude-opus-4-7"


async def test_assess_returns_synthetic_fail_on_invalid_json() -> None:
    """A model response that isn't valid JSON becomes a synthetic
    ``fail`` assessment with the parse-error captured in must-fix."""
    client = _ScriptedQAClient("not valid json {[")
    out = await assess(jd=None, resume_tailored=None, letter_tailored=None, client=client)
    assert out.status == QAStatus.FAIL
    assert out.coverage_score == 0
    assert len(out.must_fix_issues) == 1
    assert "QA agent output could not be parsed" in out.must_fix_issues[0].summary


async def test_assess_returns_synthetic_fail_on_schema_violation() -> None:
    """Valid JSON that doesn't conform to ``QAAssessment`` (eg
    coverage_score > 100) is also surfaced as the synthetic fail."""
    client = _ScriptedQAClient(
        json.dumps(
            {
                "status": "pass",
                "coverage_score": 200,  # out of range
                "consistency_score": 80,
                "format_score": 80,
                "must_fix_issues": [],
                "nice_to_fix_issues": [],
                "summary": "x",
            },
        ),
    )
    out = await assess(jd=None, resume_tailored=None, letter_tailored=None, client=client)
    assert out.status == QAStatus.FAIL


def test_parse_assessment_strips_markdown_fence() -> None:
    """The model sometimes wraps strict-JSON output in a ``` fence
    despite being told not to. The parser strips it and recovers."""
    fenced = "```json\n" + _good_assessment_json() + "\n```"
    out = _parse_assessment(fenced)
    assert out.status == QAStatus.PASS


def test_parse_assessment_strips_unfenced_first_line() -> None:
    """A bare ``` (no language tag) plus a trailing fence also strips."""
    out = _parse_assessment("```\n" + _good_assessment_json() + "\n```")
    assert out.coverage_score == 92


def test_parse_assessment_handles_text_starting_with_fence_but_no_trailing() -> None:
    """The False branches of the inner ``if lines`` guards: the input
    starts with ``` but everything else parses cleanly without a
    trailing fence to strip."""
    out = _parse_assessment("```\n" + _good_assessment_json())
    assert out.coverage_score == 92


def test_parse_assessment_handles_lonely_fence_marker() -> None:
    """A ``` line on its own (after splitlines + index 0 strip) leaves
    ``lines`` empty; the second guard short-circuits without IndexError."""
    out = _parse_assessment("```")
    # Empty after fence-stripping -> can't parse -> synthetic fail.
    assert out.status == QAStatus.FAIL


async def test_anthropic_qa_client_collects_text_blocks() -> None:
    """The Anthropic adapter concatenates every text block from the SDK
    response into one string. We stub the SDK out so the test doesn't
    hit the wire."""

    class _FakeTextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeMessage:
        def __init__(self, content: list[Any]) -> None:
            self.content = content

    class _FakeMessages:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(self, **kwargs: object) -> _FakeMessage:
            self.calls.append(kwargs)
            return _FakeMessage([_FakeTextBlock('{"status":'), _FakeTextBlock('"pass"}')])

    class _FakeAnthropic:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    fake = _FakeAnthropic()
    client = AnthropicQAClient(client=fake, default_model="claude-opus-4-7")  # type: ignore[arg-type]
    text = await client.complete(system="sys", user="usr")
    assert text == '{"status":"pass"}'
    assert fake.messages.calls[0]["model"] == "claude-opus-4-7"


async def test_anthropic_qa_client_drops_non_text_content_blocks() -> None:
    """Tool-use / image content blocks have no ``text`` attribute; the
    adapter skips them rather than crashing."""

    class _BareBlock:
        pass

    class _FakeMessages:
        async def create(self, **kwargs: object) -> Any:
            del kwargs

            class _M:
                content: ClassVar[list[Any]] = [
                    _BareBlock(),
                    type("T", (), {"text": "data"})(),
                ]

            return _M()

    class _FakeAnthropic:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    client = AnthropicQAClient(client=_FakeAnthropic(), default_model="m")  # type: ignore[arg-type]
    text = await client.complete(system="s", user="u")
    assert text == "data"


async def test_anthropic_qa_client_uses_explicit_model_override() -> None:
    """When the caller passes a ``model`` argument it overrides the default."""

    class _FakeMessages:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, object] = {}

        async def create(self, **kwargs: object) -> Any:
            self.last_kwargs = kwargs

            class _M:
                content: ClassVar[list[Any]] = []

            return _M()

    class _FakeAnthropic:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    fake = _FakeAnthropic()
    client = AnthropicQAClient(client=fake, default_model="default-model")  # type: ignore[arg-type]
    await client.complete(system="s", user="u", model="override-model")
    assert fake.messages.last_kwargs["model"] == "override-model"


@pytest.fixture(autouse=True)
def _no_anyio_backend() -> None:
    """Pytest-asyncio is already in ``auto`` mode in this repo."""
    return
