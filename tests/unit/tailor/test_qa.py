"""Coverage for the cross-artefact QA agent."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest

from jobai.api.runtime_settings import EffectiveAgentConfig
from jobai.tailor.models import QAAssessment, QAStatus
from jobai.tailor.qa import (
    AnthropicQAClient,
    SubscriptionQAClient,
    _parse_assessment,
    assess,
    build_qa_client,
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


async def test_subscription_qa_client_collects_assistant_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subscription adapter joins every TextBlock emitted by the SDK
    across assistant messages, stopping when ResultMessage arrives."""
    import claude_agent_sdk  # noqa: PLC0415

    captured_opts: dict[str, Any] = {}

    async def _fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        captured_opts["prompt"] = prompt
        captured_opts["system_prompt"] = options.system_prompt
        captured_opts["model"] = options.model
        captured_opts["max_turns"] = options.max_turns
        captured_opts["env"] = dict(options.env)
        yield claude_agent_sdk.AssistantMessage(
            content=[
                claude_agent_sdk.TextBlock(text='{"status":'),
                claude_agent_sdk.TextBlock(text='"pass"}'),
            ],
            model="claude-opus-4-7",
            parent_tool_use_id=None,
        )
        # ResultMessage shapes vary across SDK versions; the adapter only
        # cares that it's an instance of the type, so a positional-args
        # constructor with whatever the dataclass declares first is fine.
        yield _make_result_message()

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query)

    client = SubscriptionQAClient(
        default_model="claude-opus-4-7",
        oauth_token="oat-x",  # noqa: S106 - test fixture, not a real token
    )
    text = await client.complete(system="rubric", user="docs")

    assert text == '{"status":"pass"}'
    assert captured_opts["prompt"] == "docs"
    assert captured_opts["system_prompt"] == "rubric"
    assert captured_opts["model"] == "claude-opus-4-7"
    assert captured_opts["max_turns"] == 1
    assert captured_opts["env"] == {"CLAUDE_CODE_OAUTH_TOKEN": "oat-x"}


async def test_subscription_qa_client_honours_model_override_and_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No OAuth token => env stays empty (CLI falls back to its host auth);
    explicit model arg overrides the default."""
    import claude_agent_sdk  # noqa: PLC0415

    captured: dict[str, Any] = {}

    async def _fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        del prompt
        captured["model"] = options.model
        captured["env"] = dict(options.env)
        yield _make_result_message()

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query)

    client = SubscriptionQAClient(default_model="claude-default", oauth_token=None)
    text = await client.complete(system="s", user="u", model="claude-override")

    assert text == ""
    assert captured["model"] == "claude-override"
    assert captured["env"] == {}


async def test_subscription_qa_client_ignores_unhandled_message_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SystemMessage / UserMessage echoes are skipped; only AssistantMessage
    TextBlocks contribute to the returned string."""
    import claude_agent_sdk  # noqa: PLC0415

    async def _fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        del prompt, options
        yield claude_agent_sdk.SystemMessage(subtype="init", data={"hello": "world"})
        yield claude_agent_sdk.AssistantMessage(
            content=[claude_agent_sdk.TextBlock(text="ok")],
            model="m",
            parent_tool_use_id=None,
        )
        yield _make_result_message()
        # Anything emitted after ResultMessage is unreachable -- the adapter
        # ``break``s out on it -- and never gets translated.

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query)
    client = SubscriptionQAClient(default_model="m", oauth_token=None)
    assert await client.complete(system="s", user="u") == "ok"


async def test_subscription_qa_client_skips_non_text_blocks_and_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-text or non-TextBlock assistant content is silently dropped --
    only blocks where ``isinstance(block, TextBlock) and block.text`` is
    truthy contribute to the returned string."""
    import claude_agent_sdk  # noqa: PLC0415

    class _NotATextBlock:
        text = "this should not appear"

    async def _fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        del prompt, options
        # ``content`` is typed as the SDK's content-block union; we feed
        # it a stand-in via ``cast`` so the test exercises the False
        # branch of ``isinstance(block, TextBlock)`` without mypy
        # rejecting the deliberate type-mismatch.
        wrong_block: Any = _NotATextBlock()
        yield claude_agent_sdk.AssistantMessage(
            content=[
                wrong_block,
                claude_agent_sdk.TextBlock(text=""),  # empty text
                claude_agent_sdk.TextBlock(text="kept"),
            ],
            model="m",
            parent_tool_use_id=None,
        )
        yield _make_result_message()

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query)
    client = SubscriptionQAClient(default_model="m", oauth_token=None)
    assert await client.complete(system="s", user="u") == "kept"


async def test_subscription_qa_client_returns_cleanly_when_stream_closes_without_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the SDK stream ends without ever emitting ResultMessage (a
    transport-level early-close) we still return what we collected
    rather than hanging or raising. Covers the natural loop-exit branch."""
    import claude_agent_sdk  # noqa: PLC0415

    async def _fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        del prompt, options
        yield claude_agent_sdk.AssistantMessage(
            content=[claude_agent_sdk.TextBlock(text="partial")],
            model="m",
            parent_tool_use_id=None,
        )
        # No ResultMessage -- the loop falls off the end naturally.

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query)
    client = SubscriptionQAClient(default_model="m", oauth_token=None)
    assert await client.complete(system="s", user="u") == "partial"


def _make_result_message() -> Any:
    """Build a ResultMessage in a way that works across SDK versions.

    Only the *type* matters for the adapter's break-out; the field
    values aren't inspected. We construct via ``__new__`` so we don't
    have to chase whichever fields a given SDK release requires.
    """
    import claude_agent_sdk  # noqa: PLC0415

    return claude_agent_sdk.ResultMessage.__new__(claude_agent_sdk.ResultMessage)


def _cfg(**overrides: Any) -> EffectiveAgentConfig:
    """Build a :class:`EffectiveAgentConfig` with sensible defaults."""
    return EffectiveAgentConfig(
        agent_backend=overrides.get("agent_backend", "api"),
        anthropic_api_key=overrides.get("anthropic_api_key"),
        claude_code_oauth_token=overrides.get("claude_code_oauth_token"),
        anthropic_model=overrides.get("anthropic_model", "claude-opus-4-7"),
    )


def test_build_qa_client_returns_subscription_client_in_subscription_mode() -> None:
    cfg = _cfg(
        agent_backend="subscription",
        claude_code_oauth_token="oat-x",  # noqa: S106 - test fixture, not a real token
    )
    client = build_qa_client(cfg)
    assert isinstance(client, SubscriptionQAClient)
    assert client._oauth_token == "oat-x"  # noqa: S105 - asserting test-fixture value
    assert client._default_model == "claude-opus-4-7"


def test_build_qa_client_falls_back_to_api_when_subscription_lacks_token() -> None:
    """Subscription backend declared but no OAuth token => try API key path
    rather than returning None."""
    cfg = _cfg(
        agent_backend="subscription",
        claude_code_oauth_token=None,
        anthropic_api_key="sk-ant-test",
    )
    client = build_qa_client(cfg)
    assert isinstance(client, AnthropicQAClient)


def test_build_qa_client_returns_anthropic_client_with_api_key() -> None:
    cfg = _cfg(anthropic_api_key="sk-ant-test", agent_backend="api")
    client = build_qa_client(cfg)
    assert isinstance(client, AnthropicQAClient)
    assert client._default_model == "claude-opus-4-7"


def test_build_qa_client_returns_none_when_neither_credential_present() -> None:
    cfg = _cfg()  # No key, no token, default backend.
    assert build_qa_client(cfg) is None


@pytest.fixture(autouse=True)
def _no_anyio_backend() -> None:
    """Pytest-asyncio is already in ``auto`` mode in this repo."""
    return
