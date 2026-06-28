"""Mocked-client unit tests for :mod:`altendor.classify.classifier` and schema (S10).

These tests never touch the real Anthropic SDK — they wire a duck-typed fake
client into :func:`classify_post` and assert on the captured ``messages.create``
kwargs plus the returned :class:`ClassifyResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import pytest
from altendor.classify.classifier import PaperCtx, classify_post
from altendor.classify.prompts import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_TOOL_NAME
from altendor.classify.schema import (
    Endorsement,
    Flag,
    Irrelevant,
    ZeroMagnitudeError,
    parse_tool_input,
    tool_input_schema,
)
from altendor.enrich.text_resolver import ResolvedPost
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Fake Anthropic client plumbing
# ---------------------------------------------------------------------------


@dataclass
class FakeBlock:
    """Duck-types an Anthropic content block (text or tool_use)."""

    type: str
    text: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    id: str | None = None


@dataclass
class FakeUsage:
    """Duck-types ``Response.usage`` (zeros are fine for these tests)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeResponse:
    """Duck-types ``anthropic.types.Message`` enough for the classifier."""

    content: list[FakeBlock]
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    """Duck-types ``client.messages``; captures kwargs and returns a canned response."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.record_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:  # noqa: ANN401 - Anthropic SDK takes heterogeneous kwargs
        self.record_calls.append(kwargs)
        return self._response


class FakeClient:
    """Minimal duck-typed stand-in for :class:`anthropic.Anthropic`."""

    def __init__(self, response: FakeResponse) -> None:
        self.messages = FakeMessages(response)


def _as_client(fake: FakeClient) -> anthropic.Anthropic:
    """Cast helper so the type checker accepts the duck-typed fake."""
    return cast(anthropic.Anthropic, fake)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_post(text: str = "Cool paper showing X causes Y by mechanism Z.") -> ResolvedPost:
    return ResolvedPost(
        post_id="abc123",
        platform="bluesky",
        text=text,
        author_handle="example.bsky.social",
        author_id="did:plc:example",
        url="https://bsky.app/profile/example.bsky.social/post/abc123",
        created_at="2026-06-01T12:00:00+00:00",
        raw_title="Cool paper showing X causes Y by mechanism Z.",
        text_confidence="high",
    )


def _make_paper(title: str = "X causes Y: a mechanistic study") -> PaperCtx:
    return PaperCtx(
        title=title,
        abstract="We show that X causes Y through mechanism Z in N=200 subjects.",
        url="https://doi.org/10.0/example",
    )


def _tool_use_response(payload: dict[str, Any]) -> FakeResponse:
    return FakeResponse(
        content=[
            FakeBlock(type="tool_use", name=CLASSIFIER_TOOL_NAME, input=payload, id="toolu_1"),
        ],
    )


# ---------------------------------------------------------------------------
# Classifier round-trip tests (mocked client)
# ---------------------------------------------------------------------------


def test_endorsement_round_trip() -> None:
    payload = {
        "kind": "endorsement",
        "claim_text": "X causes Y via mechanism Z.",
        "magnitude_dB": 20,
        "criterion": "Support",
        "reasoning": "The post explicitly paraphrases the mechanistic claim and agrees.",
    }
    client = FakeClient(_tool_use_response(payload))

    result = classify_post(_as_client(client), _make_post(), _make_paper())

    assert isinstance(result, Endorsement)
    assert result.claim_text == "X causes Y via mechanism Z."
    assert result.magnitude_dB == 20
    assert result.criterion == "Support"
    assert result.reasoning.startswith("The post")


def test_flag_round_trip() -> None:
    payload = {
        "kind": "flag",
        "category": "methodological",
        "rationale": "The post notes the sample size of N=200 is underpowered.",
    }
    client = FakeClient(_tool_use_response(payload))

    result = classify_post(_as_client(client), _make_post(), _make_paper())

    assert isinstance(result, Flag)
    assert result.category == "methodological"
    assert "sample size" in result.rationale


def test_irrelevant_round_trip() -> None:
    payload = {
        "kind": "irrelevant",
        "reason": "Vague praise with no specific claim.",
    }
    client = FakeClient(_tool_use_response(payload))

    result = classify_post(_as_client(client), _make_post(), _make_paper())

    assert isinstance(result, Irrelevant)
    assert result.reason == "Vague praise with no specific claim."


def test_zero_magnitude_demoted_to_irrelevant() -> None:
    payload = {
        "kind": "endorsement",
        "claim_text": "X causes Y.",
        "magnitude_dB": 0,
        "criterion": "Support",
        "reasoning": "Borderline; no clear stance.",
    }
    client = FakeClient(_tool_use_response(payload))

    result = classify_post(_as_client(client), _make_post(), _make_paper())

    assert isinstance(result, Irrelevant)
    assert "Zero-magnitude" in result.reason


def test_missing_required_field_raises() -> None:
    # Flag without ``category`` — pydantic should reject it.
    payload = {
        "kind": "flag",
        "rationale": "Methodological concern but category is missing.",
    }
    client = FakeClient(_tool_use_response(payload))

    with pytest.raises(ValidationError):
        classify_post(_as_client(client), _make_post(), _make_paper())


def test_tool_use_block_extracted_from_mixed_content() -> None:
    payload = {
        "kind": "irrelevant",
        "reason": "Off-topic.",
    }
    response = FakeResponse(
        content=[
            FakeBlock(type="text", text="Thinking through the post..."),
            FakeBlock(type="tool_use", name=CLASSIFIER_TOOL_NAME, input=payload, id="toolu_x"),
        ],
    )
    client = FakeClient(response)

    result = classify_post(_as_client(client), _make_post(), _make_paper())

    assert isinstance(result, Irrelevant)
    assert result.reason == "Off-topic."


def test_system_prompt_marked_for_caching() -> None:
    payload = {"kind": "irrelevant", "reason": "Off-topic."}
    client = FakeClient(_tool_use_response(payload))

    classify_post(_as_client(client), _make_post(), _make_paper())

    assert client.messages.record_calls, "messages.create was not called"
    kwargs = client.messages.record_calls[-1]
    system = kwargs["system"]
    assert isinstance(system, list), "system should be a list for cache_control to attach"
    assert len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    assert block["text"] == CLASSIFIER_SYSTEM_PROMPT
    assert block["cache_control"] == {"type": "ephemeral"}

    # The forced-tool contract is part of the structured-output guarantee.
    assert kwargs["tool_choice"] == {"type": "tool", "name": CLASSIFIER_TOOL_NAME}
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == CLASSIFIER_TOOL_NAME
    assert tools[0]["input_schema"]["required"] == ["kind"]


def test_user_message_includes_post_and_paper() -> None:
    payload = {"kind": "irrelevant", "reason": "Off-topic."}
    client = FakeClient(_tool_use_response(payload))

    post = _make_post(text="Replication failed on the X-causes-Y claim.")
    paper = _make_paper(title="X causes Y: a mechanistic study")

    classify_post(_as_client(client), post, paper)

    kwargs = client.messages.record_calls[-1]
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, str)
    assert "Replication failed on the X-causes-Y claim." in content
    assert "X causes Y: a mechanistic study" in content


# ---------------------------------------------------------------------------
# Direct schema-validation tests
# ---------------------------------------------------------------------------


def test_parse_tool_input_endorsement() -> None:
    payload = {
        "kind": "endorsement",
        "claim_text": "X causes Y.",
        "magnitude_dB": 10,
        "criterion": "Support",
        "reasoning": "Post agrees with the mechanistic claim.",
    }
    result = parse_tool_input(payload)
    assert isinstance(result, Endorsement)
    assert result.magnitude_dB == 10


def test_parse_tool_input_flag() -> None:
    payload = {
        "kind": "flag",
        "category": "bias",
        "rationale": "Undisclosed funding mentioned in the post.",
    }
    result = parse_tool_input(payload)
    assert isinstance(result, Flag)
    assert result.category == "bias"


def test_parse_tool_input_irrelevant() -> None:
    payload = {"kind": "irrelevant", "reason": "Just emojis."}
    result = parse_tool_input(payload)
    assert isinstance(result, Irrelevant)
    assert result.reason == "Just emojis."


def test_parse_tool_input_zero_magnitude_raises() -> None:
    payload = {
        "kind": "endorsement",
        "claim_text": "X causes Y.",
        "magnitude_dB": 0,
        "criterion": "Support",
        "reasoning": "No clear stance.",
    }
    with pytest.raises(ZeroMagnitudeError):
        parse_tool_input(payload)


def test_tool_input_schema_shape() -> None:
    schema = tool_input_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["kind"]
    props = schema["properties"]
    assert set(props["kind"]["enum"]) == {"endorsement", "flag", "irrelevant"}
    assert props["magnitude_dB"]["minimum"] == -30
    assert props["magnitude_dB"]["maximum"] == 30
    assert set(props["category"]["enum"]) == {"methodological", "source", "data", "bias", "other"}
    assert set(props["criterion"]["enum"]) == {"Support", "Prior"}
