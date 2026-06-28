"""Offline tests for :mod:`altendor.route.question_router` (S15).

Two surfaces under test:

* :func:`route_paper_to_question` — exercised with a fake Anthropic client
  that captures the kwargs passed to ``messages.create`` and returns a
  canned ``tool_use`` block.
* :func:`diversify_routes` — pure-Python rebalance with no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import anthropic
import pytest
from altendor.route.question_router import (
    THREE_QUESTIONS,
    PaperForRouting,
    QuestionStub,
    diversify_routes,
    route_paper_to_question,
)

# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolUseBlock:
    """Minimal stand-in for ``anthropic.types.ToolUseBlock``."""

    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _FakeMessage:
    """Minimal stand-in for an Anthropic ``Message`` response."""

    content: list[_FakeToolUseBlock]


class _FakeMessages:
    def __init__(self, payload: dict[str, Any], tool_name: str = "route_paper") -> None:
        self._payload = payload
        self._tool_name = tool_name
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeMessage:  # noqa: ANN401 - Anthropic SDK takes heterogeneous kwargs
        self.calls.append(kwargs)
        return _FakeMessage(content=[_FakeToolUseBlock(name=self._tool_name, input=self._payload)])


class _FakeAnthropic:
    """Duck-typed stand-in exposing ``.messages.create``."""

    def __init__(self, payload: dict[str, Any], tool_name: str = "route_paper") -> None:
        self.messages = _FakeMessages(payload, tool_name=tool_name)


def _as_client(fake: _FakeAnthropic) -> anthropic.Anthropic:
    """Cast helper so the type checker accepts the duck-typed fake."""
    return cast(anthropic.Anthropic, fake)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


PAPER = PaperForRouting(
    doi="10.1000/example",
    title="A study on peer review bias",
    abstract="We investigate biases introduced during peer review of journal submissions.",
)


# ---------------------------------------------------------------------------
# route_paper_to_question
# ---------------------------------------------------------------------------


def test_routes_to_chosen_question() -> None:
    """When the model returns a valid tool call, the function returns it verbatim."""
    fake = _FakeAnthropic({"chosen_question_id": "question:peer-review", "confidence": 0.8})
    qid, conf = route_paper_to_question(_as_client(fake), PAPER)
    assert qid == "question:peer-review"
    assert conf == pytest.approx(0.8)


def test_unknown_question_id_raises() -> None:
    """Tool output with an unknown question id must raise ValueError."""
    fake = _FakeAnthropic({"chosen_question_id": "question:bogus", "confidence": 0.5})
    with pytest.raises(ValueError):
        route_paper_to_question(_as_client(fake), PAPER)


def test_confidence_out_of_range_raises() -> None:
    """Confidence outside [0, 1] must raise ValueError."""
    fake = _FakeAnthropic({"chosen_question_id": "question:peer-review", "confidence": 1.5})
    with pytest.raises(ValueError):
        route_paper_to_question(_as_client(fake), PAPER)


def test_tool_choice_forces_route_paper_tool() -> None:
    """tool_choice must force the model to call ``route_paper``."""
    fake = _FakeAnthropic({"chosen_question_id": "question:peer-review", "confidence": 0.5})
    route_paper_to_question(_as_client(fake), PAPER)
    kwargs = fake.messages.calls[0]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "route_paper"}
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "route_paper"
    enum_ids = tools[0]["input_schema"]["properties"]["chosen_question_id"]["enum"]
    assert set(enum_ids) == {q.id for q in THREE_QUESTIONS}


# ---------------------------------------------------------------------------
# diversify_routes
# ---------------------------------------------------------------------------


def _qid(i: int) -> str:
    """Shortcut: question id at index ``i`` of ``THREE_QUESTIONS``."""
    return THREE_QUESTIONS[i].id


def test_diversify_no_op_when_all_questions_have_papers() -> None:
    """If every Question already has ≥1 paper, the routing is preserved."""
    routes: dict[str, tuple[str, float]] = {
        "a": (_qid(0), 0.9),
        "b": (_qid(1), 0.8),
        "c": (_qid(2), 0.4),
    }
    out = diversify_routes(routes)
    assert out == {"a": _qid(0), "b": _qid(1), "c": _qid(2)}


def test_diversify_rescues_empty_question() -> None:
    """The lowest-confidence paper in the largest bucket fills the empty one."""
    routes: dict[str, tuple[str, float]] = {
        "a": (_qid(0), 0.9),  # donor bucket (3 papers)
        "b": (_qid(0), 0.3),  # donor bucket - lowest confidence => moves
        "c": (_qid(0), 0.7),  # donor bucket
        "d": (_qid(1), 0.6),  # neighbour - 1 paper
        # _qid(2) is empty
    }
    out = diversify_routes(routes)
    assert out["b"] == _qid(2), "lowest-confidence paper should fill the empty Question"
    # The other three stay where they were.
    assert out["a"] == _qid(0)
    assert out["c"] == _qid(0)
    assert out["d"] == _qid(1)


def test_diversify_picks_lowest_confidence_when_tied() -> None:
    """Within a donor bucket, the lower-confidence paper moves."""
    routes: dict[str, tuple[str, float]] = {
        "high": (_qid(0), 0.95),
        "low": (_qid(0), 0.20),
        "other": (_qid(1), 0.5),
        # _qid(2) is empty
    }
    out = diversify_routes(routes)
    assert out["low"] == _qid(2)
    assert out["high"] == _qid(0)
    assert out["other"] == _qid(1)


def test_diversify_no_rescue_when_no_bucket_has_two() -> None:
    """Cannot rescue if doing so would just relocate an empty: leave as-is."""
    routes: dict[str, tuple[str, float]] = {
        "a": (_qid(0), 0.8),
        "b": (_qid(1), 0.5),
        # _qid(2) is empty, but neither donor bucket has >= 2 papers.
    }
    out = diversify_routes(routes)
    assert out == {"a": _qid(0), "b": _qid(1)}


def test_diversify_preserves_paper_order() -> None:
    """The output dict preserves the input dict's key order."""
    routes: dict[str, tuple[str, float]] = {
        "a": (_qid(0), 0.9),
        "b": (_qid(0), 0.3),
        "c": (_qid(0), 0.7),
        "d": (_qid(1), 0.6),
    }
    out = diversify_routes(routes)
    assert list(out.keys()) == ["a", "b", "c", "d"]


def test_diversify_handles_two_empty_questions() -> None:
    """Two empties, one big bucket: rescue what we can.

    Bucket sizes after one move (3 → 2, fill empty): donor still has 2, so
    we move the next lowest-confidence one and fill the second empty.
    """
    routes: dict[str, tuple[str, float]] = {
        "a": (_qid(0), 0.9),
        "b": (_qid(0), 0.4),
        "c": (_qid(0), 0.6),
        # _qid(1) and _qid(2) both empty
    }
    out = diversify_routes(routes)
    assigned = sorted(out.values())
    expected = sorted([_qid(0), _qid(1), _qid(2)])
    assert assigned == expected
    # Lowest confidence ("b", 0.4) moves first to the first empty (_qid(1)).
    # Then donor has {a:0.9, c:0.6}; lower of the two is "c" -> fills _qid(2).
    assert out["b"] == _qid(1)
    assert out["c"] == _qid(2)
    assert out["a"] == _qid(0)


def test_diversify_accepts_custom_questions() -> None:
    """The function works with a custom tuple of QuestionStubs."""
    custom = (
        QuestionStub("q:one", "One?"),
        QuestionStub("q:two", "Two?"),
    )
    routes: dict[str, tuple[str, float]] = {
        "a": ("q:one", 0.9),
        "b": ("q:one", 0.2),
        # q:two empty
    }
    out = diversify_routes(routes, questions=custom)
    assert out["b"] == "q:two"
    assert out["a"] == "q:one"
