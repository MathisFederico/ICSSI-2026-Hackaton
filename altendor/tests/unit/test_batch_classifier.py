"""Mocked-client unit tests for :mod:`altendor.classify.batch` (S11).

These tests never touch the real Anthropic SDK — they wire a duck-typed
fake client into the batch driver and assert on the captured submit
payload plus the parsed :class:`BatchResultRow`s. The "results iterator"
is faked as a plain ``iter([...])`` over local dataclasses; we do not try
to emulate the SDK's streaming JSONL format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import pytest
from altendor.classify.batch import (
    BatchRequestSpec,
    BatchResultRow,
    build_batch_requests,
    parse_batch_results,
    poll_until_done,
    submit_batch,
)
from altendor.classify.classifier import DEFAULT_MODEL, PaperCtx
from altendor.classify.prompts import CLASSIFIER_TOOL_NAME
from altendor.classify.schema import Endorsement, Flag, Irrelevant
from altendor.enrich.text_resolver import ResolvedPost

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
class FakeMessage:
    """Duck-types ``MessageBatchIndividualResponse.result.message``."""

    content: list[FakeBlock]


@dataclass
class FakeError:
    """Duck-types the ``error`` payload on an errored batch result."""

    type: str
    message: str


@dataclass
class FakeResult:
    """Duck-types ``MessageBatchIndividualResponse.result``."""

    type: str
    message: FakeMessage | None = None
    error: FakeError | None = None


@dataclass
class FakeEntry:
    """Duck-types one ``MessageBatchIndividualResponse``."""

    custom_id: str
    result: FakeResult


@dataclass
class FakeBatch:
    """Duck-types ``MessageBatch`` (the object create/retrieve return)."""

    id: str
    processing_status: str


@dataclass
class FakeBatchesAPI:
    """Captures ``create`` kwargs; serves canned ``retrieve``/``results`` data."""

    create_id: str = "msgbatch_test"
    retrieve_sequence: list[FakeBatch] = field(default_factory=list)
    results_entries: list[FakeEntry] = field(default_factory=list)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieve_calls: list[str] = field(default_factory=list)
    results_calls: list[str] = field(default_factory=list)
    _retrieve_index: int = 0

    def create(self, **kwargs: Any) -> FakeBatch:  # noqa: ANN401 - heterogeneous SDK kwargs
        self.create_calls.append(kwargs)
        return FakeBatch(id=self.create_id, processing_status="in_progress")

    def retrieve(self, batch_id: str) -> FakeBatch:
        self.retrieve_calls.append(batch_id)
        if not self.retrieve_sequence:
            return FakeBatch(id=batch_id, processing_status="in_progress")
        if self._retrieve_index >= len(self.retrieve_sequence):
            return self.retrieve_sequence[-1]
        batch = self.retrieve_sequence[self._retrieve_index]
        self._retrieve_index += 1
        return batch

    def results(self, batch_id: str) -> Any:  # noqa: ANN401 - SDK returns an iterator
        self.results_calls.append(batch_id)
        return iter(self.results_entries)


class FakeMessages:
    """Duck-types ``client.messages`` exposing only ``.batches``."""

    def __init__(self, batches: FakeBatchesAPI) -> None:
        self.batches = batches


class FakeClient:
    """Minimal duck-typed stand-in for :class:`anthropic.Anthropic`."""

    def __init__(self, batches: FakeBatchesAPI | None = None) -> None:
        self.batches_api: FakeBatchesAPI = batches if batches is not None else FakeBatchesAPI()
        self.messages = FakeMessages(self.batches_api)


def _as_client(fake: FakeClient) -> anthropic.Anthropic:
    """Cast helper so the type checker accepts the duck-typed fake."""
    return cast(anthropic.Anthropic, fake)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_post(post_id: str = "abc123", text: str = "Cool paper showing X causes Y.") -> ResolvedPost:
    return ResolvedPost(
        post_id=post_id,
        platform="bluesky",
        text=text,
        author_handle="example.bsky.social",
        author_id="did:plc:example",
        url=f"https://bsky.app/profile/example/post/{post_id}",
        created_at="2026-06-01T12:00:00+00:00",
        raw_title=text,
        text_confidence="high",
    )


def _make_paper(title: str = "X causes Y: a mechanistic study") -> PaperCtx:
    return PaperCtx(
        title=title,
        abstract="We show X causes Y via Z.",
        url="https://doi.org/10.0/example",
    )


def _make_spec(custom_id: str = "row-1") -> BatchRequestSpec:
    return BatchRequestSpec(custom_id=custom_id, post=_make_post(post_id=custom_id), paper=_make_paper())


def _succeeded_entry(custom_id: str, payload: dict[str, Any]) -> FakeEntry:
    return FakeEntry(
        custom_id=custom_id,
        result=FakeResult(
            type="succeeded",
            message=FakeMessage(
                content=[
                    FakeBlock(type="tool_use", name=CLASSIFIER_TOOL_NAME, input=payload, id="toolu_1"),
                ],
            ),
        ),
    )


# ---------------------------------------------------------------------------
# build_batch_requests
# ---------------------------------------------------------------------------


def test_build_batch_requests_shape() -> None:
    specs = [_make_spec("row-1"), _make_spec("row-2")]
    rendered = build_batch_requests(specs)

    assert len(rendered) == 2
    for entry, spec in zip(rendered, specs):
        assert entry["custom_id"] == spec.custom_id
        params = entry["params"]
        assert params["model"] == DEFAULT_MODEL
        system = params["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        tools = params["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == CLASSIFIER_TOOL_NAME
        assert params["tool_choice"]["name"] == CLASSIFIER_TOOL_NAME
        messages = params["messages"]
        assert len(messages) >= 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], str)
        assert messages[0]["content"]


# ---------------------------------------------------------------------------
# submit_batch
# ---------------------------------------------------------------------------


def test_submit_batch_returns_id() -> None:
    batches = FakeBatchesAPI(create_id="msgbatch_test")
    client = FakeClient(batches)

    batch_id = submit_batch(_as_client(client), [_make_spec("row-1"), _make_spec("row-2")])

    assert batch_id == "msgbatch_test"
    assert len(batches.create_calls) == 1
    submitted = batches.create_calls[0]["requests"]
    assert [entry["custom_id"] for entry in submitted] == ["row-1", "row-2"]


def test_submit_batch_rejects_empty_specs() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="at least one"):
        submit_batch(_as_client(client), [])


def test_submit_batch_rejects_duplicate_custom_ids() -> None:
    client = FakeClient()
    specs = [_make_spec("dup"), _make_spec("dup")]
    with pytest.raises(ValueError, match="unique custom_ids"):
        submit_batch(_as_client(client), specs)


# ---------------------------------------------------------------------------
# poll_until_done
# ---------------------------------------------------------------------------


def test_poll_until_done_returns_when_ended() -> None:
    sequence = [
        FakeBatch(id="msgbatch_test", processing_status="in_progress"),
        FakeBatch(id="msgbatch_test", processing_status="in_progress"),
        FakeBatch(id="msgbatch_test", processing_status="ended"),
    ]
    batches = FakeBatchesAPI(retrieve_sequence=sequence)
    client = FakeClient(batches)

    final = poll_until_done(_as_client(client), "msgbatch_test", interval_s=0)

    assert getattr(final, "processing_status", None) == "ended"
    assert len(batches.retrieve_calls) == 3


def test_poll_until_done_timeout() -> None:
    # Always-in-progress: with timeout_s=0 we should bail after the first
    # poll because monotonic() will already have advanced past the start.
    sequence = [FakeBatch(id="msgbatch_test", processing_status="in_progress") for _ in range(5)]
    batches = FakeBatchesAPI(retrieve_sequence=sequence)
    client = FakeClient(batches)

    with pytest.raises(TimeoutError):
        poll_until_done(_as_client(client), "msgbatch_test", interval_s=0, timeout_s=0)


def test_poll_until_done_calls_on_tick() -> None:
    sequence = [
        FakeBatch(id="msgbatch_test", processing_status="in_progress"),
        FakeBatch(id="msgbatch_test", processing_status="ended"),
    ]
    batches = FakeBatchesAPI(retrieve_sequence=sequence)
    client = FakeClient(batches)
    ticks: list[object] = []

    def _record(batch: object) -> None:
        ticks.append(batch)

    poll_until_done(
        _as_client(client),
        "msgbatch_test",
        interval_s=0,
        on_tick=_record,
    )

    assert len(ticks) == 2
    assert getattr(ticks[-1], "processing_status", None) == "ended"


# ---------------------------------------------------------------------------
# parse_batch_results
# ---------------------------------------------------------------------------


def test_parse_batch_results_extracts_endorsements() -> None:
    payload_endorse = {
        "kind": "endorsement",
        "claim_text": "X causes Y via Z.",
        "magnitude_dB": 20,
        "criterion": "Support",
        "reasoning": "The post agrees with the mechanistic claim.",
    }
    payload_flag = {
        "kind": "flag",
        "category": "methodological",
        "rationale": "Sample size is underpowered.",
    }
    entries = [
        _succeeded_entry("row-1", payload_endorse),
        _succeeded_entry("row-2", payload_flag),
    ]
    batches = FakeBatchesAPI(results_entries=entries)
    client = FakeClient(batches)

    rows = parse_batch_results(_as_client(client), "msgbatch_test")

    assert len(rows) == 2
    assert all(isinstance(row, BatchResultRow) for row in rows)
    by_id = {row.custom_id: row for row in rows}
    assert isinstance(by_id["row-1"].result, Endorsement)
    assert by_id["row-1"].error_reason is None
    assert isinstance(by_id["row-2"].result, Flag)
    assert by_id["row-2"].error_reason is None


def test_parse_batch_results_handles_errored() -> None:
    entries = [
        FakeEntry(
            custom_id="row-1",
            result=FakeResult(
                type="errored",
                error=FakeError(type="overloaded_error", message="server is overloaded"),
            ),
        ),
    ]
    batches = FakeBatchesAPI(results_entries=entries)
    client = FakeClient(batches)

    rows = parse_batch_results(_as_client(client), "msgbatch_test")

    assert len(rows) == 1
    row = rows[0]
    assert row.custom_id == "row-1"
    assert row.result is None
    assert row.error_reason is not None
    assert "errored" in row.error_reason
    assert "overloaded_error" in row.error_reason


def test_parse_batch_results_handles_missing_tool_use() -> None:
    entries = [
        FakeEntry(
            custom_id="row-1",
            result=FakeResult(
                type="succeeded",
                message=FakeMessage(content=[FakeBlock(type="text", text="No tool call here.")]),
            ),
        ),
    ]
    batches = FakeBatchesAPI(results_entries=entries)
    client = FakeClient(batches)

    rows = parse_batch_results(_as_client(client), "msgbatch_test")

    assert len(rows) == 1
    row = rows[0]
    assert row.result is None
    assert row.error_reason is not None
    assert "missing_tool_use" in row.error_reason or "No tool_use" in row.error_reason


def test_parse_batch_results_demotes_zero_magnitude() -> None:
    payload = {
        "kind": "endorsement",
        "claim_text": "Borderline claim.",
        "magnitude_dB": 0,
        "criterion": "Support",
        "reasoning": "No clear stance.",
    }
    entries = [_succeeded_entry("row-1", payload)]
    batches = FakeBatchesAPI(results_entries=entries)
    client = FakeClient(batches)

    rows = parse_batch_results(_as_client(client), "msgbatch_test")

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row.result, Irrelevant)
    assert row.error_reason is None
    assert "Zero-magnitude" in row.result.reason
