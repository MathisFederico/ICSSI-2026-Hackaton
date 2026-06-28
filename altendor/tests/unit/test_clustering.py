"""Offline unit tests for :mod:`altendor.cluster.claims` (S14).

Same fake-client pattern as :mod:`test_classifier_mocked` and
:mod:`test_routing`: we wire a duck-typed stand-in into :func:`cluster_claims`
and assert on the captured ``messages.create`` kwargs plus the returned
``list[ClaimCluster]``. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
from altendor.cluster.claims import (
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    TOOL_NAME,
    ClaimCluster,
    cluster_claims,
)

# ---------------------------------------------------------------------------
# Fake Anthropic client plumbing
# ---------------------------------------------------------------------------


@dataclass
class _FakeBlock:
    """Duck-types an Anthropic ``tool_use`` content block."""

    type: str
    name: str | None = None
    input: dict[str, Any] | None = None
    id: str | None = None
    text: str | None = None


@dataclass
class _FakeResponse:
    """Duck-types ``anthropic.types.Message`` enough for the clusterer."""

    content: list[_FakeBlock] = field(default_factory=list)


class _FakeMessages:
    """Records kwargs and returns canned responses (or raises)."""

    def __init__(self, response: _FakeResponse | None, *, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:  # noqa: ANN401 - SDK takes heterogeneous kwargs
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


class _FakeClient:
    """Minimal duck-typed stand-in for :class:`anthropic.Anthropic`."""

    def __init__(self, response: _FakeResponse | None = None, *, exc: Exception | None = None) -> None:
        self.messages = _FakeMessages(response, exc=exc)


def _as_client(fake: _FakeClient) -> anthropic.Anthropic:
    """Cast helper so the type checker accepts the duck-typed fake."""
    return cast(anthropic.Anthropic, fake)


def _tool_use_response(clusters: list[dict[str, Any]]) -> _FakeResponse:
    return _FakeResponse(
        content=[_FakeBlock(type="tool_use", name=TOOL_NAME, input={"clusters": clusters}, id="toolu_1")]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_small_input_skips_llm() -> None:
    """<= MIN_CLUSTERS inputs: return one cluster per input, no API call."""
    claim_texts = {"p1": "Claim one.", "p2": "Claim two.", "p3": "Claim three."}
    client = _FakeClient(_tool_use_response([]))  # response is unused

    out = cluster_claims(_as_client(client), claim_texts)

    assert len(out) == MIN_CLUSTERS == 3
    # No call to messages.create.
    assert client.messages.calls == []
    # One-to-one identity clustering: each canonical_text equals an input text verbatim.
    canonicals = {c.canonical_text for c in out}
    assert canonicals == set(claim_texts.values())
    # Every input id appears in exactly one cluster, alone.
    flat_ids = [pid for c in out for pid in c.member_post_ids]
    assert sorted(flat_ids) == sorted(claim_texts.keys())
    for c in out:
        assert len(c.member_post_ids) == 1


def test_normal_clustering_round_trip() -> None:
    """Fake returns 4 valid clusters covering all 10 ids; sorted desc by size."""
    claim_texts = {f"p{i}": f"claim {i}" for i in range(10)}
    clusters_payload = [
        {"canonical_text": "A", "member_post_ids": ["p0", "p1", "p2", "p3"]},  # 4
        {"canonical_text": "B", "member_post_ids": ["p4", "p5", "p6"]},  # 3
        {"canonical_text": "C", "member_post_ids": ["p7", "p8"]},  # 2
        {"canonical_text": "D", "member_post_ids": ["p9"]},  # 1
    ]
    client = _FakeClient(_tool_use_response(clusters_payload))

    out = cluster_claims(_as_client(client), claim_texts)

    assert len(out) == 4
    assert all(isinstance(c, ClaimCluster) for c in out)
    # Descending order by member count.
    assert [len(c.member_post_ids) for c in out] == [4, 3, 2, 1]
    assert [c.canonical_text for c in out] == ["A", "B", "C", "D"]
    # All ids covered exactly once.
    flat = sorted(pid for c in out for pid in c.member_post_ids)
    assert flat == sorted(claim_texts)


def test_too_few_clusters_falls_back_to_mega() -> None:
    """Fake returns 1 cluster — below MIN_CLUSTERS — so we collapse to mega."""
    claim_texts = {f"p{i}": f"claim {i}" for i in range(5)}
    clusters_payload = [
        {"canonical_text": "The only cluster", "member_post_ids": ["p0", "p1", "p2", "p3", "p4"]},
    ]
    client = _FakeClient(_tool_use_response(clusters_payload))

    out = cluster_claims(_as_client(client), claim_texts)

    assert len(out) == 1
    mega = out[0]
    assert mega.canonical_text == "Various claims about this paper."
    assert mega.member_post_ids == sorted(claim_texts)


def test_too_many_clusters_clamped_to_max() -> None:
    """Fake returns 10 disjoint clusters; we merge the smallest into the largest until 7."""
    # Build 10 clusters with distinct sizes so the merge order is deterministic.
    # Cluster k has (k+1) members: c0=1, c1=2, ..., c9=10. Total members = 55.
    claim_texts: dict[str, str] = {}
    clusters_payload: list[dict[str, Any]] = []
    next_id = 0
    for k in range(10):
        ids = []
        for _ in range(k + 1):
            pid = f"p{next_id}"
            claim_texts[pid] = f"claim {next_id}"
            ids.append(pid)
            next_id += 1
        clusters_payload.append({"canonical_text": f"C{k}", "member_post_ids": ids})

    client = _FakeClient(_tool_use_response(clusters_payload))

    out = cluster_claims(_as_client(client), claim_texts)

    assert len(out) == MAX_CLUSTERS == 7
    # All input ids preserved.
    flat = sorted(pid for c in out for pid in c.member_post_ids)
    assert flat == sorted(claim_texts)
    # No duplicate ids across clusters.
    flat_dupes = [pid for c in out for pid in c.member_post_ids]
    assert len(flat_dupes) == len(set(flat_dupes))
    # The smallest three (c0=1, c1=2, c2=3) should have been merged into the
    # largest (c9=10), so one cluster contains the merged 10+1+2+3 = 16 ids.
    sizes = sorted((len(c.member_post_ids) for c in out), reverse=True)
    assert sizes[0] == 16  # 10 + 3 + 2 + 1
    # Remaining clusters keep their original sizes: 9, 8, 7, 6, 5, 4.
    assert sizes[1:] == [9, 8, 7, 6, 5, 4]


def test_missing_ids_appended_to_smallest() -> None:
    """Fake omits two input ids; they get appended to the smallest cluster."""
    claim_texts = {f"p{i}": f"claim {i}" for i in range(8)}
    # Returns 3 clusters; omit p6 and p7. Smallest cluster will be the one of size 2.
    clusters_payload = [
        {"canonical_text": "Big", "member_post_ids": ["p0", "p1", "p2"]},
        {"canonical_text": "Mid", "member_post_ids": ["p3", "p4"]},
        {"canonical_text": "Small", "member_post_ids": ["p5"]},  # smallest pre-rescue
    ]
    client = _FakeClient(_tool_use_response(clusters_payload))

    out = cluster_claims(_as_client(client), claim_texts)

    flat = sorted(pid for c in out for pid in c.member_post_ids)
    assert flat == sorted(claim_texts), "no id may be dropped"
    # p6 and p7 land in the cluster that was smallest pre-rescue ("Small").
    small_cluster = next(c for c in out if c.canonical_text == "Small")
    assert "p6" in small_cluster.member_post_ids
    assert "p7" in small_cluster.member_post_ids


def test_api_error_falls_back_to_mega() -> None:
    """``messages.create`` raises — we return a single mega-cluster."""
    claim_texts = {f"p{i}": f"claim {i}" for i in range(6)}
    client = _FakeClient(exc=RuntimeError("boom"))

    out = cluster_claims(_as_client(client), claim_texts)

    assert len(out) == 1
    assert out[0].canonical_text == "Various claims about this paper."
    assert out[0].member_post_ids == sorted(claim_texts)


def test_clusters_sorted_desc_by_member_count() -> None:
    """Sizes [2, 5, 3] → output order [5, 3, 2]."""
    # Need > MIN_CLUSTERS=3 inputs to trigger the LLM path. 10 here.
    claim_texts = {f"p{i}": f"claim {i}" for i in range(10)}
    clusters_payload = [
        {"canonical_text": "two", "member_post_ids": ["p0", "p1"]},  # 2
        {"canonical_text": "five", "member_post_ids": ["p2", "p3", "p4", "p5", "p6"]},  # 5
        {"canonical_text": "three", "member_post_ids": ["p7", "p8", "p9"]},  # 3
    ]
    client = _FakeClient(_tool_use_response(clusters_payload))

    out = cluster_claims(_as_client(client), claim_texts)

    assert [len(c.member_post_ids) for c in out] == [5, 3, 2]
    assert [c.canonical_text for c in out] == ["five", "three", "two"]


def test_tool_choice_forces_tool() -> None:
    """tool_choice must force the model to call ``cluster_claims``."""
    claim_texts = {f"p{i}": f"claim {i}" for i in range(6)}
    payload = [
        {"canonical_text": "A", "member_post_ids": ["p0", "p1"]},
        {"canonical_text": "B", "member_post_ids": ["p2", "p3"]},
        {"canonical_text": "C", "member_post_ids": ["p4", "p5"]},
    ]
    client = _FakeClient(_tool_use_response(payload))

    cluster_claims(_as_client(client), claim_texts)

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]
    assert kwargs["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == TOOL_NAME
    schema = tools[0]["input_schema"]
    assert schema["required"] == ["clusters"]
    items = schema["properties"]["clusters"]["items"]
    assert items["required"] == ["canonical_text", "member_post_ids"]
    assert schema["properties"]["clusters"]["minItems"] == MIN_CLUSTERS
    assert schema["properties"]["clusters"]["maxItems"] == MAX_CLUSTERS


def test_user_message_includes_all_ids() -> None:
    """Every input post_id must appear in the rendered user message."""
    claim_texts = {f"post-{i}": f"claim text {i}" for i in range(6)}
    payload = [
        {"canonical_text": "A", "member_post_ids": ["post-0", "post-1"]},
        {"canonical_text": "B", "member_post_ids": ["post-2", "post-3"]},
        {"canonical_text": "C", "member_post_ids": ["post-4", "post-5"]},
    ]
    client = _FakeClient(_tool_use_response(payload))

    cluster_claims(_as_client(client), claim_texts)

    kwargs = client.messages.calls[0]
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, str)
    for pid in claim_texts:
        assert pid in content, f"user message missing post id {pid!r}"


def test_member_post_ids_are_unique_within_cluster() -> None:
    """Duplicated ids within a returned cluster must be de-duplicated."""
    claim_texts = {f"p{i}": f"claim {i}" for i in range(6)}
    # First cluster has p0 listed twice and shares p1 — only the first occurrence wins.
    payload = [
        {"canonical_text": "A", "member_post_ids": ["p0", "p0", "p1"]},
        {"canonical_text": "B", "member_post_ids": ["p2", "p3"]},
        {"canonical_text": "C", "member_post_ids": ["p4", "p5"]},
    ]
    client = _FakeClient(_tool_use_response(payload))

    out = cluster_claims(_as_client(client), claim_texts)

    a_cluster = next(c for c in out if c.canonical_text == "A")
    assert a_cluster.member_post_ids.count("p0") == 1
    # And no id is dropped overall.
    flat = sorted(pid for c in out for pid in c.member_post_ids)
    assert flat == sorted(claim_texts)
