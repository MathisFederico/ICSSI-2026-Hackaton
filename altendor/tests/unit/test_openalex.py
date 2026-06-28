"""Offline tests for :mod:`altendor.sources.openalex`.

Each test stubs the network by injecting a fake :class:`httpx.Client` whose
``get`` method returns a pre-recorded response sourced from the JSON fixtures
in ``altendor/tests/fixtures/``. No test in this module hits the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from altendor.sources import openalex

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open() as f:
        return json.load(f)


class _StubResponse:
    """Minimal stand-in for :class:`httpx.Response` used by the source."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"stub status {self.status_code}")


class _StubClient:
    """Captures GETs and returns successive scripted payloads.

    ``responses`` is consumed in order; once exhausted the last response is
    returned to keep tests forgiving of extra calls.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self._index = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> _StubResponse:
        self.calls.append((url, dict(params or {})))
        payload = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return _StubResponse(payload)

    def close(self) -> None:  # pragma: no cover - never called in tests
        return None


def _as_client(stub: _StubClient) -> httpx.Client:
    """Cast a :class:`_StubClient` to :class:`httpx.Client` for the type-checker.

    The source modules only call ``.get`` and ``.close`` on the client, both of
    which the stub implements with matching shapes.
    """
    return cast(httpx.Client, stub)


def _works_payload_with_cursor(payload: dict[str, Any], next_cursor: str | None) -> dict[str, Any]:
    """Return a copy of *payload* with ``meta.next_cursor`` overridden."""
    new = json.loads(json.dumps(payload))
    new.setdefault("meta", {})["next_cursor"] = next_cursor
    return new


def test_search_dois_extracts_dois_from_fixture() -> None:
    works = _load_fixture("openalex_works_page.json")
    # Force the cursor to None so the first response terminates pagination.
    page = _works_payload_with_cursor(works, None)
    client = _StubClient([page])

    dois = openalex.search_dois("science of science", n=50, client=_as_client(client))

    assert dois == [
        "10.2307/1270020",
        "10.1016/0198-9715(90)90050-4",
        "10.2307/3498751",
        "10.1108/ir.1999.04926fae.001",
        "10.3758/bf03193146",
    ]
    # No DOI carries the URL prefix in the result list.
    assert all(not d.startswith("https://") for d in dois)


def test_search_dois_skips_missing_doi() -> None:
    page = {
        "meta": {"next_cursor": None},
        "results": [
            {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/a", "display_name": "x"},
            {"id": "https://openalex.org/W2", "doi": None, "display_name": "y"},
            {"id": "https://openalex.org/W3", "doi": "https://doi.org/10.2/b", "display_name": "z"},
        ],
    }
    client = _StubClient([page])

    dois = openalex.search_dois(n=10, client=_as_client(client))

    assert dois == ["10.1/a", "10.2/b"]


def test_search_dois_filter_selection() -> None:
    empty_page = {"meta": {"next_cursor": None}, "results": []}

    # Default keyword filter.
    keyword_client = _StubClient([empty_page])
    openalex.search_dois("network science", n=5, client=_as_client(keyword_client))
    assert keyword_client.calls, "expected at least one GET"
    _, params = keyword_client.calls[0]
    assert params["filter"] == "title_and_abstract.search:network science"

    # topic_id override beats keyword.
    topic_client = _StubClient([empty_page])
    openalex.search_dois("ignored", topic_id="T12345", n=5, client=_as_client(topic_client))
    _, params = topic_client.calls[0]
    assert params["filter"] == "topics.id:T12345"

    # concept_id override (when topic_id is unset).
    concept_client = _StubClient([empty_page])
    openalex.search_dois("ignored", concept_id="C98765", n=5, client=_as_client(concept_client))
    _, params = concept_client.calls[0]
    assert params["filter"] == "concepts.id:C98765"


def test_resolve_topic_returns_top_match() -> None:
    topics = _load_fixture("openalex_topics_search.json")
    client = _StubClient([topics])

    result = openalex.resolve_topic_by_name("science of science", client=_as_client(client))

    assert result is not None
    assert result["id"] == "T10192"
    assert result["display_name"] == "Catalytic Processes in Materials Science"
    assert isinstance(result["works_count"], int)
    assert result["works_count"] > 0


def test_resolve_topic_returns_none_when_empty() -> None:
    client = _StubClient([{"meta": {"count": 0}, "results": []}])

    assert openalex.resolve_topic_by_name("no-such-topic", client=_as_client(client)) is None


def test_polite_pool_param_present() -> None:
    empty_page = {"meta": {"next_cursor": None}, "results": []}

    works_client = _StubClient([empty_page])
    openalex.search_dois("anything", n=5, client=_as_client(works_client))
    assert works_client.calls
    for _, params in works_client.calls:
        assert params.get("mailto") == openalex.POLITE_EMAIL

    topics_client = _StubClient([{"meta": {}, "results": []}])
    openalex.resolve_topic_by_name("anything", client=_as_client(topics_client))
    assert topics_client.calls
    for _, params in topics_client.calls:
        assert params.get("mailto") == openalex.POLITE_EMAIL


def test_search_dois_paginates_via_cursor() -> None:
    """Sanity-check the cursor loop: two pages stitched together up to *n*."""
    page1 = {
        "meta": {"next_cursor": "abc"},
        "results": [
            {"id": "W1", "doi": "https://doi.org/10.1/a"},
            {"id": "W2", "doi": "https://doi.org/10.1/b"},
        ],
    }
    page2 = {
        "meta": {"next_cursor": None},
        "results": [
            {"id": "W3", "doi": "https://doi.org/10.1/c"},
        ],
    }
    client = _StubClient([page1, page2])

    dois = openalex.search_dois(n=10, client=_as_client(client))

    assert dois == ["10.1/a", "10.1/b", "10.1/c"]
    assert len(client.calls) == 2
    assert client.calls[0][1]["cursor"] == "*"
    assert client.calls[1][1]["cursor"] == "abc"


@pytest.mark.parametrize("missing_doi", [None, ""])
def test_strip_doi_handles_falsy(missing_doi: str | None) -> None:
    assert openalex._strip_doi(missing_doi) is None
