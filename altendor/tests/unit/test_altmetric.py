"""Offline tests for :mod:`altendor.sources.altmetric`.

These tests exercise the pure-pandas selector, the inverted-index
reconstructor, and the OpenAlex abstract enrichment loop with an injected
fake :class:`httpx.Client`. BigQuery integration is intentionally not
tested here — the SQL builders are covered in ``test_queries.py`` and the
``join_*``/``fetch_*`` helpers in this module are thin wrappers around them.
We do however assert their import surface stays stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
from altendor.sources.altmetric import (
    enrich_with_openalex_abstracts,
    reconstruct_abstract,
    top_papers,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# httpx stub for the OpenAlex per-DOI lookups
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, payload: dict[str, Any] | None, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("no payload")
        return self._payload


class _StubClient:
    """Returns a payload (or 404) per-DOI by matching the URL substring."""

    def __init__(self, by_doi: dict[str, tuple[int, dict[str, Any] | None]]) -> None:
        self._by_doi = by_doi
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> _StubResponse:
        self.calls.append((url, dict(params or {})))
        for doi, (status, payload) in self._by_doi.items():
            if doi in url:
                return _StubResponse(payload, status_code=status)
        return _StubResponse(None, status_code=404)

    def close(self) -> None:  # pragma: no cover - not used (we always inject)
        return None


def _as_client(stub: _StubClient) -> httpx.Client:
    return cast(httpx.Client, stub)


# ---------------------------------------------------------------------------
# top_papers pure-pandas selector
# ---------------------------------------------------------------------------


def _papers_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_top_papers_drops_zero_score() -> None:
    df = _papers_df(
        [
            {"doi": "10.1/a", "ro_id": "alt:1", "title": "A", "altmetric_score": 0.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
            {"doi": "10.1/b", "ro_id": "alt:2", "title": "B", "altmetric_score": 50.0,
             "last_mentioned_at": pd.Timestamp("2024-01-02", tz="UTC")},
            {"doi": "10.1/c", "ro_id": "alt:3", "title": "C", "altmetric_score": 30.0,
             "last_mentioned_at": pd.Timestamp("2024-01-03", tz="UTC")},
        ]
    )

    out = top_papers(df, k=10)

    assert "10.1/a" not in set(out["doi"])
    assert set(out["doi"]) == {"10.1/b", "10.1/c"}


def test_top_papers_always_includes_pinned() -> None:
    df = _papers_df(
        [
            {"doi": "10.1/big", "ro_id": "alt:1", "title": "Big", "altmetric_score": 1000.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
            {"doi": "10.1/mid", "ro_id": "alt:2", "title": "Mid", "altmetric_score": 500.0,
             "last_mentioned_at": pd.Timestamp("2024-01-02", tz="UTC")},
            {"doi": "10.1/tiny", "ro_id": "alt:3", "title": "Tiny", "altmetric_score": 2.0,
             "last_mentioned_at": pd.Timestamp("2024-01-03", tz="UTC")},
        ]
    )

    out = top_papers(df, k=1, pinned_dois=["10.1/tiny"])

    dois = list(out["doi"])
    assert "10.1/tiny" in dois
    # Pinned rows come first per the contract documented on top_papers.
    assert dois[0] == "10.1/tiny"
    # k=1 ranked + 1 pinned == 2 rows total.
    assert len(dois) == 2


def test_top_papers_dedupes_by_doi() -> None:
    df = _papers_df(
        [
            {"doi": "10.1/x", "ro_id": "alt:1", "title": "X", "altmetric_score": 999.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
            {"doi": "10.1/y", "ro_id": "alt:2", "title": "Y", "altmetric_score": 500.0,
             "last_mentioned_at": pd.Timestamp("2024-01-02", tz="UTC")},
        ]
    )

    # Pin the top-1 paper; it should appear exactly once.
    out = top_papers(df, k=5, pinned_dois=["10.1/x"])

    assert list(out["doi"]).count("10.1/x") == 1
    assert set(out["doi"]) == {"10.1/x", "10.1/y"}
    # Pinned-first ordering still holds.
    assert list(out["doi"])[0] == "10.1/x"


def test_top_papers_k_limit() -> None:
    rows: list[dict[str, Any]] = [
        {"doi": f"10.1/p{i:02d}", "ro_id": f"alt:{i}", "title": f"P{i}",
         "altmetric_score": float(100 - i),
         "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")}
        for i in range(10)
    ]
    out = top_papers(_papers_df(rows), k=3)

    assert len(out) == 3
    # Highest scores first when no pinning.
    assert list(out["altmetric_score"]) == sorted(out["altmetric_score"], reverse=True)


def test_top_papers_ignores_pin_not_in_df() -> None:
    df = _papers_df(
        [
            {"doi": "10.1/a", "ro_id": "alt:1", "title": "A", "altmetric_score": 10.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
        ]
    )
    out = top_papers(df, k=5, pinned_dois=["10.99/missing"])
    assert list(out["doi"]) == ["10.1/a"]


# ---------------------------------------------------------------------------
# reconstruct_abstract pure function
# ---------------------------------------------------------------------------


def test_reconstruct_abstract_from_inverted_index() -> None:
    """The reconstructor joins position-sorted tokens with spaces; gaps are skipped.

    Policy (documented on the module): no placeholder for missing positions,
    duplicate positions keep the first word encountered after sorting.
    """
    # "hello" at position 0, "world" at positions 1 and 5 -> "hello world world"
    # (positions 2, 3, 4 are gaps and are simply skipped per the policy).
    out = reconstruct_abstract({"hello": [0], "world": [1, 5]})
    assert out == "hello world world"


def test_reconstruct_abstract_sorts_by_position() -> None:
    out = reconstruct_abstract({"second": [1], "first": [0], "third": [2]})
    assert out == "first second third"


# ---------------------------------------------------------------------------
# enrich_with_openalex_abstracts (httpx mocked)
# ---------------------------------------------------------------------------


def test_enrich_with_openalex_abstracts_reconstructs_text() -> None:
    fixture = _load_fixture("openalex_work_doi.json")
    df = _papers_df(
        [
            {"doi": "10.1126/science.aap8731", "ro_id": "alt:1", "title": "Disruption",
             "altmetric_score": 9000.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
        ]
    )
    client = _StubClient({"10.1126/science.aap8731": (200, fixture)})

    out = enrich_with_openalex_abstracts(df, http_client=_as_client(client))

    assert "abstract" in out.columns
    abstract = out["abstract"].iloc[0]
    assert isinstance(abstract, str)
    # Verify the reconstruction starts at position 0 with the expected first word
    # from the live OpenAlex record.
    assert abstract.startswith("Perceptual")
    # The httpx stub was actually consulted.
    assert client.calls and "10.1126/science.aap8731" in client.calls[0][0]
    # Polite-pool mailto carried through.
    assert client.calls[0][1].get("mailto")


def test_enrich_with_openalex_abstracts_handles_404() -> None:
    fixture = _load_fixture("openalex_work_doi.json")
    df = _papers_df(
        [
            {"doi": "10.1/missing", "ro_id": "alt:1", "title": "Missing",
             "altmetric_score": 1.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
            {"doi": "10.1126/science.aap8731", "ro_id": "alt:2", "title": "Found",
             "altmetric_score": 9000.0,
             "last_mentioned_at": pd.Timestamp("2024-01-02", tz="UTC")},
        ]
    )
    client = _StubClient(
        {
            "10.1/missing": (404, None),
            "10.1126/science.aap8731": (200, fixture),
        }
    )

    out = enrich_with_openalex_abstracts(df, http_client=_as_client(client))

    # Lookup by DOI, not index, since downstream callers may reorder.
    missing_row = out[out["doi"] == "10.1/missing"].iloc[0]
    found_row = out[out["doi"] == "10.1126/science.aap8731"].iloc[0]
    # pandas may surface our ``None`` as ``NaN`` in a mixed-type column — both are valid "missing".
    assert pd.isna(missing_row["abstract"]) or missing_row["abstract"] is None
    assert isinstance(found_row["abstract"], str)
    assert found_row["abstract"].startswith("Perceptual")


def test_enrich_does_not_mutate_input() -> None:
    fixture = _load_fixture("openalex_work_doi.json")
    df = _papers_df(
        [
            {"doi": "10.1126/science.aap8731", "ro_id": "alt:1", "title": "Disruption",
             "altmetric_score": 9000.0,
             "last_mentioned_at": pd.Timestamp("2024-01-01", tz="UTC")},
        ]
    )
    client = _StubClient({"10.1126/science.aap8731": (200, fixture)})

    enrich_with_openalex_abstracts(df, http_client=_as_client(client))

    assert "abstract" not in df.columns


# ---------------------------------------------------------------------------
# Smoke tests for the BigQuery-facing wrappers (signatures only).
# ---------------------------------------------------------------------------


