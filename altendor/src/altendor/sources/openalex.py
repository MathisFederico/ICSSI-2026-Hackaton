"""OpenAlex source: DOI lists from keyword, topic, or concept searches.

Pulls from the public OpenAlex REST API (no auth) using the polite pool with
`mailto=` in both the User-Agent header and the query string for safety.

Two public entry points:

* :func:`search_dois` — paginated DOI harvest over ``/works`` with a default
  ``title_and_abstract.search:<query>`` filter, or a ``topics.id:<id>`` /
  ``concepts.id:<id>`` filter when the corresponding override is set.
* :func:`resolve_topic_by_name` — top-1 match against ``/topics?search=<name>``.

The client uses a synchronous :class:`httpx.Client`. ``search_dois`` accepts
an optional injected client to make the function testable without network.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

OPENALEX_BASE = "https://api.openalex.org"
POLITE_EMAIL = "mathis.federico@bycelium.com"
USER_AGENT = f"altendor (mailto:{POLITE_EMAIL})"
DEFAULT_QUERY = "science of science"
PER_PAGE = 100
DOI_URL_PREFIX = "https://doi.org/"


def _build_client() -> httpx.Client:
    """Return an :class:`httpx.Client` configured with the polite User-Agent."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
    )


def _get_with_retry(client: httpx.Client, url: str, params: dict[str, Any]) -> httpx.Response:
    """GET *url* with a single 2-second backoff retry on HTTP 429.

    Any non-429 error status raises immediately via ``raise_for_status``.
    """
    response = client.get(url, params=params)
    if response.status_code == 429:
        time.sleep(2.0)
        response = client.get(url, params=params)
    response.raise_for_status()
    return response


def _strip_doi(raw: str | None) -> str | None:
    """Strip the ``https://doi.org/`` prefix from a DOI URL.

    Returns ``None`` when *raw* is falsy or lacks the expected prefix shape.
    """
    if not raw:
        return None
    if raw.startswith(DOI_URL_PREFIX):
        return raw[len(DOI_URL_PREFIX):]
    # Tolerate bare DOIs that already lack the URL prefix.
    if raw.startswith("10."):
        return raw
    return None


def _filter_for(query: str, topic_id: str | None, concept_id: str | None) -> str:
    """Build the OpenAlex ``filter`` clause for the works endpoint.

    Precedence: ``topic_id`` > ``concept_id`` > keyword query.
    """
    if topic_id:
        return f"topics.id:{topic_id}"
    if concept_id:
        return f"concepts.id:{concept_id}"
    return f"title_and_abstract.search:{query}"


def search_dois(
    query: str = DEFAULT_QUERY,
    *,
    topic_id: str | None = None,
    concept_id: str | None = None,
    n: int = 200,
    client: httpx.Client | None = None,
) -> list[str]:
    """Return up to *n* DOIs (sans ``https://doi.org/`` prefix) matching the search.

    Args:
        query: Free-text query against ``title_and_abstract.search``. Ignored
            when *topic_id* or *concept_id* is set.
        topic_id: OpenAlex topic id (e.g. ``T12345``). If set, filters by
            ``topics.id``.
        concept_id: OpenAlex concept id (e.g. ``C12345``). If set, filters by
            ``concepts.id``. Ignored when *topic_id* is also set.
        n: Maximum number of DOIs to return. The function will paginate via
            cursor until it has *n* DOIs or OpenAlex runs out of results.
        client: Optional injected :class:`httpx.Client`. When ``None`` a
            fresh client is created and closed on exit.

    Returns:
        List of DOIs without the URL prefix, e.g. ``["10.1234/abc", ...]``.
        Results missing a DOI are silently skipped.
    """
    owns_client = client is None
    if client is None:
        client = _build_client()

    dois: list[str] = []
    cursor: str | None = "*"
    filter_clause = _filter_for(query, topic_id, concept_id)

    try:
        while cursor is not None and len(dois) < n:
            params: dict[str, Any] = {
                "filter": filter_clause,
                "per-page": min(PER_PAGE, n - len(dois)),
                "cursor": cursor,
                "mailto": POLITE_EMAIL,
            }
            response = _get_with_retry(client, f"{OPENALEX_BASE}/works", params)
            payload = response.json()

            for work in payload.get("results", []):
                doi = _strip_doi(work.get("doi"))
                if doi is None:
                    continue
                dois.append(doi)
                if len(dois) >= n:
                    break

            cursor = payload.get("meta", {}).get("next_cursor")
            if not payload.get("results"):
                # Empty page — nothing more to paginate.
                break
    finally:
        if owns_client:
            client.close()

    return dois


def resolve_topic_by_name(name: str, *, client: httpx.Client | None = None) -> dict | None:
    """Resolve a topic name to its top OpenAlex match.

    Args:
        name: Free-text topic name to search for.
        client: Optional injected :class:`httpx.Client`.

    Returns:
        ``{"id": "T...", "display_name": "...", "works_count": int}`` for the
        highest-ranked match, or ``None`` when the search returns no results.
    """
    owns_client = client is None
    if client is None:
        client = _build_client()

    try:
        params: dict[str, Any] = {
            "search": name,
            "per-page": 1,
            "mailto": POLITE_EMAIL,
        }
        response = _get_with_retry(client, f"{OPENALEX_BASE}/topics", params)
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            return None
        top = results[0]
        raw_id = top.get("id", "")
        # Topic ids come back as URLs like "https://openalex.org/T10192"; emit the bare id.
        short_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        return {
            "id": short_id,
            "display_name": top.get("display_name"),
            "works_count": int(top.get("works_count") or 0),
        }
    finally:
        if owns_client:
            client.close()
