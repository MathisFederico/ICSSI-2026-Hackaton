"""Altmetric-on-GBQ source — pipeline stage S5.

This module wraps three things:

* the ``research_outputs`` lookup (DOIs -> Altmetric ``ro_id`` rows),
* the ``posts`` fan-out for a list of ``ro_id``s,
* a per-DOI OpenAlex abstract enrichment (the GBQ table has no abstract
  column, so we ride the public OpenAlex ``/works/doi:...`` endpoint).

It also owns a tiny pure-pandas top-k selector with "pinned DOI" support
used by downstream stages to choose which papers to dive into.

Abstract reconstruction policy
------------------------------
OpenAlex stores abstracts as an "inverted index": a dict mapping each token
to the list of positions it occupies. We rebuild the abstract by emitting
the tokens in position order, **joined with a single space** and **without
inserting placeholders for missing positions**. The inverted index is
position-dense in practice (it is built from the original tokenization), so
gaps are vanishingly rare; when they do occur we prefer a slightly tighter
string over polluting the text with ``?`` placeholders that downstream
NLP/Claude calls would have to learn to ignore.

If the same position appears under multiple words (extremely unusual), we
keep the first occurrence encountered after sorting. Output is always a
``str`` — never ``None`` — when ``abstract_inverted_index`` is present.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, cast

import httpx
import pandas as pd
from google.cloud import bigquery

from altendor.bigquery.queries import (
    posts_by_research_output_ids,
    research_outputs_by_dois,
)
from altendor.sources.openalex import OPENALEX_BASE, POLITE_EMAIL, _build_client

# Polite-pool throttle between per-DOI OpenAlex calls (~20 req/s, well under the 10 burst).
_OPENALEX_SLEEP_SEC = 0.05


def join_dois_to_attention(client: bigquery.Client, dois: list[str]) -> pd.DataFrame:
    """Look up Altmetric ``research_outputs`` rows for the given *dois*.

    Args:
        client: An initialised :class:`google.cloud.bigquery.Client`.
        dois: Non-empty list of bare DOIs (e.g. ``"10.1126/science.aap8731"``).

    Returns:
        DataFrame with columns ``doi``, ``ro_id``, ``title``, ``altmetric_score``,
        and ``last_mentioned_at`` (UTC ``pd.Timestamp``). One row per matched DOI;
        DOIs not present in the Altmetric corpus are simply absent.
    """
    sql, job_config = research_outputs_by_dois(dois)
    df = client.query(sql, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
    return df


def fetch_posts_for_papers(client: bigquery.Client, ro_ids: list[str]) -> pd.DataFrame:
    """Fan out from ``ro_id``s to the mentioning posts.

    Args:
        client: An initialised :class:`google.cloud.bigquery.Client`.
        ro_ids: Non-empty list of Altmetric ``research_outputs.id`` strings.

    Returns:
        DataFrame with columns ``post_id``, ``ro_id``, ``type``, ``subtype``,
        ``date``, ``url``, ``title``, ``attention_source``, ``retweet``.
        Rows are exploded so a single post mentioning multiple research
        outputs appears once per (post, ro_id) pair.
    """
    sql, job_config = posts_by_research_output_ids(ro_ids)
    df = client.query(sql, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
    return df


def top_papers(
    df: pd.DataFrame,
    k: int = 10,
    pinned_dois: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pick the top-k papers by ``altmetric_score`` with optional pinned DOIs.

    Selection rules:

    * Rows with ``altmetric_score == 0`` are excluded from the ranked top-k
      candidate pool (a zero-score paper has no Altmetric attention, so it
      cannot be a meaningful top-pick).
    * DOIs in *pinned_dois* are always emitted — even if their score is 0 or
      they would otherwise fall outside the top-k.
    * Pinned DOIs that are not present in *df* are silently ignored (we never
      fabricate rows).
    * The output is deduplicated by ``doi`` and ordered with pinned rows
      first (in the order they appear in *pinned_dois*), then the ranked
      top-k sorted by descending ``altmetric_score``.

    Args:
        df: Output of :func:`join_dois_to_attention` (must contain ``doi``
            and ``altmetric_score`` columns).
        k: Number of ranked rows to keep after dropping zero-score ones.
        pinned_dois: Iterable of DOIs to always include.

    Returns:
        A new DataFrame with the same columns as *df*. The index is reset.
    """
    pinned_list: list[str] = list(pinned_dois or [])
    pinned_set = set(pinned_list)

    # Pinned rows first, ordered to match the caller's pinned_dois argument.
    if pinned_list:
        pinned_rows_unordered = df[df["doi"].isin(pinned_set)]
        # Preserve caller-supplied ordering of pinned DOIs (drop pins that don't appear in df).
        order_index = {doi: i for i, doi in enumerate(pinned_list)}
        pinned_rows = pinned_rows_unordered.assign(
            _pin_order=pinned_rows_unordered["doi"].map(order_index)
        ).sort_values("_pin_order").drop(columns="_pin_order")
    else:
        pinned_rows = df.iloc[0:0]

    # Ranked candidates: drop pinned (already accounted for) and zero-score rows.
    ranked_pool = df[~df["doi"].isin(pinned_set) & (df["altmetric_score"] > 0)]
    ranked = ranked_pool.sort_values("altmetric_score", ascending=False).head(k)

    combined = pd.concat([pinned_rows, ranked], ignore_index=True)
    # Dedupe by DOI; ``keep="first"`` keeps the pinned row when a pin also ranked.
    combined = combined.drop_duplicates(subset="doi", keep="first").reset_index(drop=True)
    return combined


def reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Rebuild an abstract from an OpenAlex inverted index.

    Position gaps are not filled; duplicate positions keep the first word
    encountered (after sorting by position). See the module docstring for
    the rationale.

    Args:
        inverted_index: ``{word: [pos1, pos2, ...]}`` as returned by
            OpenAlex's ``abstract_inverted_index`` field.

    Returns:
        The reconstructed abstract joined by single spaces.
    """
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda p: p[0])

    seen_positions: set[int] = set()
    words: list[str] = []
    for pos, word in pairs:
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        words.append(word)
    return " ".join(words)


def _fetch_openalex_work(doi: str, client: httpx.Client) -> dict[str, Any] | None:
    """Fetch a single OpenAlex work record by DOI.

    Returns ``None`` on 404 or network/HTTP errors. The caller is responsible
    for any throttling between calls.
    """
    url = f"{OPENALEX_BASE}/works/doi:{doi}"
    try:
        response = client.get(url, params={"mailto": POLITE_EMAIL})
    except httpx.HTTPError:
        return None
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None


def enrich_with_openalex_abstracts(
    df_papers: pd.DataFrame,
    *,
    http_client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Add an ``abstract`` column to *df_papers* by querying OpenAlex per DOI.

    Failures (404, network errors, missing ``abstract_inverted_index``) leave
    the ``abstract`` cell as ``None`` for that row. Other rows are still
    processed.

    Args:
        df_papers: DataFrame with at least a ``doi`` column.
        http_client: Optional injected :class:`httpx.Client`. When ``None``
            a fresh polite-pool client is created and closed on exit.

    Returns:
        A copy of *df_papers* with an additional ``abstract: str | None``
        column. The input DataFrame is not mutated.
    """
    owns_client = http_client is None
    if http_client is None:
        http_client = _build_client()

    abstracts: list[str | None] = []
    try:
        for i, doi in enumerate(df_papers["doi"].tolist()):
            if i > 0:
                time.sleep(_OPENALEX_SLEEP_SEC)
            if not doi:
                abstracts.append(None)
                continue
            work = _fetch_openalex_work(doi, http_client)
            if work is None:
                abstracts.append(None)
                continue
            inverted = work.get("abstract_inverted_index")
            if not inverted or not isinstance(inverted, dict):
                abstracts.append(None)
                continue
            abstracts.append(reconstruct_abstract(inverted))
    finally:
        if owns_client:
            http_client.close()

    out = df_papers.copy()
    out["abstract"] = abstracts
    return out
