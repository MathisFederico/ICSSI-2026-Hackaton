"""Unit tests for ``altendor.bigquery.queries``.

These are pure offline tests — no BigQuery calls. They pin:

* the SQL string shape (whitespace-normalized golden snippets),
* the parameter binding (no inlining of user input),
* the empty-input contract (``ValueError``).
"""

from __future__ import annotations

import re

import pytest
from altendor.bigquery.queries import (
    posts_by_research_output_ids,
    research_outputs_by_dois,
)
from google.cloud.bigquery import ArrayQueryParameter, QueryJobConfig


def _normalize(sql: str) -> str:
    """Collapse all whitespace runs to single spaces, strip ends."""
    return re.sub(r"\s+", " ", sql).strip()


def test_research_outputs_by_dois_sql_stable() -> None:
    sql, job_config = research_outputs_by_dois(["10.1000/foo", "10.1000/bar"])

    expected = _normalize(
        """
        SELECT
          id AS ro_id,
          identifiers.doi AS doi,
          title,
          altmetric_score,
          last_mentioned_at
        FROM `altmetric-endorsements.altmetric_on_gbq.research_outputs`
        WHERE identifiers.doi IN UNNEST(@dois)
        """
    )
    assert _normalize(sql) == expected

    assert isinstance(job_config, QueryJobConfig)
    params = list(job_config.query_parameters)
    assert len(params) == 1
    p = params[0]
    assert isinstance(p, ArrayQueryParameter)
    assert p.name == "dois"
    assert p.array_type == "STRING"
    assert list(p.values) == ["10.1000/foo", "10.1000/bar"]


def test_posts_by_research_output_ids_sql_stable() -> None:
    sql, job_config = posts_by_research_output_ids(["alt:1", "alt:2"])

    normalized = _normalize(sql)
    expected = _normalize(
        """
        SELECT
          p.id AS post_id,
          ro_id,
          p.type AS type,
          p.subtype AS subtype,
          p.date AS date,
          p.url AS url,
          p.title AS title,
          p.attention_source AS attention_source,
          p.retweet AS retweet
        FROM `altmetric-endorsements.altmetric_on_gbq.posts` AS p,
          UNNEST(p.research_outputs_ids) AS ro_id
        WHERE ro_id IN UNNEST(@ro_ids)
        """
    )
    assert normalized == expected

    # Spot-check the load-bearing explode pattern explicitly.
    assert "UNNEST(p.research_outputs_ids) AS ro_id" in normalized

    assert isinstance(job_config, QueryJobConfig)
    params = list(job_config.query_parameters)
    assert len(params) == 1
    p = params[0]
    assert isinstance(p, ArrayQueryParameter)
    assert p.name == "ro_ids"
    assert p.array_type == "STRING"
    assert list(p.values) == ["alt:1", "alt:2"]


def test_no_string_inlining() -> None:
    """User input must never appear in the SQL — it goes through @params only."""
    hostile_doi = "10.injection/'; DROP TABLE research_outputs; --"
    sql_ro, cfg_ro = research_outputs_by_dois([hostile_doi])
    assert hostile_doi not in sql_ro
    assert "DROP TABLE" not in sql_ro
    assert "'" not in sql_ro  # no inlined string literals at all
    assert "@dois" in sql_ro
    assert list(cfg_ro.query_parameters[0].values) == [hostile_doi]

    hostile_ro_id = "alt:'; DROP TABLE posts; --"
    sql_posts, cfg_posts = posts_by_research_output_ids([hostile_ro_id])
    assert hostile_ro_id not in sql_posts
    assert "DROP TABLE" not in sql_posts
    assert "'" not in sql_posts
    assert "@ro_ids" in sql_posts
    assert list(cfg_posts.query_parameters[0].values) == [hostile_ro_id]


def test_empty_input_returns_empty_result_safely() -> None:
    """Contract: empty input raises ValueError rather than scanning the whole table.

    Rationale documented in ``altendor.bigquery.queries`` module docstring:
    raising surfaces caller bugs eagerly and avoids a no-op BigQuery round-trip.
    """
    with pytest.raises(ValueError, match="non-empty"):
        research_outputs_by_dois([])
    with pytest.raises(ValueError, match="non-empty"):
        posts_by_research_output_ids([])
