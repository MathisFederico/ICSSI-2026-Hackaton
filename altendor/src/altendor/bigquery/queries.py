"""BigQuery query builders for the DOIs -> research_outputs -> posts workflow.

Each builder returns a ``(sql, job_config)`` tuple. The SQL contains only
named placeholders (``@dois`` / ``@ro_ids``); the values are bound via
:class:`google.cloud.bigquery.ArrayQueryParameter` on the returned
:class:`google.cloud.bigquery.QueryJobConfig`. Callers either pass the
config straight to :meth:`bigquery.Client.query` or feed the SQL through
:func:`altendor.bigquery.preflight.preflight` first.

Never inline user input into the SQL string. All filtering goes through
``UNNEST(@param)``.

Empty-input policy
------------------
Passing ``[]`` raises :class:`ValueError`. We chose this over a
trivially-empty SQL because (a) it surfaces caller bugs eagerly instead of
silently returning zero rows and (b) it avoids billing BigQuery for a
no-op dry-run round-trip.
"""

from __future__ import annotations

from google.cloud.bigquery import ArrayQueryParameter, QueryJobConfig

_PROJECT = "altmetric-endorsements"
_DATASET = "altmetric_on_gbq"


_RESEARCH_OUTPUTS_BY_DOIS_SQL = f"""
SELECT
  id AS ro_id,
  identifiers.doi AS doi,
  title,
  altmetric_score,
  last_mentioned_at
FROM `{_PROJECT}.{_DATASET}.research_outputs`
WHERE identifiers.doi IN UNNEST(@dois)
""".strip()


_POSTS_BY_RESEARCH_OUTPUT_IDS_SQL = f"""
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
FROM `{_PROJECT}.{_DATASET}.posts` AS p,
  UNNEST(p.research_outputs_ids) AS ro_id
WHERE ro_id IN UNNEST(@ro_ids)
""".strip()


def research_outputs_by_dois(dois: list[str]) -> tuple[str, QueryJobConfig]:
    """Return ``(sql, job_config)`` selecting research_outputs rows for *dois*.

    Projects: ``ro_id``, ``doi``, ``title``, ``altmetric_score``,
    ``last_mentioned_at``. Filters via ``identifiers.doi IN UNNEST(@dois)``.
    Note: ``research_outputs`` has no ``abstract`` column — fetch abstracts
    from OpenAlex (see :mod:`altendor.sources.openalex`).

    Args:
        dois: Non-empty list of DOI strings. Bound as an array parameter,
            never inlined into the SQL.

    Returns:
        A ``(sql, job_config)`` pair. ``job_config.query_parameters`` will
        contain a single ``ArrayQueryParameter("dois", "STRING", dois)``.

    Raises:
        ValueError: If *dois* is empty. See module docstring for rationale.
    """
    if not dois:
        raise ValueError("research_outputs_by_dois: dois must be a non-empty list")
    job_config = QueryJobConfig(
        query_parameters=[ArrayQueryParameter("dois", "STRING", list(dois))],
    )
    return _RESEARCH_OUTPUTS_BY_DOIS_SQL, job_config


def posts_by_research_output_ids(ro_ids: list[str]) -> tuple[str, QueryJobConfig]:
    """Return ``(sql, job_config)`` for posts mentioning any of *ro_ids*.

    Explodes ``research_outputs_ids`` server-side via ``UNNEST`` and filters
    in the ``WHERE`` clause so BigQuery can prune early. Projects:
    ``post_id``, ``ro_id`` (exploded), ``type``, ``subtype``, ``date``,
    ``url``, ``title``, ``attention_source``, ``retweet``.

    Args:
        ro_ids: Non-empty list of Altmetric research output ids (e.g.
            ``"alt:123"``). Bound as an array parameter.

    Returns:
        A ``(sql, job_config)`` pair. ``job_config.query_parameters`` will
        contain a single ``ArrayQueryParameter("ro_ids", "STRING", ro_ids)``.

    Raises:
        ValueError: If *ro_ids* is empty. See module docstring for rationale.
    """
    if not ro_ids:
        raise ValueError("posts_by_research_output_ids: ro_ids must be a non-empty list")
    job_config = QueryJobConfig(
        query_parameters=[ArrayQueryParameter("ro_ids", "STRING", list(ro_ids))],
    )
    return _POSTS_BY_RESEARCH_OUTPUT_IDS_SQL, job_config
