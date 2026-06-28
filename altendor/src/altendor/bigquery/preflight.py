"""Pre-flight BigQuery queries: estimate scan size and enforce a configurable cap.

Handles two cases:

1. **Normal datasets** — the dry run returns ``totalBytesProcessed`` directly.
2. **RLS-protected datasets** (e.g. Analytics Hub subscriptions like the
   Altmetric "Altmetrics on BigQuery" listing) — the dry-run response masks
   bytes for the consumer. We fall back to summing the metadata size of every
   referenced table as an upper bound (metadata is not RLS-masked).

In both cases the function raises :class:`QuerySizeExceeded` if the estimate
exceeds *max_bytes*, so callers can use it as a hard guard before paying for
an expensive scan.
"""

from __future__ import annotations

from google.cloud import bigquery

DEFAULT_MAX_BYTES = 50 * (1024**3)  # 50 GiB; on-demand pricing is ~$0.31 at $6.25/TB


class QuerySizeExceeded(RuntimeError):
    """The pre-flight estimate is larger than the configured cap."""


def preflight(
    client: bigquery.Client,
    sql: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    verbose: bool = True,
) -> int:
    """Validate *sql* via a dry run and return an upper-bound bytes-scanned estimate.

    Args:
        client: An initialised :class:`google.cloud.bigquery.Client`.
        sql: The query string.
        max_bytes: Cap above which :class:`QuerySizeExceeded` is raised.
            Defaults to 50 GiB.
        verbose: When True (default), print a one-line summary or a
            per-referenced-table breakdown.

    Returns:
        The estimate in bytes.

    Raises:
        ValueError: If the dry run reports query-level errors (invalid SQL,
            missing columns, missing permissions).
        QuerySizeExceeded: If the estimate exceeds *max_bytes*.
    """
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    if dry.errors:
        raise ValueError(f"Query failed validation: {dry.errors}")

    if dry.total_bytes_processed is not None:
        estimate = int(dry.total_bytes_processed)
        if verbose:
            print(f"Preflight: would scan {estimate / 1e6:.2f} MB (BigQuery dry-run estimate)")
    else:
        refs = dry._properties.get("statistics", {}).get("query", {}).get("referencedTables", [])
        estimate = 0
        if verbose:
            print("Preflight: dry-run bytes masked by row-level security; upper bound from referenced tables:")
        for r in refs:
            table = client.get_table(f"{r['projectId']}.{r['datasetId']}.{r['tableId']}")
            size = table.num_bytes or 0
            estimate += size
            if verbose:
                print(f"  - {r['datasetId']}.{r['tableId']}: {size / 1e6:.2f} MB  ({table.num_rows:,} rows)")
        if verbose:
            print(f"Upper-bound scan: {estimate / 1e6:.2f} MB")

    if estimate > max_bytes:
        raise QuerySizeExceeded(
            f"Pre-flight estimate {estimate / 1e9:.2f} GB exceeds the "
            f"{max_bytes / 1e9:.2f} GB limit. Raise `max_bytes` if you have "
            f"reviewed the query and want to proceed."
        )
    return estimate
