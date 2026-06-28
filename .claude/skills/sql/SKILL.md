---
name: sql
description: Conventions for writing SQL queries for Google BigQuery (GBQ). Use when writing, reviewing, or debugging BigQuery SQL queries.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - mcp__bigquery__execute_sql_readonly
  - mcp__bigquery__get_table_info
  - mcp__bigquery__list_table_ids
---

# Google BigQuery SQL Skill

- Schema YAML files for `dimensions-ai.data_analytics` tables live in gbq_schema/ relative to this skill [./gbq-schema].

## General Conventions

- Write all SQL queries in dedicated `.sql` files, not inline as Python strings
- Use descriptive filenames that reflect the query's purpose (e.g., `pub_counts_by_year.sql`, `grant_funding_by_org.sql`)
- Include a comment block at the top of each `.sql` file describing what the query does and any expected parameters
- Parameterize queries where possible to support reuse across projects
- always check schema memory files before writing queries against `dimensions-ai.data_analytics`
- For Dimensions data, specifically with regards to the `dimensions-ai.data_analytics.publications` the `journal` field is deprecated. Use `source` instead

## Style

- Use uppercase for SQL keywords (`SELECT`, `FROM`, `WHERE`, `JOIN`, etc.)
- Use snake_case for column aliases and CTEs
- Indent subqueries and CTEs for readability
- Prefer CTEs (`WITH` clauses) over deeply nested subqueries
- Alias all tables and use aliases consistently in column references
- Always qualify column names with table aliases to avoid ambiguity

## Joins

- **Avoid CROSS JOINs wherever possible.** They produce cartesian products and can generate massive result sets that blow up costs and processing time. If a CROSS JOIN seems necessary, confirm the intent and add a comment explaining why.
- Prefer explicit JOIN syntax (`INNER JOIN`, `LEFT JOIN`) over implicit comma-separated joins in the `FROM` clause
- Prefer explicit JOIN syntax in general, e.g. `INNER JOIN` is better than `JOIN`
- Always include the join condition in the `ON` clause, not in `WHERE`
- When joining large tables, filter each side as early as possible using subqueries or CTEs to reduce the data before the join

## Cost and Performance

- Use `LIMIT` during development and debugging to avoid scanning full tables
- Prefer `SELECT` with explicit column names over `SELECT *`
- Use partitioned and clustered columns in `WHERE` clauses when available to reduce bytes scanned
- Check estimated bytes scanned before running expensive queries on large tables
- Use `IF EXISTS` / `CREATE OR REPLACE` patterns to avoid errors on repeated runs

## Data Types

- Cast dates and timestamps explicitly rather than relying on implicit conversion
- Use `DATE` over `TIMESTAMP` when time-of-day precision is not needed
- Be explicit about `STRING` vs numeric types in comparisons and aggregations
- Convert dates to `ISO 8601 format`

## Notation and Comments
  
  - Write a 1 line (<= 100 word purpose statement for CTEs) as a comment
  - Write a comment block at the top of each sql file that labels the query, gives a `Purpose`, `Input Parameters` (if they exist), `Source Tables`, `Output columns`, & `Output Grain` (what each column "means")
    - An example is provided below
    - If reviewing an exisiting sql file that does not have this block, offer to write one


### Example Notation Block
```
-- =============================================================================
-- Portfolio Citation Impact — Resulting Publications with Citation Metrics
-- =============================================================================
-- Purpose:
--   Retrieves all publications resulting from the selected grants and
--   summarises their downstream citation reach: citing publications,
--   citing patents, citing clinical trials, and citing policy documents.
--
-- Parameters:
--   {selected_grant_ids} — comma-separated list of quoted Dimensions grant IDs
--                          e.g. 'grant.1234', 'grant.5678', 'grant.9012'
--                          Injected into the IN (...) clause at query time.
--
-- Source tables:
--   dimensions-onr.support_tables.onr_grants
--   dimensions-ai.data_analytics.publications
--   dimensions-ai.data_analytics.policy_documents
--
-- Output columns:
--   onr_contract_number — ONR contract number
--   grant_id            — Dimensions grant ID
--   resulting_pub_id    — Dimensions publication ID
--   year                — Publication year
--   n_citations         — Count of publications citing this output
--   n_patents           — Count of patents citing this output
--   n_clinical_trials   — Count of clinical trials citing this output
--   n_policy_docs       — Count of policy documents citing this output
--
-- Output grain: one row per resulting publication × grant_id
-- =============================================================================
```

## Schema and Schema Review

Examine all schema's before finalizing a query

### Dimensions Base Data
Schema's for the main dimensions tables (`dimensions-ai.data_analytics`) can be located in ~/.claude/skills/sql/gbq_schema. Each file is a YAML describing the SCHEMA of each table. 
  - Review the SCHEMA for each table before finalizing a sql query
    - Pay special attention to REPEATED fields, which require an UNNEST() to access the data
    - Pay special attention to records, which can be accessed with a `.` (e.g. p.metrics.field_citation_ratio)
    - Also, pay special attention for fields that have multipled REPEATED sub-fields or RECORDS contained with in REPEATED fields
      - These may require special combinations of UNNEST() and `.` accessor methods

If a schema needs to be found or called forward from another project or dataset (e.g. `ds-esd-shared.metrics`), this query will provide a template for how to query the schema:
```sql
SELECT
  t1.column_name AS field,
  t1.data_type AS type,
  -- t1.is_nullable AS mode,
  t2.description
FROM
  `dimensions-ai.data_analytics.INFORMATION_SCHEMA.COLUMNS` AS t1
INNER JOIN
  `dimensions-ai.data_analytics.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS` AS t2
  ON t1.table_name = t2.table_name 
  AND t1.column_name = t2.column_name
WHERE
  t1.table_name = 'publications'
ORDER BY
  t1.ordinal_position;
```

Other table names include `patents`, `source_titles`, `grants`, `datasets`, `organizations`, `org_groups`, `researchers`, `reports`, `policy_documents`, `clinical_trials`