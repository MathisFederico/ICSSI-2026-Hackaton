---
name: dimensions-api
description: Query the Dimensions API for publications, grants, patents, clinical trials, and policy documents. Use when the user mentions Dimensions, DIMENSIONS_API_KEY, DSL queries, or wants live scholarly data beyond what OpenAlex/Crossref provide.
---

# Dimensions API

Dimensions (Digital Science) indexes publications, grants, patents, clinical
trials, datasets, and policy documents in a single linked graph. The hackathon
provides `DIMENSIONS_API_KEY` in `.env.example`, so the organisers expect
participants to use it.

## Triggers

Use this skill when the user:
- Mentions Dimensions, `DIMENSIONS_API_KEY`, the Dimensions DSL, or `dimcli`.
- Wants live grant data alongside publications (Dimensions links them).
- Needs patent or clinical-trial data alongside scholarly outputs.
- Hits OpenAlex/Crossref coverage gaps and the user has a Dimensions key.

## How to use it from Python

The official Python client is `dimcli` (`pip install dimcli`). It exposes the
Dimensions DSL — a SQL-like query language — through `dsl.query(...)` or
`dsl.query_iterative(...)` for pagination.

```python
import dimcli
import os

dimcli.login(key=os.environ["DIMENSIONS_API_KEY"], endpoint="https://app.dimensions.ai/api/dsl")
dsl = dimcli.Dsl()

result = dsl.query("""
    search publications
        where research_orgs.id = "grid.266190.a"           # CU Boulder
        and year in [2020:2025]
    return publications[basics + altmetric + concepts]
        limit 100
""")
df = result.as_dataframe()
```

Key entity types: `publications`, `grants`, `patents`, `clinical_trials`,
`policy_documents`, `datasets`, `researchers`, `organizations`. Each has its
own field set — check the [DSL docs](https://docs.dimensions.ai/dsl/) for
exact field names.

## Non-obvious things to remember

- **Rate limits**: free academic keys cap at ~30 queries / 60 sec and 1000 / hour. Use `query_iterative` and a small sleep on large extracts.
- **`search ... return ...` is mandatory** — DSL queries must specify both. `limit` is mandatory for paginated returns.
- **Field selection**: `[basics]`, `[basics + altmetric]`, `[basics + concepts]`, `[extras + funders]` etc. Asking for more fields slows the query — start narrow.
- **IDs are stable but not human-readable** — researchers (`ur.xxxx`), organizations (GRID IDs like `grid.266190.a`), publications (`pub.xxxxxxxxx`). Keep a small lookup table.
- **GRID is Dimensions' org identifier**; ROR is OpenAlex's. They overlap heavily but aren't 1:1 — join via name+country fallback when crosswalking.
- **`research_orgs`** vs **`researchers`** — `research_orgs` filters on author affiliation; `researchers` filters on whether a named person is an author.
- **Dimensions citation counts differ from OpenAlex/Crossref** (different indexed corpora). Don't mix in the same table without labeling.
- **Grant currencies are not normalized** — `funding_amount` is in the awarding agency's currency. Convert before aggregating across countries.

## Cost & key hygiene

- Read the key from `os.environ["DIMENSIONS_API_KEY"]`, never hardcode.
- Don't commit any Dimensions JSON dumps containing third-party content without checking the academic API ToS — most allow research caching but not redistribution.

## See also

- [[science-of-science]] — analysis patterns and gotchas.
