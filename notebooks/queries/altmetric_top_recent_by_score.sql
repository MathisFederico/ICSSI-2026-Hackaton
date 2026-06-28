-- =============================================================================
-- Altmetric — Smoke Test (Top 5 by Attention Score)
-- =============================================================================
-- Purpose:
--   Smallest possible "hello world" query against the Altmetric research
--   outputs table. Just five rows, no joins, no date parsing, no nested
--   field access. Use this to confirm credentials, dataset access, and
--   dry-run sizing work before layering on more complexity.
--
-- Source tables:
--   `altmetric-endorsements.altmetric_on_gbq.research_outputs`
--
-- Output columns:
--   id              — Altmetric research output ID
--   altmetric_score — Altmetric Attention Score (FLOAT)
--   title           — Publication title
--
-- Output grain: one row per research output, sorted by altmetric_score DESC, LIMIT 5.
-- =============================================================================
SELECT
  ro.id,
  ro.altmetric_score,
  ro.title
FROM
  `altmetric-endorsements.altmetric_on_gbq.research_outputs` AS ro
WHERE
  ro.altmetric_score IS NOT NULL
ORDER BY
  ro.altmetric_score DESC
LIMIT 5
