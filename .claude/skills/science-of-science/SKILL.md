---
name: science-of-science
description: Analysis patterns for science-of-science work — scholarly publications, citations, careers, collaboration networks, grants, fields/topics. Use when the user is doing exploratory analysis over papers, authors, institutions, OpenAlex/Crossref data, or wants to design a hackathon-scale science-of-science study.
---

# Science of Science analysis

The ICSSI community studies how science itself works: who collaborates with
whom, how ideas spread, how careers unfold, how funding shapes output. This
skill collects analysis patterns common in this space and the gotchas that bite
hackathon-scale studies.

## Triggers

Use this skill when the user is:
- Designing an analysis over scholarly publications, citations, careers, or grants.
- Joining datasets across OpenAlex / Crossref / OpenLibrary / HathiTrust / Dimensions.
- Building topic taxonomies, collaboration networks, or career trajectories.
- Looking at field/discipline distributions, gender gaps, or institutional effects.

## Common dataset entry points

- **Tutorial abstracts** — `icssi-2026-hackathon/anthropic/tutorial_data/abstracts.json` — small set of abstracts for prompt experimentation.
- **Field taxonomy** — `icssi-2026-hackathon/anthropic/tutorial_data/field_taxonomy.md` — hierarchical field labels useful as an LLM-classification target.
- **Scientist career sample** — `icssi-2026-hackathon/anthropic/tutorial_data/scientist_career.json`.
- **Dimensions API** — for live publication/grant/patent/clinical-trial queries. See [[dimensions-api]].
- **OpenAlex / Crossref** — free, no key required, broad coverage. Good first stop before reaching for Dimensions.

## Gotchas that bite hackathon-scale work

- **Citation counts diverge across sources**. OpenAlex aggregates more broadly than Crossref; both undercount preprints. Pick one source per analysis and report it. Counts also change daily — snapshot the query date.
- **Field-Weighted Citation Impact (`fwci`)** is field-and-year normalized; comparing FWCI across fields is fine, comparing raw cited_by_count is not.
- **Author disambiguation is hard**. OpenAlex's author IDs are imperfect; the same person can have multiple IDs and the same ID can collapse multiple people. ORCID is better when present, missing for many.
- **Self-citations inflate impact**. If the user is comparing scientists or institutions, ask whether to strip self-citations.
- **Survivor bias in career data**. Parsed CVs come from people whose CVs are publicly findable — typically tenured-track, employed, English-language. Don't generalize without acknowledging it.
- **Gender inferences are noisy** — any inferred (vs self-reported) gender field needs an explicit "inferred" caveat; never publish without it.
- **OpenAlex topics are hierarchical** (4 domains → ~25 fields → ~250 subfields → ~4,500 topics). Pick the right level — topic-level is noisy at low counts.
- **Open-access status (`oa_status`)** has seven values including `bronze` (free-to-read but no license) and `diamond` (gold with no APC). Don't collapse to a binary unless the user asks.
- **Collaboration networks are temporal**. A static co-authorship graph hides that early-career and late-career collaboration patterns differ. Slice by year or career stage when possible.

## When LLMs help

LLMs (via [[claude-kit]]) are good at:
- Field/topic classification from abstracts and titles.
- Structured extraction from messy CV/grant text.
- Comparing pairs of abstracts for novelty/similarity.
- Summarizing a researcher's trajectory across decades of publications.

LLMs are *bad* at:
- Counting things accurately — use code, not the model.
- Reliable citation lookups — they hallucinate DOIs. Use Crossref/OpenAlex.
- Replicating exact analyses — temperature and model version drift matter.

## Working pattern

1. Pin the data sources and snapshot dates up front; record them in a notebook header.
2. Start with the smallest viable sample (1 CV, 10 abstracts) before scaling.
3. When LLM-classifying at scale, use Haiku + Batch API + prompt caching — see [[cost-budget]].
4. Always report sample size, source provenance, and known biases alongside any
   headline number the user wants to put on a slide.
