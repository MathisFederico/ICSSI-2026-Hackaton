# ICSSI 2026 Open Data Hackathon

This repo is the user's workspace for the **ICSSI 2026 Open Data Hackathon**, held
Sunday 28 June 2026 at the Caruthers Biotechnology Building, University of Colorado
Boulder, as the kickoff event of the International Conference on the Science of
Science & Innovation (29 June – 1 July 2026). The hackathon focuses on the
**science of science**: analysis of scholarly artifacts (papers, citations,
careers, grants, collaboration networks) using open datasets and LLMs.

The upstream submodule `icssi-2026-hackathon/` (from
`LarremoreLab/icssi-2026-hackathon`) ships tutorial notebooks for the Claude
API and reference datasets. Treat it as **read-only** — the user's own
analysis code lives at the repo root, not inside the submodule.

## Tooling & environment

- **Python**: 3.12, managed by `uv`. Run things with `uv run <cmd>` to pick up
  the venv without manual activation.
- **Lint/types**: `ruff` (line-length 120, rules E/F/I/ANN) and `ty` for type
  checking.
- **Tests**: `pytest` with `--durations=10`.
- **Nix**: `flake.nix` provides a devshell (uv, python, node, pnpm, gnumake,
  graphite-cli); `direnv` auto-loads it via `.envrc`.
- **Secrets**: `.env` is gitignored. Copy `.env.example` and fill in
  `ANTHROPIC_API_KEY` (Anthropic console) and `DIMENSIONS_API_KEY` (Dimensions
  for science-of-science queries).

## Conventions

- Keep notebooks linear (top-to-bottom executable). Move reusable logic into
  `.py` modules and import.
- Track spend with `CostTracker` on every Claude call. The hackathon runs on
  personal API keys, so cost visibility is non-negotiable.
- For Claude API code, prefer prompt caching and the Batch API on bulk runs.
  Cache writes cost ~1.25× input; cache reads ~0.10×; Batch is 0.5× both.
- The default Claude model is `claude-sonnet-4-6`; prefer `claude-haiku-4-5`
  for bulk classification. Re-check `PRICING` in `claude_kit.py` against
  <https://www.anthropic.com/pricing> before any large run.
- Don't commit `.env` or any third-party content returned from API calls.

## Skills

Use the `.claude/skills/` skills when their triggers match:

- `claude-kit` — using the tutorial's `ClaudeClient`/`CostTracker`/`Agent`/`Conversation`.
- `science-of-science` — analysis patterns over scholarly data (authorship, citations, careers, collaboration).
- `dimensions-api` — querying Dimensions for publications/grants/patents/clinical trials.
- `cost-budget` — keeping Claude API spend predictable on a personal key.
