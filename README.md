# ICSSI 2026 Open Data Hackathon

Workspace for the [ICSSI 2026 Open Data Hackathon](https://www.icssi.org/), held Sunday 28 June 2026 at the Caruthers Biotechnology Building, University of Colorado Boulder, as the kickoff of the International Conference on the Science of Science & Innovation (29 June – 1 July 2026).

The hackathon focuses on the **science of science**: papers, citations, careers, grants, and collaboration networks, analysed with open datasets and LLMs.

## Layout

- Repo root — your own analysis code, notebooks, and modules.
- `icssi-2026-hackathon/` — read-only submodule from
  [`LarremoreLab/icssi-2026-hackathon`](https://github.com/LarremoreLab/icssi-2026-hackathon)
  with the tutorial notebooks and reference datasets.

## Quick start

```bash
# clone with the tutorial submodule
git clone --recurse-submodules <this-repo>
cd ICSSI-2026-Hackaton

# secrets
cp .env.example .env
# then edit .env and fill in ANTHROPIC_API_KEY and DIMENSIONS_API_KEY

# python env (uv manages the venv)
uv sync
uv run python -c "import pandas; print(pandas.__version__)"
```

If you use [Nix](https://nixos.org/) with `direnv`, the devshell in `flake.nix`
provisions `uv`, Python 3.12, Node, `pnpm`, `make`, and `graphite-cli`, and runs
`uv sync` + `pnpm install` on entry. Just `cd` into the directory and accept
the `.envrc`.

## API keys

- **Anthropic** — get a key at <https://console.anthropic.com/> under
  *Settings → API keys*. Used by the tutorial's `ClaudeClient` and any of your
  own Claude calls.
- **Dimensions** — request access at
  <https://www.dimensions.ai/dimensions-api/>. Used for live publication,
  grant, patent, and clinical-trial queries.

`.env` is gitignored; never commit it or paste it into screenshots/chat.

## Tooling

| Tool     | Use                                                |
| -------- | -------------------------------------------------- |
| `uv`     | Python 3.12 venv and dependency management         |
| `ruff`   | Lint and import sorting (line length 120, E/F/I/ANN) |
| `ty`     | Type checking                                      |
| `pytest` | Tests (`uv run pytest`)                            |
| `pnpm`   | JS tooling (kept minimal)                          |

Common commands:

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # types
uv run jupyter lab     # notebooks
```

## Working with Claude

The tutorial submodule ships a small `claude_kit.py` with `ClaudeClient`,
`CostTracker`, `Conversation`, and `Agent` helpers. Two rules of thumb on a
personal API key:

- Wrap every Claude call in `CostTracker` so spend stays visible.
- Default to `claude-sonnet-4-6`; switch to `claude-haiku-4-5` for bulk
  classification, and use prompt caching + the Batch API for large runs.

Pricing is in the `PRICING` dict of `claude_kit.py` — double-check it against
<https://www.anthropic.com/pricing> before any large job.

## License

[MIT](LICENSE).
