---
name: claude-kit
description: Use the hackathon's claude_kit.py helpers (ClaudeClient, CostTracker, Conversation, Agent) when calling the Anthropic API. Triggers when the user writes Claude API code in this repo, asks about cost tracking, multi-turn chat, prompt caching, or the web-search/web-fetch agent.
---

# claude_kit.py

`icssi-2026-hackathon/anthropic/claude_kit.py` is a thin wrapper around the
official `anthropic` SDK. Prefer reusing these helpers over rolling new ones —
the tutorial notebooks and any take-home work expect them.

## Triggers

Use this skill when the user:
- Writes new code that calls the Anthropic API in this repo.
- Asks about token cost, prompt caching, the Batch API, or per-call spend.
- Builds a multi-turn chatbot or a research agent that uses web tools.
- Asks "how do I track how much I spent" or similar.

## The five things in claude_kit

| Symbol | What it does |
|--------|--------------|
| `PRICING`, `cost_of_usage(usage, model, used_batch_api)` | Per-million-token rates as of mid-2026; returns USD cost from a response `usage` object. Cache writes are 1.25×, cache reads 0.10× input rate. Batch API halves both rates. |
| `CostTracker` | Accumulates spend across calls; supports an optional `budget_usd` that warns once when crossed. Use `tracker.add_response(model, response, label=...)` to also bill server-side web searches ($10 / 1000). |
| `ClaudeClient` | One-line `kit.ask(prompt, system=..., model=..., label=...)` returns plain text; wraps an `anthropic.Anthropic()` and auto-tracks cost. |
| `Conversation` | Multi-turn history. `convo.send(user_text)` appends and returns the reply. Keeps full content blocks (not just text) so non-text blocks survive. |
| `Agent` | Research agent over Claude's server-side `web_search` / `web_fetch` tools. Loops up to `max_rounds`, resumes on `pause_turn`. Returns `{answer, rounds, stop_reason}`. |

## Models

Default is `claude-sonnet-4-6`. The pricing table also lists `claude-opus-4-8`,
`claude-opus-4-7`, and `claude-haiku-4-5`. Take-home notebooks default to Haiku
for cost.

When the user is iterating, suggest:
- **Haiku 4.5** for first drafts, bulk classification, or tight loops.
- **Sonnet 4.6** for the default "good enough" tier.
- **Opus 4.8** only when the task fails on Sonnet — it's 5× the input price and
  5× the output price of Sonnet.

## Non-obvious things to remember

- The web-tool IDs (`web_search_20260209`, `web_fetch_20260209`) in `Agent.__init__` are versioned and rotate periodically. If the SDK rejects them, check the docs link in the comment and update.
- `Conversation.messages` stores `response.content` (a list of blocks) for assistant turns, not a plain string. Don't `str(...)` it.
- `CostTracker.add` reads `usage.service_tier == "batch"` to detect batch responses. If you pass a `usage` object that doesn't have `service_tier`, set it explicitly or wrap the call.
- `ClaudeClient.ask` accepts either a string prompt **or** a full `messages` list — useful for prompt caching where you want a long cached system + a tiny user turn.
- Server-side web searches are billed separately from tokens. `add_response` handles this; `add(usage, ...)` alone does not.

## Working pattern

```python
from claude_kit import ClaudeClient, CostTracker

tracker = CostTracker(budget_usd=1.00)
kit = ClaudeClient(model="claude-haiku-4-5", tracker=tracker)
out = kit.ask("Classify this abstract into a field: ...", label="classify")
tracker.report()
```

For agents, instantiate once and reuse — each `Agent.run(task)` is a fresh
conversation but shares the underlying `ClaudeClient` and cost tracker.
