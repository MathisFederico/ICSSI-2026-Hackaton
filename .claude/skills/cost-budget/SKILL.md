---
name: cost-budget
description: Keep Anthropic API spend predictable on a personal key. Use whenever the user is about to run a large Claude call, batch-classify thousands of records, or asks "how much will this cost" / "am I spending too much" / "how do I budget this".
---

# Cost budgeting for the hackathon

The hackathon runs on the participant's own Anthropic key. Before any
multi-hundred-call run, do the math and put a budget in.

## Triggers

Use this skill when the user:
- Plans a large or open-ended Claude API run.
- Asks about pricing, token cost, caching, or batch processing.
- Mentions `CostTracker`, `budget_usd`, or "how do I know what I'm spending".
- Writes a loop that calls `client.messages.create` more than ~50 times.

## Pricing (mid-2026 — verify before large runs)

| Model | Input $/MTok | Output $/MTok |
|-------|-------------:|--------------:|
| claude-opus-4-8 | 5.00 | 25.00 |
| claude-opus-4-7 | 5.00 | 25.00 |
| claude-sonnet-4-6 | 3.00 | 15.00 |
| claude-haiku-4-5 | 1.00 | 5.00 |

Modifiers:
- **Cache write**: 1.25× input rate.
- **Cache read**: 0.10× input rate.
- **Batch API**: 0.5× both input and output rates.
- **Server-side web search**: $10 per 1000 searches, billed on top of tokens.

These rates ship in `icssi-2026-hackathon/anthropic/claude_kit.py` (`PRICING`)
and should be re-checked against <https://www.anthropic.com/pricing> before any
large run.

## Pre-flight estimate

Before a big loop, do this back-of-envelope:

```
n_calls × (avg_input_tokens × input_rate + avg_output_tokens × output_rate) / 1e6
```

Then add server-search cost if the agent uses `web_search`. Compare against
the user's intended budget. If it's >25%, ask before kicking off.

## Levers to cut cost

1. **Drop a tier**: Haiku is 1/3 of Sonnet, 1/5 of Opus on input. Try Haiku first; only escalate after a measurable quality miss.
2. **Prompt caching**: if a fixed system prompt or context is reused >2 times within 5 minutes, cache it. Cache reads cost 10% of input.
3. **Batch API**: for non-interactive runs (overnight, "give me an answer in <24h"), the Batch API is half price. `usage.service_tier == "batch"` reports it.
4. **Shorter outputs**: `max_tokens` defaults to 1024 in `ClaudeClient`. For classification or extraction, set it to the smallest viable number — output tokens are 5× input on Sonnet/Opus and 5× on Haiku.
5. **Filter before LLM**: don't ask Claude to classify 10k abstracts if a regex/keyword pre-filter cuts the set to 500.

## Hard rules for this repo

- **Every Claude call must go through `CostTracker`** (via `kit.ask(...)` or `tracker.add_response(...)`). No bare `client.messages.create` in committed code unless cost is explicitly out of scope.
- **Set `budget_usd` on the tracker** for any script that runs >10 minutes unattended. The tracker warns once when crossed; combine with a kill-switch if needed.
- **Print `tracker.report()` at the end of every notebook** so spend is visible in the saved output.
- **Don't put API keys in cells** — read from `os.environ`.

## See also

- [[claude-kit]] — the helpers that implement these patterns.
- `icssi-2026-hackathon/anthropic/01_cost.ipynb` — the live demo notebook for pricing, caching, and Batch API.
