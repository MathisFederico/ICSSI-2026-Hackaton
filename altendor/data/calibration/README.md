# Classifier calibration set

`labeled.jsonl` holds the **rubric anchors** the classifier sees as few-shot
exemplars (see S9 in `/PIPELINE_PLAN.md`) and the gold labels the calibration
gate measures against (S12).

## Schema (one row = one JSON object)

```json
{
  "post_text":     "string — the post the classifier sees",
  "paper_title":   "string — paper this post is about",
  "paper_abstract":"string — paper abstract (may be empty)",
  "gold": {
    "kind":         "endorsement | flag | irrelevant",

    // when kind == "endorsement"
    "claim_text":   "string — the claim this post endorses or critiques",
    "magnitude_dB": -30..30,
    "criterion":    "Support | Prior",

    // when kind == "flag"
    "category":     "methodological | source | data | bias | other",
    "rationale":    "string — short concern statement",

    // when kind == "irrelevant"
    "reason":       "string — why we drop it"
  }
}
```

## Magnitude rubric (decibans, signed)

Sign convention: **positive supports the claim, negative refutes it.**

| dB | Meaning |
|----|---------|
| **+30** | Explicit strong endorsement of a specific claim, strong language |
| **+20** | Confident positive paraphrase of a claim with reasoning |
| **+10** | Mild positive — agrees but doesn't reason |
| **0**   | Excluded — drop zero-magnitude rows |
| **−10** | Mild critique or hedge |
| **−20** | Sharp critique with reasoning |
| **−30** | Explicit refutation |

## Anchor policy

Keep at least:
- 5 rows spanning the magnitude scale (one each at ±30, ±20, ±10 or ±18)
- 3 rows covering the flag categories (methodological, data, bias)
- 2 irrelevant rows

The seed file ships 10 rows. The user expands to ~20 with real captured
posts during S12 before the calibration gate flips on.

## Calibration gate

`tests/live/test_calibration_gate.py` (added in S12) asserts:

- **MAE ≤ 8 dB** on `magnitude_dB` over rows where both predicted and
  gold are endorsements.
- **kind-F1 ≥ 0.8** on `kind ∈ {endorsement, flag, irrelevant}`.

If the gate fails, iterate on the classifier system prompt in
`altendor/classify/prompts.py` (do not add new exemplars beyond ~20 rows —
prompt drift hurts more than label scarcity).
