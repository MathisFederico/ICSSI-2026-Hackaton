"""Cached classifier system prompt + per-post user message builder (S9).

This module is **prompt scaffolding only** — it makes no Claude API calls.
The downstream S10 classifier consumes ``CLASSIFIER_SYSTEM_PROMPT`` as a
prompt-cache-friendly system message and calls :func:`build_user_message`
per post.

Determinism is load-bearing: Anthropic prompt caching only hits when the
cached prefix is byte-identical across calls, so :func:`_build_system_prompt`
sorts and renders exemplars in a stable order at module import time and
assigns the result to :data:`CLASSIFIER_SYSTEM_PROMPT` once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLASSIFIER_TOOL_NAME: str = "record_post_assessment"
"""Name of the Anthropic tool S10 will register for the classifier to call."""

DEFAULT_CALIBRATION_PATH: Path = Path(__file__).parents[3] / "data" / "calibration" / "labeled.jsonl"
"""Default path to the calibration JSONL used to render the exemplars block."""

_KIND_ORDER: dict[str, int] = {"endorsement": 0, "flag": 1, "irrelevant": 2}


def _load_calibration_rows(jsonl_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load calibration rows from a JSONL file.

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL calibration file. ``None`` uses
        :data:`DEFAULT_CALIBRATION_PATH`.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows in file order. Blank lines are skipped.
    """
    path = Path(jsonl_path) if jsonl_path is not None else DEFAULT_CALIBRATION_PATH
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _exemplar_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    """Deterministic sort key for exemplars.

    Order: kind ascending (endorsement < flag < irrelevant), then for
    endorsements by magnitude descending (so +28 lands before -28); for
    flags by category alphabetically; for irrelevant by reason. We use
    the post_text as a final tiebreaker so the order is fully stable
    even on duplicate primary keys.
    """
    gold = row.get("gold", {})
    kind = str(gold.get("kind", ""))
    kind_rank = _KIND_ORDER.get(kind, 99)

    if kind == "endorsement":
        # Sort by magnitude descending => negate for ascending sort.
        mag = int(gold.get("magnitude_dB", 0))
        secondary = -mag
        tertiary = str(gold.get("claim_text", ""))
    elif kind == "flag":
        secondary = 0
        tertiary = f"{gold.get('category', '')}|{gold.get('rationale', '')}"
    else:
        secondary = 0
        tertiary = str(gold.get("reason", ""))

    return (kind_rank, secondary, tertiary, str(row.get("post_text", "")))


def _render_gold(gold: dict[str, Any]) -> str:
    """Render the gold label as a compact, sorted-key JSON string.

    Keys are sorted so the rendering is deterministic regardless of JSON
    object key order in the source file. Compact (no indent) to keep the
    cached prefix tight.
    """
    return json.dumps(gold, sort_keys=True, ensure_ascii=False, separators=(", ", ": "))


def _render_exemplar(row: dict[str, Any], index: int, paper_ref: str) -> str:
    """Render a single exemplar as a compact Markdown block.

    ``paper_ref`` is a short label like ``Paper A`` referencing the paper-
    context table emitted alongside the exemplars; this lets us avoid
    repeating long abstracts inside every example.
    """
    post_text = str(row.get("post_text", "")).strip()
    gold = row.get("gold", {})

    return (
        f"### Example {index} ({paper_ref})\n"
        f"- Post: {post_text}\n"
        f"- Tool call: `{_render_gold(gold)}`"
    )


def build_exemplars_block(jsonl_path: Path | str | None = None) -> str:
    """Render labelled exemplars as a stable, deterministic text block.

    Rows are sorted by :func:`_exemplar_sort_key` so the produced string is
    invariant to JSONL line order. To keep the cached prefix tight we emit
    each unique paper's title+abstract once in a ``Paper Contexts`` table,
    then reference papers by short labels (``Paper A``, ``Paper B``, ...)
    inside each example. Paper labels are assigned by first-appearance
    order in the *sorted* exemplar list, so the rendering is fully
    deterministic.

    Parameters
    ----------
    jsonl_path:
        Optional override for the calibration JSONL path; defaults to
        :data:`DEFAULT_CALIBRATION_PATH`.

    Returns
    -------
    str
        Deterministic Markdown block listing paper contexts then exemplars.
    """
    rows = _load_calibration_rows(jsonl_path)
    rows_sorted = sorted(rows, key=_exemplar_sort_key)

    # Assign short, stable labels to each unique paper in sorted-row order.
    paper_label: dict[tuple[str, str], str] = {}
    paper_order: list[tuple[str, str]] = []
    for row in rows_sorted:
        title = str(row.get("paper_title", "")).strip()
        abstract = str(row.get("paper_abstract", "")).strip()
        key = (title, abstract)
        if key not in paper_label:
            paper_label[key] = f"Paper {chr(ord('A') + len(paper_order))}"
            paper_order.append(key)

    context_lines = ["## Paper Contexts"]
    for key in paper_order:
        title, abstract = key
        label = paper_label[key]
        context_lines.append(f"- {label}: _{title}_ — {abstract}")

    exemplar_lines = ["## Calibrated Examples"]
    for i, row in enumerate(rows_sorted, start=1):
        title = str(row.get("paper_title", "")).strip()
        abstract = str(row.get("paper_abstract", "")).strip()
        label = paper_label[(title, abstract)]
        exemplar_lines.append(_render_exemplar(row, i, label))

    return "\n".join(context_lines) + "\n\n" + "\n\n".join(exemplar_lines)


_ROLE_SECTION: str = (
    "## Role\n"
    "You classify a social-media post that mentions a scientific paper. "
    "Extract the specific claim (if any) the post makes about the paper, "
    "then decide: endorsement of a claim, flag of a concern, or irrelevant. "
    "You classify the post's stance, not the paper itself."
)

_DECISION_TREE_SECTION: str = (
    "## Decision Tree (pick ONE)\n"
    "1. **endorsement** — post references/paraphrases a specific claim, "
    "finding, or conclusion of the paper AND takes a stance. Fill "
    "`claim_text`; sign `magnitude_dB` positive for support, negative for "
    "refute.\n"
    "2. **flag** — post raises a methodological, source, data, bias, or "
    "other concern but engages no specific claim. Pick `category`.\n"
    "3. **irrelevant** — vague praise/criticism with no claim or concern. "
    '"Great paper!" / "Highly recommend!" → irrelevant.'
)

_DECIBAN_RUBRIC_SECTION: str = (
    "## Deciban Rubric (`magnitude_dB`, integer)\n"
    "Zero-magnitude rows are dropped; if the post does not clearly land at "
    "|10| or stronger, prefer `irrelevant`.\n\n"
    "| dB  | Meaning |\n"
    "|-----|---------|\n"
    "| +30 | Explicit strong endorsement of a specific claim, strong language |\n"
    "| +20 | Confident positive paraphrase of a claim with reasoning |\n"
    "| +10 | Mild positive — agrees but doesn't reason |\n"
    "|   0 | Excluded — drop zero-magnitude rows |\n"
    "| -10 | Mild critique or hedge |\n"
    "| -20 | Sharp critique with reasoning |\n"
    "| -30 | Explicit refutation |\n\n"
    '`criterion = "Support"` by default. Use `"Prior"` only when the post '
    "endorses a broader hypothesis the paper belongs to, not the paper's "
    "specific finding."
)

_FLAG_CATEGORIES_SECTION: str = (
    "## Flag Categories\n"
    "- `methodological` — study design, sample size, statistical analysis.\n"
    "- `source` — retracted, paywalled, broken DOI, unavailable source.\n"
    "- `data` — dataset issue, missing data, irreproducible.\n"
    "- `bias` — conflict of interest, undisclosed funding, editorial-board overlap.\n"
    "- `other` — anything else."
)

_OUTPUT_SPEC_SECTION: str = (
    "## Output Specification\n"
    f"Call `{CLASSIFIER_TOOL_NAME}` exactly once with one of these shapes "
    "(no free-text alongside). Authoritative schema lives in the tool "
    "definition; prose below is the contract.\n\n"
    "- Endorsement: "
    '`{"kind": "endorsement", "claim_text": "<one-sentence paraphrase>", '
    '"magnitude_dB": <int in [-30,30], non-zero>, '
    '"criterion": "Support"|"Prior", '
    '"reasoning": "<1-2 sentences citing the post>"}`\n'
    "- Flag: "
    '`{"kind": "flag", '
    '"category": "methodological"|"source"|"data"|"bias"|"other", '
    '"rationale": "<1-2 sentences citing the post>"}`\n'
    "- Irrelevant: "
    '`{"kind": "irrelevant", "reason": "<short reason>"}`'
)

_REFUSAL_SECTION: str = (
    "## Refusal Conditions\n"
    "- Post off-topic from the paper → `irrelevant`.\n"
    "- Non-English post: transliterate the claim into English when "
    "extractable; otherwise `irrelevant`.\n"
    "- Never invent claims the post does not make. When in doubt between "
    "`endorsement` and `irrelevant`, choose `irrelevant`."
)


def _build_system_prompt() -> str:
    """Construct the full cached system prompt.

    Called once at module import to populate :data:`CLASSIFIER_SYSTEM_PROMPT`.
    Exposed (with a leading underscore) so tests can assert byte-stability
    across repeated calls.
    """
    exemplars = build_exemplars_block()
    sections = [
        _ROLE_SECTION,
        _DECISION_TREE_SECTION,
        _DECIBAN_RUBRIC_SECTION,
        _FLAG_CATEGORIES_SECTION,
        exemplars,
        _OUTPUT_SPEC_SECTION,
        _REFUSAL_SECTION,
    ]
    return "\n\n".join(sections)


CLASSIFIER_SYSTEM_PROMPT: str = _build_system_prompt()
"""Frozen, prompt-cache-friendly system prompt for the post classifier."""


def build_user_message(
    post_text: str,
    post_url: str | None,
    paper_title: str,
    paper_abstract: str | None,
    text_confidence: str = "high",
) -> str:
    """Build the per-call user message for the classifier.

    This message is NOT prompt-cached — only the system prompt is. The user
    message carries everything that varies per post: the post text, source
    URL (if known), the paper title, the paper abstract (if known), and an
    optional low-confidence hint for cases where Altmetric truncated the
    post and downstream enrichment failed.

    Parameters
    ----------
    post_text:
        The (possibly truncated) post body.
    post_url:
        Source URL for the post; ``None`` when not available.
    paper_title:
        Title of the paper the post is mentioning.
    paper_abstract:
        Abstract of the paper, or ``None`` when not available.
    text_confidence:
        ``"high"`` (default) or ``"low"``. When ``"low"`` an explicit hint
        is appended instructing the model to be conservative.

    Returns
    -------
    str
        Markdown-formatted user message ready to send as the user turn.
    """
    abstract_line = paper_abstract.strip() if paper_abstract else "(no abstract available)"
    url_line = post_url.strip() if post_url else "(no URL available)"

    parts = [
        "## Paper",
        f"- Title: {paper_title.strip()}",
        f"- Abstract: {abstract_line}",
        "",
        "## Post",
        f"- URL: {url_line}",
        f"- Text: {post_text.strip()}",
    ]

    if text_confidence == "low":
        parts.extend(
            [
                "",
                "Note: the post text may be truncated or low-confidence "
                "(Altmetric did not return full text and downstream enrichment "
                "failed). Be conservative — prefer `irrelevant` when the "
                "available text does not clearly articulate a claim or "
                "concern.",
            ]
        )

    parts.extend(
        [
            "",
            "## Task",
            f"Classify the post by calling the `{CLASSIFIER_TOOL_NAME}` tool exactly once.",
        ]
    )

    return "\n".join(parts)
