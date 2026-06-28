"""Paper → Question router (S15).

This module routes each ingested paper to exactly ONE of the three debate
Questions backing the DeltaBay "The optimal scientific system" debate. Routing
is per-paper and is driven by a single forced tool call on the Claude API:
the classifier picks one of ``THREE_QUESTIONS`` and emits a confidence in
``[0, 1]``.

Two public entry points:

* :func:`route_paper_to_question` — single-paper classifier; returns
  ``(question_id, confidence)``.
* :func:`diversify_routes` — post-hoc rebalance to ensure that, when possible,
  no Question ends up with zero papers.

The rebalance is deliberately conservative: we only move a paper if some
Question has zero papers AND another has at least two. The lowest-confidence
paper in the largest-occupancy bucket is the one we relocate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic

TOOL_NAME: str = "route_paper"
"""Name of the Anthropic tool S15 forces the model to call."""

DEFAULT_MODEL: str = "claude-haiku-4-5"
"""Cheap, fast default. 3-way routing with clear titles doesn't need Sonnet."""

DEFAULT_MAX_TOKENS: int = 256
"""Tool call output is tiny; 256 leaves ample headroom for retries / hedging."""


@dataclass(frozen=True)
class QuestionStub:
    """A reference to one of the DeltaBay debate Questions.

    Attributes
    ----------
    id:
        Stable DeltaBay slug, e.g. ``"question:peer-review"``.
    title:
        Human-readable question title shown to the LLM in the system prompt.
    short_title:
        Optional short label used by UI/log output; not sent to the model.
    """

    id: str
    title: str
    short_title: str = ""


THREE_QUESTIONS: tuple[QuestionStub, ...] = (
    QuestionStub(
        "question:research-integrity",
        "How to preserve the research quality and integrity?",
        "Research integrity",
    ),
    QuestionStub(
        "question:measure-progress",
        "What is a good measure of scientific progress?",
        "Measure progress",
    ),
    QuestionStub(
        "question:peer-review",
        "What are the limits and keepers of the peer-review and journals scientific system?",
        "Peer review",
    ),
)
"""The three Questions backing the funding-debate. Cross-stage contract with DeltaBay."""


@dataclass(frozen=True)
class PaperForRouting:
    """Minimal paper payload required to route to a Question.

    Attributes
    ----------
    doi:
        Canonical DOI; used as the key in the routing dict downstream.
    title:
        Paper title.
    abstract:
        Paper abstract, or ``None`` when not available.
    """

    doi: str
    title: str
    abstract: str | None


def _build_system_prompt(questions: tuple[QuestionStub, ...]) -> str:
    """Build the routing system prompt enumerating the candidate Questions.

    The prompt is short and deterministic so the model behaves identically
    across calls (no exemplars needed for a 3-way routing problem).
    """
    lines = [
        "You assign a scientific paper to ONE of three meta-science debate questions.",
        "You must pick exactly one. Confidence reflects how clearly the paper's "
        "content addresses that specific question.",
        "",
        "Questions:",
    ]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. {q.title}")
    lines.append("")
    lines.append(f"Call the `{TOOL_NAME}` tool with the chosen question id and your confidence.")
    return "\n".join(lines)


def _build_user_message(paper: PaperForRouting) -> str:
    """Build the per-paper user message (title + abstract)."""
    abstract_line = paper.abstract.strip() if paper.abstract else "(no abstract available)"
    return "\n".join(
        [
            "## Paper",
            f"- Title: {paper.title.strip()}",
            f"- Abstract: {abstract_line}",
            "",
            "## Task",
            f"Route this paper by calling the `{TOOL_NAME}` tool exactly once.",
        ]
    )


def _build_tool_schema(questions: tuple[QuestionStub, ...]) -> dict[str, Any]:
    """Build the JSON-Schema for the ``route_paper`` tool's ``input_schema``."""
    return {
        "type": "object",
        "properties": {
            "chosen_question_id": {
                "type": "string",
                "enum": [q.id for q in questions],
                "description": "The DeltaBay question id the paper is routed to.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Routing confidence in [0, 1].",
            },
        },
        "required": ["chosen_question_id", "confidence"],
    }


def _extract_tool_input(response: Any) -> dict[str, Any]:  # noqa: ANN401 - Anthropic SDK Message type is loose; we duck-type
    """Pull the ``route_paper`` tool_use block's ``input`` from an Anthropic message.

    Raises
    ------
    ValueError
        When no ``tool_use`` block matching :data:`TOOL_NAME` is present.
    """
    content = getattr(response, "content", None)
    if not content:
        raise ValueError("Anthropic response had no content blocks.")
    for block in content:
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == TOOL_NAME:
            payload = getattr(block, "input", None)
            if not isinstance(payload, dict):
                raise ValueError(f"tool_use block had non-dict input: {payload!r}")
            return payload
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} in Anthropic response.")


def route_paper_to_question(
    client: anthropic.Anthropic,
    paper: PaperForRouting,
    *,
    questions: tuple[QuestionStub, ...] = THREE_QUESTIONS,
    model: str = DEFAULT_MODEL,
) -> tuple[str, float]:
    """Route a single paper to one of ``questions``.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` client (or a duck-typed stand-in exposing
        ``client.messages.create``).
    paper:
        The paper to route. Only ``title`` and ``abstract`` are read.
    questions:
        Candidate Questions. Defaults to :data:`THREE_QUESTIONS`.
    model:
        Anthropic model id. Defaults to :data:`DEFAULT_MODEL` (Haiku).

    Returns
    -------
    tuple[str, float]
        ``(chosen_question_id, confidence)`` with confidence in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the model returned no ``route_paper`` tool call, an unknown
        question id, or an out-of-range confidence.
    """
    system_text = _build_system_prompt(questions)
    user_text = _build_user_message(paper)
    tool_schema = _build_tool_schema(questions)

    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=system_text,
        messages=[{"role": "user", "content": user_text}],
        tools=[
            {
                "name": TOOL_NAME,
                "description": "Route the paper to exactly one debate question with a confidence in [0, 1].",
                "input_schema": tool_schema,
            }
        ],
        tool_choice={"type": "tool", "name": TOOL_NAME},
    )

    payload = _extract_tool_input(response)

    chosen = payload.get("chosen_question_id")
    confidence = payload.get("confidence")

    valid_ids = {q.id for q in questions}
    if not isinstance(chosen, str) or chosen not in valid_ids:
        raise ValueError(f"chosen_question_id {chosen!r} is not one of {sorted(valid_ids)!r}")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError(f"confidence must be a number; got {confidence!r}")
    confidence_f = float(confidence)
    if not (0.0 <= confidence_f <= 1.0):
        raise ValueError(f"confidence {confidence_f!r} is outside [0, 1]")

    return chosen, confidence_f


def diversify_routes(
    routes: dict[str, tuple[str, float]],
    *,
    questions: tuple[QuestionStub, ...] = THREE_QUESTIONS,
) -> dict[str, str]:
    """Post-hoc rebalance to avoid empty Questions where feasible.

    Algorithm
    ---------
    1. Start with the input routing.
    2. While some Question has zero papers AND another has ≥2:
       a. Pick an empty Question (deterministic: first in ``questions`` order).
       b. Find the largest-occupancy Question (ties broken by ``questions``
          order). Within it, pick the paper with lowest confidence (ties
          broken by paper insertion order). Reassign it to the empty Question.
    3. Stop once every Question is non-empty OR no donor bucket has ≥2.
    4. The returned dict drops confidence; key order matches the input.

    Parameters
    ----------
    routes:
        ``{paper_doi: (question_id, confidence)}`` from
        :func:`route_paper_to_question`.
    questions:
        Candidate Questions. Defaults to :data:`THREE_QUESTIONS`.

    Returns
    -------
    dict[str, str]
        ``{paper_doi: final_question_id}``, in the original key order.
    """
    # Work on a mutable copy. Insertion order is preserved by dict.
    current: dict[str, str] = {doi: qid for doi, (qid, _) in routes.items()}
    confidences: dict[str, float] = {doi: conf for doi, (_, conf) in routes.items()}
    question_ids = [q.id for q in questions]

    def _counts() -> dict[str, int]:
        counts = {qid: 0 for qid in question_ids}
        for qid in current.values():
            if qid in counts:
                counts[qid] += 1
        return counts

    while True:
        counts = _counts()
        empties = [qid for qid in question_ids if counts[qid] == 0]
        if not empties:
            break
        # Donor: largest-occupancy bucket with >= 2 papers, ties by question order.
        donor_candidates = [(qid, counts[qid]) for qid in question_ids if counts[qid] >= 2]
        if not donor_candidates:
            break
        # Stable sort: highest count first, original question order on ties.
        donor_candidates.sort(key=lambda kv: (-kv[1], question_ids.index(kv[0])))
        donor_qid = donor_candidates[0][0]
        target_qid = empties[0]

        # Pick lowest-confidence paper from the donor. Ties broken by paper
        # insertion order (dict iteration order is insertion order in 3.7+).
        donor_papers = [doi for doi, qid in current.items() if qid == donor_qid]
        # The min() call is stable: the first matching paper wins on ties.
        victim = min(donor_papers, key=lambda doi: confidences[doi])
        current[victim] = target_qid

    # Rebuild in the input order to be explicit (CPython dicts preserve order,
    # but this is the cross-version contract callers expect).
    return {doi: current[doi] for doi in routes}


__all__ = [
    "DEFAULT_MODEL",
    "PaperForRouting",
    "QuestionStub",
    "THREE_QUESTIONS",
    "TOOL_NAME",
    "diversify_routes",
    "route_paper_to_question",
]
