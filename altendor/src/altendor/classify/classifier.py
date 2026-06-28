"""Single-post classifier (S10).

Drives Claude to call the ``record_post_assessment`` tool on one
``(ResolvedPost, PaperCtx)`` pair and returns a typed
:data:`~altendor.classify.schema.ClassifyResult`.

The system prompt is sent with ``cache_control={"type": "ephemeral"}`` so
batched runs (S11) get the cache discount. We force the tool call via
``tool_choice={"type": "tool", "name": ...}``; the model still has free
reasoning budget but must emit a structured payload.

Exception policy: all Anthropic API errors propagate to the caller; the
batch driver (S11) and calibration loop (S12) own retry/backoff. The
classifier only catches :class:`ZeroMagnitudeError`, which is a known and
expected outcome of the rubric — we demote the row to :class:`Irrelevant`
in place rather than letting it crash the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from altendor.classify.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_TOOL_NAME,
    build_user_message,
)
from altendor.classify.schema import (
    ClassifyResult,
    Irrelevant,
    ZeroMagnitudeError,
    parse_tool_input,
    tool_input_schema,
)

if TYPE_CHECKING:
    import anthropic

    from altendor.enrich.text_resolver import ResolvedPost


DEFAULT_MODEL: str = "claude-sonnet-4-6"
"""Default classifier model; override per-call for cost/quality trade-offs."""

DEFAULT_MAX_TOKENS: int = 1024
"""Max tokens for the classifier response; the tool payload itself is small."""

_ZERO_MAGNITUDE_REASON: str = "Zero-magnitude endorsement dropped by classifier policy."


@dataclass(frozen=True)
class PaperCtx:
    """Paper-side context passed to the classifier."""

    title: str
    abstract: str | None
    url: str | None = None


def _extract_tool_input(response: object) -> dict[str, Any]:
    """Pull the tool-use payload out of an Anthropic ``messages.create`` response.

    Scans ``response.content`` for the first block of type ``tool_use``
    whose ``name`` matches :data:`CLASSIFIER_TOOL_NAME` and returns its
    ``.input`` dict.
    """
    content = getattr(response, "content", None)
    if content is None:
        raise ValueError("Anthropic response has no .content attribute")

    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != CLASSIFIER_TOOL_NAME:
            continue
        payload = getattr(block, "input", None)
        if not isinstance(payload, dict):
            raise ValueError(
                f"tool_use block for {CLASSIFIER_TOOL_NAME!r} has non-dict .input ({type(payload).__name__})",
            )
        return cast("dict[str, Any]", payload)

    raise ValueError(f"No tool_use block named {CLASSIFIER_TOOL_NAME!r} in classifier response")


def classify_post(
    client: anthropic.Anthropic,
    post: ResolvedPost,
    paper: PaperCtx,
    *,
    model: str = DEFAULT_MODEL,
) -> ClassifyResult:
    """Classify one ``(post, paper)`` pair into Endorsement / Flag / Irrelevant.

    Parameters
    ----------
    client:
        A configured :class:`anthropic.Anthropic` client. The caller owns
        API-key plumbing; we never read env vars here.
    post:
        Resolved post from :mod:`altendor.enrich.text_resolver`. Its
        ``text``, ``url``, and ``text_confidence`` populate the user message.
    paper:
        Paper title/abstract context.
    model:
        Anthropic model name; defaults to :data:`DEFAULT_MODEL`.

    Returns
    -------
    ClassifyResult
        One of :class:`~altendor.classify.schema.Endorsement`,
        :class:`~altendor.classify.schema.Flag`,
        :class:`~altendor.classify.schema.Irrelevant`.

    Notes
    -----
    Zero-magnitude endorsements are demoted to :class:`Irrelevant` in place
    (see :data:`_ZERO_MAGNITUDE_REASON`). All Anthropic API errors propagate
    to the caller; the batch driver in S11 owns retries.
    """
    user_message = build_user_message(
        post_text=post.text,
        post_url=post.url,
        paper_title=paper.title,
        paper_abstract=paper.abstract,
        text_confidence=post.text_confidence,
    )

    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": CLASSIFIER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_message}],
        tools=[
            {
                "name": CLASSIFIER_TOOL_NAME,
                "description": "Record the classification of this post-about-paper.",
                "input_schema": tool_input_schema(),
            },
        ],
        tool_choice={"type": "tool", "name": CLASSIFIER_TOOL_NAME},
    )

    payload = _extract_tool_input(response)
    try:
        return parse_tool_input(payload)
    except ZeroMagnitudeError:
        return Irrelevant(reason=_ZERO_MAGNITUDE_REASON)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "PaperCtx",
    "classify_post",
]
