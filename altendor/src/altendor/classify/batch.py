"""Anthropic Message Batches API wrapper for the post classifier (S11).

The batch driver mirrors :func:`altendor.classify.classifier.classify_post`
on a per-row basis but submits all requests through the Batches endpoint at
50% cost. Reusing the byte-identical system block from
:data:`altendor.classify.prompts.CLASSIFIER_SYSTEM_PROMPT` keeps prompt-cache
hits warm across the one-off and batch paths.

Lifecycle:

1. Build a list of :class:`BatchRequestSpec` (one per ``(post, paper)`` pair).
2. :func:`submit_batch` validates uniqueness/non-emptiness and posts the
   batch, returning its ``msgbatch_*`` id.
3. :func:`poll_until_done` blocks (notebook-friendly) until the batch's
   ``processing_status`` becomes ``"ended"``.
4. :func:`parse_batch_results` iterates the per-request results and turns
   each into a :class:`BatchResultRow` carrying either a parsed
   :class:`~altendor.classify.schema.ClassifyResult` or an ``error_reason``.

Policy parity with the one-off classifier:

* Zero-magnitude endorsements are demoted to
  :class:`~altendor.classify.schema.Irrelevant` in place.
* Tool-use payload extraction follows the same "first matching tool_use
  block" rule.
* All non-success outcomes (errored / canceled / expired / malformed
  payload / missing tool_use) surface as :class:`BatchResultRow` with
  ``result=None`` and a populated ``error_reason``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

from altendor.classify.classifier import DEFAULT_MODEL, PaperCtx
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


_BATCH_MAX_TOKENS: int = 1024
"""Per-request ``max_tokens`` for batch entries (matches the one-off path)."""

_ZERO_MAGNITUDE_REASON: str = "Zero-magnitude endorsement dropped by classifier policy."
"""Same reason text the one-off classifier uses; keep parity for downstream."""


@dataclass(frozen=True)
class BatchRequestSpec:
    """One ``(post, paper)`` pair to classify in a batch.

    ``custom_id`` must be unique within a single batch submission and is the
    only way to re-key results back to inputs once they come back from the
    API (results may arrive in any order).
    """

    custom_id: str
    post: ResolvedPost
    paper: PaperCtx


@dataclass(frozen=True)
class BatchResultRow:
    """One parsed batch outcome. Exactly one of ``result``/``error_reason`` is set.

    * On success: ``result`` is the parsed
      :class:`~altendor.classify.schema.ClassifyResult` (with zero-magnitude
      endorsements already demoted to :class:`Irrelevant`).
    * On any failure: ``result`` is ``None`` and ``error_reason`` carries a
      short human-readable label (errored type, expired, missing tool_use,
      malformed payload, etc.).
    """

    custom_id: str
    result: ClassifyResult | None
    error_reason: str | None


def _build_params(spec: BatchRequestSpec, model: str) -> dict[str, Any]:
    """Render the ``params`` payload for one batch entry.

    The shape mirrors :func:`altendor.classify.classifier.classify_post`'s
    ``messages.create`` kwargs exactly so the cached system prefix is
    byte-identical across one-off and batch paths.
    """
    user_message = build_user_message(
        post_text=spec.post.text,
        post_url=spec.post.url,
        paper_title=spec.paper.title,
        paper_abstract=spec.paper.abstract,
        text_confidence=spec.post.text_confidence,
    )
    return {
        "model": model,
        "max_tokens": _BATCH_MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": CLASSIFIER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [{"role": "user", "content": user_message}],
        "tools": [
            {
                "name": CLASSIFIER_TOOL_NAME,
                "description": "Record the classification of this post-about-paper.",
                "input_schema": tool_input_schema(),
            },
        ],
        "tool_choice": {"type": "tool", "name": CLASSIFIER_TOOL_NAME},
    }


def build_batch_requests(
    specs: list[BatchRequestSpec],
    *,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Render :class:`BatchRequestSpec`s into the dict shape the Batches API accepts.

    Each entry has the form ``{"custom_id": <str>, "params": {<messages.create kwargs>}}``
    where ``params`` mirrors the per-call shape produced by
    :func:`altendor.classify.classifier.classify_post` so prompt-cache hits
    survive across one-off and batch paths.

    Parameters
    ----------
    specs:
        The rows to classify. May be empty (the rendered list is also
        empty); callers that need stricter validation should use
        :func:`submit_batch`.
    model:
        Anthropic model name; defaults to
        :data:`altendor.classify.classifier.DEFAULT_MODEL`.
    """
    return [{"custom_id": spec.custom_id, "params": _build_params(spec, model)} for spec in specs]


def _check_specs(specs: list[BatchRequestSpec]) -> None:
    """Raise ``ValueError`` if *specs* is empty or has duplicate ``custom_id``s."""
    if not specs:
        raise ValueError("submit_batch requires at least one BatchRequestSpec; got empty list.")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in specs:
        if spec.custom_id in seen:
            duplicates.add(spec.custom_id)
        seen.add(spec.custom_id)
    if duplicates:
        sorted_dupes = ", ".join(sorted(duplicates))
        raise ValueError(f"submit_batch requires unique custom_ids; duplicates: {sorted_dupes}")


def submit_batch(
    client: anthropic.Anthropic,
    specs: list[BatchRequestSpec],
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """Submit a batch and return its id (e.g. ``msgbatch_...``).

    Raises
    ------
    ValueError
        If *specs* is empty or any ``custom_id`` is repeated.
    """
    _check_specs(specs)
    requests = build_batch_requests(specs, model=model)
    # The SDK types ``requests`` as ``Iterable[batch_create_params.Request]``
    # (a TypedDict). Our dicts match the runtime shape; cast to bypass the
    # nominal-typing check without dragging the TypedDict into our public API.
    batch = client.messages.batches.create(requests=cast("Any", requests))
    return str(batch.id)


def poll_until_done(
    client: anthropic.Anthropic,
    batch_id: str,
    *,
    interval_s: int = 30,
    timeout_s: int | None = None,
    on_tick: Callable[[object], None] | None = None,
) -> object:
    """Block until the batch's ``processing_status`` becomes ``"ended"``.

    Parameters
    ----------
    client:
        Configured :class:`anthropic.Anthropic` client.
    batch_id:
        The ``msgbatch_*`` id returned by :func:`submit_batch`.
    interval_s:
        Sleep duration between polls. ``0`` means tight-loop (used by tests).
    timeout_s:
        Optional overall timeout; ``None`` (default) waits forever.
    on_tick:
        Optional callable invoked with the latest ``MessageBatch`` object
        after each retrieve. Convenient for notebook progress prints.

    Returns
    -------
    object
        The final ``MessageBatch`` (``processing_status == "ended"``).

    Raises
    ------
    TimeoutError
        If *timeout_s* is set and elapsed before the batch ended.
    """
    start = time.monotonic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if on_tick is not None:
            on_tick(batch)
        status = getattr(batch, "processing_status", None)
        if status == "ended":
            return batch
        if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
            raise TimeoutError(
                f"poll_until_done: batch {batch_id!r} did not end within {timeout_s}s "
                f"(last status: {status!r}).",
            )
        if interval_s > 0:
            time.sleep(interval_s)


def _extract_tool_input_from_message(message: object) -> dict[str, Any]:
    """Pull the classifier tool-use payload from a batched message response.

    Mirrors :func:`altendor.classify.classifier._extract_tool_input` so the
    batch and one-off paths share extraction semantics.
    """
    content = getattr(message, "content", None)
    if content is None:
        raise ValueError("Anthropic message has no .content attribute")

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

    raise ValueError(f"No tool_use block named {CLASSIFIER_TOOL_NAME!r} in batched response")


def _parse_succeeded(message: object) -> tuple[ClassifyResult | None, str | None]:
    """Turn one succeeded batch message into ``(result, error_reason)``.

    Zero-magnitude endorsements are demoted to :class:`Irrelevant` in place;
    extraction or validation failures fall through to an ``error_reason``.
    """
    try:
        payload = _extract_tool_input_from_message(message)
    except ValueError as exc:
        return None, f"missing_tool_use: {exc}"
    try:
        result = parse_tool_input(payload)
    except ZeroMagnitudeError:
        return Irrelevant(reason=_ZERO_MAGNITUDE_REASON), None
    except Exception as exc:  # pydantic ValidationError, type errors, ...
        return None, f"malformed_payload: {type(exc).__name__}: {exc}"
    return result, None


def _parse_one(entry: object) -> BatchResultRow:
    """Convert one ``MessageBatchIndividualResponse``-like object to a row."""
    custom_id = cast("str", getattr(entry, "custom_id", ""))
    result_obj = getattr(entry, "result", None)
    result_type = getattr(result_obj, "type", None)

    if result_type == "succeeded":
        message = getattr(result_obj, "message", None)
        if message is None:
            return BatchResultRow(custom_id=custom_id, result=None, error_reason="succeeded_without_message")
        parsed, error_reason = _parse_succeeded(message)
        return BatchResultRow(custom_id=custom_id, result=parsed, error_reason=error_reason)

    if result_type in {"errored", "canceled", "expired"}:
        error = getattr(result_obj, "error", None)
        error_type = getattr(error, "type", None) if error is not None else None
        error_message = getattr(error, "message", None) if error is not None else None
        parts: list[str] = [str(result_type)]
        if error_type:
            parts.append(str(error_type))
        if error_message:
            parts.append(str(error_message))
        return BatchResultRow(custom_id=custom_id, result=None, error_reason=": ".join(parts))

    return BatchResultRow(
        custom_id=custom_id,
        result=None,
        error_reason=f"unknown_result_type: {result_type!r}",
    )


def parse_batch_results(client: anthropic.Anthropic, batch_id: str) -> list[BatchResultRow]:
    """Iterate the batch's results and turn each into a :class:`BatchResultRow`.

    Rows are returned in the order the API yields them; callers should
    re-key by :attr:`BatchResultRow.custom_id` when they need a specific
    ordering.
    """
    return [_parse_one(entry) for entry in client.messages.batches.results(batch_id)]


__all__ = [
    "BatchRequestSpec",
    "BatchResultRow",
    "build_batch_requests",
    "parse_batch_results",
    "poll_until_done",
    "submit_batch",
]
