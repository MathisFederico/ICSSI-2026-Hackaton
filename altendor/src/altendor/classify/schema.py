"""Discriminated-union pydantic schema for the post classifier (S10).

The classifier (see :mod:`altendor.classify.classifier`) drives Claude to
call the ``record_post_assessment`` tool with one of three payload shapes:

* :class:`Endorsement` — the post takes a stance on a specific claim.
* :class:`Flag` — the post raises a concern (methodological, source, etc.).
* :class:`Irrelevant` — vague praise/criticism with no claim or concern.

The three are discriminated by the literal ``kind`` field, which mirrors the
``oneOf`` shape we expose to Anthropic in :func:`tool_input_schema`.

Anthropic's tool-use API expects a single flat JSON-Schema object as the
``input_schema``; pydantic's auto-generated schema for a discriminated union
uses ``$defs`` + ``oneOf``, which works but is verbose. We hand-write a
compact equivalent (``required=["kind"]`` plus optional per-variant fields)
and let pydantic enforce the variant-specific constraints at parse time.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ZeroMagnitudeError(ValueError):
    """Raised by :func:`parse_tool_input` for zero-magnitude endorsements.

    Per the deciban rubric in :data:`altendor.classify.prompts.CLASSIFIER_SYSTEM_PROMPT`,
    zero-magnitude endorsements are dropped: ``magnitude_dB`` must be in
    ``[-30, 30]`` *and* non-zero. The schema enforces the range; this error
    enforces the non-zero rule and lets callers decide policy (typically:
    demote to :class:`Irrelevant`).
    """


class Endorsement(BaseModel):
    """Endorsement of a specific claim made by the paper."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["endorsement"] = "endorsement"
    claim_text: str = Field(min_length=1, description="One-sentence paraphrase of the claim being endorsed.")
    magnitude_dB: int = Field(
        ge=-30,
        le=30,
        description="Signed strength in decibans, in [-30, 30]; zero is excluded by policy.",
    )
    criterion: Literal["Support", "Prior"] = Field(
        description="'Support' for the paper's specific claim; 'Prior' for a broader hypothesis.",
    )
    reasoning: str = Field(min_length=1, description="One-to-two-sentence justification citing the post.")


class Flag(BaseModel):
    """A concern raised about the paper that engages no specific claim."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["flag"] = "flag"
    category: Literal["methodological", "source", "data", "bias", "other"] = Field(
        description="Category of the flag.",
    )
    rationale: str = Field(min_length=1, description="One-to-two-sentence justification citing the post.")


class Irrelevant(BaseModel):
    """Vague praise/criticism with no specific claim or concern."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["irrelevant"] = "irrelevant"
    reason: str = Field(min_length=1, description="Short reason the post is irrelevant.")


ClassifyResult = Annotated[
    Union[Endorsement, Flag, Irrelevant],
    Field(discriminator="kind"),
]
"""Tagged union over the three classifier outcomes, discriminated by ``kind``."""


_CLASSIFY_RESULT_ADAPTER: TypeAdapter[ClassifyResult] = TypeAdapter(ClassifyResult)


def tool_input_schema() -> dict[str, Any]:
    """Return the JSON-Schema Anthropic expects for the classifier tool input.

    Anthropic's tool spec prefers a single flat object schema; we list every
    possible field as an optional property, require only ``kind``, and rely
    on pydantic to reject variant-mismatched payloads at parse time
    (:class:`Endorsement`, :class:`Flag`, :class:`Irrelevant` all set
    ``extra="forbid"``).
    """
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["endorsement", "flag", "irrelevant"],
                "description": (
                    "Discriminator. 'endorsement' for a stance on a specific claim, "
                    "'flag' for a concern, 'irrelevant' for vague praise/criticism."
                ),
            },
            "claim_text": {
                "type": "string",
                "description": "Endorsement only: one-sentence paraphrase of the claim being endorsed.",
            },
            "magnitude_dB": {
                "type": "integer",
                "minimum": -30,
                "maximum": 30,
                "description": (
                    "Endorsement only: signed strength in decibans, in [-30, 30]; "
                    "zero-magnitude endorsements are dropped by policy."
                ),
            },
            "criterion": {
                "type": "string",
                "enum": ["Support", "Prior"],
                "description": (
                    "Endorsement only: 'Support' for the paper's specific claim, "
                    "'Prior' for a broader hypothesis the paper belongs to."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Endorsement only: 1-2 sentence justification citing the post.",
            },
            "category": {
                "type": "string",
                "enum": ["methodological", "source", "data", "bias", "other"],
                "description": "Flag only: category of the concern raised by the post.",
            },
            "rationale": {
                "type": "string",
                "description": "Flag only: 1-2 sentence justification citing the post.",
            },
            "reason": {
                "type": "string",
                "description": "Irrelevant only: short reason the post is irrelevant.",
            },
        },
        "required": ["kind"],
    }


def parse_tool_input(payload: dict[str, Any]) -> ClassifyResult:
    """Validate a tool-call payload against the discriminated union.

    Parameters
    ----------
    payload:
        Dict received from the Anthropic ``tool_use`` content block's
        ``input`` field.

    Returns
    -------
    ClassifyResult
        One of :class:`Endorsement`, :class:`Flag`, :class:`Irrelevant`.

    Raises
    ------
    ZeroMagnitudeError
        When the payload is an endorsement with ``magnitude_dB == 0``;
        callers typically demote to :class:`Irrelevant`.
    pydantic.ValidationError
        When the payload fails pydantic validation (missing required
        field, wrong type, etc.) — propagated unchanged.
    """
    result = _CLASSIFY_RESULT_ADAPTER.validate_python(payload)
    if isinstance(result, Endorsement) and result.magnitude_dB == 0:
        raise ZeroMagnitudeError("Endorsement with magnitude_dB == 0 is excluded by policy.")
    return result


__all__ = [
    "ClassifyResult",
    "Endorsement",
    "Flag",
    "Irrelevant",
    "ZeroMagnitudeError",
    "parse_tool_input",
    "tool_input_schema",
]
