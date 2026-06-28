"""Pydantic v2 schema for the neutral debate JSON emitted at the end of the pipeline.

This is the contract between the altendor pipeline (this repo) and the
downstream DeltaBay loader (stage S26, out of scope here). The shapes
crystallize the upstream producers' outputs:

* :func:`altendor.sources.altmetric.top_papers` (+ OpenAlex abstract
  enrichment) -> :class:`PaperRecord`.
* :class:`altendor.enrich.text_resolver.ResolvedPost` -> per-post fields on
  :class:`EndorsementRow` / :class:`FlagRow` and (deduped) on
  :class:`Participant`.
* :class:`altendor.classify.schema.Endorsement` / :class:`Flag` -> the two
  evidence row types, attached either to a subclaim (endorsements only) or
  to the paper-level :class:`EvidenceNode` (flags + fallback endorsements).
* :class:`altendor.cluster.claims.ClaimCluster` -> :class:`SubclaimNode`.
* :data:`altendor.route.question_router.THREE_QUESTIONS` -> the per-paper
  ``routed_question_id`` field plus the top-level ``questions`` list.

Deviations from the ticket's target shape
-----------------------------------------
* ``EndorsementRow.magnitude`` is constrained to ``[-30, 30]`` and **non-zero**
  by a Pydantic ``field_validator`` (the upstream classifier already drops
  zero-magnitude endorsements; this is defence in depth and surfaces the
  policy at the contract boundary). See also the duplicate guard in
  :func:`altendor.assemble.builder.build_intermediate`.
* Field names are exactly as in the ticket; no other deviations.

The DeltaBay loader renames snake_case here to JSON-LD camelCase and stamps
``@type`` annotations; the pipeline output stays vocabulary-neutral.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Platform values come from altendor.enrich.text_resolver.Platform; we keep the
# field as a free-form string here so the JSON contract isn't coupled to the
# specific platform enum (downstream loaders treat it as an opaque label).


class QuestionStub(BaseModel):
    """Reference to one of the three debate Questions (see ``THREE_QUESTIONS``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    short_title: str


class Participant(BaseModel):
    """A social-media account that authored at least one ingested post."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    handle: str | None
    platform: str


class SourceDocument(BaseModel):
    """The paper itself, used as the ``source_document`` on the EvidenceNode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doi: str
    url: str


class EndorsementRow(BaseModel):
    """One endorsement of either a specific subclaim or the paper as a whole.

    ``magnitude`` is constrained to ``[-30, 30] \\ {0}`` — zero-magnitude
    endorsements are dropped by classifier policy (see
    :class:`altendor.classify.schema.Endorsement` and ``ZeroMagnitudeError``).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    participant_id: str
    magnitude: int = Field(ge=-30, le=30)
    criterion: Literal["Support", "Prior"]
    created_at: str
    source_post_url: str
    source_text: str

    @field_validator("magnitude")
    @classmethod
    def _nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("magnitude must be non-zero (zero-magnitude endorsements are dropped by policy)")
        return value


class FlagRow(BaseModel):
    """One concern raised against the paper (attached to EvidenceNode, never a subclaim)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal["methodological", "source", "data", "bias", "other"]
    rationale: str
    raised_by_id: str
    raised_by_name: str
    created_at: str
    source_post_url: str


class SubclaimNode(BaseModel):
    """A canonical claim made by the paper, with the endorsements that support it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    member_post_ids: list[str]
    endorsements: list[EndorsementRow]
    flags: list[FlagRow]


class EvidenceNode(BaseModel):
    """Paper-level evidence container: the source document plus paper-level rows."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_document: SourceDocument
    endorsements: list[EndorsementRow]
    flags: list[FlagRow]


class AnswerNode(BaseModel):
    """One paper's answer to the routed debate question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    short_title: str
    evidence: EvidenceNode
    subclaims: list[SubclaimNode]


class PaperRecord(BaseModel):
    """One row from :func:`top_papers` enriched with its assembled answer."""

    model_config = ConfigDict(extra="forbid")

    doi: str
    ro_id: str
    title: str
    abstract: str | None
    altmetric_score: float
    routed_question_id: str
    answer: AnswerNode


class IntermediateDebate(BaseModel):
    """Top-level neutral debate JSON. Consumed by the DeltaBay loader (S26)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    debate_id: str
    generated_at: str
    questions: list[QuestionStub]
    participants: list[Participant]
    papers: list[PaperRecord]


__all__ = [
    "AnswerNode",
    "EndorsementRow",
    "EvidenceNode",
    "FlagRow",
    "IntermediateDebate",
    "PaperRecord",
    "Participant",
    "QuestionStub",
    "SourceDocument",
    "SubclaimNode",
]
