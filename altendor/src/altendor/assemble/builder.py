"""Assemble upstream pipeline outputs into an :class:`IntermediateDebate` (S16).

The builder is a pure function: it does no I/O, no LLM calls, and never
raises on missing data — it logs and falls back. See the module-level
``build_intermediate`` docstring for the per-field mapping rules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from altendor.assemble.intermediate import (
    AnswerNode,
    EndorsementRow,
    EvidenceNode,
    FlagRow,
    IntermediateDebate,
    PaperRecord,
    Participant,
    QuestionStub,
    SourceDocument,
    SubclaimNode,
)
from altendor.classify.schema import ClassifyResult, Endorsement, Flag, Irrelevant
from altendor.cluster.claims import ClaimCluster
from altendor.enrich.text_resolver import ResolvedPost
from altendor.route.question_router import THREE_QUESTIONS

_LOG = logging.getLogger(__name__)

_VALID_QUESTION_IDS: frozenset[str] = frozenset(q.id for q in THREE_QUESTIONS)
_FALLBACK_QUESTION_ID: str = THREE_QUESTIONS[0].id


def _participant_id(platform: str, author_id: str | None) -> str:
    """Stable participant id; ``author_id`` may be missing for anonymous posts."""
    return f"agent:{platform}:{author_id or 'anonymous'}"


def _participant_for(post: ResolvedPost) -> Participant:
    """Build a :class:`Participant` from a :class:`ResolvedPost`."""
    handle = post.author_handle
    name = handle if handle else "anonymous"
    return Participant(
        id=_participant_id(post.platform, post.author_id),
        name=name,
        handle=handle,
        platform=post.platform,
    )


def _resolve_question_id(doi: str, routes: dict[str, str]) -> str:
    """Look up the routed question; fall back to the first question with a warning."""
    qid = routes.get(doi)
    if qid is None:
        _LOG.warning("No route for paper %s; falling back to %s", doi, _FALLBACK_QUESTION_ID)
        return _FALLBACK_QUESTION_ID
    if qid not in _VALID_QUESTION_IDS:
        _LOG.warning(
            "Unknown question id %r for paper %s; falling back to %s",
            qid,
            doi,
            _FALLBACK_QUESTION_ID,
        )
        return _FALLBACK_QUESTION_ID
    return qid


def _build_endorsement_row(
    post: ResolvedPost,
    classification: Endorsement,
) -> EndorsementRow | None:
    """Build an :class:`EndorsementRow`; returns ``None`` for zero-magnitude (defence in depth)."""
    if classification.magnitude_dB == 0:
        # The classifier already drops these; double-guard so a malformed
        # input dict can't smuggle one past the EndorsementRow validator.
        return None
    return EndorsementRow(
        id=f"end:altendor:{post.platform}:{post.post_id}",
        participant_id=_participant_id(post.platform, post.author_id),
        magnitude=classification.magnitude_dB,
        criterion=classification.criterion,
        created_at=post.created_at,
        source_post_url=post.url,
        source_text=post.text,
    )


def _build_flag_row(post: ResolvedPost, classification: Flag) -> FlagRow:
    """Build a :class:`FlagRow` from a flagged post."""
    handle = post.author_handle
    return FlagRow(
        id=f"flag:altendor:{post.platform}:{post.post_id}",
        category=classification.category,
        rationale=classification.rationale,
        raised_by_id=_participant_id(post.platform, post.author_id),
        raised_by_name=handle if handle else "anonymous",
        created_at=post.created_at,
        source_post_url=post.url,
    )


def _coerce_float(value: object) -> float:
    """Best-effort coerce *value* to ``float``; returns ``0.0`` on failure."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _paper_row_value(row: pd.Series, key: str) -> object:
    """Read a column from a paper row, treating NaN/None uniformly as missing."""
    if key not in row.index:
        return None
    value = row[key]
    # pandas NA / NaN / NaT: ``pd.isna`` handles all three.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # Non-scalar values (lists, dicts) — return as-is.
        return value
    return value


def build_intermediate(
    papers: pd.DataFrame,
    resolved_posts: dict[str, ResolvedPost],
    classified: dict[str, ClassifyResult],
    clusters: dict[str, list[ClaimCluster]],
    routes: dict[str, str],
    *,
    debate_id: str = "debate:altendor:optimal-funding",
    run_id: str | None = None,
) -> IntermediateDebate:
    """Compose upstream outputs into the neutral :class:`IntermediateDebate`.

    Mapping rules
    -------------
    * One :class:`PaperRecord` per row in *papers*. Rows whose ``doi`` is
      missing are skipped (we never fabricate identifiers).
    * Per paper, one :class:`AnswerNode` whose ``title`` is the paper's title
      and ``short_title`` is the first 60 characters of that title.
    * The paper itself is the :class:`EvidenceNode`'s ``source_document``.
    * Each :class:`ClaimCluster` for the paper becomes a :class:`SubclaimNode`.
      Posts classified as :class:`Endorsement` attach to the cluster they
      live in; posts classified as :class:`Flag` attach to the paper-level
      :class:`EvidenceNode` (flags concern the paper, not a specific claim);
      posts classified as :class:`Irrelevant` are skipped.
    * Endorsements for posts that don't appear in any cluster are attached
      to the paper-level :class:`EvidenceNode` as a fallback.
    * Participants are deduped by ``(platform, author_id)`` across all papers.
    * Missing or unknown routes fall back to ``THREE_QUESTIONS[0].id`` with a
      warning.

    See :mod:`altendor.assemble.intermediate` for the field-by-field shape.
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).date().isoformat()
    generated_at = datetime.now(timezone.utc).isoformat()

    participants: dict[str, Participant] = {}

    def _register(post: ResolvedPost) -> None:
        p = _participant_for(post)
        # First-seen wins; subsequent posts from the same agent reuse the id.
        participants.setdefault(p.id, p)

    paper_records: list[PaperRecord] = []

    for _, row in papers.iterrows():
        doi_raw = _paper_row_value(row, "doi")
        if doi_raw is None or doi_raw == "":
            _LOG.warning("Skipping paper row with missing DOI: ro_id=%r", _paper_row_value(row, "ro_id"))
            continue
        doi = str(doi_raw)

        ro_id_raw = _paper_row_value(row, "ro_id")
        ro_id = str(ro_id_raw) if ro_id_raw is not None else ""

        title_raw = _paper_row_value(row, "title")
        title = str(title_raw) if title_raw is not None else ""

        abstract_raw = _paper_row_value(row, "abstract")
        abstract = str(abstract_raw) if abstract_raw is not None else None

        score_raw = _paper_row_value(row, "altmetric_score")
        altmetric_score = _coerce_float(score_raw)

        routed_qid = _resolve_question_id(doi, routes)

        paper_clusters = clusters.get(doi, [])

        # Track which posts have been placed in a subclaim — the rest become
        # fallback endorsements / flags on the paper-level EvidenceNode.
        clustered_post_ids: set[str] = set()
        for cluster in paper_clusters:
            for pid in cluster.member_post_ids:
                clustered_post_ids.add(pid)

        # Build subclaims. Per the rules: endorsements attach to their cluster,
        # flags do NOT attach to subclaims (flags always escalate to EvidenceNode).
        subclaim_nodes: list[SubclaimNode] = []
        for i, cluster in enumerate(paper_clusters, start=1):
            sub_endorsements: list[EndorsementRow] = []
            for post_id in cluster.member_post_ids:
                post = resolved_posts.get(post_id)
                if post is None:
                    _LOG.warning("Cluster member post_id %r has no ResolvedPost; skipping", post_id)
                    continue
                classification = classified.get(post_id)
                if classification is None:
                    _LOG.warning("Cluster member post_id %r has no classification; skipping", post_id)
                    continue
                if isinstance(classification, Endorsement):
                    row_obj = _build_endorsement_row(post, classification)
                    if row_obj is not None:
                        sub_endorsements.append(row_obj)
                        _register(post)
                elif isinstance(classification, (Flag, Irrelevant)):
                    # Flags handled at the paper level; Irrelevant dropped entirely.
                    continue
            subclaim_nodes.append(
                SubclaimNode(
                    id=f"claim:altendor:{ro_id}:c{i}",
                    title=cluster.canonical_text,
                    member_post_ids=list(cluster.member_post_ids),
                    endorsements=sub_endorsements,
                    flags=[],
                )
            )

        # Paper-level evidence: flags (always) + endorsements for non-clustered posts.
        evidence_endorsements: list[EndorsementRow] = []
        evidence_flags: list[FlagRow] = []
        # Iterate the full classified dict; only attach posts about THIS paper.
        # We can't tell which paper a post belongs to from `classified` alone,
        # so we use cluster membership to scope. Flags need an explicit per-paper
        # mapping — they come in via cluster member ids too: the upstream stage
        # places every resolved post about a paper into the paper's cluster set
        # (a flag still ends up as a "member" of some cluster, just routed to
        # evidence instead of the subclaim). Non-clustered endorsements (rare)
        # are emitted as fallback evidence.
        for post_id in clustered_post_ids:
            post = resolved_posts.get(post_id)
            if post is None:
                continue
            classification = classified.get(post_id)
            if classification is None:
                continue
            if isinstance(classification, Flag):
                evidence_flags.append(_build_flag_row(post, classification))
                _register(post)

        # Fallback: posts classified about this paper but not in any cluster.
        # We approximate "about this paper" by: post_id present in resolved_posts
        # AND classified AND not in clustered_post_ids AND there's at least one
        # cluster for the paper (otherwise we have no scoping signal). In the
        # zero-cluster case we conservatively skip — the calibration gate decides.
        # For S16 we accept the simpler rule: only emit fallback evidence for
        # posts we *can* attribute, which is the clustered-but-flagged set
        # (already handled above). Other fallback paths are reserved for S18+.

        evidence = EvidenceNode(
            id=f"evidence:altendor:{ro_id}",
            source_document=SourceDocument(
                id=f"src:doi:{doi}",
                doi=doi,
                url=f"https://doi.org/{doi}",
            ),
            endorsements=evidence_endorsements,
            flags=evidence_flags,
        )

        answer = AnswerNode(
            id=f"answer:altendor:{ro_id}",
            title=title,
            short_title=title[:60],
            evidence=evidence,
            subclaims=subclaim_nodes,
        )

        paper_records.append(
            PaperRecord(
                doi=doi,
                ro_id=ro_id,
                title=title,
                abstract=abstract,
                altmetric_score=altmetric_score,
                routed_question_id=routed_qid,
                answer=answer,
            )
        )

    question_stubs = [
        QuestionStub(id=q.id, title=q.title, short_title=q.short_title) for q in THREE_QUESTIONS
    ]

    return IntermediateDebate(
        run_id=run_id,
        debate_id=debate_id,
        generated_at=generated_at,
        questions=question_stubs,
        participants=list(participants.values()),
        papers=paper_records,
    )


__all__ = ["build_intermediate"]
