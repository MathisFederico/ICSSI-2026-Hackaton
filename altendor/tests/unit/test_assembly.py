"""Unit tests for the assembly stage (S16/S17).

These tests exercise the full integration boundary:
:func:`altendor.assemble.builder.build_intermediate` plus
:func:`altendor.assemble.deltabay_writer.write_debate_json`. They are pure
input -> output tests (no module-export or signature checks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from altendor.assemble.builder import build_intermediate
from altendor.assemble.deltabay_writer import write_debate_json
from altendor.assemble.intermediate import IntermediateDebate
from altendor.classify.schema import ClassifyResult, Endorsement, Flag, Irrelevant
from altendor.cluster.claims import ClaimCluster
from altendor.enrich.text_resolver import Platform, ResolvedPost, TextConfidence
from altendor.route.question_router import THREE_QUESTIONS

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_resolved_post(
    post_id: str,
    *,
    platform: Platform = "bluesky",
    text: str = "post text",
    author_handle: str | None = "alice.bsky",
    author_id: str | None = "did:plc:alice",
    url: str = "https://bsky.app/profile/alice.bsky/post/abc",
    created_at: str = "2026-06-20T12:00:00+00:00",
    raw_title: str = "raw title",
    text_confidence: TextConfidence = "high",
) -> ResolvedPost:
    """Build a :class:`ResolvedPost` with sensible defaults for tests."""
    return ResolvedPost(
        post_id=post_id,
        platform=platform,
        text=text,
        author_handle=author_handle,
        author_id=author_id,
        url=url,
        created_at=created_at,
        raw_title=raw_title,
        text_confidence=text_confidence,
    )


def make_endorsement(
    *,
    claim_text: str = "The method improves recall.",
    magnitude_dB: int = 12,
    criterion: Literal["Support", "Prior"] = "Support",
    reasoning: str = "The post explicitly endorses the claim.",
) -> Endorsement:
    """Build an :class:`Endorsement` with sensible defaults."""
    return Endorsement(
        claim_text=claim_text,
        magnitude_dB=magnitude_dB,
        criterion=criterion,
        reasoning=reasoning,
    )


def make_flag(
    *,
    category: Literal["methodological", "source", "data", "bias", "other"] = "methodological",
    rationale: str = "The post raises a sample-size concern.",
) -> Flag:
    """Build a :class:`Flag` with sensible defaults."""
    return Flag(category=category, rationale=rationale)


def make_irrelevant(*, reason: str = "Vague praise.") -> Irrelevant:
    """Build an :class:`Irrelevant` with sensible defaults."""
    return Irrelevant(reason=reason)


def make_papers_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a papers DataFrame matching ``top_papers`` + abstract-enrichment schema."""
    defaults = {
        "doi": "10.1234/example",
        "ro_id": "alt-1",
        "title": "An Example Paper",
        "altmetric_score": 100.0,
        "last_mentioned_at": pd.Timestamp("2026-06-20T12:00:00", tz="UTC"),
        "abstract": "Example abstract.",
    }
    filled = [{**defaults, **row} for row in rows]
    return pd.DataFrame(filled)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_round_trip_through_writer(tmp_path: Path) -> None:
    """One paper, two posts in a single cluster, route to peer-review.

    Build -> write -> read -> assert equality with the built model.
    """
    papers = make_papers_df([{"doi": "10.1234/peer", "ro_id": "alt-99", "title": "Peer review paper"}])
    posts = {
        "p1": make_resolved_post("p1", text="endorses claim 1"),
        "p2": make_resolved_post("p2", author_handle="bob.bsky", author_id="did:plc:bob", text="also endorses"),
    }
    classified: dict[str, ClassifyResult] = {
        "p1": make_endorsement(magnitude_dB=10),
        "p2": make_endorsement(magnitude_dB=5),
    }
    clusters = {
        "10.1234/peer": [ClaimCluster(canonical_text="Recall is higher", member_post_ids=["p1", "p2"])],
    }
    routes = {"10.1234/peer": "question:peer-review"}

    built = build_intermediate(
        papers=papers,
        resolved_posts=posts,
        classified=classified,
        clusters=clusters,
        routes=routes,
        run_id="2026-06-28-001",
    )

    out_path = tmp_path / "debate.json"
    write_debate_json(built, out_path)

    loaded = IntermediateDebate.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert loaded == built


def test_endorsement_attaches_to_correct_subclaim() -> None:
    """Three posts split across two clusters; assert per-cluster endorsement membership."""
    papers = make_papers_df([{"doi": "10.1234/x", "ro_id": "alt-1"}])
    posts = {
        "p1": make_resolved_post("p1"),
        "p2": make_resolved_post("p2", author_id="did:plc:bob"),
        "p3": make_resolved_post("p3", author_id="did:plc:carol"),
    }
    classified: dict[str, ClassifyResult] = {
        "p1": make_endorsement(magnitude_dB=8),
        "p2": make_endorsement(magnitude_dB=6),
        "p3": make_endorsement(magnitude_dB=4),
    }
    clusters = {
        "10.1234/x": [
            ClaimCluster(canonical_text="cluster A canonical", member_post_ids=["p1", "p2"]),
            ClaimCluster(canonical_text="cluster B canonical", member_post_ids=["p3"]),
        ],
    }
    routes = {"10.1234/x": "question:peer-review"}

    idb = build_intermediate(papers, posts, classified, clusters, routes)

    assert len(idb.papers) == 1
    subclaims = idb.papers[0].answer.subclaims
    assert len(subclaims) == 2
    a_endorsement_ids = {e.id for e in subclaims[0].endorsements}
    b_endorsement_ids = {e.id for e in subclaims[1].endorsements}
    assert a_endorsement_ids == {"end:altendor:bluesky:p1", "end:altendor:bluesky:p2"}
    assert b_endorsement_ids == {"end:altendor:bluesky:p3"}


def test_flag_attaches_to_evidence_not_subclaim() -> None:
    """A flagged post in a cluster routes to EvidenceNode.flags, not the subclaim."""
    papers = make_papers_df([{"doi": "10.1234/y", "ro_id": "alt-2"}])
    posts = {
        "p1": make_resolved_post("p1"),
        "p2": make_resolved_post("p2", author_id="did:plc:bob"),
    }
    classified: dict[str, ClassifyResult] = {
        "p1": make_endorsement(magnitude_dB=9),
        "p2": make_flag(),
    }
    clusters = {
        "10.1234/y": [ClaimCluster(canonical_text="some claim", member_post_ids=["p1", "p2"])],
    }
    routes = {"10.1234/y": "question:peer-review"}

    idb = build_intermediate(papers, posts, classified, clusters, routes)

    paper = idb.papers[0]
    # The endorsement should be on the subclaim.
    assert len(paper.answer.subclaims) == 1
    assert len(paper.answer.subclaims[0].endorsements) == 1
    assert paper.answer.subclaims[0].endorsements[0].id == "end:altendor:bluesky:p1"
    # The subclaim must NOT carry flags.
    assert paper.answer.subclaims[0].flags == []
    # The flag must be on the EvidenceNode.
    assert len(paper.answer.evidence.flags) == 1
    assert paper.answer.evidence.flags[0].id == "flag:altendor:bluesky:p2"


def test_irrelevant_post_dropped() -> None:
    """Irrelevant classifications are silently dropped."""
    papers = make_papers_df([{"doi": "10.1234/z", "ro_id": "alt-3"}])
    posts = {
        "p1": make_resolved_post("p1"),
        "p2": make_resolved_post("p2", author_id="did:plc:bob"),
    }
    classified: dict[str, ClassifyResult] = {
        "p1": make_endorsement(magnitude_dB=7),
        "p2": make_irrelevant(),
    }
    clusters = {
        "10.1234/z": [ClaimCluster(canonical_text="claim", member_post_ids=["p1", "p2"])],
    }
    routes = {"10.1234/z": "question:peer-review"}

    idb = build_intermediate(papers, posts, classified, clusters, routes)

    paper = idb.papers[0]
    # Only the endorsement survives.
    total_endorsements = sum(len(sc.endorsements) for sc in paper.answer.subclaims) + len(
        paper.answer.evidence.endorsements
    )
    assert total_endorsements == 1
    assert paper.answer.evidence.flags == []


def test_participants_deduplicated_across_papers() -> None:
    """Two papers, three posts, two sharing the same (platform, author_id)."""
    papers = make_papers_df(
        [
            {"doi": "10.1234/a", "ro_id": "alt-A"},
            {"doi": "10.1234/b", "ro_id": "alt-B"},
        ]
    )
    posts = {
        "p1": make_resolved_post("p1", author_handle="alice.bsky", author_id="did:plc:alice"),
        "p2": make_resolved_post("p2", author_handle="alice.bsky", author_id="did:plc:alice"),
        "p3": make_resolved_post("p3", author_handle="bob.bsky", author_id="did:plc:bob"),
    }
    classified: dict[str, ClassifyResult] = {
        "p1": make_endorsement(magnitude_dB=8),
        "p2": make_endorsement(magnitude_dB=8),
        "p3": make_endorsement(magnitude_dB=8),
    }
    clusters = {
        "10.1234/a": [ClaimCluster(canonical_text="claim A", member_post_ids=["p1", "p3"])],
        "10.1234/b": [ClaimCluster(canonical_text="claim B", member_post_ids=["p2"])],
    }
    routes = {
        "10.1234/a": "question:peer-review",
        "10.1234/b": "question:research-integrity",
    }

    idb = build_intermediate(papers, posts, classified, clusters, routes)

    assert len(idb.participants) == 2
    ids = {p.id for p in idb.participants}
    assert ids == {"agent:bluesky:did:plc:alice", "agent:bluesky:did:plc:bob"}


def test_paper_without_route_falls_back_to_first_question() -> None:
    """An unrouted paper picks ``THREE_QUESTIONS[0].id`` as its routed question."""
    papers = make_papers_df([{"doi": "10.1234/u", "ro_id": "alt-U"}])
    posts: dict[str, ResolvedPost] = {}
    classified: dict[str, ClassifyResult] = {}
    clusters: dict[str, list[ClaimCluster]] = {}
    routes: dict[str, str] = {}

    idb = build_intermediate(papers, posts, classified, clusters, routes)

    assert idb.papers[0].routed_question_id == THREE_QUESTIONS[0].id


def test_zero_magnitude_endorsement_dropped() -> None:
    """An endorsement with magnitude_dB=0 is dropped (defence in depth)."""
    papers = make_papers_df([{"doi": "10.1234/zero", "ro_id": "alt-Z"}])
    posts = {"p1": make_resolved_post("p1")}
    # Bypass the classifier's ZeroMagnitudeError by constructing the model
    # directly via model_construct (no validation), so we exercise the
    # builder's defence-in-depth filter.
    bad = Endorsement.model_construct(
        kind="endorsement",
        claim_text="x",
        magnitude_dB=0,
        criterion="Support",
        reasoning="y",
    )
    classified: dict[str, ClassifyResult] = {"p1": bad}
    clusters = {"10.1234/zero": [ClaimCluster(canonical_text="claim", member_post_ids=["p1"])]}
    routes = {"10.1234/zero": "question:peer-review"}

    idb = build_intermediate(papers, posts, classified, clusters, routes)

    paper = idb.papers[0]
    total_endorsements = sum(len(sc.endorsements) for sc in paper.answer.subclaims) + len(
        paper.answer.evidence.endorsements
    )
    assert total_endorsements == 0


def test_writer_creates_parent_directory(tmp_path: Path) -> None:
    """write_debate_json mkdirs missing parents."""
    papers = make_papers_df([{"doi": "10.1234/w", "ro_id": "alt-W"}])
    idb = build_intermediate(papers, {}, {}, {}, {})

    out_path = tmp_path / "nested" / "subdir" / "debate.json"
    assert not out_path.parent.exists()

    write_debate_json(idb, out_path)

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").endswith("\n")


def test_writer_output_round_trips_via_model_validate_json(tmp_path: Path) -> None:
    """Write -> read -> re-validate yields an equal model instance."""
    papers = make_papers_df(
        [
            {"doi": "10.1234/r1", "ro_id": "alt-R1", "title": "Round trip paper 1"},
            {"doi": "10.1234/r2", "ro_id": "alt-R2", "title": "Round trip paper 2"},
        ]
    )
    posts = {
        "p1": make_resolved_post("p1"),
        "p2": make_resolved_post("p2", author_id="did:plc:bob"),
    }
    classified: dict[str, ClassifyResult] = {
        "p1": make_endorsement(magnitude_dB=11),
        "p2": make_flag(),
    }
    clusters = {
        "10.1234/r1": [ClaimCluster(canonical_text="claim1", member_post_ids=["p1", "p2"])],
        "10.1234/r2": [],
    }
    routes = {
        "10.1234/r1": "question:peer-review",
        "10.1234/r2": "question:measure-progress",
    }
    built = build_intermediate(papers, posts, classified, clusters, routes)

    out_path = tmp_path / "round_trip.json"
    write_debate_json(built, out_path)

    text = out_path.read_text(encoding="utf-8")
    reloaded = IntermediateDebate.model_validate_json(text)

    assert reloaded == built


def test_paper_with_missing_doi_skipped() -> None:
    """A row whose ``doi`` is None is excluded from the output."""
    papers = make_papers_df(
        [
            {"doi": None, "ro_id": "alt-missing", "title": "Missing DOI"},
            {"doi": "10.1234/ok", "ro_id": "alt-ok", "title": "Has DOI"},
        ]
    )
    idb = build_intermediate(papers, {}, {}, {}, {})

    assert len(idb.papers) == 1
    assert idb.papers[0].doi == "10.1234/ok"
