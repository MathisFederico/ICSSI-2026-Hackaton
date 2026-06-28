"""Unit tests for :mod:`altendor.traverse.replies` (S13).

These tests run fully offline: the bluesky/reddit fetchers and the Anthropic
classifier are all stubbed via ``monkeypatch``. No network, no Anthropic key.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import anthropic
import pytest
from altendor.classify import classifier as classifier_module
from altendor.classify.classifier import PaperCtx
from altendor.classify.schema import (
    ClassifyResult,
    Endorsement,
    Flag,
    Irrelevant,
)
from altendor.enrich import bluesky as bluesky_module
from altendor.enrich import reddit as reddit_module
from altendor.enrich.bluesky import BskyPost
from altendor.enrich.reddit import RedditNode
from altendor.enrich.text_resolver import ResolvedPost
from altendor.traverse import replies as traverse_module
from altendor.traverse.replies import (
    TraversalRow,
    TraversalSeed,
    traverse_depth1,
)

# ---------------------------------------------------------------------------
# Builders for ResolvedPost / classification results / seeds
# ---------------------------------------------------------------------------


def _bsky_resolved(
    *,
    post_id: str = "at://did:plc:seed/app.bsky.feed.post/seedrkey",
    author_handle: str = "seed.bsky.social",
    author_did: str = "did:plc:seed",
    text: str = "Seed bluesky post.",
) -> ResolvedPost:
    return ResolvedPost(
        post_id=post_id,
        platform="bluesky",
        text=text,
        author_handle=author_handle,
        author_id=author_did,
        url=f"https://bsky.app/profile/{author_handle}/post/{post_id.rsplit('/', 1)[-1]}",
        created_at="2026-06-01T12:00:00+00:00",
        raw_title=text,
        text_confidence="high",
    )


def _reddit_resolved(
    *,
    post_id: str = "t3_seed",
    author: str = "seed_user",
    text: str = "Seed reddit post.",
    url: str = "https://www.reddit.com/r/Sci/comments/seed/title/",
) -> ResolvedPost:
    return ResolvedPost(
        post_id=post_id,
        platform="reddit",
        text=text,
        author_handle=author,
        author_id=author,
        url=url,
        created_at="2026-06-01T12:00:00+00:00",
        raw_title=text,
        text_confidence="high",
    )


def _twitter_resolved() -> ResolvedPost:
    return ResolvedPost(
        post_id="tw-1",
        platform="twitter",
        text="A tweet.",
        author_handle="tw_user",
        author_id="tw-user-id",
        url="https://twitter.com/tw_user/status/1",
        created_at="2026-06-01T12:00:00+00:00",
        raw_title="A tweet.",
        text_confidence="low",
    )


def _endorsement() -> Endorsement:
    return Endorsement(
        claim_text="X causes Y.",
        magnitude_dB=20,
        criterion="Support",
        reasoning="Post explicitly endorses the mechanistic claim.",
    )


def _flag() -> Flag:
    return Flag(category="methodological", rationale="Sample size N=10 too low.")


def _irrelevant() -> Irrelevant:
    return Irrelevant(reason="Vague praise.")


def _paper() -> PaperCtx:
    return PaperCtx(title="X causes Y", abstract="we show X causes Y", url=None)


def _seed(post: ResolvedPost, result: ClassifyResult) -> TraversalSeed:
    return TraversalSeed(post=post, paper=_paper(), result=result)


def _as_client(obj: object) -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, obj)


# ---------------------------------------------------------------------------
# Bluesky/Reddit reply builders
# ---------------------------------------------------------------------------


def _bsky_post(
    *,
    rkey: str,
    did: str = "did:plc:replier",
    handle: str = "replier.bsky.social",
    text: str = "reply text",
) -> BskyPost:
    return BskyPost(
        at_uri=f"at://{did}/app.bsky.feed.post/{rkey}",
        cid=f"cid-{rkey}",
        text=text,
        author_did=did,
        author_handle=handle,
        created_at="2026-06-02T12:00:00.000Z",
    )


def _reddit_node(
    *,
    cid: str,
    author: str = "replier_user",
    body: str = "reply body",
    permalink_suffix: str = "reply",
) -> RedditNode:
    return RedditNode(
        id=f"t1_{cid}",
        kind="comment",
        author=author,
        body=body,
        created_utc=1750000000.0,
        permalink=f"https://www.reddit.com/r/Sci/comments/seed/title/{permalink_suffix}/",
    )


# ---------------------------------------------------------------------------
# Stub plumbing
# ---------------------------------------------------------------------------


def _patch_classifier(monkeypatch: pytest.MonkeyPatch, result_for: dict[str, ClassifyResult] | None = None) -> list[ResolvedPost]:
    """Replace ``classify_post`` so it returns a canned ``Irrelevant`` (or per-text overrides).

    Records every ``post`` it was called with into the returned list.
    """
    record: list[ResolvedPost] = []
    overrides = result_for or {}

    def fake_classify(
        client: object,
        post: ResolvedPost,
        paper: PaperCtx,
        *,
        model: str = "ignored",
    ) -> ClassifyResult:
        record.append(post)
        if post.text in overrides:
            return overrides[post.text]
        return Irrelevant(reason=f"stub-classified {post.post_id}")

    # The traverse module imports ``classify_post`` directly, so we patch it
    # on the traverse module (where the name is bound), not just on
    # ``altendor.classify.classifier``.
    monkeypatch.setattr(traverse_module, "classify_post", fake_classify)
    monkeypatch.setattr(classifier_module, "classify_post", fake_classify)
    return record


def _patch_bsky_thread(
    monkeypatch: pytest.MonkeyPatch,
    *,
    by_uri: dict[str, list[BskyPost]],
) -> list[str]:
    """Replace ``bluesky.get_thread`` to return canned posts per AT URI.

    The first entry of each list is treated as the root (mirrors the real
    function). Records every URI it was asked about into the returned list.
    """
    seen: list[str] = []

    async def fake_get_thread(
        at_uri: str,
        *,
        depth: int = 1,
        parent_height: int = 0,
        session: object = None,
    ) -> list[BskyPost]:
        seen.append(at_uri)
        return list(by_uri.get(at_uri, []))

    monkeypatch.setattr(bluesky_module, "get_thread", fake_get_thread)
    monkeypatch.setattr(traverse_module.bluesky, "get_thread", fake_get_thread)
    return seen


def _patch_reddit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nodes_by_url: dict[str, RedditNode | None],
    replies_by_id: dict[str, list[RedditNode]],
) -> dict[str, list[str]]:
    """Replace ``reddit.resolve_node`` + ``reddit.get_replies`` with stubs.

    Records resolve-node URLs and get-replies parent ids for assertions.
    """
    seen: dict[str, list[str]] = {"resolve": [], "replies": []}

    def fake_resolve_node(url: str, *, client: object = None) -> RedditNode | None:
        seen["resolve"].append(url)
        return nodes_by_url.get(url)

    def fake_get_replies(
        node: RedditNode,
        *,
        depth: int = 1,
        max_per_parent: int = 20,
        client: object = None,
    ) -> list[RedditNode]:
        seen["replies"].append(node.id)
        out = replies_by_id.get(node.id, [])
        return list(out[:max_per_parent])

    monkeypatch.setattr(reddit_module, "resolve_node", fake_resolve_node)
    monkeypatch.setattr(reddit_module, "get_replies", fake_get_replies)
    monkeypatch.setattr(traverse_module.reddit, "resolve_node", fake_resolve_node)
    monkeypatch.setattr(traverse_module.reddit, "get_replies", fake_get_replies)
    return seen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_traverse_only_processes_endorsements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-endorsement seeds are silently dropped before any fetch."""
    endorse_seed = _seed(_bsky_resolved(post_id="at://did:plc:e/app.bsky.feed.post/r1"), _endorsement())
    flag_seed = _seed(_bsky_resolved(post_id="at://did:plc:f/app.bsky.feed.post/r2"), _flag())
    irrelevant_seed = _seed(_bsky_resolved(post_id="at://did:plc:i/app.bsky.feed.post/r3"), _irrelevant())

    fetched_uris = _patch_bsky_thread(
        monkeypatch,
        by_uri={
            endorse_seed.post.post_id: [
                _bsky_post(rkey="r1"),  # root
                _bsky_post(rkey="child1", did="did:plc:other"),
            ],
        },
    )
    classified = _patch_classifier(monkeypatch)

    rows = asyncio.run(
        traverse_depth1(
            _as_client(object()),
            [endorse_seed, flag_seed, irrelevant_seed],
        ),
    )

    # Only the endorsement seed's URI is fetched.
    assert fetched_uris == [endorse_seed.post.post_id]
    # And only its one non-root child was classified.
    assert len(classified) == 1
    assert len(rows) == 1
    assert rows[0].parent_post_id == endorse_seed.post.post_id


def test_traverse_only_processes_bluesky_and_reddit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endorsements on twitter (or any non-{bluesky,reddit} platform) are skipped."""
    twitter_seed = _seed(_twitter_resolved(), _endorsement())

    fetched_uris = _patch_bsky_thread(monkeypatch, by_uri={})
    reddit_calls = _patch_reddit(monkeypatch, nodes_by_url={}, replies_by_id={})
    classified = _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), [twitter_seed]))

    assert rows == []
    assert fetched_uris == []
    assert reddit_calls == {"resolve": [], "replies": []}
    assert classified == []


def test_traverse_drops_self_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply whose author matches the seed's author is excluded from output."""
    seed = _seed(_bsky_resolved(post_id="at://did:plc:seed/app.bsky.feed.post/r1"), _endorsement())

    self_reply = _bsky_post(rkey="self", did="did:plc:seed", handle="seed.bsky.social", text="self reply")
    other_reply = _bsky_post(rkey="other", did="did:plc:other", handle="other.bsky.social", text="other reply")

    _patch_bsky_thread(
        monkeypatch,
        by_uri={seed.post.post_id: [_bsky_post(rkey="r1"), self_reply, other_reply]},
    )
    _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), [seed]))

    assert len(rows) == 1
    assert rows[0].reply.text == "other reply"
    assert rows[0].reply.author_id == "did:plc:other"


def test_traverse_dedups_across_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two seeds whose replies share a (platform, post_id) only produce one row."""
    seed_a = _seed(_bsky_resolved(post_id="at://did:plc:a/app.bsky.feed.post/a"), _endorsement())
    seed_b = _seed(_bsky_resolved(post_id="at://did:plc:b/app.bsky.feed.post/b"), _endorsement())

    shared = _bsky_post(rkey="shared", did="did:plc:other", text="shared reply")

    _patch_bsky_thread(
        monkeypatch,
        by_uri={
            seed_a.post.post_id: [_bsky_post(rkey="a"), shared],
            seed_b.post.post_id: [_bsky_post(rkey="b"), shared],
        },
    )
    classified = _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), [seed_a, seed_b]))

    # The shared reply is classified once and surfaces under the first seed.
    assert len(rows) == 1
    assert rows[0].parent_post_id == seed_a.post.post_id
    assert rows[0].reply.post_id == shared.at_uri
    assert len(classified) == 1


def test_traverse_respects_max_per_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed with 30 replies is capped at ``max_per_parent``."""
    seed = _seed(_bsky_resolved(post_id="at://did:plc:s/app.bsky.feed.post/s"), _endorsement())

    children = [_bsky_post(rkey=f"c{i}", did=f"did:plc:r{i}") for i in range(30)]
    _patch_bsky_thread(
        monkeypatch,
        by_uri={seed.post.post_id: [_bsky_post(rkey="s")] + children},
    )
    _patch_classifier(monkeypatch)

    rows = asyncio.run(
        traverse_depth1(_as_client(object()), [seed], max_per_parent=5),
    )

    assert len(rows) == 5
    assert [r.reply.post_id for r in rows] == [c.at_uri for c in children[:5]]


def test_traverse_respects_max_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global cap ``max_total`` truncates the run after that many replies."""
    # Five seeds, each with 10 unique children — 50 in total.
    seeds: list[TraversalSeed] = []
    by_uri: dict[str, list[BskyPost]] = {}
    for s in range(5):
        seed_uri = f"at://did:plc:s{s}/app.bsky.feed.post/s{s}"
        seeds.append(_seed(_bsky_resolved(post_id=seed_uri), _endorsement()))
        children = [_bsky_post(rkey=f"s{s}_c{i}", did=f"did:plc:r{s}_{i}") for i in range(10)]
        by_uri[seed_uri] = [_bsky_post(rkey=f"root{s}")] + children

    _patch_bsky_thread(monkeypatch, by_uri=by_uri)
    classified = _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), seeds, max_total=20))

    assert len(rows) == 20
    assert len(classified) == 20


def test_traverse_returns_traversal_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each output row carries parent_post_id, a ResolvedPost reply, and the stubbed result."""
    seed = _seed(_bsky_resolved(post_id="at://did:plc:s/app.bsky.feed.post/s"), _endorsement())
    child = _bsky_post(rkey="c1", did="did:plc:r", handle="r.bsky.social", text="meaningful reply")

    _patch_bsky_thread(monkeypatch, by_uri={seed.post.post_id: [_bsky_post(rkey="s"), child]})
    canned = Endorsement(
        claim_text="X causes Y.",
        magnitude_dB=10,
        criterion="Support",
        reasoning="Reply also endorses the claim.",
    )
    _patch_classifier(monkeypatch, result_for={"meaningful reply": canned})

    rows = asyncio.run(traverse_depth1(_as_client(object()), [seed]))

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, TraversalRow)
    assert row.parent_post_id == seed.post.post_id
    assert isinstance(row.reply, ResolvedPost)
    assert row.reply.platform == "bluesky"
    assert row.reply.text == "meaningful reply"
    assert row.reply.post_id == child.at_uri
    assert row.reply.url == f"https://bsky.app/profile/{child.author_handle}/post/c1"
    assert row.result is canned


def test_traverse_handles_resolve_failure_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reddit seed whose resolve_node returns ``None`` contributes 0 rows; other seeds still run."""
    # Seed 1: reddit, will fail to resolve.
    reddit_seed = _seed(
        _reddit_resolved(post_id="t3_dead", url="https://www.reddit.com/r/Sci/comments/dead/x/"),
        _endorsement(),
    )
    # Seed 2: bluesky, will succeed.
    bsky_seed = _seed(_bsky_resolved(post_id="at://did:plc:s/app.bsky.feed.post/s"), _endorsement())

    _patch_reddit(
        monkeypatch,
        nodes_by_url={reddit_seed.post.url: None},
        replies_by_id={},
    )
    _patch_bsky_thread(
        monkeypatch,
        by_uri={
            bsky_seed.post.post_id: [
                _bsky_post(rkey="s"),
                _bsky_post(rkey="c1", did="did:plc:other", text="bsky reply"),
            ],
        },
    )
    _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), [reddit_seed, bsky_seed]))

    # Only the bluesky seed contributes.
    assert len(rows) == 1
    assert rows[0].parent_post_id == bsky_seed.post.post_id
    assert rows[0].reply.text == "bsky reply"


def test_traverse_bsky_reply_root_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_thread`` returns [root, child]; only the child appears in output."""
    seed = _seed(_bsky_resolved(post_id="at://did:plc:s/app.bsky.feed.post/s"), _endorsement())

    root_post = _bsky_post(rkey="s", did="did:plc:s", handle="seed.bsky.social", text="seed root text")
    child_post = _bsky_post(rkey="c1", did="did:plc:other", handle="other.bsky.social", text="child text")

    _patch_bsky_thread(monkeypatch, by_uri={seed.post.post_id: [root_post, child_post]})
    _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), [seed]))

    assert len(rows) == 1
    assert rows[0].reply.post_id == child_post.at_uri
    assert rows[0].reply.text == "child text"
    # Make sure no row carries the seed/root post_id.
    assert all(r.reply.post_id != root_post.at_uri for r in rows)


def test_traverse_reddit_seed_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reddit seeds: resolve_node is called on seed.url, then get_replies returns children
    that are converted to ResolvedPost via the reddit adapter."""
    seed = _seed(
        _reddit_resolved(
            post_id="t3_live",
            author="seed_user",
            url="https://www.reddit.com/r/Sci/comments/live/x/",
        ),
        _endorsement(),
    )

    parent_node = RedditNode(
        id="t3_live",
        kind="submission",
        author="seed_user",
        body="seed body",
        created_utc=1750000000.0,
        permalink=seed.post.url,
    )
    self_reply = _reddit_node(cid="self", author="seed_user", body="self reply")
    other_reply = _reddit_node(cid="other", author="other_user", body="external reply")

    reddit_calls = _patch_reddit(
        monkeypatch,
        nodes_by_url={seed.post.url: parent_node},
        replies_by_id={parent_node.id: [self_reply, other_reply]},
    )
    _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), [seed]))

    assert reddit_calls["resolve"] == [seed.post.url]
    assert reddit_calls["replies"] == [parent_node.id]
    # Self-reply dropped; only the external reply survives.
    assert len(rows) == 1
    assert rows[0].reply.platform == "reddit"
    assert rows[0].reply.text == "external reply"
    assert rows[0].reply.author_handle == "other_user"
    assert rows[0].reply.post_id == "t1_other"


def test_traverse_returns_empty_when_no_seeds_qualify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty input -> empty output, and we never touch any backend."""
    fetched_uris = _patch_bsky_thread(monkeypatch, by_uri={})
    reddit_calls = _patch_reddit(monkeypatch, nodes_by_url={}, replies_by_id={})
    classified = _patch_classifier(monkeypatch)

    rows = asyncio.run(traverse_depth1(_as_client(object()), []))

    assert rows == []
    assert fetched_uris == []
    assert reddit_calls == {"resolve": [], "replies": []}
    assert classified == []


def test_bsky_to_resolved_url_rebuilt_from_handle_and_rkey() -> None:
    """``_bsky_to_resolved`` rebuilds the bsky.app URL when handle and rkey are present."""
    from altendor.traverse.replies import _bsky_to_resolved

    p = BskyPost(
        at_uri="at://did:plc:abc/app.bsky.feed.post/xyz789",
        cid="cid-1",
        text="hello",
        author_did="did:plc:abc",
        author_handle="carl.bsky.social",
        created_at="2026-06-01T12:00:00.000Z",
    )
    resolved = _bsky_to_resolved(p)
    assert resolved.post_id == p.at_uri
    assert resolved.platform == "bluesky"
    assert resolved.url == "https://bsky.app/profile/carl.bsky.social/post/xyz789"
    assert resolved.author_handle == "carl.bsky.social"
    assert resolved.author_id == "did:plc:abc"


def test_reddit_to_resolved_carries_permalink_and_author() -> None:
    """``_reddit_to_resolved`` maps node attributes into a ResolvedPost."""
    from altendor.traverse.replies import _reddit_to_resolved

    n = RedditNode(
        id="t1_xyz",
        kind="comment",
        author="alice",
        body="reply body",
        created_utc=1750000000.0,
        permalink="https://www.reddit.com/r/Sci/comments/abc/x/xyz/",
    )
    resolved = _reddit_to_resolved(n)
    assert resolved.post_id == "t1_xyz"
    assert resolved.platform == "reddit"
    assert resolved.text == "reply body"
    assert resolved.url == n.permalink
    assert resolved.author_handle == "alice"
    assert resolved.author_id == "alice"
    # ``created_utc`` (epoch seconds) was normalised to ISO-8601.
    assert "T" in resolved.created_at


# Silence "imported but unused" for the helpers we re-export for clarity.
_ = (Any, Flag, Irrelevant)
