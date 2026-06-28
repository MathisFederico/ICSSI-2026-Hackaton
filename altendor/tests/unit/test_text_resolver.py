"""Unit tests for :mod:`altendor.enrich.text_resolver`.

All tests run offline: the dispatched ``bluesky.resolve_post`` and
``reddit.resolve_node`` calls are monkeypatched, so no network or
Reddit credentials are required.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from altendor.enrich import bluesky, reddit, text_resolver
from altendor.enrich.bluesky import BskyPost
from altendor.enrich.reddit import RedditNode
from altendor.enrich.text_resolver import ResolvedPost, resolve_full_text


def _row(**overrides: object) -> dict[str, Any]:
    """Build a minimal Altmetric ``posts`` row dict with sensible defaults."""
    base: dict[str, Any] = {
        "post_id": "post-1",
        "type": "tweet",
        "subtype": None,
        "date": "2026-06-01T12:00:00+00:00",
        "url": "https://example.com/post",
        "title": "a short title",
        "attention_source": {"screen_name": "alice", "user_id": "uid-1"},
        "retweet": False,
    }
    base.update(overrides)
    return base


def test_dispatch_bluesky(monkeypatch: pytest.MonkeyPatch) -> None:
    """``type=bsky`` calls bluesky.resolve_post and copies its text/handle/did."""
    fake = BskyPost(
        at_uri="at://did:plc:abc/app.bsky.feed.post/xyz",
        cid="cid-1",
        text="full bluesky post body",
        author_did="did:plc:abc",
        author_handle="carl.bsky.social",
        created_at="2026-06-01T12:00:00.000Z",
    )

    async def fake_resolve_post(url: str, *, session: object = None) -> BskyPost:
        return fake

    monkeypatch.setattr(bluesky, "resolve_post", fake_resolve_post)

    row = _row(
        type="bsky",
        url="https://bsky.app/profile/carl.bsky.social/post/xyz",
        title="raw posts.title (ignored when resolver succeeds)",
    )

    result = asyncio.run(resolve_full_text(row))

    assert isinstance(result, ResolvedPost)
    assert result.platform == "bluesky"
    assert result.text == "full bluesky post body"
    assert result.author_handle == "carl.bsky.social"
    assert result.author_id == "did:plc:abc"
    assert result.text_confidence == "high"
    assert result.raw_title == row["title"]


def test_dispatch_reddit(monkeypatch: pytest.MonkeyPatch) -> None:
    """``type=rdt`` calls reddit.resolve_node and copies body/author into the result."""
    fake_node = RedditNode(
        id="t3_abc",
        kind="submission",
        author="alice",
        body="full reddit submission body",
        created_utc=1750000000.0,
        permalink="https://www.reddit.com/r/Scientometrics/comments/abc/title/",
    )

    def fake_resolve_node(url: str, *, client: object = None) -> RedditNode:
        return fake_node

    monkeypatch.setattr(reddit, "resolve_node", fake_resolve_node)

    row = _row(
        type="rdt",
        url="https://www.reddit.com/r/Scientometrics/comments/abc/title/",
        title="raw posts.title",
    )

    result = asyncio.run(resolve_full_text(row))

    assert result.platform == "reddit"
    assert result.text == "full reddit submission body"
    assert result.author_handle == "alice"
    assert result.author_id == "alice"
    assert result.text_confidence == "high"
    # created_utc (epoch seconds) was normalised to ISO 8601.
    assert result.created_at.startswith("2025-") or result.created_at.startswith("2026-")
    assert "T" in result.created_at


def test_dispatch_twitter_uses_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """``type=tweet`` never invokes a resolver; ``text == posts.title``."""
    called = {"bsky": 0, "reddit": 0}

    async def boom_bsky(url: str, *, session: object = None) -> None:
        called["bsky"] += 1
        return None

    def boom_reddit(url: str, *, client: object = None) -> None:
        called["reddit"] += 1
        return None

    monkeypatch.setattr(bluesky, "resolve_post", boom_bsky)
    monkeypatch.setattr(reddit, "resolve_node", boom_reddit)

    row = _row(type="tweet", title="a tweet")
    result = asyncio.run(resolve_full_text(row))

    assert result.platform == "twitter"
    assert result.text == "a tweet"
    assert result.text_confidence == "low"  # len("a tweet") < 140
    assert called == {"bsky": 0, "reddit": 0}


def test_unknown_type_falls_back_to_title() -> None:
    """An unrecognised ``type`` keeps the title and uses platform=``other``."""
    row = _row(type="weird", title="some unknown platform post")
    result = asyncio.run(resolve_full_text(row))

    assert result.platform == "other"
    assert result.text == "some unknown platform post"
    assert result.raw_title == "some unknown platform post"


def test_short_title_marked_low_confidence() -> None:
    """Fallback path: titles shorter than 140 chars -> ``low``."""
    row = _row(type="msm", title="A short news headline")
    result = asyncio.run(resolve_full_text(row))

    assert result.platform == "news"
    assert result.text_confidence == "low"
    assert len(result.raw_title) < 140


def test_long_title_marked_high_confidence() -> None:
    """Fallback path: titles >= 140 chars are treated as high confidence."""
    long_title = "x" * 200
    row = _row(type="blog", title=long_title)
    result = asyncio.run(resolve_full_text(row))

    assert result.platform == "blog"
    assert result.text == long_title
    assert len(result.raw_title) >= 140
    assert result.text_confidence == "high"


def test_resolver_failure_falls_back_to_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """When bluesky.resolve_post returns ``None`` we fall back to ``posts.title``."""

    async def returns_none(url: str, *, session: object = None) -> None:
        return None

    monkeypatch.setattr(bluesky, "resolve_post", returns_none)

    short_row = _row(type="bsky", title="short", url="https://bsky.app/profile/x/post/y")
    short_result = asyncio.run(resolve_full_text(short_row))
    assert short_result.platform == "bluesky"
    assert short_result.text == "short"
    assert short_result.text_confidence == "low"

    long_row = _row(type="bsky", title="L" * 200, url="https://bsky.app/profile/x/post/y")
    long_result = asyncio.run(resolve_full_text(long_row))
    assert long_result.text_confidence == "high"
    assert long_result.text == "L" * 200


def test_fallback_pulls_author_from_attention_source() -> None:
    """When falling back to the title, author_handle/author_id come from attention_source."""
    row = _row(
        type="tweet",
        title="a tweet",
        attention_source={"screen_name": "alice", "user_id": "uid-42"},
    )
    result = asyncio.run(resolve_full_text(row))
    assert result.author_handle == "alice"
    assert result.author_id == "uid-42"


def test_fallback_handles_missing_attention_source() -> None:
    """A missing/None ``attention_source`` yields ``None`` for both author fields."""
    row = _row(type="tweet", title="a tweet", attention_source=None)
    result = asyncio.run(resolve_full_text(row))
    assert result.author_handle is None
    assert result.author_id is None


def test_post_id_fallback_to_id_key() -> None:
    """Either ``post_id`` or ``id`` is accepted as the post identifier."""
    row = _row(title="x")
    row.pop("post_id")
    row["id"] = "from-id-key"
    result = asyncio.run(resolve_full_text(row))
    assert result.post_id == "from-id-key"


def test_known_alt_types_all_map() -> None:
    """Every documented Altmetric ``type`` lands on the right platform literal."""
    expected = {
        "tweet": "twitter",
        "blog": "blog",
        "msm": "news",
        "wikipedia": "wikipedia",
        "video": "video",
        "patent": "patent",
        "policy": "policy",
        "podcast": "podcast",
        "peer_review": "peer_review",
    }
    for alt_type, platform in expected.items():
        row = _row(type=alt_type, title=f"title for {alt_type}")
        result = asyncio.run(resolve_full_text(row))
        assert result.platform == platform, alt_type
        assert result.text == row["title"]


def test_text_resolver_re_exports_resolved_post() -> None:
    """``ResolvedPost`` is part of the public API surface of the module."""
    assert hasattr(text_resolver, "ResolvedPost")
    assert hasattr(text_resolver, "resolve_full_text")
