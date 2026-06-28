"""Offline fixture-replay tests for ``altendor.enrich.bluesky``.

These tests stub :class:`aiohttp.ClientSession.get` so no network is required.
The stub mimics the async-context-manager protocol that ``aiohttp`` exposes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl, urlparse

import aiohttp
import pytest
from altendor.enrich.bluesky import BskyPost, _flatten_thread, get_thread, resolve_post


def _as_session(stub: object) -> aiohttp.ClientSession:
    """Tell the type-checker the stub is a real ClientSession."""
    return cast(aiohttp.ClientSession, stub)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
THREAD_FIXTURE = FIXTURES / "bluesky_thread.json"
HANDLE_FIXTURE = FIXTURES / "bluesky_resolve_handle.json"


# ---------------------------------------------------------------------------
# aiohttp stub
# ---------------------------------------------------------------------------


class _StubResponse:
    """Async context manager that mimics ``aiohttp.ClientResponse``."""

    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_StubResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self) -> object:
        return self._payload


class AiohttpStub:
    """A minimal stand-in for :class:`aiohttp.ClientSession`.

    Routes are matched on a substring of the request URL. Each route may be a
    static ``(status, payload)`` tuple or a callable accepting the parsed
    query dict and returning ``(status, payload)``.
    """

    def __init__(self, routes: Iterable[tuple[str, Any]]) -> None:
        self._routes: list[tuple[str, Any]] = list(routes)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def _resolve(self, url: str, params: dict[str, str]) -> tuple[int, Any]:
        for needle, value in self._routes:
            if needle in url:
                if callable(value):
                    return value(params)
                return value
        return 404, {"error": "NotFound", "message": f"no stub for {url}"}

    def get(self, url: str, params: dict[str, str] | None = None) -> _StubResponse:
        params = dict(params or {})
        # Merge any query already on the URL (mirrors aiohttp behaviour for test inspection).
        merged = dict(parse_qsl(urlparse(url).query))
        merged.update(params)
        self.calls.append((url, merged))
        status, payload = self._resolve(url, merged)
        return _StubResponse(status, payload)

    async def close(self) -> None:
        return None


class ExplodingStub:
    """Raises :class:`aiohttp.ClientError` on every ``.get()``."""

    def get(self, url: str, params: dict[str, str] | None = None) -> "ExplodingStub":
        return self

    async def __aenter__(self) -> "ExplodingStub":
        raise aiohttp.ClientError("boom")

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def thread_payload() -> dict[str, Any]:
    return json.loads(THREAD_FIXTURE.read_text())


@pytest.fixture
def handle_payload() -> dict[str, Any]:
    return json.loads(HANDLE_FIXTURE.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_uri_construction(handle_payload: dict[str, Any]) -> None:
    """A bsky.app URL is converted to ``at://<did>/app.bsky.feed.post/<rkey>``."""
    stub = AiohttpStub(
        [
            ("com.atproto.identity.resolveHandle", (200, handle_payload)),
            # Return a not-found thread so ``resolve_post`` short-circuits after URI build.
            ("app.bsky.feed.getPostThread", (200, {"thread": {}})),
        ]
    )

    async def go() -> None:
        await resolve_post(
            "https://bsky.app/profile/carlbergstrom.com/post/3mpclwr75w22e",
            session=_as_session(stub),
        )

    asyncio.run(go())

    # The thread call must use the resolved AT URI.
    thread_calls = [c for c in stub.calls if "getPostThread" in c[0]]
    assert thread_calls, "expected getPostThread to be invoked"
    expected_uri = f"at://{handle_payload['did']}/app.bsky.feed.post/3mpclwr75w22e"
    assert thread_calls[0][1]["uri"] == expected_uri


def test_resolve_post_from_url_extracts_text(
    handle_payload: dict[str, Any],
    thread_payload: dict[str, Any],
) -> None:
    """``resolve_post`` returns a BskyPost whose text matches the fixture's record.text."""
    stub = AiohttpStub(
        [
            ("com.atproto.identity.resolveHandle", (200, handle_payload)),
            ("app.bsky.feed.getPostThread", (200, thread_payload)),
        ]
    )

    async def go() -> BskyPost | None:
        return await resolve_post(
            "https://bsky.app/profile/carlbergstrom.com/post/3mpclwr75w22e",
            session=_as_session(stub),
        )

    post = asyncio.run(go())
    assert post is not None
    expected_text = thread_payload["thread"]["post"]["record"]["text"]
    assert post.text == expected_text
    assert post.at_uri == thread_payload["thread"]["post"]["uri"]
    assert post.author_handle == thread_payload["thread"]["post"]["author"]["handle"]
    assert post.author_did == thread_payload["thread"]["post"]["author"]["did"]
    assert post.created_at == thread_payload["thread"]["post"]["record"]["createdAt"]


def test_get_thread_flattens_replies(thread_payload: dict[str, Any]) -> None:
    """Flat output is BFS: root first, then all level-1 replies, then level-2, etc."""
    stub = AiohttpStub([("app.bsky.feed.getPostThread", (200, thread_payload))])

    async def go() -> list[BskyPost]:
        return await get_thread(
            thread_payload["thread"]["post"]["uri"],
            depth=2,
            session=_as_session(stub),
        )

    flat = asyncio.run(go())

    # Sanity: there is at least one L1 and one L2 in the fixture.
    root = thread_payload["thread"]
    l1_replies = root.get("replies") or []
    assert l1_replies, "fixture should have level-1 replies"
    l2_count = sum(len(r.get("replies") or []) for r in l1_replies)
    assert l2_count > 0, "fixture should have level-2 replies for a meaningful BFS test"

    # Root first.
    assert flat[0].at_uri == root["post"]["uri"]
    assert flat[0].text == root["post"]["record"]["text"]

    # BFS ordering: every level-1 reply appears before any level-2 reply.
    l1_uris = {r["post"]["uri"] for r in l1_replies if "post" in r}
    l2_uris: set[str] = set()
    for r in l1_replies:
        for rr in r.get("replies") or []:
            if "post" in rr:
                l2_uris.add(rr["post"]["uri"])

    # Find positions in the flat output.
    positions = {p.at_uri: idx for idx, p in enumerate(flat)}
    for u in l1_uris:
        if u in positions:
            for v in l2_uris:
                if v in positions:
                    assert positions[u] < positions[v], (
                        f"BFS violated: L1 {u} (pos {positions[u]}) should precede L2 {v} (pos {positions[v]})"
                    )

    # And every flattened uri is unique.
    uris = [p.at_uri for p in flat]
    assert len(uris) == len(set(uris))


def test_get_thread_honours_depth(thread_payload: dict[str, Any]) -> None:
    """``depth=0`` returns only the root; ``depth=1`` excludes level-2."""
    stub = AiohttpStub([("app.bsky.feed.getPostThread", (200, thread_payload))])

    async def go(d: int) -> list[BskyPost]:
        return await get_thread(
            thread_payload["thread"]["post"]["uri"],
            depth=d,
            session=_as_session(stub),
        )

    only_root = asyncio.run(go(0))
    assert len(only_root) == 1
    assert only_root[0].at_uri == thread_payload["thread"]["post"]["uri"]

    depth_one = asyncio.run(go(1))
    # depth=1 must contain root + every L1 with a ``post`` field, and no more.
    expected = 1 + sum(1 for r in (thread_payload["thread"].get("replies") or []) if "post" in r)
    assert len(depth_one) == expected


def test_resolve_returns_none_on_404(handle_payload: dict[str, Any]) -> None:
    """A 404 from getPostThread yields ``None``."""
    stub = AiohttpStub(
        [
            ("com.atproto.identity.resolveHandle", (200, handle_payload)),
            ("app.bsky.feed.getPostThread", (404, {"error": "NotFound"})),
        ]
    )

    async def go() -> BskyPost | None:
        return await resolve_post(
            "https://bsky.app/profile/carlbergstrom.com/post/doesnotexist",
            session=_as_session(stub),
        )

    assert asyncio.run(go()) is None


def test_resolve_returns_none_on_client_error() -> None:
    """Network errors are swallowed and surfaced as ``None``."""
    stub = ExplodingStub()

    async def go() -> BskyPost | None:
        return await resolve_post(
            "at://did:plc:tbqqvyv6pjjww44glrmycaxl/app.bsky.feed.post/3mpclwr75w22e",
            session=_as_session(stub),
        )

    assert asyncio.run(go()) is None


def test_resolve_rejects_unknown_url() -> None:
    """A non-bsky URL returns ``None`` without making any network call."""
    stub = AiohttpStub([])

    async def go() -> BskyPost | None:
        return await resolve_post(
            "https://example.com/not-a-bsky-url",
            session=_as_session(stub),
        )

    assert asyncio.run(go()) is None
    assert stub.calls == []


def test_flatten_handles_missing_post_node() -> None:
    """Blocked / not-found nodes (no ``post`` field) are skipped, not crashed on."""
    tree = {
        "post": {
            "uri": "at://did:plc:x/app.bsky.feed.post/root",
            "cid": "rootcid",
            "author": {"did": "did:plc:x", "handle": "x.bsky.social"},
            "record": {"text": "root", "createdAt": "2026-01-01T00:00:00Z"},
        },
        "replies": [
            {"blocked": True},  # no ``post`` — should be skipped.
            {
                "post": {
                    "uri": "at://did:plc:y/app.bsky.feed.post/r1",
                    "cid": "r1cid",
                    "author": {"did": "did:plc:y", "handle": "y.bsky.social"},
                    "record": {"text": "reply", "createdAt": "2026-01-02T00:00:00Z"},
                },
                "replies": [],
            },
        ],
    }
    flat = _flatten_thread(tree, depth=1)
    assert [p.text for p in flat] == ["root", "reply"]


