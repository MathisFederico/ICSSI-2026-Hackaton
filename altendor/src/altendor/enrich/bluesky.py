"""Bluesky AT Protocol enrichment — see S6 in PIPELINE_PLAN.md.

Resolves ``bsky.app`` post URLs (or raw ``at://`` URIs) to their text and
descendants via Bluesky's public XRPC endpoints. No auth required.

Public API:

- :func:`resolve_post` — URL/URI → :class:`BskyPost` (or ``None`` on failure).
- :func:`get_thread` — AT URI → flattened list of :class:`BskyPost` (root + BFS descendants).

All network calls are wrapped in ``try/except`` and return ``None``/``[]`` on
failure (logged at WARNING). A module-level :class:`asyncio.Semaphore` caps
concurrent requests at 8 across the whole module so notebook users can fan out
freely without hammering the public endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

XRPC_BASE = "https://public.api.bsky.app/xrpc"

# Cap concurrent requests across the module. Bluesky's public endpoint is
# unauthenticated but rate-limited; 8 in flight is a friendly default.
_SEMAPHORE = asyncio.Semaphore(8)

_BSKY_APP_URL_RE = re.compile(r"https?://bsky\.app/profile/([^/]+)/post/([a-zA-Z0-9]+)")
_AT_URI_RE = re.compile(r"at://(did:[a-z]+:[a-zA-Z0-9._:-]+)/app\.bsky\.feed\.post/([a-zA-Z0-9]+)$")


@dataclass(frozen=True)
class BskyPost:
    """A flattened Bluesky post view."""

    at_uri: str
    cid: str
    text: str
    author_did: str
    author_handle: str
    created_at: str


async def _xrpc_get(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Call an XRPC endpoint and return the JSON body, or ``None`` on failure."""
    url = f"{XRPC_BASE}/{endpoint}"
    async with _SEMAPHORE:
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Bluesky XRPC %s returned %s (params=%s)", endpoint, resp.status, params)
                    return None
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Bluesky XRPC %s failed: %s", endpoint, exc)
            return None


async def _resolve_handle(session: aiohttp.ClientSession, handle: str) -> str | None:
    """Resolve a handle (e.g. ``alice.bsky.social``) to its DID."""
    data = await _xrpc_get(session, "com.atproto.identity.resolveHandle", {"handle": handle})
    if not data:
        return None
    did = data.get("did")
    if not isinstance(did, str):
        logger.warning("resolveHandle returned no did for %s: %s", handle, data)
        return None
    return did


async def _url_to_at_uri(session: aiohttp.ClientSession, url_or_uri: str) -> str | None:
    """Normalise a ``bsky.app`` URL or AT URI into a canonical AT URI."""
    if _AT_URI_RE.match(url_or_uri):
        return url_or_uri
    m = _BSKY_APP_URL_RE.match(url_or_uri)
    if not m:
        logger.warning("Not a recognised Bluesky URL or AT URI: %s", url_or_uri)
        return None
    handle, rkey = m.group(1), m.group(2)
    # Handles are usually a domain (carlbergstrom.com) or *.bsky.social, but a
    # bsky.app URL may also embed the DID directly.
    did = handle if handle.startswith("did:") else await _resolve_handle(session, handle)
    if not did:
        return None
    return f"at://{did}/app.bsky.feed.post/{rkey}"


def _post_view_to_bsky_post(post: dict[str, Any]) -> BskyPost | None:
    """Convert one XRPC ``postView`` dict to a :class:`BskyPost`."""
    try:
        author = post.get("author") or {}
        record = post.get("record") or {}
        return BskyPost(
            at_uri=post["uri"],
            cid=post["cid"],
            text=str(record.get("text", "")),
            author_did=str(author.get("did", "")),
            author_handle=str(author.get("handle", "")),
            created_at=str(record.get("createdAt", "")),
        )
    except (KeyError, TypeError) as exc:
        logger.warning("Malformed Bluesky postView: %s (%s)", exc, post.get("uri"))
        return None


def _flatten_thread(thread: dict[str, Any], depth: int) -> list[BskyPost]:
    """BFS-walk the nested ``thread`` dict, returning root + descendants <= ``depth``."""
    out: list[BskyPost] = []
    # Queue of (node, level). Level 0 = root.
    queue: deque[tuple[dict[str, Any], int]] = deque([(thread, 0)])
    while queue:
        node, level = queue.popleft()
        if not isinstance(node, dict):
            continue
        # Skip not-found / blocked thread nodes (they lack a ``post`` field).
        post = node.get("post")
        if isinstance(post, dict):
            bp = _post_view_to_bsky_post(post)
            if bp is not None:
                out.append(bp)
        if level < depth:
            replies = node.get("replies") or []
            for child in replies:
                queue.append((child, level + 1))
    return out


async def resolve_post(
    url_or_uri: str,
    *,
    session: aiohttp.ClientSession | None = None,
) -> BskyPost | None:
    """Resolve a ``bsky.app`` URL or AT URI to a single :class:`BskyPost`.

    Returns ``None`` on any failure (network, 404, malformed URL, etc.).
    Callers may pass their own :class:`aiohttp.ClientSession` for connection
    pooling; otherwise a short-lived session is created.
    """
    owns_session = session is None
    if session is None:
        session = aiohttp.ClientSession()
    try:
        at_uri = await _url_to_at_uri(session, url_or_uri)
        if at_uri is None:
            return None
        data = await _xrpc_get(
            session,
            "app.bsky.feed.getPostThread",
            {"uri": at_uri, "depth": 0, "parentHeight": 0},
        )
        if not data:
            return None
        thread = data.get("thread")
        if not isinstance(thread, dict):
            logger.warning("getPostThread returned no thread for %s", at_uri)
            return None
        flat = _flatten_thread(thread, depth=0)
        return flat[0] if flat else None
    finally:
        if owns_session:
            await session.close()


async def get_thread(
    at_uri: str,
    *,
    depth: int = 1,
    parent_height: int = 0,
    session: aiohttp.ClientSession | None = None,
) -> list[BskyPost]:
    """Return the root post plus its descendants down to ``depth``, flattened (BFS).

    Returns an empty list on failure. Root is always first if present.
    """
    owns_session = session is None
    if session is None:
        session = aiohttp.ClientSession()
    try:
        data = await _xrpc_get(
            session,
            "app.bsky.feed.getPostThread",
            {"uri": at_uri, "depth": depth, "parentHeight": parent_height},
        )
        if not data:
            return []
        thread = data.get("thread")
        if not isinstance(thread, dict):
            logger.warning("getPostThread returned no thread for %s", at_uri)
            return []
        return _flatten_thread(thread, depth=depth)
    finally:
        if owns_session:
            await session.close()
