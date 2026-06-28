"""Post text-resolver dispatcher (stage S8).

Single async entry point that takes one Altmetric ``posts`` row dict (from
:func:`altendor.bigquery.queries.posts_by_research_output_ids`) and returns
a normalized :class:`ResolvedPost` carrying the post's full text.

Dispatches by ``post.type``:

* ``"bsky"`` -> :func:`altendor.enrich.bluesky.resolve_post` (async).
* ``"rdt"``  -> :func:`altendor.enrich.reddit.resolve_node` (sync).
* every other Altmetric ``type`` keeps the raw ``posts.title`` as the text
  (no free public API to enrich Twitter/blog/news/etc. in v1).

The resolver is conservative: any failure of the underlying call (missing
client, network error, deleted content) cleanly falls back to the raw
title with ``text_confidence="low"`` so downstream stages can decide
whether to keep or drop the row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from altendor.enrich import bluesky, reddit

if TYPE_CHECKING:
    import aiohttp
    import praw


Platform = Literal[
    "bluesky",
    "reddit",
    "twitter",
    "blog",
    "news",
    "wikipedia",
    "video",
    "patent",
    "policy",
    "podcast",
    "peer_review",
    "other",
]

TextConfidence = Literal["high", "low"]


# Altmetric ``posts.type`` -> our platform literal. Anything not listed here
# falls back to ``"other"``.
_TYPE_TO_PLATFORM: dict[str, Platform] = {
    "bsky": "bluesky",
    "rdt": "reddit",
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


_HIGH_CONFIDENCE_TITLE_LEN = 140


@dataclass(frozen=True)
class ResolvedPost:
    """A normalized, platform-agnostic view of one Altmetric post row.

    ``text`` is the best available representation of the post's content: the
    resolver's body when enrichment succeeded, otherwise the raw
    ``posts.title`` from BigQuery (kept on :attr:`raw_title` for traceability).
    ``text_confidence`` is ``"low"`` when the resolver fell back to a short
    raw title (``len(raw_title) < 140``), ``"high"`` otherwise.
    """

    post_id: str
    platform: Platform
    text: str
    author_handle: str | None
    author_id: str | None
    url: str
    created_at: str
    raw_title: str
    text_confidence: TextConfidence


def _post_id(post_row: dict[str, Any]) -> str:
    """Return the post's id, tolerating either ``post_id`` or ``id``."""
    value = post_row.get("post_id")
    if value is None:
        value = post_row.get("id")
    return "" if value is None else str(value)


def _coerce_iso(value: object) -> str | None:
    """Normalise a created-at value (Timestamp / datetime / float / str) to ISO 8601.

    Returns ``None`` when *value* is missing or cannot be coerced — callers
    fall through to the next-best source.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (int, float)):
        # Reddit's ``created_utc`` is seconds since the epoch.
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return str(iso())
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _fallback_author(post_row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull ``screen_name``/``user_id`` from ``attention_source`` if present."""
    source = post_row.get("attention_source")
    if not isinstance(source, dict):
        return None, None
    screen_name = source.get("screen_name")
    user_id = source.get("user_id")
    handle = str(screen_name) if isinstance(screen_name, str) and screen_name else None
    author_id = str(user_id) if user_id is not None and str(user_id) else None
    return handle, author_id


def _platform_for(post_type: object) -> Platform:
    """Map an Altmetric ``posts.type`` value to our platform literal."""
    if not isinstance(post_type, str):
        return "other"
    return _TYPE_TO_PLATFORM.get(post_type, "other")


def _confidence_for(raw_title: str) -> TextConfidence:
    """Fallback-to-title path: confidence is ``"low"`` for short titles."""
    return "low" if len(raw_title) < _HIGH_CONFIDENCE_TITLE_LEN else "high"


def _fallback_resolved(
    post_row: dict[str, Any],
    platform: Platform,
    raw_title: str,
) -> ResolvedPost:
    """Build a :class:`ResolvedPost` from ``posts.title`` (no enrichment)."""
    handle, author_id = _fallback_author(post_row)
    created_at = _coerce_iso(post_row.get("date")) or ""
    return ResolvedPost(
        post_id=_post_id(post_row),
        platform=platform,
        text=raw_title,
        author_handle=handle,
        author_id=author_id,
        url=str(post_row.get("url") or ""),
        created_at=created_at,
        raw_title=raw_title,
        text_confidence=_confidence_for(raw_title),
    )


async def _resolve_bluesky(
    post_row: dict[str, Any],
    raw_title: str,
    session: aiohttp.ClientSession | None,
) -> ResolvedPost:
    """Dispatch a ``type=bsky`` row through the Bluesky XRPC resolver."""
    url = str(post_row.get("url") or "")
    bsky_post = None
    if url:
        bsky_post = await bluesky.resolve_post(url, session=session)
    if bsky_post is None:
        return _fallback_resolved(post_row, "bluesky", raw_title)
    created_at = _coerce_iso(bsky_post.created_at) or _coerce_iso(post_row.get("date")) or ""
    return ResolvedPost(
        post_id=_post_id(post_row),
        platform="bluesky",
        text=bsky_post.text,
        author_handle=bsky_post.author_handle or None,
        author_id=bsky_post.author_did or None,
        url=url,
        created_at=created_at,
        raw_title=raw_title,
        text_confidence="high",
    )


def _resolve_reddit(
    post_row: dict[str, Any],
    raw_title: str,
    client: praw.Reddit | None,
) -> ResolvedPost:
    """Dispatch a ``type=rdt`` row through the PRAW resolver."""
    url = str(post_row.get("url") or "")
    node = None
    if url:
        node = reddit.resolve_node(url, client=client)
    if node is None:
        return _fallback_resolved(post_row, "reddit", raw_title)
    created_at = _coerce_iso(node.created_utc) or _coerce_iso(post_row.get("date")) or ""
    author = node.author or None
    return ResolvedPost(
        post_id=_post_id(post_row),
        platform="reddit",
        text=node.body,
        author_handle=author,
        author_id=author,
        url=url,
        created_at=created_at,
        raw_title=raw_title,
        text_confidence="high",
    )


async def resolve_full_text(
    post_row: dict[str, Any],
    *,
    bsky_session: aiohttp.ClientSession | None = None,
    reddit_client: object | None = None,
) -> ResolvedPost:
    """Resolve one Altmetric ``posts`` row to a :class:`ResolvedPost`.

    Dispatches by ``post_row["type"]``:

    * ``"bsky"`` -> awaits :func:`bluesky.resolve_post`.
    * ``"rdt"``  -> calls :func:`reddit.resolve_node` (sync; not awaited).
    * everything else -> keeps :data:`posts.title` as the text.

    On any failure from the underlying resolver (returns ``None``) the row
    falls back to ``posts.title`` and ``text_confidence`` becomes
    ``"low"`` for titles shorter than 140 characters, ``"high"`` otherwise.
    """
    raw_title = str(post_row.get("title") or "")
    post_type = post_row.get("type")

    if post_type == "bsky":
        return await _resolve_bluesky(post_row, raw_title, bsky_session)
    if post_type == "rdt":
        # ``reddit_client`` is typed as ``object`` in the public signature so
        # callers needn't import praw; ``resolve_node`` itself accepts None
        # and will try to build a client from env vars.
        from typing import cast

        client = cast("praw.Reddit | None", reddit_client)
        return _resolve_reddit(post_row, raw_title, client)

    platform = _platform_for(post_type)
    return _fallback_resolved(post_row, platform, raw_title)


__all__ = [
    "Platform",
    "ResolvedPost",
    "TextConfidence",
    "resolve_full_text",
]
