"""Depth-1 reply traversal — stage S13.

For each post that the classifier marked as an :class:`Endorsement`, fetch
its depth-1 replies (Bluesky + Reddit only — Twitter/blog/news/etc. have no
free public reply API in v1) and re-classify them through the same
:func:`altendor.classify.classifier.classify_post`. Returns one
:class:`TraversalRow` per reply.

Behaviour highlights:

* Only seeds with ``post.platform in {"bluesky", "reddit"}`` and
  ``result.kind == "endorsement"`` are processed; everything else is silently
  dropped.
* Replies authored by the seed's author are dropped — self-replies don't count
  as third-party endorsements/flags.
* Replies are de-duplicated by ``(platform, post_id)`` **before** classifying so
  we don't pay Anthropic twice for a post that's reachable from multiple seeds.
* Global cap ``MAX_REPLIES_TOTAL`` and per-parent cap ``MAX_REPLIES_PER_PARENT``
  bound the work; seeds are processed in order so the cap is deterministic.
* Bluesky uses the already-existing module-level semaphore in
  :mod:`altendor.enrich.bluesky`; we fan out seeds with ``asyncio.gather``.
  Reddit (PRAW) is sync and called serially — the free Reddit rate limits are
  tight enough that there's no point parallelising at the seed level.
* Per-seed exceptions are logged at ``WARNING`` and that seed contributes
  zero rows; the rest of the traversal continues.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from altendor.classify.classifier import PaperCtx, classify_post
from altendor.classify.schema import ClassifyResult, Endorsement
from altendor.enrich import bluesky, reddit
from altendor.enrich.bluesky import BskyPost
from altendor.enrich.reddit import RedditNode
from altendor.enrich.text_resolver import ResolvedPost, _coerce_iso

if TYPE_CHECKING:
    import aiohttp
    import anthropic
    import praw

logger = logging.getLogger(__name__)


MAX_REPLIES_TOTAL: int = 200
"""Global cap on the number of replies classified across the whole traversal."""

MAX_REPLIES_PER_PARENT: int = 20
"""Per-seed cap on the number of replies fetched from one parent post."""

BLUESKY_CONCURRENCY: int = 8
"""Concurrent in-flight Bluesky thread fetches at the seed level."""

REDDIT_CONCURRENCY: int = 4
"""Concurrent in-flight Reddit reply fetches at the seed level (reserved)."""


@dataclass(frozen=True)
class TraversalSeed:
    """One classified endorsement to traverse from.

    ``post`` must have ``platform`` in ``{"bluesky", "reddit"}`` and
    ``result.kind`` must be ``"endorsement"`` for the seed to be processed;
    seeds that don't meet both conditions are silently skipped.
    """

    post: ResolvedPost
    paper: PaperCtx
    result: ClassifyResult


@dataclass(frozen=True)
class TraversalRow:
    """One traversed reply, re-classified through the same classifier."""

    parent_post_id: str
    reply: ResolvedPost
    result: ClassifyResult


# ---------------------------------------------------------------------------
# Native-type -> ResolvedPost adapters
# ---------------------------------------------------------------------------


def _bsky_to_resolved(p: BskyPost) -> ResolvedPost:
    """Adapt a :class:`BskyPost` (XRPC view) to a :class:`ResolvedPost`.

    The post_id is the AT URI (mirrors what ``text_resolver._resolve_bluesky``
    stores for seed posts); the URL is rebuilt from the handle + rkey when
    possible and otherwise falls back to the AT URI itself.
    """
    rkey = p.at_uri.rsplit("/", 1)[-1] if p.at_uri else ""
    if p.author_handle and rkey:
        url = f"https://bsky.app/profile/{p.author_handle}/post/{rkey}"
    else:
        url = p.at_uri
    created_at = _coerce_iso(p.created_at) or ""
    return ResolvedPost(
        post_id=p.at_uri,
        platform="bluesky",
        text=p.text,
        author_handle=p.author_handle or None,
        author_id=p.author_did or None,
        url=url,
        created_at=created_at,
        raw_title=p.text,
        text_confidence="high",
    )


def _reddit_to_resolved(n: RedditNode) -> ResolvedPost:
    """Adapt a :class:`RedditNode` (PRAW view) to a :class:`ResolvedPost`."""
    created_at = _coerce_iso(n.created_utc) or ""
    author = n.author or None
    return ResolvedPost(
        post_id=n.id,
        platform="reddit",
        text=n.body,
        author_handle=author,
        author_id=author,
        url=n.permalink,
        created_at=created_at,
        raw_title=n.body,
        text_confidence="high",
    )


# ---------------------------------------------------------------------------
# Per-seed reply fetchers (return list of (reply_post, native_author_key))
# ---------------------------------------------------------------------------


def _seed_author_key(seed: TraversalSeed) -> str | None:
    """Author identity to compare against for self-reply detection.

    For Bluesky we compare on the DID (stored on ``ResolvedPost.author_id`` by
    :func:`text_resolver._resolve_bluesky`); for Reddit on the username string
    (stored on both ``author_handle`` and ``author_id``).
    """
    return seed.post.author_id or seed.post.author_handle


async def _fetch_bluesky_replies(
    seed: TraversalSeed,
    *,
    session: aiohttp.ClientSession | None,
    max_per_parent: int,
) -> list[ResolvedPost]:
    """Fetch depth-1 Bluesky replies for one seed, drop root + self-replies."""
    flat = await bluesky.get_thread(seed.post.post_id, depth=1, session=session)
    # Root is always first when present; the rest are descendants.
    descendants = flat[1:]
    seed_author = _seed_author_key(seed)
    out: list[ResolvedPost] = []
    for bsky_post in descendants:
        if seed_author is not None and bsky_post.author_did == seed_author:
            continue
        out.append(_bsky_to_resolved(bsky_post))
        if len(out) >= max_per_parent:
            break
    return out


def _fetch_reddit_replies(
    seed: TraversalSeed,
    *,
    client: praw.Reddit | None,
    max_per_parent: int,
) -> list[ResolvedPost]:
    """Fetch depth-1 Reddit replies for one seed, drop self-replies."""
    node = reddit.resolve_node(seed.post.url, client=client)
    if node is None:
        return []
    children = reddit.get_replies(node, depth=1, max_per_parent=max_per_parent, client=client)
    seed_author = _seed_author_key(seed)
    out: list[ResolvedPost] = []
    for child in children:
        if seed_author is not None and child.author == seed_author:
            continue
        out.append(_reddit_to_resolved(child))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _is_endorsement_seed(seed: TraversalSeed) -> bool:
    """Filter: only Bluesky/Reddit endorsements are traversed in v1."""
    if seed.post.platform not in ("bluesky", "reddit"):
        return False
    return isinstance(seed.result, Endorsement)


async def _gather_replies(
    seeds: list[TraversalSeed],
    *,
    bsky_session: aiohttp.ClientSession | None,
    reddit_client: praw.Reddit | None,
    max_per_parent: int,
) -> list[tuple[TraversalSeed, list[ResolvedPost]]]:
    """Fetch replies for each seed, preserving input order.

    Bluesky seeds are fanned out concurrently via ``asyncio.gather`` (the
    underlying XRPC client has its own module-level semaphore at size 8, so we
    don't double-bound here). Reddit seeds are handled serially — PRAW is sync
    and Reddit's free rate limits are tight enough that there's no win from
    fanning out at the seed level.
    """
    bsky_tasks: dict[int, asyncio.Task[list[ResolvedPost]]] = {}
    reddit_indices: list[int] = []

    for idx, seed in enumerate(seeds):
        if seed.post.platform == "bluesky":
            bsky_tasks[idx] = asyncio.create_task(
                _fetch_bluesky_replies(seed, session=bsky_session, max_per_parent=max_per_parent),
            )
        else:
            reddit_indices.append(idx)

    bsky_results: dict[int, list[ResolvedPost] | BaseException] = {}
    if bsky_tasks:
        gathered = await asyncio.gather(*bsky_tasks.values(), return_exceptions=True)
        for idx, result in zip(bsky_tasks.keys(), gathered, strict=True):
            bsky_results[idx] = result

    out: list[tuple[TraversalSeed, list[ResolvedPost]]] = []
    for idx, seed in enumerate(seeds):
        replies: list[ResolvedPost]
        if idx in bsky_results:
            result = bsky_results[idx]
            if isinstance(result, BaseException):
                logger.warning("Bluesky traversal for seed %r failed: %s", seed.post.post_id, result)
                replies = []
            else:
                replies = result
        else:
            try:
                replies = _fetch_reddit_replies(seed, client=reddit_client, max_per_parent=max_per_parent)
            except Exception as exc:  # noqa: BLE001 - bound the blast radius of per-seed errors
                logger.warning("Reddit traversal for seed %r failed: %s", seed.post.post_id, exc)
                replies = []
        out.append((seed, replies))
    return out


async def traverse_depth1(
    anthropic_client: anthropic.Anthropic,
    seeds: list[TraversalSeed],
    *,
    bsky_session: aiohttp.ClientSession | None = None,
    reddit_client: object | None = None,
    max_total: int = MAX_REPLIES_TOTAL,
    max_per_parent: int = MAX_REPLIES_PER_PARENT,
) -> list[TraversalRow]:
    """Traverse depth-1 replies for each endorsement seed and re-classify them.

    Parameters
    ----------
    anthropic_client:
        Anthropic client passed through to :func:`classify_post`. Never read
        directly here.
    seeds:
        List of :class:`TraversalSeed`. Non-endorsement seeds and seeds on
        platforms other than ``bluesky``/``reddit`` are silently dropped.
    bsky_session:
        Optional shared :class:`aiohttp.ClientSession` for Bluesky XRPC calls.
        When ``None``, each underlying call creates a short-lived session.
    reddit_client:
        Optional ``praw.Reddit`` instance. Typed as ``object`` so callers
        needn't import praw; passed straight through to the reddit module.
    max_total:
        Hard global cap on the number of replies classified across the whole
        traversal. Defaults to :data:`MAX_REPLIES_TOTAL`.
    max_per_parent:
        Hard per-seed cap on the number of replies fetched from one parent
        post. Defaults to :data:`MAX_REPLIES_PER_PARENT`.

    Returns
    -------
    list[TraversalRow]
        One row per re-classified reply, deduplicated by
        ``(platform, post_id)`` across the run.
    """
    # 1) Filter seeds in input order.
    filtered = [s for s in seeds if _is_endorsement_seed(s)]
    if not filtered:
        return []

    # 2) Fetch replies for each seed.
    reddit_typed = cast("praw.Reddit | None", reddit_client)
    per_seed = await _gather_replies(
        filtered,
        bsky_session=bsky_session,
        reddit_client=reddit_typed,
        max_per_parent=max_per_parent,
    )

    # 3) Dedup by (platform, post_id) across the whole run, respecting input
    #    order so the global cap is deterministic. We pair each surviving reply
    #    with the seed it first showed up under.
    seen: set[tuple[str, str]] = set()
    to_classify: list[tuple[TraversalSeed, ResolvedPost]] = []
    for seed, replies in per_seed:
        for reply in replies:
            key = (reply.platform, reply.post_id)
            if key in seen:
                continue
            seen.add(key)
            to_classify.append((seed, reply))
            if len(to_classify) >= max_total:
                break
        if len(to_classify) >= max_total:
            break

    # 4) Re-classify each surviving reply. Classifier call is sync; the spec
    #    says callers can wrap the whole thing in ``asyncio.to_thread`` if they
    #    need to overlap with other I/O.
    rows: list[TraversalRow] = []
    for seed, reply in to_classify:
        try:
            result = classify_post(anthropic_client, reply, seed.paper)
        except Exception as exc:  # noqa: BLE001 - one bad classify shouldn't kill the run
            logger.warning("Reclassify failed for reply %r: %s", reply.post_id, exc)
            continue
        rows.append(
            TraversalRow(
                parent_post_id=seed.post.post_id,
                reply=reply,
                result=result,
            ),
        )
    return rows


__all__ = [
    "BLUESKY_CONCURRENCY",
    "MAX_REPLIES_PER_PARENT",
    "MAX_REPLIES_TOTAL",
    "REDDIT_CONCURRENCY",
    "TraversalRow",
    "TraversalSeed",
    "traverse_depth1",
]
