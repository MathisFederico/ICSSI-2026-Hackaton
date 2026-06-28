"""Reddit text + reply resolution via PRAW (stage S7).

Resolves a Reddit submission or comment URL to a normalized
:class:`RedditNode` and walks depth-1 (optionally deeper) reply trees.

Auth uses a *script-type* Reddit app via env vars:

* ``REDDIT_CLIENT_ID``
* ``REDDIT_CLIENT_SECRET``
* ``REDDIT_USER_AGENT``

If any of these is missing, :func:`get_reddit_client` returns ``None`` so
callers can choose to fall back rather than crash. All PRAW/prawcore
network errors are caught and logged at ``WARNING``; the functions then
return ``None`` (for :func:`resolve_node`) or ``[]`` (for
:func:`get_replies`).

Internal helpers take ``object`` parameters and access attributes by
name. PRAW models are not type-stub-annotated in this repo, and the same
helpers are exercised by lightweight test doubles — duck typing keeps
both paths honest without inventing a Protocol that pretends to match
PRAW's real signatures.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    import praw

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
DELETED_AUTHOR = "[deleted]"


@dataclass(frozen=True)
class RedditNode:
    """Normalized Reddit node (submission or comment).

    ``id`` is prefixed: ``t3_`` for submissions, ``t1_`` for comments.
    ``author`` is the string ``"[deleted]"`` (not ``None``) when the
    account or content has been removed. ``permalink`` is an absolute URL.
    """

    id: str
    kind: Literal["submission", "comment"]
    author: str
    body: str
    created_utc: float
    permalink: str


def _author_name(author: object) -> str:
    """Return the author name string, mapping ``None``/missing to ``"[deleted]"``.

    PRAW exposes ``submission.author`` as a ``Redditor`` (with ``.name``) or
    ``None`` for removed/deleted accounts.
    """
    if author is None:
        return DELETED_AUTHOR
    name = getattr(author, "name", None)
    if not isinstance(name, str) or not name:
        return DELETED_AUTHOR
    return name


def _absolute_permalink(permalink: str) -> str:
    """Prefix a Reddit-relative permalink with the canonical base URL."""
    if not permalink:
        return ""
    if permalink.startswith("http://") or permalink.startswith("https://"):
        return permalink
    if not permalink.startswith("/"):
        permalink = "/" + permalink
    return REDDIT_BASE + permalink


def _is_comment_url(url: str) -> bool:
    """Heuristic: a Reddit comment permalink has a trailing comment id segment.

    Submission: ``/r/<sub>/comments/<id36>/<slug>/`` (5 segments).
    Comment:    ``/r/<sub>/comments/<id36>/<slug>/<comment_id36>/`` (6+).
    """
    path = urlparse(url).path.strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 6 and parts[0] == "r" and parts[2] == "comments":
        return True
    return False


def get_reddit_client() -> praw.Reddit | None:
    """Build a read-only PRAW client from env vars, or return ``None``.

    Returns ``None`` (without raising) when any of the three required env
    vars is missing or empty. This lets pipeline stages fall back to a
    cached or html-scraped path instead of crashing on absent creds.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")

    if not client_id or not client_secret or not user_agent:
        logger.info("Reddit credentials missing; get_reddit_client() returning None")
        return None

    try:
        import praw  # local import — keeps module import cheap and ty-clean
    except ImportError:
        logger.warning("praw is not installed; cannot build Reddit client")
        return None

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_updates=False,
        )
        reddit.read_only = True
        return reddit
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to build PRAW client: %s", exc)
        return None


def _submission_to_node(submission: object) -> RedditNode:
    """Convert a PRAW (or fixture) submission object to a :class:`RedditNode`.

    Reads attributes by name so this function works for both ``praw.models.
    Submission`` instances and lightweight test doubles.
    """
    selftext = cast(str, getattr(submission, "selftext", "") or "")
    title = cast(str, getattr(submission, "title", "") or "")
    body = selftext or title
    return RedditNode(
        id=f"t3_{getattr(submission, 'id')}",
        kind="submission",
        author=_author_name(getattr(submission, "author", None)),
        body=body,
        created_utc=float(getattr(submission, "created_utc", 0.0)),
        permalink=_absolute_permalink(cast(str, getattr(submission, "permalink", ""))),
    )


def _comment_to_node(comment: object) -> RedditNode:
    """Convert a PRAW (or fixture) comment object to a :class:`RedditNode`."""
    return RedditNode(
        id=f"t1_{getattr(comment, 'id')}",
        kind="comment",
        author=_author_name(getattr(comment, "author", None)),
        body=cast(str, getattr(comment, "body", "") or ""),
        created_utc=float(getattr(comment, "created_utc", 0.0)),
        permalink=_absolute_permalink(cast(str, getattr(comment, "permalink", ""))),
    )


def resolve_node(url: str, *, client: praw.Reddit | None = None) -> RedditNode | None:
    """Resolve a Reddit URL to a :class:`RedditNode`.

    Submission URLs use ``client.submission(url=url)`` and set the body to
    ``selftext or title`` (link-posts have empty selftext). Comment URLs
    use ``client.comment(url=url)`` and copy ``comment.body``.

    Returns ``None`` when *client* is ``None`` or when PRAW raises any
    ``PRAWException``/``prawcore`` error.
    """
    if client is None:
        client = get_reddit_client()
        if client is None:
            return None

    try:
        if _is_comment_url(url):
            comment = client.comment(url=url)
            _maybe_refresh(comment)
            return _comment_to_node(comment)
        submission = client.submission(url=url)
        return _submission_to_node(submission)
    except Exception as exc:
        # praw.exceptions.PRAWException and prawcore.exceptions.* live in
        # optional dep packages; catch broadly and log once at WARNING.
        logger.warning("resolve_node(%r) failed: %s", url, exc)
        return None


def _maybe_refresh(obj: object) -> None:
    """Best-effort fetch — PRAW comments are lazy until accessed."""
    refresh = getattr(obj, "refresh", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:  # pragma: no cover - tolerate stale/missing
            pass


def _replace_more_safe(forest: object) -> None:
    """Call ``replace_more(limit=0)`` on a comment forest, ignoring failures."""
    if forest is None:
        return
    fn = getattr(forest, "replace_more", None)
    if callable(fn):
        try:
            fn(limit=0)
        except Exception:  # pragma: no cover - tolerate fixtures w/o network
            pass


def _children_of_submission(submission: object) -> list[object]:
    """Return top-level comments of a submission (after ``replace_more``)."""
    forest = getattr(submission, "comments", None)
    if forest is None:
        return []
    _replace_more_safe(forest)
    return list(forest)


def _children_of_comment(comment: object) -> list[object]:
    """Return direct replies of a comment (after ``replace_more``)."""
    replies = getattr(comment, "replies", None)
    if replies is None:
        return []
    _replace_more_safe(replies)
    return list(replies)


def _walk(
    parent: object,
    *,
    depth: int,
    max_per_parent: int,
    parent_is_submission: bool,
    acc: list[RedditNode],
) -> None:
    """Recursive helper for :func:`get_replies`.

    Walks at most ``depth`` levels deep, capping each parent's children at
    ``max_per_parent`` and appending normalized nodes to ``acc``.
    """
    if depth <= 0:
        return

    children: list[object]
    if parent_is_submission:
        children = _children_of_submission(parent)
    else:
        children = _children_of_comment(parent)

    for child in children[:max_per_parent]:
        node = _comment_to_node(child)
        acc.append(node)
        if depth > 1:
            _walk(
                child,
                depth=depth - 1,
                max_per_parent=max_per_parent,
                parent_is_submission=False,
                acc=acc,
            )


def get_replies(
    node: RedditNode,
    *,
    depth: int = 1,
    max_per_parent: int = 20,
    client: praw.Reddit | None = None,
) -> list[RedditNode]:
    """Fetch replies below *node*, depth-1 by default.

    For a submission, ``replace_more(limit=0)`` drops "load more" stubs and
    we return top-level comments (capped at *max_per_parent*). For a
    comment, the equivalent walks ``comment.replies``. Setting *depth* > 1
    recurses, capping each parent's child count at *max_per_parent*.

    Returns ``[]`` on missing client or any PRAW/prawcore error.
    """
    if depth < 1:
        return []

    if client is None:
        client = get_reddit_client()
        if client is None:
            return []

    try:
        # Re-hydrate the parent from PRAW so we can walk its forest. Use the
        # absolute permalink URL since that is what we always store.
        parent: object
        if node.kind == "submission":
            parent = client.submission(url=node.permalink)
            is_submission = True
        else:
            parent = client.comment(url=node.permalink)
            _maybe_refresh(parent)
            is_submission = False

        out: list[RedditNode] = []
        _walk(
            parent,
            depth=depth,
            max_per_parent=max_per_parent,
            parent_is_submission=is_submission,
            acc=out,
        )
        return out
    except Exception as exc:
        logger.warning("get_replies(%s) failed: %s", node.id, exc)
        return []


# ---------------------------------------------------------------------------
# Test helpers (exposed so the unit tests can drive the walker directly).
# ---------------------------------------------------------------------------


def _walk_from_parent(
    parent: object,
    *,
    depth: int = 1,
    max_per_parent: int = 20,
    parent_is_submission: bool = True,
) -> list[RedditNode]:
    """Public-for-tests entry point: walk a fixture-built parent directly.

    Bypasses the PRAW client step so tests can pass a duck-typed object
    with ``.comments``/``.replies`` already populated.
    """
    out: list[RedditNode] = []
    _walk(
        parent,
        depth=depth,
        max_per_parent=max_per_parent,
        parent_is_submission=parent_is_submission,
        acc=out,
    )
    return out
