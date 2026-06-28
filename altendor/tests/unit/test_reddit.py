"""Unit tests for :mod:`altendor.enrich.reddit`.

PRAW is not exercised against the live API — these tests use a tiny
``FakePraw`` duck-typed stub that mirrors the attributes the production
code reads (``id``, ``selftext``, ``title``, ``author``, ``created_utc``,
``permalink``, ``comments``, ``replies``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, cast

import pytest
from altendor.enrich.reddit import (
    RedditNode,
    _submission_to_node,
    _walk_from_parent,
    get_reddit_client,
    get_replies,
    resolve_node,
)

if TYPE_CHECKING:
    import praw

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Fake PRAW objects — pure duck-typing, no inheritance from praw classes.
# ---------------------------------------------------------------------------


@dataclass
class FakeAuthor:
    """Mimic ``praw.models.Redditor`` (only the attributes we use)."""

    name: str


@dataclass
class FakeCommentForest:
    """Mimic ``praw.models.comment_forest.CommentForest``.

    Iterates the wrapped list; ``replace_more`` is a no-op so tests can run
    fully offline.
    """

    items: list[FakeComment] = field(default_factory=list)

    def replace_more(self, limit: int = 0) -> None:  # noqa: ANN001 - PRAW signature
        return None

    def __iter__(self) -> Iterator[FakeComment]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class FakeComment:
    """Mimic ``praw.models.Comment``."""

    id: str
    body: str
    author: FakeAuthor | None
    created_utc: float
    permalink: str
    replies: FakeCommentForest = field(default_factory=FakeCommentForest)


@dataclass
class FakeSubmission:
    """Mimic ``praw.models.Submission``."""

    id: str
    selftext: str
    title: str
    author: FakeAuthor | None
    created_utc: float
    permalink: str
    comments: FakeCommentForest = field(default_factory=FakeCommentForest)


class FakePraw:
    """Mimic ``praw.Reddit``. Holds a registry of submissions/comments
    keyed by URL and returns them on lookup."""

    def __init__(
        self,
        *,
        submissions: dict[str, FakeSubmission] | None = None,
        comments: dict[str, FakeComment] | None = None,
    ) -> None:
        self._submissions = submissions or {}
        self._comments = comments or {}
        self.read_only = True

    def submission(self, *, url: str) -> FakeSubmission:
        return self._submissions[url]

    def comment(self, *, url: str) -> FakeComment:
        return self._comments[url]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _author_from_json(value: str | None) -> FakeAuthor | None:
    return FakeAuthor(name=value) if value is not None else None


def _comment_from_json(blob: dict[str, Any]) -> FakeComment:
    return FakeComment(
        id=blob["id"],
        body=blob["body"],
        author=_author_from_json(blob.get("author")),
        created_utc=float(blob["created_utc"]),
        permalink=blob["permalink"],
        replies=FakeCommentForest(items=[_comment_from_json(c) for c in blob.get("replies", [])]),
    )


def _submission_from_json(blob: dict[str, Any]) -> FakeSubmission:
    return FakeSubmission(
        id=blob["id"],
        selftext=blob.get("selftext", ""),
        title=blob.get("title", ""),
        author=_author_from_json(blob.get("author")),
        created_utc=float(blob["created_utc"]),
        permalink=blob["permalink"],
        comments=FakeCommentForest(items=[_comment_from_json(c) for c in blob.get("comments", [])]),
    )


@pytest.fixture
def submission_blob() -> dict[str, Any]:
    with (FIXTURES / "reddit_submission.json").open() as fh:
        return json.load(fh)


@pytest.fixture
def submission(submission_blob: dict[str, Any]) -> FakeSubmission:
    return _submission_from_json(submission_blob)


@pytest.fixture
def submission_url(submission: FakeSubmission) -> str:
    return "https://www.reddit.com" + submission.permalink


@pytest.fixture
def fake_client(submission: FakeSubmission, submission_url: str) -> FakePraw:
    return FakePraw(submissions={submission_url: submission})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_submission_uses_selftext(
    fake_client: FakePraw, submission_url: str, submission: FakeSubmission
) -> None:
    node = resolve_node(submission_url, client=cast("praw.Reddit", fake_client))
    assert node is not None
    assert node.kind == "submission"
    assert node.id == f"t3_{submission.id}"
    assert node.body == "Body of the post."
    assert node.author == "user_name"
    assert node.permalink.startswith("https://www.reddit.com/r/Scientometrics/")


def test_resolve_submission_falls_back_to_title_when_selftext_empty() -> None:
    sub = FakeSubmission(
        id="emptyself",
        selftext="",
        title="Only the title survives",
        author=FakeAuthor(name="bob"),
        created_utc=1700000000.0,
        permalink="/r/Scientometrics/comments/emptyself/only_title/",
    )
    url = "https://www.reddit.com" + sub.permalink
    client = FakePraw(submissions={url: sub})

    node = resolve_node(url, client=cast("praw.Reddit", client))
    assert node is not None
    assert node.body == "Only the title survives"


def test_get_replies_depth_1(submission: FakeSubmission) -> None:
    parent_node = _submission_to_node(submission)
    # Drive the walker directly with the fixture parent (bypasses client lookup).
    replies = _walk_from_parent(submission, depth=1, max_per_parent=20, parent_is_submission=True)

    assert parent_node.kind == "submission"
    assert len(replies) == 2
    assert {r.kind for r in replies} == {"comment"}
    assert {r.id for r in replies} == {"t1_cmt1", "t1_cmt2"}


def test_deleted_author_normalized_to_string(submission: FakeSubmission) -> None:
    replies = _walk_from_parent(submission, depth=1, max_per_parent=20, parent_is_submission=True)
    # cmt2 has author=None in the fixture.
    cmt2 = next(r for r in replies if r.id == "t1_cmt2")
    assert cmt2.author == "[deleted]"
    # And the non-deleted one is unchanged.
    cmt1 = next(r for r in replies if r.id == "t1_cmt1")
    assert cmt1.author == "alice"


def test_max_per_parent_caps_results() -> None:
    big_sub = FakeSubmission(
        id="big",
        selftext="lots",
        title="big thread",
        author=FakeAuthor(name="op"),
        created_utc=1700000000.0,
        permalink="/r/Scientometrics/comments/big/big_thread/",
        comments=FakeCommentForest(
            items=[
                FakeComment(
                    id=f"c{i}",
                    body=f"reply {i}",
                    author=FakeAuthor(name=f"u{i}"),
                    created_utc=1700000000.0 + i,
                    permalink=f"/r/Scientometrics/comments/big/big_thread/c{i}/",
                )
                for i in range(30)
            ]
        ),
    )

    replies = _walk_from_parent(big_sub, depth=1, max_per_parent=5, parent_is_submission=True)
    assert len(replies) == 5
    assert [r.id for r in replies] == [f"t1_c{i}" for i in range(5)]


def test_get_reddit_client_returns_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    assert get_reddit_client() is None

    # Partial creds also yield None — all three are required.
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    assert get_reddit_client() is None


def test_get_replies_uses_client_to_rehydrate_parent(
    fake_client: FakePraw, submission: FakeSubmission, submission_url: str
) -> None:
    """End-to-end through ``get_replies``: passes a RedditNode and expects the
    fake client's ``.submission(url=...)`` lookup to succeed."""
    # Build a node whose permalink matches the URL the fake client was keyed by.
    node = RedditNode(
        id=f"t3_{submission.id}",
        kind="submission",
        author="user_name",
        body="Body of the post.",
        created_utc=1700000000.0,
        permalink=submission_url,
    )
    replies = get_replies(node, depth=1, max_per_parent=20, client=cast("praw.Reddit", fake_client))
    assert len(replies) == 2
    assert {r.id for r in replies} == {"t1_cmt1", "t1_cmt2"}
