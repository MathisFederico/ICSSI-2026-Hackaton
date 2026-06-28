"""Unit tests for :mod:`altendor.classify.prompts`.

These tests pin the cache-stability contract for the classifier system
prompt (S9). The prompt is consumed by S10 with prompt caching enabled,
so byte-stability and deterministic exemplar ordering are non-negotiable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from altendor.classify.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    DEFAULT_CALIBRATION_PATH,
    _build_system_prompt,
    build_exemplars_block,
    build_user_message,
)


def _load_calibration_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_system_prompt_is_stable() -> None:
    """Two builds of the system prompt must be byte-identical."""
    first = _build_system_prompt()
    second = _build_system_prompt()
    assert first == second
    # And the module-level constant matches a fresh build.
    assert CLASSIFIER_SYSTEM_PROMPT == first


def test_exemplars_block_sorts_deterministically(tmp_path: Path) -> None:
    """Shuffling the JSONL must not change the rendered block."""
    rows = _load_calibration_rows(DEFAULT_CALIBRATION_PATH)
    assert len(rows) >= 2

    rng = random.Random(20260628)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    # Guard the test itself: confirm the shuffle actually permuted the rows.
    assert shuffled != rows, "shuffle was a no-op; pick a different seed"

    shuffled_path = tmp_path / "labeled_shuffled.jsonl"
    with shuffled_path.open("w", encoding="utf-8") as fh:
        for row in shuffled:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    baseline = build_exemplars_block()
    shuffled_block = build_exemplars_block(shuffled_path)
    assert baseline == shuffled_block


def test_build_user_message_handles_missing_url_and_abstract() -> None:
    """When optional fields are None we substitute placeholders instead of `None`."""
    msg = build_user_message(
        post_text="Post body.",
        post_url=None,
        paper_title="Paper",
        paper_abstract=None,
    )
    assert "None" not in msg
    assert "no URL available" in msg
    assert "no abstract available" in msg


def test_build_user_message_low_confidence_hint() -> None:
    """A `text_confidence="low"` argument must surface a conservative-routing hint."""
    high = build_user_message(
        post_text="Post body.",
        post_url=None,
        paper_title="Paper",
        paper_abstract="Abstract.",
        text_confidence="high",
    )
    low = build_user_message(
        post_text="Post body.",
        post_url=None,
        paper_title="Paper",
        paper_abstract="Abstract.",
        text_confidence="low",
    )
    assert "truncated" not in high
    assert "truncated" in low


