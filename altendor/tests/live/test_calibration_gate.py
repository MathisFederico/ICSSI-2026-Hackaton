"""Live calibration-gate test for the S10 classifier (stage S12).

Excluded from the default test invocation by the ``live`` marker (see root
``pyproject.toml``: ``addopts = "--durations=10 -m 'not live'"``). Run
explicitly via ``uv run pytest -m live altendor/tests/live/`` once
``ANTHROPIC_API_KEY`` is in env.

The test runs the real classifier across the 10-row calibration set and
fails the build if either gate slips: MAE > 8 dB on the endorsement subset
or macro-F1 < 0.8 across the three kinds. The intent is to halt the
pipeline whenever the classifier prompt drifts past calibration.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import pytest
from altendor.classify.calibration import (
    load_labeled_jsonl,
    render_report_markdown,
    score,
)
from altendor.classify.classifier import PaperCtx, classify_post
from altendor.enrich.text_resolver import ResolvedPost

LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration" / "labeled.jsonl"


@pytest.mark.live
def test_calibration_gate_passes() -> None:
    """Real-classifier gate. Requires ``ANTHROPIC_API_KEY`` in env.

    Asserts MAE <= 8 dB on the endorsement subset and macro-F1 >= 0.8 across
    the three kinds. Halts the pipeline if classifier prompt drifts.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    examples = load_labeled_jsonl(LABELS_PATH)
    client = anthropic.Anthropic()
    pairs = []
    for ex in examples:
        post = ResolvedPost(
            post_id="calib",
            platform="other",
            text=ex.post_text,
            author_handle=None,
            author_id=None,
            url="",
            created_at="",
            raw_title=ex.post_text,
            text_confidence="high",
        )
        paper = PaperCtx(title=ex.paper_title, abstract=ex.paper_abstract or None)
        result = classify_post(client, post, paper)
        pairs.append((ex, result))

    report = score(pairs)
    print(render_report_markdown(report))  # captured by pytest -s
    assert report.passes, (
        f"calibration failed: MAE={report.mae_dB}, F1={report.kind_f1_macro}\n"
        f"{render_report_markdown(report)}"
    )
