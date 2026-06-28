"""Offline unit tests for :mod:`altendor.classify.calibration` (S12).

No Anthropic API calls — :class:`ClassifyResult` instances are constructed
directly so we can exercise the metrics module in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from altendor.classify.calibration import (
    CalibrationReport,
    LabeledExample,
    load_labeled_jsonl,
    render_report_markdown,
    score,
)
from altendor.classify.schema import ClassifyResult, Endorsement, Flag, Irrelevant

_FlagCategory = Literal["methodological", "source", "data", "bias", "other"]
_Criterion = Literal["Support", "Prior"]

_REPO_LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration" / "labeled.jsonl"
_VALID_KINDS = {"endorsement", "flag", "irrelevant"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ex(
    *,
    kind: str = "endorsement",
    magnitude_dB: int | None = None,
    category: str | None = None,
    post_text: str = "post",
    paper_title: str = "T",
    paper_abstract: str = "A",
) -> LabeledExample:
    gold: dict[str, object] = {"kind": kind}
    if kind == "endorsement":
        gold["magnitude_dB"] = magnitude_dB if magnitude_dB is not None else 20
        gold["claim_text"] = "claim"
        gold["criterion"] = "Support"
    elif kind == "flag":
        gold["category"] = category or "methodological"
        gold["rationale"] = "rationale"
    elif kind == "irrelevant":
        gold["reason"] = "reason"
    return LabeledExample(
        post_text=post_text,
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        gold=gold,
    )


def _endorsement(magnitude_dB: int = 20, criterion: _Criterion = "Support") -> ClassifyResult:
    return Endorsement(
        claim_text="claim",
        magnitude_dB=magnitude_dB,
        criterion=criterion,
        reasoning="because.",
    )


def _flag(category: _FlagCategory = "methodological") -> ClassifyResult:
    return Flag(category=category, rationale="rationale")


def _irrelevant() -> ClassifyResult:
    return Irrelevant(reason="vague")


# ---------------------------------------------------------------------------
# load_labeled_jsonl
# ---------------------------------------------------------------------------


def test_load_labeled_jsonl_parses_seed_file() -> None:
    examples = load_labeled_jsonl(_REPO_LABELS_PATH)
    assert len(examples) == 10
    for ex in examples:
        assert ex.gold.get("kind") in _VALID_KINDS


def test_load_labeled_jsonl_raises_on_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"post_text": "ok", "paper_title": "t", "paper_abstract": "a", "gold": {"kind": "irrelevant", "reason": "x"}}\n'
        "this is not json\n",
        encoding="utf-8",
    )
    with pytest.raises(json.JSONDecodeError):
        load_labeled_jsonl(bad)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_perfect_match() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        (_ex(kind="endorsement", magnitude_dB=20), _endorsement(20)),
        (_ex(kind="flag", category="methodological"), _flag("methodological")),
        (_ex(kind="irrelevant"), _irrelevant()),
    ]
    report = score(pairs)
    assert report.kind_f1_macro == pytest.approx(1.0)
    assert report.mae_dB == pytest.approx(0.0)
    assert report.passes is True


def test_score_all_wrong_kind() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        (_ex(kind="endorsement", magnitude_dB=15), _irrelevant()),
        (_ex(kind="flag"), _irrelevant()),
        (_ex(kind="irrelevant"), _irrelevant()),
    ]
    report = score(pairs)
    # Only irrelevant has any TP; endorsement & flag F1 are 0; irrelevant has FP > 0.
    assert report.kind_f1_macro < 0.8
    assert report.passes is False


def test_score_mae_computation() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        (_ex(kind="endorsement", magnitude_dB=20), _endorsement(22)),
        (_ex(kind="endorsement", magnitude_dB=-10), _endorsement(-8)),
    ]
    report = score(pairs)
    assert report.mae_dB == pytest.approx(2.0)
    assert report.n_endorsements_both == 2


def test_score_mae_only_over_both_endorsement_subset() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        # Match: counted in MAE.
        (_ex(kind="endorsement", magnitude_dB=20), _endorsement(20)),
        # Gold endorsement but predicted flag — excluded from MAE, kept in F1.
        (_ex(kind="endorsement", magnitude_dB=15), _flag("methodological")),
    ]
    report = score(pairs)
    assert report.n_endorsements_both == 1
    assert report.mae_dB == pytest.approx(0.0)
    # Confusion still records the endorsement->flag mistake.
    assert report.confusion["endorsement"]["flag"] == 1


def test_score_empty_endorsement_subset_pass_trivially() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        (_ex(kind="flag", category="bias"), _flag("bias")),
        (_ex(kind="irrelevant"), _irrelevant()),
    ]
    report = score(pairs)
    assert report.mae_dB is None
    assert report.n_endorsements_both == 0
    # MAE side passes trivially when no data; F1 here is high so overall passes.
    assert report.passes is True


def test_confusion_matrix_shape() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        (_ex(kind="endorsement", magnitude_dB=20), _endorsement(20)),
        (_ex(kind="flag"), _flag()),
        (_ex(kind="irrelevant"), _irrelevant()),
    ]
    report = score(pairs)
    assert set(report.confusion.keys()) == _VALID_KINDS
    for k in _VALID_KINDS:
        assert set(report.confusion[k].keys()) == _VALID_KINDS


def test_passes_requires_both_thresholds() -> None:
    # Case 1: perfect MAE but F1 below threshold.
    weak_f1 = CalibrationReport(
        n_total=10,
        n_endorsements_both=2,
        mae_dB=0.0,
        kind_f1_macro=0.7,
        per_kind_f1={"endorsement": 0.7, "flag": 0.7, "irrelevant": 0.7},
        confusion={k: {kk: 0 for kk in _VALID_KINDS} for k in _VALID_KINDS},
    )
    assert weak_f1.passes is False

    # Case 2: F1 fine but MAE too high.
    weak_mae = CalibrationReport(
        n_total=10,
        n_endorsements_both=2,
        mae_dB=12.0,
        kind_f1_macro=0.85,
        per_kind_f1={"endorsement": 0.85, "flag": 0.85, "irrelevant": 0.85},
        confusion={k: {kk: 0 for kk in _VALID_KINDS} for k in _VALID_KINDS},
    )
    assert weak_mae.passes is False

    # Sanity: both at threshold passes.
    ok = CalibrationReport(
        n_total=10,
        n_endorsements_both=2,
        mae_dB=8.0,
        kind_f1_macro=0.8,
        per_kind_f1={"endorsement": 0.8, "flag": 0.8, "irrelevant": 0.8},
        confusion={k: {kk: 0 for kk in _VALID_KINDS} for k in _VALID_KINDS},
    )
    assert ok.passes is True


def test_render_report_markdown_includes_metrics() -> None:
    pairs: list[tuple[LabeledExample, ClassifyResult]] = [
        (_ex(kind="endorsement", magnitude_dB=20), _endorsement(22)),
        (_ex(kind="flag"), _flag()),
        (_ex(kind="irrelevant"), _irrelevant()),
    ]
    report = score(pairs)
    md = render_report_markdown(report)
    assert "MAE" in md
    assert "F1" in md
    assert "Confusion Matrix" in md
    # Confusion-matrix headers (the kind names should appear in the header row).
    assert "endorsement" in md
    assert "flag" in md
    assert "irrelevant" in md
