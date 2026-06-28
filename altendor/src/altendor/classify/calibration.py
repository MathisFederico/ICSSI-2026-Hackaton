"""Calibration metrics for the S10 post classifier (stage S12).

Pure utility module — **no Anthropic API calls live here**. Loads the labelled
JSONL anchor set, scores classifier predictions against gold labels, and
produces a markdown report used by the live calibration gate and the S21
notebook.

Two metrics gate the classifier:

* **Macro-F1** over the three kinds (``endorsement``/``flag``/``irrelevant``).
* **Mean absolute error in decibans** (``MAE``) over the subset of rows where
  both gold and prediction are endorsements. When that subset is empty the
  MAE side of the gate passes trivially (``mae_dB is None``).

The live test in ``altendor/tests/live/test_calibration_gate.py`` runs the
real classifier over the calibration set and asserts both thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from altendor.classify.schema import ClassifyResult, Endorsement, Flag, Irrelevant

_KINDS: tuple[str, str, str] = ("endorsement", "flag", "irrelevant")
"""Canonical kind order used for confusion-matrix rendering."""


@dataclass(frozen=True)
class LabeledExample:
    """One row of the calibration JSONL.

    ``gold`` is kept as a raw dict so the schema stays loose — gold labels
    may have variant-specific fields (``magnitude_dB``, ``category``,
    ``reason``) without us needing to mirror the discriminated-union schema
    on the gold side.
    """

    post_text: str
    paper_title: str
    paper_abstract: str
    gold: dict  # raw dict from JSONL


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregated calibration metrics produced by :func:`score`."""

    n_total: int
    n_endorsements_both: int
    mae_dB: float | None
    kind_f1_macro: float
    per_kind_f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    threshold_mae: float = 8.0
    threshold_f1: float = 0.8

    @property
    def passes(self) -> bool:
        """True iff (MAE is N/A or MAE <= threshold) AND macro-F1 >= threshold."""
        mae_ok = self.mae_dB is None or self.mae_dB <= self.threshold_mae
        f1_ok = self.kind_f1_macro >= self.threshold_f1
        return mae_ok and f1_ok


def load_labeled_jsonl(path: Path | str) -> list[LabeledExample]:
    """Parse a JSONL of labelled rows.

    Blank lines are skipped. Malformed JSON raises :class:`json.JSONDecodeError`
    — callers can let that propagate (the live test should fail loudly on a
    corrupted anchor file).
    """
    p = Path(path)
    rows: list[LabeledExample] = []
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                LabeledExample(
                    post_text=str(obj.get("post_text", "")),
                    paper_title=str(obj.get("paper_title", "")),
                    paper_abstract=str(obj.get("paper_abstract", "")),
                    gold=dict(obj.get("gold", {})),
                )
            )
    return rows


def gold_kind(example: LabeledExample) -> str:
    """Return the gold ``kind`` string for an example."""
    return str(example.gold.get("kind", ""))


def predict_to_kind(result: ClassifyResult) -> str:
    """Return the predicted ``kind`` string for a :class:`ClassifyResult`."""
    if isinstance(result, Endorsement):
        return "endorsement"
    if isinstance(result, Flag):
        return "flag"
    if isinstance(result, Irrelevant):
        return "irrelevant"
    # Shouldn't happen — the discriminated union is exhaustive.
    raise TypeError(f"Unknown ClassifyResult variant: {type(result).__name__}")


def gold_magnitude(example: LabeledExample) -> int | None:
    """Return the gold endorsement magnitude, or ``None`` for non-endorsements."""
    if gold_kind(example) != "endorsement":
        return None
    mag = example.gold.get("magnitude_dB")
    if mag is None:
        return None
    return int(mag)


def predict_magnitude(result: ClassifyResult) -> int | None:
    """Return the predicted endorsement magnitude, or ``None`` for non-endorsements."""
    if isinstance(result, Endorsement):
        return int(result.magnitude_dB)
    return None


def _empty_confusion() -> dict[str, dict[str, int]]:
    """Build a 3x3 confusion matrix initialised to zero."""
    return {true_kind: {pred_kind: 0 for pred_kind in _KINDS} for true_kind in _KINDS}


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from (TP, FP, FN) with the explicit-zero convention.

    If either precision or recall is zero (no positive predictions OR no
    positive labels), F1 is defined as 0.0 — never NaN.
    """
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass
class _Counts:
    """Per-kind TP/FP/FN bucket used while iterating the pair stream."""

    tp: int = 0
    fp: int = 0
    fn: int = 0


def score(
    pairs: Iterable[tuple[LabeledExample, ClassifyResult]],
    *,
    threshold_mae: float = 8.0,
    threshold_f1: float = 0.8,
) -> CalibrationReport:
    """Aggregate calibration metrics over (label, prediction) pairs.

    Computes:

    * The 3x3 confusion matrix keyed by canonical kind names.
    * Per-kind F1 (with the explicit ``F1=0`` convention when either side
      has no positives — see :func:`_f1_from_counts`).
    * Macro-F1 as the unweighted mean of the per-kind F1s.
    * MAE in decibans, restricted to the subset of rows where BOTH gold
      and prediction are endorsements. If that subset is empty, ``mae_dB``
      is set to ``None`` and the MAE side of the gate passes trivially.
    """
    counts: dict[str, _Counts] = {k: _Counts() for k in _KINDS}
    confusion = _empty_confusion()

    n_total = 0
    abs_errors: list[int] = []

    for example, result in pairs:
        n_total += 1
        true_k = gold_kind(example)
        pred_k = predict_to_kind(result)

        if true_k in confusion and pred_k in confusion[true_k]:
            confusion[true_k][pred_k] += 1

        for kind in _KINDS:
            if true_k == kind and pred_k == kind:
                counts[kind].tp += 1
            elif true_k != kind and pred_k == kind:
                counts[kind].fp += 1
            elif true_k == kind and pred_k != kind:
                counts[kind].fn += 1

        if true_k == "endorsement" and pred_k == "endorsement":
            g_mag = gold_magnitude(example)
            p_mag = predict_magnitude(result)
            if g_mag is not None and p_mag is not None:
                abs_errors.append(abs(g_mag - p_mag))

    per_kind_f1: dict[str, float] = {
        kind: _f1_from_counts(counts[kind].tp, counts[kind].fp, counts[kind].fn) for kind in _KINDS
    }
    # Macro-F1 averages over kinds that actually appear in either gold or
    # prediction. Kinds with zero presence on both axes contribute no signal
    # and would otherwise drag the average toward 0 unfairly (e.g. a label
    # set with only flag/irrelevant rows would otherwise score 2/3 max).
    present_kinds = [
        kind for kind in _KINDS if counts[kind].tp + counts[kind].fp + counts[kind].fn > 0
    ]
    if present_kinds:
        kind_f1_macro = sum(per_kind_f1[k] for k in present_kinds) / len(present_kinds)
    else:
        kind_f1_macro = 0.0

    n_endorsements_both = len(abs_errors)
    mae_dB: float | None = sum(abs_errors) / n_endorsements_both if n_endorsements_both > 0 else None

    return CalibrationReport(
        n_total=n_total,
        n_endorsements_both=n_endorsements_both,
        mae_dB=mae_dB,
        kind_f1_macro=kind_f1_macro,
        per_kind_f1=per_kind_f1,
        confusion=confusion,
        threshold_mae=threshold_mae,
        threshold_f1=threshold_f1,
    )


def render_report_markdown(report: CalibrationReport) -> str:
    """Render a :class:`CalibrationReport` as a markdown block.

    Format is stable enough for notebook 4 and CI logs: header summary,
    per-kind F1 table, then the 3x3 confusion matrix with gold rows and
    predicted columns.
    """
    if report.mae_dB is None:
        mae_line = f"- MAE (dB, endorsement subset): n/a (no overlapping endorsements; threshold {report.threshold_mae})"
    else:
        mae_line = (
            f"- MAE (dB, endorsement subset): {report.mae_dB:.2f}  "
            f"(threshold {report.threshold_mae}, n={report.n_endorsements_both})"
        )

    lines: list[str] = [
        "## Calibration Report",
        f"- n_total: {report.n_total}",
        mae_line,
        f"- Macro-F1: {report.kind_f1_macro:.3f}  (threshold {report.threshold_f1})",
        f"- Passes gate: {report.passes}",
        "",
        "### Per-kind F1",
        "| Kind | F1 |",
        "|------|-----|",
    ]
    for kind in _KINDS:
        lines.append(f"| {kind} | {report.per_kind_f1.get(kind, 0.0):.3f} |")

    lines.extend(
        [
            "",
            "### Confusion Matrix (rows = gold, cols = predicted)",
            "| gold \\ pred | " + " | ".join(_KINDS) + " |",
            "|" + "|".join(["---"] * (len(_KINDS) + 1)) + "|",
        ]
    )
    for true_kind in _KINDS:
        row_counts = [str(report.confusion.get(true_kind, {}).get(pred_kind, 0)) for pred_kind in _KINDS]
        lines.append(f"| {true_kind} | " + " | ".join(row_counts) + " |")

    return "\n".join(lines)


__all__ = [
    "CalibrationReport",
    "LabeledExample",
    "gold_kind",
    "gold_magnitude",
    "load_labeled_jsonl",
    "predict_magnitude",
    "predict_to_kind",
    "render_report_markdown",
    "score",
]
