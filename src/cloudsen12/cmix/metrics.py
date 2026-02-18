"""CMIX evaluation metrics.

Unlike the CloudSEN12 evaluation (patch-level BOA with median aggregation),
CMIX aggregates all labeled pixels across all 29 scenes into a single global
confusion matrix and derives metrics from it.

Metrics reported
----------------
OA  : Overall Accuracy          = (TP + TN) / N
BOA : Balanced Overall Accuracy = 0.5 * (PA + TN_rate)
PA  : Producer's Accuracy       = TP / (TP + FN)  [recall / sensitivity]
UA  : User's Accuracy           = TP / (TP + FP)  [precision / PPV]
"""

from typing import Dict, NamedTuple

import numpy as np


class BinaryConfusion(NamedTuple):
    """Counts from a 2x2 binary confusion matrix."""

    tp: int
    fn: int
    fp: int
    tn: int


def _safe_div(numerator: float, denominator: float) -> float:
    """Return NaN when denominator is zero."""
    return float("nan") if denominator == 0 else numerator / denominator


def compute_binary_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> BinaryConfusion:
    """Compute TP, FN, FP, TN from binary 0/1 arrays.

    Args:
        y_true: Reference binary labels (1=positive, 0=negative).
        y_pred: Predicted binary labels (1=positive, 0=negative).

    Returns:
        BinaryConfusion namedtuple.
    """
    y_true = y_true.ravel().astype(bool)
    y_pred = y_pred.ravel().astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    return BinaryConfusion(tp=tp, fn=fn, fp=fp, tn=tn)


def compute_cmix_metrics(confusion: BinaryConfusion) -> Dict[str, float]:
    """Compute OA, BOA, PA, and UA from binary confusion counts.

    Args:
        confusion: BinaryConfusion namedtuple.

    Returns:
        Dictionary with keys 'OA', 'BOA', 'PA', 'UA'.
    """
    tp, fn, fp, tn = confusion.tp, confusion.fn, confusion.fp, confusion.tn
    n = tp + fn + fp + tn

    oa = _safe_div(tp + tn, n)
    pa = _safe_div(tp, tp + fn)
    ua = _safe_div(tp, tp + fp)
    tn_rate = _safe_div(tn, tn + fp)
    boa = 0.5 * (pa + tn_rate) if not (np.isnan(pa) or np.isnan(tn_rate)) else float("nan")

    return {"OA": oa, "BOA": boa, "PA": pa, "UA": ua}


def accumulate_global_confusion(
    confusions: list,
) -> BinaryConfusion:
    """Sum BinaryConfusion counts from multiple scenes into one.

    Args:
        confusions: List of BinaryConfusion namedtuples (one per scene).

    Returns:
        Aggregated BinaryConfusion.
    """
    tp = sum(c.tp for c in confusions)
    fn = sum(c.fn for c in confusions)
    fp = sum(c.fp for c in confusions)
    tn = sum(c.tn for c in confusions)
    return BinaryConfusion(tp=tp, fn=fn, fp=fp, tn=tn)


def format_cmix_results_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    """Format CMIX results as a plain-text comparison table.

    Args:
        results: Nested dict structured as:
            {model_name: {experiment_name: {metric: value}}}

    Returns:
        Formatted string ready for printing or saving.
    """
    experiments = list(next(iter(results.values())).keys())
    metrics = ["OA", "BOA", "PA", "UA"]

    lines = []
    sep = "=" * 80

    for exp in experiments:
        lines.append(f"\n{sep}")
        lines.append(f"Experiment: {exp}")
        lines.append(sep)

        header = f"{'Model':<30s}" + "".join(f"  {m:>8s}" for m in metrics)
        lines.append(header)
        lines.append("-" * 80)

        for model_name, exp_dict in results.items():
            row = f"{model_name:<30s}"
            for m in metrics:
                val = exp_dict[exp].get(m, float("nan"))
                row += f"  {val:>8.4f}" if not np.isnan(val) else f"  {'nan':>8s}"
            lines.append(row)

        lines.append(sep)

    return "\n".join(lines)