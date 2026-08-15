"""Dependency-light metrics and uncertainty utilities."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} != {pred.shape}")
    residual = pred - true
    absolute = np.abs(residual)
    mae = float(absolute.mean())
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(true - true.mean())))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator else 0.0
    target_sum = float(np.sum(np.abs(true)))
    pmae = float(np.sum(absolute) / target_sum) if target_sum else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "pmae": pmae}


def multilabel_metrics(
    y_true: np.ndarray, y_probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=bool)
    probability = np.asarray(y_probability, dtype=float)
    if true.shape != probability.shape:
        raise ValueError(f"Shape mismatch: {true.shape} != {probability.shape}")
    predicted = probability >= threshold

    tp = np.logical_and(predicted, true).sum(axis=0).astype(float)
    fp = np.logical_and(predicted, ~true).sum(axis=0).astype(float)
    fn = np.logical_and(~predicted, true).sum(axis=0).astype(float)

    precision_per_label = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_per_label = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_per_label = np.divide(
        2 * precision_per_label * recall_per_label,
        precision_per_label + recall_per_label,
        out=np.zeros_like(tp),
        where=(precision_per_label + recall_per_label) > 0,
    )

    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(precision_per_label.mean()),
        "macro_recall": float(recall_per_label.mean()),
        "macro_f1": float(f1_per_label.mean()),
    }


def conformal_absolute_error(
    y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.95
) -> np.ndarray:
    """Return per-target absolute residual quantiles for simple prediction intervals."""

    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("Expected equally shaped 2D arrays")
    residuals = np.abs(true - pred)
    try:
        return np.quantile(residuals, coverage, axis=0, method="higher")
    except TypeError:  # NumPy < 1.22 compatibility
        return np.quantile(residuals, coverage, axis=0, interpolation="higher")

