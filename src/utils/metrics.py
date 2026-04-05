"""
src/utils/metrics.py
======================
Evaluation metrics for all model types.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve,
)
from typing import Dict, Tuple, Optional
import warnings


def anomaly_detection_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
) -> Dict[str, float]:
    """
    Full evaluation suite for anomaly detection.

    Args:
        y_true:           Binary ground truth (1 = anomaly/failure)
        anomaly_scores:   Continuous anomaly scores (higher = more anomalous)
        threshold:        Fixed threshold. If None, uses percentile on normal samples.
        threshold_percentile: Percentile of scores on NORMAL samples for threshold.

    Returns:
        dict with AUC-ROC, AUC-PR, F1, precision, recall, confusion matrix values
    """
    if threshold is None:
        normal_scores = anomaly_scores[y_true == 0]
        threshold = np.percentile(normal_scores, threshold_percentile)

    y_pred = (anomaly_scores > threshold).astype(int)

    # Handle edge case: all predicted same class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_roc = roc_auc_score(y_true, anomaly_scores) if len(np.unique(y_true)) > 1 else 0.5
        auc_pr = average_precision_score(y_true, anomaly_scores) if len(np.unique(y_true)) > 1 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "false_alarm_rate": float(fp / (fp + tn + 1e-8)),
        "detection_rate": float(tp / (tp + fn + 1e-8)),
    }


def rul_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluation metrics for RUL (Remaining Useful Life) prediction.

    Includes RMSE, MAE, and the NASA CMAPSS scoring function.
    """
    diff = y_pred - y_true
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae = float(np.abs(diff).mean())
    # NASA scoring: asymmetric exponential (penalises late predictions more)
    nasa_score = float(
        np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1).sum()
    )
    # R² coefficient
    ss_res = (diff ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return {
        "rmse": rmse,
        "mae": mae,
        "nasa_score": nasa_score,
        "r2": r2,
        "mean_error": float(diff.mean()),
        "std_error": float(diff.std()),
    }


def early_detection_lead_time(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average lead time (how many timesteps before failure the alarm fires).

    Args:
        y_true:          Binary labels (1 = failure)
        anomaly_scores:  Per-timestep anomaly scores
        threshold:       Alarm threshold
        timestamps:      Optional array of actual timestamps/indices

    Returns:
        dict with mean/median lead time stats
    """
    if timestamps is None:
        timestamps = np.arange(len(y_true))

    alarm_flags = anomaly_scores > threshold
    failure_times = timestamps[y_true == 1]

    if len(failure_times) == 0:
        return {"mean_lead_time": np.nan, "median_lead_time": np.nan, "n_detected": 0}

    lead_times = []
    for t_fail in failure_times:
        # Find first alarm before this failure
        pre_alarm_times = timestamps[(alarm_flags) & (timestamps < t_fail)]
        if len(pre_alarm_times) > 0:
            lead_times.append(float(t_fail - pre_alarm_times[-1]))

    return {
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else 0.0,
        "median_lead_time": float(np.median(lead_times)) if lead_times else 0.0,
        "n_detected": len(lead_times),
        "n_failures": len(failure_times),
        "detection_rate": len(lead_times) / len(failure_times),
    }


def print_metrics_table(metrics: Dict[str, Dict[str, float]]):
    """Pretty-print a dict of model_name → metrics dict."""
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Recall':>8}")
    print("-" * 70)
    for model_name, m in metrics.items():
        print(
            f"{model_name:<30} {m.get('auc_roc', 0):>8.4f} "
            f"{m.get('auc_pr', 0):>8.4f} {m.get('f1', 0):>8.4f} "
            f"{m.get('recall', 0):>8.4f}"
        )
    print("=" * 70)
