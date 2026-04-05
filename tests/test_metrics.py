"""
tests/test_metrics.py
======================
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import (
    anomaly_detection_metrics,
    rul_metrics,
    early_detection_lead_time,
    print_metrics_table,
)


class TestAnomalyDetectionMetrics:
    """Test anomaly detection metric computation."""

    def test_perfect_classifier(self):
        """Perfect classifier should have AUC-ROC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.8, 0.9, 0.7, 0.9, 0.85])

        metrics = anomaly_detection_metrics(y_true, scores)
        assert metrics["auc_roc"] == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier(self):
        """Random scores should give AUC-ROC ≈ 0.5."""
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        np.random.seed(42)
        scores = np.random.rand(100)

        metrics = anomaly_detection_metrics(y_true, scores)
        assert abs(metrics["auc_roc"] - 0.5) < 0.15

    def test_confusion_matrix_values(self):
        """Test with known predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.15, 0.6])
        threshold = 0.5

        metrics = anomaly_detection_metrics(y_true, scores, threshold=threshold)
        # scores > 0.5: indices 2(0.9,T), 3(0.8,T), 5(0.7,T), 7(0.6,T) → y_true all 1 → 4 TP
        # scores < 0.5: indices 0(0.1,F), 1(0.2,F), 4(0.3,F), 6(0.15,F) → y_true all 0 → 4 TN
        assert metrics["tp"] == 4
        assert metrics["tn"] == 4
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_threshold_percentile(self):
        """Default threshold should be at specified percentile of normal scores."""
        y_true = np.concatenate([np.zeros(100), np.ones(20)])
        np.random.seed(42)
        normal_scores = np.random.beta(2, 10, 100)  # low scores for normal
        anomaly_scores = np.random.beta(10, 2, 20)  # high scores for anomalies
        scores = np.concatenate([normal_scores, anomaly_scores])

        metrics = anomaly_detection_metrics(y_true, scores, threshold_percentile=95)
        assert 0 < metrics["threshold"] < 1
        assert metrics["false_alarm_rate"] <= 0.06  # ~5% false positive rate

    def test_all_metrics_present(self):
        """All expected metric keys should be returned."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9])

        metrics = anomaly_detection_metrics(y_true, scores)
        expected_keys = {"auc_roc", "auc_pr", "f1", "precision", "recall",
                         "threshold", "tp", "fp", "tn", "fn",
                         "false_alarm_rate", "detection_rate"}
        assert set(metrics.keys()) == expected_keys


class TestRULMetrics:
    """Test RUL prediction metrics."""

    def test_perfect_prediction(self):
        """Perfect predictions should give RMSE = 0."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])

        metrics = rul_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_constant_error(self):
        """Constant error should give expected RMSE."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 22, 32])  # +2 each

        metrics = rul_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(2.0)
        assert metrics["mae"] == pytest.approx(2.0)

    def test_nasa_score_asymmetric(self):
        """NASA score should penalize over-prediction more."""
        # Same absolute error, different NASA score
        y_true1 = np.array([50])
        y_pred1 = np.array([70])  # over-prediction by 20
        y_pred2 = np.array([30])  # under-prediction by 20

        metrics1 = rul_metrics(y_true1, y_pred1)
        metrics2 = rul_metrics(y_true1, y_pred2)

        # Over-prediction should have higher penalty
        assert metrics1["nasa_score"] > metrics2["nasa_score"]

    def test_r2_negative(self):
        """Very bad predictions can give negative R²."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([100, 100, 100, 100, 100])

        metrics = rul_metrics(y_true, y_pred)
        assert metrics["r2"] < 0


class TestEarlyDetectionLeadTime:
    """Test lead time computation."""

    def test_early_detection(self):
        """Alarm before failure should give positive lead time."""
        y_true = np.array([0, 0, 0, 1, 0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.7, 0.9, 0.1, 0.1, 0.8, 0.9])
        threshold = 0.5

        result = early_detection_lead_time(y_true, scores, threshold)
        assert result["n_detected"] >= 1
        assert result["mean_lead_time"] > 0

    def test_no_detection(self):
        """No alarms before failure should give zero detection."""
        y_true = np.array([0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.1, 0.3])
        threshold = 0.5

        result = early_detection_lead_time(y_true, scores, threshold)
        assert result["n_detected"] == 0
        assert result["detection_rate"] == 0.0

    def test_no_failures(self):
        """No failures in data should return NaN lead time."""
        y_true = np.zeros(10)
        scores = np.random.rand(10)
        threshold = 0.5

        result = early_detection_lead_time(y_true, scores, threshold)
        assert result["n_detected"] == 0
        assert np.isnan(result["mean_lead_time"])


class TestPrintMetricsTable:
    """Test metrics table printing (smoke test)."""

    def test_print_does_not_crash(self, capsys):
        """Printing should not raise exceptions."""
        metrics = {
            "Model A": {"auc_roc": 0.94, "auc_pr": 0.80, "f1": 0.81, "recall": 0.79},
            "Model B": {"auc_roc": 0.96, "auc_pr": 0.85, "f1": 0.84, "recall": 0.82},
        }
        print_metrics_table(metrics)
        captured = capsys.readouterr()
        assert "Model A" in captured.out
        assert "Model B" in captured.out
