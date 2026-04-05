"""
tests/test_preprocessor.py
============================
Unit tests for the preprocessor module.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessor import (
    fit_scaler,
    transform_data,
    winsorize,
    impute_missing,
    create_windows,
    split_data,
)


class TestFitScaler:
    """Test scaler fitting."""

    def test_standard_scaler(self):
        """Standard scaler should zero-mean, unit-variance."""
        X = np.random.randn(100, 10, 3).astype(np.float32) * 5 + 10
        scaler = fit_scaler(X, scaler_type="standard")
        X_norm = transform_data(X, scaler)

        mean = X_norm.mean()
        std = X_norm.std()
        assert abs(mean) < 0.05
        assert abs(std - 1.0) < 0.05

    def test_robust_scaler(self):
        """Robust scaler should handle outliers."""
        X = np.random.randn(100, 10, 3).astype(np.float32)
        X[0] = 1000  # extreme outlier
        scaler = fit_scaler(X, scaler_type="robust")
        X_norm = transform_data(X, scaler)
        # Should not be NaN
        assert not np.isnan(X_norm).any()

    def test_normal_only_mask(self):
        """Scaler fit on normal samples only should differ from all samples."""
        X = np.random.randn(200, 5, 2).astype(np.float32)
        X[100:] += 50  # anomalous samples shifted
        mask = np.concatenate([np.ones(100), np.zeros(100)]).astype(bool)

        scaler_normal = fit_scaler(X, scaler_type="standard", normal_only_mask=mask)
        scaler_all = fit_scaler(X, scaler_type="standard")

        # They should have different parameters
        assert not np.allclose(scaler_normal.mean_, scaler_all.mean_)


class TestWinsorize:
    """Test outlier clipping."""

    def test_clips_extreme_values(self):
        """Extreme values should be clipped to percentile bounds."""
        X = np.zeros((10, 5, 1), dtype=np.float32)
        X[0, 0, 0] = 1000  # extreme outlier

        X_clipped, bounds = winsorize(X, lower_percentile=5, upper_percentile=95)
        assert X_clipped[0, 0, 0] < 1000

    def test_preserves_normal_data(self):
        """Normal data within bounds should be unchanged."""
        X = np.random.randn(100, 5, 2).astype(np.float32)
        X_clipped, _ = winsorize(X, lower_percentile=1, upper_percentile=99)
        # Most values should be unchanged
        pct_changed = (X_clipped != X).mean()
        assert pct_changed < 0.05  # <5% should change


class TestImputeMissing:
    """Test missing value imputation."""

    def test_ffill_bfill(self):
        """Forward fill + backward fill should remove NaN."""
        df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, 5]})
        result = impute_missing(df, strategy="ffill_bfill")
        assert result.isna().sum().sum() == 0
        assert result["a"].tolist() == [1.0, 1.0, 1.0, 4.0, 5.0]

    def test_interpolate(self):
        """Linear interpolation should fill gaps."""
        df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, 5]})
        result = impute_missing(df, strategy="interpolate")
        assert result.isna().sum().sum() == 0
        # Middle values should be interpolated
        assert result["a"].iloc[1] == pytest.approx(2.0)
        assert result["a"].iloc[2] == pytest.approx(3.0)

    def test_zero_strategy(self):
        """Zero strategy should fill with 0."""
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        result = impute_missing(df, strategy="zero")
        assert result["a"].tolist() == [1.0, 0.0, 3.0]

    def test_max_gap(self):
        """Gaps larger than max_gap should remain for final fill."""
        df = pd.DataFrame({"a": [1] + [np.nan] * 20 + [5]})
        result = impute_missing(df, strategy="ffill_bfill", max_gap=5)
        # Large gap should still have NaN after limited ffill/bfill
        # Then final fillna(0) handles them
        assert result.isna().sum().sum() == 0


class TestCreateWindows:
    """Test sliding window creation."""

    def test_basic_windows(self):
        """Should create correct number of windows."""
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        y = np.arange(10, dtype=np.float32)

        X_w, y_w, _ = create_windows(X, y, window_size=3, step_size=1)
        assert X_w.shape == (8, 3, 2)  # (10 - 3 + 1) windows
        assert len(y_w) == 8

    def test_strategy_any(self):
        """'any' strategy should label window if any timestep is 1."""
        X = np.zeros((10, 2), dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        y[3] = 1

        X_w, y_w, _ = create_windows(X, y, window_size=5, step_size=1, strategy="any")
        # Window [0:5] contains index 3 which is 1
        assert y_w[0] == 1

    def test_no_labels(self):
        """Should work without labels or RUL."""
        X = np.random.randn(20, 3).astype(np.float32)
        X_w, y_w, rul_w = create_windows(X, window_size=5, step_size=1)
        assert X_w.shape == (16, 5, 3)
        assert y_w is None
        assert rul_w is None


class TestSplitData:
    """Test data splitting."""

    def test_split_proportions(self):
        """Split should produce approximate proportions."""
        X = np.random.randn(1000, 10, 3).astype(np.float32)
        y = np.concatenate([np.zeros(800), np.ones(200)])
        rul = np.arange(1000, dtype=np.float32)

        result = split_data(X, y, rul, val_split=0.15, test_split=0.15, random_seed=42)

        assert len(result["X_train"]) == 700
        assert len(result["X_val"]) == 150
        assert len(result["X_test"]) == 150

    def test_stratification(self):
        """Stratification should preserve class ratio."""
        X = np.random.randn(200, 5, 2).astype(np.float32)
        y = np.concatenate([np.zeros(160), np.ones(40)])  # 20% positive
        rul = np.arange(200, dtype=np.float32)

        result = split_data(X, y, rul, val_split=0.2, test_split=0.2,
                            random_seed=42, stratify=True)

        train_ratio = result["y_train"].mean()
        val_ratio = result["y_val"].mean()
        test_ratio = result["y_test"].mean()

        # All should be approximately 20% positive
        assert abs(train_ratio - 0.2) < 0.05
        assert abs(val_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05
