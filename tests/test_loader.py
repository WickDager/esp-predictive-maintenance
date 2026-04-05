"""
tests/test_loader.py
=====================
Unit tests for data loading and sliding window logic.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    _sliding_window,
    _compute_rul,
    _split_and_scale,
    _cmapss_windows,
    TimeSeriesDataset,
    make_dataloaders,
)


class TestSlidingWindow:
    """Test the sliding window creation function."""

    def test_basic_window_shape(self):
        """Windows should have shape (num_windows, window_size, features)."""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.zeros(100, dtype=np.float32)
        rul = np.arange(100, dtype=np.float32)

        X_w, y_w, rul_w = _sliding_window(X, y, rul, window_size=10, step_size=1)

        expected_windows = (100 - 10 + 1) // 1
        assert X_w.shape == (expected_windows, 10, 5)
        assert len(y_w) == expected_windows
        assert len(rul_w) == expected_windows

    def test_step_size(self):
        """Larger step size should produce fewer windows."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.zeros(100, dtype=np.float32)
        rul = np.zeros(100, dtype=np.float32)

        X_w1, _, _ = _sliding_window(X, y, rul, window_size=10, step_size=1)
        X_w5, _, _ = _sliding_window(X, y, rul, window_size=10, step_size=5)

        assert len(X_w1) > len(X_w5)

    def test_window_content(self):
        """Each window should contain the correct sequential data."""
        X = np.arange(30).reshape(10, 3).astype(np.float32)
        y = np.arange(10, dtype=np.float32)
        rul = np.arange(10, dtype=np.float32)

        X_w, y_w, rul_w = _sliding_window(X, y, rul, window_size=3, step_size=1)

        # First window should be rows 0,1,2
        np.testing.assert_array_equal(X_w[0], X[0:3])
        # Last timestep label
        assert y_w[0] == y[2]
        assert rul_w[0] == rul[2]

    def test_label_is_last_timestep(self):
        """Label should come from the last timestep of each window."""
        X = np.random.randn(20, 2).astype(np.float32)
        y = np.zeros(20, dtype=np.float32)
        y[9] = 1  # failure at index 9
        rul = np.arange(20, dtype=np.float32)

        X_w, y_w, rul_w = _sliding_window(X, y, rul, window_size=5, step_size=1)

        # Window ending at index 9 should have label 1
        assert y_w[5] == 1  # window [5:10] ends at index 9
        # Window ending at index 8 should have label 0
        assert y_w[4] == 0  # window [4:9] ends at index 8


class TestComputeRUL:
    """Test RUL computation from binary failure flags."""

    def test_rul_decreases_toward_failure(self):
        """RUL should decrease by 1 each step toward a failure."""
        flags = np.array([0, 0, 0, 1, 0, 0])
        rul = _compute_rul(flags)
        assert rul[0] == 3  # 3 steps to failure
        assert rul[1] == 2
        assert rul[2] == 1
        assert rul[3] == 0  # at failure

    def test_no_failure_means_no_rul(self):
        """If no failure, RUL should be -1."""
        flags = np.zeros(10, dtype=np.float32)
        rul = _compute_rul(flags)
        assert (rul == -1).all()

    def test_rul_at_failure_is_zero(self):
        """RUL at the failure timestep should be 0."""
        flags = np.zeros(20, dtype=np.float32)
        flags[15] = 1
        rul = _compute_rul(flags)
        assert rul[15] == 0


class TestSplitAndScale:
    """Test train/val/test splitting and scaling."""

    def test_split_sizes(self):
        """Split should produce correct proportions."""
        X = np.random.randn(1000, 10, 5).astype(np.float32)
        y = np.concatenate([np.zeros(800), np.ones(200)])
        rul = np.arange(1000, dtype=np.float32)
        feature_names = [f"f{i}" for i in range(5)]

        data = _split_and_scale(X, y, rul, feature_names,
                                 val_split=0.15, test_split=0.15,
                                 random_seed=42)

        total = len(X)
        assert abs(len(data["X_train"]) / total - 0.70) < 0.05
        assert abs(len(data["X_val"]) / total - 0.15) < 0.05
        assert abs(len(data["X_test"]) / total - 0.15) < 0.05

    def test_scaler_fit_on_train_only(self):
        """Scaler should be fit only on training data."""
        X = np.random.randn(500, 10, 3).astype(np.float32)
        y = np.concatenate([np.zeros(400), np.ones(100)])
        rul = np.arange(500, dtype=np.float32)

        data = _split_and_scale(X, y, rul, ["f0", "f1", "f2"],
                                 val_split=0.15, test_split=0.15,
                                 random_seed=42)

        # Training data should be approximately zero-mean, unit-variance
        train_mean = data["X_train"][data["y_train"] == 0].mean()
        train_std = data["X_train"][data["y_train"] == 0].std()
        assert abs(train_mean) < 0.1, f"Train mean too far from 0: {train_mean}"
        assert abs(train_std - 1.0) < 0.2, f"Train std too far from 1: {train_std}"


class TestTimeSeriesDataset:
    """Test the PyTorch Dataset wrapper."""

    def test_dataset_length(self):
        """Dataset length should match number of samples."""
        X = np.random.randn(100, 10, 5).astype(np.float32)
        y = np.ones(100, dtype=np.float32)
        ds = TimeSeriesDataset(X, y)
        assert len(ds) == 100

    def test_dataset_item(self):
        """Each item should have X and optionally y, rul."""
        X = np.random.randn(50, 10, 3).astype(np.float32)
        y = np.ones(50, dtype=np.float32)
        rul = np.arange(50, dtype=np.float32)

        ds = TimeSeriesDataset(X, y, rul)
        item = ds[0]
        assert "X" in item
        assert "y" in item
        assert "rul" in item
        assert item["X"].shape == (10, 3)

    def test_without_labels(self):
        """Dataset without labels should only return X."""
        X = np.random.randn(30, 5, 2).astype(np.float32)
        ds = TimeSeriesDataset(X)
        item = ds[0]
        assert "X" in item
        assert "y" not in item
        assert "rul" not in item
