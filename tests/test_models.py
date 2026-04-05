"""
tests/test_models.py
=====================
Unit tests for model architectures (forward pass, save/load, etc.).
"""

import numpy as np
import pytest
import torch
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lstm_autoencoder import LSTMAutoencoder, mc_dropout_anomaly_scores
from src.models.transformer_model import TransformerAutoencoder
from src.models.rul_predictor import RULPredictor, AsymmetricRULLoss


class TestLSTMAutoencoder:
    """Test LSTM Autoencoder."""

    def setup_method(self):
        self.model = LSTMAutoencoder(
            input_size=10, hidden_size=32, num_layers=2,
            latent_size=8, dropout=0.2, seq_len=20,
        )

    def test_forward_shape(self):
        """Forward pass should return correct output shapes."""
        x = torch.randn(4, 20, 10)
        x_hat, z = self.model(x, teacher_forcing_ratio=0.0)
        assert x_hat.shape == (4, 20, 10)
        assert z.shape == (4, 8)

    def test_reconstruction_loss(self):
        """Loss should be a scalar tensor."""
        x = torch.randn(4, 20, 10)
        loss = self.model.reconstruction_loss(x, teacher_forcing_ratio=0.5)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_anomaly_score(self):
        """Anomaly score should be per-sample."""
        x = torch.randn(8, 20, 10)
        scores = self.model.anomaly_score(x)
        assert scores.shape == (8,)

    def test_save_and_load(self):
        """Model should save and load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "pytorch_model.bin"))
            assert os.path.exists(os.path.join(tmpdir, "config.json"))

            loaded = LSTMAutoencoder.from_pretrained(tmpdir)
            assert loaded.input_size == self.model.input_size
            assert loaded.hidden_size == self.model.hidden_size
            assert loaded.latent_size == self.model.latent_size
            assert loaded.seq_len == self.model.seq_len

    def test_mc_dropout(self):
        """MC Dropout should produce uncertainty estimates."""
        x = torch.randn(5, 20, 10)
        mean, std, all_scores = mc_dropout_anomaly_scores(self.model, x, n_samples=10)
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert all_scores.shape == (10, 5)
        # Standard deviations should be positive
        assert (std >= 0).all()

    def test_predict_requires_threshold(self):
        """Predict should fail if threshold not set."""
        x = torch.randn(2, 20, 10)
        with pytest.raises(AssertionError):
            self.model.predict(x)

    def test_calibrate_threshold_with_array(self):
        """Threshold calibration should work with numpy arrays."""
        normal_data = np.random.randn(100, 20, 10).astype(np.float32)
        device = torch.device("cpu")
        threshold = self.model.calibrate_threshold(normal_data, device, percentile=95)
        assert 0 < threshold < 10  # reasonable MSE range


class TestTransformerAutoencoder:
    """Test Transformer Autoencoder."""

    def setup_method(self):
        self.model = TransformerAutoencoder(
            input_size=10, d_model=32, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=64, dropout=0.1, seq_len=20,
        )

    def test_forward_shape(self):
        """Forward pass should return correct shapes."""
        x = torch.randn(4, 20, 10)
        x_hat, latent = self.model(x)
        assert x_hat.shape == (4, 20, 10)
        assert latent.shape == (4, 32)

    def test_reconstruction_loss(self):
        """Loss should be a scalar."""
        x = torch.randn(4, 20, 10)
        loss = self.model.reconstruction_loss(x)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_save_and_load_full_config(self):
        """All hyperparameters should be preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            loaded = TransformerAutoencoder.from_pretrained(tmpdir)

            assert loaded.d_model == self.model.d_model
            assert loaded.encoder.layers[0].self_attn.num_heads == self.model.encoder.layers[0].self_attn.num_heads
            assert len(loaded.encoder.layers) == len(self.model.encoder.layers)
            assert len(loaded.decoder.layers) == len(self.model.decoder.layers)

    def test_anomaly_score(self):
        """Anomaly score should be per-sample."""
        x = torch.randn(6, 20, 10)
        scores = self.model.anomaly_score(x)
        assert scores.shape == (6,)

    def test_calibrate_threshold_with_array(self):
        """Threshold calibration should work with numpy arrays."""
        normal_data = np.random.randn(50, 20, 10).astype(np.float32)
        device = torch.device("cpu")
        threshold = self.model.calibrate_threshold(normal_data, device, percentile=95)
        assert 0 < threshold < 10


class TestRULPredictor:
    """Test RUL Predictor."""

    def setup_method(self):
        self.model = RULPredictor(
            input_size=10, hidden_size=32, num_layers=2,
            dropout=0.2, output_range=(0, 130),
        )

    def test_forward_shape(self):
        """Forward should return (batch,) predictions."""
        x = torch.randn(8, 20, 10)
        pred = self.model(x)
        assert pred.shape == (8,)

    def test_output_bounded(self):
        """Predictions should be within output_range."""
        x = torch.randn(100, 20, 10)
        pred = self.model(x)
        assert (pred >= 0).all()
        assert (pred <= 130).all()

    def test_predict_with_uncertainty(self):
        """MC Dropout should return mean, std, CI."""
        x = torch.randn(5, 20, 10)
        result = self.model.predict_with_uncertainty(x, n_samples=10)
        assert "mean" in result
        assert "std" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["mean"].shape == (5,)
        assert result["std"].shape == (5,)
        # CI low should be <= mean <= CI high
        assert (result["ci_low"] <= result["mean"]).all()
        assert (result["ci_high"] >= result["mean"]).all()

    def test_save_and_load(self):
        """Model should save and load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            loaded = RULPredictor.from_pretrained(tmpdir)
            assert loaded.input_size == self.model.input_size
            assert loaded.hidden_size == self.model.hidden_size


class TestAsymmetricRULLoss:
    """Test asymmetric RUL loss."""

    def test_overprediction_penalized_more(self):
        """Over-prediction should have higher loss."""
        criterion = AsymmetricRULLoss(alpha=2.0)
        target = torch.tensor([50.0])

        pred_over = torch.tensor([70.0])  # over-predict by 20
        pred_under = torch.tensor([30.0])  # under-predict by 20

        loss_over = criterion(pred_over, target)
        loss_under = criterion(pred_under, target)

        assert loss_over > loss_under

    def test_zero_loss_for_perfect(self):
        """Perfect prediction should give zero loss."""
        criterion = AsymmetricRULLoss()
        target = torch.tensor([50.0])
        pred = torch.tensor([50.0])
        loss = criterion(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
