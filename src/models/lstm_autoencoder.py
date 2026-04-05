"""
src/models/lstm_autoencoder.py
================================
LSTM Autoencoder for unsupervised anomaly detection in ESP sensor data.

Architecture:
  Encoder: Bidirectional LSTM → latent vector z
  Decoder: LSTM (teacher-forced during training) → reconstructed sequence

Anomaly score = Mean Squared Reconstruction Error (MSE) per window.
Threshold = 95th percentile of reconstruction error on normal (healthy) windows.

Monte Carlo Dropout:
  Enabled at inference time by keeping dropout active.
  Run N forward passes → get distribution over reconstruction errors
  → anomaly score + uncertainty bounds.

References:
  - Malhotra et al. (2016) "LSTM-based Encoder-Decoder for Multi-sensor
    Anomaly Detection", ICML Anomaly Detection Workshop
  - Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import json
import os


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder → latent vector."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        latent_size: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # Project to latent space
        self.fc_z = nn.Linear(hidden_size * self.num_directions, latent_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            z: (batch, latent_size)  — latent representation
            hidden: last LSTM hidden state for decoder init
        """
        out, (h, c) = self.lstm(x)  # out: (batch, seq, hidden*directions)
        # Take the last timestep's output (or mean pooling)
        last_out = out[:, -1, :]   # (batch, hidden*directions)
        z = self.layer_norm(self.fc_z(self.dropout(last_out)))
        return z, (h, c)


class LSTMDecoder(nn.Module):
    """Unidirectional LSTM decoder → reconstructed sequence."""

    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        seq_len: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Project latent → decoder hidden state
        self.fc_init = nn.Linear(latent_size, hidden_size * num_layers)
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_size) — latent vector from encoder
            target: (batch, seq_len, output_size) — ground truth for teacher forcing
            teacher_forcing_ratio: probability of using ground truth input

        Returns:
            reconstructed: (batch, seq_len, output_size)
        """
        batch = z.size(0)
        device = z.device

        # Initialize decoder hidden state from latent vector
        h_0 = self.fc_init(z).view(self.num_layers, batch, self.hidden_size)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0.contiguous(), c_0.contiguous())

        # Start token: zeros
        decoder_input = torch.zeros(batch, 1, self.fc_out.out_features).to(device)
        outputs = []

        for t in range(self.seq_len):
            out, hidden = self.lstm(decoder_input, hidden)
            pred = self.fc_out(self.dropout(out))  # (batch, 1, output_size)
            outputs.append(pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = pred.detach()

        return torch.cat(outputs, dim=1)  # (batch, seq_len, output_size)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder for time-series anomaly detection.

    Typical usage:
        # Training
        model = LSTMAutoencoder(input_size=52, hidden_size=128, ...)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model.reconstruction_loss(batch_x)

        # Anomaly scoring
        scores = model.anomaly_score(batch_x)   # (batch,)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        latent_size: int = 32,
        dropout: float = 0.3,
        seq_len: int = 50,
        bidirectional_encoder: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_size=latent_size,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
        )
        self.decoder = LSTMDecoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            seq_len=seq_len,
            dropout=dropout,
        )

        self.threshold: Optional[float] = None  # set after calibration on normal data

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            x_hat: reconstructed (batch, seq_len, input_size)
            z:     latent codes   (batch, latent_size)
        """
        z, _ = self.encoder(x)
        x_hat = self.decoder(z, target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return x_hat, z

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """MSE reconstruction loss."""
        x_hat, _ = self.forward(x, teacher_forcing_ratio)
        return nn.functional.mse_loss(x_hat, x, reduction=reduction)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window reconstruction error (MSE, mean over time and features).
        Higher = more anomalous.
        """
        was_training = self.training
        self.eval()
        x_hat, _ = self.forward(x, teacher_forcing_ratio=0.0)
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))  # (batch,)
        if was_training:
            self.train()
        return scores

    def calibrate_threshold(
        self,
        normal_loader,
        device: torch.device,
        percentile: float = 95.0,
    ) -> float:
        """
        Compute anomaly threshold from normal (healthy) data.

        Args:
            normal_loader: DataLoader containing only normal windows,
                          OR np.ndarray of shape (N, seq_len, features).
            device: Torch device.
            percentile: Threshold percentile (95 = 5% false-positive rate).

        Returns:
            threshold: float, stored in self.threshold
        """
        self.eval()
        all_scores = []

        # Support raw numpy arrays as input
        if isinstance(normal_loader, np.ndarray):
            X = torch.from_numpy(normal_loader).float().to(device)
            with torch.no_grad():
                scores = self.anomaly_score(X)
                all_scores.append(scores.cpu().numpy())
        else:
            with torch.no_grad():
                for batch in normal_loader:
                    x = batch["X"].to(device)
                    scores = self.anomaly_score(x)
                    all_scores.append(scores.cpu().numpy())

        all_scores = np.concatenate(all_scores)
        self.threshold = float(np.percentile(all_scores, percentile))
        return self.threshold

    def predict(
        self,
        x: torch.Tensor,
        return_scores: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Binary anomaly prediction.

        Returns:
            dict with keys: "labels" (0/1), "scores", optionally "x_hat"
        """
        assert self.threshold is not None, (
            "Call calibrate_threshold() on normal data first."
        )
        scores = self.anomaly_score(x)
        labels = (scores > self.threshold).long()
        result = {"labels": labels, "scores": scores}
        if return_scores:
            result["x_hat"] = self.forward(x, teacher_forcing_ratio=0.0)[0]
        return result

    def get_config(self) -> dict:
        return {
            "model_type": "LSTMAutoencoder",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "seq_len": self.seq_len,
            "threshold": self.threshold,
        }

    def save_pretrained(self, save_dir: str):
        """Save model weights + config (HuggingFace-style)."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.get_config(), f, indent=2)
        print(f"Model saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> "LSTMAutoencoder":
        """Load model from directory."""
        with open(os.path.join(load_dir, "config.json")) as f:
            config = json.load(f)
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            latent_size=config["latent_size"],
            seq_len=config["seq_len"],
        )
        state = torch.load(
            os.path.join(load_dir, "pytorch_model.bin"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.threshold = config.get("threshold")
        return model


# ──────────────────────────────────────────────────────────────────
# Monte Carlo Dropout inference
# ──────────────────────────────────────────────────────────────────

def mc_dropout_anomaly_scores(
    model: LSTMAutoencoder,
    x: torch.Tensor,
    n_samples: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout: run N stochastic forward passes to estimate
    prediction uncertainty.

    By enabling dropout at inference, each pass gives a slightly different
    reconstruction error → empirical distribution of anomaly scores.

    Args:
        model: LSTMAutoencoder with dropout
        x: (batch, seq_len, features) input tensor
        n_samples: number of MC samples

    Returns:
        mean_scores:  (batch,) mean anomaly score
        std_scores:   (batch,) standard deviation (epistemic uncertainty)
        all_scores:   (n_samples, batch) all individual scores
    """
    # Enable dropout at inference by switching to train() mode selectively
    model.train()  # activates dropout
    all_scores = []
    with torch.no_grad():
        for _ in range(n_samples):
            scores = model.anomaly_score(x)
            all_scores.append(scores.cpu().numpy())
    model.eval()

    all_scores = np.stack(all_scores)  # (n_samples, batch)
    return all_scores.mean(0), all_scores.std(0), all_scores
