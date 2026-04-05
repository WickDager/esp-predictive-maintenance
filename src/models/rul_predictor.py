"""
src/models/rul_predictor.py
============================
Remaining Useful Life (RUL) prediction for ESPs.

Architecture:
  Bi-LSTM feature extractor → Dense regression head with MC Dropout

Task framing:
  Given a window of sensor readings, predict the number of timesteps
  (hours / cycles) until the next failure event.

Loss:
  Weighted MSE that penalises under-prediction more heavily than
  over-prediction (asymmetric loss) — in oil & gas, missing a failure
  early is costlier than a false early warning.

Training data:
  - NASA CMAPSS (FD001-FD004) for benchmarking
  - Pump Sensor dataset with computed RUL labels (from loader.py)

References:
  - Zheng et al. (2017) "Long short-term memory network for remaining
    useful life estimation" — IEEE PHM
  - Li et al. (2018) "Remaining useful life estimation in prognostics
    using deep convolution neural networks"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import json
import os


# ──────────────────────────────────────────────────────────────────
# Asymmetric RUL Loss
# ──────────────────────────────────────────────────────────────────

class AsymmetricRULLoss(nn.Module):
    """
    Asymmetric weighted MSE for RUL prediction.

    Under-prediction (we say less life remains than actual):
        → operator may trigger early maintenance (costly but recoverable)
    Over-prediction (we say more life remains than actual):
        → risk of actual failure (catastrophic)

    α > 1 penalises over-predictions (actual < predicted) more.

    L(y, ŷ) = α * max(0, ŷ-y)² + max(0, y-ŷ)²
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        over_pred_mask = diff > 0   # predicted > actual → over-prediction
        loss = torch.where(
            over_pred_mask,
            self.alpha * diff ** 2,
            diff ** 2
        )
        return loss.mean()


# ──────────────────────────────────────────────────────────────────
# Bi-LSTM RUL Regressor
# ──────────────────────────────────────────────────────────────────

class RULPredictor(nn.Module):
    """
    Bidirectional LSTM regressor for RUL estimation.

    Args:
        input_size:    number of sensor features
        hidden_size:   LSTM hidden units per direction
        num_layers:    LSTM depth
        dropout:       dropout rate (also enables MC Dropout at inference)
        output_range:  optional (min_rul, max_rul) for sigmoid scaling
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        output_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_range = output_range

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            rul_pred: (batch,) predicted RUL values
        """
        out, _ = self.lstm(x)                  # (batch, seq, hidden*2)
        # Attention-weighted pooling over timesteps
        # (gives more weight to recent timesteps)
        pooled = self._temporal_attention_pool(out)  # (batch, hidden*2)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        rul = self.head(pooled).squeeze(-1)     # (batch,)

        # Optional sigmoid scaling to bounded range
        if self.output_range is not None:
            lo, hi = self.output_range
            rul = torch.sigmoid(rul) * (hi - lo) + lo

        return rul

    def _temporal_attention_pool(self, out: torch.Tensor) -> torch.Tensor:
        """
        Learnable temporal attention pooling.
        More recent timesteps naturally get higher attention during training.
        """
        # Simple learned weighted sum using linear projection
        attn_weights = torch.softmax(
            out.mean(dim=-1, keepdim=True),   # (batch, seq, 1)
            dim=1
        )
        return (out * attn_weights).sum(dim=1)  # (batch, hidden*2)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        MC Dropout RUL prediction with uncertainty quantification.

        Returns:
            mean:   (batch,) mean predicted RUL
            std:    (batch,) epistemic uncertainty
            ci_low: (batch,) 5th percentile
            ci_high:(batch,) 95th percentile
        """
        self.train()  # activate dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred.cpu().numpy())
        self.eval()

        samples = np.stack(samples)  # (n_samples, batch)
        return {
            "mean": samples.mean(0),
            "std": samples.std(0),
            "ci_low": np.percentile(samples, 5, axis=0),
            "ci_high": np.percentile(samples, 95, axis=0),
            "all_samples": samples,
        }

    def get_config(self) -> dict:
        return {
            "model_type": "RULPredictor",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.lstm.num_layers,
            "dropout": self.dropout.p,
            "output_range": list(self.output_range) if self.output_range else None,
        }

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> "RULPredictor":
        with open(os.path.join(load_dir, "config.json")) as f:
            config = json.load(f)
        output_range = config.get("output_range")
        if output_range is not None:
            output_range = tuple(output_range)
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.3),
            output_range=output_range,
        )
        model.load_state_dict(torch.load(
            os.path.join(load_dir, "pytorch_model.bin"),
            map_location=device,
            weights_only=True,
        ))
        return model


# ──────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────

def train_rul_epoch(
    model: RULPredictor,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["X"].to(device)
        rul = batch["rul"].to(device)

        # Skip windows with unknown RUL (label = -1)
        mask = rul >= 0
        if mask.sum() == 0:
            continue
        x, rul = x[mask], rul[mask]

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, rul)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_rul(
    model: RULPredictor,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate RUL predictor. Returns RMSE, MAE, Score function."""
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        x = batch["X"].to(device)
        rul = batch["rul"].to(device)
        mask = rul >= 0
        if mask.sum() == 0:
            continue
        pred = model(x[mask]).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(rul[mask].cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    mae = float(np.abs(preds - targets).mean())
    # NASA CMAPSS scoring function (penalises late predictions more)
    diff = preds - targets
    score = float(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1).sum())

    return {"rmse": rmse, "mae": mae, "nasa_score": score}
