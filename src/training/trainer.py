"""
src/training/trainer.py
=========================
Generic training loop for LSTM Autoencoder and Transformer Autoencoder.

Features:
  - Early stopping with configurable patience
  - Learning rate scheduling
  - Gradient clipping
  - Checkpoint saving (best model by val loss)
  - Tensorboard/CSV logging
  - Colab-friendly progress bars (tqdm)
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when val loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            (score < self.best_score - self.min_delta) if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class AutoencoderTrainer:
    """
    Trainer for LSTM Autoencoder and Transformer Autoencoder.

    Both models use the same training objective: minimise MSE
    reconstruction loss on input windows (unsupervised).

    Example usage:
        trainer = AutoencoderTrainer(
            model=lstm_ae,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir="checkpoints/lstm_ae",
        )
        history = trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str = "checkpoints",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 15,
        scheduler_T0: int = 20,
        teacher_forcing_ratio: float = 0.5,
        teacher_forcing_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.gradient_clip = gradient_clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )
        self.early_stopper = EarlyStopping(patience=early_stopping_patience)
        self.criterion = nn.MSELoss()

        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self._log_file = os.path.join(save_dir, "training_log.csv")
        self._init_log()

    def _init_log(self):
        with open(self._log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_sec"])

    def _log_epoch(self, epoch, train_loss, val_loss, lr, elapsed):
        with open(self._log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{lr:.8f}", f"{elapsed:.1f}"])

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x = batch["X"].to(self.device)

            self.optimizer.zero_grad()

            # Handle both LSTM AE (teacher forcing) and Transformer AE
            if hasattr(self.model, 'reconstruction_loss'):
                # Check if model accepts teacher_forcing_ratio
                import inspect
                sig = inspect.signature(self.model.reconstruction_loss)
                if 'teacher_forcing_ratio' in sig.parameters:
                    loss = self.model.reconstruction_loss(
                        x, teacher_forcing_ratio=self.teacher_forcing_ratio
                    )
                else:
                    loss = self.model.reconstruction_loss(x)
            else:
                x_hat, _ = self.model(x)
                loss = self.criterion(x_hat, x)

            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            x = batch["X"].to(self.device)
            if hasattr(self.model, 'reconstruction_loss'):
                import inspect
                sig = inspect.signature(self.model.reconstruction_loss)
                if 'teacher_forcing_ratio' in sig.parameters:
                    loss = self.model.reconstruction_loss(x, teacher_forcing_ratio=0.0)
                else:
                    loss = self.model.reconstruction_loss(x)
            else:
                x_hat, _ = self.model(x)
                loss = self.criterion(x_hat, x)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, num_epochs: int = 100) -> Dict[str, list]:
        """
        Full training loop.

        Returns:
            history dict with train/val loss curves.
        """
        best_val_loss = float("inf")
        best_ckpt_path = os.path.join(self.save_dir, "best_model.pt")

        pbar = tqdm(range(1, num_epochs + 1), desc="Training")
        for epoch in pbar:
            t0 = time.time()

            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            self.scheduler.step(val_loss)
            # Decay teacher forcing ratio (curriculum learning)
            self.teacher_forcing_ratio = max(
                0.0, self.teacher_forcing_ratio - self.teacher_forcing_decay
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            self._log_epoch(epoch, train_loss, val_loss, current_lr, elapsed)

            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "lr": f"{current_lr:.6f}",
            })

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }, best_ckpt_path)

            # Early stopping
            if self.early_stopper(val_loss):
                logger.info(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

        # Load best weights
        if not os.path.exists(best_ckpt_path):
            logger.warning(f"No checkpoint saved at {best_ckpt_path}. Training may have diverged (NaN losses).")
            logger.warning("Returning history without loading best model.")
            return self.history

        checkpoint = torch.load(best_ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

        return self.history

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training curves."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(self.history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(self.history["val_loss"], label="Val", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Reconstruction Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].semilogy(self.history["lr"])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
