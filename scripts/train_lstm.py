"""
scripts/train_lstm.py
======================
Train LSTM Autoencoder for ESP anomaly detection.

Usage:
    python scripts/train_lstm.py --config configs/lstm_config.yaml
    python scripts/train_lstm.py --config configs/lstm_config.yaml --dataset synthetic
    python scripts/train_lstm.py --data_path data/raw/synthetic_esp.csv
"""

import argparse
import os
import sys
import yaml
import logging
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import generate_esp_dataset, SYNTHETIC_SENSOR_COLS
from src.data.loader import (
    load_pump_sensor, TimeSeriesDataset, make_dataloaders,
    PUMP_SENSOR_COLS,
)
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.training.trainer import AutoencoderTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--config", default="configs/lstm_config.yaml",
                        help="Path to LSTM config YAML")
    parser.add_argument("--training_config", default="configs/training_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--dataset", choices=["pump_sensor", "synthetic"],
                        default="synthetic", help="Dataset to use")
    parser.add_argument("--data_path", default=None,
                        help="Direct path to CSV file (overrides --dataset)")
    parser.add_argument("--save_dir", default="checkpoints/lstm_ae",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detects if not provided)")
    args = parser.parse_args()

    # Load configs
    lstm_cfg = load_config(args.config)
    train_cfg = load_config(args.training_config)

    model_cfg = lstm_cfg["model"]
    anomaly_cfg = lstm_cfg["anomaly"]
    data_cfg = train_cfg["data"]
    training_cfg = train_cfg["training"]
    opt_cfg = train_cfg["optimizer"]
    sched_cfg = train_cfg["scheduler"]

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────
    if args.data_path:
        # Use provided CSV path
        logger.info(f"Loading data from {args.data_path}")
        # Determine if it's pump sensor or synthetic by checking columns
        import pandas as pd
        df = pd.read_csv(args.data_path, nrows=1)
        if "sensor_01" in df.columns:
            # Pump sensor format
            data = load_pump_sensor(
                args.data_path,
                window_size=data_cfg["window_size"],
                step_size=data_cfg["step_size"],
                val_split=data_cfg["val_split"],
                test_split=data_cfg["test_split"],
                random_seed=data_cfg["random_seed"],
            )
        else:
            # Assume synthetic — load with synthetic columns
            from src.data.loader import _sliding_window, _compute_rul, _split_and_scale
            df_full = pd.read_csv(args.data_path, parse_dates=["timestamp"])
            df_full = df_full.sort_values("timestamp").reset_index(drop=True)
            df_full["failure"] = (df_full["machine_status"] == "BROKEN").astype(int)
            rul_series = _compute_rul(df_full["failure"].values)
            X_raw = df_full[SYNTHETIC_SENSOR_COLS].values.astype(np.float32)
            y_raw = df_full["failure"].values.astype(np.float32)
            rul_raw = rul_series.astype(np.float32)
            X_windows, y_windows, rul_windows = _sliding_window(
                X_raw, y_raw, rul_raw,
                window_size=data_cfg["window_size"],
                step_size=data_cfg["step_size"],
            )
            data = _split_and_scale(
                X_windows, y_windows, rul_windows, SYNTHETIC_SENSOR_COLS,
                val_split=data_cfg["val_split"],
                test_split=data_cfg["test_split"],
                random_seed=data_cfg["random_seed"],
            )
    elif args.dataset == "synthetic":
        logger.info("Generating synthetic dataset...")
        df_synth = generate_esp_dataset(
            n_wells=20, timesteps_per_well=3000,
            failure_prob=0.6, random_seed=42,
        )
        df_synth["failure"] = (df_synth["machine_status"] == "BROKEN").astype(int)
        from src.data.loader import _sliding_window, _compute_rul, _split_and_scale
        rul_series = _compute_rul(df_synth["failure"].values)
        X_raw = df_synth[SYNTHETIC_SENSOR_COLS].values.astype(np.float32)
        y_raw = df_synth["failure"].values.astype(np.float32)
        rul_raw = rul_series.astype(np.float32)
        X_windows, y_windows, rul_windows = _sliding_window(
            X_raw, y_raw, rul_raw,
            window_size=data_cfg["window_size"],
            step_size=data_cfg["step_size"],
        )
        data = _split_and_scale(
            X_windows, y_windows, rul_windows, SYNTHETIC_SENSOR_COLS,
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            random_seed=data_cfg["random_seed"],
        )
    else:
        # Pump sensor
        raw_path = Path(train_cfg["paths"]["raw_data_dir"]) / "pump_sensor.csv"
        data = load_pump_sensor(
            str(raw_path),
            window_size=data_cfg["window_size"],
            step_size=data_cfg["step_size"],
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            random_seed=data_cfg["random_seed"],
        )

    # Create dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        data,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 0),
    )

    n_features = data["X_train"].shape[-1]
    logger.info(f"Input features: {n_features}, Window size: {data_cfg['window_size']}")

    # ── Build model ───────────────────────────────────────────────
    model = LSTMAutoencoder(
        input_size=n_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        latent_size=model_cfg["latent_size"],
        dropout=model_cfg["dropout"],
        seq_len=data_cfg["window_size"],
        bidirectional_encoder=model_cfg["bidirectional_encoder"],
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ─────────────────────────────────────────────────────
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        gradient_clip=training_cfg.get("gradient_clip", 1.0),
        early_stopping_patience=training_cfg.get("early_stopping_patience", 15),
        scheduler_T0=sched_cfg.get("T_0", 20),
        teacher_forcing_ratio=model_cfg.get("teacher_forcing_ratio", 0.5),
    )

    history = trainer.train(num_epochs=training_cfg["num_epochs"])

    # ── Calibrate threshold ───────────────────────────────────────
    logger.info("Calibrating anomaly threshold on normal training data...")
    normal_mask = data["y_train"] == 0
    normal_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(data["X_train"][normal_mask]).float()
        ),
        batch_size=training_cfg["batch_size"],
        shuffle=False,
    )
    # Wrap to match expected batch["X"] structure
    class NormalDataset:
        def __init__(self, X):
            self.X = torch.from_numpy(X).float()
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return {"X": self.X[idx]}
    normal_loader = torch.utils.data.DataLoader(
        NormalDataset(data["X_train"][normal_mask]),
        batch_size=training_cfg["batch_size"],
        shuffle=False,
    )
    model.calibrate_threshold(normal_loader, device, percentile=anomaly_cfg["threshold_percentile"])
    logger.info(f"Anomaly threshold: {model.threshold:.6f}")

    # ── Save final model ──────────────────────────────────────────
    model.save_pretrained(args.save_dir)
    logger.info(f"Model saved to {args.save_dir}")

    # ── Plot training curves ──────────────────────────────────────
    fig = trainer.plot_history(save_path=os.path.join(args.save_dir, "training_history.png"))
    logger.info("Training history plot saved.")


if __name__ == "__main__":
    main()
