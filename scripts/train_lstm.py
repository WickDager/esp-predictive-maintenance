"""
scripts/train_lstm.py
======================
Train LSTM Autoencoder for ESP anomaly detection.

Usage:
    python scripts/train_lstm.py --config configs/lstm_config.yaml
    python scripts/train_lstm.py --dataset synthetic
    python scripts/train_lstm.py --data_path data/raw/synthetic_esp.csv
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import (
    generate_esp_dataset,
    SYNTHETIC_SENSOR_COLS,
)
from src.data.loader import (
    load_pump_sensor,
    _sliding_window,
    _compute_rul,
    _split_and_scale,
)
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.training.trainer import AutoencoderTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml(path):
    with open(path) as f:
        return __import__("yaml").safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--config", default="configs/lstm_config.yaml")
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--dataset", choices=["pump_sensor", "synthetic"],
                        default="synthetic")
    parser.add_argument("--data_path", default=None, help="Direct CSV path")
    parser.add_argument("--save_dir", default="checkpoints/lstm_ae")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    lstm_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.training_config)

    model_cfg = lstm_cfg["model"]
    anomaly_cfg = lstm_cfg["anomaly"]
    data_cfg = train_cfg["data"]
    training_cfg = train_cfg["training"]
    opt_cfg = train_cfg["optimizer"]
    sched_cfg = train_cfg["scheduler"]

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("Using device: %s", device)

    # ── Load data ─────────────────────────────────────────────────
    if args.data_path:
        df = pd.read_csv(args.data_path, nrows=1)
        if "sensor_01" in df.columns:
            data = load_pump_sensor(
                args.data_path, window_size=data_cfg["window_size"],
                step_size=data_cfg["step_size"],
                val_split=data_cfg["val_split"],
                test_split=data_cfg["test_split"],
                random_seed=data_cfg["random_seed"],
            )
        else:
            data = _load_csv(args.data_path, data_cfg)
    elif args.dataset == "synthetic":
        logger.info("Generating synthetic dataset...")
        df_s = generate_esp_dataset(
            n_wells=20, timesteps_per_well=3000,
            failure_prob=0.6, random_seed=42,
        )
        data = _df_to_data(df_s, SYNTHETIC_SENSOR_COLS, data_cfg)
    else:
        raw_path = Path(train_cfg["paths"]["raw_data_dir"]) / "pump_sensor.csv"
        data = load_pump_sensor(
            str(raw_path), window_size=data_cfg["window_size"],
            step_size=data_cfg["step_size"],
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            random_seed=data_cfg["random_seed"],
        )

    # ── Create dataloaders ────────────────────────────────────────
    n_workers = training_cfg.get("num_workers", 0)
    batch_size = training_cfg["batch_size"]

    class NormalDataset:
        def __init__(self, X):
            self.X = torch.from_numpy(X).float()
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return {"X": self.X[idx]}

    train_ds = __import__("src.data.loader", fromlist=["TimeSeriesDataset"]).TimeSeriesDataset(
        data["X_train"], data["y_train"],
    )
    val_ds = __import__("src.data.loader", fromlist=["TimeSeriesDataset"]).TimeSeriesDataset(
        data["X_val"], data["y_val"],
    )
    test_ds = __import__("src.data.loader", fromlist=["TimeSeriesDataset"]).TimeSeriesDataset(
        data["X_test"], data["y_test"],
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=n_workers)

    n_features = data["X_train"].shape[-1]
    logger.info("Input features: %d, Window size: %d", n_features, data_cfg["window_size"])

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
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

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

    trainer.train(num_epochs=training_cfg["num_epochs"])

    # ── Calibrate threshold ───────────────────────────────────────
    logger.info("Calibrating anomaly threshold...")
    normal_mask = data["y_train"] == 0
    normal_loader = DataLoader(
        NormalDataset(data["X_train"][normal_mask]),
        batch_size=batch_size, shuffle=False,
    )
    model.calibrate_threshold(
        normal_loader, device, percentile=anomaly_cfg["threshold_percentile"],
    )
    logger.info("Anomaly threshold: %.6f", model.threshold)

    # ── Save ──────────────────────────────────────────────────────
    model.save_pretrained(args.save_dir)
    trainer.plot_history(save_path=os.path.join(args.save_dir, "training_history.png"))
    logger.info("Model saved to %s", args.save_dir)


def _df_to_data(df, sensor_cols, data_cfg):
    """Convert raw dataframe to windowed data dict."""
    df["failure"] = (df["machine_status"] == "BROKEN").astype(int)
    rul_series = _compute_rul(df["failure"].values)
    X_raw = df[sensor_cols].values.astype(np.float32)
    y_raw = df["failure"].values.astype(np.float32)
    rul_raw = rul_series.astype(np.float32)
    X_w, y_w, rul_w = _sliding_window(
        X_raw, y_raw, rul_raw,
        window_size=data_cfg["window_size"],
        step_size=data_cfg["step_size"],
    )
    return _split_and_scale(
        X_w, y_w, rul_w, sensor_cols,
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        random_seed=data_cfg["random_seed"],
    )


def _load_csv(path, data_cfg):
    """Load a CSV with synthetic-style columns."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return _df_to_data(df, SYNTHETIC_SENSOR_COLS, data_cfg)


if __name__ == "__main__":
    main()
