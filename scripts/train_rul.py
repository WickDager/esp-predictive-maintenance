"""
scripts/train_rul.py
=====================
Train Bi-LSTM RUL (Remaining Useful Life) predictor.

Usage:
    python scripts/train_rul.py --dataset cmapss --cmapss_subset FD001
    python scripts/train_rul.py --dataset pump_sensor
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import (
    generate_esp_dataset,
    SYNTHETIC_SENSOR_COLS,
)
from src.data.loader import (
    load_cmapss,
    load_pump_sensor,
    _sliding_window,
    _compute_rul,
    _split_and_scale,
    TimeSeriesDataset,
)
from src.models.rul_predictor import (
    RULPredictor,
    AsymmetricRULLoss,
    train_rul_epoch,
    evaluate_rul,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml(path):
    with open(path) as f:
        return __import__("yaml").safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train RUL Predictor")
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--dataset",
                        choices=["cmapss", "pump_sensor", "synthetic"],
                        default="cmapss")
    parser.add_argument("--cmapss_subset", default="FD001")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--data_path", default=None, help="Direct CSV path")
    parser.add_argument("--save_dir", default="checkpoints/rul")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_cfg = load_yaml(args.training_config)
    data_cfg = train_cfg["data"]
    training_cfg = train_cfg["training"]
    opt_cfg = train_cfg["optimizer"]

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("Using device: %s", device)

    # ── Load data ─────────────────────────────────────────────────
    if args.dataset == "cmapss":
        data_dir = args.data_dir or str(
            Path(train_cfg["paths"]["raw_data_dir"]) / "cmapss"
        )
        data = load_cmapss(
            data_dir, subset=args.cmapss_subset,
            window_size=data_cfg["window_size"],
            step_size=data_cfg["step_size"],
            val_split=data_cfg["val_split"],
            random_seed=data_cfg["random_seed"],
        )
    elif args.dataset == "synthetic":
        if args.data_path:
            df_s = pd.read_csv(args.data_path, parse_dates=["timestamp"])
        else:
            df_s = generate_esp_dataset(
                n_wells=20, timesteps_per_well=3000,
                failure_prob=0.6, random_seed=42,
            )
        data = _df_to_data(df_s, SYNTHETIC_SENSOR_COLS, data_cfg)
    else:
        path = args.data_path or str(
            Path(train_cfg["paths"]["raw_data_dir"]) / "pump_sensor.csv"
        )
        data = load_pump_sensor(
            path, window_size=data_cfg["window_size"],
            step_size=data_cfg["step_size"],
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            random_seed=data_cfg["random_seed"],
        )

    batch_size = training_cfg["batch_size"]
    n_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        TimeSeriesDataset(data["X_train"], data["y_train"],
                          data.get("rul_train")),
        batch_size=batch_size, shuffle=True, num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TimeSeriesDataset(data["X_val"], data["y_val"],
                          data.get("rul_val")),
        batch_size=batch_size, shuffle=False, num_workers=n_workers,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(data["X_test"], data["y_test"],
                          data.get("rul_test")),
        batch_size=batch_size, shuffle=False, num_workers=n_workers,
    )

    n_features = data["X_train"].shape[-1]
    logger.info("Input features: %d, Window size: %d",
                n_features, data_cfg["window_size"])

    rul_valid = data["rul_train"][data["rul_train"] >= 0]
    output_range = (0.0, float(np.percentile(rul_valid, 99))) if len(rul_valid) > 0 else None

    # ── Build model ───────────────────────────────────────────────
    model = RULPredictor(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        output_range=output_range,
    )
    logger.info("Model parameters: %s",
                f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Training ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5,
    )
    criterion = AsymmetricRULLoss(alpha=2.0)

    best_val_rmse = float("inf")
    patience = training_cfg.get("early_stopping_patience", 15)
    counter = 0
    gradient_clip = training_cfg.get("gradient_clip", 1.0)
    num_epochs = training_cfg["num_epochs"]

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Starting RUL training for %d epochs...", num_epochs)
    for epoch in range(1, num_epochs + 1):
        train_loss = train_rul_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=gradient_clip,
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate_rul(model, val_loader, device)
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_rmse=%.2f | val_mae=%.2f",
                epoch, train_loss, val_metrics["rmse"], val_metrics["mae"],
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                counter = 0
                model.save_pretrained(args.save_dir)
                logger.info("  -> Saved best model (RMSE=%.2f)", best_val_rmse)
            else:
                counter += 5
                if counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

    # ── Final evaluation ──────────────────────────────────────────
    logger.info("Loading best model for final evaluation...")
    model = RULPredictor.from_pretrained(args.save_dir, device=str(device))
    test_metrics = evaluate_rul(model, test_loader, device)
    logger.info("Test RMSE: %.2f", test_metrics["rmse"])
    logger.info("Test MAE:  %.2f", test_metrics["mae"])
    logger.info("Test NASA Score: %.0f", test_metrics["nasa_score"])

    # ── MC Dropout evaluation ─────────────────────────────────────
    mc_samples = train_cfg.get("uncertainty", {}).get("mc_dropout_samples", 50)
    mc_results = []
    mc_targets = []
    for batch in test_loader:
        x = batch["X"].to(device)
        rul = batch["rul"].to(device)
        mask = rul >= 0
        if mask.sum() == 0:
            continue
        mc = model.predict_with_uncertainty(x[mask], n_samples=mc_samples)
        mc_results.append(mc)
        mc_targets.append(rul[mask].cpu().numpy())

    if mc_results:
        all_mean = np.concatenate([r["mean"] for r in mc_results])
        all_std = np.concatenate([r["std"] for r in mc_results])
        all_targets = np.concatenate(mc_targets)

        in_ci = ((all_targets >= all_mean - 1.645 * all_std)
                 & (all_targets <= all_mean + 1.645 * all_std))
        coverage = in_ci.mean()
        avg_uncertainty = all_std.mean()

        logger.info("MC Dropout - Avg uncertainty: %.2f", avg_uncertainty)
        logger.info("MC Dropout - 90%% CI coverage: %.1f%%", coverage * 100)

    logger.info("Model saved to %s", args.save_dir)


def _df_to_data(df, sensor_cols, data_cfg):
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


if __name__ == "__main__":
    main()
