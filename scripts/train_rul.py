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
import yaml
import logging
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import generate_esp_dataset, SYNTHETIC_SENSOR_COLS
from src.data.loader import (
    load_cmapss, load_pump_sensor,
    _sliding_window, _compute_rul, _split_and_scale,
    make_dataloaders,
)
from src.models.rul_predictor import (
    RULPredictor, AsymmetricRULLoss, train_rul_epoch, evaluate_rul,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train RUL Predictor")
    parser.add_argument("--training_config", default="configs/training_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--dataset", choices=["cmapss", "pump_sensor", "synthetic"],
                        default="cmapss", help="Dataset to use")
    parser.add_argument("--cmapss_subset", default="FD001",
                        help="CMAPSS subset (FD001-FD004)")
    parser.add_argument("--data_dir", default=None,
                        help="Path to CMAPSS data directory")
    parser.add_argument("--data_path", default=None,
                        help="Direct path to CSV (pump sensor or synthetic)")
    parser.add_argument("--save_dir", default="checkpoints/rul",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detects if not provided)")
    args = parser.parse_args()

    train_cfg = load_config(args.training_config)
    data_cfg = train_cfg["data"]
    training_cfg = train_cfg["training"]
    opt_cfg = train_cfg["optimizer"]

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────
    if args.dataset == "cmapss":
        if args.data_dir:
            data_dir = args.data_dir
        else:
            data_dir = Path(train_cfg["paths"]["raw_data_dir"]) / "cmapss"
        data = load_cmapss(
            str(data_dir),
            subset=args.cmapss_subset,
            window_size=data_cfg["window_size"],
            step_size=data_cfg["step_size"],
            val_split=data_cfg["val_split"],
            random_seed=data_cfg["random_seed"],
        )
    elif args.dataset == "synthetic":
        if args.data_path:
            import pandas as pd
            df_synth = pd.read_csv(args.data_path, parse_dates=["timestamp"])
        else:
            df_synth = generate_esp_dataset(
                n_wells=20, timesteps_per_well=3000,
                failure_prob=0.6, random_seed=42,
            )
        df_synth["failure"] = (df_synth["machine_status"] == "BROKEN").astype(int)
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
        if args.data_path:
            data = load_pump_sensor(
                args.data_path,
                window_size=data_cfg["window_size"],
                step_size=data_cfg["step_size"],
                val_split=data_cfg["val_split"],
                test_split=data_cfg["test_split"],
                random_seed=data_cfg["random_seed"],
            )
        else:
            raw_path = Path(train_cfg["paths"]["raw_data_dir"]) / "pump_sensor.csv"
            data = load_pump_sensor(
                str(raw_path),
                window_size=data_cfg["window_size"],
                step_size=data_cfg["step_size"],
                val_split=data_cfg["val_split"],
                test_split=data_cfg["test_split"],
                random_seed=data_cfg["random_seed"],
            )

    # Create dataloaders (include RUL labels)
    train_loader, val_loader, test_loader = make_dataloaders(
        data,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 0),
        include_rul=True,
    )

    n_features = data["X_train"].shape[-1]
    logger.info(f"Input features: {n_features}, Window size: {data_cfg['window_size']}")

    # Clip RUL range for output scaling (optional)
    rul_valid = data["rul_train"][data["rul_train"] >= 0]
    if len(rul_valid) > 0:
        max_rul = float(np.percentile(rul_valid, 99))
        output_range = (0.0, max_rul)
    else:
        output_range = None

    # ── Build model ───────────────────────────────────────────────
    model = RULPredictor(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        output_range=output_range,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
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

    logger.info(f"Starting RUL training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_rul_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=gradient_clip,
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate_rul(model, val_loader, device)
            logger.info(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.2f} | val_mae={val_metrics['mae']:.2f}"
            )

            # Save best
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                counter = 0
                model.save_pretrained(args.save_dir)
                logger.info(f"  → Saved best model (RMSE={best_val_rmse:.2f})")
            else:
                counter += 5
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    # ── Final evaluation ──────────────────────────────────────────
    logger.info("Loading best model for final evaluation...")
    model = RULPredictor.from_pretrained(args.save_dir, device=str(device))
    test_metrics = evaluate_rul(model, test_loader, device)
    logger.info(f"Test RMSE: {test_metrics['rmse']:.2f}")
    logger.info(f"Test MAE:  {test_metrics['mae']:.2f}")
    logger.info(f"Test NASA Score: {test_metrics['nasa_score']:.0f}")

    # ── MC Dropout evaluation ─────────────────────────────────────
    mc_samples = train_cfg.get("uncertainty", {}).get("mc_dropout_samples", 50)
    model.eval()
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

        # Coverage of 90% CI
        in_ci = (all_targets >= all_mean - 1.645 * all_std) & \
                (all_targets <= all_mean + 1.645 * all_std)
        coverage = in_ci.mean()
        avg_uncertainty = all_std.mean()

        logger.info(f"MC Dropout — Avg uncertainty: {avg_uncertainty:.2f}")
        logger.info(f"MC Dropout — 90% CI coverage: {coverage:.1%}")

    logger.info(f"Model saved to {args.save_dir}")


if __name__ == "__main__":
    main()
