"""
scripts/evaluate.py
====================
Full evaluation suite for all trained ESP models.

Evaluates:
  - LSTM Autoencoder: AUC-ROC, AUC-PR, F1, lead time
  - Transformer Autoencoder: same metrics
  - RUL Predictor: RMSE, MAE, NASA Score, MC Dropout coverage
  - Survival Analysis: Concordance index, hazard ratios

Usage:
    python scripts/evaluate.py --model lstm --data_path data/raw/synthetic_esp.csv
    python scripts/evaluate.py --model all --dataset synthetic
    python scripts/evaluate.py --model rul --dataset cmapss --cmapss_subset FD001
"""

import argparse
import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import generate_esp_dataset, SYNTHETIC_SENSOR_COLS
from src.data.loader import (
    load_pump_sensor, load_cmapss,
    _sliding_window, _compute_rul, _split_and_scale,
    make_dataloaders,
)
from src.models.lstm_autoencoder import LSTMAutoencoder, mc_dropout_anomaly_scores
from src.models.transformer_model import TransformerAutoencoder
from src.models.rul_predictor import RULPredictor, evaluate_rul
from src.utils.metrics import (
    anomaly_detection_metrics, rul_metrics,
    early_detection_lead_time, print_metrics_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data_for_eval(dataset: str, data_path: str, window_size: int = 50,
                       step_size: int = 1, val_split: float = 0.15,
                       test_split: float = 0.15, random_seed: int = 42,
                       cmapss_subset: str = "FD001", data_dir: str = None):
    """Load data for evaluation, matching the training format."""
    if dataset == "synthetic":
        if data_path:
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        else:
            df = generate_esp_dataset(
                n_wells=20, timesteps_per_well=3000,
                failure_prob=0.6, random_seed=random_seed,
            )
        df["failure"] = (df["machine_status"] == "BROKEN").astype(int)
        rul_series = _compute_rul(df["failure"].values)
        X_raw = df[SYNTHETIC_SENSOR_COLS].values.astype(np.float32)
        y_raw = df["failure"].values.astype(np.float32)
        rul_raw = rul_series.astype(np.float32)
        X_windows, y_windows, rul_windows = _sliding_window(
            X_raw, y_raw, rul_raw,
            window_size=window_size, step_size=step_size,
        )
        return _split_and_scale(
            X_windows, y_windows, rul_windows, SYNTHETIC_SENSOR_COLS,
            val_split=val_split, test_split=test_split,
            random_seed=random_seed,
        )
    elif dataset == "pump_sensor":
        path = data_path or "data/raw/pump_sensor.csv"
        return load_pump_sensor(
            path, window_size=window_size, step_size=step_size,
            val_split=val_split, test_split=test_split,
            random_seed=random_seed,
        )
    elif dataset == "cmapss":
        data_dir = data_dir or "data/raw/cmapss"
        return load_cmapss(
            data_dir, subset=cmapss_subset,
            window_size=window_size, step_size=step_size,
            val_split=val_split, random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def evaluate_autoencoder(model, test_loader, device, data, model_name="Model"):
    """Evaluate an autoencoder on test data."""
    model.eval()
    model = model.to(device)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["X"].to(device)
            y = batch["y"].cpu().numpy()
            scores = model.anomaly_score(x).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(y)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    metrics = anomaly_detection_metrics(all_labels, all_scores)
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} — Anomaly Detection Metrics")
    logger.info(f"{'='*60}")
    for k, v in metrics.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    # MC Dropout evaluation
    logger.info(f"\nRunning MC Dropout uncertainty analysis (50 samples)...")
    # Gather all test data
    all_X = []
    for batch in test_loader:
        all_X.append(batch["X"])
    X_all = torch.cat(all_X, dim=0).to(device)

    mc_mean, mc_std, _ = mc_dropout_anomaly_scores(model, X_all, n_samples=50)

    # Uncertainty quality: average coefficient of variation
    cv = (mc_std / (mc_mean + 1e-8)).mean()
    logger.info(f"  Avg MC std (uncertainty): {mc_std.mean():.6f}")
    logger.info(f"  Avg CV: {cv:.4f}")

    return metrics


def evaluate_rul_model(model, test_loader, device, model_name="RUL Predictor"):
    """Evaluate RUL predictor."""
    model.eval()
    model = model.to(device)

    test_metrics = evaluate_rul(model, test_loader, device)
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} — RUL Prediction Metrics")
    logger.info(f"{'='*60}")
    for k, v in test_metrics.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    # MC Dropout
    logger.info(f"\nRunning MC Dropout uncertainty analysis (50 samples)...")
    all_mc_mean = []
    all_mc_std = []
    all_targets = []

    for batch in test_loader:
        x = batch["X"].to(device)
        rul = batch["rul"].to(device)
        mask = rul >= 0
        if mask.sum() == 0:
            continue
        mc = model.predict_with_uncertainty(x[mask], n_samples=50)
        all_mc_mean.append(mc["mean"])
        all_mc_std.append(mc["std"])
        all_targets.append(rul[mask].cpu().numpy())

    if all_mc_mean:
        mc_mean = np.concatenate(all_mc_mean)
        mc_std = np.concatenate(all_mc_std)
        targets = np.concatenate(all_targets)

        # 90% CI coverage
        in_ci = (targets >= mc_mean - 1.645 * mc_std) & \
                (targets <= mc_mean + 1.645 * mc_std)
        coverage = in_ci.mean()
        avg_width = (2 * 1.645 * mc_std).mean()

        logger.info(f"  Avg MC std: {mc_std.mean():.2f}")
        logger.info(f"  90% CI avg width: {avg_width:.2f}")
        logger.info(f"  90% CI coverage: {coverage:.1%}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ESP models")
    parser.add_argument("--model", choices=["lstm", "transformer", "rul", "all"],
                        default="all", help="Model(s) to evaluate")
    parser.add_argument("--dataset", choices=["pump_sensor", "cmapss", "synthetic"],
                        default="synthetic", help="Dataset")
    parser.add_argument("--data_path", default=None, help="Direct CSV path")
    parser.add_argument("--data_dir", default=None, help="Data directory (CMAPSS)")
    parser.add_argument("--cmapss_subset", default="FD001")
    parser.add_argument("--device", default=None)
    parser.add_argument("--results_dir", default="results",
                        help="Directory to save evaluation results")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading {args.dataset} data...")
    data = load_data_for_eval(
        dataset=args.dataset,
        data_path=args.data_path,
        cmapss_subset=args.cmapss_subset,
        data_dir=args.data_dir,
    )

    _, _, test_loader = make_dataloaders(
        data, batch_size=128, num_workers=0, include_rul=True,
    )

    all_metrics = {}

    # ── LSTM Autoencoder ──────────────────────────────────────────
    if args.model in ("lstm", "all"):
        lstm_path = "checkpoints/lstm_ae"
        if os.path.exists(os.path.join(lstm_path, "pytorch_model.bin")):
            logger.info("Loading LSTM Autoencoder...")
            lstm_model = LSTMAutoencoder.from_pretrained(lstm_path, device=str(device))
            metrics = evaluate_autoencoder(
                lstm_model, test_loader, device, data,
                model_name="LSTM Autoencoder",
            )
            all_metrics["LSTM Autoencoder"] = metrics
        else:
            logger.warning(f"No LSTM checkpoint found at {lstm_path}, skipping.")

    # ── Transformer Autoencoder ───────────────────────────────────
    if args.model in ("transformer", "all"):
        transformer_path = "checkpoints/transformer_ae"
        if os.path.exists(os.path.join(transformer_path, "pytorch_model.bin")):
            logger.info("Loading Transformer Autoencoder...")
            tf_model = TransformerAutoencoder.from_pretrained(
                transformer_path, device=str(device)
            )
            metrics = evaluate_autoencoder(
                tf_model, test_loader, device, data,
                model_name="Transformer Autoencoder",
            )
            all_metrics["Transformer Autoencoder"] = metrics
        else:
            logger.warning(f"No Transformer checkpoint found at {transformer_path}, skipping.")

    # ── RUL Predictor ─────────────────────────────────────────────
    if args.model in ("rul", "all"):
        rul_path = "checkpoints/rul"
        if os.path.exists(os.path.join(rul_path, "pytorch_model.bin")):
            logger.info("Loading RUL Predictor...")
            rul_model = RULPredictor.from_pretrained(rul_path, device=str(device))
            metrics = evaluate_rul_model(
                rul_model, test_loader, device,
                model_name="RUL Predictor",
            )
            all_metrics["RUL Predictor"] = metrics
        else:
            logger.warning(f"No RUL checkpoint found at {rul_path}, skipping.")

    # ── Print summary ─────────────────────────────────────────────
    if all_metrics:
        print_metrics_table(all_metrics)

        # Save to JSON
        results_path = os.path.join(args.results_dir, "evaluation_results.json")
        # Convert numpy types for JSON serialization
        serializable = {}
        for model_name, m in all_metrics.items():
            serializable[model_name] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in m.items()
            }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
    else:
        logger.warning("No models were evaluated (no checkpoints found).")


if __name__ == "__main__":
    main()
