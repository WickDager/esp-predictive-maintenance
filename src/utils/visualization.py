"""
src/utils/visualization.py
============================
Plotting utilities for ESP Predictive Maintenance project.

All plots are Matplotlib-based and return Figure objects (display or save).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Tuple, Dict


# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FAIL_COLOR = "#E24B4A"
NORMAL_COLOR = "#1D9E75"
PRED_COLOR = "#378ADD"
UNCERTAINTY_COLOR = "#EF9F27"


def plot_sensor_overview(
    df: pd.DataFrame,
    sensor_cols: List[str],
    label_col: Optional[str] = "machine_status",
    timestamp_col: Optional[str] = "timestamp",
    n_cols: int = 3,
    figsize_per_plot: Tuple = (6, 2.5),
    title: str = "ESP Sensor Overview",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel sensor time-series with failure annotation.

    Args:
        df: Sensor dataframe.
        sensor_cols: Columns to plot.
        label_col: Column for failure shading (optional).
        timestamp_col: X-axis timestamp column.
        n_cols: Number of columns in subplot grid.
        title: Overall figure title.
    """
    n_plots = len(sensor_cols)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )
    x = df[timestamp_col] if timestamp_col and timestamp_col in df.columns else df.index

    for idx, col in enumerate(sensor_cols):
        row, col_idx = divmod(idx, n_cols)
        ax = axes[row][col_idx]
        ax.plot(x, df[col], color=PRED_COLOR, linewidth=0.8, alpha=0.9)

        # Shade failure regions
        if label_col and label_col in df.columns:
            failure_mask = df[label_col] == "BROKEN" if df[label_col].dtype == object \
                else df[label_col] == 1
            _shade_failures(ax, x, failure_mask)

        ax.set_title(col, fontsize=9, fontweight="medium")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        row, col_idx = divmod(idx, n_cols)
        axes[row][col_idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="medium", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_anomaly_scores(
    timestamps: np.ndarray,
    scores: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    mc_std: Optional[np.ndarray] = None,
    title: str = "Anomaly Score Over Time",
    ylabel: str = "Reconstruction Error (MSE)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot anomaly scores with optional:
      - Ground truth failure shading
      - Threshold line
      - MC Dropout uncertainty band
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Uncertainty band (MC Dropout)
    if mc_std is not None:
        ax.fill_between(
            timestamps, scores - 2 * mc_std, scores + 2 * mc_std,
            alpha=0.25, color=UNCERTAINTY_COLOR, label="±2σ (MC Dropout)"
        )

    # Anomaly score line
    ax.plot(timestamps, scores, color=PRED_COLOR, linewidth=1.2, label="Anomaly score", zorder=3)

    # Threshold
    if threshold is not None:
        ax.axhline(threshold, color=FAIL_COLOR, linestyle="--", linewidth=1.5,
                   label=f"Threshold = {threshold:.4f}", alpha=0.8)

    # Failure shading
    if y_true is not None:
        _shade_failures(ax, timestamps, y_true == 1, label="Known failure")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rul_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ci_low: Optional[np.ndarray] = None,
    ci_high: Optional[np.ndarray] = None,
    title: str = "RUL Prediction vs Ground Truth",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter + line plots for RUL prediction.

    Left: Scatter (predicted vs actual)
    Right: Time-series view (aligned by index)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: scatter ──────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=PRED_COLOR)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual RUL", fontsize=11)
    ax.set_ylabel("Predicted RUL", fontsize=11)
    ax.set_title("Predicted vs Actual RUL", fontsize=12)
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    ax.text(0.05, 0.93, f"RMSE = {rmse:.1f}", transform=ax.transAxes, fontsize=10,
            color="red", va="top")
    ax.legend()

    # ── Right: time-series ─────────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(y_true))
    ax.plot(x, y_true, color=NORMAL_COLOR, linewidth=1.5, label="Actual RUL")
    ax.plot(x, y_pred, color=PRED_COLOR, linewidth=1.2, linestyle="--", label="Predicted RUL")
    if ci_low is not None and ci_high is not None:
        ax.fill_between(x, ci_low, ci_high, alpha=0.2, color=UNCERTAINTY_COLOR,
                        label="90% CI (MC Dropout)")
    ax.set_xlabel("Sample index", fontsize=11)
    ax.set_ylabel("Remaining Useful Life (timesteps)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reconstruction_comparison(
    x_original: np.ndarray,
    x_reconstructed: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    n_sensors_to_show: int = 6,
    title: str = "Sensor Reconstruction (LSTM Autoencoder)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of original vs reconstructed sensor windows.

    Args:
        x_original:      (batch, seq_len, n_sensors)
        x_reconstructed: (batch, seq_len, n_sensors)
        sample_idx:      Which batch sample to plot.
    """
    orig = x_original[sample_idx]         # (seq_len, n_sensors)
    recon = x_reconstructed[sample_idx]   # (seq_len, n_sensors)

    n_sensors = min(n_sensors_to_show, orig.shape[1])
    sensor_names = sensor_names or [f"Sensor {i}" for i in range(n_sensors)]
    t = np.arange(orig.shape[0])

    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, n_sensors * 2), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i in range(n_sensors):
        ax = axes[i]
        ax.plot(t, orig[:, i], color=NORMAL_COLOR, linewidth=1.5, label="Original")
        ax.plot(t, recon[:, i], color=PRED_COLOR, linewidth=1.2,
                linestyle="--", label="Reconstructed")
        # Shade reconstruction error
        ax.fill_between(t, orig[:, i], recon[:, i], alpha=0.25,
                        color=FAIL_COLOR, label="Error")
        ax.set_ylabel(sensor_names[i], fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(title, fontsize=12)

    axes[-1].set_xlabel("Timestep within window", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of mean |SHAP| values.

    Args:
        shap_values: (n_samples, n_features) SHAP values array.
        feature_names: Feature name labels.
        max_display: Top-N features to show.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, max_display * 0.35)))
    colors = [FAIL_COLOR if v > mean_abs.mean() else PRED_COLOR
              for v in mean_abs[order]]
    bars = ax.barh(
        y=np.array(feature_names)[order],
        width=mean_abs[order],
        color=colors, alpha=0.85, edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP Value| (impact on failure probability)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.axvline(mean_abs.mean(), color="gray", linestyle="--",
               alpha=0.6, label="Mean importance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────

def _shade_failures(ax, x, failure_mask, label="Failure", alpha=0.15):
    """Shade regions where failure_mask is True."""
    shaded = False
    for i in range(len(failure_mask)):
        if failure_mask[i] and not shaded:
            start = x[i]
            shaded = True
        elif not failure_mask[i] and shaded:
            ax.axvspan(start, x[i - 1], color=FAIL_COLOR, alpha=alpha,
                       label=label if i == 1 else "")
            shaded = False
    if shaded:
        ax.axvspan(start, x[-1], color=FAIL_COLOR, alpha=alpha, label=label)
