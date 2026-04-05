"""
app/gradio_app.py
==================
Interactive Gradio demo for ESP Predictive Maintenance.

Features:
  1. Upload a CSV of sensor readings → get anomaly score + RUL prediction
  2. Simulate a well in real-time with sliders for each sensor
  3. View reconstruction plot (which sensors triggered the alarm)
  4. SHAP explanation of the prediction

Deploy to HuggingFace Spaces:
  - Create a new Space (Gradio SDK)
  - Push this file as app.py
  - Add requirements.txt to the Space repo

Run locally:
  python app/gradio_app.py
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(".."))

from src.data.synthetic_generator import generate_esp_dataset, SYNTHETIC_SENSOR_COLS
from src.data.loader import _sliding_window, _compute_rul, _split_and_scale
from src.models.lstm_autoencoder import LSTMAutoencoder, mc_dropout_anomaly_scores
from src.models.rul_predictor import RULPredictor

# ──────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")  # CPU for inference in demo
WINDOW_SIZE = 50
SENSOR_COLS = SYNTHETIC_SENSOR_COLS

# Load or build demo models
def _load_or_build_demo_models():
    """Load trained models if available, otherwise build untrained demo models."""
    lstm_path = os.path.join(os.path.dirname(__file__), "../checkpoints/lstm_ae")
    rul_path  = os.path.join(os.path.dirname(__file__), "../checkpoints/rul")

    n_feat = len(SENSOR_COLS)

    # LSTM Autoencoder
    if os.path.exists(os.path.join(lstm_path, "pytorch_model.bin")):
        print("Loading trained LSTM Autoencoder...")
        lstm_model = LSTMAutoencoder.from_pretrained(lstm_path, device=str(DEVICE))
    else:
        print("No checkpoint found. Using untrained demo model...")
        lstm_model = LSTMAutoencoder(
            input_size=n_feat, hidden_size=64, num_layers=2,
            latent_size=16, seq_len=WINDOW_SIZE
        )
        # Set a demo threshold
        lstm_model.threshold = 0.02

    lstm_model.eval()

    # RUL Predictor
    if os.path.exists(os.path.join(rul_path, "pytorch_model.bin")):
        print("Loading trained RUL Predictor...")
        rul_model = RULPredictor.from_pretrained(rul_path, device=str(DEVICE))
    else:
        rul_model = RULPredictor(
            input_size=n_feat, hidden_size=64, num_layers=2,
            output_range=(0, 200)
        )

    rul_model.eval()
    return lstm_model, rul_model


LSTM_MODEL, RUL_MODEL = _load_or_build_demo_models()


# ──────────────────────────────────────────────────────────────────
# Helper: generate a demo well time series
# ──────────────────────────────────────────────────────────────────

def generate_demo_well(failure_mode: str, n_steps: int = 500):
    """Generate a synthetic well for live demo."""
    from src.data.synthetic_generator import generate_esp_dataset
    df = generate_esp_dataset(
        n_wells=1,
        timesteps_per_well=n_steps,
        failure_prob=0.0 if failure_mode == "normal" else 1.0,
        random_seed=42,
    )
    # Patch failure mode if specified
    if failure_mode != "normal":
        from src.data.synthetic_generator import (
            _inject_gas_locking, _inject_abrasive_wear,
            _inject_motor_overheating, _inject_scale_buildup
        )
        rng = np.random.default_rng(42)
        failure_start = int(n_steps * 0.6)
        signals = {col: df[col].values.copy() for col in SENSOR_COLS}

        if failure_mode == "gas_locking":
            curr, temp, pres, flow = _inject_gas_locking(
                signals["motor_current_A"], signals["motor_temperature_C"],
                signals["intake_pressure_psi"], signals["flow_rate_bpd"],
                failure_start, n_steps, rng
            )
            df["motor_current_A"] = curr
            df["motor_temperature_C"] = temp
            df["intake_pressure_psi"] = pres
            df["flow_rate_bpd"] = flow
        elif failure_mode == "abrasive_wear":
            vx, vy, ip = _inject_abrasive_wear(
                signals["vibration_x_g"], signals["vibration_y_g"],
                signals["intake_pressure_psi"], failure_start, n_steps, rng
            )
            df["vibration_x_g"] = vx; df["vibration_y_g"] = vy
            df["intake_pressure_psi"] = ip
        elif failure_mode == "motor_overheating":
            curr, temp, res = _inject_motor_overheating(
                signals["motor_current_A"], signals["motor_temperature_C"],
                signals["winding_resistance_ohm"], failure_start, n_steps, rng
            )
            df["motor_current_A"] = curr; df["motor_temperature_C"] = temp
            df["winding_resistance_ohm"] = res
        elif failure_mode == "scale_buildup":
            dp, fl = _inject_scale_buildup(
                signals["discharge_pressure_psi"], signals["flow_rate_bpd"],
                failure_start, n_steps, rng
            )
            df["discharge_pressure_psi"] = dp; df["flow_rate_bpd"] = fl

        df["machine_status"] = np.where(df.index >= failure_start, "BROKEN", "NORMAL")
    return df


# ──────────────────────────────────────────────────────────────────
# Tab 1: Live simulation
# ──────────────────────────────────────────────────────────────────

def run_simulation(failure_mode: str, n_steps: int, mc_samples: int):
    """Generate well data and run anomaly detection + RUL prediction."""
    df = generate_demo_well(failure_mode, n_steps=n_steps)
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().fillna(0)

    # Normalise with simple z-score per sensor
    mean = df[SENSOR_COLS].mean()
    std  = df[SENSOR_COLS].std().replace(0, 1)
    X_norm = ((df[SENSOR_COLS] - mean) / std).values.astype(np.float32)

    # Build windows
    if len(X_norm) < WINDOW_SIZE:
        return None, "Need more data (increase n_steps)", ""

    X_windows = np.stack([
        X_norm[i:i + WINDOW_SIZE]
        for i in range(0, len(X_norm) - WINDOW_SIZE + 1, 5)
    ]).astype(np.float32)

    y_true = np.array([
        1 if (df["machine_status"].iloc[i + WINDOW_SIZE - 1] == "BROKEN") else 0
        for i in range(0, len(X_norm) - WINDOW_SIZE + 1, 5)
    ])

    X_t = torch.from_numpy(X_windows).float()

    # MC Dropout anomaly scores
    mc_mean, mc_std, _ = mc_dropout_anomaly_scores(LSTM_MODEL, X_t, n_samples=mc_samples)

    # RUL prediction
    rul_mc = RUL_MODEL.predict_with_uncertainty(X_t, n_samples=mc_samples)

    threshold = LSTM_MODEL.threshold or float(np.percentile(mc_mean[y_true == 0], 95))

    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = np.arange(len(mc_mean))

    # Panel 1: Key sensors
    ax = axes[0]
    for col_idx, col in enumerate(["motor_current_A", "motor_temperature_C",
                                     "vibration_x_g", "flow_rate_bpd"]):
        # Downsample to window pace
        vals = df[col].values[WINDOW_SIZE-1::5][:len(t)]
        ax.plot(t, vals / (vals.max() + 1e-8), label=col, linewidth=1, alpha=0.8)
    # Shade failure
    fail_mask = y_true == 1
    if fail_mask.any():
        ax.fill_between(t, 0, 1, where=fail_mask,
                        color="#E24B4A", alpha=0.15, label="Failure region")
    ax.set_ylabel("Normalised value", fontsize=9)
    ax.set_title("Sensor signals (normalised)", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # Panel 2: Anomaly score
    ax = axes[1]
    ax.fill_between(t, mc_mean - 2 * mc_std, mc_mean + 2 * mc_std,
                    alpha=0.2, color="#EF9F27", label="±2σ MC Dropout")
    ax.plot(t, mc_mean, color="#378ADD", linewidth=1.5, label="Anomaly score")
    ax.axhline(threshold, color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")
    if fail_mask.any():
        ax.fill_between(t, 0, mc_mean.max() * 1.1, where=fail_mask,
                        color="#E24B4A", alpha=0.1)
    ax.set_ylabel("Recon. error (MSE)", fontsize=9)
    ax.set_title("LSTM Autoencoder anomaly score (MC Dropout uncertainty)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    # Panel 3: RUL
    ax = axes[2]
    ax.fill_between(t, rul_mc["ci_low"], rul_mc["ci_high"],
                    alpha=0.2, color="#EF9F27", label="90% CI")
    ax.plot(t, rul_mc["mean"], color="#1D9E75", linewidth=1.5, label="Predicted RUL")
    ax.axhline(48, color="#E24B4A", linestyle=":", linewidth=1,
               label="Alert: RUL < 48 h")
    ax.set_ylabel("RUL (timesteps)", fontsize=9)
    ax.set_xlabel("Window index", fontsize=9)
    ax.set_title("Bi-LSTM RUL prediction with uncertainty", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        f"ESP Predictive Maintenance — Mode: {failure_mode.replace('_', ' ').title()}",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    # ── Summary text ─────────────────────────────────────────────
    n_alarms = int((mc_mean > threshold).sum())
    last_rul  = float(rul_mc["mean"][-1])
    last_std  = float(rul_mc["std"][-1])
    status    = "🔴 FAILURE DETECTED" if n_alarms > 0 else "🟢 NORMAL OPERATION"

    summary = f"""
**{status}**

| Metric | Value |
|--------|-------|
| Failure mode | {failure_mode.replace('_', ' ').title()} |
| Alarm triggers | {n_alarms} / {len(mc_mean)} windows |
| Latest anomaly score | {mc_mean[-1]:.5f} |
| Threshold | {threshold:.5f} |
| Predicted RUL | **{last_rul:.0f}** ± {last_std:.0f} timesteps |
| Urgent alert | {'⚠️ YES (RUL < 48)' if last_rul < 48 else '✅ NO'} |
"""
    return fig, summary, f"Analysed {len(mc_mean)} windows from {n_steps} sensor readings."


# ──────────────────────────────────────────────────────────────────
# Tab 2: CSV Upload
# ──────────────────────────────────────────────────────────────────

def predict_from_csv(file_obj):
    """Accept a CSV upload and return predictions."""
    if file_obj is None:
        return None, "No file uploaded.", ""

    try:
        df = pd.read_csv(file_obj.name)
    except Exception as e:
        return None, f"Error reading CSV: {e}", ""

    # Find available sensor columns
    available = [c for c in SENSOR_COLS if c in df.columns]
    if len(available) < 3:
        return None, (
            f"CSV must contain sensor columns. Expected: {SENSOR_COLS[:5]}...\n"
            f"Found: {list(df.columns)}\n\n"
            "Tip: Use the synthetic data generator to produce a compatible CSV:\n"
            "  python -c \"from src.data.synthetic_generator import *; "
            "df=generate_esp_dataset(); df.to_csv('my_data.csv',index=False)\""
        ), ""

    df[available] = df[available].ffill().fillna(0)
    X_raw = df[available].values.astype(np.float32)

    # Pad missing sensors with zeros
    if len(available) < len(SENSOR_COLS):
        n_missing = len(SENSOR_COLS) - len(available)
        X_raw = np.concatenate([X_raw, np.zeros((len(X_raw), n_missing))], axis=1)

    # Normalise
    mean = X_raw.mean(axis=0)
    std  = X_raw.std(axis=0) + 1e-8
    X_norm = ((X_raw - mean) / std).astype(np.float32)

    if len(X_norm) < WINDOW_SIZE:
        return None, f"Need at least {WINDOW_SIZE} rows. Got {len(X_norm)}.", ""

    X_windows = np.stack([
        X_norm[i:i + WINDOW_SIZE]
        for i in range(0, len(X_norm) - WINDOW_SIZE + 1, 5)
    ]).astype(np.float32)

    X_t = torch.from_numpy(X_windows).float()
    mc_mean, mc_std, _ = mc_dropout_anomaly_scores(LSTM_MODEL, X_t, n_samples=30)
    rul_mc = RUL_MODEL.predict_with_uncertainty(X_t, n_samples=30)
    threshold = LSTM_MODEL.threshold or float(np.percentile(mc_mean, 95))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(mc_mean))

    ax = axes[0]
    ax.fill_between(t, mc_mean - 2 * mc_std, mc_mean + 2 * mc_std,
                    alpha=0.2, color="#EF9F27", label="±2σ")
    ax.plot(t, mc_mean, color="#378ADD", linewidth=1.5, label="Anomaly score")
    ax.axhline(threshold, color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_title("Anomaly Score", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.fill_between(t, rul_mc["ci_low"], rul_mc["ci_high"], alpha=0.2,
                    color="#EF9F27", label="90% CI")
    ax.plot(t, rul_mc["mean"], color="#1D9E75", linewidth=1.5, label="Predicted RUL")
    ax.axhline(48, color="#E24B4A", linestyle=":", linewidth=1, label="Alert threshold")
    ax.set_xlabel("Window index")
    ax.set_title("RUL Prediction", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()

    n_alarms = int((mc_mean > threshold).sum())
    summary = f"**{n_alarms} alarm windows** detected out of {len(mc_mean)} total. " \
              f"Final RUL estimate: **{rul_mc['mean'][-1]:.0f}** ± {rul_mc['std'][-1]:.0f} timesteps."
    return fig, summary, f"Processed {len(df)} rows from uploaded CSV."


# ──────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────

TITLE = "# ESP Predictive Maintenance — Interactive Demo"
DESCRIPTION = """
Predictive maintenance for **Electric Submersible Pumps (ESPs)** using deep learning.

**Models**: LSTM Autoencoder (anomaly detection) + Bi-LSTM (RUL prediction) with Monte Carlo Dropout uncertainty.

**Data**: Synthetic physics-based ESP sensor simulation. See [GitHub repo](https://github.com/YOUR_USERNAME/esp-predictive-maintenance) for full code.
"""

with gr.Blocks(title="ESP Predictive Maintenance") as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ── Tab 1: Simulation ─────────────────────────────────────
        with gr.TabItem("Live Simulation"):
            gr.Markdown("### Simulate an ESP well and detect failures in real-time")
            with gr.Row():
                failure_mode = gr.Dropdown(
                    choices=["normal", "gas_locking", "abrasive_wear",
                             "motor_overheating", "scale_buildup"],
                    value="gas_locking",
                    label="Failure Mode",
                )
                n_steps = gr.Slider(200, 2000, value=600, step=100,
                                    label="Simulation length (timesteps)")
                mc_samples = gr.Slider(10, 100, value=30, step=10,
                                       label="MC Dropout samples")

            run_btn = gr.Button("Run Simulation", variant="primary")

            with gr.Row():
                plot_out = gr.Plot(label="Results")
            with gr.Row():
                summary_out = gr.Markdown()
                info_out    = gr.Textbox(label="Info", interactive=False)

            run_btn.click(
                fn=run_simulation,
                inputs=[failure_mode, n_steps, mc_samples],
                outputs=[plot_out, summary_out, info_out],
            )

        # ── Tab 2: CSV Upload ─────────────────────────────────────
        with gr.TabItem("Upload CSV"):
            gr.Markdown(
                "### Upload your own ESP sensor CSV\n"
                f"Expected sensor columns: `{', '.join(SENSOR_COLS[:5])}...`\n\n"
                "Minimum **50 rows** required (one row per sensor reading)."
            )
            file_input = gr.File(label="Upload CSV file", file_types=[".csv"])
            upload_btn = gr.Button("Predict", variant="primary")
            with gr.Row():
                csv_plot    = gr.Plot()
                csv_summary = gr.Markdown()
                csv_info    = gr.Textbox(label="Info", interactive=False)

            upload_btn.click(
                fn=predict_from_csv,
                inputs=[file_input],
                outputs=[csv_plot, csv_summary, csv_info],
            )

        # ── Tab 3: About ──────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown("""
## About this project

**Author**: Your Name | BSc Oil & Gas Engineering + MSc Data Science

### Models
| Model | Architecture | Task |
|-------|-------------|------|
| LSTM Autoencoder | Bi-LSTM encoder → LSTM decoder | Unsupervised anomaly detection |
| RUL Predictor | Bi-LSTM + attention pooling | Remaining Useful Life regression |

Both models use **Monte Carlo Dropout** for uncertainty quantification —
each prediction comes with a confidence interval, not just a point estimate.

### ESP Failure Modes Detected
- **Gas locking** — free gas enters the pump, causing current oscillations
- **Abrasive wear** — sand/solids erosion increases vibration
- **Motor overheating** — winding insulation degradation (temperature + resistance)
- **Scale buildup** — carbonate/sulfate deposition increases differential pressure

### Links
- [GitHub Repository](https://github.com/YOUR_USERNAME/esp-predictive-maintenance)
- [HuggingFace Models](https://huggingface.co/YOUR_USERNAME)
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   # Set True to get a public Gradio link
        debug=False,
    )
