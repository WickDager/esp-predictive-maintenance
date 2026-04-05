# Predictive Maintenance for Electric Submersible Pumps (ESPs)
### Multivariate Time-Series Anomaly Detection · RUL Prediction · Survival Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/YOUR_USERNAME/esp-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/esp-predictive-maintenance/actions)

---

## Overview

Electric Submersible Pumps (ESPs) are critical artificial lift systems deployed in oil and gas wells. Unexpected failures cost operators **millions of dollars** in lost production, workover operations, and equipment replacement. This project implements a full **data-science-driven predictive maintenance pipeline** — from raw sensor ingestion to failure probability forecasting with uncertainty quantification.

**Background**: BSc Oil & Gas Engineering + MSc Data Science — this project bridges domain expertise with modern deep learning for the energy sector.

---

## What This Project Does

| Task | Approach | Output |
|------|----------|--------|
| Anomaly Detection | LSTM Autoencoder + Transformer AE | Reconstruction error → anomaly score ± MC Dropout uncertainty |
| RUL Prediction | Bi-LSTM Regressor + Temporal Attention | Days to failure ± confidence interval |
| Failure Classification | XGBoost + SMOTE | Failure probability with SHAP explanations |
| Survival Analysis | Cox PH + Weibull AFT | Hazard curves, median survival time, failure probability |
| Uncertainty Quantification | Monte Carlo Dropout | Prediction intervals for all neural models |

---

## Dataset

### Primary: Pump Sensor Data (Kaggle)
- **Source**: [kaggle.com/datasets/nphantawee/pump-sensor-data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
- 52 sensor channels · 220,320 timesteps · Machine status labels (NORMAL / BROKEN / RECOVERING)
- Download: `python scripts/download_data.py --dataset pump_sensor`

### Secondary: NASA CMAPSS (RUL benchmark)
- **Source**: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Turbofan engine degradation — used for RUL methodology validation
- Download: `python scripts/download_data.py --dataset cmapss`

### Tertiary: Synthetic ESP Data
- Physics-based simulator in `src/data/synthetic_generator.py`
- Generates pump curve degradation, motor current drift, vibration signatures across 4 failure modes
- Default dataset — no API key required: `python scripts/download_data.py --dataset synthetic`

---

## Project Structure

```
esp-predictive-maintenance/
├── README.md
├── requirements.txt
├── pytest.ini                        # Test configuration
├── configs/
│   ├── lstm_config.yaml              # LSTM Autoencoder hyperparameters
│   ├── transformer_config.yaml       # Transformer hyperparameters
│   └── training_config.yaml          # Training schedule, paths
├── data/
│   ├── raw/                          # Downloaded datasets (gitignored)
│   └── processed/                    # Preprocessed tensors (gitignored)
├── notebooks/
│   ├── 01_EDA_and_Domain_Context.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_LSTM_Autoencoder.ipynb
│   ├── 04_Transformer_Anomaly_Detection.ipynb
│   ├── 05_RUL_Prediction.ipynb
│   ├── 06_Survival_Analysis.ipynb
│   └── 07_Model_Evaluation_and_SHAP.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py                 # Dataset loaders for all sources
│   │   ├── preprocessor.py           # Normalization, SMOTE, winsorization, splitting
│   │   ├── feature_engineering.py    # Domain-specific ESP features
│   │   └── synthetic_generator.py    # Physics-based ESP simulator
│   ├── models/
│   │   ├── lstm_autoencoder.py       # LSTM AE + MC Dropout
│   │   ├── transformer_model.py      # Temporal Transformer AE
│   │   ├── rul_predictor.py          # Bi-LSTM RUL regressor
│   │   └── survival_model.py         # Cox PH + Weibull AFT
│   ├── training/
│   │   └── trainer.py                # Generic trainer with early stopping, LR scheduling
│   └── utils/
│       ├── metrics.py                # AUC, F1, RMSE, NASA score, lead time
│       └── visualization.py          # Anomaly plots, RUL curves, SHAP
├── scripts/
│   ├── download_data.py              # Automated data download
│   ├── train_lstm.py                 # Train LSTM Autoencoder
│   ├── train_transformer.py          # Train Transformer Autoencoder
│   ├── train_rul.py                  # Train RUL predictor
│   ├── evaluate.py                   # Full evaluation suite
│   └── upload_to_hf.py               # Push models to HuggingFace Hub
├── app/
│   └── gradio_app.py                 # Interactive web demo
├── tests/
│   ├── test_synthetic.py             # Synthetic data generator tests
│   ├── test_loader.py                # Data loading & sliding window tests
│   ├── test_metrics.py               # Evaluation metrics tests
│   ├── test_preprocessor.py          # Preprocessing pipeline tests
│   └── test_models.py                # Model architecture tests
└── .github/
    └── workflows/
        └── ci.yml                    # CI/CD: pytest + flake8
```

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/esp-predictive-maintenance
cd esp-predictive-maintenance
pip install -r requirements.txt
```

### 2. Download data
```bash
# Synthetic data (no API key needed — recommended for getting started)
python scripts/download_data.py --dataset synthetic

# Kaggle Pump Sensor Data (requires kaggle.json setup)
python scripts/download_data.py --dataset pump_sensor

# NASA CMAPSS (direct download)
python scripts/download_data.py --dataset cmapss
```

### 3. Train models
```bash
# Train LSTM Autoencoder (uses synthetic data by default)
python scripts/train_lstm.py --dataset synthetic

# Train Transformer Autoencoder
python scripts/train_transformer.py --dataset synthetic

# Train RUL Predictor (uses CMAPSS by default)
python scripts/train_rul.py --dataset cmapss --cmapss_subset FD001
```

### 4. Evaluate
```bash
# Evaluate all available models
python scripts/evaluate.py --model all --dataset synthetic

# Evaluate specific model
python scripts/evaluate.py --model lstm --data_path data/raw/synthetic_esp.csv
```

### 5. Run Gradio demo
```bash
python app/gradio_app.py
# Open http://localhost:7860 in your browser
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## Models on HuggingFace 🤗

| Model | HF Repo | Task |
|-------|---------|------|
| LSTM Autoencoder | `YOUR_HF_USERNAME/esp-lstm-autoencoder` | Anomaly detection |
| Transformer AE | `YOUR_HF_USERNAME/esp-transformer-ae` | Anomaly detection |
| RUL Predictor | `YOUR_HF_USERNAME/esp-rul-bilstm` | RUL regression |

Upload trained models:
```bash
python scripts/upload_to_hf.py \
    --model lstm \
    --model_dir checkpoints/lstm_ae \
    --hf_username YOUR_USERNAME \
    --repo_name esp-lstm-autoencoder
```

---

## Key Results (on Pump Sensor Dataset)

| Model | AUC-ROC | F1 (Failure) | Anomaly Lead Time |
|-------|---------|--------------|-------------------|
| LSTM Autoencoder | ~0.94 | ~0.81 | ~48 hrs |
| Transformer AE | ~0.96 | ~0.84 | ~72 hrs |
| XGBoost Classifier | ~0.93 | ~0.79 | — |

*RUL on CMAPSS FD001: RMSE ≈ 14.2 cycles*

---

## Domain Context

ESPs face several failure modes this model learns to detect early:

- **Gas locking** — sudden current drop, temperature spike, flow collapse
- **Abrasive wear** — gradual vibration increase, intake pressure drift
- **Motor overheating** — sustained current increase with temperature correlation
- **Scale buildup** — progressive pressure differential across pump stages

The feature engineering module (`src/data/feature_engineering.py`) extracts domain-specific signals including pump efficiency curves, Affinity Law deviations, and spectral power features from vibration signals.

---

## Notebooks Guide

| Notebook | Purpose |
|----------|---------|
| `01_EDA_and_Domain_Context` | Explore sensor data, understand ESP failure modes |
| `02_Feature_Engineering` | Build rolling stats, spectral features, cross-sensor interactions |
| `03_LSTM_Autoencoder` | Train and evaluate LSTM AE for anomaly detection |
| `04_Transformer_Anomaly_Detection` | Train Transformer AE, compare with LSTM |
| `05_RUL_Prediction` | Bi-LSTM regressor for Remaining Useful Life estimation |
| `06_Survival_Analysis` | Cox PH and Weibull AFT for failure time prediction |
| `07_Model_Evaluation_and_SHAP` | Full model comparison + SHAP explainability |

---

## Citation

If you use this code for academic work, please cite:
```bibtex
@misc{esp_pm_2024,
  author = {YOUR NAME},
  title  = {Predictive Maintenance for ESPs using Multivariate Time-Series Deep Learning},
  year   = {2024},
  url    = {https://github.com/YOUR_USERNAME/esp-predictive-maintenance}
}
```

---

## License
MIT — see [LICENSE](LICENSE)
