"""
src/data/loader.py
==================
Dataset loaders for:
  - Pump Sensor Data (Kaggle)     : 52 sensors, NORMAL/BROKEN/RECOVERING labels
  - NASA CMAPSS                   : Turbofan RUL benchmark (FD001-FD004)
  - Synthetic ESP data            : From synthetic_generator.py

Each loader returns a unified dict:
  {
    "X_train":   np.ndarray  (N_train, window, features),
    "X_val":     np.ndarray,
    "X_test":    np.ndarray,
    "y_train":   np.ndarray  (binary failure label),
    "y_val":     np.ndarray,
    "y_test":    np.ndarray,
    "rul_train": np.ndarray  (remaining useful life, -1 if unknown),
    "rul_val":   np.ndarray,
    "rul_test":  np.ndarray,
    "feature_names": list[str],
    "scaler":    fitted sklearn scaler,
  }
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Pump Sensor Dataset (primary, from Kaggle)
# ──────────────────────────────────────────────────────────────────

PUMP_SENSOR_COLS = [f"sensor_{i:02d}" for i in range(1, 53)]
PUMP_STATUS_COL = "machine_status"
PUMP_STATUS_MAP = {"NORMAL": 0, "RECOVERING": 0, "BROKEN": 1}


def load_pump_sensor(
    data_path: str,
    window_size: int = 50,
    step_size: int = 1,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    sensor_cols: Optional[list] = None,
) -> Dict:
    """
    Load Kaggle Pump Sensor dataset.

    Download first:
        kaggle datasets download -d nphantawee/pump-sensor-data -p data/raw/
        unzip data/raw/pump-sensor-data.zip -d data/raw/

    Args:
        data_path: Path to 'sensor.csv' file.
        window_size: Sliding window length in timesteps.
        step_size: Step between consecutive windows.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        random_seed: Reproducibility seed.
        sensor_cols: Subset of sensor columns. None = all 52.

    Returns:
        Unified data dict.
    """
    logger.info(f"Loading pump sensor data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cols = sensor_cols if sensor_cols else PUMP_SENSOR_COLS
    # Drop sensors with >20% NaN
    valid_cols = [c for c in cols if c in df.columns and df[c].isna().mean() < 0.2]
    df[valid_cols] = df[valid_cols].ffill().bfill()

    # Binary failure label
    df["failure"] = df[PUMP_STATUS_COL].map(PUMP_STATUS_MAP).fillna(0).astype(int)

    # Compute RUL: for each window ending before a failure, count steps to failure
    failure_idx = df[df["failure"] == 1].index.tolist()
    rul_series = _compute_rul(df["failure"].values)

    X_raw = df[valid_cols].values.astype(np.float32)
    y_raw = df["failure"].values.astype(np.float32)
    rul_raw = rul_series.astype(np.float32)

    # Sliding window
    X_windows, y_windows, rul_windows = _sliding_window(
        X_raw, y_raw, rul_raw, window_size, step_size
    )
    logger.info(f"Windows created: {X_windows.shape}  failures: {y_windows.sum():.0f}")

    return _split_and_scale(
        X_windows, y_windows, rul_windows, valid_cols,
        val_split, test_split, random_seed
    )


# ──────────────────────────────────────────────────────────────────
# NASA CMAPSS Dataset (for RUL benchmark)
# ──────────────────────────────────────────────────────────────────

CMAPSS_SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
CMAPSS_OP_COLS = ["op1", "op2", "op3"]
# Sensors with near-zero variance in FD001 (drop them)
CMAPSS_DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

CMAPSS_SUBSETS = {
    "FD001": {"train": "train_FD001.txt", "test": "test_FD001.txt", "rul": "RUL_FD001.txt"},
    "FD002": {"train": "train_FD002.txt", "test": "test_FD002.txt", "rul": "RUL_FD002.txt"},
    "FD003": {"train": "train_FD003.txt", "test": "test_FD003.txt", "rul": "RUL_FD003.txt"},
    "FD004": {"train": "train_FD004.txt", "test": "test_FD004.txt", "rul": "RUL_FD004.txt"},
}


def load_cmapss(
    data_dir: str,
    subset: str = "FD001",
    window_size: int = 30,
    step_size: int = 1,
    clip_rul: int = 130,
    val_split: float = 0.15,
    random_seed: int = 42,
) -> Dict:
    """
    Load NASA CMAPSS turbofan dataset for RUL prediction.

    Download from:
        https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
    Or via the download script:
        python scripts/download_data.py --dataset cmapss

    Args:
        data_dir: Directory containing train_FD001.txt etc.
        subset: One of FD001, FD002, FD003, FD004.
        window_size: Timesteps per window.
        step_size: Sliding window step.
        clip_rul: Piecewise-linear RUL clipping (standard in literature).
        val_split: Fraction of training engines for validation.
        random_seed: Seed.

    Returns:
        Unified data dict.
    """
    logger.info(f"Loading CMAPSS {subset}")
    files = CMAPSS_SUBSETS[subset]

    col_names = ["unit", "cycle"] + CMAPSS_OP_COLS + CMAPSS_SENSOR_COLS
    train_df = pd.read_csv(
        os.path.join(data_dir, files["train"]),
        sep=" ", header=None, names=col_names, index_col=False
    ).dropna(axis=1)

    test_df = pd.read_csv(
        os.path.join(data_dir, files["test"]),
        sep=" ", header=None, names=col_names, index_col=False
    ).dropna(axis=1)

    test_rul = pd.read_csv(
        os.path.join(data_dir, files["rul"]),
        sep=" ", header=None, names=["RUL"], index_col=False
    ).dropna(axis=1)

    # Compute RUL for training data
    max_cycle = train_df.groupby("unit")["cycle"].max().reset_index()
    max_cycle.columns = ["unit", "max_cycle"]
    train_df = train_df.merge(max_cycle, on="unit")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df["RUL"] = train_df["RUL"].clip(upper=clip_rul)

    # Feature selection: drop low-variance sensors
    use_sensors = [s for s in CMAPSS_SENSOR_COLS if s not in CMAPSS_DROP_SENSORS]
    feature_cols = use_sensors  # ops omitted for simplicity (can be added)

    # Build per-engine windows
    units = train_df["unit"].unique()
    val_units = np.random.default_rng(random_seed).choice(
        units, size=int(len(units) * val_split), replace=False
    )
    train_units = [u for u in units if u not in val_units]

    X_train, y_train, rul_train = _cmapss_windows(
        train_df[train_df["unit"].isin(train_units)], feature_cols, window_size, step_size
    )
    X_val, y_val, rul_val = _cmapss_windows(
        train_df[train_df["unit"].isin(val_units)], feature_cols, window_size, step_size
    )

    # Test set: take last `window_size` cycles per engine
    X_test_list = []
    for unit_id, grp in test_df.groupby("unit"):
        seq = grp[feature_cols].values[-window_size:]
        if len(seq) < window_size:
            seq = np.pad(seq, ((window_size - len(seq), 0), (0, 0)), mode="edge")
        X_test_list.append(seq)
    X_test = np.stack(X_test_list).astype(np.float32)
    rul_test = test_rul["RUL"].values.astype(np.float32)
    # Binary label: failure within 30 cycles
    y_test = (rul_test <= 30).astype(np.float32)

    # Fit scaler on training data
    scaler = StandardScaler()
    n_train, seq_len, n_feat = X_train.shape
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(n_train, seq_len, n_feat)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    logger.info(f"CMAPSS {subset}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "rul_train": rul_train, "rul_val": rul_val, "rul_test": rul_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


# ──────────────────────────────────────────────────────────────────
# PyTorch Dataset wrappers
# ──────────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for windowed time-series data.

    Args:
        X: (N, seq_len, features) float32 array
        y: (N,) binary labels (optional)
        rul: (N,) remaining useful life values (optional)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        rul: Optional[np.ndarray] = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.rul = torch.from_numpy(rul).float() if rul is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {"X": self.X[idx]}
        if self.y is not None:
            sample["y"] = self.y[idx]
        if self.rul is not None:
            sample["rul"] = self.rul[idx]
        return sample


def make_dataloaders(
    data: Dict,
    batch_size: int = 128,
    num_workers: int = 0,
    include_rul: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from unified data dict."""
    train_ds = TimeSeriesDataset(
        data["X_train"], data["y_train"],
        data["rul_train"] if include_rul else None
    )
    val_ds = TimeSeriesDataset(
        data["X_val"], data["y_val"],
        data["rul_val"] if include_rul else None
    )
    test_ds = TimeSeriesDataset(
        data["X_test"], data["y_test"],
        data["rul_test"] if include_rul else None
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────

def _sliding_window(
    X: np.ndarray,
    y: np.ndarray,
    rul: np.ndarray,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply sliding window to raw arrays."""
    X_windows, y_windows, rul_windows = [], [], []
    n = len(X)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(X[start:end])
        # Label = 1 if any failure in window, weighted toward last timestep
        y_windows.append(y[end - 1])
        rul_windows.append(rul[end - 1])
    return (
        np.stack(X_windows).astype(np.float32),
        np.array(y_windows).astype(np.float32),
        np.array(rul_windows).astype(np.float32),
    )


def _compute_rul(failure_flags: np.ndarray) -> np.ndarray:
    """
    Compute per-timestep RUL from binary failure flags.
    RUL = steps until next failure. -1 if no future failure.
    """
    n = len(failure_flags)
    rul = np.full(n, -1, dtype=np.float32)
    next_fail = n  # sentinel
    for i in range(n - 1, -1, -1):
        if failure_flags[i] == 1:
            next_fail = i
        rul[i] = next_fail - i if next_fail < n else -1
    return rul


def _split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    rul: np.ndarray,
    feature_names: list,
    val_split: float,
    test_split: float,
    random_seed: int,
) -> Dict:
    """Train/val/test split + StandardScaler fit on train only."""
    X_tmp, X_test, y_tmp, y_test, rul_tmp, rul_test = train_test_split(
        X, y, rul, test_size=test_split, random_state=random_seed, stratify=y
    )
    val_frac = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val, rul_train, rul_val = train_test_split(
        X_tmp, y_tmp, rul_tmp, test_size=val_frac,
        random_state=random_seed, stratify=y_tmp
    )

    # Fit scaler on train normal windows only (standard practice)
    normal_mask = y_train == 0
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    train_normal = X_train[normal_mask].reshape(-1, n_feat)

    # Safety check: ensure we have enough normal samples
    if len(train_normal) == 0:
        raise ValueError(
            f"No normal (non-failure) samples in training set. "
            f"Cannot fit scaler. y_train distribution: {np.unique(y_train, return_counts=True)}"
        )

    scaler.fit(train_normal)

    # Clip scaler's scale_ to avoid division by near-zero (prevents NaN/Inf)
    scaler.scale_ = np.clip(scaler.scale_, 1e-8, None)

    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    # Final safety: replace any remaining NaN/Inf with clipped values
    X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
    X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
    X_test = np.clip(np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)

    logger.info(
        f"Split: train={X_train.shape} (fail={y_train.mean():.3f}) | "
        f"val={X_val.shape} | test={X_test.shape}"
    )

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "rul_train": rul_train,
        "rul_val": rul_val,
        "rul_test": rul_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def _cmapss_windows(df, feature_cols, window_size, step_size):
    """Build windows per engine unit for CMAPSS."""
    X_list, y_list, rul_list = [], [], []
    for _, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle")
        vals = grp[feature_cols].values.astype(np.float32)
        rul_vals = grp["RUL"].values.astype(np.float32)
        n = len(vals)
        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            X_list.append(vals[start:end])
            rul_list.append(rul_vals[end - 1])
            y_list.append(1 if rul_vals[end - 1] <= 30 else 0)
    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list).astype(np.float32),
        np.stack(rul_list).astype(np.float32),
    )
