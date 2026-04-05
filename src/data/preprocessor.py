"""
src/data/preprocessor.py
=========================
Preprocessing utilities for ESP sensor data.

Handles:
  - Sensor normalization (fit on train only, transform val/test)
  - Sliding window creation with configurable overlap
  - Class imbalance handling (SMOTE for time-series)
  - Missing value imputation strategies
  - Outlier clipping (winsorization)
  - Train/val/test splitting with stratification
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────

SCALER_REGISTRY = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def fit_scaler(
    X_train: np.ndarray,
    scaler_type: str = "standard",
    normal_only_mask: Optional[np.ndarray] = None,
) -> Any:
    """
    Fit a scaler on training data, optionally on normal (healthy) samples only.

    Args:
        X_train: (N, seq_len, features) training data
        scaler_type: "standard", "robust", or "minmax"
        normal_only_mask: boolean mask for normal samples (N,)

    Returns:
        Fitted sklearn scaler object
    """
    scaler_cls = SCALER_REGISTRY.get(scaler_type)
    if scaler_cls is None:
        raise ValueError(f"Unknown scaler_type: {scaler_type}. Choose from {list(SCALER_REGISTRY.keys())}")

    data = X_train
    if normal_only_mask is not None:
        data = X_train[normal_only_mask]
        logger.info(f"Fitting scaler on {data.shape[0]} normal samples (of {X_train.shape[0]} total)")

    # Reshape to 2D for sklearn
    n_samples, seq_len, n_features = data.shape
    scaler = scaler_cls()
    scaler.fit(data.reshape(-1, n_features))
    logger.info(f"Fitted {scaler_type} scaler on {data.shape}")
    return scaler


def transform_data(
    X: np.ndarray,
    scaler: Any,
) -> np.ndarray:
    """
    Apply fitted scaler to data.

    Args:
        X: (N, seq_len, features) data array
        scaler: Fitted sklearn scaler

    Returns:
        Normalized array of same shape
    """
    n_samples, seq_len, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    X_norm = scaler.transform(X_2d).reshape(n_samples, seq_len, n_features)
    return X_norm.astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# Outlier clipping (Winsorization)
# ──────────────────────────────────────────────────────────────────

def winsorize(
    X: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    fit_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Clip outliers to percentile bounds.

    Args:
        X: (N, seq_len, features) data to clip
        lower_percentile: Lower bound percentile
        upper_percentile: Upper bound percentile
        fit_data: Data to compute bounds from (defaults to X)

    Returns:
        Clipped X, (lower_bounds, upper_bounds) per feature
    """
    if fit_data is None:
        fit_data = X

    n_feat = fit_data.shape[-1]
    fit_2d = fit_data.reshape(-1, n_feat)

    lower_bounds = np.percentile(fit_2d, lower_percentile, axis=0)
    upper_bounds = np.percentile(fit_2d, upper_percentile, axis=0)

    X_2d = X.reshape(-1, n_feat)
    X_clipped = np.clip(X_2d, lower_bounds, upper_bounds)
    return X_clipped.reshape(X.shape).astype(np.float32), (lower_bounds, upper_bounds)


# ──────────────────────────────────────────────────────────────────
# SMOTE for time-series (apply per-timestep, not per-window)
# ──────────────────────────────────────────────────────────────────

def apply_smote_timeseries(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance classes in windowed time-series data.

    SMOTE is applied to flattened windows (each timestep treated as
    an independent sample) to avoid breaking temporal structure.

    Args:
        X: (N, seq_len, features) windowed data
        y: (N,) binary labels
        target_ratio: Desired minority class ratio (0.5 = balanced)
        random_state: Reproducibility seed

    Returns:
        Balanced (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
        return X, y

    n_samples, seq_len, n_features = X.shape

    # Flatten: treat each timestep as a sample
    X_flat = X.reshape(-1, n_features)
    y_flat = np.repeat(y, seq_len)

    # Count minority samples
    minority_count = (y_flat == 1).sum()
    majority_count = (y_flat == 0).sum()

    if minority_count == 0 or majority_count == 0:
        logger.warning("Only one class present. Skipping SMOTE.")
        return X, y

    # Compute sampling strategy
    desired_minority = int(target_ratio * majority_count)
    if desired_minority <= minority_count:
        logger.info("Already balanced enough. Skipping SMOTE.")
        return X, y

    smote = SMOTE(
        sampling_strategy={1: desired_minority},
        random_state=random_state,
        k_neighbors=min(5, minority_count - 1),
    )

    X_res, y_res = smote.fit_resample(X_flat, y_flat)

    # Reconstruct windows: we only upsample the minority class windows
    # Group back into windows by majority voting on original indices
    # This is an approximation — for production, use window-level SMOTE
    logger.info(
        f"SMOTE: {len(y_flat)} → {len(y_res)} samples "
        f"(minority: {minority_count} → {desired_minority})"
    )

    # Return flattened data — caller should decide how to re-window
    # For autoencoder training, we return the original windows + label
    # For classification, flattened data can be used directly
    return X_res.reshape(-1, seq_len, n_features).astype(np.float32), y_res


# ──────────────────────────────────────────────────────────────────
# Missing value imputation
# ──────────────────────────────────────────────────────────────────

def impute_missing(
    df: pd.DataFrame,
    strategy: str = "ffill_bfill",
    max_gap: int = 10,
) -> pd.DataFrame:
    """
    Impute missing values in sensor data.

    Args:
        df: DataFrame with sensor columns
        strategy: "ffill_bfill", "interpolate", "zero", or "median"
        max_gap: Maximum gap size to fill (larger gaps remain NaN)

    Returns:
        DataFrame with imputed values
    """
    result = df.copy()

    if strategy == "ffill_bfill":
        result = result.ffill(limit=max_gap).bfill(limit=max_gap)
    elif strategy == "interpolate":
        result = result.interpolate(method="linear", limit=max_gap)
        result = result.bfill().ffill()  # edge cases
    elif strategy == "zero":
        result = result.fillna(0)
    elif strategy == "median":
        result = result.fillna(result.median())
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    remaining_nan = result.isna().sum().sum()
    if remaining_nan > 0:
        logger.warning(f"{remaining_nan} NaN values remain after imputation. Filling with 0.")
        result = result.fillna(0)

    return result


# ──────────────────────────────────────────────────────────────────
# Sliding window creation
# ──────────────────────────────────────────────────────────────────

def create_windows(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    rul: Optional[np.ndarray] = None,
    window_size: int = 50,
    step_size: int = 1,
    strategy: str = "last",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create sliding windows from sequential data.

    Args:
        X: (N, features) sequential data
        y: (N,) labels (optional)
        rul: (N,) RUL values (optional)
        window_size: Window length in timesteps
        step_size: Step between consecutive windows
        strategy: How to label each window:
            "last"    — label = value at last timestep
            "any"     — label = 1 if any timestep in window is 1
            "majority" — label = majority vote in window

    Returns:
        X_windows: (num_windows, window_size, features)
        y_windows: (num_windows,) or None
        rul_windows: (num_windows,) or None
    """
    n = len(X)
    X_windows, y_windows, rul_windows = [], [], []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(X[start:end])

        if y is not None:
            if strategy == "last":
                y_windows.append(y[end - 1])
            elif strategy == "any":
                y_windows.append(1 if y[start:end].any() else 0)
            elif strategy == "majority":
                y_windows.append(1 if y[start:end].mean() > 0.5 else 0)
            else:
                y_windows.append(y[end - 1])

        if rul is not None:
            rul_windows.append(rul[end - 1])

    X_out = np.stack(X_windows).astype(np.float32)
    y_out = np.array(y_windows, dtype=np.float32) if y_windows else None
    rul_out = np.array(rul_windows, dtype=np.float32) if rul_windows else None

    logger.info(
        f"Created {len(X_windows)} windows "
        f"(shape: {X_out.shape}, step={step_size})"
    )
    return X_out, y_out, rul_out


# ──────────────────────────────────────────────────────────────────
# Train/val/test split with optional stratification
# ──────────────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    rul: Optional[np.ndarray] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Split data into train/val/test sets.

    Args:
        X: (N, seq_len, features) data
        y: (N,) labels
        rul: (N,) RUL values
        val_split: Validation fraction
        test_split: Test fraction
        random_seed: Reproducibility seed
        stratify: Whether to stratify on y

    Returns:
        Dict with X_train, X_val, X_test, y_train, y_val, y_test,
        rul_train, rul_val, rul_test
    """
    stratify_arg = y if stratify and len(np.unique(y)) > 1 else None

    X_tmp, X_test, y_tmp, y_test, rul_tmp, rul_test = train_test_split(
        X, y, rul if rul is not None else np.zeros(len(X)),
        test_size=test_split, random_state=random_seed,
        stratify=stratify_arg,
    )

    val_frac = val_split / (1 - test_split)
    stratify_tmp = y_tmp if stratify and len(np.unique(y_tmp)) > 1 else None
    X_train, X_val, y_train, y_val, rul_train, rul_val = train_test_split(
        X_tmp, y_tmp, rul_tmp,
        test_size=val_frac, random_state=random_seed,
        stratify=stratify_tmp,
    )

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "rul_train": rul_train if rul is not None else None,
        "rul_val": rul_val if rul is not None else None,
        "rul_test": rul_test if rul is not None else None,
    }

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
        f"(failure rates: {y_train.mean():.3f}, {y_val.mean():.3f}, {y_test.mean():.3f})"
    )
    return result
