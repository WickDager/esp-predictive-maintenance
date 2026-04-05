"""
src/data/feature_engineering.py
=================================
Domain-specific feature engineering for Electric Submersible Pumps (ESPs).

Features are grounded in petroleum engineering:
  - Pump efficiency and deviation from manufacturer pump curves
  - Affinity Law deviations (Q ∝ N, H ∝ N², P ∝ N³)
  - Spectral features from vibration signals (FFT-based)
  - Statistical rolling features (mean, std, kurtosis, skewness)
  - Rate-of-change (first derivative) features
  - Cross-sensor correlations relevant to ESP failure modes

Failure mode mapping to features:
  - Gas locking      : sudden drop in current, temperature spike
  - Abrasive wear    : gradual vibration increase, intake pressure drift
  - Motor overheating: sustained current + temperature correlation
  - Scale buildup    : progressive pressure differential across stages
"""

import numpy as np
import pandas as pd
from scipy import signal  # noqa: F401
from scipy import stats  # noqa: F401
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ──────────────────────────────────────────────────────────────────
# High-level pipeline entry point
# ──────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    freq_hz: float = 1.0,        # sampling frequency (1 Hz typical for ESP SCADA)
    rolling_windows: List[int] = [10, 30, 60],
    fft_n_components: int = 5,
    include_cross_sensor: bool = True,
    pump_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: DataFrame with timestamp index and sensor_cols.
        sensor_cols: List of raw sensor column names.
        freq_hz: Sensor sampling frequency in Hz.
        rolling_windows: Rolling stat window sizes (in samples).
        fft_n_components: Top N FFT magnitudes to extract.
        include_cross_sensor: Whether to add cross-sensor interaction features.
        pump_config: Optional dict with pump design parameters:
            {"rated_flow_bpd": 2000, "rated_head_ft": 8000,
             "rated_power_hp": 100, "rated_rpm": 3500}

    Returns:
        DataFrame with original sensors + engineered features.
    """
    result = df[sensor_cols].copy()

    # 1. Rolling statistical features
    rolling_feats = rolling_statistics(df[sensor_cols], windows=rolling_windows)
    result = pd.concat([result, rolling_feats], axis=1)

    # 2. Rate-of-change (first & second derivative)
    roc_feats = rate_of_change(df[sensor_cols])
    result = pd.concat([result, roc_feats], axis=1)

    # 3. Spectral features from vibration proxy
    # Attempt to auto-detect vibration-like columns
    vib_cols = _detect_vibration_cols(sensor_cols, df)
    if vib_cols:
        spec_feats = spectral_features(df[vib_cols], freq_hz=freq_hz,
                                       n_components=fft_n_components)
        result = pd.concat([result, spec_feats], axis=1)

    # 4. Cross-sensor interactions (domain-driven)
    if include_cross_sensor:
        cross_feats = cross_sensor_features(df[sensor_cols], sensor_cols)
        result = pd.concat([result, cross_feats], axis=1)

    # 5. Pump curve deviation (if config provided)
    if pump_config:
        curve_feats = pump_curve_features(df[sensor_cols], sensor_cols, pump_config)
        result = pd.concat([result, curve_feats], axis=1)

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.ffill().fillna(0)

    return result


# ──────────────────────────────────────────────────────────────────
# Rolling statistics
# ──────────────────────────────────────────────────────────────────

def rolling_statistics(
    df: pd.DataFrame,
    windows: List[int] = [10, 30, 60],
) -> pd.DataFrame:
    """
    Per-sensor rolling mean, std, min, max, kurtosis, skewness.

    These capture both the operating state and the rate of degradation:
    - Rolling std increase → growing vibration / instability
    - Rolling kurtosis spike → impulsive fault signatures
    """
    feat_dict = {}
    for col in df.columns:
        for w in windows:
            r = df[col].rolling(w, min_periods=1)
            feat_dict[f"{col}_rmean_{w}"] = r.mean()
            feat_dict[f"{col}_rstd_{w}"] = r.std()
            feat_dict[f"{col}_rmin_{w}"] = r.min()
            feat_dict[f"{col}_rmax_{w}"] = r.max()
            feat_dict[f"{col}_rkurt_{w}"] = r.kurt()
            feat_dict[f"{col}_rskew_{w}"] = r.skew()
    return pd.DataFrame(feat_dict, index=df.index)


# ──────────────────────────────────────────────────────────────────
# Rate of change (velocity + acceleration)
# ──────────────────────────────────────────────────────────────────

def rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    First and second derivatives of each sensor channel.

    - First derivative  : rate of change (velocity)
    - Second derivative : acceleration (curvature of trend)

    Sudden acceleration in motor current → gas slug / blockage event.
    Sustained first-derivative increase in temperature → thermal runaway.
    """
    feat_dict = {}
    for col in df.columns:
        feat_dict[f"{col}_diff1"] = df[col].diff(1)
        feat_dict[f"{col}_diff2"] = df[col].diff(2)
        # Exponentially weighted moving average of derivative (smoothed trend)
        feat_dict[f"{col}_ewm_diff"] = df[col].diff(1).ewm(span=10).mean()
    return pd.DataFrame(feat_dict, index=df.index)


# ──────────────────────────────────────────────────────────────────
# Spectral features (FFT-based)
# ──────────────────────────────────────────────────────────────────

def spectral_features(
    df: pd.DataFrame,
    freq_hz: float = 1.0,
    n_components: int = 5,
    frame_size: int = 64,
) -> pd.DataFrame:
    """
    Extract dominant FFT magnitudes and their frequencies.

    Vibration spectra are classic for bearing and mechanical wear detection:
    - Fundamental frequency shift → impeller eccentricity
    - Harmonic growth → rotor imbalance
    - Sub-harmonic emergence → bearing damage

    Uses a short-time approach: FFT over the last `frame_size` samples.
    """
    feat_dict = {col: {f: [] for f in
                       [f"{col}_fft_mag_{i}" for i in range(n_components)] +
                       [f"{col}_spectral_centroid", f"{col}_spectral_entropy"]}
                 for col in df.columns}  # noqa: F841

    result_rows = []
    for i in range(len(df)):
        row = {}
        for col in df.columns:
            start = max(0, i - frame_size + 1)
            segment = df[col].iloc[start:i + 1].values
            if len(segment) < 4:
                for k in range(n_components):
                    row[f"{col}_fft_mag_{k}"] = 0.0
                row[f"{col}_spectral_centroid"] = 0.0
                row[f"{col}_spectral_entropy"] = 0.0
                continue

            freqs = np.fft.rfftfreq(len(segment), d=1.0 / freq_hz)
            mags = np.abs(np.fft.rfft(segment))
            # Top-N magnitudes (excluding DC)
            top_idx = np.argsort(mags[1:])[::-1][:n_components] + 1
            for k in range(n_components):
                row[f"{col}_fft_mag_{k}"] = mags[top_idx[k]] if k < len(top_idx) else 0.0
            # Spectral centroid (weighted mean frequency)
            mag_sum = mags.sum() + 1e-10
            row[f"{col}_spectral_centroid"] = float(np.sum(freqs * mags) / mag_sum)
            # Spectral entropy
            p = mags / mag_sum
            row[f"{col}_spectral_entropy"] = float(-np.sum(p * np.log2(p + 1e-10)))
        result_rows.append(row)

    return pd.DataFrame(result_rows, index=df.index)


# ──────────────────────────────────────────────────────────────────
# Cross-sensor features (domain-driven interactions)
# ──────────────────────────────────────────────────────────────────

def cross_sensor_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
) -> pd.DataFrame:
    """
    Physics-informed cross-sensor interaction features for ESPs.

    Key interactions:
      - Current × Voltage → Apparent power (motor load)
      - Temperature / Current → Specific heat index (overheating indicator)
      - Intake Pressure – Discharge Pressure → Differential pressure (scale / blockage)
      - Vibration × Current correlation → Mechanical-electrical coupling
      - Flow × Head → Hydraulic power (vs shaft power = efficiency)
    """
    feat_dict = {}

    # Auto-detect candidate columns by name heuristics
    current_cols = _find_cols(sensor_cols, ["current", "amps", "motor_i", "sensor_11"])
    voltage_cols = _find_cols(sensor_cols, ["voltage", "volts", "motor_v", "sensor_12"])
    temp_cols = _find_cols(sensor_cols, ["temp", "temperature", "motor_t", "sensor_05"])
    pressure_cols = _find_cols(sensor_cols, ["pressure", "press", "psi", "sensor_07", "sensor_08"])
    vibration_cols = _find_cols(sensor_cols, ["vibr", "vib", "accel", "sensor_00", "sensor_01"])

    # Power proxy: I × V
    if current_cols and voltage_cols:
        feat_dict["power_proxy"] = df[current_cols[0]] * df[voltage_cols[0]]
        feat_dict["power_rolling_mean_30"] = feat_dict["power_proxy"].rolling(30, min_periods=1).mean()

    # Thermal load index: T / I (overheating per amp)
    if temp_cols and current_cols:
        feat_dict["thermal_load_idx"] = (
            df[temp_cols[0]] / (df[current_cols[0]].abs() + 1e-3)
        )

    # Differential pressure (pump performance indicator)
    if len(pressure_cols) >= 2:
        feat_dict["diff_pressure"] = df[pressure_cols[1]] - df[pressure_cols[0]]
        feat_dict["diff_pressure_roc"] = feat_dict["diff_pressure"].diff(1)

    # Vibration-current coupling (normalized cross-product)
    if vibration_cols and current_cols:
        vib = df[vibration_cols[0]]
        cur = df[current_cols[0]]
        feat_dict["vib_current_coupling"] = (
            (vib - vib.rolling(30, min_periods=1).mean()) *
            (cur - cur.rolling(30, min_periods=1).mean())
        ).rolling(10, min_periods=1).mean()

    # Rolling Pearson correlation between first two sensors (captures regime changes)
    if len(sensor_cols) >= 2:
        feat_dict["sensor_01_02_corr"] = (
            df[sensor_cols[0]].rolling(30, min_periods=2)
            .corr(df[sensor_cols[1]])
        )

    return pd.DataFrame(feat_dict, index=df.index).fillna(0)


# ──────────────────────────────────────────────────────────────────
# Pump curve deviation features
# ──────────────────────────────────────────────────────────────────

def pump_curve_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    pump_config: dict,
) -> pd.DataFrame:
    """
    Deviation from manufacturer pump curves using Affinity Laws.

    Affinity Laws for centrifugal pumps:
        Q₂/Q₁ = N₂/N₁          (flow vs speed)
        H₂/H₁ = (N₂/N₁)²       (head vs speed)
        P₂/P₁ = (N₂/N₁)³       (power vs speed)

    When degradation occurs, the actual operating point deviates from
    the scaled curve — this deviation is a powerful predictive feature.

    Args:
        pump_config: dict with keys:
            rated_flow_bpd, rated_head_ft, rated_power_hp, rated_rpm
    """
    feat_dict = {}
    rated_flow = pump_config.get("rated_flow_bpd", 2000)
    rated_head = pump_config.get("rated_head_ft", 8000)
    rated_power = pump_config.get("rated_power_hp", 100)
    rated_rpm = pump_config.get("rated_rpm", 3500)

    # Proxy: use sensor columns as best approximations
    flow_col = _find_cols(sensor_cols, ["flow", "rate", "bpd", "sensor_02"])
    head_col = _find_cols(sensor_cols, ["head", "pressure", "psi", "sensor_07"])
    power_col = _find_cols(sensor_cols, ["power", "watt", "kw", "sensor_11"])
    rpm_col = _find_cols(sensor_cols, ["rpm", "speed", "freq", "sensor_04"])

    if rpm_col:
        rpm = df[rpm_col[0]] + 1e-3
        speed_ratio = rpm / rated_rpm
        feat_dict["speed_ratio"] = speed_ratio

        if flow_col:
            expected_flow = rated_flow * speed_ratio
            feat_dict["flow_deviation"] = df[flow_col[0]] - expected_flow
            feat_dict["flow_deviation_pct"] = feat_dict["flow_deviation"] / (rated_flow + 1e-3)

        if head_col:
            expected_head = rated_head * speed_ratio ** 2
            feat_dict["head_deviation"] = df[head_col[0]] - expected_head
            feat_dict["head_deviation_pct"] = feat_dict["head_deviation"] / (rated_head + 1e-3)

        if power_col:
            expected_power = rated_power * speed_ratio ** 3
            feat_dict["power_deviation"] = df[power_col[0]] - expected_power

    # Pump efficiency proxy: (hydraulic power out) / (shaft power in)
    if flow_col and head_col and power_col:
        # η = (Q × ρ × g × H) / P  — simplified proportional form
        hydraulic_power = df[flow_col[0]] * df[head_col[0]]
        shaft_power = df[power_col[0]] + 1e-3
        feat_dict["pump_efficiency_proxy"] = hydraulic_power / shaft_power
        feat_dict["efficiency_rolling_mean_60"] = (
            feat_dict["pump_efficiency_proxy"].rolling(60, min_periods=1).mean()
        )
        feat_dict["efficiency_rolling_std_60"] = (
            feat_dict["pump_efficiency_proxy"].rolling(60, min_periods=1).std()
        )

    return pd.DataFrame(feat_dict, index=df.index).fillna(0)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _find_cols(sensor_cols: List[str], keywords: List[str]) -> List[str]:
    """Find sensor columns matching any keyword (case-insensitive)."""
    matches = []
    for col in sensor_cols:
        if any(kw.lower() in col.lower() for kw in keywords):
            matches.append(col)
    return matches


def _detect_vibration_cols(sensor_cols: List[str], df: pd.DataFrame) -> List[str]:
    """
    Heuristically detect vibration-like columns:
    high-variance sensors with near-zero mean.
    """
    vib_kw = ["vibr", "vib", "accel", "acc", "sensor_00", "sensor_01"]
    explicit = _find_cols(sensor_cols, vib_kw)
    if explicit:
        return explicit
    # Fallback: highest variance columns
    variances = df[sensor_cols].var().sort_values(ascending=False)
    return variances.head(2).index.tolist()
