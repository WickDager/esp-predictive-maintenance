"""
src/data/synthetic_generator.py
================================
Physics-based synthetic ESP sensor data generator.

Generates realistic multivariate time-series data that simulates
common ESP failure modes. Useful for:
  - Demonstrating the model on any machine (no data download required)
  - Ablation studies with known ground-truth failure timing
  - Augmenting real datasets

Modeled sensors (ESP-like):
  - motor_current_A        : Motor current draw (Amps)
  - motor_temperature_C    : Motor winding temperature (°C)
  - intake_pressure_psi    : Pump intake pressure (PSI)
  - discharge_pressure_psi : Pump discharge pressure (PSI)
  - vibration_x_g          : Vibration, X-axis (g)
  - vibration_y_g          : Vibration, Y-axis (g)
  - flow_rate_bpd          : Produced fluid flow rate (barrels/day)
  - motor_speed_rpm        : Motor speed (RPM)
  - winding_resistance_ohm : Motor winding resistance (Ohms)
  - fluid_temperature_C    : Produced fluid temperature (°C)

Failure modes:
  - "gas_locking"       : Current drop, temperature spike, pressure oscillation
  - "abrasive_wear"     : Gradual vibration increase, intake pressure drift
  - "motor_overheating" : Sustained current + temperature rise
  - "scale_buildup"     : Progressive differential pressure increase
  - "normal"            : No failure, just operational noise
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────

def generate_esp_dataset(
    n_wells: int = 10,
    timesteps_per_well: int = 5000,
    failure_prob: float = 0.6,
    sampling_interval_sec: int = 60,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic ESP sensor dataset for multiple wells.

    Args:
        n_wells: Number of simulated ESP installations.
        timesteps_per_well: Sensor readings per well.
        failure_prob: Fraction of wells that experience a failure.
        sampling_interval_sec: Seconds between readings.
        random_seed: Reproducibility seed.

    Returns:
        DataFrame with columns: [timestamp, well_id, sensor_cols...,
                                  machine_status, failure_mode, rul]
    """
    rng = np.random.default_rng(random_seed)
    all_records = []

    for well_id in range(n_wells):
        seed_i = int(rng.integers(0, 100000))

        # Decide failure mode for this well
        if rng.random() < failure_prob:
            mode = rng.choice(["gas_locking", "abrasive_wear",
                               "motor_overheating", "scale_buildup"])
            failure_start = int(rng.uniform(0.5, 0.9) * timesteps_per_well)
        else:
            mode = "normal"
            failure_start = timesteps_per_well + 1  # never

        df_well = _simulate_well(
            well_id=well_id,
            n_steps=timesteps_per_well,
            failure_mode=mode,
            failure_start=failure_start,
            sampling_interval_sec=sampling_interval_sec,
            rng=np.random.default_rng(seed_i),
        )
        all_records.append(df_well)

    dataset = pd.concat(all_records, ignore_index=True)
    return dataset


# ──────────────────────────────────────────────────────────────────
# Per-well simulation
# ──────────────────────────────────────────────────────────────────

def _simulate_well(
    well_id: int,
    n_steps: int,
    failure_mode: str,
    failure_start: int,
    sampling_interval_sec: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate one ESP well's sensor trajectory."""

    t = np.arange(n_steps)

    # ── Baseline (healthy) sensor values ──────────────────────────
    # Add well-specific offset to simulate different operating conditions
    offset_current = rng.uniform(-5, 5)
    offset_temp = rng.uniform(-10, 10)
    offset_flow = rng.uniform(-200, 200)

    # Diurnal-like oscillation (surface temperature affects motor temp)
    diurnal = np.sin(2 * np.pi * t / (24 * 3600 / sampling_interval_sec))

    motor_current = (
        80 + offset_current
        + 3 * np.sin(2 * np.pi * t / 300)           # production cycle
        + 0.5 * diurnal
        + rng.normal(0, 0.8, n_steps)
    )
    motor_temp = (
        85 + offset_temp
        + 0.05 * (motor_current - 80)                # thermal coupling to current
        + 2 * diurnal
        + rng.normal(0, 1.2, n_steps)
    )
    intake_pressure = (
        1200 + rng.normal(0, 15, n_steps)
        - 0.02 * t                                    # reservoir depletion trend
    )
    discharge_pressure = (
        4500 + rng.normal(0, 20, n_steps)
    )
    vib_x = rng.normal(0, 0.05, n_steps) + 0.02 * np.abs(np.sin(2 * np.pi * t / 120))
    vib_y = rng.normal(0, 0.04, n_steps) + 0.015 * np.abs(np.sin(2 * np.pi * t / 118))
    flow_rate = (
        2000 + offset_flow
        + 50 * np.sin(2 * np.pi * t / 500)
        + rng.normal(0, 30, n_steps)
    )
    motor_speed = 3500 + rng.normal(0, 10, n_steps)
    winding_resistance = 2.5 + rng.normal(0, 0.05, n_steps)
    fluid_temp = (
        65 + rng.normal(0, 2, n_steps)
        + 0.003 * motor_temp                          # heat transfer from motor
    )

    # ── Inject failure-mode degradation ───────────────────────────
    if failure_mode == "gas_locking":
        motor_current, motor_temp, intake_pressure, flow_rate = _inject_gas_locking(
            motor_current, motor_temp, intake_pressure, flow_rate,
            failure_start, n_steps, rng
        )
    elif failure_mode == "abrasive_wear":
        vib_x, vib_y, intake_pressure = _inject_abrasive_wear(
            vib_x, vib_y, intake_pressure, failure_start, n_steps, rng
        )
    elif failure_mode == "motor_overheating":
        motor_current, motor_temp, winding_resistance = _inject_motor_overheating(
            motor_current, motor_temp, winding_resistance, failure_start, n_steps, rng
        )
    elif failure_mode == "scale_buildup":
        discharge_pressure, flow_rate = _inject_scale_buildup(
            discharge_pressure, flow_rate, failure_start, n_steps, rng
        )

    # ── Labels ────────────────────────────────────────────────────
    machine_status = np.where(t >= failure_start, "BROKEN", "NORMAL")
    # Brief RECOVERING period after failure (if we simulate restart)
    recovering_end = min(failure_start + 50, n_steps)
    machine_status[failure_start:recovering_end] = "RECOVERING"

    rul = np.where(
        t < failure_start,
        failure_start - t,
        0
    ).astype(float)

    start_ts = datetime(2023, 1, 1)
    timestamps = [start_ts + timedelta(seconds=int(i * sampling_interval_sec))
                  for i in range(n_steps)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "well_id": well_id,
        "motor_current_A": motor_current,
        "motor_temperature_C": motor_temp,
        "intake_pressure_psi": intake_pressure,
        "discharge_pressure_psi": discharge_pressure,
        "vibration_x_g": vib_x,
        "vibration_y_g": vib_y,
        "flow_rate_bpd": flow_rate,
        "motor_speed_rpm": motor_speed,
        "winding_resistance_ohm": winding_resistance,
        "fluid_temperature_C": fluid_temp,
        "machine_status": machine_status,
        "failure_mode": failure_mode,
        "rul": rul,
    })
    return df


# ──────────────────────────────────────────────────────────────────
# Failure mode injectors
# ──────────────────────────────────────────────────────────────────

def _inject_gas_locking(current, temp, intake_p, flow, start, n, rng):
    """
    Gas locking: gas slug enters pump suction.
    Signature: sudden current drop + spike, temperature surge,
               oscillating intake pressure, near-zero flow.
    """
    if start >= n:
        return current, temp, intake_p, flow

    # Pre-failure: gradual intake pressure oscillation (gas accumulation)
    pre_start = max(0, start - 200)
    for i in range(pre_start, start):
        amplitude = (i - pre_start) / (start - pre_start + 1) * 50
        intake_p[i] += amplitude * np.sin(2 * np.pi * i / 20)

    # At failure: severe current drop then spike, flow collapse
    for i in range(start, n):
        phase = i - start
        current[i] = current[i] * 0.3 + 15 * np.sin(2 * np.pi * phase / 10) + rng.normal(0, 5)
        temp[i] = temp[i] + min(phase * 0.3, 40) + rng.normal(0, 2)
        flow[i] = max(0, flow[i] * np.exp(-0.05 * phase) + rng.normal(0, 20))
        intake_p[i] += 30 * np.sin(2 * np.pi * phase / 8)

    return current, temp, intake_p, flow


def _inject_abrasive_wear(vib_x, vib_y, intake_p, start, n, rng):
    """
    Abrasive wear: sand/solids eroding impellers and bearings.
    Signature: gradual vibration increase (both axes), drift in intake pressure.
    """
    if start >= n:
        return vib_x, vib_y, intake_p

    for i in range(start, n):
        wear = (i - start) / (n - start)
        vib_x[i] += wear * 0.8 + rng.normal(0, 0.05)
        vib_y[i] += wear * 0.6 + rng.normal(0, 0.04)
        intake_p[i] -= wear * 150

    return vib_x, vib_y, intake_p


def _inject_motor_overheating(current, temp, resistance, start, n, rng):
    """
    Motor overheating: winding insulation degradation.
    Signature: increasing winding resistance, current stays up, temperature climbs.
    """
    if start >= n:
        return current, temp, resistance

    for i in range(start, n):
        heat_factor = (i - start) / (n - start)
        resistance[i] += heat_factor * 1.5 + rng.normal(0, 0.02)
        temp[i] += heat_factor * 60 + rng.normal(0, 2)
        current[i] += heat_factor * 10 + rng.normal(0, 1)

    return current, temp, resistance


def _inject_scale_buildup(discharge_p, flow, start, n, rng):
    """
    Scale buildup (carbonate/sulfate): progressive internal restriction.
    Signature: steadily rising discharge pressure, declining flow rate.
    """
    if start >= n:
        return discharge_p, flow

    for i in range(start, n):
        scale = (i - start) / (n - start)
        discharge_p[i] += scale * 800 + rng.normal(0, 20)
        flow[i] -= scale * 800 + rng.normal(0, 15)
        flow[i] = max(0, flow[i])

    return discharge_p, flow


# ──────────────────────────────────────────────────────────────────
# Convenience: get sensor column names for this synthetic dataset
# ──────────────────────────────────────────────────────────────────

SYNTHETIC_SENSOR_COLS = [
    "motor_current_A", "motor_temperature_C", "intake_pressure_psi",
    "discharge_pressure_psi", "vibration_x_g", "vibration_y_g",
    "flow_rate_bpd", "motor_speed_rpm", "winding_resistance_ohm",
    "fluid_temperature_C",
]


if __name__ == "__main__":
    print("Generating synthetic ESP dataset...")
    df = generate_esp_dataset(n_wells=20, timesteps_per_well=3000, random_seed=42)
    df.to_csv("data/raw/synthetic_esp.csv", index=False)
    print(f"Saved {len(df):,} rows. Failure distribution:")
    print(df.groupby(["failure_mode", "machine_status"]).size())
