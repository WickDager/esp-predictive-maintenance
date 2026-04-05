"""
tests/test_synthetic.py
========================
Unit tests for the synthetic ESP data generator.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import (
    generate_esp_dataset,
    SYNTHETIC_SENSOR_COLS,
    _inject_gas_locking,
    _inject_abrasive_wear,
    _inject_motor_overheating,
    _inject_scale_buildup,
)


class TestGenerateESPDataset:
    """Test the main synthetic data generator."""

    def test_basic_generation(self):
        """Test that dataset generation produces expected shape."""
        df = generate_esp_dataset(n_wells=3, timesteps_per_well=500, random_seed=42)
        assert len(df) == 3 * 500
        assert "timestamp" in df.columns
        assert "well_id" in df.columns
        assert "machine_status" in df.columns
        assert "failure_mode" in df.columns
        assert "rul" in df.columns

    def test_sensor_columns(self):
        """All expected sensor columns should be present."""
        df = generate_esp_dataset(n_wells=1, timesteps_per_well=100, random_seed=42)
        for col in SYNTHETIC_SENSOR_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nan_values(self):
        """Synthetic data should have no NaN values."""
        df = generate_esp_dataset(n_wells=5, timesteps_per_well=200, random_seed=42)
        for col in SYNTHETIC_SENSOR_COLS:
            assert df[col].isna().sum() == 0, f"NaN values in {col}"

    def test_failure_modes(self):
        """All failure modes should be representable."""
        modes = set()
        for _ in range(100):
            df = generate_esp_dataset(
                n_wells=10, timesteps_per_well=100,
                failure_prob=1.0,
                random_seed=np.random.randint(0, 10000),
            )
            modes.update(df["failure_mode"].unique())
        # Should see all 4 failure modes across many runs
        expected = {"gas_locking", "abrasive_wear", "motor_overheating", "scale_buildup"}
        assert expected.issubset(modes)

    def test_normal_wells(self):
        """Wells with failure_prob=0 should all be NORMAL."""
        df = generate_esp_dataset(
            n_wells=5, timesteps_per_well=100,
            failure_prob=0.0, random_seed=42,
        )
        assert (df["machine_status"] == "NORMAL").all()
        assert (df["failure_mode"] == "normal").all()

    def test_rul_values(self):
        """RUL should decrease toward failure and be 0 at failure."""
        df = generate_esp_dataset(n_wells=1, timesteps_per_well=1000,
                                   failure_prob=1.0, random_seed=42)
        failure_start = (df["machine_status"] == "BROKEN").idxmax()
        # RUL should be 0 at and after failure start
        rul_at_failure = df.loc[failure_start, "rul"]
        assert rul_at_failure == 0.0

    def test_reproducibility(self):
        """Same seed should produce same data."""
        df1 = generate_esp_dataset(
            n_wells=3, timesteps_per_well=200, random_seed=123,
        )
        df2 = generate_esp_dataset(
            n_wells=3, timesteps_per_well=200, random_seed=123,
        )
        for col in SYNTHETIC_SENSOR_COLS:
            np.testing.assert_array_equal(df1[col].values, df2[col].values)

    def test_well_ids_are_unique(self):
        """Each well should have a unique well_id."""
        df = generate_esp_dataset(n_wells=10, timesteps_per_well=100, random_seed=42)
        assert df["well_id"].nunique() == 10


class TestFailureModeInjectors:
    """Test individual failure mode injection functions."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.n = 500
        self.start = 300

        # Baseline signals
        self.current = np.ones(self.n) * 80 + self.rng.normal(0, 0.5, self.n)
        self.temp = np.ones(self.n) * 85 + self.rng.normal(0, 0.5, self.n)
        self.intake_p = np.ones(self.n) * 1200 + self.rng.normal(0, 5, self.n)
        self.flow = np.ones(self.n) * 2000 + self.rng.normal(0, 10, self.n)
        self.vib_x = np.ones(self.n) * 0.05 + self.rng.normal(0, 0.01, self.n)
        self.vib_y = np.ones(self.n) * 0.04 + self.rng.normal(0, 0.01, self.n)
        self.discharge_p = np.ones(self.n) * 4500 + self.rng.normal(0, 10, self.n)
        self.resistance = np.ones(self.n) * 2.5 + self.rng.normal(0, 0.01, self.n)

    def test_gas_locking_reduces_flow(self):
        """Gas locking should cause flow to decrease after failure."""
        curr, temp, ip, flow = _inject_gas_locking(
            self.current.copy(), self.temp.copy(),
            self.intake_p.copy(), self.flow.copy(),
            self.start, self.n, self.rng
        )
        pre_flow = flow[:self.start].mean()
        post_flow = flow[self.start:].mean()
        assert post_flow < pre_flow, "Flow should decrease during gas locking"

    def test_gas_locking_current_oscillation(self):
        """Gas locking should increase current variance."""
        curr, _, _, _ = _inject_gas_locking(
            self.current.copy(), self.temp.copy(),
            self.intake_p.copy(), self.flow.copy(),
            self.start, self.n, self.rng
        )
        pre_std = curr[:self.start].std()
        post_std = curr[self.start:].std()
        assert post_std > pre_std, "Current variance should increase during gas locking"

    def test_abrasive_wear_increases_vibration(self):
        """Abrasive wear should increase vibration."""
        vx, vy, _ = _inject_abrasive_wear(
            self.vib_x.copy(), self.vib_y.copy(),
            self.intake_p.copy(), self.start, self.n, self.rng
        )
        pre_vib = self.vib_x[:self.start].std()
        post_vib = vx[self.start:].std()
        assert post_vib > pre_vib, "Vibration should increase during abrasive wear"

    def test_motor_overheating_increases_temp(self):
        """Motor overheating should increase temperature."""
        _, temp, _ = _inject_motor_overheating(
            self.current.copy(), self.temp.copy(),
            self.resistance.copy(), self.start, self.n, self.rng
        )
        pre_temp = self.temp[:self.start].mean()
        post_temp = temp[self.start:].mean()
        assert post_temp > pre_temp, "Temperature should increase during overheating"

    def test_motor_overheating_increases_resistance(self):
        """Motor overheating should increase winding resistance."""
        _, _, res = _inject_motor_overheating(
            self.current.copy(), self.temp.copy(),
            self.resistance.copy(), self.start, self.n, self.rng
        )
        pre_res = self.resistance[:self.start].mean()
        post_res = res[self.start:].mean()
        assert post_res > pre_res, "Resistance should increase during overheating"

    def test_scale_buildup_increases_pressure(self):
        """Scale buildup should increase discharge pressure."""
        dp, _ = _inject_scale_buildup(
            self.discharge_p.copy(), self.flow.copy(),
            self.start, self.n, self.rng
        )
        pre_dp = self.discharge_p[:self.start].mean()
        post_dp = dp[self.start:].mean()
        assert post_dp > pre_dp, "Discharge pressure should increase during scale buildup"

    def test_scale_buildup_reduces_flow(self):
        """Scale buildup should reduce flow rate."""
        _, fl = _inject_scale_buildup(
            self.discharge_p.copy(), self.flow.copy(),
            self.start, self.n, self.rng
        )
        pre_flow = self.flow[:self.start].mean()
        post_flow = fl[self.start:].mean()
        assert post_flow < pre_flow, "Flow should decrease during scale buildup"

    def test_no_injection_when_start_at_end(self):
        """Injectors should return unchanged when start >= n."""
        curr, temp, ip, flow = _inject_gas_locking(
            self.current.copy(), self.temp.copy(),
            self.intake_p.copy(), self.flow.copy(),
            self.n + 100, self.n, self.rng
        )
        np.testing.assert_array_equal(curr, self.current)
