"""
src/models/survival_model.py
==============================
Survival analysis for ESP failure prediction.

Models implemented:
  1. Cox Proportional Hazards (CoxPH)   — semiparametric, no distributional assumption
  2. Weibull Accelerated Failure Time (AFT) — parametric, interpretable scale/shape
  3. DeepHit wrapper                    — optional deep survival (requires pycox)

Key outputs:
  - Hazard function h(t|x): instantaneous failure rate at time t given covariates x
  - Survival function S(t|x) = P(T > t | x): probability of surviving beyond time t
  - Median survival time: t such that S(t|x) = 0.5
  - Expected failure time: ∫₀^∞ S(t|x) dt

Covariates (features) can be:
  - Engineered statistical features (mean, std, trend) over a window
  - Latent representations from the LSTM/Transformer autoencoder (z-scores)
  - Domain-specific aggregates (efficiency, differential pressure trend)

References:
  - Cox (1972) "Regression models and life-tables"
  - Kalbfleisch & Prentice (2002) "The Statistical Analysis of Failure Time Data"
  - Lee et al. (2018) "DeepHit: A Deep Learning Approach to Survival Analysis
    with Competing Risks"
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
import warnings
import pickle
import os

# lifelines is the main survival analysis library
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    warnings.warn("lifelines not installed. Run: pip install lifelines")


# ──────────────────────────────────────────────────────────────────
# Feature preparation for survival models
# ──────────────────────────────────────────────────────────────────

def prepare_survival_dataframe(
    df: pd.DataFrame,
    sensor_cols: List[str],
    time_col: str = "rul",
    event_col: str = "failure",
    groupby_col: Optional[str] = None,
    agg_window: int = 100,
) -> pd.DataFrame:
    """
    Prepare a per-event-unit survival DataFrame for lifelines.

    In survival analysis we need one row per "subject" (per well / per run):
      - T: time-to-event (RUL at start of monitoring)
      - E: event observed (1 = failure occurred, 0 = censored)
      - Covariates: summary statistics of the sensor signals

    Args:
        df: Raw sensor dataframe (long format, one row per timestep).
        sensor_cols: Sensor feature columns.
        time_col: Column containing remaining useful life.
        event_col: Binary column (1 = failure event, 0 = censored).
        groupby_col: Column identifying each unit (well_id, unit, etc.).
        agg_window: Use first `agg_window` timesteps for covariate estimation
                    (simulates early-life monitoring).

    Returns:
        Survival DataFrame with columns: [duration, event, covariate1, ...]
    """
    if groupby_col is None:
        # Treat entire df as one unit
        groups = [("all", df)]
    else:
        groups = df.groupby(groupby_col)

    records = []
    for unit_id, group in groups:
        group = group.sort_index()
        early = group.iloc[:agg_window]  # early-life window for covariates

        row = {
            "unit_id": unit_id,
            # Duration: max RUL at start (before any degradation)
            "duration": float(group[time_col].max()) if time_col in group.columns
                        else float(len(group)),
            # Event: did failure occur during observation?
            "event": int(group[event_col].max()) if event_col in group.columns else 0,
        }
        # Aggregate covariates from early-life window
        for col in sensor_cols:
            if col not in early.columns:
                continue
            vals = early[col].dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std())
            row[f"{col}_trend"] = float(np.polyfit(range(len(vals)), vals, 1)[0])
            row[f"{col}_max"] = float(vals.max())

        records.append(row)

    surv_df = pd.DataFrame(records).set_index("unit_id")
    surv_df = surv_df.replace([np.inf, -np.inf], np.nan).dropna()
    return surv_df


# ──────────────────────────────────────────────────────────────────
# Cox Proportional Hazards
# ──────────────────────────────────────────────────────────────────

class CoxPHModel:
    """
    Wrapper around lifelines CoxPHFitter for ESP failure prediction.

    The Cox model estimates:
      h(t|x) = h₀(t) × exp(β₁x₁ + β₂x₂ + ... + βₚxₚ)

    where h₀(t) is the baseline hazard (non-parametric).
    β coefficients are interpretable: exp(β) = hazard ratio.
    """

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0):
        """
        Args:
            penalizer: L1/L2 regularisation strength.
            l1_ratio:  0 = L2 (Ridge), 1 = L1 (Lasso), 0.5 = ElasticNet.
        """
        assert LIFELINES_AVAILABLE, "pip install lifelines"
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.feature_cols: Optional[List[str]] = None

    def fit(
        self,
        surv_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "CoxPHModel":
        """
        Fit the Cox model.

        Args:
            surv_df: Survival DataFrame (from prepare_survival_dataframe).
            duration_col: Time-to-event column.
            event_col: Binary event indicator column.

        Returns:
            self (for chaining)
        """
        self.feature_cols = [c for c in surv_df.columns
                              if c not in [duration_col, event_col]]
        self.model.fit(
            surv_df,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )
        return self

    def predict_survival_function(
        self,
        covariates: pd.DataFrame,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict S(t|x) for new observations.

        Returns:
            DataFrame where index = time, columns = sample units.
        """
        return self.model.predict_survival_function(covariates, times=times)

    def predict_median_survival(self, covariates: pd.DataFrame) -> np.ndarray:
        """Predicted median survival time (50th percentile of failure time)."""
        return self.model.predict_median(covariates).values

    def predict_failure_probability(
        self,
        covariates: pd.DataFrame,
        horizon: float,
    ) -> np.ndarray:
        """
        P(failure by time `horizon`) = 1 - S(horizon|x).
        Useful for: "probability of failure within next 30 days".
        """
        sf = self.predict_survival_function(covariates, times=[horizon])
        return 1.0 - sf.values.flatten()

    def print_summary(self):
        """Print model coefficients and statistical significance."""
        self.model.print_summary()

    def get_hazard_ratios(self) -> pd.DataFrame:
        """
        Return hazard ratios (exp(β)) with 95% CI.
        HR > 1: covariate increases failure risk.
        HR < 1: covariate reduces failure risk.
        """
        summary = self.model.summary
        return summary[["coef", "exp(coef)", "p", "exp(coef) lower 95%",
                         "exp(coef) upper 95%"]].round(4)

    def concordance_index(
        self,
        surv_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> float:
        """C-index: discrimination ability (0.5 = random, 1.0 = perfect)."""
        predicted_risk = self.model.predict_partial_hazard(surv_df)
        return concordance_index(
            surv_df[duration_col],
            -predicted_risk,
            surv_df[event_col],
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "CoxPHModel":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ──────────────────────────────────────────────────────────────────
# Weibull AFT Model
# ──────────────────────────────────────────────────────────────────

class WeibullAFTModel:
    """
    Weibull Accelerated Failure Time model.

    Assumes: log(T) = β₀ + β₁x₁ + ... + βₚxₚ + σε
    where ε ~ Extreme Value distribution → T ~ Weibull.

    More interpretable than Cox when distributional assumption holds.
    Provides direct estimates of expected lifetime and scale parameter.
    """

    def __init__(self):
        assert LIFELINES_AVAILABLE, "pip install lifelines"
        self.model = WeibullAFTFitter()

    def fit(
        self,
        surv_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "WeibullAFTModel":
        self.model.fit(
            surv_df, duration_col=duration_col, event_col=event_col,
            show_progress=False,
        )
        return self

    def predict_median(self, covariates: pd.DataFrame) -> np.ndarray:
        return self.model.predict_median(covariates).values

    def predict_expectation(self, covariates: pd.DataFrame) -> np.ndarray:
        """Predicted mean failure time E[T|x]."""
        return self.model.predict_expectation(covariates).values

    def predict_survival_function(self, covariates: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_survival_function(covariates)

    def print_summary(self):
        self.model.print_summary()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "WeibullAFTModel":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ──────────────────────────────────────────────────────────────────
# Utility: Plot survival curves
# ──────────────────────────────────────────────────────────────────

def plot_survival_curves(
    cox_model: CoxPHModel,
    sample_covariates: pd.DataFrame,
    labels: Optional[List[str]] = None,
    title: str = "Predicted Survival Functions",
    ax=None,
):
    """
    Plot S(t|x) for multiple covariate profiles.

    Args:
        cox_model: Fitted CoxPHModel.
        sample_covariates: DataFrame, one row per profile to plot.
        labels: Legend labels (defaults to row index).
        title: Plot title.
        ax: Matplotlib axis.

    Returns:
        ax: Matplotlib axis with the plot.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    sf = cox_model.predict_survival_function(sample_covariates)
    labels = labels or [str(i) for i in sample_covariates.index]

    for col, label in zip(sf.columns, labels):
        ax.plot(sf.index, sf[col], label=label, linewidth=2)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="50% survival")
    ax.set_xlabel("Time (timesteps / hours)", fontsize=12)
    ax.set_ylabel("Survival Probability S(t|x)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    return ax
