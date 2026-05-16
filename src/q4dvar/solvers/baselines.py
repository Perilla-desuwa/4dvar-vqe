from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q4dvar.data_loader import Lorenz96Dataset
from q4dvar.models.lorenz96 import Lorenz96Model
from q4dvar.problem import Array
from q4dvar.solvers.classical import rmse


@dataclass(frozen=True)
class BaselineResult:
    name: str
    analysis: Array
    rmse_vs_truth: float | None


def observed_baseline(dataset: Lorenz96Dataset) -> BaselineResult:
    """Use noisy observations directly as the analysis."""

    analysis = dataset.observed.copy()
    return BaselineResult(
        name="observed",
        analysis=analysis,
        rmse_vs_truth=_score_if_truth_available(dataset, analysis),
    )


def free_run_baseline(dataset: Lorenz96Dataset, model: Lorenz96Model | None = None) -> BaselineResult:
    """Forecast freely from the first observation without further updates."""

    model = model or Lorenz96Model(state_dim=dataset.state_dim)
    analysis = model.forecast(dataset.observed[0], dataset.n_times)
    return BaselineResult(
        name="free_run",
        analysis=analysis,
        rmse_vs_truth=_score_if_truth_available(dataset, analysis),
    )


def optimal_interpolation_baseline(
    dataset: Lorenz96Dataset,
    background_std: float = 1.0,
    observation_std: float = 0.5,
    model: Lorenz96Model | None = None,
) -> BaselineResult:
    """Run a scalar-gain 3D-Var/OI update at each observation time."""

    if background_std <= 0.0:
        raise ValueError("background_std must be positive.")
    if observation_std <= 0.0:
        raise ValueError("observation_std must be positive.")

    model = model or Lorenz96Model(state_dim=dataset.state_dim)
    gain = background_std**2 / (background_std**2 + observation_std**2)
    analysis = np.zeros_like(dataset.observed, dtype=np.float64)
    forecast = dataset.observed[0].copy()

    for time_index, observed in enumerate(dataset.observed):
        if time_index > 0:
            forecast = model.advance(analysis[time_index - 1], model.steps_per_obs)
        analysis[time_index] = forecast + gain * (observed - forecast)

    return BaselineResult(
        name="optimal_interpolation",
        analysis=analysis,
        rmse_vs_truth=_score_if_truth_available(dataset, analysis),
    )


def stochastic_enkf_baseline(
    dataset: Lorenz96Dataset,
    ensemble_size: int = 80,
    observation_std: float = 0.5,
    initial_spread: float = 2.0,
    inflation: float = 1.08,
    seed: int = 0,
    model: Lorenz96Model | None = None,
) -> BaselineResult:
    """Run a simple stochastic EnKF with full-state Lorenz96 observations."""

    if ensemble_size < 2:
        raise ValueError("ensemble_size must be at least 2.")
    if observation_std <= 0.0:
        raise ValueError("observation_std must be positive.")
    if initial_spread <= 0.0:
        raise ValueError("initial_spread must be positive.")

    model = model or Lorenz96Model(state_dim=dataset.state_dim)
    rng = np.random.default_rng(seed)
    ensemble = dataset.observed[0] + rng.normal(
        0.0,
        initial_spread,
        size=(ensemble_size, dataset.state_dim),
    )
    analysis = np.zeros_like(dataset.observed, dtype=np.float64)

    for time_index, observed in enumerate(dataset.observed):
        if time_index > 0:
            ensemble = np.asarray([model.advance(member, model.steps_per_obs) for member in ensemble], dtype=np.float64)

        mean = np.mean(ensemble, axis=0)
        anomalies = (ensemble - mean) * inflation
        ensemble = mean + anomalies
        ensemble = _stochastic_enkf_update(ensemble, observed, observation_std, rng)
        analysis[time_index] = np.mean(ensemble, axis=0)

    return BaselineResult(
        name="stochastic_enkf",
        analysis=analysis,
        rmse_vs_truth=_score_if_truth_available(dataset, analysis),
    )


def _stochastic_enkf_update(
    forecast_ensemble: Array,
    observed: Array,
    observation_std: float,
    rng: np.random.Generator,
) -> Array:
    ensemble_size, state_dim = forecast_ensemble.shape
    mean = np.mean(forecast_ensemble, axis=0)
    anomalies = forecast_ensemble - mean
    forecast_cov = (anomalies.T @ anomalies) / float(ensemble_size - 1)
    innovation_cov = forecast_cov + np.eye(state_dim, dtype=np.float64) * observation_std**2
    kalman_gain = np.linalg.solve(innovation_cov, forecast_cov).T

    perturbed_observations = observed + rng.normal(0.0, observation_std, size=forecast_ensemble.shape)
    innovations = perturbed_observations - forecast_ensemble
    return forecast_ensemble + innovations @ kalman_gain.T


def _score_if_truth_available(dataset: Lorenz96Dataset, analysis: Array) -> float | None:
    if not dataset.has_truth:
        return None
    mask = np.isfinite(dataset.truth)
    return rmse(analysis[mask], dataset.truth[mask])
