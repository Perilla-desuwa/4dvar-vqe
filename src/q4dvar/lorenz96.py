from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q4dvar.data_loader import Lorenz96Dataset
from q4dvar.toy_model import Array, AssimilationProblem


@dataclass(frozen=True)
class Lorenz96Model:
    """40-dimensional Lorenz96 model integrated with fixed-step RK4."""

    state_dim: int = 40
    forcing: float = 8.0
    dt: float = 0.05
    steps_per_obs: int = 2
    name: str = "lorenz96"

    def tendency(self, state: Array) -> Array:
        """Evaluate dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F."""

        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape ({self.state_dim},), got {state.shape}.")

        return (
            (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1)
            - state
            + self.forcing
        ).astype(np.float64)

    def step(self, state: Array) -> Array:
        """Advance one model step with fourth-order Runge-Kutta."""

        state = np.asarray(state, dtype=np.float64)
        k1 = self.tendency(state)
        k2 = self.tendency(state + 0.5 * self.dt * k1)
        k3 = self.tendency(state + 0.5 * self.dt * k2)
        k4 = self.tendency(state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def advance(self, initial_state: Array, steps: int) -> Array:
        """Advance a state by a given number of RK4 model steps."""

        state = np.asarray(initial_state, dtype=np.float64)
        for _ in range(steps):
            state = self.step(state)
        return state

    def forecast(self, initial_state: Array, window: int) -> Array:
        """Return model states at observation times over a 4D-Var window."""

        states = []
        state = np.asarray(initial_state, dtype=np.float64)
        for _ in range(window):
            states.append(state)
            state = self.advance(state, self.steps_per_obs)
        return np.asarray(states, dtype=np.float64)


def make_lorenz96_problem(
    dataset: Lorenz96Dataset,
    start_index: int = 0,
    window: int = 12,
    background: Array | None = None,
    background_std: float = 1.0,
    observation_std: float = 0.5,
    model: Lorenz96Model | None = None,
) -> AssimilationProblem:
    """Build a classic 4D-Var problem from a contiguous Lorenz96 data window."""

    model = model or Lorenz96Model(state_dim=dataset.state_dim)
    _validate_window(dataset, start_index, window)
    if model.state_dim != dataset.state_dim:
        raise ValueError(f"Model dimension {model.state_dim} does not match dataset dimension {dataset.state_dim}.")

    observations = dataset.observed[start_index : start_index + window]
    if np.isnan(observations).any():
        raise ValueError("Lorenz96 4D-Var problem requires complete observations in the selected window.")

    if background is None:
        background = observations[0].copy()
    background = np.asarray(background, dtype=np.float64)
    if background.shape != (dataset.state_dim,):
        raise ValueError(f"Expected background shape ({dataset.state_dim},), got {background.shape}.")

    truth_initial = dataset.truth[start_index]
    observation = np.eye(dataset.state_dim, dtype=np.float64)
    background_cov = np.eye(dataset.state_dim, dtype=np.float64) * background_std**2
    observation_cov = np.eye(dataset.state_dim, dtype=np.float64) * observation_std**2

    return AssimilationProblem(
        model=model,
        observation=observation,
        background=background,
        background_cov=background_cov,
        observation_cov=observation_cov,
        observations=observations.astype(np.float64),
        truth_initial=truth_initial.astype(np.float64),
        name="lorenz96",
    )


def _validate_window(dataset: Lorenz96Dataset, start_index: int, window: int) -> None:
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    if window <= 0:
        raise ValueError("window must be positive.")
    if start_index + window > dataset.n_times:
        raise ValueError(
            f"Window [{start_index}, {start_index + window}) exceeds dataset length {dataset.n_times}."
        )
