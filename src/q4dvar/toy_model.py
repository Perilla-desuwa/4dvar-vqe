from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class AssimilationProblem:
    """4D-Var problem over a short observation window."""

    model: "ForecastModel"
    observation: Array
    background: Array
    background_cov: Array
    observation_cov: Array
    observations: Array
    truth_initial: Array
    name: str = "toy"


class ForecastModel(Protocol):
    """Minimal forecast-model interface used by the assimilation solvers."""

    state_dim: int
    name: str

    def forecast(self, initial_state: Array, window: int) -> Array:
        """Return model states at the observation times."""


@dataclass(frozen=True)
class LinearModel:
    """Linear model x_{t+1} = M x_t."""

    transition: Array
    name: str = "linear-rotation"

    @property
    def state_dim(self) -> int:
        return int(self.transition.shape[0])

    def forecast(self, initial_state: Array, window: int) -> Array:
        states = []
        state = initial_state.astype(np.float64)
        for _ in range(window):
            states.append(state)
            state = self.transition @ state
        return np.asarray(states, dtype=np.float64)


@dataclass(frozen=True)
class Lorenz63Model:
    """Lorenz-63 model integrated with a fixed-step RK4 scheme."""

    dt: float = 0.01
    steps_per_obs: int = 10
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    name: str = "lorenz63"

    @property
    def state_dim(self) -> int:
        return 3

    def tendency(self, state: Array) -> Array:
        x_value, y_value, z_value = state
        return np.asarray(
            [
                self.sigma * (y_value - x_value),
                x_value * (self.rho - z_value) - y_value,
                x_value * y_value - self.beta * z_value,
            ],
            dtype=np.float64,
        )

    def step(self, state: Array) -> Array:
        k1 = self.tendency(state)
        k2 = self.tendency(state + 0.5 * self.dt * k1)
        k3 = self.tendency(state + 0.5 * self.dt * k2)
        k4 = self.tendency(state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forecast(self, initial_state: Array, window: int) -> Array:
        states = []
        state = initial_state.astype(np.float64)
        for _ in range(window):
            states.append(state)
            for _ in range(self.steps_per_obs):
                state = self.step(state)
        return np.asarray(states, dtype=np.float64)


def rotation_dynamics(angle_degrees: float = 18.0, damping: float = 0.97) -> Array:
    """Return a stable 2D linear dynamics matrix for the toy forecast model."""

    angle = np.deg2rad(angle_degrees)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return damping * rotation


def generate_problem(seed: int = 7, window: int = 6) -> AssimilationProblem:
    """Generate synthetic observations from a known initial state."""

    rng = np.random.default_rng(seed)
    model = LinearModel(rotation_dynamics())
    observation = np.array([[1.0, 0.35]], dtype=np.float64)
    truth_initial = np.array([1.2, -0.7], dtype=np.float64)
    background = truth_initial + np.array([0.55, -0.35], dtype=np.float64)
    background_cov = np.diag([0.45**2, 0.45**2]).astype(np.float64)
    observation_cov = np.array([[0.08**2]], dtype=np.float64)

    states = model.forecast(truth_initial, window)
    noise = rng.normal(0.0, np.sqrt(observation_cov[0, 0]), size=(window, 1))
    observations = states @ observation.T + noise

    return AssimilationProblem(
        model=model,
        observation=observation,
        background=background,
        background_cov=background_cov,
        observation_cov=observation_cov,
        observations=observations.astype(np.float64),
        truth_initial=truth_initial,
        name="linear-rotation",
    )


def generate_lorenz63_problem(seed: int = 7, window: int = 25) -> AssimilationProblem:
    """Generate a Lorenz-63 4D-Var problem with partial noisy observations."""

    rng = np.random.default_rng(seed)
    model = Lorenz63Model()
    observation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    truth_initial = np.array([-8.0, 7.0, 27.0], dtype=np.float64)
    background = truth_initial + np.array([2.2, -1.7, 1.4], dtype=np.float64)
    background_cov = np.diag([2.0**2, 2.0**2, 2.0**2]).astype(np.float64)
    observation_cov = np.diag([0.8**2, 0.8**2]).astype(np.float64)

    states = model.forecast(truth_initial, window)
    obs_std = np.sqrt(np.diag(observation_cov))
    noise = rng.normal(0.0, obs_std, size=(window, observation.shape[0]))
    observations = states @ observation.T + noise

    return AssimilationProblem(
        model=model,
        observation=observation,
        background=background,
        background_cov=background_cov,
        observation_cov=observation_cov,
        observations=observations.astype(np.float64),
        truth_initial=truth_initial,
        name="lorenz63",
    )


def propagate(model: ForecastModel, initial_state: Array, window: int) -> Array:
    """Propagate an initial state with any supported model."""

    return model.forecast(initial_state, window)
