from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from q4dvar.models.toy import LinearModel, propagate
from q4dvar.problem import AssimilationProblem


Array = NDArray[np.float64]


def cost(problem: AssimilationProblem, initial_state: Array) -> float:
    """Evaluate the classic strong-constraint 4D-Var objective J(x0)."""

    background_delta = initial_state - problem.background
    background_precision = np.linalg.inv(problem.background_cov)
    observation_precision = np.linalg.inv(problem.observation_cov)

    value = 0.5 * background_delta.T @ background_precision @ background_delta
    states = propagate(problem.model, initial_state, len(problem.observations))
    for state, observed in zip(states, problem.observations):
        innovation = problem.observation @ state - observed
        value += 0.5 * innovation.T @ observation_precision @ innovation
    return float(value)


def quadratic_form(problem: AssimilationProblem) -> tuple[Array, Array, float]:
    """Return A, b, c such that J(x) = 0.5 * x.T A x - b.T x + c."""

    background_precision = np.linalg.inv(problem.background_cov)
    observation_precision = np.linalg.inv(problem.observation_cov)
    a_matrix = background_precision.copy()
    b_vector = background_precision @ problem.background
    c_value = 0.5 * problem.background.T @ background_precision @ problem.background

    if not isinstance(problem.model, LinearModel):
        raise TypeError("quadratic_form is only available for LinearModel problems.")

    model_power = np.eye(problem.model.transition.shape[0], dtype=np.float64)
    for observed in problem.observations:
        linear_obs = problem.observation @ model_power
        a_matrix += linear_obs.T @ observation_precision @ linear_obs
        b_vector += linear_obs.T @ observation_precision @ observed
        c_value += 0.5 * observed.T @ observation_precision @ observed
        model_power = problem.model.transition @ model_power

    return a_matrix, b_vector, float(c_value)


def solve_classical(problem: AssimilationProblem) -> Array:
    """Solve the 4D-Var problem with an exact or numerical classical baseline."""

    if not isinstance(problem.model, LinearModel):
        result = minimize(
            lambda state: cost(problem, np.asarray(state, dtype=np.float64)),
            problem.background,
            method="Powell",
            options={"maxiter": 800, "xtol": 1e-7, "ftol": 1e-7},
        )
        if not result.success:
            # The current state is still useful for a demo, but make failure visible.
            print(f"Warning: classical optimizer did not fully converge: {result.message}")
        return np.asarray(result.x, dtype=np.float64)

    a_matrix, b_vector, _ = quadratic_form(problem)
    return np.linalg.solve(a_matrix, b_vector)


def rmse(estimate: Array, truth: Array) -> float:
    """Root mean squared error for reporting."""

    return float(np.sqrt(np.mean((estimate - truth) ** 2)))
