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

