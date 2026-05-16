from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def plot_state_trajectories(
    truth: Array,
    background: Array,
    analysis: Array,
    output_path: str | Path,
    title: str,
    component_names: tuple[str, ...] | None = None,
) -> Path:
    """Plot truth, background forecast, and assimilated forecast trajectories."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_components = truth.shape[1]
    names = component_names or tuple(f"x{i}" for i in range(n_components))
    times = np.arange(truth.shape[0])

    figure, axes = plt.subplots(n_components, 1, figsize=(10, 2.5 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]

    for index, axis in enumerate(axes):
        axis.plot(times, truth[:, index], label="truth", color="black", linewidth=2.0)
        axis.plot(times, background[:, index], label="background", linestyle="--", color="tab:red")
        axis.plot(times, analysis[:, index], label="analysis", color="tab:blue")
        axis.set_ylabel(names[index])
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("observation time index")
    axes[0].legend(loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_phase_trajectory_comparison(
    truth: Array,
    observed: Array,
    background: Array,
    analysis: Array,
    output_path: str | Path,
    dims: tuple[int, int] = (0, 1),
    title: str = "Lorenz96 phase trajectory comparison",
) -> Path:
    """Plot 2D phase trajectories for truth, observations, background, and analysis."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x_dim, y_dim = dims

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(
        truth[:, x_dim],
        truth[:, y_dim],
        label="truth",
        color="black",
        linewidth=2.0,
    )
    axis.scatter(
        observed[:, x_dim],
        observed[:, y_dim],
        label="observed",
        color="tab:orange",
        s=14,
        alpha=0.6,
    )
    axis.plot(
        background[:, x_dim],
        background[:, y_dim],
        label="unassimilated",
        color="tab:red",
        linestyle="--",
        linewidth=1.7,
    )
    axis.plot(
        analysis[:, x_dim],
        analysis[:, y_dim],
        label="assimilated",
        color="tab:blue",
        linewidth=1.8,
    )

    axis.scatter(truth[0, x_dim], truth[0, y_dim], color="black", s=45, marker="o", label="start")
    axis.set_xlabel(f"x{int(x_dim)}")
    axis.set_ylabel(f"x{int(y_dim)}")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
