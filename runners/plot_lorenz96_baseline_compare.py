from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from q4dvar.data_loader import Lorenz96Dataset, load_lorenz96_csv
from q4dvar.models.lorenz96 import Lorenz96Model
from q4dvar.plotting import plot_phase_trajectories
from q4dvar.solvers.baselines import (
    free_run_baseline,
    observed_baseline,
    optimal_interpolation_baseline,
    stochastic_enkf_baseline,
)
from q4dvar.solvers.classical import rmse
from q4dvar.solvers.qubo import run_sliding_window_qubo


DEFAULT_TRAIN_PATH = Path("气象海洋/气象海洋/小规模测试/lorenz96_train.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/lorenz96_baseline_solver_x0_x1.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot truth, initial, baseline, and solver-QUBO trajectories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_PATH, help="Lorenz96 CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG path.")
    parser.add_argument("--limit", type=int, default=120, help="Number of observation times to plot.")
    parser.add_argument("--dim-x", type=int, default=0, help="X-axis state dimension.")
    parser.add_argument("--dim-y", type=int, default=1, help="Y-axis state dimension.")
    parser.add_argument(
        "--baseline",
        choices=["observed", "free_run", "oi", "enkf"],
        default="observed",
        help="Classical baseline to compare against QUBO 4D-Var.",
    )
    parser.add_argument("--window", type=int, default=6, help="Number of observation times per 4D-Var window.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to window - 1.")
    parser.add_argument("--block-size", type=int, default=10, help="State dimensions per local QUBO block.")
    parser.add_argument("--block-stride", type=int, default=None, help="Dimension stride between QUBO blocks.")
    parser.add_argument(
        "--block-selection",
        choices=["cyclic", "gradient", "hessian"],
        default="cyclic",
        help="Policy for selecting QUBO state-dimension blocks.",
    )
    parser.add_argument("--bits-per-dim", type=int, default=3, help="Binary variables per optimized dimension.")
    parser.add_argument("--radius", type=float, default=0.4, help="Maximum absolute increment per dimension.")
    parser.add_argument("--block-passes", type=int, default=5, help="Number of QUBO block sweep passes per window.")
    parser.add_argument("--time-sweeps", type=int, default=3, help="Number of full passes over all time windows.")
    parser.add_argument("--ensemble-size", type=int, default=80, help="EnKF ensemble size if --baseline enkf.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-window progress logs.")
    parser.add_argument(
        "--solver",
        choices=["qaoa", "greedy"],
        default="greedy",
        help="Solver to use for 4D-Var.",
    )
    args = parser.parse_args()

    dataset = _slice_dataset(load_lorenz96_csv(args.input), args.limit)
    model = Lorenz96Model(state_dim=dataset.state_dim)
    initial = model.forecast(dataset.observed[0], dataset.n_times)
    baseline = _run_baseline(args.baseline, dataset, args.ensemble_size, args.seed)
    solver = run_sliding_window_qubo(
        dataset,
        window=args.window,
        stride=args.stride,
        block_size=args.block_size,
        block_stride=args.block_stride,
        block_selection=args.block_selection,
        bits_per_dim=args.bits_per_dim,
        radius=args.radius,
        outer_loops=args.block_passes,
        time_sweeps=args.time_sweeps,
        seed=args.seed,
        solver=args.solver,
        verbose=not args.quiet,
    )

    output = plot_phase_trajectories(
        {
            "truth": dataset.truth,
            "initial": initial,
            "baseline": baseline.analysis,
            "greedy": solver.analysis,
        },
        output_path=args.output,
        dims=(args.dim_x, args.dim_y),
        title=f"Lorenz96 x{args.dim_x}-x{args.dim_y}: truth, initial, baseline, solver",
    )

    print("Lorenz96 baseline vs solver plot")
    print("=" * 36)
    print(f"output:        {output}")
    print(f"points:        {dataset.n_times}")
    print(f"dims:          x{args.dim_x}, x{args.dim_y}")
    print(f"baseline:      {baseline.name}")
    print(f"initial RMSE:  {_score(dataset, initial):.6f}")
    if baseline.rmse_vs_truth is not None:
        print(f"baseline RMSE: {baseline.rmse_vs_truth:.6f}")
    if solver.rmse_vs_truth is not None:
        print(f"solver RMSE:   {solver.rmse_vs_truth:.6f}")


def _slice_dataset(dataset: Lorenz96Dataset, limit: int) -> Lorenz96Dataset:
    plot_limit = min(limit, dataset.n_times)
    if plot_limit == dataset.n_times:
        return dataset
    return Lorenz96Dataset(
        time_steps=dataset.time_steps[:plot_limit],
        truth=dataset.truth[:plot_limit],
        observed=dataset.observed[:plot_limit],
        observed_mask=dataset.observed_mask[:plot_limit],
        source_path=dataset.source_path,
    )


def _run_baseline(name: str, dataset: Lorenz96Dataset, ensemble_size: int, seed: int):
    if name == "observed":
        return observed_baseline(dataset)
    if name == "free_run":
        return free_run_baseline(dataset)
    if name == "enkf":
        return stochastic_enkf_baseline(dataset, ensemble_size=ensemble_size, seed=seed)
    return optimal_interpolation_baseline(dataset)


def _score(dataset: Lorenz96Dataset, analysis) -> float:
    mask = np.isfinite(dataset.truth)
    return rmse(analysis[mask], dataset.truth[mask])


if __name__ == "__main__":
    main()
