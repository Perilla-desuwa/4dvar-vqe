from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from q4dvar.data_loader import Lorenz96Dataset, load_lorenz96_csv
from q4dvar.models.lorenz96 import Lorenz96Model
from q4dvar.problem import Array
from q4dvar.solvers.classical import rmse
from q4dvar.solvers.qubo import run_sliding_window_qubo
from q4dvar.solvers.second_order_qubo import run_sliding_window_second_order_qubo


DEFAULT_INPUT_PATH = Path("气象海洋/气象海洋/小规模测试/lorenz96_test_1.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures")


@dataclass(frozen=True)
class MethodResult:
    name: str
    analysis: Array
    rmse: float
    error_energy: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Lorenz96 QUBO variants and block-selection policies.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Lorenz96 CSV file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG outputs.")
    parser.add_argument("--limit", type=int, default=120, help="Number of observation times to show in trajectory plots.")
    parser.add_argument("--dim-x", type=int, default=0, help="X-axis state dimension for phase plots.")
    parser.add_argument("--dim-y", type=int, default=1, help="Y-axis state dimension for phase plots.")
    parser.add_argument("--window", type=int, default=6, help="Number of observation times per 4D-Var window.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to window - 1.")
    parser.add_argument("--block-size", type=int, default=6, help="State dimensions per first-order QUBO block.")
    parser.add_argument("--second-order-block-size", type=int, default=2, help="State dimensions per second-order block.")
    parser.add_argument("--block-stride", type=int, default=None, help="Dimension stride between QUBO blocks.")
    parser.add_argument("--bits-per-dim", type=int, default=4, help="Binary variables per optimized dimension.")
    parser.add_argument(
        "--order-bits-per-dim",
        type=int,
        default=3,
        help="Binary variables per dimension for the linear vs second-order comparison.",
    )
    parser.add_argument("--radius", type=float, default=0.4, help="Maximum absolute increment per dimension.")
    parser.add_argument("--outer-loops", type=int, default=12, help="Number of QUBO block sweep passes per window.")
    parser.add_argument("--time-sweeps", type=int, default=3, help="Number of full passes over all time windows.")
    parser.add_argument(
        "--time-sweep-mode",
        choices=["carry", "background"],
        default="carry",
        help="How later time sweeps initialize each window.",
    )
    parser.add_argument("--solver", choices=["greedy", "qaoa"], default="greedy", help="QUBO backend.")
    parser.add_argument("--qaoa-reps", type=int, default=1, help="QAOA depth p.")
    parser.add_argument("--qaoa-shots", type=int, default=512, help="Shots per QAOA circuit.")
    parser.add_argument("--qaoa-optimizer-iterations", type=int, default=20, help="COBYLA iterations for QAOA angles.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-window progress logs.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    dataset = load_lorenz96_csv(args.input)
    plot_dataset = _slice_dataset(dataset, args.limit)
    model = Lorenz96Model(state_dim=dataset.state_dim)
    initial = model.forecast(dataset.observed[0], dataset.n_times)
    order_note = _parameter_note(
        args,
        dataset,
        bits_per_dim=args.order_bits_per_dim,
        label="linear/2nd-order",
        plot_points=plot_dataset.n_times,
    )
    block_note = _parameter_note(
        args,
        dataset,
        bits_per_dim=args.bits_per_dim,
        label="block selection",
        plot_points=plot_dataset.n_times,
    )

    linear = _run_linear(
        "linear QUBO",
        dataset,
        args,
        block_selection="hessian",
        bits_per_dim=args.order_bits_per_dim,
    )
    second_order = _run_second_order("2nd-order QUBO", dataset, args)

    block_results = [
        _run_linear(selection, dataset, args, block_selection=selection, bits_per_dim=args.bits_per_dim)
        for selection in ("cyclic", "gradient", "hessian")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_traj = args.output_dir / "lorenz96_test_1_linear_vs_second_order_trajectory.png"
    linear_energy = args.output_dir / "lorenz96_test_1_linear_vs_second_order_energy.png"
    block_traj = args.output_dir / "lorenz96_test_1_block_selection_trajectory.png"
    block_energy = args.output_dir / "lorenz96_test_1_block_selection_energy.png"

    _plot_phase(
        plot_dataset,
        {"initial": initial, linear.name: linear.analysis, second_order.name: second_order.analysis},
        linear_traj,
        dims=(args.dim_x, args.dim_y),
        title="Lorenz96 test: linear vs second-order QUBO",
        parameter_note=order_note,
    )
    _plot_metric_bars(
        [linear, second_order],
        linear_energy,
        title="Linear vs second-order QUBO metrics",
        parameter_note=order_note,
    )
    _plot_phase(
        plot_dataset,
        {result.name: result.analysis for result in block_results},
        block_traj,
        dims=(args.dim_x, args.dim_y),
        title="Lorenz96 test: block-selection policies",
        parameter_note=block_note,
    )
    _plot_metric_bars(block_results, block_energy, title="Block-selection policy metrics", parameter_note=block_note)

    print("Lorenz96 QUBO variant comparison")
    print("=" * 38)
    print(f"input:                     {args.input}")
    print(f"metric points/state_dim:   {dataset.n_times}/{dataset.state_dim}")
    print(f"plot points:               {plot_dataset.n_times}")
    print(f"solver:                    {args.solver}")
    print(f"linear vs second-order:    {linear_traj}")
    print(f"linear metric bars:        {linear_energy}")
    print(f"block trajectory:          {block_traj}")
    print(f"block metric bars:         {block_energy}")
    for result in [linear, second_order, *block_results]:
        print(f"{result.name:18s} rmse={result.rmse:.6f} error_energy={result.error_energy:.6f}")


def _run_linear(
    name: str,
    dataset: Lorenz96Dataset,
    args: argparse.Namespace,
    block_selection: str,
    bits_per_dim: int,
) -> MethodResult:
    result = run_sliding_window_qubo(
        dataset,
        window=args.window,
        stride=args.stride,
        block_size=args.block_size,
        block_stride=args.block_stride,
        block_selection=block_selection,
        bits_per_dim=bits_per_dim,
        radius=args.radius,
        outer_loops=args.outer_loops,
        time_sweeps=args.time_sweeps,
        time_sweep_mode=args.time_sweep_mode,
        seed=args.seed,
        solver=args.solver,
        qaoa_reps=args.qaoa_reps,
        qaoa_shots=args.qaoa_shots,
        qaoa_optimizer_iterations=args.qaoa_optimizer_iterations,
        verbose=not args.quiet,
    )
    return _method_result(name, dataset, result.analysis)


def _run_second_order(name: str, dataset: Lorenz96Dataset, args: argparse.Namespace) -> MethodResult:
    result = run_sliding_window_second_order_qubo(
        dataset,
        window=args.window,
        stride=args.stride,
        block_size=args.second_order_block_size,
        block_stride=args.block_stride,
        block_selection="hessian",
        bits_per_dim=args.order_bits_per_dim,
        radius=args.radius,
        outer_loops=args.outer_loops,
        time_sweeps=args.time_sweeps,
        time_sweep_mode=args.time_sweep_mode,
        seed=args.seed,
        solver=args.solver,
        qaoa_reps=args.qaoa_reps,
        qaoa_shots=args.qaoa_shots,
        qaoa_optimizer_iterations=args.qaoa_optimizer_iterations,
        verbose=not args.quiet,
    )
    return _method_result(name, dataset, result.analysis)


def _method_result(name: str, dataset: Lorenz96Dataset, analysis: Array) -> MethodResult:
    mask = np.isfinite(dataset.truth)
    error = analysis[mask] - dataset.truth[mask]
    return MethodResult(
        name=name,
        analysis=analysis,
        rmse=rmse(analysis[mask], dataset.truth[mask]),
        error_energy=0.5 * float(np.mean(error * error)),
    )


def _plot_phase(
    dataset: Lorenz96Dataset,
    trajectories: dict[str, Array],
    output_path: Path,
    dims: tuple[int, int],
    title: str,
    parameter_note: str,
) -> None:
    x_dim, y_dim = dims
    n_points = dataset.n_times
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    axis.plot(dataset.truth[:, x_dim], dataset.truth[:, y_dim], label="truth", color="black", linewidth=2.2)
    axis.scatter(
        dataset.observed[:, x_dim],
        dataset.observed[:, y_dim],
        label="observed",
        color="tab:orange",
        s=12,
        alpha=0.45,
    )

    styles = [
        {"color": "tab:red", "linestyle": "--", "linewidth": 1.7},
        {"color": "tab:blue", "linestyle": "-", "linewidth": 1.9},
        {"color": "tab:green", "linestyle": "-.", "linewidth": 1.9},
        {"color": "tab:purple", "linestyle": ":", "linewidth": 2.1},
    ]
    for style, (label, values) in zip(styles, trajectories.items()):
        axis.plot(values[:n_points, x_dim], values[:n_points, y_dim], label=label, **style)

    axis.set_xlabel(f"x{int(x_dim)}")
    axis.set_ylabel(f"x{int(y_dim)}")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.text(0.5, 0.012, parameter_note, ha="center", va="bottom", fontsize=8.5)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_metric_bars(results: list[MethodResult], output_path: Path, title: str, parameter_note: str) -> None:
    labels = [result.name for result in results]
    x_positions = np.arange(len(labels))
    width = 0.38

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    rmse_values = [result.rmse for result in results]
    energy_values = [result.error_energy for result in results]

    axes[0].bar(x_positions, rmse_values, width=0.65, color="tab:blue")
    axes[0].set_title("RMSE vs truth")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x_positions, energy_values, width=0.65, color="tab:purple")
    axes[1].set_title("Truth-error energy")
    axes[1].set_ylabel(r"$0.5\,\mathrm{mean}((x_a-x_t)^2)$")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    figure.suptitle(title)
    figure.text(0.5, 0.012, parameter_note, ha="center", va="bottom", fontsize=8.5)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _parameter_note(
    args: argparse.Namespace,
    dataset: Lorenz96Dataset,
    bits_per_dim: int,
    label: str,
    plot_points: int,
) -> str:
    stride = args.stride if args.stride is not None else max(1, args.window - 1)
    block_stride = args.block_stride if args.block_stride is not None else args.block_size
    return (
        f"data={Path(args.input).name}, metric_points={dataset.n_times}, plot_points={plot_points}, dim={dataset.state_dim}, "
        f"test={label}, solver={args.solver}, window/stride={args.window}/{stride}\n"
        f"block/stride={args.block_size}/{block_stride}, second-order block={args.second_order_block_size}, "
        f"bits={bits_per_dim}, radius={args.radius}, outer_loops={args.outer_loops}, "
        f"time_sweeps={args.time_sweeps}, mode={args.time_sweep_mode}"
    )


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


if __name__ == "__main__":
    main()
