from __future__ import annotations

import argparse
from pathlib import Path

from q4dvar.data_loader import load_lorenz96_csv
from q4dvar.lorenz96 import Lorenz96Model
from q4dvar.plotting import plot_phase_trajectory_comparison
from q4dvar.qubo_4dvar import run_sliding_window_qubo


DEFAULT_TRAIN_PATH = Path("气象海洋/气象海洋/小规模测试/lorenz96_train.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/lorenz96_qubo_phase_x0_x1.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Lorenz96 truth/observed/background/QUBO phase trajectories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_PATH, help="Official Lorenz96 CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG path.")
    parser.add_argument("--limit", type=int, default=120, help="Number of observation times to plot.")
    parser.add_argument("--dim-x", type=int, default=0, help="X-axis state dimension.")
    parser.add_argument("--dim-y", type=int, default=1, help="Y-axis state dimension.")
    parser.add_argument("--window", type=int, default=6, help="Number of observation times per 4D-Var window.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to window - 1.")
    parser.add_argument("--block-size", type=int, default=10, help="State dimensions per local QUBO block.")
    parser.add_argument("--bits-per-dim", type=int, default=3, help="Binary variables per optimized dimension.")
    parser.add_argument("--radius", type=float, default=0.4, help="Maximum absolute increment per dimension.")
    parser.add_argument("--outer-loops", type=int, default=1, help="Number of QUBO block sweep passes per window.")
    parser.add_argument("--solver", choices=["qaoa", "greedy"], default="qaoa", help="QUBO backend.")
    parser.add_argument("--qaoa-reps", type=int, default=1, help="QAOA depth p.")
    parser.add_argument("--qaoa-shots", type=int, default=256, help="Shots per QAOA circuit.")
    parser.add_argument(
        "--qaoa-optimizer-iterations",
        type=int,
        default=0,
        help="COBYLA iterations for QAOA angle tuning; 0 uses fixed angles.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the QUBO backend.")
    args = parser.parse_args()

    dataset = load_lorenz96_csv(args.input)
    plot_limit = min(args.limit, dataset.n_times)
    plot_dataset = dataset
    if plot_limit < dataset.n_times:
        plot_dataset = type(dataset)(
            time_steps=dataset.time_steps[:plot_limit],
            truth=dataset.truth[:plot_limit],
            observed=dataset.observed[:plot_limit],
            observed_mask=dataset.observed_mask[:plot_limit],
            source_path=dataset.source_path,
        )

    result = run_sliding_window_qubo(
        plot_dataset,
        window=args.window,
        stride=args.stride,
        block_size=args.block_size,
        bits_per_dim=args.bits_per_dim,
        radius=args.radius,
        outer_loops=args.outer_loops,
        seed=args.seed,
        solver=args.solver,
        qaoa_reps=args.qaoa_reps,
        qaoa_shots=args.qaoa_shots,
        qaoa_optimizer_iterations=args.qaoa_optimizer_iterations,
    )
    model = Lorenz96Model(state_dim=plot_dataset.state_dim)
    background = model.forecast(plot_dataset.observed[0], plot_dataset.n_times)

    output = plot_phase_trajectory_comparison(
        truth=plot_dataset.truth,
        observed=plot_dataset.observed,
        background=background,
        analysis=result.analysis,
        output_path=args.output,
        dims=(args.dim_x, args.dim_y),
        title=f"Lorenz96 x{args.dim_x}-x{args.dim_y}: truth, observed, free run, QUBO analysis",
    )

    print("Lorenz96 phase plot")
    print("=" * 24)
    print(f"output:             {output}")
    print(f"points:             {plot_dataset.n_times}")
    print(f"dims:               x{args.dim_x}, x{args.dim_y}")
    print(f"solver:             {args.solver}")
    if args.solver == "qaoa":
        print(f"QAOA reps/shots:    {args.qaoa_reps}/{args.qaoa_shots}")
    print(f"max QUBO variables: {result.max_qubo_variables}")
    if result.rmse_vs_truth is not None:
        print(f"RMSE vs truth:      {result.rmse_vs_truth:.6f}")


if __name__ == "__main__":
    main()
