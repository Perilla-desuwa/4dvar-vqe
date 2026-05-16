from __future__ import annotations

import argparse
import csv
from pathlib import Path

from q4dvar.data_loader import load_lorenz96_csv
from q4dvar.qubo_4dvar import run_sliding_window_qubo


DEFAULT_TRAIN_PATH = Path("气象海洋/气象海洋/小规模测试/lorenz96_train.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/lorenz96_train_qubo_result.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sliding-window Lorenz96 incremental QUBO 4D-Var.")
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_PATH, help="Official Lorenz96 CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Prediction CSV output path.")
    parser.add_argument("--window", type=int, default=8, help="Number of observation times per 4D-Var window.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to window - 1.")
    parser.add_argument("--block-size", type=int, default=10, help="State dimensions per local QUBO block.")
    parser.add_argument("--bits-per-dim", type=int, default=3, help="Binary variables per optimized dimension.")
    parser.add_argument("--radius", type=float, default=0.6, help="Maximum absolute increment per dimension.")
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
    result = run_sliding_window_qubo(
        dataset,
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
    write_prediction_csv(args.output, result.time_steps, result.analysis)

    print("Sliding-window Lorenz96 QUBO 4D-Var")
    print("=" * 40)
    print(f"input:              {args.input}")
    print(f"output:             {args.output}")
    print(f"states:             {result.analysis.shape}")
    print(f"window/stride:      {args.window}/{args.stride or max(1, args.window - 1)}")
    print(f"block/bits:         {args.block_size}/{args.bits_per_dim}")
    print(f"solver:             {args.solver}")
    if args.solver == "qaoa":
        print(f"QAOA reps/shots:    {args.qaoa_reps}/{args.qaoa_shots}")
    print(f"max QUBO variables: {result.max_qubo_variables}")
    if result.rmse_vs_truth is not None:
        print(f"RMSE vs truth:      {result.rmse_vs_truth:.6f}")


def write_prediction_csv(path: Path, time_steps, analysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_step", "dimension", "predicted_value"])
        for row_index, time_step in enumerate(time_steps):
            for dimension, value in enumerate(analysis[row_index]):
                writer.writerow([int(time_step), dimension, float(value)])


if __name__ == "__main__":
    main()
