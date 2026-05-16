from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from q4dvar.models.lorenz96 import Lorenz96Model


DEFAULT_OUTPUT_PATH = Path("outputs/datasets/lorenz96_synthetic.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic long-form Lorenz96 CSV dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--state-dim", type=int, default=40, help="Lorenz96 state dimension.")
    parser.add_argument("--n-times", type=int, default=500, help="Number of observation times to write.")
    parser.add_argument("--forcing", type=float, default=8.0, help="Lorenz96 forcing F.")
    parser.add_argument("--dt", type=float, default=0.05, help="RK4 model time step.")
    parser.add_argument("--steps-per-obs", type=int, default=2, help="RK4 steps between observation rows.")
    parser.add_argument("--obs-std", type=float, default=0.5, help="Observation noise standard deviation.")
    parser.add_argument("--spinup-steps", type=int, default=200, help="RK4 steps before writing the first row.")
    parser.add_argument("--initial-std", type=float, default=1.0, help="Random perturbation std around forcing.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    if args.state_dim < 4:
        raise ValueError("Lorenz96 state_dim must be at least 4.")
    if args.n_times <= 0:
        raise ValueError("n_times must be positive.")
    if args.steps_per_obs <= 0:
        raise ValueError("steps_per_obs must be positive.")

    rng = np.random.default_rng(args.seed)
    model = Lorenz96Model(
        state_dim=args.state_dim,
        forcing=args.forcing,
        dt=args.dt,
        steps_per_obs=args.steps_per_obs,
    )
    state = args.forcing + rng.normal(0.0, args.initial_std, size=args.state_dim)
    state = model.advance(state, args.spinup_steps)

    truth = np.zeros((args.n_times, args.state_dim), dtype=np.float64)
    observed = np.zeros_like(truth)
    for time_index in range(args.n_times):
        truth[time_index] = state
        observed[time_index] = state + rng.normal(0.0, args.obs_std, size=args.state_dim)
        state = model.advance(state, args.steps_per_obs)

    write_lorenz96_csv(args.output, truth, observed, args.steps_per_obs)
    print("Synthetic Lorenz96 dataset")
    print("=" * 28)
    print(f"output:        {args.output}")
    print(f"state_dim:     {args.state_dim}")
    print(f"n_times:       {args.n_times}")
    print(f"rows:          {args.n_times * args.state_dim}")
    print(f"steps_per_obs: {args.steps_per_obs}")
    print(f"obs_std:       {args.obs_std}")


def write_lorenz96_csv(path: Path, truth: np.ndarray, observed: np.ndarray, steps_per_obs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_step", "dimension", "true_value", "observed_value"])
        for time_index in range(truth.shape[0]):
            time_step = time_index * steps_per_obs
            for dimension in range(truth.shape[1]):
                writer.writerow(
                    [
                        int(time_step),
                        int(dimension),
                        float(truth[time_index, dimension]),
                        float(observed[time_index, dimension]),
                    ]
                )


if __name__ == "__main__":
    main()
