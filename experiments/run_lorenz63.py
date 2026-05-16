from __future__ import annotations

import argparse
from pathlib import Path

from q4dvar.classical_4dvar import cost, rmse, solve_classical
from q4dvar.plotting import plot_state_trajectories
from q4dvar.toy_model import generate_lorenz63_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Lorenz-63 4D-Var trajectory demo.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic observations.")
    parser.add_argument("--window", type=int, default=25, help="Assimilation window length.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lorenz63_trajectories.png"),
        help="Path for the trajectory comparison figure.",
    )
    args = parser.parse_args()

    problem = generate_lorenz63_problem(seed=args.seed, window=args.window)
    analysis_initial = solve_classical(problem)

    truth_trajectory = problem.model.forecast(problem.truth_initial, args.window)
    background_trajectory = problem.model.forecast(problem.background, args.window)
    analysis_trajectory = problem.model.forecast(analysis_initial, args.window)
    output_path = plot_state_trajectories(
        truth=truth_trajectory,
        background=background_trajectory,
        analysis=analysis_trajectory,
        output_path=args.output,
        title="Lorenz-63 4D-Var Trajectory Comparison",
        component_names=("x", "y", "z"),
    )

    print("Lorenz-63 4D-Var trajectory demo")
    print("=" * 36)
    print(f"Truth initial state:       {problem.truth_initial}")
    print(f"Background initial state:  {problem.background}")
    print(f"Analysis initial state:    {analysis_initial}")
    print()
    print(f"Background initial RMSE:   {rmse(problem.background, problem.truth_initial):.6f}")
    print(f"Analysis initial RMSE:     {rmse(analysis_initial, problem.truth_initial):.6f}")
    print(f"Background cost:           {cost(problem, problem.background):.6f}")
    print(f"Analysis cost:             {cost(problem, analysis_initial):.6f}")
    print(f"Figure written to:         {output_path}")


if __name__ == "__main__":
    main()
