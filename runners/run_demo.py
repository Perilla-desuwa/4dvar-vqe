from __future__ import annotations

import argparse

from q4dvar.models.toy import generate_problem
from q4dvar.solvers.classical import cost, rmse, solve_classical
from q4dvar.solvers.vqe import solve_vqe


def aer_backend_name() -> str:
    """Report whether qiskit-aer is available in the active environment."""

    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        return "qiskit-aer not installed"
    return AerSimulator(method="statevector").name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal toy 4D-Var + VQE demo.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic observations.")
    parser.add_argument("--window", type=int, default=6, help="Assimilation window length.")
    parser.add_argument("--bits-per-dim", type=int, default=2, help="Binary grid bits per state dimension.")
    parser.add_argument("--maxiter", type=int, default=600, help="COBYLA iterations for VQE.")
    args = parser.parse_args()

    problem = generate_problem(seed=args.seed, window=args.window)
    classical = solve_classical(problem)
    quantum = solve_vqe(
        problem,
        bits_per_dim=args.bits_per_dim,
        maxiter=args.maxiter,
    )

    print("Toy quantum-assisted 4D-Var demo")
    print("=" * 36)
    print(f"Aer backend:          {aer_backend_name()}")
    print(f"Model:                {problem.model.name}")
    print(f"Truth initial state:  {problem.truth_initial}")
    print(f"Background state:     {problem.background}")
    print()
    print("Classical 4D-Var baseline")
    print(f"  analysis state:     {classical}")
    print(f"  cost:               {cost(problem, classical):.6f}")
    print(f"  RMSE vs truth:      {rmse(classical, problem.truth_initial):.6f}")
    print()
    print("VQE over discretized 4D-Var cost")
    print(f"  n_qubits:           {quantum.n_qubits}")
    print(f"  best bitstring:     {quantum.bitstring}")
    print(f"  grid state:         {quantum.state}")
    print(f"  grid cost:          {quantum.cost:.6f}")
    print(f"  expectation:        {quantum.expectation:.6f}")
    print(f"  grid min cost:      {quantum.grid_min_cost:.6f}")
    print(f"  gap to grid min:    {quantum.gap_to_grid_min:.6f}")
    print(f"  RMSE vs truth:      {rmse(quantum.state, problem.truth_initial):.6f}")
    print(f"  optimizer success:  {quantum.optimizer_success}")
    print(f"  optimizer message:  {quantum.optimizer_message}")


if __name__ == "__main__":
    main()
