# Quantum-Assisted Toy 4D-Var Demo

This repository is a minimal runnable template for a quantum hackathon project on
data assimilation in meteorology and oceanography.

The demo keeps the first version intentionally small but extensible:

- pluggable forecast models, currently a linear rotation model and Lorenz-63;
- synthetic observations over a short 4D-Var window;
- a classical 4D-Var baseline, exact for the linear model and numerical for
  nonlinear models;
- a VQE path that discretizes the 4D-Var cost into a diagonal Hamiltonian and
  searches for a low-cost initial state with Qiskit;
- a plotting helper for truth, background forecast, and analysis forecast
  trajectories.

This is not yet a claim that VQE replaces a production adjoint model. It is a
template for testing the end-to-end story: model forecast, observations,
classical baseline, quantum optimization hook, and comparable metrics.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python experiments\run_demo.py
```

Run the Lorenz-63 trajectory demo:

```powershell
.\.venv\Scripts\python experiments\run_lorenz63.py
```

The Lorenz script writes `outputs/lorenz63_trajectories.png`, comparing the
true trajectory, the unassimilated background forecast, and the assimilated
forecast.

Useful knobs:

```powershell
.\.venv\Scripts\python experiments\run_demo.py --maxiter 1000
```

Increasing `--bits-per-dim` gives a finer grid but also increases the number of
qubits and makes the VQE optimization less stable. Keep the default first for a
smoke test.

## Project Layout

```text
src/q4dvar/
  toy_model.py        model interface, linear model, Lorenz-63, synthetic obs
  classical_4dvar.py  4D-Var cost, exact linear solve, nonlinear optimizer
  quantum_vqe.py      grid encoding, diagonal Hamiltonian, VQE solver
  plotting.py         trajectory comparison plots
experiments/
  run_demo.py         linear model + VQE smoke test
  run_lorenz63.py     Lorenz-63 classical 4D-Var trajectory demo
```

## Current Mathematical Mapping

For a linear model and linear observation operator, the toy 4D-Var objective is

```text
J(x0) = 1/2 ||x0 - xb||^2_B^-1
      + 1/2 sum_t ||H M_t x0 - y_t||^2_R^-1
```

The classical baseline solves this quadratic objective directly for the linear
model. For nonlinear models like Lorenz-63, it minimizes the same 4D-Var cost
numerically over the initial state.

The quantum demo samples the objective on a binary grid, expands the resulting
diagonal energy table as a Pauli-Z Hamiltonian, and uses VQE to find a low-cost
grid point. The current VQE smoke test is kept on the 2D linear model so it
stays small enough to run quickly.

## Next Steps

- Replace synthetic observations with the official dataset loader when it is
  available.
- Add a QAOA or quantum annealing-style solver for QUBO-shaped variants.
- Add a Lorenz-96 model once the baseline interface is stable.
- Benchmark regular Qiskit Aer against the competition GPU-enabled Aer build.
