# Quantum-Assisted 4D-Var for Lorenz96

This repository contains a runnable prototype for 40-dimensional Lorenz96 data
assimilation. The current main path solves a sliding-window incremental 4D-Var
problem with local QUBO subproblems, and includes classical reference baselines
for comparison.

The implementation is intentionally hybrid:

- Lorenz96 dynamics are integrated with fixed-step RK4.
- Long-form CSV files are loaded into dense `time x dimension` arrays.
- Each 4D-Var window is linearized around the current guess trajectory.
- A local state increment is encoded as a QUBO with at most 30 binary variables.
- The default QUBO backend is sampled QAOA on Qiskit Aer; a greedy backend is
  kept for fast debugging and ablation.

This is a working research prototype. The quantum part is isolated behind the
QUBO solver interface so QAOA settings, annealing-style solvers, or a stronger
optimizer can be swapped in without rewriting the Lorenz96 assimilation loop.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

Run the Lorenz96 sliding-window QUBO/QAOA pipeline on a local train file:

```powershell
.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "气象海洋\气象海洋\小规模测试\lorenz96_train.csv" `
  --output "outputs\lorenz96_train_qubo_result.csv" `
  --window 6 `
  --block-size 10 `
  --bits-per-dim 3 `
  --radius 0.4 `
  --solver qaoa `
  --qaoa-shots 256
```

Generate a synthetic Lorenz96 CSV with custom dimension and length:

```powershell
.\.venv\Scripts\python runners\generate_lorenz96_dataset.py `
  --output "outputs\datasets\lorenz96_dim80_t200.csv" `
  --state-dim 80 `
  --n-times 200 `
  --steps-per-obs 2 `
  --obs-std 0.5
```

The default full-size configuration uses `10 x 3 = 30` QUBO variables per local
block. On a CPU simulator this can be slow, because many sampled QAOA circuits
are executed across the sliding windows.

For quick debugging, use smaller local QUBOs or the greedy backend:

```powershell
.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "气象海洋\气象海洋\小规模测试\lorenz96_train.csv" `
  --block-size 4 `
  --bits-per-dim 2 `
  --solver qaoa `
  --qaoa-shots 64

.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "气象海洋\气象海洋\小规模测试\lorenz96_train.csv" `
  --solver greedy
```

## Classical Baselines

Run lightweight reference methods on the same Lorenz96 CSV:

```powershell
.\.venv\Scripts\python runners\run_lorenz96_baselines.py `
  --input "气象海洋\气象海洋\小规模测试\lorenz96_train.csv" `
  --output-dir "outputs\baselines"
```

Implemented baselines:

- `observed`: directly uses noisy observations as the analysis.
- `free_run`: forecasts from the first observation without later updates.
- `optimal_interpolation`: scalar-gain 3D-Var/OI cycling.
- `stochastic_enkf`: a full-state stochastic ensemble Kalman filter with
  configurable ensemble size, inflation, and observation noise.

These baselines are intended to provide a local reference curve before tuning
the QUBO/QAOA path.

## Visualization

Generate a 2D phase plot for the first two Lorenz96 dimensions. The plot
contains truth, noisy observations, the unassimilated free run, and the
assimilated trajectory.

```powershell
.\.venv\Scripts\python runners\plot_lorenz96_qubo_phase.py `
  --input "气象海洋\气象海洋\小规模测试\lorenz96_train.csv" `
  --output "outputs\lorenz96_qubo_phase_x0_x1.png" `
  --limit 120 `
  --window 6 `
  --block-size 10 `
  --bits-per-dim 3 `
  --radius 0.4 `
  --solver qaoa
```

Use `--dim-x` and `--dim-y` to plot other state dimensions.

## Project Layout

```text
src/q4dvar/
  problem.py          shared Array, ForecastModel, AssimilationProblem types
  data_loader.py      Lorenz96 CSV loader
  plotting.py         trajectory and phase-space plots
  models/
    toy.py            toy linear and Lorenz-63 models
    lorenz96.py       Lorenz96 RK4 model and dataset-to-problem builder
  solvers/
    classical.py      public wrapper for the classic 4D-Var solver
    baselines.py      observed/free-run/OI/EnKF baselines
    qubo.py           public wrapper for incremental QUBO 4D-Var
    greedy.py         public wrapper for the greedy QUBO backend
    qaoa.py           public wrapper for the sampled QAOA backend
    vqe.py            public wrapper for the earlier VQE demo
    4dvar/
      classical.py    classic 4D-Var objective and solver
      qubo.py         incremental 4D-Var QUBO construction and dispatcher
      greedy.py       greedy QUBO backend entry point
      qaoa.py         sampled QAOA QUBO backend entry point
      vqe.py          VQE demo for small toy problems
runners/
  run_lorenz96_qubo.py        Lorenz96 QUBO/QAOA run
  run_lorenz96_baselines.py   classical baseline runner
  generate_lorenz96_dataset.py synthetic Lorenz96 CSV generator
  plot_lorenz96_qubo_phase.py Lorenz96 2D trajectory comparison plot
  run_demo.py                 linear model + VQE smoke test
  run_lorenz63.py             Lorenz-63 classical 4D-Var trajectory demo
```

## Data Format

The expected CSV format is long-form:

```text
time_step,dimension,true_value,observed_value
0,0,...
0,1,...
...
2,0,...
```

`data_loader.load_lorenz96_csv()` converts it to:

- `time_steps`: sorted observation times.
- `truth`: shape `(n_times, 40)`, with `NaN` allowed for hidden test truth.
- `observed`: shape `(n_times, 40)`.
- `observed_mask`: boolean observation availability mask.

The loader validates required columns, duplicate `(time_step, dimension)` rows,
dimension bounds, and complete observations.

## Method

The strong-constraint 4D-Var objective over one observation window is:

```text
J(x0) = 1/2 ||x0 - xb||^2_B^-1
      + 1/2 sum_k ||H M_k(x0) - y_k||^2_R^-1
```

For Lorenz96, `H = I`, `R = 0.5^2 I`, and `M_k` is the RK4 forecast operator.
To obtain a QUBO, the code uses an incremental 4D-Var approximation around the
current guess `xg`:

```text
x0 = xg + delta
M_k(x0) ~= M_k(xg) + T_k delta
```

After linearization, each local increment block has a quadratic objective:

```text
J(delta) = 1/2 delta^T A delta + g^T delta + const
```

The block increment is discretized with `bits_per_dim` binary variables per
state dimension, producing a QUBO:

```text
min_q q^T Q q + const, q in {0, 1}^n
```

With the default `block_size=10` and `bits_per_dim=3`, each QUBO uses exactly
30 binary variables. The QAOA backend converts `Q` to Ising `Z` and `ZZ` terms,
runs a shallow sampled QAOA circuit on Aer, and accepts an increment only if it
improves the original nonlinear window cost.

## Useful Parameters

- `--window`: number of observation times in one 4D-Var window.
- `--stride`: step between windows; defaults to `window - 1`.
- `--block-size`: number of state dimensions optimized per QUBO block.
- `--bits-per-dim`: binary resolution per state dimension.
- `--radius`: maximum local increment magnitude encoded by the QUBO grid.
- `--solver`: `qaoa` for the quantum path, `greedy` for fast debugging.
- `--qaoa-reps`: QAOA depth `p`.
- `--qaoa-shots`: shots per sampled QAOA circuit.
- `--qaoa-optimizer-iterations`: optional COBYLA tuning of QAOA angles. `0`
  uses fixed angles, which is much faster for end-to-end runs.

## Outputs

`runners/run_lorenz96_qubo.py` and `runners/run_lorenz96_baselines.py`
write prediction CSVs:

```text
time_step,dimension,predicted_value
0,0,...
0,1,...
```

`runners/plot_lorenz96_qubo_phase.py` writes a PNG phase plot, usually under
`outputs/`.

## Legacy Toy Demos

The original toy scripts are still available:

```powershell
.\.venv\Scripts\python runners\run_demo.py
.\.venv\Scripts\python runners\run_lorenz63.py
```

They are useful for quick checks of the earlier VQE and classical 4D-Var code,
but the Lorenz96 QUBO/QAOA path is the main implementation.
