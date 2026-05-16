# Lorenz96 的量子辅助增量 4D-Var 方法说明

## 1. 方法概述

本项目实现了一种面向 Lorenz96 系统的数据同化方法。整体思路是把经典滚动窗口 4D-Var 与局部 QUBO 子问题结合起来，再用量子线路后端或快速近似后端求解这些 QUBO 子问题。

方法设计围绕三个约束展开：

- 预报模型必须保留 Lorenz96 的非线性动力学，数值积分使用四阶 Runge-Kutta。
- 优化问题不只做单时刻去噪，而是利用一个时间窗口内的多时刻观测。
- 每次 QUBO/量子线路调用都必须控制在固定量子比特预算内；默认配置每次最多使用 30 个二进制变量。

当前代码支持：

- 经典参考方法：直接观测、自由积分、最优插值/OI、随机 EnKF。
- Greedy QUBO 后端，用于快速调参和消融实验。
- 基于 Qiskit Aer 采样的 QAOA 后端。
- 面向小规模 toy problem 的 VQE 示例代码。

## 2. 动力学模型

系统状态记为：

```text
x(t) = [x_0(t), x_1(t), ..., x_{d-1}(t)]
```

其中 `d` 是可配置的状态维度。Lorenz96 动力学方程为：

```text
dx_i / dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
```

边界条件为循环边界。默认强迫项为：

```text
F = 8
```

预报模型使用四阶 Runge-Kutta 积分：

```text
x_{k+1} = RK4(x_k, dt)
```

观测模型为全状态带噪观测：

```text
y_k = x_k + epsilon_k,    epsilon_k ~ N(0, sigma_obs^2 I)
```

默认观测噪声标准差为：

```text
sigma_obs = 0.5
```

## 3. 强约束 4D-Var 目标函数

在一个同化窗口内，经典强约束 4D-Var 的目标函数为：

```text
J(x_0) =
  1/2 ||x_0 - x_b||^2_{B^{-1}}
  + 1/2 sum_k ||M_k(x_0) - y_k||^2_{R^{-1}}
```

其中：

- `x_0` 是当前窗口内需要优化的初始状态。
- `x_b` 是背景场。
- `M_k` 是从窗口初始时刻传播到第 `k` 个观测时刻的 Lorenz96 RK4 预报算子。
- `y_k` 是第 `k` 个观测时刻的观测值。
- `B` 和 `R` 分别是背景误差协方差矩阵和观测误差协方差矩阵。

当前实现使用对角协方差：

```text
B = sigma_b^2 I
R = sigma_obs^2 I
```

## 4. 增量线性化

直接把完整非线性 4D-Var 问题编码成 QUBO 并不现实。因此当前算法采用增量 4D-Var 思路：在当前猜测轨迹附近做局部线性化。

设：

```text
x_0 = x_g + delta
```

其中 `x_g` 是当前初猜状态，`delta` 是待求增量。在短窗口内，模型轨迹近似为：

```text
M_k(x_g + delta) ~= M_k(x_g) + T_k delta
```

其中 `T_k` 是切线传播矩阵。代码中用有限差分近似 `T_k`。把这个近似代入 4D-Var 目标函数后，可以得到关于 `delta` 的二次型：

```text
J(delta) = 1/2 delta^T A delta + g^T delta + const
```

这个二次型就是从 4D-Var 过渡到 QUBO 的关键。

## 5. 局部 Block QUBO 编码

完整状态维度可能远大于量子比特预算，因此代码不会一次优化全部状态维度，而是按维度 block 局部优化。

对于一个维度集合 `S`，只编码该局部 block 上的增量 `delta_S`。每个状态维度使用 `bits_per_dim` 个二进制变量。因此单次 QUBO 变量数为：

```text
n_qubits = block_size * bits_per_dim
```

默认完整配置为：

```text
block_size = 10
bits_per_dim = 3
n_qubits = 30
```

这样每次 QUBO/量子线路调用都不超过 30 个变量。

每个连续增量会被映射到一个二进制网格：

```text
delta_i = offset_i + sum_j weight_j q_{ij}
q_{ij} in {0, 1}
```

把二进制编码代回二次目标函数后，可以得到标准 QUBO：

```text
min_q q^T Q q + const
```

其中 `Q` 是 QUBO 矩阵。

## 6. Window 与 Block Sweep

算法中有两种局部性：

- `window`：时间方向的局部性。它控制一次 4D-Var 目标函数包含多少个观测时刻。
- `block`：状态维度方向的局部性。它控制一次 QUBO 优化多少个状态维度。

对于每一个时间窗口，算法流程为：

```text
for outer_loop in 1..N:
    for each state block:
        build local QUBO
        solve QUBO
        decode increment
        accept increment only if nonlinear window cost decreases
```

`block_stride` 可以小于 `block_size`，从而允许 block 重叠。例如：

```text
block_size = 10
block_stride = 5

blocks:
0..9, 5..14, 10..19, ...
```

重叠 block 可以减轻硬切分状态维度带来的边界效应，也更适合 Lorenz96 这种局部耦合系统。

## 7. QAOA 后端

QAOA 后端会把 QUBO 转换成 Ising 哈密顿量。对于二进制变量 `q_i`，使用映射：

```text
q_i = (1 - z_i) / 2
```

代价哈密顿量形式为：

```text
H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
```

QAOA 电路在代价演化和混合演化之间交替：

```text
U_C(gamma) = exp(-i gamma H_C)
U_M(beta)  = exp(-i beta sum_i X_i)
```

初态为均匀叠加态：

```text
|+>^{\otimes n}
```

经过 `p` 层 QAOA 后：

```text
|psi> = product_l U_M(beta_l) U_C(gamma_l) |+>^{\otimes n}
```

实现中使用 Qiskit Aer 对线路采样，并从采样得到的 bitstring 中选择 QUBO 能量最低的解。代码也支持用 COBYLA 调整 QAOA 角度，但在端到端大规模实验中，固定浅层角度更快。

## 8. 非线性代价接受准则

由于 QUBO 来自局部线性化，QUBO 的最优解不能无条件接受。解码出候选增量后，算法会重新计算原始非线性 4D-Var 窗口代价。

只有当：

```text
J_nonlinear(candidate) <= J_nonlinear(current)
```

时，才接受该 block 的更新。

这个规则类似 trust-region 保护，可以避免线性化误差导致轨迹变差。

## 9. 经典 Baseline

项目中包含四类经典参考方法：

- `observed`：直接使用带噪观测作为分析场。
- `free_run`：从第一个观测值出发自由积分，不再吸收后续观测。
- `optimal_interpolation`：标量增益的 3D-Var/OI cycling。
- `stochastic_enkf`：全状态随机 EnKF。

这些 baseline 用于区分“单纯观测去噪”和“利用动力学一致性”的效果差异。

## 10. 示例结果

在一个 40 维 Lorenz96 数据集上，取前 120 个观测时刻，曾得到如下代表性 RMSE：

```text
initial/free run RMSE:  4.363070
OI baseline RMSE:      0.409237
greedy QUBO RMSE:      0.189511
```

示例二维相图路径：

```text
outputs/lorenz96_baseline_solver_x0_x1.png
```

该图在 `(x_0, x_1)` 平面中对比了真实轨迹、初始自由积分轨迹、经典 baseline 轨迹和 QUBO 辅助分析轨迹。

## 11. 复现实验

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

生成合成数据集：

```powershell
.\.venv\Scripts\python runners\generate_lorenz96_dataset.py `
  --output "outputs\datasets\lorenz96_dim80_t200.csv" `
  --state-dim 80 `
  --n-times 200 `
  --steps-per-obs 2 `
  --obs-std 0.5
```

运行 greedy QUBO 4D-Var：

```powershell
.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "outputs\datasets\lorenz96_dim80_t200.csv" `
  --output "outputs\lorenz96_dim80_t200_greedy.csv" `
  --window 8 `
  --stride 4 `
  --block-size 10 `
  --block-stride 5 `
  --bits-per-dim 3 `
  --radius 0.4 `
  --solver greedy
```

运行采样式 QAOA：

```powershell
.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "outputs\datasets\lorenz96_dim80_t200.csv" `
  --output "outputs\lorenz96_dim80_t200_qaoa.csv" `
  --window 8 `
  --stride 4 `
  --block-size 10 `
  --block-stride 5 `
  --bits-per-dim 3 `
  --radius 0.4 `
  --solver qaoa `
  --qaoa-reps 1 `
  --qaoa-shots 256
```

生成对比图：

```powershell
.\.venv\Scripts\python runners\plot_lorenz96_baseline_compare.py `
  --input "outputs\datasets\lorenz96_dim80_t200.csv" `
  --output "outputs\lorenz96_dim80_t200_compare.png" `
  --limit 120 `
  --baseline observed `
  --solver greedy `
  --window 8 `
  --stride 4 `
  --block-size 10 `
  --block-stride 5 `
  --bits-per-dim 3
```

## 12. 方法总结

本方法可以概括为一种量子辅助增量 4D-Var：

- Lorenz96 提供非线性动力学约束。
- 滚动时间窗口保证线性化近似在局部时间内有效。
- 局部 block QUBO 保证每次量子调用不超过变量预算。
- QAOA 提供基于量子线路的 QUBO 求解路径。
- Greedy QUBO 提供快速可扩展的对照后端。
- 非线性代价接受准则避免线性化代理问题产生有害更新。

核心思想不是把完整高维非线性轨迹一次性塞进一个量子线路，而是把数据同化问题拆成许多小规模、局部、具有动力学意义的 QUBO 子问题。
# Quantum-Assisted Incremental 4D-Var for Lorenz96

## 1. Overview

This project implements a hybrid data-assimilation method for the Lorenz96
system. The method combines a classical rolling-window 4D-Var formulation with
local QUBO subproblems solved by a quantum-inspired or quantum-circuit backend.

The implementation is designed around three constraints:

- The forecast model must preserve Lorenz96 dynamics through RK4 integration.
- The optimization problem should use observations over a time window, not just
  independent single-time denoising.
- Each QUBO/quantum call must stay within a configurable qubit budget; the
  default configuration uses at most 30 binary variables per call.

The current implementation supports:

- Classical baselines: direct observation, free run, optimal interpolation, and
  stochastic EnKF.
- Greedy QUBO backend for fast ablation.
- QAOA backend using Qiskit Aer sampling.
- VQE demo code for small toy problems.

## 2. Dynamical Model

The state is a vector

```text
x(t) = [x_0(t), x_1(t), ..., x_{d-1}(t)]
```

where `d` is configurable. The Lorenz96 dynamics are

```text
dx_i / dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
```

with cyclic boundary conditions. The default forcing is `F = 8`.

The forecast model uses fourth-order Runge-Kutta integration:

```text
x_{k+1} = RK4(x_k, dt)
```

Observations are full-state noisy measurements:

```text
y_k = x_k + epsilon_k,    epsilon_k ~ N(0, sigma_obs^2 I)
```

The default observation noise standard deviation is `sigma_obs = 0.5`.

## 3. Strong-Constraint 4D-Var Objective

For one assimilation window, the classical strong-constraint 4D-Var objective is

```text
J(x_0) =
  1/2 ||x_0 - x_b||^2_{B^{-1}}
  + 1/2 sum_k ||M_k(x_0) - y_k||^2_{R^{-1}}
```

where:

- `x_0` is the initial state to optimize in the current window.
- `x_b` is the background state.
- `M_k` is the Lorenz96 RK4 forecast from the window initial time to observation
  time `k`.
- `y_k` is the observation at time `k`.
- `B` and `R` are background and observation error covariance matrices.

The current implementation uses diagonal covariance matrices:

```text
B = sigma_b^2 I
R = sigma_obs^2 I
```

## 4. Incremental Linearization

Directly encoding the full nonlinear 4D-Var problem as a QUBO is not practical.
Instead, the solver uses an incremental 4D-Var approximation around a current
guess trajectory.

Let

```text
x_0 = x_g + delta
```

where `x_g` is the current guess. For short windows, the model trajectory is
linearized as

```text
M_k(x_g + delta) ~= M_k(x_g) + T_k delta
```

where `T_k` is approximated by finite differences. Substituting this into the
4D-Var objective gives a quadratic approximation:

```text
J(delta) = 1/2 delta^T A delta + g^T delta + const
```

This quadratic form is the bridge from 4D-Var to QUBO.

## 5. Block-Local QUBO Encoding

The full state dimension can be much larger than the qubit budget, so the
increment is optimized block by block.

For a block of dimensions `S`, only the local increment `delta_S` is encoded.
Each dimension uses `bits_per_dim` binary variables. Therefore:

```text
n_qubits = block_size * bits_per_dim
```

The default full-size setting is:

```text
block_size = 10
bits_per_dim = 3
n_qubits = 30
```

This satisfies the per-call qubit constraint.

Each local continuous increment is mapped to a binary grid:

```text
delta_i = offset_i + sum_j weight_j q_{ij}
q_{ij} in {0, 1}
```

After substituting the binary encoding into the quadratic objective, the solver
obtains

```text
min_q q^T Q q + const
```

where `Q` is the QUBO matrix.

## 6. Window and Block Sweeps

The algorithm has two independent forms of locality:

- `window`: time locality. It controls how many observation times are included
  in one 4D-Var objective.
- `block`: state locality. It controls how many state dimensions are optimized
  in one QUBO call.

For each time window:

```text
for outer_loop in 1..N:
    for each state block:
        build local QUBO
        solve QUBO
        decode increment
        accept increment only if nonlinear window cost decreases
```

The block stride can be smaller than the block size, allowing overlapping
blocks:

```text
block_size = 10
block_stride = 5

blocks:
0..9, 5..14, 10..19, ...
```

Overlapping blocks improve smoothness and reduce artifacts from hard dimension
partitioning.

## 7. QAOA Backend

The QAOA backend converts the QUBO into an Ising Hamiltonian over `Z` and `ZZ`
terms.

For binary variables `q_i`, the mapping is

```text
q_i = (1 - z_i) / 2
```

The cost Hamiltonian has the form

```text
H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
```

The QAOA circuit alternates between:

```text
U_C(gamma) = exp(-i gamma H_C)
U_M(beta)  = exp(-i beta sum_i X_i)
```

The circuit starts from a uniform superposition:

```text
|+>^{\otimes n}
```

and applies `p` layers:

```text
|psi> = product_l U_M(beta_l) U_C(gamma_l) |+>^{\otimes n}
```

The implementation samples bitstrings using Qiskit Aer and selects the sampled
bitstring with the lowest QUBO energy. Optional COBYLA tuning of QAOA angles is
available, but fixed shallow angles are faster for large end-to-end runs.

## 8. Acceptance Rule

Because the QUBO is built from a local linearization, the QUBO optimum is not
accepted blindly. After decoding the proposed increment, the solver evaluates
the original nonlinear 4D-Var window cost.

The increment is accepted only if:

```text
J_nonlinear(candidate) <= J_nonlinear(current)
```

This trust-region-like rule prevents poor linearized updates from degrading the
trajectory.

## 9. Classical Baselines

The project includes four reference baselines:

- `observed`: directly use noisy observations.
- `free_run`: start from the first observation and forecast without updates.
- `optimal_interpolation`: scalar-gain 3D-Var/OI cycling.
- `stochastic_enkf`: full-state stochastic EnKF.

These baselines help separate the benefit of dynamical consistency from simple
observation denoising.

## 10. Example Results

On a 40-dimensional Lorenz96 dataset, using the first 120 observation times,
the following representative RMSE values were observed:

```text
initial/free run RMSE:  4.363070
OI baseline RMSE:      0.409237
greedy QUBO RMSE:      0.189511
```

An example phase-space comparison is available at:

```text
outputs/lorenz96_baseline_solver_x0_x1.png
```

This figure compares truth, the initial free run, the classical baseline, and
the QUBO-assisted analysis trajectory in the `(x_0, x_1)` plane.

## 11. Reproducibility

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

Generate a synthetic dataset:

```powershell
.\.venv\Scripts\python runners\generate_lorenz96_dataset.py `
  --output "outputs\datasets\lorenz96_dim80_t200.csv" `
  --state-dim 80 `
  --n-times 200 `
  --steps-per-obs 2 `
  --obs-std 0.5
```

Run greedy QUBO 4D-Var:

```powershell
.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "outputs\datasets\lorenz96_dim80_t200.csv" `
  --output "outputs\lorenz96_dim80_t200_greedy.csv" `
  --window 8 `
  --stride 4 `
  --block-size 10 `
  --block-stride 5 `
  --bits-per-dim 3 `
  --radius 0.4 `
  --solver greedy
```

Run sampled QAOA:

```powershell
.\.venv\Scripts\python runners\run_lorenz96_qubo.py `
  --input "outputs\datasets\lorenz96_dim80_t200.csv" `
  --output "outputs\lorenz96_dim80_t200_qaoa.csv" `
  --window 8 `
  --stride 4 `
  --block-size 10 `
  --block-stride 5 `
  --bits-per-dim 3 `
  --radius 0.4 `
  --solver qaoa `
  --qaoa-reps 1 `
  --qaoa-shots 256
```

Generate a comparison plot:

```powershell
.\.venv\Scripts\python runners\plot_lorenz96_baseline_compare.py `
  --input "outputs\datasets\lorenz96_dim80_t200.csv" `
  --output "outputs\lorenz96_dim80_t200_compare.png" `
  --limit 120 `
  --baseline observed `
  --solver greedy `
  --window 8 `
  --stride 4 `
  --block-size 10 `
  --block-stride 5 `
  --bits-per-dim 3
```

## 12. Summary

The method is a quantum-assisted incremental 4D-Var algorithm:

- Lorenz96 provides the nonlinear dynamical constraint.
- Rolling windows keep the linearization valid.
- Block-local QUBOs keep each quantum call within the qubit budget.
- QAOA provides a circuit-based QUBO backend.
- Greedy QUBO provides a fast control backend for scaling experiments.
- Nonlinear acceptance prevents harmful updates from the linearized surrogate.

The key idea is not to encode the full nonlinear high-dimensional trajectory in
one quantum circuit. Instead, the method decomposes the assimilation problem
into many small, local, dynamically meaningful QUBO subproblems.
