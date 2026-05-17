from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from q4dvar.data_loader import Lorenz96Dataset
from q4dvar.models.lorenz96 import Lorenz96Model, make_lorenz96_problem
from q4dvar.problem import Array, AssimilationProblem
from q4dvar.solvers.classical import rmse


QuboSolverName = Literal["greedy", "qaoa"]
BlockSelectionName = Literal["cyclic", "gradient", "hessian"]
TimeSweepMode = Literal["carry", "background"]


@dataclass(frozen=True)
class QuboEncoding:
    """Binary grid encoding for a local Lorenz96 increment."""

    dimensions: tuple[int, ...]
    bits_per_dim: int
    radius: float
    offsets: Array
    weights: Array

    @property
    def n_variables(self) -> int:
        return len(self.dimensions) * self.bits_per_dim


@dataclass(frozen=True)
class QuboProblem:
    """Upper-triangular QUBO matrix and metadata for one local increment."""

    matrix: Array
    constant: float
    encoding: QuboEncoding

    @property
    def n_variables(self) -> int:
        return int(self.matrix.shape[0])


@dataclass(frozen=True)
class QuboSolveResult:
    bits: np.ndarray
    energy: float
    n_variables: int
    n_sweeps: int


@dataclass(frozen=True)
class WindowAssimilationResult:
    initial_state: Array
    forecast: Array
    window_cost: float
    max_qubo_variables: int


@dataclass(frozen=True)
class SlidingAssimilationResult:
    analysis: Array
    time_steps: np.ndarray
    rmse_vs_truth: float | None
    max_qubo_variables: int


def build_incremental_qubo(
    problem: AssimilationProblem,
    initial_guess: Array,
    dimensions: tuple[int, ...],
    bits_per_dim: int = 3,
    radius: float = 0.6,
    finite_difference_eps: float = 1e-4,
) -> QuboProblem:
    """Linearize a Lorenz96 4D-Var window and encode a local increment as QUBO."""

    if len(dimensions) * bits_per_dim > 30:
        raise ValueError("QUBO exceeds the configured 30-variable/qubit limit.")

    model = problem.model
    base_trajectory = model.forecast(initial_guess, len(problem.observations))
    residual = (base_trajectory - problem.observations).reshape(-1)
    tangent = _finite_difference_tangent(problem, initial_guess, base_trajectory, dimensions, finite_difference_eps)

    background_precision = 1.0 / float(problem.background_cov[0, 0])
    observation_precision = 1.0 / float(problem.observation_cov[0, 0])
    a_matrix = background_precision * np.eye(len(dimensions), dtype=np.float64)
    a_matrix += observation_precision * (tangent.T @ tangent)
    g_vector = observation_precision * (tangent.T @ residual)

    encoding = _make_encoding(dimensions, bits_per_dim, radius)
    transform = _binary_transform(encoding)
    offset = encoding.offsets

    hessian_q = transform.T @ a_matrix @ transform
    linear_q = transform.T @ (a_matrix @ offset + g_vector)
    constant = float(0.5 * offset.T @ a_matrix @ offset + g_vector.T @ offset)
    matrix = _to_upper_qubo(hessian_q, linear_q)

    return QuboProblem(matrix=matrix, constant=constant, encoding=encoding)


def decode_increment(qubo: QuboProblem, bits: np.ndarray) -> tuple[tuple[int, ...], Array]:
    """Decode QUBO bits into state dimensions and increment values."""

    bits = np.asarray(bits, dtype=np.float64)
    if bits.shape != (qubo.n_variables,):
        raise ValueError(f"Expected {qubo.n_variables} bits, got shape {bits.shape}.")

    transform = _binary_transform(qubo.encoding)
    increment = qubo.encoding.offsets + transform @ bits
    return qubo.encoding.dimensions, increment.astype(np.float64)






def solve_qubo_greedy(
    qubo: QuboProblem,
    sweeps: int = 6,
    restarts: int = 4,
    seed: int = 0,
) -> QuboSolveResult:
    """Small dependency-free QUBO optimizer."""

    rng = np.random.default_rng(seed)
    best_bits = np.zeros(qubo.n_variables, dtype=np.int8)
    best_energy = _qubo_energy(qubo.matrix, best_bits) + qubo.constant

    starts = [np.zeros(qubo.n_variables, dtype=np.int8)]
    starts.extend(rng.integers(0, 2, size=qubo.n_variables, dtype=np.int8) for _ in range(restarts))

    for start in starts:
        bits = start.copy()
        energy = _qubo_energy(qubo.matrix, bits) + qubo.constant
        for _ in range(sweeps):
            improved = False
            for index in rng.permutation(qubo.n_variables):
                bits[index] ^= 1
                candidate = _qubo_energy(qubo.matrix, bits) + qubo.constant
                if candidate <= energy:
                    energy = candidate
                    improved = True
                else:
                    bits[index] ^= 1
            if not improved:
                break
        if energy < best_energy:
            best_energy = energy
            best_bits = bits.copy()

    return QuboSolveResult(
        bits=best_bits,
        energy=float(best_energy),
        n_variables=qubo.n_variables,
        n_sweeps=sweeps,
    )


def solve_qubo_qaoa(
    qubo: QuboProblem,
    reps: int = 1,
    shots: int = 256,
    optimizer_iterations: int = 0,
    seed: int = 0,
) -> QuboSolveResult:
    """Solve a QUBO with shallow sampled QAOA on Qiskit Aer."""

    if qubo.n_variables > 30:
        raise ValueError("QAOA backend refuses QUBOs above the configured 30-qubit limit.")
    if reps <= 0:
        raise ValueError("reps must be positive.")
    if shots <= 0:
        raise ValueError("shots must be positive.")

    scale = max(float(np.max(np.abs(qubo.matrix))), 1.0)
    scaled_matrix = qubo.matrix / scale
    rng = np.random.default_rng(seed)
    backend = AerSimulator(seed_simulator=seed)
    theta = np.concatenate(
        [
            np.full(reps, 0.08, dtype=np.float64),
            np.full(reps, 0.45, dtype=np.float64),
        ]
    )

    def sample(theta_values: Array, sample_seed: int) -> tuple[np.ndarray, float]:
        gammas = theta_values[:reps]
        betas = theta_values[reps:]
        circuit = _build_qaoa_circuit(scaled_matrix, gammas, betas)
        counts = backend.run(circuit, shots=shots, seed_simulator=sample_seed).result().get_counts()
        return _best_sample_from_counts(counts, qubo)

    def objective(theta_values: Array) -> float:
        _, energy = sample(np.asarray(theta_values, dtype=np.float64), seed)
        return energy

    if optimizer_iterations > 0:
        result = minimize(
            objective,
            theta,
            method="COBYLA",
            options={"maxiter": optimizer_iterations, "rhobeg": 0.25},
        )
        theta = np.asarray(result.x, dtype=np.float64)

    best_bits = np.zeros(qubo.n_variables, dtype=np.int8)
    best_energy = _qubo_energy(qubo.matrix, best_bits) + qubo.constant
    for _ in range(max(1, min(4, optimizer_iterations + 1))):
        bits, energy = sample(theta, int(rng.integers(0, 2**31 - 1)))
        if energy < best_energy:
            best_bits = bits
            best_energy = energy

    return QuboSolveResult(
        bits=best_bits,
        energy=float(best_energy),
        n_variables=qubo.n_variables,
        n_sweeps=optimizer_iterations,
    )


def solve_qubo(
    qubo: QuboProblem,
    solver: QuboSolverName = "qaoa",
    seed: int = 0,
    qaoa_reps: int = 1,
    qaoa_shots: int = 512,
    qaoa_optimizer_iterations: int = 20,
) -> QuboSolveResult:
    """Dispatch to the selected QUBO backend."""

    if solver == "qaoa":
        return solve_qubo_qaoa(
            qubo,
            reps=qaoa_reps,
            shots=qaoa_shots,
            optimizer_iterations=qaoa_optimizer_iterations,
            seed=seed,
        )
    if solver == "greedy":
        return solve_qubo_greedy(qubo, seed=seed)
    raise ValueError(f"Unknown QUBO solver: {solver}.")


def assimilate_window_qubo(
    problem: AssimilationProblem,
    block_size: int = 10,
    block_stride: int | None = None,
    block_selection: BlockSelectionName = "cyclic",
    bits_per_dim: int = 3,
    radius: float = 0.6,
    outer_loops: int = 1,
    seed: int = 0,
    solver: QuboSolverName = "qaoa",
    qaoa_reps: int = 1,
    qaoa_shots: int = 512,
    qaoa_optimizer_iterations: int = 20,
    finite_difference_eps: float = 1e-4,
    verbose: bool = False,
) -> WindowAssimilationResult:
    """Optimize one 4D-Var window by sweeping local QUBO increments."""

    state = problem.background.copy()
    max_qubo_variables = 0
    rng = np.random.default_rng(seed)

    for outer_index in range(outer_loops):
        blocks = _select_dimension_blocks(
            problem,
            state,
            block_size=block_size,
            block_stride=block_stride,
            block_selection=block_selection,
            finite_difference_eps=finite_difference_eps,
        )
        for block_number, dimensions in enumerate(blocks, start=1):
            if verbose:
                print(
                    f"[{solver}]   outer {outer_index + 1}/{outer_loops} "
                    f"block {block_number}/{len(blocks)} selection={block_selection} "
                    f"dims={_format_dimensions(dimensions)}"
                )
            current_cost = _classic_window_cost(problem, state)
            qubo = build_incremental_qubo(
                problem,
                state,
                dimensions,
                bits_per_dim=bits_per_dim,
                radius=radius,
                finite_difference_eps=finite_difference_eps,
            )
            max_qubo_variables = max(max_qubo_variables, qubo.n_variables)
            solved = solve_qubo(
                qubo,
                solver=solver,
                seed=int(rng.integers(0, 2**31 - 1)),
                qaoa_reps=qaoa_reps,
                qaoa_shots=qaoa_shots,
                qaoa_optimizer_iterations=qaoa_optimizer_iterations,
            )
            selected_dims, increment = decode_increment(qubo, solved.bits)
            candidate = state.copy()
            candidate[list(selected_dims)] += increment
            candidate_cost = _classic_window_cost(problem, candidate)
            accepted = candidate_cost <= current_cost
            if accepted:
                state = candidate
            if verbose:
                status = "accepted" if accepted else "rejected"
                print(
                    f"[{solver}]   block {block_number}/{len(blocks)} {status} "
                    f"dims={_format_dimensions(selected_dims)} "
                    f"cost={current_cost:.6f}->{candidate_cost:.6f} "
                    f"qubo_vars={qubo.n_variables}"
                )

    forecast = problem.model.forecast(state, len(problem.observations))
    return WindowAssimilationResult(
        initial_state=state,
        forecast=forecast,
        window_cost=_classic_window_cost(problem, state),
        max_qubo_variables=max_qubo_variables,
    )


def run_sliding_window_qubo(
    dataset: Lorenz96Dataset,
    window: int = 8,
    stride: int | None = None,
    block_size: int = 10,
    block_stride: int | None = None,
    block_selection: BlockSelectionName = "cyclic",
    bits_per_dim: int = 3,
    radius: float = 0.6,
    outer_loops: int = 1,
    time_sweeps: int = 1,
    time_sweep_mode: TimeSweepMode = "carry",
    background_std: float = 1.0,
    observation_std: float = 0.5,
    seed: int = 0,
    solver: QuboSolverName = "qaoa",
    qaoa_reps: int = 1,
    qaoa_shots: int = 512,
    qaoa_optimizer_iterations: int = 20,
    finite_difference_eps: float = 1e-4,
    verbose: bool = True,
) -> SlidingAssimilationResult:
    """Run sequential sliding-window Lorenz96 QUBO assimilation over a dataset."""

    if time_sweeps <= 0:
        raise ValueError("time_sweeps must be positive.")
    if time_sweep_mode not in ("carry", "background"):
        raise ValueError("time_sweep_mode must be 'carry' or 'background'.")
    if stride is None:
        stride = max(1, window - 1)
    effective_block_stride = block_size if block_stride is None else block_stride
    model = Lorenz96Model(state_dim=dataset.state_dim)
    analysis = np.full_like(dataset.observed, np.nan, dtype=np.float64)
    max_qubo_variables = 0
    rng = np.random.default_rng(seed)
    window_starts = list(range(0, dataset.n_times, stride))
    carried_backgrounds: dict[int, Array] = {}

    for time_sweep in range(time_sweeps):
        if verbose:
            print(
                f"[{solver}] time_sweep {time_sweep + 1}/{time_sweeps} "
                f"windows={len(window_starts)} mode={time_sweep_mode}"
            )
        for window_number, start_index in enumerate(window_starts, start=1):
            local_window = min(window, dataset.n_times - start_index)
            if local_window <= 0:
                break
            window_started = time.perf_counter()
            if verbose:
                end_index = start_index + local_window
                print(
                    f"[{solver}] time_sweep {time_sweep + 1}/{time_sweeps} "
                    f"window {window_number}/{len(window_starts)} "
                    f"indices={start_index}..{end_index - 1} start={start_index} length={local_window} "
                    f"block={block_size} block_stride={effective_block_stride} "
                    f"block_selection={block_selection} bits={bits_per_dim} "
                    f"mode={time_sweep_mode}"
                )

            if time_sweep_mode == "carry" and start_index in carried_backgrounds:
                background = carried_backgrounds[start_index].copy()
            else:
                background = _background_for_window(dataset, analysis, start_index)
            problem = make_lorenz96_problem(
                dataset,
                start_index=start_index,
                window=local_window,
                background=background,
                background_std=background_std,
                observation_std=observation_std,
                model=model,
            )
            result = assimilate_window_qubo(
                problem,
                block_size=block_size,
                block_stride=block_stride,
                block_selection=block_selection,
                bits_per_dim=bits_per_dim,
                radius=radius,
                outer_loops=outer_loops,
                seed=int(rng.integers(0, 2**31 - 1)),
                solver=solver,
                qaoa_reps=qaoa_reps,
                qaoa_shots=qaoa_shots,
                qaoa_optimizer_iterations=qaoa_optimizer_iterations,
                finite_difference_eps=finite_difference_eps,
                verbose=verbose,
            )
            end_index = start_index + local_window
            analysis[start_index:end_index] = result.forecast[:local_window]
            if time_sweep_mode == "carry":
                carried_backgrounds[start_index] = result.initial_state.copy()
            max_qubo_variables = max(max_qubo_variables, result.max_qubo_variables)
            if verbose:
                elapsed = time.perf_counter() - window_started
                print(
                    f"[{solver}] time_sweep {time_sweep + 1}/{time_sweeps} "
                    f"window {window_number}/{len(window_starts)} done "
                    f"cost={result.window_cost:.6f} max_qubo_vars={result.max_qubo_variables} "
                    f"elapsed={elapsed:.2f}s"
                )

    score = None
    if dataset.has_truth:
        score = rmse(analysis[np.isfinite(dataset.truth)], dataset.truth[np.isfinite(dataset.truth)])

    return SlidingAssimilationResult(
        analysis=analysis,
        time_steps=dataset.time_steps,
        rmse_vs_truth=score,
        max_qubo_variables=max_qubo_variables,
    )


def _finite_difference_tangent(
    problem: AssimilationProblem,
    initial_guess: Array,
    base_trajectory: Array,
    dimensions: tuple[int, ...],
    eps: float,
) -> Array:
    columns = []
    for dim in dimensions:
        perturbed = initial_guess.copy()
        perturbed[dim] += eps
        trajectory = problem.model.forecast(perturbed, len(problem.observations))
        columns.append(((trajectory - base_trajectory) / eps).reshape(-1))
    return np.column_stack(columns).astype(np.float64)


def _select_dimension_blocks(
    problem: AssimilationProblem,
    state: Array,
    block_size: int,
    block_stride: int | None,
    block_selection: BlockSelectionName,
    finite_difference_eps: float,
) -> list[tuple[int, ...]]:
    state_dim = int(problem.background.shape[0])
    if block_selection == "cyclic":
        return [
            tuple(range(start, min(start + block_size, state_dim)))
            for start in _block_starts(state_dim, block_size, block_stride)
        ]

    # Adaptive policies choose one active block from the current linearized
    # window. Multiple active blocks are obtained by increasing outer_loops so
    # each loop can re-linearize around the updated state.
    max_blocks = 1
    hessian, gradient = _window_quadratic_model(problem, state, finite_difference_eps)
    if block_selection == "gradient":
        return _gradient_blocks(gradient, block_size, max_blocks)
    if block_selection == "hessian":
        return _hessian_coupled_blocks(hessian, gradient, block_size, max_blocks)
    raise ValueError(f"Unknown block_selection: {block_selection}.")


def _window_quadratic_model(problem: AssimilationProblem, state: Array, finite_difference_eps: float) -> tuple[Array, Array]:
    dimensions = tuple(range(problem.background.shape[0]))
    base_trajectory = problem.model.forecast(state, len(problem.observations))
    residual = (base_trajectory - problem.observations).reshape(-1)
    tangent = _finite_difference_tangent(problem, state, base_trajectory, dimensions, finite_difference_eps)

    background_precision = 1.0 / float(problem.background_cov[0, 0])
    observation_precision = 1.0 / float(problem.observation_cov[0, 0])
    hessian = background_precision * np.eye(len(dimensions), dtype=np.float64)
    hessian += observation_precision * (tangent.T @ tangent)
    gradient = background_precision * (state - problem.background)
    gradient += observation_precision * (tangent.T @ residual)
    return hessian, gradient


def _gradient_blocks(gradient: Array, block_size: int, max_blocks: int) -> list[tuple[int, ...]]:
    order = np.argsort(-np.abs(gradient))
    blocks = []
    used: set[int] = set()
    for seed in order:
        if int(seed) in used:
            continue
        block = []
        for dim in order:
            dim_int = int(dim)
            if dim_int in used:
                continue
            block.append(dim_int)
            used.add(dim_int)
            if len(block) == block_size:
                break
        if block:
            blocks.append(tuple(sorted(block)))
        if len(blocks) >= max_blocks or len(used) == gradient.shape[0]:
            break
    return blocks


def _hessian_coupled_blocks(hessian: Array, gradient: Array, block_size: int, max_blocks: int) -> list[tuple[int, ...]]:
    seed_order = np.argsort(-np.abs(gradient))
    blocks = []
    used: set[int] = set()
    all_dims = set(range(gradient.shape[0]))

    for seed in seed_order:
        seed_int = int(seed)
        if seed_int in used:
            continue
        block = [seed_int]
        used.add(seed_int)
        while len(block) < block_size and used != all_dims:
            candidates = np.asarray(sorted(all_dims - used), dtype=np.int64)
            coupling = np.sum(np.abs(hessian[np.ix_(candidates, block)]), axis=1)
            next_dim = int(candidates[int(np.argmax(coupling))])
            block.append(next_dim)
            used.add(next_dim)
        blocks.append(tuple(sorted(block)))
        if len(blocks) >= max_blocks or used == all_dims:
            break
    return blocks


def _block_starts(state_dim: int, block_size: int, block_stride: int | None) -> list[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    stride = block_size if block_stride is None else block_stride
    if stride <= 0:
        raise ValueError("block_stride must be positive.")
    if block_size > state_dim:
        return [0]

    starts = list(range(0, state_dim - block_size + 1, stride))
    final_start = state_dim - block_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _format_dimensions(dimensions: tuple[int, ...]) -> str:
    if not dimensions:
        return "[]"
    if len(dimensions) == 1:
        return f"[{dimensions[0]}]"
    if all(right == left + 1 for left, right in zip(dimensions, dimensions[1:])):
        return f"[{dimensions[0]}..{dimensions[-1]}]"
    return "[" + ",".join(str(dim) for dim in dimensions) + "]"


def _build_qaoa_circuit(matrix: Array, gammas: Array, betas: Array) -> QuantumCircuit:
    n_qubits = int(matrix.shape[0])
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))

    z_coefficients, zz_terms = _qubo_to_ising_terms(matrix)
    for gamma, beta in zip(gammas, betas):
        for qubit, coefficient in enumerate(z_coefficients):
            if abs(coefficient) > 1e-12:
                circuit.rz(2.0 * gamma * coefficient, qubit)
        for left, right, coefficient in zz_terms:
            if abs(coefficient) > 1e-12:
                circuit.rzz(2.0 * gamma * coefficient, left, right)
        for qubit in range(n_qubits):
            circuit.rx(2.0 * beta, qubit)

    circuit.measure_all()
    return circuit


def _qubo_to_ising_terms(matrix: Array) -> tuple[Array, list[tuple[int, int, float]]]:
    n_variables = int(matrix.shape[0])
    z_coefficients = np.zeros(n_variables, dtype=np.float64)
    zz_terms: list[tuple[int, int, float]] = []

    for row in range(n_variables):
        z_coefficients[row] -= 0.5 * matrix[row, row]
        for col in range(row + 1, n_variables):
            value = matrix[row, col]
            if abs(value) <= 1e-12:
                continue
            z_coefficients[row] -= 0.25 * value
            z_coefficients[col] -= 0.25 * value
            zz_terms.append((row, col, 0.25 * float(value)))

    return z_coefficients, zz_terms


def _best_sample_from_counts(counts: dict[str, int], qubo: QuboProblem) -> tuple[np.ndarray, float]:
    best_bits = np.zeros(qubo.n_variables, dtype=np.int8)
    best_energy = _qubo_energy(qubo.matrix, best_bits) + qubo.constant

    for bitstring in counts:
        compact = bitstring.replace(" ", "")
        bits = np.asarray([int(bit) for bit in compact[::-1]], dtype=np.int8)
        bits = bits[: qubo.n_variables]
        energy = _qubo_energy(qubo.matrix, bits) + qubo.constant
        if energy < best_energy:
            best_bits = bits
            best_energy = energy

    return best_bits, float(best_energy)


def _make_encoding(dimensions: tuple[int, ...], bits_per_dim: int, radius: float) -> QuboEncoding:
    if bits_per_dim <= 0:
        raise ValueError("bits_per_dim must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    n_levels = 2**bits_per_dim
    step = 2.0 * radius / float(n_levels - 1)
    offsets = np.full(len(dimensions), -radius, dtype=np.float64)
    weights = np.asarray([step * 2**bit for bit in range(bits_per_dim)], dtype=np.float64)
    return QuboEncoding(
        dimensions=dimensions,
        bits_per_dim=bits_per_dim,
        radius=radius,
        offsets=offsets,
        weights=weights,
    )


def _binary_transform(encoding: QuboEncoding) -> Array:
    transform = np.zeros((len(encoding.dimensions), encoding.n_variables), dtype=np.float64)
    for dim_index in range(len(encoding.dimensions)):
        for bit in range(encoding.bits_per_dim):
            transform[dim_index, dim_index * encoding.bits_per_dim + bit] = encoding.weights[bit]
    return transform


def _to_upper_qubo(hessian: Array, linear: Array) -> Array:
    matrix = np.triu(hessian).astype(np.float64)
    diagonal = 0.5 * np.diag(hessian) + linear
    np.fill_diagonal(matrix, diagonal)
    return matrix


def _qubo_energy(matrix: Array, bits: np.ndarray) -> float:
    bits_float = bits.astype(np.float64)
    return float(bits_float @ matrix @ bits_float)


def _classic_window_cost(problem: AssimilationProblem, initial_state: Array) -> float:
    background_delta = initial_state - problem.background
    background_var = float(problem.background_cov[0, 0])
    observation_var = float(problem.observation_cov[0, 0])
    trajectory = problem.model.forecast(initial_state, len(problem.observations))
    value = 0.5 * float(background_delta @ background_delta) / background_var
    residual = trajectory - problem.observations
    value += 0.5 * float(np.sum(residual * residual)) / observation_var
    return value


def _background_for_window(dataset: Lorenz96Dataset, analysis: Array, start_index: int) -> Array:
    if np.isfinite(analysis[start_index]).all():
        return analysis[start_index].copy()
    return dataset.observed[start_index].copy()
