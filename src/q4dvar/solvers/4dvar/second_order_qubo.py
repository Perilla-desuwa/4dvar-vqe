from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from q4dvar.data_loader import Lorenz96Dataset
from q4dvar.models.lorenz96 import Lorenz96Model, make_lorenz96_problem
from q4dvar.problem import Array, AssimilationProblem
from q4dvar.solvers.classical import rmse

from .qubo import (
    BlockSelectionName,
    QuboEncoding,
    QuboProblem,
    QuboSolverName,
    SlidingAssimilationResult,
    TimeSweepMode,
    WindowAssimilationResult,
    _background_for_window,
    _binary_transform,
    _classic_window_cost,
    _format_dimensions,
    _make_encoding,
    _select_dimension_blocks,
    _to_upper_qubo,
    solve_qubo,
)


@dataclass(frozen=True)
class SecondOrderQuboProblem:
    """QUBO with original bits plus auxiliary pair-product bits."""

    qubo: QuboProblem
    original_bit_count: int
    pair_indices: tuple[tuple[int, int], ...]
    penalty_strength: float


def build_second_order_incremental_qubo(
    problem: AssimilationProblem,
    initial_guess: Array,
    dimensions: tuple[int, ...],
    bits_per_dim: int = 2,
    radius: float = 0.4,
    finite_difference_eps: float = 1e-3,
    penalty_strength: float = 20.0,
    max_qubo_variables: int = 30,
) -> SecondOrderQuboProblem:
    """Build a second-order incremental 4D-Var QUBO with auxiliary variables."""

    encoding = _make_encoding(dimensions, bits_per_dim, radius)
    original_bits = encoding.n_variables
    pair_indices = tuple((left, right) for left in range(original_bits) for right in range(left + 1, original_bits))
    total_variables = original_bits + len(pair_indices)
    if total_variables > max_qubo_variables:
        raise ValueError(
            "Second-order QUBO exceeds variable limit: "
            f"original_bits={original_bits}, auxiliary={len(pair_indices)}, total={total_variables}, "
            f"limit={max_qubo_variables}."
        )

    base_trajectory = problem.model.forecast(initial_guess, len(problem.observations))
    residual = (base_trajectory - problem.observations).reshape(-1)
    tangent, hessian = _finite_difference_second_order(
        problem,
        initial_guess,
        base_trajectory,
        dimensions,
        finite_difference_eps,
    )

    transform = _binary_transform(encoding)
    offset = encoding.offsets
    coeff_constant, coeff_matrix = _second_order_residual_coefficients(
        residual,
        tangent,
        hessian,
        transform,
        offset,
        pair_indices,
    )

    observation_precision = 1.0 / float(problem.observation_cov[0, 0])
    hessian_z = observation_precision * (coeff_matrix.T @ coeff_matrix)
    linear_z = observation_precision * (coeff_constant @ coeff_matrix)
    constant = 0.5 * observation_precision * float(coeff_constant @ coeff_constant)

    background_precision = 1.0 / float(problem.background_cov[0, 0])
    background_delta = initial_guess[list(dimensions)] - problem.background[list(dimensions)]
    background_offset = background_delta + offset
    hessian_z[:original_bits, :original_bits] += background_precision * (transform.T @ transform)
    linear_z[:original_bits] += background_precision * (transform.T @ background_offset)
    constant += 0.5 * background_precision * float(background_offset @ background_offset)

    matrix = _to_upper_qubo(hessian_z, linear_z)
    _add_pair_product_penalties(matrix, original_bits, pair_indices, penalty_strength)

    lifted_encoding = QuboEncoding(
        dimensions=encoding.dimensions,
        bits_per_dim=encoding.bits_per_dim,
        radius=encoding.radius,
        offsets=encoding.offsets,
        weights=encoding.weights,
    )
    return SecondOrderQuboProblem(
        qubo=QuboProblem(matrix=matrix, constant=constant, encoding=lifted_encoding),
        original_bit_count=original_bits,
        pair_indices=pair_indices,
        penalty_strength=penalty_strength,
    )


def decode_second_order_increment(problem: SecondOrderQuboProblem, bits: np.ndarray) -> tuple[tuple[int, ...], Array]:
    """Decode only the original bits; auxiliary bits are used only during optimization."""

    original = np.asarray(bits[: problem.original_bit_count], dtype=np.float64)
    transform = _binary_transform(problem.qubo.encoding)
    increment = problem.qubo.encoding.offsets + transform @ original
    return problem.qubo.encoding.dimensions, increment.astype(np.float64)


def assimilate_window_second_order_qubo(
    problem: AssimilationProblem,
    block_size: int = 3,
    block_stride: int | None = None,
    block_selection: BlockSelectionName = "cyclic",
    bits_per_dim: int = 2,
    radius: float = 0.4,
    outer_loops: int = 1,
    seed: int = 0,
    solver: QuboSolverName = "greedy",
    penalty_strength: float = 20.0,
    finite_difference_eps: float = 1e-3,
    qaoa_reps: int = 1,
    qaoa_shots: int = 512,
    qaoa_optimizer_iterations: int = 20,
    verbose: bool = False,
) -> WindowAssimilationResult:
    """Optimize one window using second-order incremental QUBO blocks."""

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
                    f"[second-order-{solver}]   outer {outer_index + 1}/{outer_loops} "
                    f"block {block_number}/{len(blocks)} selection={block_selection} "
                    f"dims={_format_dimensions(dimensions)}"
                )
            current_cost = _classic_window_cost(problem, state)
            second_order = build_second_order_incremental_qubo(
                problem,
                state,
                dimensions,
                bits_per_dim=bits_per_dim,
                radius=radius,
                finite_difference_eps=finite_difference_eps,
                penalty_strength=penalty_strength,
            )
            max_qubo_variables = max(max_qubo_variables, second_order.qubo.n_variables)
            solved = solve_qubo(
                second_order.qubo,
                solver=solver,
                seed=int(rng.integers(0, 2**31 - 1)),
                qaoa_reps=qaoa_reps,
                qaoa_shots=qaoa_shots,
                qaoa_optimizer_iterations=qaoa_optimizer_iterations,
            )
            selected_dims, increment = decode_second_order_increment(second_order, solved.bits)
            candidate = state.copy()
            candidate[list(selected_dims)] += increment
            candidate_cost = _classic_window_cost(problem, candidate)
            accepted = candidate_cost <= current_cost
            if accepted:
                state = candidate
            if verbose:
                status = "accepted" if accepted else "rejected"
                print(
                    f"[second-order-{solver}]   block {block_number}/{len(blocks)} {status} "
                    f"dims={_format_dimensions(selected_dims)} "
                    f"cost={current_cost:.6f}->{candidate_cost:.6f} "
                    f"qubo_vars={second_order.qubo.n_variables}"
                )

    forecast = problem.model.forecast(state, len(problem.observations))
    return WindowAssimilationResult(
        initial_state=state,
        forecast=forecast,
        window_cost=_classic_window_cost(problem, state),
        max_qubo_variables=max_qubo_variables,
    )


def run_sliding_window_second_order_qubo(
    dataset: Lorenz96Dataset,
    window: int = 6,
    stride: int | None = None,
    block_size: int = 3,
    block_stride: int | None = None,
    block_selection: BlockSelectionName = "cyclic",
    bits_per_dim: int = 2,
    radius: float = 0.4,
    outer_loops: int = 1,
    time_sweeps: int = 1,
    time_sweep_mode: TimeSweepMode = "carry",
    background_std: float = 1.0,
    observation_std: float = 0.5,
    seed: int = 0,
    solver: QuboSolverName = "greedy",
    penalty_strength: float = 20.0,
    finite_difference_eps: float = 1e-3,
    qaoa_reps: int = 1,
    qaoa_shots: int = 512,
    qaoa_optimizer_iterations: int = 20,
    verbose: bool = True,
) -> SlidingAssimilationResult:
    """Run sliding-window second-order QUBO assimilation over a dataset."""

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
                f"[second-order-{solver}] time_sweep {time_sweep + 1}/{time_sweeps} "
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
                    f"[second-order-{solver}] time_sweep {time_sweep + 1}/{time_sweeps} "
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
            assim_problem = make_lorenz96_problem(
                dataset,
                start_index=start_index,
                window=local_window,
                background=background,
                background_std=background_std,
                observation_std=observation_std,
                model=model,
            )
            result = assimilate_window_second_order_qubo(
                assim_problem,
                block_size=block_size,
                block_stride=block_stride,
                block_selection=block_selection,
                bits_per_dim=bits_per_dim,
                radius=radius,
                outer_loops=outer_loops,
                seed=int(rng.integers(0, 2**31 - 1)),
                solver=solver,
                penalty_strength=penalty_strength,
                finite_difference_eps=finite_difference_eps,
                qaoa_reps=qaoa_reps,
                qaoa_shots=qaoa_shots,
                qaoa_optimizer_iterations=qaoa_optimizer_iterations,
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
                    f"[second-order-{solver}] time_sweep {time_sweep + 1}/{time_sweeps} "
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


def _finite_difference_second_order(
    problem: AssimilationProblem,
    initial_guess: Array,
    base_trajectory: Array,
    dimensions: tuple[int, ...],
    eps: float,
) -> tuple[Array, Array]:
    output_size = int(base_trajectory.size)
    n_dims = len(dimensions)
    tangent = np.zeros((output_size, n_dims), dtype=np.float64)
    hessian = np.zeros((output_size, n_dims, n_dims), dtype=np.float64)

    for local_index, dim in enumerate(dimensions):
        plus_state = initial_guess.copy()
        minus_state = initial_guess.copy()
        plus_state[dim] += eps
        minus_state[dim] -= eps
        plus_traj = problem.model.forecast(plus_state, len(problem.observations)).reshape(-1)
        minus_traj = problem.model.forecast(minus_state, len(problem.observations)).reshape(-1)
        base_flat = base_trajectory.reshape(-1)
        tangent[:, local_index] = (plus_traj - minus_traj) / (2.0 * eps)
        hessian[:, local_index, local_index] = (plus_traj - 2.0 * base_flat + minus_traj) / eps**2

    for left in range(n_dims):
        for right in range(left + 1, n_dims):
            f_pp = _forecast_pair(problem, initial_guess, dimensions[left], dimensions[right], eps, eps)
            f_pm = _forecast_pair(problem, initial_guess, dimensions[left], dimensions[right], eps, -eps)
            f_mp = _forecast_pair(problem, initial_guess, dimensions[left], dimensions[right], -eps, eps)
            f_mm = _forecast_pair(problem, initial_guess, dimensions[left], dimensions[right], -eps, -eps)
            mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
            hessian[:, left, right] = mixed
            hessian[:, right, left] = mixed

    return tangent, hessian


def _forecast_pair(
    problem: AssimilationProblem,
    initial_guess: Array,
    left_dim: int,
    right_dim: int,
    left_delta: float,
    right_delta: float,
) -> Array:
    state = initial_guess.copy()
    state[left_dim] += left_delta
    state[right_dim] += right_delta
    return problem.model.forecast(state, len(problem.observations)).reshape(-1)


def _second_order_residual_coefficients(
    residual: Array,
    tangent: Array,
    hessian: Array,
    transform: Array,
    offset: Array,
    pair_indices: tuple[tuple[int, int], ...],
) -> tuple[Array, Array]:
    original_bits = transform.shape[1]
    total_variables = original_bits + len(pair_indices)
    constant = residual + tangent @ offset + 0.5 * np.einsum("i,oij,j->o", offset, hessian, offset)
    coefficients = np.zeros((residual.shape[0], total_variables), dtype=np.float64)

    coefficients[:, :original_bits] = tangent @ transform
    coefficients[:, :original_bits] += np.einsum("iq,oij,j->oq", transform, hessian, offset)
    diagonal_quadratic = 0.5 * np.einsum("iq,oij,jq->oq", transform, hessian, transform)
    coefficients[:, :original_bits] += diagonal_quadratic

    for pair_offset, (left, right) in enumerate(pair_indices):
        variable_index = original_bits + pair_offset
        coefficients[:, variable_index] = np.einsum(
            "i,oij,j->o",
            transform[:, left],
            hessian,
            transform[:, right],
        )

    return constant, coefficients


def _add_pair_product_penalties(
    matrix: Array,
    original_bits: int,
    pair_indices: tuple[tuple[int, int], ...],
    strength: float,
) -> None:
    for pair_offset, (left, right) in enumerate(pair_indices):
        aux = original_bits + pair_offset
        matrix[aux, aux] += 3.0 * strength
        _add_upper_pair(matrix, left, right, strength)
        _add_upper_pair(matrix, left, aux, -2.0 * strength)
        _add_upper_pair(matrix, right, aux, -2.0 * strength)


def _add_upper_pair(matrix: Array, left: int, right: int, value: float) -> None:
    row, col = (left, right) if left <= right else (right, left)
    matrix[row, col] += value
