from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from q4dvar.problem import AssimilationProblem
from q4dvar.solvers.classical import cost


Array = NDArray[np.float64]


@dataclass(frozen=True)
class VQEResult:
    state: Array
    cost: float
    bitstring: str
    expectation: float
    grid_min_cost: float
    gap_to_grid_min: float
    optimizer_success: bool
    optimizer_message: str
    n_qubits: int


def build_ansatz(n_qubits: int, layers: int = 2) -> tuple[QuantumCircuit, ParameterVector]:
    """Build a small hardware-efficient ansatz for the VQE demo."""

    parameters = ParameterVector("theta", length=2 * layers * n_qubits)
    circuit = QuantumCircuit(n_qubits)
    parameter_index = 0

    for _ in range(layers):
        for qubit in range(n_qubits):
            circuit.ry(parameters[parameter_index], qubit)
            parameter_index += 1
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        if n_qubits > 2:
            circuit.cx(n_qubits - 1, 0)
        for qubit in range(n_qubits):
            circuit.rz(parameters[parameter_index], qubit)
            parameter_index += 1

    return circuit, parameters


def decode_grid_state(
    basis_index: int,
    n_dims: int,
    bits_per_dim: int,
    lower: float,
    upper: float,
) -> Array:
    """Decode a computational basis index into a low-dimensional state vector."""

    levels = np.linspace(lower, upper, 2**bits_per_dim, dtype=np.float64)
    values = []
    for dim in range(n_dims):
        start = dim * bits_per_dim
        level_index = 0
        for bit in range(bits_per_dim):
            level_index |= ((basis_index >> (start + bit)) & 1) << bit
        values.append(levels[level_index])
    return np.asarray(values, dtype=np.float64)


def diagonal_cost_values(
    problem: AssimilationProblem,
    bits_per_dim: int = 2,
    lower: float = -2.0,
    upper: float = 2.0,
) -> tuple[Array, list[Array]]:
    """Evaluate the 4D-Var objective on a binary grid."""

    n_dims = problem.background.shape[0]
    n_qubits = n_dims * bits_per_dim
    values = np.zeros(2**n_qubits, dtype=np.float64)
    states = []

    for index in range(2**n_qubits):
        state = decode_grid_state(index, n_dims, bits_per_dim, lower, upper)
        states.append(state)
        values[index] = cost(problem, state)

    return values, states


def diagonal_hamiltonian(values: Array, n_qubits: int) -> SparsePauliOp:
    """Create a Pauli-Z expansion for a diagonal cost Hamiltonian."""

    paulis: list[str] = []
    coefficients: list[float] = []
    n_states = 2**n_qubits

    for mask in range(n_states):
        eigen_products = np.ones(n_states, dtype=np.float64)
        label = ["I"] * n_qubits
        for qubit in range(n_qubits):
            if (mask >> qubit) & 1:
                label[n_qubits - 1 - qubit] = "Z"
                bit_values = np.array(
                    [1.0 if ((index >> qubit) & 1) == 0 else -1.0 for index in range(n_states)],
                    dtype=np.float64,
                )
                eigen_products *= bit_values
        coefficient = float(np.dot(values, eigen_products) / n_states)
        if abs(coefficient) > 1e-12:
            paulis.append("".join(label))
            coefficients.append(coefficient)

    return SparsePauliOp(paulis, coefficients)


def statevector_from_parameters(
    ansatz: QuantumCircuit,
    parameters: ParameterVector,
    values: Array,
    backend: AerSimulator,
) -> Statevector:
    """Bind ansatz parameters and return an Aer-simulated exact statevector."""

    bound = ansatz.assign_parameters({parameter: value for parameter, value in zip(parameters, values)})
    bound.save_statevector()
    result = backend.run(bound).result()
    return result.get_statevector(bound)


def solve_vqe(
    problem: AssimilationProblem,
    bits_per_dim: int = 2,
    grid_lower: float = -2.0,
    grid_upper: float = 2.0,
    layers: int = 2,
    maxiter: int = 600,
    seed: int = 11,
) -> VQEResult:
    """Approximate the best grid state with VQE over a diagonal Hamiltonian."""

    grid_costs, grid_states = diagonal_cost_values(problem, bits_per_dim, grid_lower, grid_upper)
    n_qubits = problem.background.shape[0] * bits_per_dim
    hamiltonian = diagonal_hamiltonian(grid_costs, n_qubits)
    ansatz, parameters = build_ansatz(n_qubits, layers)
    backend = AerSimulator(method="statevector")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-0.2, 0.2, size=len(parameters))

    def objective(theta: Array) -> float:
        statevector = statevector_from_parameters(ansatz, parameters, theta, backend)
        return float(np.real(statevector.expectation_value(hamiltonian)))

    result = minimize(
        objective,
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.6},
    )

    final_statevector = statevector_from_parameters(
        ansatz,
        parameters,
        np.asarray(result.x, dtype=np.float64),
        backend,
    )
    probabilities = final_statevector.probabilities()
    best_index = int(np.argmax(probabilities))
    best_state = grid_states[best_index]
    bitstring = format(best_index, f"0{n_qubits}b")[::-1]
    expectation = float(np.real(final_statevector.expectation_value(hamiltonian)))
    grid_min_cost = float(np.min(grid_costs))
    gap_to_grid_min = expectation - grid_min_cost
    converged_to_grid_min = gap_to_grid_min < 1e-2
    message = (
        "Energy converged to the discretized grid minimum."
        if converged_to_grid_min
        else str(result.message)
    )

    return VQEResult(
        state=best_state,
        cost=cost(problem, best_state),
        bitstring=bitstring,
        expectation=expectation,
        grid_min_cost=grid_min_cost,
        gap_to_grid_min=gap_to_grid_min,
        optimizer_success=bool(result.success or converged_to_grid_min),
        optimizer_message=message,
        n_qubits=n_qubits,
    )
