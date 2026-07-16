"""First- and second-order Trotter product formulas."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp

from atlas.circuits.evolution.base import EvolutionCircuitSpec, EvolutionMethod
from atlas.circuits.initial_states import InitialStateSpec
from atlas.physics.hamiltonians import Hamiltonian


def _pauli_terms(operator: SparsePauliOp) -> list[tuple[Pauli, complex]]:
    """Return ``(Pauli, coefficient)`` pairs with deterministic ordering."""

    return list(zip(operator.paulis, operator.coeffs))


def _apply_pauli_evolution(
    circuit: QuantumCircuit,
    pauli: Pauli,
    time: float,
) -> None:
    circuit.append(PauliEvolutionGate(pauli, time=time), circuit.qubits)


class _ProductFormulaBase(EvolutionMethod):
    """Shared helpers for Trotter-style product formulas."""

    def __init__(self, num_trotter_steps: int = 1):
        if num_trotter_steps < 1:
            raise ValueError("num_trotter_steps must be at least 1.")
        self.num_trotter_steps = num_trotter_steps

    def _compose_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
        method_name: str,
        step_builder,
    ) -> EvolutionCircuitSpec:
        circuit = QuantumCircuit(hamiltonian.num_qubits)
        circuit.compose(initial_state.preparation_circuit(), inplace=True)

        step_time = evolution_time / self.num_trotter_steps
        terms = _pauli_terms(hamiltonian.operator())

        for _ in range(self.num_trotter_steps):
            step_builder(circuit, terms, step_time)

        return EvolutionCircuitSpec(
            circuit=circuit,
            num_qubits=hamiltonian.num_qubits,
            num_trotter_steps=self.num_trotter_steps,
            evolution_time=evolution_time,
            method_name=method_name,
        )


class LieTrotter(_ProductFormulaBase):
    """First-order Lie-Trotter product formula."""

    @property
    def name(self) -> str:
        return "lie"

    def build_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
    ) -> EvolutionCircuitSpec:
        def step_builder(circuit, terms, step_time):
            for pauli, coeff in terms:
                _apply_pauli_evolution(circuit, pauli, step_time * coeff.real)

        return self._compose_circuit(
            hamiltonian,
            initial_state,
            evolution_time,
            self.name,
            step_builder,
        )


class StrangTrotter(_ProductFormulaBase):
    """Second-order Strang splitting (symmetric Trotter)."""

    @property
    def name(self) -> str:
        return "strang"

    def build_circuit(
        self,
        hamiltonian: Hamiltonian,
        initial_state: InitialStateSpec,
        evolution_time: float,
    ) -> EvolutionCircuitSpec:
        def step_builder(circuit, terms, step_time):
            half_time = step_time / 2
            for pauli, coeff in terms:
                _apply_pauli_evolution(circuit, pauli, half_time * coeff.real)
            for pauli, coeff in reversed(terms):
                _apply_pauli_evolution(circuit, pauli, half_time * coeff.real)

        return self._compose_circuit(
            hamiltonian,
            initial_state,
            evolution_time,
            self.name,
            step_builder,
        )
