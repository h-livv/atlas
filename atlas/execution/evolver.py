"""Local statevector evolver for Hamiltonian simulation circuits.

Distinct from ``estimator.py``: evolution runs a complete circuit once and
returns the final statevector rather than evaluating repeated expectation
values during optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class EvolutionResult:
    """Outcome of evolving a circuit to its final statevector."""

    statevector: np.ndarray
    num_qubits: int


class StatevectorEvolver:
    """Evolves a quantum circuit via exact statevector simulation."""

    def evolve(self, circuit: QuantumCircuit) -> EvolutionResult:
        """Simulate ``circuit`` and return the final statevector."""

        statevector = np.asarray(Statevector(circuit))
        return EvolutionResult(
            statevector=statevector,
            num_qubits=circuit.num_qubits,
        )
